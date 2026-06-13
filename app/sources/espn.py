"""ESPN 公共 JSON 接口：球队近况（主力源，无反爬、无需 key）。

注意：ESPN 没有 xG 数据；xG/xGA 只有 FBref 降级源可用时才有。
"""
import difflib
import logging
import time
from datetime import datetime, timedelta, timezone

import requests

from config import USER_AGENT

log = logging.getLogger("espn")

SEARCH = "https://site.web.api.espn.com/apis/common/v3/search"
SCHEDULE = "https://site.api.espn.com/apis/site/v2/sports/soccer/all/teams/{team_id}/schedule"
SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"

_team_id_cache = {}


class EspnError(Exception):
    pass


def _get(url, params=None, retries=3):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(
                url, params=params, headers={"User-Agent": USER_AGENT}, timeout=30
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_err = e
            log.warning("ESPN 请求失败 (第%d次): %s", attempt, e)
            time.sleep(2 ** attempt)
    raise EspnError(f"ESPN 请求 {url} 重试 {retries} 次仍失败: {last_err}")


def _find_team_id(team_en):
    if team_en in _team_id_cache:
        return _team_id_cache[team_en]
    data = _get(SEARCH, {"query": team_en, "limit": 10, "type": "team"})
    for item in data.get("items", []):
        if item.get("type") != "team":
            continue
        # 优先取有足球默认联赛的条目（搜索结果跨运动项目）
        slug = item.get("defaultLeagueSlug", "")
        if slug and not slug.isdigit():
            _team_id_cache[team_en] = item["id"]
            return item["id"]
    # 退而求其次取第一个 team
    for item in data.get("items", []):
        if item.get("type") == "team":
            _team_id_cache[team_en] = item["id"]
            return item["id"]
    raise EspnError(f"ESPN 搜索 '{team_en}' 未找到球队")


def get_recent_matches(team_en, limit=10):
    """返回 {source, matches: [{date, comp, venue, opponent, result, gf, ga, xg, xga}]}"""
    team_id = _find_team_id(team_en)
    data = _get(SCHEDULE.format(team_id=team_id))
    played = []
    for ev in data.get("events", []):
        comp = ev["competitions"][0]
        state = (comp.get("status", {}).get("type") or {}).get("state")
        if state != "post":
            continue
        sides = {c["homeAway"]: c for c in comp["competitors"]}
        home, away = sides.get("home"), sides.get("away")
        if not home or not away:
            continue
        is_home = str(home["team"]["id"]) == str(team_id)
        me, opp = (home, away) if is_home else (away, home)
        try:
            gf = int(float(me.get("score", {}).get("value", 0)))
            ga = int(float(opp.get("score", {}).get("value", 0)))
        except (TypeError, ValueError):
            continue
        result = "W" if gf > ga else ("L" if gf < ga else "D")
        played.append({
            "date": ev.get("date", "")[:10],
            "comp": (ev.get("league") or {}).get("name", ""),
            "venue": "主" if is_home else "客",
            "opponent": opp["team"]["displayName"],
            "result": result,
            "gf": gf,
            "ga": ga,
            "xg": None,   # ESPN 无 xG
            "xga": None,
        })
    played.sort(key=lambda m: m["date"], reverse=True)
    if not played:
        raise EspnError(f"ESPN 球队 {team_en} (id={team_id}) 没有已完赛记录")
    return {"source": "espn", "team_id": team_id, "matches": played[:limit]}


# 各数据源的国家队写法差异（ESPN/FIFA/The Odds API 各有习惯）
_ALIASES = {
    "czechia": "czech republic",
    "türkiye": "turkey",
    "united states": "usa",
    "côte d'ivoire": "ivory coast",
    "cote d'ivoire": "ivory coast",
    "korea republic": "south korea",
    "ir iran": "iran",
    "congo dr": "dr congo",
    "democratic republic of the congo": "dr congo",
    "cabo verde": "cape verde",
    "curacao": "curaçao",
    "bosnia and herzegovina": "bosnia & herzegovina",
    "united arab emirates": "uae",
}


def _canon(name):
    n = name.lower().strip()
    return _ALIASES.get(n, n)


def _similar(a, b):
    a, b = _canon(a), _canon(b)
    if a == b or a in b or b in a:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _parse_espn_date(s):
    for fmt in ("%Y-%m-%dT%H:%MZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


SCHEDULE_LEAGUES = ("all", "fifa.world")  # 世界杯赛果在 fifa.world 专属赛程里


def get_match_venue(home_en, away_en, kickoff_utc_iso):
    """从世界杯赛程板按赛日 + 队名匹配，返回 (球场名, 城市) 或 None。无需 key。"""
    date = datetime.strptime(kickoff_utc_iso, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc)
    # 跨时区，查赛日当天和前一天
    for delta in (0, -1, 1):
        ymd = (date + timedelta(days=delta)).strftime("%Y%m%d")
        try:
            data = _get(SCOREBOARD, {"dates": ymd})
        except EspnError:
            continue
        for ev in data.get("events", []):
            comp = ev["competitions"][0]
            names = [c["team"]["displayName"] for c in comp["competitors"]]
            if (any(_similar(home_en, n) >= 0.7 for n in names)
                    and any(_similar(away_en, n) >= 0.7 for n in names)):
                v = comp.get("venue", {})
                addr = v.get("address") or {}
                if v.get("fullName"):
                    return v["fullName"], addr.get("city", "")
    return None


def get_match_result(home_en, away_en, kickoff_utc_iso):
    """回填赛果：从主队赛程里按时间 + 对手名定位比赛。

    返回 (主队进球, 客队进球) —— 以我们库里的主客方向为准；找不到返回 None。
    依次尝试通用赛程和世界杯专属赛程。
    """
    team_id = _find_team_id(home_en)
    kickoff = datetime.strptime(kickoff_utc_iso, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )
    events = []
    for league in SCHEDULE_LEAGUES:
        url = (f"https://site.api.espn.com/apis/site/v2/sports/soccer/"
               f"{league}/teams/{team_id}/schedule")
        try:
            events.extend(_get(url).get("events", []))
        except EspnError as e:
            log.warning("ESPN %s 赛程查询失败: %s", league, e)
    for ev in events:
        comp = ev["competitions"][0]
        state = (comp.get("status", {}).get("type") or {}).get("state")
        evdt = _parse_espn_date(ev.get("date", ""))
        if state != "post" or evdt is None:
            continue
        if abs(evdt - kickoff) > timedelta(hours=6):
            continue
        sides = {c["homeAway"]: c for c in comp["competitors"]}
        h, a = sides.get("home"), sides.get("away")
        if not h or not a:
            continue
        h_name, a_name = h["team"]["displayName"], a["team"]["displayName"]
        try:
            h_goals = int(float(h["score"]["value"]))
            a_goals = int(float(a["score"]["value"]))
        except (KeyError, TypeError, ValueError):
            continue
        # 对手名校验 + 主客方向对齐
        if _similar(home_en, h_name) >= 0.7 and _similar(away_en, a_name) >= 0.7:
            return h_goals, a_goals
        if _similar(home_en, a_name) >= 0.7 and _similar(away_en, h_name) >= 0.7:
            return a_goals, h_goals  # ESPN 主客与我们相反，翻转比分
    return None
