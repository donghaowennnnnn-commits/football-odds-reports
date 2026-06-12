"""The Odds API（主数据源）。

事件列表接口免费，赔率接口每次调用消耗 = 市场数（3）个额度。
额度余量记录在日志里（响应头 x-requests-remaining）。
"""
import difflib
import json
import logging
import subprocess
import time
from datetime import datetime, timedelta, timezone

import requests

from config import (
    BOOKMAKERS, LOG_DIR, MARKETS, ODDS_API_KEY, PROJECT_DIR,
    QUOTA_WARN_THRESHOLD, USER_AGENT,
)

BASE = "https://api.the-odds-api.com/v4"
log = logging.getLogger("odds_api")

# 中文队名 -> The Odds API 英文名（主要覆盖国家队；俱乐部可直接用英文名添加）
ZH_EN = {
    "墨西哥": "Mexico", "南非": "South Africa", "韩国": "South Korea",
    "捷克": "Czech Republic", "加拿大": "Canada", "波黑": "Bosnia & Herzegovina",
    "美国": "USA", "巴拉圭": "Paraguay", "卡塔尔": "Qatar", "瑞士": "Switzerland",
    "巴西": "Brazil", "摩洛哥": "Morocco", "海地": "Haiti", "苏格兰": "Scotland",
    "澳大利亚": "Australia", "土耳其": "Turkey", "德国": "Germany",
    "库拉索": "Curaçao", "荷兰": "Netherlands", "日本": "Japan",
    "科特迪瓦": "Ivory Coast", "厄瓜多尔": "Ecuador", "瑞典": "Sweden",
    "突尼斯": "Tunisia", "阿根廷": "Argentina", "法国": "France",
    "英格兰": "England", "西班牙": "Spain", "葡萄牙": "Portugal",
    "比利时": "Belgium", "意大利": "Italy", "克罗地亚": "Croatia",
    "乌拉圭": "Uruguay", "哥伦比亚": "Colombia", "智利": "Chile", "秘鲁": "Peru",
    "伊朗": "Iran", "沙特": "Saudi Arabia", "沙特阿拉伯": "Saudi Arabia",
    "乌兹别克斯坦": "Uzbekistan", "约旦": "Jordan", "伊拉克": "Iraq",
    "塞内加尔": "Senegal", "加纳": "Ghana", "尼日利亚": "Nigeria",
    "喀麦隆": "Cameroon", "埃及": "Egypt", "阿尔及利亚": "Algeria",
    "奥地利": "Austria", "波兰": "Poland", "丹麦": "Denmark", "挪威": "Norway",
    "威尔士": "Wales", "爱尔兰": "Ireland", "希腊": "Greece",
    "塞尔维亚": "Serbia", "乌克兰": "Ukraine", "俄罗斯": "Russia",
    "新西兰": "New Zealand", "巴拿马": "Panama", "哥斯达黎加": "Costa Rica",
    "洪都拉斯": "Honduras", "牙买加": "Jamaica", "佛得角": "Cape Verde",
    "斯洛文尼亚": "Slovenia", "斯洛伐克": "Slovakia", "罗马尼亚": "Romania",
    "匈牙利": "Hungary", "芬兰": "Finland", "冰岛": "Iceland", "阿联酋": "UAE",
    "刚果(金)": "DR Congo",
}

# 英文 -> 中文（显示用反向表）
EN_ZH = {}
for _zh, _en in ZH_EN.items():
    EN_ZH.setdefault(_en, _zh)
EN_ZH["Saudi Arabia"] = "沙特阿拉伯"


# 淘汰赛占位队名（对阵未定时 API 的写法），显示为"待定"
_TBD_PATTERNS = ("winner", "runner", "loser", "tbd", "qualifier",
                 "play-off", "playoff", "to be")


def to_chinese(name):
    """显示用：英文队名转中文；淘汰赛占位名显示"待定"；无对照时原样返回。"""
    n = (name or "").strip()
    if n in ZH_EN:
        return n
    low = n.lower()
    if any(p in low for p in _TBD_PATTERNS):
        return "待定"
    return EN_ZH.get(n, n)


class OddsApiError(Exception):
    pass


_QUOTA_MARKER = LOG_DIR / ".quota_warned"


def _notify_mac(title, text):
    """弹 macOS 系统通知（launchd 后台运行时也有效）。"""
    try:
        subprocess.run(
            ["osascript", "-e",
             f'display notification "{text}" with title "{title}"'],
            timeout=10, check=False,
        )
    except Exception as e:
        log.warning("发送系统通知失败: %s", e)


def _save_quota(remaining):
    """额度余量落盘（quota.json），供总览页展示。"""
    try:
        (PROJECT_DIR / "quota.json").write_text(json.dumps({
            "remaining": int(float(remaining)),
            "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }), encoding="utf-8")
    except Exception:
        pass


def _maybe_warn_quota(remaining):
    """额度低于阈值时弹通知 + 记 ERROR 日志，24 小时内最多提醒一次。"""
    try:
        rem = int(float(remaining))
    except (TypeError, ValueError):
        return
    if rem >= QUOTA_WARN_THRESHOLD:
        return
    if (_QUOTA_MARKER.exists()
            and time.time() - _QUOTA_MARKER.stat().st_mtime < 86400):
        return
    _QUOTA_MARKER.touch()
    days = rem / 80  # 全量密集模式日耗约 80
    log.error("The Odds API 额度仅剩 %d 次（警戒线 %d），按当前强度约可用 %.1f 天，"
              "请准备新 key（替换 .env 中的 ODDS_API_KEY 即可）",
              rem, QUOTA_WARN_THRESHOLD, days)
    _notify_mac("足球数据采集",
                f"API 额度仅剩 {rem} 次（约 {days:.1f} 天），请准备新 key 替换 .env")


def to_english(name):
    return ZH_EN.get(name.strip(), name.strip())


def _norm(name):
    return name.lower().replace("&", "and").replace("ç", "c").strip()


def _similar(a, b):
    a, b = _norm(a), _norm(b)
    if a == b or a in b or b in a:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _get(path, params, retries=3):
    params = dict(params, apiKey=ODDS_API_KEY)
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(
                f"{BASE}{path}", params=params,
                headers={"User-Agent": USER_AGENT}, timeout=30,
            )
            remaining = resp.headers.get("x-requests-remaining")
            if remaining is not None:
                log.info("The Odds API 额度剩余: %s", remaining)
                _maybe_warn_quota(remaining)
                _save_quota(remaining)
            if resp.status_code == 401:
                _notify_mac("足球数据采集",
                            "The Odds API 额度已耗尽或 key 失效，已切换备用数据源")
                raise OddsApiError("API key 无效或额度耗尽 (401)")
            resp.raise_for_status()
            return resp.json()
        except OddsApiError:
            raise
        except Exception as e:  # 网络错误/5xx，指数退避重试
            last_err = e
            wait = 2 ** attempt
            log.warning("请求 %s 失败 (第%d次): %s，%ds 后重试", path, attempt, e, wait)
            time.sleep(wait)
    raise OddsApiError(f"请求 {path} 重试 {retries} 次仍失败: {last_err}")


def list_soccer_sports():
    sports = _get("/sports", {})  # 免费接口
    return [s["key"] for s in sports if s["group"] == "Soccer" and s["active"]]


def find_event(home, away, kickoff_utc_iso, sport_key=None):
    """按队名 + 开赛时间匹配事件。返回 (sport_key, event_id, home_en, away_en) 或 None。

    /events 接口免费，所以可以扫多个联赛。
    """
    home_en, away_en = to_english(home), to_english(away)
    kickoff = datetime.strptime(kickoff_utc_iso, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )
    sport_keys = [sport_key] if sport_key else list_soccer_sports()

    near_misses = []
    for sk in sport_keys:
        try:
            events = _get(f"/sports/{sk}/events", {})
        except OddsApiError as e:
            log.warning("拉取 %s 赛程失败: %s", sk, e)
            continue
        for ev in events:
            commence = datetime.strptime(
                ev["commence_time"], "%Y-%m-%dT%H:%M:%SZ"
            ).replace(tzinfo=timezone.utc)
            if abs(commence - kickoff) > timedelta(hours=3):
                continue
            score = min(_similar(home_en, ev["home_team"]),
                        _similar(away_en, ev["away_team"]))
            swapped = min(_similar(home_en, ev["away_team"]),
                          _similar(away_en, ev["home_team"]))
            if max(score, swapped) >= 0.75:
                if swapped > score:
                    log.warning(
                        "主客场与 API 相反: 输入 %s vs %s，API %s vs %s（以 API 为准）",
                        home, away, ev["home_team"], ev["away_team"],
                    )
                return sk, ev["id"], ev["home_team"], ev["away_team"]
            near_misses.append(
                f"{sk}: {ev['home_team']} vs {ev['away_team']} @ {ev['commence_time']}"
            )
    if near_misses:
        log.error(
            "未匹配到事件 (%s vs %s @ %s)。时间窗口内的候选:\n  %s\n"
            "可用 track.py set-event <比赛ID> <sport_key> <event_id> 手动指定",
            home, away, kickoff_utc_iso, "\n  ".join(near_misses[:10]),
        )
    else:
        log.error("未匹配到事件，开赛时间 ±3 小时内无任何赛程 (%s vs %s)", home, away)
    return None


def fetch_odds(sport_key, event_id, home_en, away_en):
    """抓单场赔率，返回标准化行: [{bookmaker, market, line, home, draw, away}]"""
    data = _get(
        f"/sports/{sport_key}/events/{event_id}/odds",
        {
            "markets": MARKETS,
            "bookmakers": ",".join(BOOKMAKERS),
            "oddsFormat": "decimal",
        },
    )
    rows = _normalize_rows(data)
    if not rows:
        raise OddsApiError(f"事件 {event_id} 返回了空赔率（可能尚未开盘）")
    return rows


def fetch_sport_odds(sport_key):
    """批量抓整个赛事所有场次的赔率（与单场同价：3 个额度）。

    返回 {event_id: {home, away, commence, rows}}。
    跟踪同赛事多场比赛时用这个，额度消耗不随场次增加。
    """
    data = _get(
        f"/sports/{sport_key}/odds",
        {
            "markets": MARKETS,
            "bookmakers": ",".join(BOOKMAKERS),
            "oddsFormat": "decimal",
        },
    )
    return {
        ev["id"]: {
            "home": ev.get("home_team"),
            "away": ev.get("away_team"),
            "commence": ev.get("commence_time"),
            "rows": _normalize_rows(ev),
        }
        for ev in data
    }


def _normalize_rows(event):
    """把 API 的单个事件 JSON 转成标准化赔率行（队名取自事件本身）。"""
    home_en, away_en = event.get("home_team", ""), event.get("away_team", "")
    rows = []
    for bm in event.get("bookmakers", []):
        for market in bm.get("markets", []):
            mkey = market["key"]
            outcomes = market.get("outcomes", [])
            if mkey == "h2h":
                prices = {_norm(o["name"]): o["price"] for o in outcomes}
                rows.append({
                    "bookmaker": bm["key"], "market": "1x2", "line": None,
                    "home": prices.get(_norm(home_en)),
                    "draw": prices.get("draw"),
                    "away": prices.get(_norm(away_en)),
                })
            elif mkey == "spreads":
                # 同一让球线的主/客两条 outcome 配对（line 取主队让球数）
                by_line = {}
                for o in outcomes:
                    if o.get("point") is None:
                        continue
                    key = abs(o["point"])
                    by_line.setdefault(key, {})[_norm(o["name"])] = o
                for pair in by_line.values():
                    h, a = pair.get(_norm(home_en)), pair.get(_norm(away_en))
                    if h and a:
                        rows.append({
                            "bookmaker": bm["key"], "market": "ah",
                            "line": h["point"],
                            "home": h["price"], "draw": None, "away": a["price"],
                        })
            elif mkey == "totals":
                by_line = {}
                for o in outcomes:
                    if o.get("point") is None:
                        continue
                    by_line.setdefault(o["point"], {})[o["name"].lower()] = o["price"]
                for line, pair in by_line.items():
                    rows.append({
                        "bookmaker": bm["key"], "market": "ou", "line": line,
                        "home": pair.get("over"),  # home 列存大球赔率
                        "draw": None,
                        "away": pair.get("under"),  # away 列存小球赔率
                    })
    return rows
