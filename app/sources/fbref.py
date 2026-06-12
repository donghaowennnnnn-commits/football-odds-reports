"""FBref 球队数据：近 10 场赛果 + xG/xGA（如有）。

遵守 robots.txt（请求间隔 >= FBREF_MIN_DELAY 秒），失败抛异常由上层记日志。
"""
import logging
import time
import urllib.robotparser
from urllib.parse import quote, urljoin

import requests
from bs4 import BeautifulSoup

from config import FBREF_MIN_DELAY, USER_AGENT

BASE = "https://fbref.com"
log = logging.getLogger("fbref")

_robots = None
_last_request = 0.0


class FbrefError(Exception):
    pass


def _allowed(url):
    global _robots
    if _robots is None:
        _robots = urllib.robotparser.RobotFileParser(BASE + "/robots.txt")
        try:
            _robots.read()
        except Exception as e:
            log.warning("读取 fbref robots.txt 失败: %s（保守起见放行）", e)
            _robots.allow_all = True
    return _robots.can_fetch(USER_AGENT, url)


def _get(url, retries=3):
    global _last_request
    if not _allowed(url):
        raise FbrefError(f"robots.txt 不允许抓取 {url}")
    last_err = None
    for attempt in range(1, retries + 1):
        elapsed = time.monotonic() - _last_request
        if elapsed < FBREF_MIN_DELAY:
            time.sleep(FBREF_MIN_DELAY - elapsed)
        _last_request = time.monotonic()
        try:
            resp = requests.get(
                url, headers={"User-Agent": USER_AGENT}, timeout=30,
                allow_redirects=True,
            )
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 30))
                log.warning("fbref 限流 429，等待 %ds", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_err = e
            log.warning("fbref 请求失败 (第%d次): %s", attempt, e)
            time.sleep(2 ** attempt)
    raise FbrefError(f"fbref 请求 {url} 重试 {retries} 次仍失败: {last_err}")


def _find_team_url(team_en):
    """通过站内搜索找到球队页。搜索唯一命中时 fbref 会直接 302 到球队页。"""
    url = f"{BASE}/en/search/search.fcgi?search={quote(team_en)}"
    resp = _get(url)
    if "/squads/" in resp.url:
        return resp.url
    soup = BeautifulSoup(resp.text, "lxml")
    for div in soup.select("div.search-item-name"):
        a = div.find("a", href=True)
        if a and "/squads/" in a["href"]:
            return urljoin(BASE, a["href"])
    raise FbrefError(f"fbref 搜索 '{team_en}' 未找到球队页")


def get_recent_matches(team_en, limit=10):
    """返回 {team_url, matches: [{date, comp, venue, opponent, result, gf, ga, xg, xga}]}"""
    team_url = _find_team_url(team_en)
    resp = _get(team_url)
    soup = BeautifulSoup(resp.text, "lxml")
    table = soup.find("table", id=lambda x: x and x.startswith("matchlogs"))
    if table is None:
        raise FbrefError(f"球队页 {team_url} 没有 Scores & Fixtures 表")

    matches = []
    for tr in table.find("tbody").find_all("tr"):
        def cell(stat):
            td = tr.find(attrs={"data-stat": stat})
            return td.get_text(strip=True) if td else ""

        result = cell("result")
        if not result:  # 未踢的场次
            continue
        matches.append({
            "date": cell("date"),
            "comp": cell("comp"),
            "venue": cell("venue"),
            "opponent": cell("opponent"),
            "result": result,           # W / D / L
            "gf": cell("goals_for") or cell("gf"),
            "ga": cell("goals_against") or cell("ga"),
            "xg": cell("xg_for") or cell("xg") or None,
            "xga": cell("xg_against") or cell("xga") or None,
        })
    return {"team_url": team_url, "matches": matches[-limit:]}
