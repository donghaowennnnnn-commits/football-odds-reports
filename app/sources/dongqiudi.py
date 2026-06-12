"""懂球帝伤停数据（尽力而为的备用源）。

懂球帝网页是 JS 渲染的 SPA，没有稳定的公开接口，这里尝试它的移动端
搜索/球队接口；任何一步失败都返回 None 并记 warning，不让整次抓取失败。
"""
import logging
import random
import time

import requests

from config import SCRAPE_DELAY_RANGE, USER_AGENT

log = logging.getLogger("dongqiudi")

SEARCH_API = "https://api.dongqiudi.com/search"


def _sleep():
    time.sleep(random.uniform(*SCRAPE_DELAY_RANGE))


def get_injuries(team_name_zh):
    """返回 [{player, reason, status}] 或 None（拿不到时）。"""
    try:
        _sleep()
        resp = requests.get(
            SEARCH_API,
            params={"keywords": team_name_zh, "type": "all", "page": 1},
            headers={"User-Agent": USER_AGENT},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        teams = data.get("teams") or []
        if not teams:
            log.warning("懂球帝搜索 '%s' 无球队结果，伤停数据缺失", team_name_zh)
            return None
        team_id = teams[0].get("team_id")
        _sleep()
        resp = requests.get(
            f"https://api.dongqiudi.com/data/team/{team_id}/injury",
            headers={"User-Agent": USER_AGENT},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        items = data.get("data") or []
        injuries = [
            {
                "player": it.get("person_name") or it.get("name"),
                "reason": it.get("reason") or it.get("injury_type"),
                "status": it.get("status") or it.get("return_date"),
            }
            for it in items
        ]
        return injuries
    except Exception as e:
        log.warning("懂球帝伤停抓取失败 (%s): %s —— 本次伤停数据缺失", team_name_zh, e)
        return None
