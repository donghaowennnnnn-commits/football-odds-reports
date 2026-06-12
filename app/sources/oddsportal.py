"""OddsPortal 降级源（Playwright 模拟真实浏览器）。

仅在 The Odds API 不可用时由 scrape.py 调用。需要额外安装：
    ./venv/bin/pip install playwright
    ./venv/bin/playwright install chromium

注意：OddsPortal 页面结构变动频繁，本模块按 2026-06 的页面结构编写，
解析失败时抛 OddsPortalError，由上层记日志。bet365 的赔率只能从这里获得。
"""
import logging
import random
import re
import time
from urllib.parse import quote

from config import SCRAPE_DELAY_RANGE, USER_AGENT

log = logging.getLogger("oddsportal")

BASE = "https://www.oddsportal.com"
TARGET_BOOKMAKERS = {"pinnacle", "bet365", "william hill", "unibet", "1xbet"}


class OddsPortalError(Exception):
    pass


def _require_playwright():
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
        return sync_playwright
    except ImportError:
        raise OddsPortalError(
            "未安装 playwright。启用 OddsPortal 降级源请执行:\n"
            "  ./venv/bin/pip install playwright && ./venv/bin/playwright install chromium"
        )


def _sleep():
    time.sleep(random.uniform(*SCRAPE_DELAY_RANGE))


def _norm_team(s):
    return re.sub(r"[^a-z]", "", s.lower())


def fetch_odds(home_en, away_en):
    """搜索比赛并抓取 1X2 / 亚让 / 大小球。返回与 odds_api.fetch_odds 同构的行。"""
    sync_playwright = _require_playwright()
    rows = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(user_agent=USER_AGENT, locale="en-US")
        page = ctx.new_page()
        try:
            # 1. 站内搜索找比赛页
            _sleep()
            page.goto(f"{BASE}/search/results/{quote(home_en)}/", timeout=60000)
            page.wait_for_load_state("networkidle", timeout=30000)
            match_url = None
            for a in page.locator("a[href*='/football/']").all():
                href = a.get_attribute("href") or ""
                text = _norm_team(a.inner_text())
                if _norm_team(home_en) in text and _norm_team(away_en) in text:
                    match_url = href if href.startswith("http") else BASE + href
                    break
            if not match_url:
                raise OddsPortalError(
                    f"OddsPortal 搜索未找到 {home_en} vs {away_en}"
                )
            log.info("OddsPortal 比赛页: %s", match_url)

            # 2. 三个盘口分别在不同 tab：#1X2、#ah（让球）、#over-under
            for fragment, market in [
                ("#1X2;2", "1x2"), ("#ah;2", "ah"), ("#over-under;2", "ou"),
            ]:
                _sleep()
                page.goto(match_url + fragment, timeout=60000)
                page.wait_for_load_state("networkidle", timeout=30000)
                rows.extend(_parse_market(page, market))
        finally:
            browser.close()
    if not rows:
        raise OddsPortalError("OddsPortal 页面解析结果为空（页面结构可能已变化）")
    return rows


def _parse_market(page, market):
    """解析当前页面上每家公司的一行赔率。"""
    out = []
    for row in page.locator("div[data-testid='over-under-expanded-row'],"
                            " div[data-testid='odd-container'],"
                            " div.border-black-borders.flex").all():
        try:
            text = row.inner_text().strip()
        except Exception:
            continue
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if len(lines) < 3:
            continue
        name = lines[0].lower()
        if not any(b in name for b in TARGET_BOOKMAKERS):
            continue
        nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", "\n".join(lines[1:]))]
        if market == "1x2" and len(nums) >= 3:
            out.append({"bookmaker": name, "market": "1x2", "line": None,
                        "home": nums[0], "draw": nums[1], "away": nums[2]})
        elif market in ("ah", "ou") and len(nums) >= 3:
            # 第一个数字是盘口线，后两个是两边赔率
            out.append({"bookmaker": name, "market": market, "line": nums[0],
                        "home": nums[1], "draw": None, "away": nums[2]})
    log.info("OddsPortal %s 解析到 %d 行", market, len(out))
    return out
