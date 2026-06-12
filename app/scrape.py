#!/usr/bin/env python3
"""单次抓取入口（launchd 每 30 分钟调用一次）。

实际是否抓取由内部规则决定:
  - 距开赛 > 48h: 跳过
  - 48h ~ 3h:    距上次抓取 >= 4 小时才抓
  - <= 3h:       距上次抓取 >= 30 分钟就抓
  - 已开赛:      标记 finished，不再抓

手动调试:
    python scrape.py --force            # 忽略频率限制立即抓全部
    python scrape.py --match-id 1       # 只抓某场
    python scrape.py --odds-only        # 只抓赔率不抓球队数据
"""
import argparse
import json
import logging
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler

import db
from config import (
    AUTO_TRACK_SPORTS, INTERVAL_FAR_MIN, INTERVAL_NEAR_MIN, LOG_DIR,
    ODDS_API_KEY, REPORT_WINDOW_HOURS, TEAM_STATS_INTERVAL_MIN,
    WINDOW_FAR_HOURS, WINDOW_NEAR_HOURS,
)
from sources import dongqiudi, espn, fbref, odds_api

log = logging.getLogger("scrape")


def setup_logging():
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    fh = RotatingFileHandler(
        LOG_DIR / "scrape.log", maxBytes=5 * 1024 * 1024, backupCount=3,
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(fh)
    root.addHandler(sh)


def parse_utc(iso):
    return datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def odds_due(match, now, conn):
    """根据距开赛时间和上次抓取时间判断这场比赛现在该不该抓赔率。"""
    kickoff = parse_utc(match["kickoff_utc"])
    hours_left = (kickoff - now).total_seconds() / 3600
    if hours_left > WINDOW_FAR_HOURS:
        return False, f"距开赛还有 {hours_left:.1f}h (>48h)，跳过"
    interval = INTERVAL_NEAR_MIN if hours_left <= WINDOW_NEAR_HOURS else INTERVAL_FAR_MIN
    last = db.last_snapshot_time(conn, match["id"])
    if last:
        minutes_since = (now - parse_utc(last)).total_seconds() / 60
        if minutes_since < interval:
            return False, (f"上次抓取在 {minutes_since:.0f} 分钟前 "
                           f"(当前频率每 {interval} 分钟)，跳过")
    return True, f"距开赛 {hours_left:.1f}h，应抓取"


def _resolve_event(conn, match):
    """补齐比赛的 sport_key / event_id，返回 (sport_key, event_id, home_en, away_en)。"""
    sport_key, event_id = match["sport_key"], match["odds_api_event_id"]
    home_en = match["home_team_en"] or odds_api.to_english(match["home_team"])
    away_en = match["away_team_en"] or odds_api.to_english(match["away_team"])
    if not event_id:
        found = odds_api.find_event(
            match["home_team"], match["away_team"], match["kickoff_utc"], sport_key,
        )
        if not found:
            return None
        sport_key, event_id, home_en, away_en = found
        db.set_event(conn, match["id"], sport_key, event_id, home_en, away_en)
    return sport_key, event_id, home_en, away_en


def _oddsportal_fallback(conn, match, home_en, away_en):
    from sources import oddsportal  # 延迟导入，playwright 未装时主路径不受影响
    rows = oddsportal.fetch_odds(home_en, away_en)
    ts = db.insert_snapshots(conn, match["id"], "oddsportal", rows)
    log.info("比赛 #%d 赔率入库 %d 行 (oddsportal @ %s)", match["id"], len(rows), ts)


def scrape_odds_batch(conn, due_matches, errors):
    """批量抓赔率：同一赛事的多场比赛合并为一次 API 调用（同价 3 额度）。

    返回 {match_id: 是否成功}。任何一场失败都单独降级处理，不互相影响。
    """
    ok = {}
    groups = {}  # sport_key -> [(match, event_id, home_en, away_en)]
    for match in due_matches:
        resolved = _resolve_event(conn, match) if ODDS_API_KEY else None
        if resolved:
            groups.setdefault(resolved[0], []).append((match, *resolved[1:]))
        else:
            # 无 key 或匹配失败 → 直接走 OddsPortal
            home_en = match["home_team_en"] or odds_api.to_english(match["home_team"])
            away_en = match["away_team_en"] or odds_api.to_english(match["away_team"])
            label = f"#{match['id']} {match['home_team']} vs {match['away_team']}"
            try:
                _oddsportal_fallback(conn, match, home_en, away_en)
                ok[match["id"]] = True
            except Exception as e:
                log.error("%s 赔率抓取失败: %s", label, e)
                errors.append(f"{label}: {e}")
                ok[match["id"]] = False

    for sport_key, items in groups.items():
        batch = None
        try:
            batch = odds_api.fetch_sport_odds(sport_key)
            log.info("批量抓取 [%s]: API 返回 %d 个事件，本轮需要其中 %d 场",
                     sport_key, len(batch), len(items))
        except odds_api.OddsApiError as e:
            log.error("批量抓取 [%s] 失败: %s，逐场降级 OddsPortal", sport_key, e)

        for match, event_id, home_en, away_en in items:
            label = f"#{match['id']} {match['home_team']} vs {match['away_team']}"
            rows = ((batch or {}).get(event_id) or {}).get("rows") or []
            if rows:
                ts = db.insert_snapshots(conn, match["id"], "odds_api", rows)
                log.info("比赛 #%d 赔率入库 %d 行 (odds_api @ %s)",
                         match["id"], len(rows), ts)
                ok[match["id"]] = True
                continue
            if batch is not None:
                log.warning("%s 不在批量结果中（可能尚未开盘），降级 OddsPortal", label)
            try:
                _oddsportal_fallback(conn, match, home_en, away_en)
                ok[match["id"]] = True
            except Exception as e:
                log.error("%s 两个赔率源都失败: %s", label, e)
                errors.append(f"{label}: {e}")
                ok[match["id"]] = False

        # 批量响应里的其他场次：免费数据顺带存档（passive），
        # 以后临时想跟某场，track.py add 会自动接上这些历史快照
        if batch:
            served = {eid for _, eid, _, _ in items}
            archived = 0
            for eid, ev in batch.items():
                if eid in served or not ev["rows"]:
                    continue
                existing = db.get_match_by_event(conn, eid)
                mid = existing["id"] if existing else db.add_passive_match(
                    conn, ev["home"], ev["away"], ev["commence"], sport_key, eid)
                if (existing and ev["commence"]
                        and existing["kickoff_utc"] != ev["commence"]):
                    db.update_kickoff(conn, mid, ev["commence"])
                    log.info("比赛 #%d 开赛时间已更新: %s → %s",
                             mid, existing["kickoff_utc"], ev["commence"])
                db.insert_snapshots(conn, mid, "odds_api", ev["rows"])
                archived += 1
            if archived:
                log.info("[%s] 顺带存档其他 %d 场的赔率快照（零额度成本）",
                         sport_key, archived)
    return ok


def scrape_team_stats(conn, match, now, force=False):
    """两队近况(ESPN 主源, FBref 备用补 xG) + 伤停(懂球帝尽力而为)。

    单队/单源失败不影响其他部分。
    """
    errors = []
    for team_zh, team_en in [
        (match["home_team"], match["home_team_en"] or odds_api.to_english(match["home_team"])),
        (match["away_team"], match["away_team_en"] or odds_api.to_english(match["away_team"])),
    ]:
        last = db.last_team_stats_time(conn, team_zh)
        if last and not force:
            minutes = (now - parse_utc(last)).total_seconds() / 60
            if minutes < TEAM_STATS_INTERVAL_MIN:
                log.info("%s 球队数据 %.0f 分钟前已抓，跳过", team_zh, minutes)
                continue
        recent, source = None, None
        try:
            recent = espn.get_recent_matches(team_en)
            source = "espn"
            log.info("%s: ESPN 近 %d 场已抓取", team_zh, len(recent["matches"]))
        except espn.EspnError as e:
            log.error("%s: ESPN 抓取失败: %s，尝试 FBref", team_zh, e)
            try:
                recent = fbref.get_recent_matches(team_en)
                source = "fbref"
                log.info("%s: FBref 近 %d 场已抓取", team_zh, len(recent["matches"]))
            except fbref.FbrefError as e2:
                log.error("%s: FBref 也失败: %s", team_zh, e2)
                errors.append(f"{team_zh}/近况: espn: {e}; fbref: {e2}")
        injuries = dongqiudi.get_injuries(team_zh)  # 失败返回 None 已记日志
        if recent or injuries:
            db.insert_team_stats(
                conn, team_zh,
                source or "dongqiudi",
                json.dumps(recent, ensure_ascii=False) if recent else None,
                json.dumps(injuries, ensure_ascii=False) if injuries else None,
            )
    return errors


def place_paper_bets(conn, match):
    """进入临场窗口的首轮：按当下最优价记录三笔模拟注单（幂等，一场一次）。"""
    if db.get_paper_bets(conn, match["id"]):
        return
    from analyze import analyze as run_analysis
    try:
        res = run_analysis(match["id"])
    except SystemExit:
        return
    label = f"#{match['id']} {match['home_team']} vs {match['away_team']}"
    for mk in ("ah", "ou"):
        rec = res["recs"].get(mk)
        if rec:
            db.insert_paper_bet(conn, match["id"], mk, rec["label"],
                                rec["bookmaker"], rec["line"], rec["side"],
                                rec["odds"], rec["ev"])
    if res["scores"]:
        top = res["scores"][0]
        # 波胆注额 = 其他玩法的 1/10（高赔率玩法小仓位，符合实际投注习惯）
        db.insert_paper_bet(conn, match["id"], "cs", top["score"], None,
                            None, None, round(top["fair"], 2), None, stake=0.1)

    # 平行实验组：检测到职业资金强信号（让球线移动）时，额外按资金方向
    # 落一注（该方向的最优价），与 EV 策略分账复盘——验证"该不该跟资金"
    flow = res.get("flow") or {}
    if flow.get("strong") and flow.get("direction") and res["ah"]:
        side = "home" if flow["direction"] > 0 else "away"
        cands = []
        for r in res["ah"]:
            if side == "home":
                cands.append((r["ev_home"], f"主让 {r['line']:+.2f}",
                              r["line"], r["home_odds"], r["bookmaker"]))
            else:
                cands.append((r["ev_away"], f"客受让 {(-r['line']) + 0.0:+.2f}",
                              r["line"], r["away_odds"], r["bookmaker"]))
        ev_, pick, line, odds_, bk = max(cands)
        db.insert_paper_bet(conn, match["id"], "ah", pick, bk, line, side,
                            odds_, ev_, strategy="flow")
        log.info("%s 顺资金策略加注: %s @ %.2f（资金方向%s）",
                 label, pick, odds_, "主队" if side == "home" else "客队")
    log.info("%s 模拟下注已落单（亚盘/大小球注额 1，波胆注额 0.1）", label)


def settle_paper_bets(conn):
    """有赛果的未结算注单全部结算。"""
    from analyze import settle_handicap, settle_total
    n = 0
    for b in db.unsettled_paper_bets(conn):
        diff = b["hs"] - b["aws"]
        total = b["hs"] + b["aws"]
        stake = b["stake"] if b["stake"] is not None else 1.0
        if b["market"] == "ah":
            result, pnl = settle_handicap(b["line"], b["side"], b["odds"], diff)
        elif b["market"] == "ou":
            result, pnl = settle_total(b["line"], b["side"], b["odds"], total)
        else:  # cs 波胆
            hit = b["pick"] == f"{b['hs']}-{b['aws']}"
            result = "赢" if hit else "输"
            pnl = round(b["odds"] - 1, 4) if hit else -1.0
        pnl = round(pnl * stake, 4)  # 盈亏按注额折算
        db.settle_paper_bet(conn, b["id"], result, pnl)
        n += 1
        log.info("模拟注单结算: 比赛#%d %s %s → %s (%+.2f)",
                 b["match_id"], b["market"], b["pick"], result, pnl)
    return n


def backfill_results(conn, now):
    """已完赛但没比分的比赛（含被动存档）回填赛果，每轮最多 10 场。

    赛果是校准验证（calibrate.py）的原料：模型预测 + 实际比分 = 可对账。
    """
    cutoff = (now - timedelta(hours=2, minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = db.matches_needing_result(conn, cutoff, limit=10)
    filled = []
    for m in rows:
        home_en = m["home_team_en"] or odds_api.to_english(m["home_team"])
        away_en = m["away_team_en"] or odds_api.to_english(m["away_team"])
        label = f"#{m['id']} {m['home_team']} vs {m['away_team']}"
        try:
            result = espn.get_match_result(home_en, away_en, m["kickoff_utc"])
        except espn.EspnError as e:
            log.warning("%s 赛果查询失败: %s（稍后重试）", label, e)
            db.bump_result_attempts(conn, m["id"])
            continue
        if result:
            db.set_result(conn, m["id"], result[0], result[1], "espn")
            log.info("%s 赛果回填: %d-%d", label, result[0], result[1])
            filled.append(m["id"])
        else:
            db.bump_result_attempts(conn, m["id"])
            log.warning("%s 暂未找到赛果（第 %d 次尝试，6 次后放弃）",
                        label, m["result_attempts"] + 1)
    return filled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="忽略频率限制")
    parser.add_argument("--match-id", type=int, help="只处理指定比赛")
    parser.add_argument("--odds-only", action="store_true", help="跳过球队数据")
    args = parser.parse_args()

    setup_logging()
    conn = db.connect()
    run_id = db.start_run(conn)
    now = datetime.now(timezone.utc)
    errors = []
    scraped = 0

    matches = db.list_matches(conn, status="tracking")
    n_manual = len(matches)
    if AUTO_TRACK_SPORTS:
        # 配置中的赛事整体自动密集跟踪（被动存档的场次升格参与调度）
        auto = [
            m for m in db.list_matches(conn, status="passive")
            if m["sport_key"] in AUTO_TRACK_SPORTS
            and parse_utc(m["kickoff_utc"]) > now
        ]
        matches = matches + auto
    if args.match_id:
        matches = [m for m in matches if m["id"] == args.match_id]
        if not matches:
            log.error("比赛 #%s 不存在或不在跟踪中", args.match_id)
    log.info("跟踪中的比赛: %d 场（手动 %d + 自动赛事跟踪 %d）",
             len(matches), n_manual, len(matches) - n_manual)

    # 第一遍：筛出本轮需要抓赔率的比赛
    due_matches = []
    for match in matches:
        label = f"#{match['id']} {match['home_team']} vs {match['away_team']}"
        if parse_utc(match["kickoff_utc"]) <= now:
            db.set_status(conn, match["id"], "finished")
            log.info("%s 已开赛，标记 finished", label)
            continue
        due, reason = (True, "强制抓取") if args.force else odds_due(match, now, conn)
        log.info("%s: %s", label, reason)
        if due:
            due_matches.append(match)

    # 第二遍：按赛事分组批量抓赔率（同赛事多场 = 一次调用）
    ok = scrape_odds_batch(conn, due_matches, errors) if due_matches else {}
    scraped = sum(1 for v in ok.values() if v)

    # 模拟下注：所有进入临场 3 小时窗口的比赛，首轮自动落单
    for match in due_matches:
        if not ok.get(match["id"]):
            continue
        hours_left = (parse_utc(match["kickoff_utc"]) - now).total_seconds() / 3600
        if hours_left <= WINDOW_NEAR_HOURS:
            try:
                place_paper_bets(conn, match)
            except Exception as e:
                log.error("比赛 #%d 模拟下注失败: %s", match["id"], e)
                errors.append(f"#{match['id']}/paper: {e}")

    # 第三遍：球队数据（仅重点场、本轮抓取过的）
    for match in due_matches:
        if match["status"] == "tracking" and not args.odds_only:
            errors.extend(scrape_team_stats(conn, match, now, force=args.force))

    # 自动报告：所有进入开赛前 24h 窗口的比赛每轮刷新——
    # 不依赖"本轮是否正式抓取"（被动存档的顺风车让快照始终新鲜，
    # 但也让远期比赛几乎不会触发正式抓取，故报告必须独立判断）
    for match in matches:
        kickoff = parse_utc(match["kickoff_utc"])
        if kickoff <= now:
            continue
        hours_left = (kickoff - now).total_seconds() / 3600
        if hours_left > REPORT_WINDOW_HOURS:
            continue
        label = f"#{match['id']} {match['home_team']} vs {match['away_team']}"
        try:
            from analyze import generate_html
            out, _ = generate_html(match["id"])
            log.info("%s 赛前解析报告(HTML)已更新: %s", label, out)
        except (Exception, SystemExit) as e:  # 无快照等情况跳过即可
            log.warning("%s 生成报告跳过: %s", label, e)

    # 第四遍：回填赛果 + 结算注单 + 为完赛比赛生成定稿报告（含比分和结算）
    if not args.odds_only:
        try:
            filled = backfill_results(conn, now)
            settle_paper_bets(conn)
            for mid in filled:
                try:
                    from analyze import generate_html
                    generate_html(mid)
                    log.info("比赛 #%d 完赛定稿报告已生成", mid)
                except (Exception, SystemExit) as e:
                    log.warning("比赛 #%d 定稿报告跳过: %s", mid, e)
        except Exception as e:
            log.error("赛果回填/结算异常: %s", e)
            errors.append(f"backfill: {e}")

    # 刷新总览主页（reports/index.html，系统唯一入口）
    try:
        from analyze import build_index
        build_index()
    except Exception as e:
        log.warning("刷新总览主页失败: %s", e)

    # 发布到公网（GitHub Pages）——失败只记日志，不影响主流程
    try:
        import subprocess
        from config import PROJECT_DIR
        r = subprocess.run(
            ["/bin/bash", str(PROJECT_DIR / "publish.sh")],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode != 0:
            log.warning("公网发布失败: %s", (r.stderr or r.stdout).strip()[:300])
    except Exception as e:
        log.warning("公网发布异常: %s", e)

    status = "ok" if not errors else ("partial" if scraped else "error")
    db.finish_run(conn, run_id, status, "; ".join(errors))
    log.info("本次运行结束: %s，抓取 %d 场，错误 %d 个", status, scraped, len(errors))
    if errors:
        for e in errors:
            log.error("  - %s", e)


if __name__ == "__main__":
    main()
