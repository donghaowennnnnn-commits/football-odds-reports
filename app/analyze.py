#!/usr/bin/env python3
"""盘口数学解析 + PDF 报告。

用法:
    python analyze.py <比赛ID>            # 生成 reports/match_<ID>_盘口解析.pdf

方法（纯数学，不引入主观判断）:
 1. 取 Pinnacle 最新 1X2 赔率，用幂法去水得公平胜平负概率
    （幂法对冷门方向去水更多，修正 favourite-longshot bias）
 2. 用泊松 + Dixon-Coles 低比分修正模型拟合 (λ主, λ客, ρ)，
    同时匹配公平 1X2 概率、让球盘零 EV、大小球零 EV 三组市场条件；
    ρ（低比分相关系数）由市场平局概率反推，不取经验值
 3. 由模型推导任意让球线/大小球线的公平赔率，与各家实盘对比算期望值(EV)
 4. 波胆 = 修正后比分矩阵的概率排名

局限: 模型校准基于 Pinnacle，对 Pinnacle 自身的盘 EV 必然接近 0，
价值信号主要体现在其他公司偏离 Pinnacle 的地方；伤停/阵容等信息不在模型内。
"""
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import db
from config import PROJECT_DIR, TZ
from sources.odds_api import to_chinese
from sources import espn, venues

MAX_GOALS = 12
# 锚点：Pinnacle（职业盘）+ Betfair 交易所（真实买卖撮合价），加权平均。
# 让球/大小球校准条件取第一个有数据的锚（Betfair 交易所无亚盘）。
ANCHOR_BOOKS = ["pinnacle", "betfair_ex_eu"]


def load_calibration():
    """读取 calibrate.py 写出的校准反馈（不存在或损坏时用默认值）。

    教训只有在样本量过统计门槛后才会写进这个文件并生效，
    单场比赛的偶然结果不会扭曲模型。
    """
    f = PROJECT_DIR / "calibration.json"
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


# ---------------- 概率模型 ----------------

def poisson_pmf(lam, k):
    return math.exp(-lam) * lam ** k / math.factorial(k)


def score_matrix(lh, la, rho=0.0):
    """泊松比分矩阵 + Dixon-Coles 低比分修正。

    ρ<0 时提高 0-0/1-1、降低 1-0/0-1 的概率（修正泊松独立性假设
    对低比分的系统性低估）。修正后重新归一化。
    """
    mat = [[poisson_pmf(lh, i) * poisson_pmf(la, j) for j in range(MAX_GOALS + 1)]
           for i in range(MAX_GOALS + 1)]
    if rho:
        mat[0][0] *= 1 - lh * la * rho
        mat[0][1] *= 1 + lh * rho
        mat[1][0] *= 1 + la * rho
        mat[1][1] *= 1 - rho
        total = sum(sum(row) for row in mat)
        mat = [[p / total for p in row] for row in mat]
    return mat


def outcome_probs(mat):
    ph = sum(mat[i][j] for i in range(MAX_GOALS + 1) for j in range(MAX_GOALS + 1) if i > j)
    pd = sum(mat[i][i] for i in range(MAX_GOALS + 1))
    pa = sum(mat[i][j] for i in range(MAX_GOALS + 1) for j in range(MAX_GOALS + 1) if i < j)
    return ph, pd, pa


def diff_dist(mat):
    """净胜球分布 P(主-客 = d)"""
    dist = {}
    for i in range(MAX_GOALS + 1):
        for j in range(MAX_GOALS + 1):
            dist[i - j] = dist.get(i - j, 0.0) + mat[i][j]
    return dist


def total_dist(mat):
    dist = {}
    for i in range(MAX_GOALS + 1):
        for j in range(MAX_GOALS + 1):
            dist[i + j] = dist.get(i + j, 0.0) + mat[i][j]
    return dist


def _half_lines(line):
    """四分之一盘拆成两个子盘，整盘/半盘返回自身。"""
    if abs(line * 4 - round(line * 4)) > 1e-9 or abs(line * 2 - round(line * 2)) < 1e-9:
        return [line]
    return [line - 0.25, line + 0.25]


def ev_handicap(ddist, line, odds, side):
    """让球盘 EV（每 1 单位投注）。side='home' 买主队 line（如 -1.25），
    side='away' 买客队接受 +line。"""
    evs = []
    for sub in _half_lines(line):
        win = push = lose = 0.0
        for d, p in ddist.items():
            margin = (d + sub) if side == "home" else -(d + sub)
            if margin > 1e-9:
                win += p
            elif margin < -1e-9:
                lose += p
            else:
                push += p
        evs.append(win * (odds - 1) - lose)
    return sum(evs) / len(evs)


def ev_total(tdist, line, odds, side):
    """大小球 EV。side='over'/'under'。"""
    evs = []
    for sub in _half_lines(line):
        win = lose = 0.0
        for t, p in tdist.items():
            margin = (t - sub) if side == "over" else (sub - t)
            if margin > 1e-9:
                win += p
            elif margin < -1e-9:
                lose += p
        evs.append(win * (odds - 1) - lose)
    return sum(evs) / len(evs)


def handicap_probs(ddist, line, side):
    """让球盘的 (赢盘, 走盘, 输盘) 概率，四分之一盘按两个子盘平均。"""
    win = push = lose = 0.0
    subs = _half_lines(line)
    for sub in subs:
        for d, p in ddist.items():
            margin = (d + sub) if side == "home" else -(d + sub)
            if margin > 1e-9:
                win += p
            elif margin < -1e-9:
                lose += p
            else:
                push += p
    n = len(subs)
    return win / n, push / n, lose / n


def total_probs(tdist, line, side):
    """大小球的 (赢, 走, 输) 概率。side='over'/'under'。"""
    win = push = lose = 0.0
    subs = _half_lines(line)
    for sub in subs:
        for t, p in tdist.items():
            margin = (t - sub) if side == "over" else (sub - t)
            if margin > 1e-9:
                win += p
            elif margin < -1e-9:
                lose += p
            else:
                push += p
    n = len(subs)
    return win / n, push / n, lose / n


def settle_handicap(line, side, odds, diff):
    """按实际净胜球结算让球注单（本金 1）。返回 (结果, 盈亏)。"""
    subs = _half_lines(line)
    pnl, marks = 0.0, []
    for sub in subs:
        margin = (diff + sub) if side == "home" else -(diff + sub)
        if margin > 1e-9:
            pnl += (odds - 1) / len(subs)
            marks.append("w")
        elif margin < -1e-9:
            pnl -= 1 / len(subs)
            marks.append("l")
        else:
            marks.append("p")
    return _settle_label(marks), round(pnl, 4)


def settle_total(line, side, odds, total):
    """按实际总进球结算大小球注单（本金 1）。"""
    subs = _half_lines(line)
    pnl, marks = 0.0, []
    for sub in subs:
        margin = (total - sub) if side == "over" else (sub - total)
        if margin > 1e-9:
            pnl += (odds - 1) / len(subs)
            marks.append("w")
        elif margin < -1e-9:
            pnl -= 1 / len(subs)
            marks.append("l")
        else:
            marks.append("p")
    return _settle_label(marks), round(pnl, 4)


def _settle_label(marks):
    if all(m == "w" for m in marks):
        return "赢"
    if all(m == "l" for m in marks):
        return "输"
    if all(m == "p" for m in marks):
        return "走"
    return "赢半" if "w" in marks else "输半"


def fair_odds(ev_func, dist, line, side):
    """解 EV(odds)=0 的公平赔率（二分法）。"""
    lo, hi = 1.001, 100.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if ev_func(dist, line, mid, side) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def power_demargin(odds_list):
    """幂法去水：解 k 使 Σ(1/o)^k = 1。

    比等比例法对低概率（高赔率）方向去水更多，
    修正庄家在冷门上加更厚水位的 favourite-longshot bias。
    """
    inv = [1 / o for o in odds_list]
    lo, hi = 0.5, 5.0
    for _ in range(80):
        k = (lo + hi) / 2
        if sum(p ** k for p in inv) > 1:
            lo = k
        else:
            hi = k
    k = (lo + hi) / 2
    probs = [p ** k for p in inv]
    s = sum(probs)
    return [p / s for p in probs]


def fit_model(p_home, p_draw, p_away,
              ah_line, fair_ah_home, ou_line, fair_over_odds):
    """网格搜索 (λ主, λ客, ρ)。

    约束: 公平 1X2 三概率 + 让球盘零 EV + 大小球零 EV。
    ρ 不取经验值，由市场（主要是平局概率）反推，限制在 [-0.35, 0.10]。
    """
    def loss(lh, la, rho):
        mat = score_matrix(lh, la, rho)
        ph, pd, pa = outcome_probs(mat)
        e = (ph - p_home) ** 2 + (pd - p_draw) ** 2 + (pa - p_away) ** 2
        if ou_line is not None:
            e += ev_total(total_dist(mat), ou_line, fair_over_odds, "over") ** 2
        if ah_line is not None:
            e += ev_handicap(diff_dist(mat), ah_line, fair_ah_home, "home") ** 2
        return e

    lh0, la0, r0 = 1.5, 1.0, -0.08
    step_l, step_r = 0.25, 0.08
    best = None
    for _ in range(4):  # 逐级细化
        candidates = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                for k in range(-3, 4):
                    lh = lh0 + i * step_l
                    la = la0 + j * step_l
                    rho = min(0.10, max(-0.35, r0 + k * step_r))
                    if lh < 0.05 or la < 0.05:
                        continue
                    candidates.append((loss(lh, la, rho), lh, la, rho))
        best, lh0, la0, r0 = min(candidates)
        step_l /= 3
        step_r /= 3
    return lh0, la0, r0


# ---------------- 数据读取 ----------------

def latest_odds(conn, match_id):
    """返回 (latest, history)。

    latest:  {market: {bookmaker: 最新一行}} —— EV 表用真实最新报价
    history: {(market, bookmaker): [按时间升序的所有行]} —— 模型校准用平滑值
    """
    rows = db.get_snapshots(conn, match_id)
    latest, history = {}, {}
    for r in rows:
        latest.setdefault(r["market"], {})[r["bookmaker"]] = r
        history.setdefault((r["market"], r["bookmaker"]), []).append(r)
    return latest, history


def smooth_quote(rows):
    """快照平滑：取与最新盘口线一致的最近 ≤3 帧加权平均（新帧权重大）。

    避免模型校准恰好撞上某家公司调价瞬间的单帧噪声。
    EV 表不用平滑值——那里比较的是此刻真实可成交的价格。
    """
    last = rows[-1]
    use = [r for r in rows if r["line"] == last["line"]][-3:]
    weights = [0.2, 0.3, 0.5][-len(use):]
    total_w = sum(weights)

    def avg(field):
        vals = [r[field] for r in use]
        if any(v is None for v in vals):
            return last[field]
        return sum(v * w for v, w in zip(vals, weights)) / total_w

    return {
        "line": last["line"], "fetched_at": last["fetched_at"],
        "home_odds": avg("home_odds"), "draw_odds": avg("draw_odds"),
        "away_odds": avg("away_odds"),
    }


def demargin(odds_list, method="power"):
    if method == "proportional":
        inv = [1 / o for o in odds_list]
        s = sum(inv)
        return [p / s for p in inv]
    return power_demargin(odds_list)


# ---------------- 依据链（自动生成解读文字） ----------------

def _team_form(conn, team):
    row = conn.execute(
        "SELECT * FROM team_stats WHERE team=? ORDER BY fetched_at DESC LIMIT 1",
        (team,),
    ).fetchone()
    if not row or not row["recent_matches_json"]:
        return None
    ms = json.loads(row["recent_matches_json"]).get("matches", [])
    if not ms:
        return None
    w = sum(1 for m in ms if m.get("result") == "W")
    d = sum(1 for m in ms if m.get("result") == "D")
    l = sum(1 for m in ms if m.get("result") == "L")

    def _int(v):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return 0

    gf = sum(_int(m.get("gf")) for m in ms)
    ga = sum(_int(m.get("ga")) for m in ms)
    return {"w": w, "d": d, "l": l, "gf": gf, "ga": ga,
            "points": 3 * w + d, "n": len(ms)}


def _build_fundamentals(conn, match):
    """基本面对比（不消耗赔率 API）：球场 / 天气 / 海拔 / 主场 / 攻防近况 +
    可选人工补充（战术/球星，notes/<id>.md）。全部为事实数据或可计算项。"""
    home, away = match["home_team"], match["away_team"]
    home_en = match["home_team_en"] or to_chinese(home)
    away_en = match["away_team_en"] or to_chinese(away)

    # 球场（缓存优先，否则查 ESPN 赛程板）
    vname, vcity = match["venue_name"], match["venue_city"]
    if not vname:
        try:
            got = espn.get_match_venue(home_en, away_en, match["kickoff_utc"])
            if got:
                vname, vcity = got
                db.set_venue(conn, match["id"], vname, vcity)
        except Exception:
            pass

    venue = venues.lookup(vname, vcity) if vname else None
    weather = None
    if venue:
        ymd = datetime.strptime(match["kickoff_utc"], "%Y-%m-%dT%H:%M:%SZ").strftime(
            "%Y-%m-%d")
        weather = venues.get_weather(venue["lat"], venue["lon"], ymd)

    # 主场 / 海拔优势（可计算）
    adv = []
    if venue:
        vc = venue.get("country")
        # 东道主在本国作战
        for team_en, team_zh, who in ((match["home_team_en"], home, "主队"),
                                      (match["away_team_en"], away, "客队")):
            if team_en and vc and team_en == vc:
                adv.append(f"{who} {to_chinese(team_zh)} 为东道主、本土作战，"
                           f"享主场氛围与无旅行/时差消耗")
        if venue["alt"] >= 1500:
            adv.append(f"高海拔球场（{venue['alt']} 米）：体能消耗与传球轨迹受影响，"
                       f"对非高原球队不利，墨西哥等高原球队适应更佳")
        if "顶棚" in (venue.get("roof") or "") and weather and \
                (weather.get("rain") or 0) >= 50:
            adv.append("可开合顶棚球场，雨天可闭顶，天气影响有限")

    # 攻防近况对比（优先用已抓的 team_stats；非重点场即时取 ESPN，免费）
    fh, fa = _team_form(conn, home), _team_form(conn, away)
    if not (fh and fa):
        for team_zh, team_en, slot in ((home, home_en, "h"), (away, away_en, "a")):
            if (slot == "h" and fh) or (slot == "a" and fa):
                continue
            try:
                rec = espn.get_recent_matches(team_en)
                db.insert_team_stats(conn, team_zh, "espn",
                                     json.dumps(rec, ensure_ascii=False), None)
            except Exception:
                pass
        fh, fa = _team_form(conn, home), _team_form(conn, away)
    form_cmp = None
    if fh and fa:
        form_cmp = {
            "home": {"name": to_chinese(home), **fh,
                     "agf": fh["gf"] / fh["n"], "aga": fh["ga"] / fh["n"]},
            "away": {"name": to_chinese(away), **fa,
                     "agf": fa["gf"] / fa["n"], "aga": fa["ga"] / fa["n"]},
        }

    # 人工补充（战术/球星等编辑性内容，纯数据回路无法生成，可选）
    note_md = None
    nf = PROJECT_DIR / "notes" / f"{match['id']}.md"
    if nf.exists():
        note_md = nf.read_text(encoding="utf-8").strip()

    return {"venue": venue, "weather": weather, "advantages": adv,
            "form_cmp": form_cmp, "note_md": note_md}


def _build_basis(conn, match, history, anchors, p_home, p_away):
    """从快照历史 + 球队数据生成依据链文字，按权重分层。"""
    home, away = match["home_team"], match["away_team"]

    # ① 市场锚点（主依据）
    anchor_lines = []
    for bk, probs, mg, _ in anchors:
        anchor_lines.append(
            f"{bk}: 主胜 {probs[0]:.1%} / 平 {probs[1]:.1%} / 客胜 {probs[2]:.1%}"
            f"（抽水 {mg:.1f}%）")
    if len(anchors) >= 2:
        gap = abs(anchors[0][1][0] - anchors[1][1][0]) * 100
        verdict = ("两锚点高度一致，市场共识可信度高" if gap < 2 else
                   "两锚点存在分歧，结论的不确定性高于平时")
        anchor_lines.append(f"做市商与交易所的主胜概率差 {gap:.1f} 个百分点——{verdict}")

    # ② 盘口走势（解读层）
    n_snaps = conn.execute(
        "SELECT COUNT(DISTINCT fetched_at) c FROM odds_snapshots WHERE match_id=?",
        (match["id"],)).fetchone()["c"]
    trend_lines = [f"观察窗口内共 {n_snaps} 个快照时点："]
    t = history.get(("1x2", "pinnacle"))
    if t and len(t) >= 2 and t[0]["home_odds"] and t[-1]["home_odds"]:
        f0, l0 = t[0], t[-1]
        pct = (l0["home_odds"] / f0["home_odds"] - 1) * 100
        if pct < -0.5:
            d = "走低——资金温和流向主队"
        elif pct > 0.5:
            d = "走高——市场对主队降温"
        else:
            d = "基本持平——市场判断稳定"
        trend_lines.append(
            f"Pinnacle 主胜赔率 {f0['home_odds']:.2f} → {l0['home_odds']:.2f}"
            f"（{pct:+.1f}%），{d}")
    t = history.get(("ah", "pinnacle"))
    if t and len(t) >= 2:
        f0, l0 = t[0], t[-1]
        if f0["line"] != l0["line"]:
            d = "升盘，职业资金压主队" if (l0["line"] or 0) < (f0["line"] or 0) \
                else "退盘，对主队信心减弱"
            trend_lines.append(
                f"让球线 {f0['line']:+.2f} → {l0['line']:+.2f}（{d}——这是快照序列里"
                f"权重最高的信号）")
        elif f0["home_odds"] and l0["home_odds"]:
            wp = (l0["home_odds"] / f0["home_odds"] - 1) * 100
            trend_lines.append(
                f"让球线稳定在 {l0['line']:+.2f}，主队水位 {f0['home_odds']:.2f} → "
                f"{l0['home_odds']:.2f}（{wp:+.1f}%，{'向主队方向收紧' if wp < -0.5 else ('向客队方向放宽' if wp > 0.5 else '无方向性变化')}）")
    if len(trend_lines) == 1:
        trend_lines.append("快照尚少，走势信号待积累（临场 3 小时为密集观察期）")

    # ③ 纸面近况（辅助层，不进入计算）
    form_lines = []
    fh, fa = _team_form(conn, home), _team_form(conn, away)
    consistency = None
    if fh and fa:
        form_lines.append(
            f"{home}: 近 {fh['n']} 场 {fh['w']}胜{fh['d']}平{fh['l']}负，"
            f"进 {fh['gf']} 失 {fh['ga']}（积 {fh['points']} 分）")
        form_lines.append(
            f"{away}: 近 {fa['n']} 场 {fa['w']}胜{fa['d']}平{fa['l']}负，"
            f"进 {fa['gf']} 失 {fa['ga']}（积 {fa['points']} 分）")
        market_fav_home = p_home > p_away
        form_fav_home = fh["points"] > fa["points"]
        if market_fav_home == form_fav_home:
            consistency = ("纸面近况与市场方向<b>一致</b>，相互印证；"
                           "近况信息已包含在价格中，不重复计入模型。")
        else:
            consistency = ("纸面近况与市场方向<b>背离</b>——市场坚持的判断里"
                           "含有近况之外的信息（伤停、轮换、未公开情报），"
                           "此时应以市场为准，并留意临场盘口异动。")
    else:
        form_lines.append("本场未抓取球队近况（非手动跟踪的重点场），"
                          "可用 track.py add 升级后获得")

    return {
        "anchor": anchor_lines,
        "trend": trend_lines,
        "form": form_lines,
        "consistency": consistency,
    }


def _pct(a, b):
    if not a or not b:
        return None
    return (b / a - 1) * 100


def _money_flow(history):
    """从快照序列做规则化的资金流向 + 庄家行为解读。

    全部为价格行为的间接推断（真实投注量不可见），阈值规则：
      - 让球线移动 = 强信号；水位/赔率变动 >1% = 温和信号；以下为噪声
      - 软庄滞后 = 锚点动 >1% 而它 <0.3%；逆共识 = 与锚点反向且 >1.5%
    """
    flow = {"sharp": [], "behavior": [], "verdict": "", "lag_books": set()}

    def first_last(mk, bk):
        rows = history.get((mk, bk))
        if not rows or len(rows) < 2:
            return None, None
        return rows[0], rows[-1]

    direction = 0  # 正=资金偏向主队，负=偏向客队

    # 1) Pinnacle 亚盘（最强信号源）
    f, l = first_last("ah", "pinnacle")
    if f and l:
        if f["line"] != l["line"]:
            if (l["line"] or 0) < (f["line"] or 0):
                flow["sharp"].append(
                    f"Pinnacle 让球线 {f['line']:+.2f} → {l['line']:+.2f}（升盘）"
                    f"——职业资金重仓主队的强信号")
                direction += 2
            else:
                flow["sharp"].append(
                    f"Pinnacle 让球线 {f['line']:+.2f} → {l['line']:+.2f}（退盘）"
                    f"——职业资金看衰主队的强信号")
                direction -= 2
        else:
            wp = _pct(f["home_odds"], l["home_odds"])
            if wp is not None and wp < -1:
                flow["sharp"].append(
                    f"让球线稳定 {l['line']:+.2f}，主队水位收紧 {wp:+.1f}%"
                    f"——资金温和流向主队")
                direction += 1
            elif wp is not None and wp > 1:
                flow["sharp"].append(
                    f"让球线稳定 {l['line']:+.2f}，主队水位放宽 {wp:+.1f}%"
                    f"——资金温和流向客队方向")
                direction -= 1
            else:
                flow["sharp"].append("亚盘线与水位均稳定——无方向性资金动作")

    # 2) Pinnacle 1X2 佐证
    f, l = first_last("1x2", "pinnacle")
    pin_pct = None
    if f and l:
        pin_pct = _pct(f["home_odds"], l["home_odds"])
        if pin_pct is not None and abs(pin_pct) > 1:
            d = "走低（向主队）" if pin_pct < 0 else "走高（对主队降温）"
            flow["sharp"].append(f"Pinnacle 主胜赔率 {f['home_odds']:.2f} → "
                                 f"{l['home_odds']:.2f}（{pin_pct:+.1f}%，{d}）")
            direction += -1 if pin_pct > 0 else 1

    # 3) Betfair 交易所确认（交易所价格 = 真实成交资金）
    f, l = first_last("1x2", "betfair_ex_eu")
    if f and l and pin_pct is not None:
        bf_pct = _pct(f["home_odds"], l["home_odds"])
        if bf_pct is not None and abs(pin_pct) > 1:
            if bf_pct * pin_pct > 0:
                flow["sharp"].append("Betfair 交易所同向移动——资金流向获独立确认，"
                                     "可信度高")
            else:
                flow["sharp"].append("Betfair 交易所未同步——Pinnacle 的移动可能是"
                                     "风控调价而非资金驱动，降级处理")

    # 综合判定（“明确”一词只授予让球线移动这种强信号）
    line_moved = any("升盘" in s or "退盘" in s for s in flow["sharp"])
    flow["direction"] = direction   # >0 资金偏主队, <0 偏客队
    flow["strong"] = line_moved
    if direction > 0:
        side_txt = "主队"
    elif direction < 0:
        side_txt = "客队方向（对主队降温）"
    if direction == 0:
        flow["verdict"] = "无明显方向性资金"
    elif line_moved:
        flow["verdict"] = f"职业资金明确流向{side_txt}（含让球线移动的强信号）"
    elif abs(direction) >= 2:
        flow["verdict"] = f"多个温和信号一致：资金偏向{side_txt}"
    else:
        flow["verdict"] = f"资金温和偏向{side_txt}"

    # 4) 软庄行为检测（滞后 / 逆共识 / 同步）
    for (mk, bk), rows in history.items():
        if mk != "1x2" or bk in ANCHOR_BOOKS or len(rows) < 2:
            continue
        s_pct = _pct(rows[0]["home_odds"], rows[-1]["home_odds"])
        if s_pct is None or pin_pct is None:
            continue
        if abs(pin_pct) > 1 and abs(s_pct) < 0.3:
            flow["behavior"].append(
                f"{bk}: 锚点移动 {pin_pct:+.1f}% 期间按兵不动——滞后跟盘，"
                f"其旧价值得对照 EV 表检查")
            flow["lag_books"].add(bk)
        elif abs(s_pct) > 1.5 and s_pct * pin_pct < 0:
            flow["behavior"].append(
                f"{bk}: 与市场共识反向调价（{s_pct:+.1f}% vs 锚点 {pin_pct:+.1f}%）"
                f"——疑似诱导性定价或本地风控，对其报价保持警惕")
    if not flow["behavior"]:
        flow["behavior"].append("各软庄与锚点基本同步，未检测到滞后或逆共识行为")

    return flow


# ---------------- 分析主流程 ----------------

def analyze(match_id):
    conn = db.connect()
    match = db.get_match(conn, match_id)
    if not match:
        sys.exit(f"比赛 #{match_id} 不存在")
    odds, history = latest_odds(conn, match_id)
    if "1x2" not in odds:
        sys.exit("没有 1X2 赔率快照，先跑 scrape.py")

    cfg = load_calibration()
    devig = cfg.get("devig_method", "power")
    anchor_w = cfg.get("anchor_weights", {})

    # --- 1X2 锚点：Pinnacle + Betfair 交易所加权（平滑后的报价） ---
    anchors = []  # (bookmaker, [ph,pd,pa], margin, fetched_at)
    for bk in ANCHOR_BOOKS:
        rows = history.get(("1x2", bk))
        if not rows:
            continue
        sm = smooth_quote(rows)
        quotes = [sm["home_odds"], sm["draw_odds"], sm["away_odds"]]
        if any(q is None for q in quotes):
            continue
        anchors.append((bk, demargin(quotes, devig),
                        (sum(1 / q for q in quotes) - 1) * 100, sm["fetched_at"]))
    if not anchors:  # 锚点公司都没数据时退化用任意一家
        bk = sorted(odds["1x2"])[0]
        r = odds["1x2"][bk]
        quotes = [r["home_odds"], r["draw_odds"], r["away_odds"]]
        anchors.append((bk, demargin(quotes, devig),
                        (sum(1 / q for q in quotes) - 1) * 100, r["fetched_at"]))

    total_w = sum(anchor_w.get(bk, 1.0) for bk, *_ in anchors)
    blend = [
        sum(probs[i] * anchor_w.get(bk, 1.0) for bk, probs, _, _ in anchors) / total_w
        for i in range(3)
    ]
    s = sum(blend)
    p_home, p_draw, p_away = (p / s for p in blend)
    margin = anchors[0][2]            # 显示主锚（通常 Pinnacle）的抽水
    anchor_desc = "+".join(bk for bk, *_ in anchors)
    anchor_fetched = max(ts for *_, ts in anchors)

    # --- 让球/大小球校准条件：第一个有数据的锚（交易所无亚盘，通常是 Pinnacle）---
    base_book = next(
        (bk for bk in ANCHOR_BOOKS
         if odds.get("ah", {}).get(bk) or odds.get("ou", {}).get(bk)),
        None,
    )
    ou_line = fair_over = None
    ah_line = fair_ah_home = None
    if base_book:
        rows = history.get(("ou", base_book))
        if rows:
            sm = smooth_quote(rows)
            if sm["home_odds"] and sm["away_odds"]:
                ou_line = sm["line"]
                po, _pu = demargin([sm["home_odds"], sm["away_odds"]], devig)
                fair_over = 1 / po
        rows = history.get(("ah", base_book))
        if rows:
            sm = smooth_quote(rows)
            if sm["home_odds"] and sm["away_odds"]:
                ah_line = sm["line"]
                pah, _ = demargin([sm["home_odds"], sm["away_odds"]], devig)
                fair_ah_home = 1 / pah

    lh, la, rho = fit_model(p_home, p_draw, p_away,
                            ah_line, fair_ah_home, ou_line, fair_over)
    mat = score_matrix(lh, la, rho)
    mh, md, ma = outcome_probs(mat)
    ddist, tdist = diff_dist(mat), total_dist(mat)

    # 各家让球盘/大小球 EV
    ah_rows, ou_rows = [], []
    for bk, r in sorted(odds.get("ah", {}).items()):
        ah_rows.append({
            "bookmaker": bk, "line": r["line"],
            "home_odds": r["home_odds"], "away_odds": r["away_odds"],
            "fair_home": fair_odds(ev_handicap, ddist, r["line"], "home"),
            "fair_away": fair_odds(ev_handicap, ddist, r["line"], "away"),
            "ev_home": ev_handicap(ddist, r["line"], r["home_odds"], "home") * 100,
            "ev_away": ev_handicap(ddist, r["line"], r["away_odds"], "away") * 100,
        })
    for bk, r in sorted(odds.get("ou", {}).items()):
        ou_rows.append({
            "bookmaker": bk, "line": r["line"],
            "over_odds": r["home_odds"], "under_odds": r["away_odds"],
            "fair_over": fair_odds(ev_total, tdist, r["line"], "over"),
            "fair_under": fair_odds(ev_total, tdist, r["line"], "under"),
            "ev_over": ev_total(tdist, r["line"], r["home_odds"], "over") * 100,
            "ev_under": ev_total(tdist, r["line"], r["away_odds"], "under") * 100,
        })

    # 各玩法推荐：全场所有(线, 方向, 公司)组合里 EV 最优的入口
    recs = {}
    if ah_rows:
        cands = []
        for r in ah_rows:
            cands.append(("home", f"主让 {r['line']:+.2f}", r["line"],
                          r["ev_home"], r["home_odds"], r["bookmaker"]))
            cands.append(("away", f"客受让 {(-r['line']) + 0.0:+.2f}", r["line"],
                          r["ev_away"], r["away_odds"], r["bookmaker"]))
        side, label, line, ev, best_odds, bk = max(cands, key=lambda x: x[3])
        w, p, l = handicap_probs(ddist, line, side)
        recs["ah"] = {"label": label, "ev": ev, "odds": best_odds, "bookmaker": bk,
                      "win": w, "push": p, "lose": l, "side": side, "line": line}
    if ou_rows:
        cands = []
        for r in ou_rows:
            cands.append(("over", f"大 {r['line']:.2f}", r["line"],
                          r["ev_over"], r["over_odds"], r["bookmaker"]))
            cands.append(("under", f"小 {r['line']:.2f}", r["line"],
                          r["ev_under"], r["under_odds"], r["bookmaker"]))
        side, label, line, ev, best_odds, bk = max(cands, key=lambda x: x[3])
        w, p, l = total_probs(tdist, line, side)
        recs["ou"] = {"label": label, "ev": ev, "odds": best_odds, "bookmaker": bk,
                      "win": w, "push": p, "lose": l, "side": side, "line": line}

    # 波胆 top 8
    scores = sorted(
        ((mat[i][j], i, j) for i in range(7) for j in range(7)),
        reverse=True,
    )[:8]
    score_rows = [
        {"score": f"{i}-{j}", "prob": p * 100, "fair": 1 / p}
        for p, i, j in scores
    ]

    basis = _build_basis(conn, match, history, anchors, p_home, p_away)
    flow = _money_flow(history)

    # 推荐 × 资金方向 一致性核对（亚盘）：
    # 数学入口选价格，资金方向是另一根轴——冲突时显性警示而非暗中加权
    ah_rec = recs.get("ah")
    if ah_rec and flow["direction"] != 0:
        rec_home = ah_rec["side"] == "home"
        flow_home = flow["direction"] > 0
        strength = "强信号·让球线移动" if flow["strong"] else "温和信号"
        if rec_home == flow_home:
            ah_rec["flow_note"] = f"✓ 与职业资金方向一致（{strength}），两轴互证"
            ah_rec["flow_align"] = True
        else:
            ah_rec["flow_note"] = (
                f"⚠ 与职业资金方向背离（资金偏向{'主队' if flow_home else '客队'}，"
                f"{strength}）——本推荐取“价格最优”，方向派的成绩见顺资金策略复盘")
            ah_rec["flow_align"] = False
    paper_bets = [dict(b) for b in db.get_paper_bets(conn, match_id)]
    fundamentals = _build_fundamentals(conn, match)

    # 走势序列（HTML 报告的曲线图用）
    series = {}
    for key, name in [
        (("1x2", "pinnacle"), "Pinnacle 主胜赔率"),
        (("1x2", "betfair_ex_eu"), "Betfair 主胜赔率"),
        (("ah", "pinnacle"), "Pinnacle 亚盘主队水位"),
    ]:
        rows = history.get(key)
        if rows:
            pts = [(r["fetched_at"], r["home_odds"]) for r in rows if r["home_odds"]]
            if len(pts) >= 2:
                series[name] = pts

    return {
        "match": match, "fetched_at": anchor_fetched, "base_book": anchor_desc,
        "basis": basis, "flow": flow, "paper_bets": paper_bets, "series": series,
        "fundamentals": fundamentals,
        "margin": margin,
        "devig": devig,
        "calibration_n": cfg.get("n_matches"),
        "calibration_date": cfg.get("updated"),
        "fair_1x2": (p_home, p_draw, p_away),
        "model_1x2": (mh, md, ma),
        "lambdas": (lh, la),
        "rho": rho,
        "exp_total": lh + la,
        "ah": ah_rows, "ou": ou_rows, "scores": score_rows, "recs": recs,
        "odds_1x2": odds["1x2"],
    }


# ---------------- PDF 输出 ----------------

def fmt_local(utc_iso):
    dt = datetime.strptime(utc_iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return dt.astimezone(TZ).strftime("%Y-%m-%d %H:%M")


# ---------------- HTML 主报告（赛博朋克风横版仪表盘） ----------------

CYBER_CSS = """
:root{--bg:#070b18;--card:#0d1430;--card2:#0a102a;--line:rgba(0,212,255,.18);
--cyan:#00d4ff;--mag:#ff2e88;--vio:#8b5cf6;--ink:#dfe7ff;--mut:#7a86ad;
--good:#2bd97c;--warn:#ffb454;--bad:#ff5d6c}
*{box-sizing:border-box;margin:0;padding:0}
body{background:radial-gradient(1200px 600px at 80% -10%,rgba(255,46,136,.13),transparent 60%),
radial-gradient(1000px 500px at -10% 10%,rgba(0,212,255,.12),transparent 55%),var(--bg);
color:var(--ink);font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;
padding:26px 32px;min-width:1100px}
.hero{border:1px solid var(--line);border-radius:18px;padding:24px 30px;
background:linear-gradient(135deg,rgba(0,212,255,.10),rgba(255,46,136,.08) 60%,transparent);
box-shadow:0 0 28px rgba(0,212,255,.10),inset 0 0 60px rgba(0,0,0,.35)}
.tag{font-size:11px;letter-spacing:.35em;color:var(--cyan);margin-bottom:10px}
h1{font-size:32px;font-weight:800}
h1 .vs{color:var(--mag);font-size:20px;margin:0 12px;text-shadow:0 0 14px rgba(255,46,136,.8)}
.chips{margin-top:12px;display:flex;gap:10px;flex-wrap:wrap}
.chip{font-size:12px;color:var(--mut);border:1px solid var(--line);border-radius:999px;
padding:4px 12px;background:rgba(0,0,0,.25)}
.chip b{color:var(--ink)}
.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:18px;margin-top:18px}
.card{grid-column:span 6;border:1px solid var(--line);border-radius:16px;
background:linear-gradient(180deg,var(--card),var(--card2));padding:18px 20px;
box-shadow:0 0 18px rgba(0,212,255,.06)}
.card.w12{grid-column:span 12}
.ct{display:flex;align-items:center;gap:10px;margin-bottom:12px}
.ct .ico{width:26px;height:26px;border-radius:8px;display:flex;align-items:center;
justify-content:center;background:linear-gradient(135deg,var(--cyan),var(--vio));
color:#04101e;font-weight:700;font-size:13px;box-shadow:0 0 12px rgba(0,212,255,.55)}
.ct h2{font-size:15px;letter-spacing:.06em}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th{color:#03142a;background:linear-gradient(90deg,rgba(0,212,255,.95),rgba(139,92,246,.9));
padding:7px 8px;text-align:center;font-weight:700}
th:first-child{border-radius:8px 0 0 8px}th:last-child{border-radius:0 8px 8px 0}
td{padding:7px 8px;text-align:center;border-bottom:1px solid rgba(122,134,173,.15)}
td:first-child{text-align:left;color:var(--mut)}
tr:hover td{background:rgba(0,212,255,.05)}
.pos{color:var(--good);font-weight:700}.neg{color:var(--bad)}
.rec{margin-top:12px;border:1px dashed rgba(0,212,255,.4);border-radius:12px;
padding:10px 14px;font-size:13px;background:rgba(0,212,255,.05);line-height:1.7}
.rec .pick{color:var(--cyan);font-weight:800;text-shadow:0 0 10px rgba(0,212,255,.6)}
.badge{display:inline-block;border-radius:999px;padding:2px 10px;font-size:11px;
font-weight:700;margin-left:8px}
.b-good{background:rgba(43,217,124,.15);color:var(--good);border:1px solid rgba(43,217,124,.5)}
.b-mid{background:rgba(0,212,255,.12);color:var(--cyan);border:1px solid rgba(0,212,255,.45)}
.b-warn{background:rgba(255,180,84,.12);color:var(--warn);border:1px solid rgba(255,180,84,.5)}
ul{list-style:none}
li{font-size:12.5px;padding:4px 0 4px 16px;position:relative;line-height:1.6}
li:before{content:"";position:absolute;left:2px;top:11px;width:6px;height:6px;
border-radius:50%;background:var(--cyan);box-shadow:0 0 8px var(--cyan)}
.mut{color:var(--mut)}
.small{font-size:11px;color:var(--mut);line-height:1.65;margin-top:10px}
.kpis{display:flex;gap:14px;margin:12px 0 2px}
.kpi{flex:1;border:1px solid var(--line);border-radius:12px;padding:10px;
text-align:center;background:rgba(0,0,0,.22)}
.kpi .v{font-size:21px;font-weight:800;color:var(--cyan);text-shadow:0 0 12px rgba(0,212,255,.5)}
.kpi .l{font-size:11px;color:var(--mut);margin-top:3px}
.verdict{font-size:14px;font-weight:700;color:var(--mag);
text-shadow:0 0 12px rgba(255,46,136,.5);margin-bottom:8px}
.sub{font-size:12.5px;font-weight:700;color:var(--cyan);margin:12px 0 4px}
.legend{display:flex;gap:16px;font-size:11px;color:var(--mut);margin-top:6px}
.legend i{display:inline-block;width:18px;height:3px;border-radius:2px;margin-right:5px;
vertical-align:middle}
.footer{margin-top:20px;font-size:11px;color:var(--mut);line-height:1.7;
border-top:1px solid var(--line);padding-top:12px}
svg text{font-family:inherit}
.cols3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:18px}
.cols2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.card{overflow-x:auto}
@media (max-width:900px){
  body{min-width:0;padding:14px}
  h1{font-size:22px}
  .grid{grid-template-columns:1fr}
  .card,.card.w12{grid-column:span 1}
  .cols3,.cols2{grid-template-columns:1fr}
  .kpis{flex-wrap:wrap}.kpi{min-width:40%}
}
"""

CHART_COLORS = ["#00d4ff", "#ff2e88", "#8b5cf6"]


def _svg_chart(series_list, width=640, height=200):
    """多条折线的霓虹走势图（纯 SVG，无 JS 依赖）。"""
    def ep(ts):
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").timestamp()

    all_x = [ep(t) for _, _, pts in series_list for t, _ in pts]
    all_y = [v for _, _, pts in series_list for _, v in pts]
    x0, x1 = min(all_x), max(all_x)
    y0, y1 = min(all_y), max(all_y)
    if x1 - x0 < 1:
        x1 = x0 + 1
    pad = (y1 - y0) * 0.2 or 0.04
    y0, y1 = y0 - pad, y1 + pad
    L, R, T, B = 44, 14, 12, 24

    def X(x):
        return L + (x - x0) / (x1 - x0) * (width - L - R)

    def Y(y):
        return T + (y1 - y) / (y1 - y0) * (height - T - B)

    s = [f'<svg viewBox="0 0 {width} {height}" width="100%">']
    for i in range(4):  # 横向网格
        yy = y0 + (y1 - y0) * i / 3
        s.append(f'<line x1="{L}" y1="{Y(yy):.1f}" x2="{width - R}" y2="{Y(yy):.1f}"'
                 f' stroke="rgba(122,134,173,.18)" stroke-width="1"/>')
        s.append(f'<text x="{L - 6}" y="{Y(yy) + 4:.1f}" font-size="10"'
                 f' fill="#7a86ad" text-anchor="end">{yy:.2f}</text>')
    for label, color, pts in series_list:
        coords = " ".join(f"{X(ep(t)):.1f},{Y(v):.1f}" for t, v in pts)
        s.append(f'<polyline points="{coords}" fill="none" stroke="{color}"'
                 f' stroke-width="2.2" stroke-linejoin="round"'
                 f' style="filter:drop-shadow(0 0 4px {color})"/>')
        lt, lv = pts[-1]
        s.append(f'<circle cx="{X(ep(lt)):.1f}" cy="{Y(lv):.1f}" r="3.5"'
                 f' fill="{color}" style="filter:drop-shadow(0 0 6px {color})"/>')
    s.append(f'<text x="{L}" y="{height - 6}" font-size="10" fill="#7a86ad">'
             f'{fmt_local(series_list[0][2][0][0])}</text>')
    s.append(f'<text x="{width - R}" y="{height - 6}" font-size="10" fill="#7a86ad"'
             f' text-anchor="end">{fmt_local(series_list[0][2][-1][0])}</text>')
    s.append("</svg>")
    return "".join(s)


def _html_rec(name, rec, lag_books):
    if rec["ev"] > 15:
        # 错盘护栏: EV 好到不真实，大概率是数据毛刺或庄家错盘——
        # 错盘注单可被"明显错误"条款单方面作废，不应当作机会推荐
        badge = ('<span class="badge b-warn">⚠ 疑似错盘或数据异常 · '
                 '先核实可成交性，勿当作机会</span>')
    elif rec["ev"] > 0:
        badge = '<span class="badge b-good">推荐 · 正期望</span>'
    elif rec["ev"] > -2.5:
        badge = '<span class="badge b-mid">可参与 · 损耗在抽水内</span>'
    else:
        badge = '<span class="badge b-warn">建议观望 · 损耗过高</span>'
    note = ""
    if rec["ev"] > 0 and rec["bookmaker"] in lag_books:
        note = ('<br/><span class="pos">↳ 成色: 滞后旧价（锚点已动该公司未跟）'
                '——真实捡漏窗口，可能随时关闭</span>')
    flow_html = ""
    if rec.get("flow_note"):
        fc = "#2bd97c" if rec.get("flow_align") else "#ffb454"
        flow_html = f'<br/><span style="color:{fc}">{rec["flow_note"]}</span>'
    if rec["win"] < rec["lose"]:
        flow_html += ('<br/><span class="mut">▲ 注意: 该方向是模型分布中的'
                      '少数派（胜率低于对侧）——推荐理由纯粹是价格最接近公平，'
                      '属“低胜率·高赔率”入口，与波胆首选可能反向，两者同源于'
                      '同一分布并不矛盾</span>')
    return (f'<div class="rec"><b>{name}推荐</b>: '
            f'<span class="pick">「{rec["label"]}」</span> @ {rec["odds"]:.2f}'
            f'（{rec["bookmaker"]}，全场最优价）{badge}<br/>'
            f'<span class="mut">模型概率: 赢 {rec["win"]:.1%} / 走 {rec["push"]:.1%}'
            f' / 输 {rec["lose"]:.1%} ｜ EV {rec["ev"]:+.1f}%</span>{flow_html}{note}</div>')


def _ev_cls(v):
    return ' class="pos"' if v > 0 else ""


def build_html(res, out_path):
    m = res["match"]
    hz, az = to_chinese(m["home_team"]), to_chinese(m["away_team"])
    ph, pd_, pa = res["fair_1x2"]
    mh, md, ma = res["model_1x2"]
    lh, la = res["lambdas"]
    lag = res["flow"]["lag_books"]
    P = []  # html parts

    P.append(f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{hz} vs {az} · 盘口解析</title>
<style>{CYBER_CSS}</style></head><body>
<div class="hero"><div class="tag">FOOTBALL ODDS INTELLIGENCE</div>
<h1>{hz}<span class="vs">VS</span>{az}</h1>
<div class="chips">
<span class="chip">赛事 <b>{m['competition'] or '—'}</b></span>
<span class="chip">开赛 <b>{fmt_local(m['kickoff_utc'])}</b>（北京时间）</span>
<span class="chip">快照 <b>{fmt_local(res['fetched_at'])}</b></span>
<span class="chip">基准 <b>{res['base_book']}</b></span>
</div></div><div class="grid">""")

    # 模型参数
    P.append(f"""<div class="card"><div class="ct"><div class="ico">■</div>
<h2>模型参数 · 由市场赔率反推，无主观输入</h2></div>
<table><tr><th></th><th>主胜</th><th>平局</th><th>客胜</th></tr>
<tr><td>市场公平概率</td><td>{ph:.1%}</td><td>{pd_:.1%}</td><td>{pa:.1%}</td></tr>
<tr><td>模型拟合概率</td><td>{mh:.1%}</td><td>{md:.1%}</td><td>{ma:.1%}</td></tr></table>
<div class="kpis">
<div class="kpi"><div class="v">{lh:.2f}</div><div class="l">λ {hz}</div></div>
<div class="kpi"><div class="v">{la:.2f}</div><div class="l">λ {az}</div></div>
<div class="kpi"><div class="v">{res['exp_total']:.2f}</div><div class="l">总进球期望</div></div>
<div class="kpi"><div class="v">{res['rho']:+.3f}</div><div class="l">低比分相关 ρ</div></div>
</div>
<div class="small">幂法去水（主锚抽水 {res['margin']:.1f}%）· 双锚加权 · 近 3 帧平滑 ·
泊松 + Dixon-Coles · 满足 1X2 / 让球 / 大小球三组零期望约束</div></div>""")

    # 走势曲线
    if res["series"]:
        sl = [(name, CHART_COLORS[i % 3], pts)
              for i, (name, pts) in enumerate(res["series"].items())]
        legend = "".join(f'<span><i style="background:{c}"></i>{n}</span>'
                         for n, c, _ in sl)
        P.append(f"""<div class="card"><div class="ct"><div class="ico">~</div>
<h2>盘口走势 · 职业资金的脚印</h2></div>
{_svg_chart(sl)}<div class="legend">{legend}</div></div>""")

    # 基本面对比（球场/天气/海拔/主场/攻防近况 + 可选人工补充）
    fund = res["fundamentals"]
    fb = []
    v, w = fund["venue"], fund["weather"]
    if v:
        info = (f"{v['stadium']} · {v['city']} ｜ 海拔 {v['alt']} 米 ｜ "
                f"{v['roof']} ｜ 容量约 {v['cap']} 千")
        fb.append(f'<div class="sub">🏟 球场</div><ul><li>{info}</li></ul>')
    if w and w.get("tmax") is not None:
        wt = (f"赛日 {w['tmin']:.0f}~{w['tmax']:.0f}°C ｜ 降水概率 "
              f"{w.get('rain', 0):.0f}% ｜ 最大风速 {w.get('wind', 0):.0f} km/h")
        fb.append(f'<div class="sub">⛅ 天气（赛日预报）</div><ul><li>{wt}</li></ul>')
    if fund["advantages"]:
        adv = "".join(f"<li>{a}</li>" for a in fund["advantages"])
        fb.append(f'<div class="sub">⚑ 场地 / 主场要素</div><ul>{adv}</ul>')
    fc = fund["form_cmp"]
    if fc:
        h, a = fc["home"], fc["away"]
        fb.append(
            '<div class="sub">📊 近况攻防对比（近 N 场场均）</div>'
            '<table><tr><th>球队</th><th>战绩</th><th>场均进球</th>'
            '<th>场均失球</th><th>积分</th></tr>'
            f'<tr><td>{h["name"]}</td><td>{h["w"]}胜{h["d"]}平{h["l"]}负</td>'
            f'<td>{h["agf"]:.1f}</td><td>{h["aga"]:.1f}</td><td>{h["points"]}</td></tr>'
            f'<tr><td>{a["name"]}</td><td>{a["w"]}胜{a["d"]}平{a["l"]}负</td>'
            f'<td>{a["agf"]:.1f}</td><td>{a["aga"]:.1f}</td><td>{a["points"]}</td></tr>'
            '</table>')
    if fund["note_md"]:
        note_html = fund["note_md"].replace("\n", "<br/>")
        fb.append(f'<div class="sub">📝 战术 / 球星补充（人工）</div>'
                  f'<div class="rec">{note_html}</div>')
    if fb:
        note_hint = ("" if fund["note_md"] else
                     '<div class="small">战术体系、球星状态等编辑性内容需人工/专家'
                     '补充（纯数据回路不臆造），可在 notes/&lt;比赛ID&gt;.md 中添加后自动并入。</div>')
        P.append(f"""<div class="card w12"><div class="ct"><div class="ico">◉</div>
<h2>基本面对比 · 助读（不消耗赔率额度，独立于模型计算）</h2></div>
{"".join(fb)}
<div class="small">本卡为事实/可计算数据（球场、赛日天气、海拔、主场、近况攻防），
独立于盘口模型、仅供理解双方；模型概率不使用本卡信息（已被市场价格消化）。</div>
{note_hint}</div>""")

    # 亚盘
    rows = "".join(
        f"<tr><td>{r['bookmaker']}</td><td>{r['line']:+.2f}</td>"
        f"<td>{r['home_odds']:.2f}</td><td>{r['away_odds']:.2f}</td>"
        f"<td>{r['fair_home']:.2f}</td><td>{r['fair_away']:.2f}</td>"
        f"<td{_ev_cls(r['ev_home'])}>{r['ev_home']:+.1f}%</td>"
        f"<td{_ev_cls(r['ev_away'])}>{r['ev_away']:+.1f}%</td></tr>"
        for r in res["ah"])
    P.append(f"""<div class="card"><div class="ct"><div class="ico">◆</div>
<h2>亚洲让球盘</h2></div>
<table><tr><th>公司</th><th>盘口</th><th>主赔</th><th>客赔</th><th>公平主</th>
<th>公平客</th><th>主EV</th><th>客EV</th></tr>{rows}</table>
{_html_rec('亚盘', res['recs']['ah'], lag) if res['recs'].get('ah') else ''}</div>""")

    # 大小球
    rows = "".join(
        f"<tr><td>{r['bookmaker']}</td><td>{r['line']:.2f}</td>"
        f"<td>{r['over_odds']:.2f}</td><td>{r['under_odds']:.2f}</td>"
        f"<td>{r['fair_over']:.2f}</td><td>{r['fair_under']:.2f}</td>"
        f"<td{_ev_cls(r['ev_over'])}>{r['ev_over']:+.1f}%</td>"
        f"<td{_ev_cls(r['ev_under'])}>{r['ev_under']:+.1f}%</td></tr>"
        for r in res["ou"])
    P.append(f"""<div class="card"><div class="ct"><div class="ico">●</div>
<h2>大小球盘</h2></div>
<table><tr><th>公司</th><th>盘口</th><th>大球</th><th>小球</th><th>公平大</th>
<th>公平小</th><th>大EV</th><th>小EV</th></tr>{rows}</table>
{_html_rec('大小球', res['recs']['ou'], lag) if res['recs'].get('ou') else ''}</div>""")

    # 波胆
    top = res["scores"]
    rows = "".join(
        f"<tr><td>{r['score']}</td><td>{r['prob']:.1f}%</td><td>{r['fair']:.1f}</td></tr>"
        for r in top)
    P.append(f"""<div class="card"><div class="ct"><div class="ico">★</div>
<h2>波胆 · 比分概率排名</h2></div>
<table><tr><th>比分(主-客)</th><th>模型概率</th><th>公平赔率（低于即亏）</th></tr>{rows}</table>
<div class="rec"><b>波胆推荐</b>: <span class="pick">「{top[0]['score']}」</span>
（{top[0]['prob']:.1f}%，市价 &gt; {top[0]['fair']:.1f} 才值得）、
次选 {top[1]['score']}（需 &gt; {top[1]['fair']:.1f}）、{top[2]['score']}
（需 &gt; {top[2]['fair']:.1f}）<br/><span class="mut">已做 Dixon-Coles 低比分修正
（ρ={res['rho']:.3f}）；波胆方差极大，仓位应远小于亚盘/大小球</span></div></div>""")

    # 模拟下注
    MKT = {"ah": "亚盘", "ou": "大小球", "cs": "波胆"}
    if res["paper_bets"]:
        rows, total_pnl, n_settled = "", 0.0, 0
        for b in res["paper_bets"]:
            settled = b["result"] is not None
            if settled:
                total_pnl += b["pnl"]
                n_settled += 1
            pnl_html = (f'<span class="{"pos" if b["pnl"] > 0 else "neg"}">'
                        f'{b["pnl"]:+.2f}</span>' if settled else "—")
            stake = b["stake"] if b["stake"] is not None else 1.0
            strat = ("顺资金" if (b["strategy"] if "strategy" in b.keys()
                     else b.get("strategy")) == "flow" else "EV最优")
            rows += (f"<tr><td>{MKT.get(b['market'])}</td><td>{b['pick']}</td>"
                     f"<td>{strat}</td>"
                     f"<td>{b['bookmaker'] or '模型公平价'}</td><td>{b['odds']:.2f}</td>"
                     f"<td>{stake:g}</td><td>{fmt_local(b['placed_at'])}</td>"
                     f"<td>{b['result'] or '未结算'}</td><td>{pnl_html}</td></tr>")
        summary = (f'<div class="rec">已结算 {n_settled} 注，合计盈亏 '
                   f'<b class="{"pos" if total_pnl >= 0 else "neg"}">{total_pnl:+.2f}</b>'
                   f'</div>' if n_settled else "")
        body = (f"<table><tr><th>玩法</th><th>选择</th><th>策略</th><th>公司</th>"
                f"<th>赔率</th><th>注额</th><th>落单时间</th><th>状态</th><th>盈亏</th></tr>"
                f"{rows}</table>{summary}")
    else:
        body = ('<div class="rec mut">尚未落单——系统将在开赛前 3 小时的首轮抓取时，'
                '按届时全场最优价自动记录三笔模拟注单（亚盘/大小球/波胆各一），'
                '赛果回填后自动结算。</div>')
    P.append(f"""<div class="card"><div class="ct"><div class="ico">※</div>
<h2>模拟下注 · 复盘数据，非真实投注</h2></div>{body}
<div class="small">策略说明: <b>EV最优</b> = 全场损耗最小入口（纯数学）；
<b>顺资金</b> = 检测到让球线移动的强信号时按职业资金方向加注（平行实验组）——
两条策略分账复盘，用数据裁决“该不该跟资金方向”。
注额规则: 亚盘/大小球每注 1，<b>波胆每注 0.1</b>——高赔率玩法
按现实投注习惯采用小仓位，与前文“波胆仓位应远小于亚盘/大小球”的提示一致。
波胆按模型公平赔率记账（市价不可采集），其复盘指标为命中率校准；
亚盘/大小球按落单时刻真实报价记账。<br/>
<b>注单与上方推荐可能方向不同</b>: 注单在开赛前 3 小时按当时最优入口锁定、永不改动
（复盘需要忠实的入场价）；推荐栏随每轮最新盘口实时重算——两者分歧说明落单后盘口
发生了移动，EV 接近时（如平手盘）排名极易翻面，不代表判断反转。</div></div>""")

    # 依据链（含第④层：资金流向与盘口行为）
    basis = res["basis"]
    flow = res["flow"]
    a = "".join(f"<li>{x}</li>" for x in basis["anchor"])
    t = "".join(f"<li>{x}</li>" for x in basis["trend"])
    f_ = "".join(f"<li>{x}</li>" for x in basis["form"])
    sharp = "".join(f"<li>{x}</li>" for x in flow["sharp"])
    behav = "".join(f"<li>{x}</li>" for x in flow["behavior"])
    cons = (f'<div class="rec" style="margin-top:14px">交叉验证: '
            f'{basis["consistency"]}</div>' if basis["consistency"] else "")
    P.append(f"""<div class="card w12"><div class="ct"><div class="ico">▲</div>
<h2>结论的依据链 · 按权重排序</h2></div>
<div class="cols3">
<div><div class="sub">① 市场锚点（主依据，模型全部输入）</div><ul>{a}</ul></div>
<div><div class="sub">② 盘口走势（解读层，不进入计算）</div><ul>{t}</ul></div>
<div><div class="sub">③ 纸面近况（辅助层，不进入计算）</div><ul>{f_}</ul></div></div>
<div style="border-top:1px dashed rgba(0,212,255,.25);margin:16px 0 12px"></div>
<div class="ct" style="margin-bottom:6px"><div class="ico">→</div>
<h2>④ 资金流向与盘口行为 · 盯盘序列分析</h2></div>
<div class="verdict">综合判定: {flow['verdict']}</div>
<div class="cols2">
<div><div class="sub">职业资金方向</div><ul>{sharp}</ul></div>
<div><div class="sub">庄家行为检测</div><ul>{behav}</ul></div></div>
{cons}<div class="small">模型概率 100% 来自第①层市场价格——公开信息已被职业资金消化
进价格，重复计入等于二次计权；②③④层用于解读与预警，不进入推荐计算。
真实投注量不可见，资金流向为价格行为的间接推断，“疑似诱导”仅为模式标记。</div></div>""")

    calib = ""
    if res.get("calibration_n"):
        calib = (f"模型参数含 {res['calibration_n']} 场历史赛果回测校准"
                 f"（更新于 {res['calibration_date']}）。")
    P.append(f"""</div><div class="footer">本报告为纯数学推导（市场赔率去水 + 泊松模型），
不构成投注建议。模型以基准公司为锚，EV 为相对锚点的偏差；伤停、阵容等场外信息未纳入模型。
{calib} 博彩有风险，本系统仅用于数据分析研究。</div></body></html>""")

    Path(out_path).write_text("".join(P), encoding="utf-8")
    return out_path


def generate_html(match_id, out_path=None):
    """生成 HTML 主报告（固定文件名覆盖更新）。"""
    res = analyze(match_id)
    out_dir = PROJECT_DIR / "reports"
    out_dir.mkdir(exist_ok=True)
    if out_path is None:
        out_path = out_dir / f"match_{match_id}_盘口解析.html"
    build_html(res, out_path)
    build_index()
    return out_path, res


def build_index():
    """总览主页 reports/index.html——系统的唯一入口。

    列出全部比赛（按开赛时间），有报告的可点击；每 5 分钟自动刷新页面。
    """
    conn = db.connect()
    out_dir = PROJECT_DIR / "reports"
    out_dir.mkdir(exist_ok=True)
    now_iso = db.utcnow_iso()

    matches = conn.execute(
        "SELECT m.*, (SELECT COUNT(DISTINCT fetched_at) FROM odds_snapshots s"
        "  WHERE s.match_id = m.id) AS n_snaps"
        " FROM matches m ORDER BY m.kickoff_utc",
    ).fetchall()
    bets = conn.execute(
        "SELECT COUNT(*) n, COALESCE(SUM(pnl), 0) pnl,"
        " SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) wins"
        " FROM paper_bets WHERE result IS NOT NULL").fetchone()
    n_open = conn.execute(
        "SELECT COUNT(*) n FROM paper_bets WHERE result IS NULL").fetchone()["n"]

    # 推荐成功率：只统计系统推荐（EV 最优策略）的已结算注单，按市场分。
    # 成功 = 盈利注（含赢半），失败 = 亏损注（含输半），走盘不计入分母。
    def market_stats(market):
        rows = conn.execute(
            "SELECT pnl, stake FROM paper_bets WHERE result IS NOT NULL"
            " AND strategy='ev' AND market=?", (market,)).fetchall()
        wins = sum(1 for r in rows if r["pnl"] > 0)
        loses = sum(1 for r in rows if r["pnl"] < 0)
        decided = wins + loses
        pnl = sum(r["pnl"] for r in rows)
        invested = sum((r["stake"] or 1) for r in rows)
        return {
            "n": len(rows), "wins": wins, "loses": loses, "decided": decided,
            "rate": (wins / decided * 100) if decided else None,
            "pnl": pnl, "roi": (pnl / invested * 100) if invested else None,
        }

    ah_stat, ou_stat = market_stats("ah"), market_stats("ou")

    upcoming, finished = [], []
    for r in matches:
        (upcoming if r["kickoff_utc"] > now_iso else finished).append(r)

    def row_html(r, show_score=False):
        rid = r["id"]
        fname = f"match_{rid}_盘口解析.html"
        # 报告可能在本轮生成目录，也可能在已发布的 docs 累积目录（云端）
        report = out_dir / fname
        if not report.exists():
            report = PROJECT_DIR.parent / "docs" / fname
        if report.exists():
            link = (f'<a href="match_{rid}_盘口解析.html" style="color:#00d4ff">'
                    f'打开报告 →</a>')
        else:
            link = f'<span class="mut">analyze.py {rid} 可生成</span>'
        if show_score:  # 已完赛行：统一显示"已完赛"，不再显示跟踪类型
            status = '<span class="badge b-mid">已完赛</span>'
        else:
            status = {"tracking": '<span class="badge b-good">重点</span>',
                      "passive": '<span class="badge b-mid">自动</span>'}.get(
                r["status"], f'<span class="badge b-warn">{r["status"]}</span>')
        score = ""
        if show_score:
            score = (f"<td><b>{r['home_score']}-{r['away_score']}</b></td>"
                     if r["home_score"] is not None else "<td>—</td>")
        return (f"<tr><td>{fmt_local(r['kickoff_utc'])}</td>"
                f"<td style='text-align:left;color:#dfe7ff'>"
                f"{to_chinese(r['home_team'])} vs {to_chinese(r['away_team'])}</td>"
                f"<td>{status}</td>{score}<td>{r['n_snaps']}</td><td>{link}</td></tr>")

    up_rows = "".join(row_html(r) for r in upcoming[:24])
    fin_rows = "".join(row_html(r, show_score=True)
                       for r in reversed(finished[-20:]))

    # API 额度（quota.json 由抓取时落盘）
    quota_html = ""
    qf = PROJECT_DIR / "quota.json"
    if qf.exists():
        try:
            q = json.loads(qf.read_text(encoding="utf-8"))
            rem = q.get("remaining", 0)
            qc = "#2bd97c" if rem > 150 else ("#ffb454" if rem > 60 else "#ff5d6c")
            quota_html = (
                f'<div class="kpi"><div class="v" style="color:{qc};'
                f'text-shadow:0 0 12px {qc}66">{rem}</div>'
                f'<div class="l">API 额度剩余 · {fmt_local(q["updated"])}</div></div>')
        except Exception:
            pass

    pnl_cls = "pos" if bets["pnl"] >= 0 else "neg"

    # 推荐战绩卡：让球盘 / 大小球 成功率（只算 EV 推荐注单）
    def stat_block(title, s):
        if not s["decided"]:
            inner = ('<div class="v mut">—</div>'
                     '<div class="l">尚无已结算推荐</div>')
        else:
            rc = "#2bd97c" if s["rate"] >= 50 else "#ff5d6c"
            roi_cls = "pos" if (s["roi"] or 0) >= 0 else "neg"
            inner = (
                f'<div class="v" style="color:{rc};text-shadow:0 0 12px {rc}66">'
                f'{s["rate"]:.0f}%</div>'
                f'<div class="l">{title}成功率 · {s["wins"]}胜{s["loses"]}负'
                f'（{s["n"]}注）｜ ROI <span class="{roi_cls}">'
                f'{s["roi"]:+.1f}%</span></div>')
        return f'<div class="kpi">{inner}</div>'

    total_decided = ah_stat["decided"] + ou_stat["decided"]
    sample_note = ("" if total_decided >= 20 else
                   '<div class="small">⚠ 样本不足（&lt;20 注），成功率波动极大仅供观察；'
                   '统计学有效结论需积累至 20+ 注后参考每日校准报告。</div>')
    perf_card = f"""<div class="card w12"><div class="ct"><div class="ico">⌖</div>
<h2>推荐战绩 · 仅统计系统推荐（EV 最优）注单</h2></div>
<div class="kpis">{stat_block("让球盘", ah_stat)}{stat_block("大小球", ou_stat)}</div>
{sample_note}
<div class="small">成功率 = 盈利注 ÷（盈利注＋亏损注），走盘不计入；含赢半/输半按盈亏方向归类。
此处仅反映“推荐方向命中率”，真实价值看 ROI——成功率高而 ROI 为负属正常（赔率＜2 时）。</div></div>"""

    html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="300">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>足球盘口情报站 · 总览</title><style>{CYBER_CSS}</style></head><body>
<div class="hero"><div class="tag">FOOTBALL ODDS INTELLIGENCE · DASHBOARD</div>
<h1>盘口情报站<span class="vs">//</span>总览</h1>
<div class="chips">
<span class="chip">更新 <b>{fmt_local(now_iso)}</b>（每 5 分钟自动刷新）</span>
<span class="chip">跟踪中 <b>{len(upcoming)}</b> 场</span>
<span class="chip">已完赛 <b>{len(finished)}</b> 场</span>
</div></div>
<div class="grid">
<div class="card w12"><div class="kpis">
<div class="kpi"><div class="v">{len(upcoming)}</div><div class="l">未开赛场次</div></div>
<div class="kpi"><div class="v">{n_open}</div><div class="l">未结算模拟注单</div></div>
<div class="kpi"><div class="v">{bets['n']}</div><div class="l">已结算注单</div></div>
<div class="kpi"><div class="v {pnl_cls}">{bets['pnl']:+.2f}</div>
<div class="l">模拟盈亏（亚盘/大小球注1，波胆注0.1）</div></div>
{quota_html}
</div></div>
{perf_card}
<div class="card w12"><div class="ct"><div class="ico">▶</div>
<h2>未开赛 · 最近 24 场（共 {len(upcoming)} 场待赛，新场次随官方赛程自动补入）</h2></div>
<table><tr><th>开赛(北京)</th><th>比赛</th><th>状态</th><th>快照</th><th>报告</th></tr>
{up_rows}</table>
<div class="small">「重点」= 手动跟踪（含球队近况数据）；「自动」= 全赛事密集跟踪。
报告在<b>开赛前 24 小时</b>起自动生成（远期每 6 小时、临场 3 小时每 30 分钟刷新），
模拟注单于开赛前 3 小时自动落单。</div></div>
<div class="card w12"><div class="ct"><div class="ico">✓</div>
<h2>已完赛 · 最近 20 场</h2></div>
<table><tr><th>开赛(北京)</th><th>比赛</th><th>状态</th><th>比分</th><th>快照</th>
<th>报告</th></tr>{fin_rows}</table></div>
</div><div class="footer">纯数据分析研究系统 · 不构成投注建议</div></body></html>"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    return out_dir / "index.html"


def build_pdf(res, out_path):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.pdfmetrics import registerFontFamily
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )

    # 嵌入系统中文字体（CID 字体不嵌入，部分查看器渲染不出中文）
    F = "CJK"
    for path, idx in [
        ("/System/Library/Fonts/Supplemental/Songti.ttc", 0),
        ("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", None),
        ("/System/Library/Fonts/STHeiti Light.ttc", 0),
    ]:
        try:
            if idx is None:
                pdfmetrics.registerFont(TTFont(F, path))
            else:
                pdfmetrics.registerFont(TTFont(F, path, subfontIndex=idx))
            break
        except Exception:
            continue
    registerFontFamily(F, normal=F, bold=F, italic=F, boldItalic=F)

    # —— 配色：深海蓝 × 奶白（模块化卡片风）——
    PRIMARY = colors.HexColor("#122E8A")    # 深海蓝（主色）
    PRIMARY_D = colors.HexColor("#0E2266")  # 更深一档（强调文字）
    BAND = colors.HexColor("#F5EFEA")       # 奶白横幅
    ZEBRA = colors.HexColor("#FAF7F3")      # 斑马纹（奶白提亮）
    GRID = colors.HexColor("#E4DCD2")       # 表格线（暖灰）
    INK = colors.HexColor("#2E2E2E")        # 正文
    MUT = colors.HexColor("#92897E")        # 弱化文字（暖灰）
    GOOD = "#3E8E5A"                        # 推荐/盈利 绿
    WARN = "#C29036"                        # 观望 琥珀

    h1 = ParagraphStyle("h1", fontName=F, fontSize=18, leading=24,
                        spaceAfter=6, textColor=colors.white)
    body = ParagraphStyle("body", fontName=F, fontSize=10, leading=15,
                          textColor=INK)
    small = ParagraphStyle("small", fontName=F, fontSize=8.5, leading=12,
                           textColor=MUT)
    sec_icon = ParagraphStyle("si", fontName=F, fontSize=11,
                              textColor=colors.white, alignment=1)
    sec_title = ParagraphStyle("st", fontName=F, fontSize=12.5, leading=16,
                               textColor=PRIMARY_D)

    def section(icon, title):
        """模块化节标题：珊瑚色图标块 + 浅粉横幅。"""
        t = Table(
            [[Paragraph(f"<b>{icon}</b>", sec_icon),
              Paragraph(f"<b>{title}</b>", sec_title)]],
            colWidths=[9 * mm, None], rowHeights=[8.5 * mm],
        )
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), PRIMARY),
            ("BACKGROUND", (1, 0), (1, 0), BAND),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (1, 0), (1, 0), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 1),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ]))
        return [Spacer(1, 5 * mm), t, Spacer(1, 3 * mm)]

    def tbl(data, widths=None, highlight_col=None):
        t = Table(data, colWidths=widths)
        style = [
            ("FONTNAME", (0, 0), (-1, -1), F),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("TEXTCOLOR", (0, 1), (-1, -1), INK),
            ("LINEBELOW", (0, 0), (-1, 0), 0.8, PRIMARY_D),
            ("GRID", (0, 0), (-1, -1), 0.4, GRID),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ZEBRA]),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
        ]
        return t, style

    m = res["match"]
    ph, pd_, pa = res["fair_1x2"]
    mh, md, ma = res["model_1x2"]
    lh, la = res["lambdas"]
    story = []

    sub_white = ParagraphStyle("sw", fontName=F, fontSize=9, leading=13,
                               textColor=colors.HexColor("#DCE3F7"))
    banner = Table([[[
        Paragraph(f"<b>{m['home_team']} vs {m['away_team']}</b> · 盘口数学解析", h1),
        Paragraph(
            f"赛事: {m['competition'] or '—'} ｜ 开赛: {fmt_local(m['kickoff_utc'])}"
            f"（北京时间）｜ 赔率快照: {fmt_local(res['fetched_at'])} ｜ "
            f"基准: {res['base_book']}", sub_white),
    ]]])
    banner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PRIMARY),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(banner)

    # 一、模型
    story.extend(section("■", "一、模型参数（由市场赔率反推，无主观输入）"))
    devig_name = "幂法" if res["devig"] == "power" else "等比例法"
    calib_note = ""
    if res.get("calibration_n"):
        calib_note = (f"模型参数含 {res['calibration_n']} 场历史赛果的回测校准"
                      f"（更新于 {res['calibration_date']}）。")
    story.append(Paragraph(
        f"以 <b>{res['base_book']}</b> 双锚加权的 1X2 赔率<b>{devig_name}去水</b>"
        f"（主锚抽水 {res['margin']:.1f}%），快照经近 3 帧平滑消除单帧噪声，"
        f"再用<b>泊松 + Dixon-Coles 低比分修正</b>模型拟合，同时满足 1X2、让球盘、"
        f"大小球三组市场零期望条件。低比分相关系数 ρ = {res['rho']:.3f}"
        f"（由市场反推）。{calib_note}", body))
    story.append(Spacer(1, 2 * mm))
    data = [
        ["", "主胜", "平局", "客负" if False else "客胜"],
        ["市场公平概率", f"{ph*100:.1f}%", f"{pd_*100:.1f}%", f"{pa*100:.1f}%"],
        ["模型拟合概率", f"{mh*100:.1f}%", f"{md*100:.1f}%", f"{ma*100:.1f}%"],
    ]
    t, st = tbl(data, [40 * mm, 30 * mm, 30 * mm, 30 * mm])
    t.setStyle(TableStyle(st))
    story.append(t)
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(
        f"期望进球: {m['home_team']} <b>{lh:.2f}</b> ─ {m['away_team']} <b>{la:.2f}</b>"
        f"（总进球期望 {res['exp_total']:.2f}）", body))

    def rec_paragraph(name, rec):
        """生成「推荐」行：最优入口 + 模型概率 + EV 分级。"""
        if rec["ev"] > 0:
            tier, color = "推荐（正期望，存在数学价值）", GOOD
        elif rec["ev"] > -2.5:
            tier, color = "可参与（损耗在正常抽水以内）", "#122E8A"
        else:
            tier, color = "建议观望（损耗高于正常抽水）", WARN
        note = ""
        if rec["ev"] > 0 and rec["bookmaker"] in res["flow"]["lag_books"]:
            note = (f"<br/><font color='{GOOD}'>↳ 成色标注: 该价为滞后旧价"
                    f"（锚点已移动而该公司未跟）——真实捡漏窗口，可能随时关闭</font>")
        return Paragraph(
            f"<b>{name}推荐</b>: <font color='{color}'><b>「{rec['label']}」</b></font>"
            f" @ {rec['odds']:.2f}（{rec['bookmaker']}，全场最优价）｜ "
            f"模型概率: 赢 {rec['win']:.1%} / 走 {rec['push']:.1%} / "
            f"输 {rec['lose']:.1%} ｜ EV {rec['ev']:+.1f}% —— "
            f"<font color='{color}'>{tier}</font>{note}", body)

    # 二、让球盘
    story.extend(section("◆", "二、亚洲让球盘解析"))
    if res["ah"]:
        data = [["公司", "盘口", "主队赔率", "客队赔率",
                 "公平主赔", "公平客赔", "主队EV", "客队EV"]]
        for r in res["ah"]:
            data.append([
                r["bookmaker"], f"{r['line']:+.2f}",
                f"{r['home_odds']:.2f}", f"{r['away_odds']:.2f}",
                f"{r['fair_home']:.2f}", f"{r['fair_away']:.2f}",
                f"{r['ev_home']:+.1f}%", f"{r['ev_away']:+.1f}%",
            ])
        t, st = tbl(data)
        # EV 列着色
        for ri, r in enumerate(res["ah"], start=1):
            if r["ev_home"] > 0:
                st.append(("TEXTCOLOR", (6, ri), (6, ri), colors.HexColor("#3E8E5A")))
            if r["ev_away"] > 0:
                st.append(("TEXTCOLOR", (7, ri), (7, ri), colors.HexColor("#3E8E5A")))
        t.setStyle(TableStyle(st))
        story.append(t)
        story.append(Spacer(1, 2 * mm))
        story.append(rec_paragraph("亚盘", res["recs"]["ah"]))
    else:
        story.append(Paragraph("暂无让球盘数据。", body))

    # 三、大小球
    story.extend(section("●", "三、大小球盘解析"))
    if res["ou"]:
        data = [["公司", "盘口", "大球赔率", "小球赔率",
                 "公平大球", "公平小球", "大球EV", "小球EV"]]
        for r in res["ou"]:
            data.append([
                r["bookmaker"], f"{r['line']:.2f}",
                f"{r['over_odds']:.2f}", f"{r['under_odds']:.2f}",
                f"{r['fair_over']:.2f}", f"{r['fair_under']:.2f}",
                f"{r['ev_over']:+.1f}%", f"{r['ev_under']:+.1f}%",
            ])
        t, st = tbl(data)
        for ri, r in enumerate(res["ou"], start=1):
            if r["ev_over"] > 0:
                st.append(("TEXTCOLOR", (6, ri), (6, ri), colors.HexColor("#3E8E5A")))
            if r["ev_under"] > 0:
                st.append(("TEXTCOLOR", (7, ri), (7, ri), colors.HexColor("#3E8E5A")))
        t.setStyle(TableStyle(st))
        story.append(t)
        story.append(Spacer(1, 2 * mm))
        story.append(rec_paragraph("大小球", res["recs"]["ou"]))
    else:
        story.append(Paragraph("暂无大小球数据。", body))

    # 四、波胆
    story.extend(section("★", "四、波胆（正确比分）概率排名"))
    data = [["比分(主-客)", "模型概率", "公平赔率（低于此值即亏）"]]
    for r in res["scores"]:
        data.append([r["score"], f"{r['prob']:.1f}%", f"{r['fair']:.1f}"])
    t, st = tbl(data, [40 * mm, 40 * mm, 60 * mm])
    t.setStyle(TableStyle(st))
    story.append(t)
    story.append(Spacer(1, 2 * mm))
    top = res["scores"]
    story.append(Paragraph(
        f"<b>波胆推荐</b>: <font color='#1a3e6e'><b>「{top[0]['score']}」</b></font>"
        f"（{top[0]['prob']:.1f}%，市价 &gt; {top[0]['fair']:.1f} 才值得）、"
        f"次选 <b>{top[1]['score']}</b>（{top[1]['prob']:.1f}%，需 &gt; {top[1]['fair']:.1f}）、"
        f"<b>{top[2]['score']}</b>（{top[2]['prob']:.1f}%，需 &gt; {top[2]['fair']:.1f}）。"
        f"前两项合计覆盖 {top[0]['prob'] + top[1]['prob']:.0f}% 概率。"
        f"低比分概率已做 Dixon-Coles 修正（ρ={res['rho']:.3f}），"
        f"但波胆方差极大，仓位应远小于亚盘/大小球。", body))

    # 五、依据链
    basis = res["basis"]
    story.extend(section("▲", "五、结论的依据链（按权重排序）"))
    story.append(Paragraph(
        "<b>① 市场锚点（主依据，模型计算的全部输入）</b>", body))
    for ln in basis["anchor"]:
        story.append(Paragraph("· " + ln, body))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(
        "<b>② 盘口走势（解读层：职业资金的脚印，不进入计算）</b>", body))
    for ln in basis["trend"]:
        story.append(Paragraph("· " + ln, body))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(
        "<b>③ 纸面近况（辅助层：人工参考，不进入计算）</b>", body))
    for ln in basis["form"]:
        story.append(Paragraph("· " + ln, body))
    if basis["consistency"]:
        story.append(Spacer(1, 2 * mm))
        story.append(Paragraph("<b>交叉验证</b>: " + basis["consistency"], body))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(
        "权重说明: 模型概率 100% 来自第①层的市场价格——纸面实力等公开信息"
        "已被全球职业资金消化进价格里，重复计入等于二次计权。②③层的价值在于"
        "解读（资金往哪边动）与预警（近况和价格背离时提示场外信息存在）。", small))

    # 六、资金流向与盘口行为
    flow = res["flow"]
    story.extend(section("→", "六、资金流向与盘口行为解读（盯盘序列分析）"))
    story.append(Paragraph(
        f"<b>综合判定: <font color='#122E8A'>{flow['verdict']}</font></b>", body))
    story.append(Spacer(1, 1.5 * mm))
    story.append(Paragraph("<b>职业资金方向</b>（让球线移动 &gt; 水位变化 &gt; "
                           "赔率漂移，按信号强度排序）:", body))
    for ln in flow["sharp"]:
        story.append(Paragraph("· " + ln, body))
    story.append(Spacer(1, 1.5 * mm))
    story.append(Paragraph("<b>庄家行为检测</b>（逐家对照锚点）:", body))
    for ln in flow["behavior"]:
        story.append(Paragraph("· " + ln, body))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(
        "声明: 本节为快照序列的规则化解读——真实投注量不可见，资金流向是"
        "从价格行为反推的间接证据；“疑似诱导”仅为模式标记，风控调价、"
        "本地客群策略均可产生相同表象。本节不进入推荐计算，定位为决策参考与预警。",
        small))

    # 七、模拟下注
    story.extend(section("※", "七、模拟下注记录（复盘数据，非真实投注）"))
    MKT = {"ah": "亚盘", "ou": "大小球", "cs": "波胆"}
    if res["paper_bets"]:
        data = [["玩法", "选择", "公司", "赔率", "注额", "落单时间", "状态", "盈亏"]]
        total_pnl, n_settled = 0.0, 0
        for b in res["paper_bets"]:
            settled = b["result"] is not None
            if settled:
                total_pnl += b["pnl"]
                n_settled += 1
            stake = b["stake"] if b["stake"] is not None else 1.0
            data.append([
                MKT.get(b["market"], b["market"]), b["pick"],
                b["bookmaker"] or "模型公平价", f"{b['odds']:.2f}", f"{stake:g}",
                fmt_local(b["placed_at"]), b["result"] or "未结算",
                f"{b['pnl']:+.2f}" if settled else "—",
            ])
        t, st = tbl(data)
        for ri, b in enumerate(res["paper_bets"], start=1):
            if b["result"] and b["pnl"] > 0:
                st.append(("TEXTCOLOR", (7, ri), (7, ri), colors.HexColor("#3E8E5A")))
            elif b["result"] and b["pnl"] < 0:
                st.append(("TEXTCOLOR", (7, ri), (7, ri), colors.HexColor("#B0413E")))
        t.setStyle(TableStyle(st))
        story.append(t)
        if n_settled:
            story.append(Spacer(1, 2 * mm))
            story.append(Paragraph(
                f"本场已结算 {n_settled} 注，合计盈亏 <b>{total_pnl:+.2f}</b>"
                f"（亚盘/大小球注 1，波胆注 0.1）。", body))
    else:
        story.append(Paragraph(
            "尚未落单。系统将在<b>开赛前 3 小时的首轮抓取</b>时，按届时全场最优价"
            "自动记录三笔模拟注单（亚盘 / 大小球 / 波胆各一，每注本金 1），"
            "赛果回填后自动结算，汇总复盘见每日校准报告。", body))
    story.append(Spacer(1, 1.5 * mm))
    story.append(Paragraph(
        "注: 波胆市价无法采集，按模型公平赔率记账——其复盘指标是命中率校准，"
        "不代表真实可得收益。亚盘/大小球按落单时刻的真实报价记账。", small))

    # 风险提示
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph(
        "说明: 本报告为纯数学推导（市场赔率去水 + 泊松模型），不构成投注建议。"
        "模型以基准公司为锚，EV 为相对该锚点的偏差；伤停、阵容轮换等信息未纳入模型。"
        "博彩有风险，本系统仅用于数据分析研究。", small))

    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm,
    )
    doc.build(story)


def generate_pdf(match_id, keep_copy=False):
    """跑完整解析并生成 PDF（每场固定一个文件，覆盖更新），返回输出路径。

    keep_copy=True 时额外保留一份带时间戳的副本。
    """
    res = analyze(match_id)
    out_dir = PROJECT_DIR / "reports"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"match_{match_id}_盘口解析.pdf"
    build_pdf(res, out_path)
    if keep_copy:
        copy = out_dir / f"match_{match_id}_盘口解析_{datetime.now(TZ):%m%d_%H%M}.pdf"
        copy.write_bytes(out_path.read_bytes())
    return out_path, res


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    match_id = int(sys.argv[1])
    out, res = generate_html(match_id)

    lh, la = res["lambdas"]
    print(f"模型: λ主={lh:.2f} λ客={la:.2f} ρ={res['rho']:.3f} "
          f"总进球期望={res['exp_total']:.2f}")
    mh, md, ma = res["model_1x2"]
    ph, pd_, pa = res["fair_1x2"]
    print(f"公平概率 主/平/客: {ph:.3f}/{pd_:.3f}/{pa:.3f}  "
          f"模型拟合: {mh:.3f}/{md:.3f}/{ma:.3f}")
    print(f"HTML 主报告已生成: {out}")
    if "--pdf" in sys.argv:  # 明确要求时才出 PDF（沿用原模板）
        pdf_out, _ = generate_pdf(match_id, keep_copy="--keep" in sys.argv)
        print(f"PDF 已生成: {pdf_out}")


if __name__ == "__main__":
    main()
