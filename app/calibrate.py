#!/usr/bin/env python3
"""校准验证：用已完赛比赛（收盘赔率 + 实际赛果）给模型对账。

用法:
    python calibrate.py            # 输出校准报告并更新 calibration.json

对账内容:
  1. 去水方法对比（幂法 vs 等比例法）：哪种的收盘概率更接近实际结果
  2. 锚点对比（Pinnacle 单锚 vs Pinnacle+Betfair 双锚）
  3. 平局/大小球/大比分的系统性偏差检查

教训回流规则（写入 calibration.json，analyze.py 自动读取）:
  - 样本 >= 20 场且差异显著，才切换去水方法或锚点权重
  - 不足门槛的发现只记录在 notes 里，不影响模型 ——
    单场比赛说明不了任何问题，这是设计原则不是缺陷
"""
import json
import math
import sys
from datetime import datetime, timezone

import db
from analyze import (
    ANCHOR_BOOKS, analyze, demargin, diff_dist, ev_total, fit_model,
    latest_odds, score_matrix, smooth_quote, total_dist,
)
from config import PROJECT_DIR

MIN_N_SWITCH = 20    # 切换模型参数的最小样本量
MIN_N_TAIL = 30      # 评论尾部偏差的最小样本量
SIG_LOGLOSS = 0.002  # 对数损失差异的显著性阈值


def outcome_index(hs, as_):
    return 0 if hs > as_ else (1 if hs == as_ else 2)


def logloss(prob_outcome):
    return -math.log(max(prob_outcome, 1e-9))


def main():
    conn = db.connect()
    matches = db.finished_with_results(conn)
    if not matches:
        print("还没有'已完赛 + 有比分 + 有赔率快照'的比赛，等开赛后自动积累。")
        _write_config(0, "power", {}, ["尚无可对账的比赛"])
        return

    per_method = {"power": [], "proportional": []}   # 每场的 logloss
    blend_ll, pinn_ll = [], []                        # 双锚 vs 单锚
    draw_pred, draw_real = [], []
    tail_pred, tail_real = [], []
    score_hits, n_score = 0, 0

    for m in matches:
        hs, as_ = m["home_score"], m["away_score"]
        idx = outcome_index(hs, as_)
        _, history = latest_odds(conn, m["id"])

        # 收盘 1X2（平滑后），按锚点逐家取
        closing = {}
        for bk in ANCHOR_BOOKS:
            rows = history.get(("1x2", bk))
            if not rows:
                continue
            sm = smooth_quote(rows)
            q = [sm["home_odds"], sm["draw_odds"], sm["away_odds"]]
            if all(q):
                closing[bk] = q
        if not closing:
            continue
        primary = closing.get("pinnacle") or list(closing.values())[0]

        # 1) 去水方法对比（用主锚收盘价）
        for method in per_method:
            probs = demargin(primary, method)
            per_method[method].append(logloss(probs[idx]))

        # 2) 单锚 vs 双锚（都用幂法）
        p_pinn = demargin(primary, "power")
        pinn_ll.append(logloss(p_pinn[idx]))
        if len(closing) >= 2:
            ps = [demargin(q, "power") for q in closing.values()]
            blend = [sum(p[i] for p in ps) / len(ps) for i in range(3)]
            blend_ll.append(logloss(blend[idx]))
        else:
            blend_ll.append(pinn_ll[-1])

        # 3) 模型级检查：平局概率 / 尾部（净胜>=3）/ 波胆 top3
        try:
            res = analyze(m["id"])
        except SystemExit:
            continue
        mh, md, ma = res["model_1x2"]
        draw_pred.append(md)
        draw_real.append(1 if idx == 1 else 0)
        lh, la = res["lambdas"]
        dd = diff_dist(score_matrix(lh, la, res["rho"]))
        tail_pred.append(sum(p for d, p in dd.items() if abs(d) >= 3))
        tail_real.append(1 if abs(hs - as_) >= 3 else 0)
        top3 = {r["score"] for r in res["scores"][:3]}
        n_score += 1
        if f"{hs}-{as_}" in top3:
            score_hits += 1

    n = len(per_method["power"])
    print(f"=== 校准报告（{n} 场有效样本）===\n")

    notes = []
    devig_method = "power"
    anchor_weights = {}

    if n:
        ll_pow = sum(per_method["power"]) / n
        ll_pro = sum(per_method["proportional"]) / n
        print(f"去水方法  幂法 logloss={ll_pow:.4f} | 等比例 logloss={ll_pro:.4f}"
              f"（越低越好）")
        if n >= MIN_N_SWITCH and ll_pro < ll_pow - SIG_LOGLOSS:
            devig_method = "proportional"
            notes.append(f"[已生效] {n} 场回测显示等比例去水更准，已切换")
        elif n < MIN_N_SWITCH:
            notes.append(f"[样本不足 {n}/{MIN_N_SWITCH}] 去水方法对比仅记录，维持幂法")

        ll_blend = sum(blend_ll) / len(blend_ll)
        ll_pinn = sum(pinn_ll) / len(pinn_ll)
        print(f"锚点      双锚 logloss={ll_blend:.4f} | Pinnacle 单锚={ll_pinn:.4f}")
        if n >= MIN_N_SWITCH and ll_pinn < ll_blend - SIG_LOGLOSS:
            anchor_weights = {"pinnacle": 1.0, "betfair_ex_eu": 0.0}
            notes.append(f"[已生效] {n} 场回测显示 Pinnacle 单锚更准，已调整权重")
        elif n < MIN_N_SWITCH:
            notes.append(f"[样本不足 {n}/{MIN_N_SWITCH}] 锚点对比仅记录，维持双锚等权")

    if draw_pred:
        dp, dr = sum(draw_pred) / len(draw_pred), sum(draw_real) / len(draw_real)
        print(f"平局      模型平均预测 {dp:.1%} | 实际发生率 {dr:.1%}")
        if len(draw_pred) >= MIN_N_SWITCH and abs(dp - dr) > 0.05:
            notes.append(f"[观察] 平局概率偏差 {dp - dr:+.1%}（{len(draw_pred)} 场），"
                         f"持续存在可考虑调整 ρ 搜索范围")
    if tail_pred:
        tp, tr = sum(tail_pred) / len(tail_pred), sum(tail_real) / len(tail_real)
        print(f"大胜(净3+) 模型平均预测 {tp:.1%} | 实际发生率 {tr:.1%}")
        if len(tail_pred) >= MIN_N_TAIL and tr > tp * 1.3:
            notes.append(f"[观察] 大比分被低估（预测 {tp:.1%} vs 实际 {tr:.1%}），"
                         f"建议启用全梯子深抓校准尾部")
    if n_score:
        print(f"波胆      top3 命中率 {score_hits}/{n_score} = {score_hits/n_score:.1%}"
              f"（随机基线约 25~30%）")

    # 模拟下注复盘
    bets = conn.execute(
        "SELECT market, strategy, result, pnl, stake FROM paper_bets"
        " WHERE result IS NOT NULL"
    ).fetchall()
    if bets:
        print("\n=== 模拟下注复盘（按策略分账；亚盘/大小球注 1，波胆注 0.1）===")
        by_mkt = {}
        for b in bets:
            strat = b["strategy"] or "ev"
            key = b["market"] if strat == "ev" else f"{b['market']}·顺资金"
            by_mkt.setdefault(key, []).append(b)
        names = {"ah": "亚盘", "ou": "大小球", "cs": "波胆",
                 "ah·顺资金": "亚盘·顺资金"}
        for mk, rows_ in sorted(by_mkt.items()):
            n_ = len(rows_)
            wins = sum(1 for b in rows_ if b["pnl"] > 0)
            pnl = sum(b["pnl"] for b in rows_)
            total_stake = sum(b["stake"] if b["stake"] is not None else 1.0
                              for b in rows_)
            roi = pnl / total_stake * 100
            tag = "（公平赔率记账，看命中率校准）" if mk == "cs" else ""
            print(f"{names.get(mk, mk):<4} {n_} 注 | 赢 {wins} ({wins/n_:.0%}) | "
                  f"累计盈亏 {pnl:+.2f} | ROI {roi:+.1f}%{tag}")
            if mk != "cs" and n_ >= MIN_N_SWITCH:
                notes.append(f"[复盘] {names[mk]} {n_} 注 ROI {roi:+.1f}%"
                             f"——{'策略呈正期望，值得深入' if roi > 0 else 'ROI 为负，与负 EV 入场一致，勿当作可盈利策略'}")

    _write_config(n, devig_method, anchor_weights, notes)
    print(f"\n校准配置已写入 calibration.json（{len(notes)} 条记录）")
    for note in notes:
        print(f"  - {note}")


def _write_config(n, devig_method, anchor_weights, notes):
    cfg = {
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "n_matches": n,
        "devig_method": devig_method,
        "anchor_weights": anchor_weights,
        "notes": notes,
    }
    (PROJECT_DIR / "calibration.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
