"""
格状题型 live 测试：提取格子 -> 选异常 -> 点击 -> 提交 -> 读 hCaptcha 回执

这是本次会话里第一次有了**可用的坐标生成**（针对格状子类）：
`research/grid_cells3.py` 用「角色块质心反推格距」定出 4x4 格框，
拼图核验（research_out/grid3/cells.png）确认每格取到一个完整角色。

于是可以真正闭环了：把候选点按坐标点出去，让 hCaptcha 判对错。
判据是服务端的通过/不通过 —— 这是唯一的真值来源。

同时做一件之前做不到的事：**对比不同选点规则的表现**，因为单靠方差分析
未必能定出"规律"到底是列重复还是别的。这里支持两种规则：
    col   —— 与本列其它格最不像的那两个（列重复假设）
    global—— 与全场所有格最不像的那两个（整体少数派假设）
每道题只提交一次（提交会消耗题目），所以规则在题与题之间轮换。

用法:
    python tools/eval_grid.py 10            # 最多测 10 道格状题
产物:
    research_out/eval_grid/report.json / summary.txt
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

from hcaptcha_solver import (
    HCaptchaSolver, HTML_TEMPLATE, CHALLENGE_IFRAME_SELECTOR, REFRESH_SELECTOR,
)
from canvas_actions import CanvasGeometry
from research.grid_cells3 import find_blobs, build_cells, cell_patch, similarity, crop_puzzle

OUT = os.path.join(os.environ.get("RESEARCH_OUT", "research_out"), "eval_grid")
SITEKEY = "a5f74b19-9e45-40e0-b45d-47ff91b7a6c2"
URL = "https://accounts.hcaptcha.com/demo"

GET_CANVAS = r"""() => {
    try {
        const c = document.querySelector('canvas');
        const r = c.getBoundingClientRect();
        return {ok: true, data: c.toDataURL('image/png'),
                rect: {x: r.x, y: r.y, w: r.width, h: r.height},
                buf: {w: c.width, h: c.height}};
    } catch (e) { return {ok: false, err: String(e)}; }
}"""


def pick_candidates(rgb_full, rule):
    """
    返回 (候选[(x,y),...], 诊断信息)；坐标为裁剪后拼图区坐标

    rule:
      col/global —— 前两轮的规则（保留以便对比）
      axis       —— 本轮新增：先判断规律在哪个方向自洽，再在该方向上找偏离

    为什么要 axis：上一轮 0/10 的原因定位到"候选分几乎并列"。
    回看唯一能肉眼核验的那道题（pat_01），发现按列看答案是清楚的
    （列0 全绿 / 列1 全粉 / 列2 三兔一青 / 列3 三蓝一红 -> 异常就是那只青的和红的），
    而我的分数把列与行平均了 —— **把一个自洽方向和一个不自洽方向混在一起**，
    信号被稀释。所以改成：先估哪个方向自洽，只在该方向上算偏离。
    """
    rgb, top = crop_puzzle(rgb_full)
    blobs = find_blobs(rgb)
    cells, geom = build_cells(rgb, blobs, 4, debug=False)
    if not cells or len(cells) < 16:
        return None, {"error": f"格提取不足 ({len(cells) if cells else 0})"}
    patches = {k: cell_patch(rgb, t) for k, t in cells.items()}

    # 16x16 相似度矩阵（对称）
    keys = sorted(patches)
    S = {k: {} for k in keys}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            v = similarity(patches[a], patches[b])
            S[a][b] = v
            S[b][a] = v

    def line_stats(axis):
        """
        对每条线(行或列)：
          consens = 与线内其它格平均相似度最高的那个（代表该线的"主流角色"）
          within  = 线内所有两两相似度的均值（衡量这条线有多自洽）
          dev[k]  = 1 - 相似度(k, consens)
        返回 (整体自洽度, {格: 偏离度})
        """
        within_all, dev = [], {}
        for idx in range(4):
            line = [k for k in keys if (k[1] if axis == "col" else k[0]) == idx]
            if len(line) < 2:
                continue
            best_c, best_v = None, -1
            for k in line:
                others = [S[k][o] for o in line if o != k]
                m = float(np.mean(others))
                if m > best_v:
                    best_v, best_c = m, k
            within_all.append(best_v)
            for k in line:
                if k != best_c:
                    dev[k] = 1.0 - S[k][best_c]
        return float(np.mean(within_all)) if within_all else 0.0, dev

    col_cons, col_dev = line_stats("col")
    row_cons, row_dev = line_stats("row")
    axis = "col" if col_cons >= row_cons else "row"
    dev = col_dev if axis == "col" else row_dev

    if rule in ("axis", "second"):
        ranked = sorted(dev.items(), key=lambda kv: -kv[1])
        if rule == "second":
            # 「次强偏离」策略：检验"最极端的离群不是答案、中间档才是"这一假设。
            # 对每条线取**第二强**偏离的格子，再在全部这些格子里取前 2。
            per_line = {}
            for k, v in dev.items():
                per_line.setdefault(k[0] if axis == "row" else k[1], []).append((k, v))
            cand = []
            for _, lst in per_line.items():
                lst.sort(key=lambda kv: -kv[1])
                if len(lst) >= 2:
                    cand.append(lst[1])
            cand.sort(key=lambda kv: -kv[1])
            ranked = cand + ranked
        picks = [k for k, _ in ranked[:2]]
        diag = {"rule": rule, "picks": [list(k) for k in picks],
                "col_consistency": round(col_cons, 4),
                "row_consistency": round(row_cons, 4),
                "chosen_axis": axis,
                "top_scores": [round(v, 4) for _, v in ranked[:4]],
                "margin_2_3": round(ranked[2][1] - ranked[1][1], 4) if len(ranked) > 2 else None,
                "n_cells": len(cells), "top": int(top)}
        pts = [(int(cells[k]["cx"]), int(cells[k]["cy"])) for k in picks]
        return pts, diag

    # --- 旧规则（对比用）---
    def score(axis_):
        out = {}
        for (r, c), p in patches.items():
            others = [patches[q] for q in patches
                      if (q[1] == c if axis_ == "col" else q[0] == r) and q != (r, c)]
            if others:
                out[(r, c)] = float(np.mean([similarity(p, o) for o in others]))
        return out

    sc, sr = score("col"), score("row")
    if rule == "col":
        ranked = sorted({k: (sc[k] + sr.get(k, sc[k])) / 2 for k in sc}.items(),
                        key=lambda kv: kv[1])
    else:
        g = {}
        for k, p in patches.items():
            others = [v for q, v in patches.items() if q != k]
            g[k] = float(np.mean([similarity(p, o) for o in others]))
        ranked = sorted(g.items(), key=lambda kv: kv[1])

    picks = [k for k, _ in ranked[:2]]
    pts = [(int(cells[k]["cx"]), int(cells[k]["cy"])) for k in picks]
    diag = {"rule": rule, "picks": [list(k) for k in picks],
            "top_scores": [round(v, 3) for _, v in ranked[:4]],
            "n_cells": len(cells), "top": int(top)}
    return pts, diag


def main():
    n_want = next((int(a) for a in sys.argv[1:] if a.isdigit()), 10)
    os.makedirs(OUT, exist_ok=True)
    records = []
    dist = {}

    solver = HCaptchaSolver(headless=True, widget_retries=3)
    solver.__enter__()
    try:
        page = solver.browser.new_page()
        html = HTML_TEMPLATE.replace("{{SITEKEY}}", SITEKEY)
        for u in (URL, URL + "/", URL.rstrip("/")):
            page.route(u, lambda r: r.fulfill(status=200, content_type="text/html", body=html))
        widget = solver._wait_for_widget(page, URL, tries=12)
        widget.evaluate("() => document.querySelector('#checkbox').click()")
        print("widget 就绪", flush=True)

        tried, rounds = 0, 0
        while tried < n_want and rounds < n_want * 10:
            rounds += 1
            challenge = None
            for _ in range(14):
                time.sleep(1.4)
                cf = solver._get_challenge_frame(page)
                if cf:
                    challenge = cf
                    break
            if not challenge:
                continue
            prompt = solver._read_prompt(challenge)
            low = prompt.lower()
            is_pat = ("pattern" in low or "does not follow" in low)
            kind = "pattern" if is_pat else ("drag" if "drag" in low else "other")
            dist[kind] = dist.get(kind, 0) + 1

            if is_pat:
                cap = challenge.evaluate(GET_CANVAS)
                if not cap.get("ok"):
                    continue
                raw = base64.b64decode(cap["data"].split(",", 1)[1])
                arr = np.asarray(Image.open(io.BytesIO(raw)))
                rule = os.environ.get("SOLVER_SELECT", "axis")   # axis=最极端离群；second=次强偏离
                pts, diag = pick_candidates(arr, rule)
                if not pts:
                    print(f"[{rounds:3d}] 跳过: {diag.get('error')}", flush=True)
                else:
                    tried += 1
                    print(f"[{rounds:3d}] 第{tried}题 rule={rule} 格数={diag['n_cells']} "
                          f"top分={diag['top_scores']}", flush=True)
                    iframe_box = page.query_selector(CHALLENGE_IFRAME_SELECTOR).bounding_box()
                    geom = CanvasGeometry(iframe_box, cap["rect"], (cap["buf"]["w"], cap["buf"]["h"]))
                    top = diag.pop("top", 0)
                    rec = {"trial": tried, **diag, "prompt": prompt}

                    # 保存本次挑战图与候选位置，供事后离线复查。
                    # 之前没存 → live 失败无法分析，这正是"离线估的命中率解释不了 0/13"
                    # 时无法判断是估计偏差还是 live 另有失败的原因。补上这个仪器缺口。
                    try:
                        tdir = os.path.join(OUT, f"trial_{tried:02d}")
                        os.makedirs(tdir, exist_ok=True)
                        with open(os.path.join(tdir, "canvas.png"), "wb") as f:
                            f.write(raw)
                        import cv2 as _cv2
                        vis = arr[top:, :, :3].copy()
                        picked = []
                        for (bx, by) in pts:
                            picked.append([int(bx), int(by)])
                            _cv2.drawMarker(vis, (int(bx), int(by)),
                                            (255, 0, 255), _cv2.MARKER_CROSS, 30, 4)
                        Image.fromarray(vis).save(os.path.join(tdir, "picks.png"))
                        rec["saved_dir"] = os.path.basename(tdir)
                        rec["picked_positions"] = picked
                    except Exception as _e:
                        rec["save_error"] = f"{type(_e).__name__}: {_e}"

                    for (bx, by) in pts:
                        px, py = geom.to_page(bx, by + top)
                        page.mouse.move(px, py)
                        page.mouse.click(px, py)
                        time.sleep(0.35)
                    challenge.evaluate(
                        "() => { const b=document.querySelector('.button-submit'); if(b) b.click(); }")
                    time.sleep(2.5)
                    err = challenge.evaluate(
                        "() => { const e=document.querySelector('.display-error');"
                        " return e && e.offsetParent!==null ? e.innerText : null; }")
                    rec["hcaptcha_error"] = err
                    try:
                        tok = solver._get_token(page)
                        rec["solved"] = True
                        rec["token_len"] = len(tok)
                    except Exception:
                        rec["solved"] = False
                    records.append(rec)
                    print(f"        提交回执={err!r}  解题={rec['solved']}", flush=True)
                    if rec["solved"]:
                        break

            try:
                challenge.evaluate(
                    f"() => {{ const r=document.querySelector('{REFRESH_SELECTOR}'); if(r) r.click(); }}")
            except Exception:
                pass
            time.sleep(2.0)
    finally:
        try:
            solver.__exit__(None, None, None)
        except Exception:
            pass

    ok = sum(1 for r in records if r.get("solved"))
    summary = {"attempted": len(records), "solved": ok,
               "success_rate": round(ok / len(records), 3) if records else 0.0,
               "challenge_types": dist,
               "by_rule": {}}
    for rule in ("col", "global"):
        sub = [r for r in records if r.get("rule") == rule]
        if sub:
            summary["by_rule"][rule] = {
                "n": len(sub), "solved": sum(1 for r in sub if r.get("solved"))}

    lines = [
        f"题型分布           : {dist}",
        f"格状题尝试次数     : {len(records)}",
        f"成功拿到 token     : {ok}/{len(records)}"
        + (f"  ({summary['success_rate']:.0%})" if records else ""),
    ]
    for rule, v in summary["by_rule"].items():
        lines.append(f"  规则 {rule:7s}: {v['solved']}/{v['n']}")
    if records:
        lines.append("")
        lines.append("各次回执:")
        for r in records:
            lines.append(f"  第{r['trial']}题 rule={r.get('rule')} "
                         f"picks={r.get('picks')} 回执={r.get('hcaptcha_error')!r}")

    with open(os.path.join(OUT, "report.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\n报告: {OUT}/report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
