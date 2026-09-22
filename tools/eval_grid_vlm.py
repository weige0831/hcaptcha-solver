"""
本地 VLM 解题实验：让 Moondream2 直接说出「哪两格不同」，再按坐标点击提交

为什么做这个：用户要求**不依赖外部服务**的方案。前面已排除的是
"传统 CV 判据"（格状 0/93）与"Moondream2 的 point() 接口"（实例定位原语，非推理）。
但**从没测过**用结构化提问让本地 VLM 直接回答格子坐标 —— 它此前答出过连贯的
"Top center"，说明有能力，只是我没用对提问方式。

架构：VLM 推理必须在 .venv-vlm（transformers 4.52）里跑，浏览器在主环境，
所以用子进程交接：主环境存图 -> 子进程出答案 -> 主环境点提交。

用法:
    python tools/eval_grid_vlm.py 5       # 最多测 5 道格状题
产物:
    research_out/eval_grid_vlm/<轮次>/{canvas.png, answer.json}
    research_out/eval_grid_vlm/report.json
"""

from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

from hcaptcha_solver import (
    HCaptchaSolver, HTML_TEMPLATE, CHALLENGE_IFRAME_SELECTOR, REFRESH_SELECTOR,
)
from canvas_actions import CanvasGeometry
from research.grid_cells3 import crop_puzzle, find_blobs, build_cells

OUT = os.path.join(os.environ.get("RESEARCH_OUT", "research_out"), "eval_grid_vlm")
VENV_PY = os.path.join(".venv-vlm", "Scripts", "python.exe")
SITEKEY = os.environ.get("HC_SITEKEY", "338af34c-7bcb-4c7c-900b-acbec73d7d43")
URL = os.environ.get("HC_URL", "https://democaptcha.com/demo-form-eng/hcaptcha.html")

GET_CANVAS = r"""() => {
    try {
        const c = document.querySelector('canvas');
        const r = c.getBoundingClientRect();
        return {ok: true, data: c.toDataURL('image/png'),
                rect: {x: r.x, y: r.y, w: r.width, h: r.height},
                buf: {w: c.width, h: c.height}};
    } catch (e) { return {ok: false, err: String(e)}; }
}"""


def main():
    n_want = next((int(a) for a in sys.argv[1:] if a.isdigit()), 5)
    os.makedirs(OUT, exist_ok=True)
    if not os.path.exists(VENV_PY):
        print(f"❌ 找不到隔离环境 {VENV_PY}（VLM 需要 transformers 4.52）")
        return 2

    records = []
    solver = HCaptchaSolver(headless=True, widget_retries=3)
    solver.__enter__()
    try:
        page = solver.browser.new_page()
        html = HTML_TEMPLATE.replace("{{SITEKEY}}", SITEKEY)
        for u in (URL, URL + "/", URL.rstrip("/")):
            page.route(u, lambda r: r.fulfill(status=200, content_type="text/html", body=html))
        w = solver._wait_for_widget(page, URL, tries=12)
        w.evaluate("() => document.querySelector('#checkbox').click()")
        print("widget 就绪", flush=True)

        tried, rounds = 0, 0
        while tried < n_want and rounds < n_want * 40:
            rounds += 1
            ch = None
            for _ in range(14):
                time.sleep(1.4)
                cf = solver._get_challenge_frame(page)
                if cf:
                    ch = cf
                    break
            if not ch:
                if rounds % 5 == 0:
                    print(f"   [轮 {rounds}] 尚无挑战框...", flush=True)
                continue
            prompt = solver._read_prompt(ch)
            cap = ch.evaluate(GET_CANVAS)
            if not cap.get("ok"):
                continue
            arr = np.asarray(Image.open(io.BytesIO(base64.b64decode(cap["data"].split(",", 1)[1]))))
            rgb, top = crop_puzzle(arr)
            cells, _ = build_cells(rgb, find_blobs(rgb), 4, debug=False)
            if not cells or len(cells) < 16:
                if rounds % 5 == 0:
                    print(f"   [轮 {rounds}] 拿到挑战但被判为非格状（散点），刷新", flush=True)
                try:
                    ch.evaluate(f"() => {{ const r=document.querySelector('{REFRESH_SELECTOR}'); if(r) r.click(); }}")
                except Exception:
                    pass
                time.sleep(2.0)
                continue

            tried += 1
            d = os.path.join(OUT, f"trial_{tried:02d}")
            os.makedirs(d, exist_ok=True)
            canvas_path = os.path.join(d, "canvas.png")
            with open(canvas_path, "wb") as f:
                f.write(base64.b64decode(cap["data"].split(",", 1)[1]))
            ans_path = os.path.join(d, "answer.json")
            rec = {"trial": tried, "prompt": prompt}
            print(f"\n[trial {tried}] 题面: {prompt[:50]}", flush=True)

            t0 = time.time()
            try:
                subprocess.run([VENV_PY, os.path.join("research", "vlm_grid_answer.py"),
                                canvas_path, ans_path],
                               timeout=900, check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.TimeoutExpired:
                rec["error"] = "vlm_timeout"
            vlm = {}
            if os.path.exists(ans_path):
                vlm = json.loads(open(ans_path, encoding="utf-8").read())
            rec["vlm_raw"] = vlm.get("raw", "")
            rec["vlm_seconds"] = vlm.get("seconds")
            rec["vlm_cells"] = vlm.get("cells", [])
            print(f"   VLM 用时 {vlm.get('seconds')}s  回答: {vlm.get('raw','')[:90]!r}", flush=True)
            print(f"   解析出格子: {vlm.get('cells')}", flush=True)

            picks = [tuple(k) for k in vlm.get("cells", []) if tuple(k) in cells]
            if len(picks) < 2:
                rec["result"] = "vlm_no_usable_answer"
                print("   ⚠️ VLM 未给出可用的 2 个格子，本题跳过提交", flush=True)
            else:
                iframe = page.query_selector(CHALLENGE_IFRAME_SELECTOR).bounding_box()
                geom = CanvasGeometry(iframe, cap["rect"], (cap["buf"]["w"], cap["buf"]["h"]))
                for k in picks:
                    t = cells[k]
                    px, py = geom.to_page(t["cx"], t["cy"] + top)
                    page.mouse.move(px, py)
                    page.mouse.click(px, py)
                    time.sleep(0.4)
                try:
                    ch.evaluate("() => { const b=document.querySelector('.button-submit'); if(b) b.click(); }")
                except Exception:
                    pass
                time.sleep(2.5)
                err = ch.evaluate(
                    "() => { const e=document.querySelector('.display-error');"
                    " return e && e.offsetParent!==null ? e.innerText : null; }")
                rec["hcaptcha_error"] = err
                rec["picks"] = [list(k) for k in picks]
                try:
                    tok = solver._get_token(page)
                    rec["result"] = "SOLVED"
                    rec["token_len"] = len(tok)
                except Exception:
                    rec["result"] = "rejected"
                print(f"   提交回执={err!r} -> {rec['result']}", flush=True)
                if rec["result"] == "SOLVED":
                    records.append(rec)
                    break
            rec["elapsed"] = round(time.time() - t0, 1)
            records.append(rec)

            try:
                ch.evaluate(f"() => {{ const r=document.querySelector('{REFRESH_SELECTOR}'); if(r) r.click(); }}")
            except Exception:
                pass
            time.sleep(2.0)
    finally:
        try:
            solver.__exit__(None, None, None)
        except Exception:
            pass

    solved = sum(1 for r in records if r.get("result") == "SOLVED")
    sub = [r for r in records if r.get("result") in ("SOLVED", "rejected")]
    lines = [
        f"本地 VLM（Moondream2）直答格子坐标 —— 不依赖任何外部服务",
        f"格状题尝试: {len(records)}   其中真正提交: {len(sub)}   成功: {solved}",
    ]
    if sub:
        lines.append(f"提交成功率: {solved}/{len(sub)}")
    if records:
        lines.append(f"VLM 平均用时: {np.mean([r.get('vlm_seconds') or 0 for r in records]):.0f}s")
    for r in records:
        lines.append(f"  trial{r['trial']}: cells={r.get('vlm_cells')} -> {r.get('result')}"
                     f"  err={str(r.get('hcaptcha_error'))[:30]}")
    with open(os.path.join(OUT, "report.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": {"tried": len(records), "submitted": len(sub), "solved": solved},
                   "records": records}, f, ensure_ascii=False, indent=2)
    print("\n" + "\n".join(lines))
    print(f"\n报告: {OUT}/report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
