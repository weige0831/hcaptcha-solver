"""
10 次一组的评测脚本

测量两件事：
  1. 流水线可靠性 —— widget 渲染成功率、挑战获取率、限流情况
     （这是任何解题器的**上限**：流水线拿不到题，再强的求解器也没用）
  2. 端到端成功率 —— 是否最终拿到 token

用法:
    python tools/eval_harness.py            # 默认 10 次
    python tools/eval_harness.py 5          # 指定次数
    python tools/eval_harness.py 10 --attempt-canvas
        --attempt-canvas 会真的用 canvas_strategy 猜坐标并提交。
        注意：canvas_strategy 的旋转估计在真实图标上判别余量仅 ~0.001
        （见 README「旋转估计为何不可用」），坐标基本是噪声，
        提交只会消耗 hCaptcha 的容忍度，默认关闭。
产物:
    research_out/eval/report.json + summary.txt
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hcaptcha_solver import (
    HCaptchaSolver,
    HTML_TEMPLATE,
    CHALLENGE_IFRAME_SELECTOR,
    is_unsupported_challenge,
)

SITEKEY = os.environ.get("HC_SITEKEY", "a5f74b19-9e45-40e0-b45d-47ff91b7a6c2")
URL = os.environ.get("HC_URL", "https://accounts.hcaptcha.com/demo")
OUT = os.path.join(os.environ.get("RESEARCH_OUT", "research_out"), "eval")


def classify(prompt: str) -> str:
    p = (prompt or "").lower()
    if "drag" in p or "into their outlines" in p:
        return "drag"
    if "break the pattern" in p or "does not follow" in p or "follow the pattern" in p:
        return "pattern"
    if "broken spot" in p:
        return "brokenspot"
    if "click each image" in p or "select all" in p:
        return "grid"
    return "other"


def run_trial(idx: int, solver, attempt_canvas: bool) -> dict:
    """
    单次试验。注意：浏览器由调用方复用，这里只开新 page。

    Camoufox/Playwright 的 sync API 不能在一个进程里反复 __enter__/__exit__
    （第二次会抛 "Sync API inside the asyncio loop"），所以浏览器必须复用，
    每次试验只开一个新页面 —— 页面级隔离足以保证试验相互独立。
    """
    rec = {"trial": idx, "started": time.time()}
    t0 = time.time()
    page = None
    try:
        page = solver.browser.new_page()
        html = HTML_TEMPLATE.replace("{{SITEKEY}}", SITEKEY)
        for u in (URL, URL + "/", URL.rstrip("/")):
            page.route(u, lambda r: r.fulfill(status=200, content_type="text/html", body=html))

        # --- 1. widget ---
        attempts = 0
        widget = None
        for attempt in range(1, solver.widget_retries + 1):
            attempts = attempt
            page.goto(URL, wait_until="domcontentloaded", timeout=60000)
            for _ in range(16):
                time.sleep(1.5)
                cf = solver._find_widget(page)
                if cf:
                    widget = cf
                    break
            if widget:
                break
            time.sleep(3)
        rec["widget_attempts"] = attempts
        rec["widget_ready"] = bool(widget)
        if not widget:
            rec["error"] = "widget_never_rendered"
            return rec

        # --- 2. 挑战 ---
        widget.evaluate("() => document.querySelector('#checkbox').click()")
        challenge = None
        for _ in range(16):
            time.sleep(1.5)
            cf = solver._get_challenge_frame(page)
            if cf:
                challenge = cf
                break
        rec["has_challenge"] = bool(challenge)
        if not challenge:
            # 可能直接通过了
            try:
                rec["token"] = solver._get_token(page)[:24] + "..."
                rec["solved"] = True
                return rec
            except Exception:
                rec["error"] = "no_challenge"
                return rec

        prompt = solver._read_prompt(challenge)
        rec["prompt"] = prompt
        rec["type"] = classify(prompt)

        # --- 3. 求解 ---
        if rec["type"] == "grid":
            # 照片网格路径（当下遇不到）
            rec["solve_attempted"] = "grid"
            rec["note"] = "grid 路径未经验证"
        elif attempt_canvas:
            rec["solve_attempted"] = "canvas"
            try:
                from canvas_strategy import analyze
                import base64
                import numpy as np
                from PIL import Image
                import io as _io

                cap = challenge.evaluate("""() => {
                    try {
                        const c = document.querySelector('canvas');
                        const r = c.getBoundingClientRect();
                        return {ok: true, data: c.toDataURL('image/png'),
                                rect: {x: r.x, y: r.y, w: r.width, h: r.height},
                                buf: {w: c.width, h: c.height}};
                    } catch (e) { return {ok: false, err: String(e)}; }
                }""")
                if not cap.get("ok"):
                    rec["error"] = "canvas_capture_failed"
                    return rec
                raw = base64.b64decode(cap["data"].split(",", 1)[1])
                arr = np.asarray(Image.open(_io.BytesIO(raw)))
                rgb = arr[:, :, :3].copy()
                alpha = arr[:, :, 3] if arr.shape[2] == 4 else None
                res = analyze(rgb, alpha)
                rec["icons_detected"] = len(res["icons"])
                rec["outliers"] = res["outliers"]
                rec["angle_scores"] = [round(i["angle_score"], 3) for i in res["icons"]]

                iframe_box = page.query_selector(CHALLENGE_IFRAME_SELECTOR).bounding_box()
                scale = cap["buf"]["w"] / cap["rect"]["w"]
                for (bx, by) in res["click_points"]:
                    ax = iframe_box["x"] + cap["rect"]["x"] + bx / scale
                    ay = iframe_box["y"] + cap["rect"]["y"] + by / scale
                    page.mouse.click(ax, ay)
                    time.sleep(0.4)
                challenge.evaluate(
                    "() => { const b=document.querySelector('.button-submit'); if(b) b.click(); }")
                time.sleep(2)
                err = challenge.evaluate(
                    "() => { const e=document.querySelector('.display-error');"
                    " return e && e.offsetParent!==null ? e.innerText : null; }")
                rec["hcaptcha_error"] = err
            except Exception as e:
                rec["canvas_error"] = f"{type(e).__name__}: {e}"
        else:
            rec["solve_attempted"] = None
            rec["note"] = "未尝试（canvas 求解未实现）"

        # --- 4. 结果 ---
        try:
            rec["token"] = solver._get_token(page)[:24] + "..."
            rec["solved"] = True
        except Exception:
            rec["solved"] = False

    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["trace"] = traceback.format_exc()[-400:]
    finally:
        try:
            if page:
                page.close()
        except Exception:
            pass
        rec["seconds"] = round(time.time() - t0, 1)
    return rec


def main():
    n = 10
    attempt_canvas = "--attempt-canvas" in sys.argv
    for a in sys.argv[1:]:
        if a.isdigit():
            n = int(a)
    os.makedirs(OUT, exist_ok=True)

    records = []
    with HCaptchaSolver(headless=True, widget_retries=4) as solver:
        for i in range(1, n + 1):
            print(f"\n===== 第 {i}/{n} 次 =====", flush=True)
            rec = run_trial(i, solver, attempt_canvas)
            records.append(rec)
            print(f"  widget就绪={rec.get('widget_ready')} (尝试{rec.get('widget_attempts')}次) "
                  f"拿到题={rec.get('has_challenge')} 类型={rec.get('type')} "
                  f"解题={rec.get('solved')} 耗时={rec.get('seconds')}s "
                  f"{'错误=' + str(rec.get('error')) if rec.get('error') else ''}", flush=True)
            time.sleep(5)

    total = len(records)
    widget_ok = sum(1 for r in records if r.get("widget_ready"))
    chal_ok = sum(1 for r in records if r.get("has_challenge"))
    solved = sum(1 for r in records if r.get("solved"))
    types = {}
    for r in records:
        types[r.get("type", "n/a")] = types.get(r.get("type", "n/a"), 0) + 1

    summary = {
        "trials": total,
        "widget_ready": widget_ok,
        "widget_rate": round(widget_ok / total, 3),
        "challenge_obtained": chal_ok,
        "challenge_rate": round(chal_ok / total, 3),
        "solved": solved,
        "solve_rate": round(solved / total, 3),
        "end_to_end_rate": round(solved / total, 3),
        "challenge_types": types,
        "avg_seconds": round(sum(r.get("seconds", 0) for r in records) / total, 1),
        "attempt_canvas": attempt_canvas,
    }

    with open(os.path.join(OUT, "report.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f, ensure_ascii=False, indent=2)

    lines = [
        f"试验次数           : {total}",
        f"widget 就绪        : {widget_ok}/{total}  ({summary['widget_rate']:.0%})",
        f"拿到挑战           : {chal_ok}/{total}  ({summary['challenge_rate']:.0%})",
        f"成功拿到 token     : {solved}/{total}  ({summary['solve_rate']:.0%})",
        f"端到端成功率       : {summary['end_to_end_rate']:.0%}",
        f"题型分布           : {types}",
        f"平均单次耗时       : {summary['avg_seconds']}s",
        f"是否尝试 canvas    : {attempt_canvas}",
        "",
        "注: 流水线可靠性（widget / 拿题）是任何求解器的成功率上限。",
    ]
    with open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\n报告: {OUT}/report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
