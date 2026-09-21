"""
针对拖拽题型的定点验证

为什么需要单独测：hCaptcha 每次下发的题型是随机的（实测有的批次 10/10 全是
pattern，有的批次 4/10 是 drag）。跑评测碰运气可能整批都遇不到拖拽题，
那样「拖拽已实现」就只是没验证过的说法。这个脚本反复刷新直到拿到拖拽题，
再执行一次拖拽并测量 canvas 像素变化。

判据：拖拽后 canvas 像素必须发生变化。若像素毫无变化，说明拖拽事件序列
（pointer down -> move -> up）没有被 canvas 接收，实现需要调整。

用法:
    python tools/verify_drag.py
产物:
    research_out/drag/before.png / after.png / log.json
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

from hcaptcha_solver import (
    HCaptchaSolver,
    HTML_TEMPLATE,
    CHALLENGE_IFRAME_SELECTOR,
    REFRESH_SELECTOR,
    is_unsupported_challenge,
)
from canvas_actions import CanvasGeometry, drag_and_drop

SITEKEY = os.environ.get("HC_SITEKEY", "a5f74b19-9e45-40e0-b45d-47ff91b7a6c2")
URL = os.environ.get("HC_URL", "https://accounts.hcaptcha.com/demo")
OUT = os.path.join(os.environ.get("RESEARCH_OUT", "research_out"), "drag")

GET_CANVAS = r"""() => {
    try {
        const c = document.querySelector('canvas');
        const r = c.getBoundingClientRect();
        return {ok: true, data: c.toDataURL('image/png'),
                rect: {x: r.x, y: r.y, w: r.width, h: r.height},
                buf: {w: c.width, h: c.height}};
    } catch (e) { return {ok: false, err: String(e)}; }
}"""


def save_png(data_url: str, path: str) -> None:
    with open(path, "wb") as f:
        f.write(base64.b64decode(data_url.split(",", 1)[1]))


def diff_ratio(a_path: str, b_path: str) -> tuple[float, int]:
    a = np.asarray(Image.open(a_path).convert("RGB"), dtype=np.int16)
    b = np.asarray(Image.open(b_path).convert("RGB"), dtype=np.int16)
    if a.shape != b.shape:
        return 1.0, -1
    d = np.abs(a - b).sum(axis=2)
    changed = int((d > 30).sum())
    return changed / d.size, changed


def main():
    os.makedirs(OUT, exist_ok=True)
    log = {"attempts": []}

    with HCaptchaSolver(headless=True, widget_retries=5) as solver:
        page = solver.browser.new_page()
        html = HTML_TEMPLATE.replace("{{SITEKEY}}", SITEKEY)
        for u in (URL, URL + "/", URL.rstrip("/")):
            page.route(u, lambda r: r.fulfill(status=200, content_type="text/html", body=html))

        widget = solver._wait_for_widget(page, URL)
        widget.evaluate("() => document.querySelector('#checkbox').click()")

        challenge, prompt = None, ""
        for round_no in range(18):
            challenge = None
            for _ in range(16):
                time.sleep(1.5)
                cf = solver._get_challenge_frame(page)
                if cf:
                    challenge = cf
                    break
            if not challenge:
                print(f"[{round_no+1}] 挑战未出现，重试")
                continue
            prompt = solver._read_prompt(challenge)
            low = prompt.lower()
            print(f"[{round_no+1}] 题面: {prompt!r}")
            if "drag" in low or "into their outlines" in low:
                print("    -> 拿到拖拽题")
                break
            try:
                challenge.evaluate(
                    f"() => {{ const r=document.querySelector('{REFRESH_SELECTOR}'); if(r) r.click(); }}")
            except Exception:
                pass
            time.sleep(3.0)
        else:
            print("❌ 18 次刷新都没遇到拖拽题（题型由服务端随机决定）")
            log["got_drag_challenge"] = False
            with open(os.path.join(OUT, "log.json"), "w", encoding="utf-8") as f:
                json.dump(log, f, ensure_ascii=False, indent=2)
            return 1

        log["got_drag_challenge"] = True
        log["prompt"] = prompt

        cap = challenge.evaluate(GET_CANVAS)
        if not cap.get("ok"):
            print("canvas 抓取失败:", cap.get("err"))
            return 1
        before = os.path.join(OUT, "before.png")
        save_png(cap["data"], before)
        print(f"   canvas CSS {cap['rect']['w']}x{cap['rect']['h']}  "
              f"缓冲 {cap['buf']['w']}x{cap['buf']['h']}")

        # 做一次大跨度拖拽。注意有些拖拽题需要先选中源物体（截图里见过 Move 手柄），
        # 所以这里先点一下源位置再拖，两条路径都覆盖到。
        iframe_box = page.query_selector(CHALLENGE_IFRAME_SELECTOR).bounding_box()
        geom = CanvasGeometry(iframe_box, cap["rect"],
                              (cap["buf"]["w"], cap["buf"]["h"]))

        bw, bh = cap["buf"]["w"], cap["buf"]["h"]
        x1, y1 = bw * 0.30, bh * 0.55
        x2, y2 = bw * 0.72, bh * 0.78
        p1 = geom.to_page(x1, y1)
        p2 = geom.to_page(x2, y2)
        print(f"   先点击源位置 页面({p1[0]:.0f},{p1[1]:.0f})")
        page.mouse.move(p1[0], p1[1])
        page.mouse.click(p1[0], p1[1])
        time.sleep(0.6)
        print(f"   拖拽 缓冲({x1:.0f},{y1:.0f}) -> ({x2:.0f},{y2:.0f})   "
              f"页面({p1[0]:.0f},{p1[1]:.0f}) -> ({p2[0]:.0f},{p2[1]:.0f})")
        drag_and_drop(page, p1[0], p1[1], p2[0], p2[1])

        cap2 = challenge.evaluate(GET_CANVAS)
        if not cap2.get("ok"):
            print("二次抓取失败")
            return 1
        after = os.path.join(OUT, "after.png")
        save_png(cap2["data"], after)

        ratio, px = diff_ratio(before, after)
        log["diff_ratio"] = round(ratio, 6)
        log["diff_pixels"] = px
        print(f"\n拖拽后像素变化: {px} px  比例 {ratio:.6f}")
        verdict = "✅ canvas 接收到了拖拽" if ratio > 0.0005 else "❌ 像素无变化，拖拽未被接收"
        log["drag_registered"] = ratio > 0.0005
        print(verdict)

        # 再提交一次，看服务端反应
        try:
            challenge.evaluate(
                "() => { const b=document.querySelector('.button-submit'); if(b) b.click(); }")
            time.sleep(2)
            err = challenge.evaluate(
                "() => { const e=document.querySelector('.display-error');"
                " return e && e.offsetParent!==null ? e.innerText : null; }")
            log["hcaptcha_error"] = err
            print(f"提交后返回: {err!r}")
        except Exception as e:
            print("提交异常:", e)

    with open(os.path.join(OUT, "log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"产物: {OUT}/before.png  {OUT}/after.png  {OUT}/log.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
