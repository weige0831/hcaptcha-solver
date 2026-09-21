"""
拖拽 live 验证（修正瞄准后）

上一轮结论是"0 像素变化 -> 手势未被接收"，但看了 before.png 才发现是**我瞄错了**：
这类题是左侧面板放可拖拽的方块（各带 Move 手柄），右侧面板是白色虚线空轮廓。
我上次从 (0.5*W, 0.45*H) 开始拖 —— 那里在右侧面板的空白处，
根本没抓住任何东西，所以没有像素变化。0 像素不能推出"手势未被接收"。

这次修正三点：
  1. 起点放在**左侧可拖拽方块**上（据截图约为缓冲坐标 (0.135W, 0.40H)）
  2. 终点放到某个**虚线轮廓**上（约 (0.60W, 0.53H)）
  3. 关键改进：在**按住不放的中途**抓一帧。如果方块跟着光标走，
     说明手势确实被接收了 —— 这比只看首尾两帧更能定性，
     也避免了"松手后回弹"导致的误判。

用法:
    python tools/verify_drag_live2.py
产物:
    research_out/drag_live/before.png / middrag.png / after.png / log.json
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
    HCaptchaSolver, HTML_TEMPLATE, CHALLENGE_IFRAME_SELECTOR,
    REFRESH_SELECTOR,
)
from canvas_actions import CanvasGeometry

OUT = os.path.join(os.environ.get("RESEARCH_OUT", "research_out"), "drag_live")

# 备用 sitekey 上次成功下发了拖拽题
SITEKEY = "338af34c-7bcb-4c7c-900b-acbec73d7d43"
URL = "https://democaptcha.com/demo-form-eng/hcaptcha.html"

GET_CANVAS = r"""() => {
    try {
        const c = document.querySelector('canvas');
        const r = c.getBoundingClientRect();
        return {ok: true, data: c.toDataURL('image/png'),
                rect: {x: r.x, y: r.y, w: r.width, h: r.height},
                buf: {w: c.width, h: c.height}};
    } catch (e) { return {ok: false, err: String(e)}; }
}"""


def save(data_url, path):
    with open(path, "wb") as f:
        f.write(base64.b64decode(data_url.split(",", 1)[1]))


def diff(p1, p2):
    a = np.asarray(Image.open(p1).convert("RGB"), dtype=np.int16)
    b = np.asarray(Image.open(p2).convert("RGB"), dtype=np.int16)
    if a.shape != b.shape:
        return 1.0, -1
    d = np.abs(a - b).sum(axis=2)
    n = int((d > 30).sum())
    return n / d.size, n


def main():
    os.makedirs(OUT, exist_ok=True)
    log = {}

    solver = HCaptchaSolver(headless=True, widget_retries=3)
    solver.__enter__()
    try:
        page = solver.browser.new_page()
        html = HTML_TEMPLATE.replace("{{SITEKEY}}", SITEKEY)
        for u in (URL, URL + "/", URL.rstrip("/")):
            page.route(u, lambda r: r.fulfill(status=200, content_type="text/html", body=html))

        widget = solver._wait_for_widget(page, URL, tries=12)
        print("widget 就绪")
        widget.evaluate("() => document.querySelector('#checkbox').click()")

        challenge, prompt = None, ""
        for rd in range(1, 31):
            challenge = None
            for _ in range(14):
                time.sleep(1.5)
                cf = solver._get_challenge_frame(page)
                if cf:
                    challenge = cf
                    break
            if not challenge:
                continue
            prompt = solver._read_prompt(challenge)
            if "drag" in prompt.lower() or "into their outlines" in prompt.lower():
                print(f"[{rd}] 拿到拖拽题: {prompt!r}")
                break
            try:
                challenge.evaluate(
                    f"() => {{ const r=document.querySelector('{REFRESH_SELECTOR}'); if(r) r.click(); }}")
            except Exception:
                pass
            time.sleep(2.2)
        else:
            print("❌ 30 次刷新未遇到拖拽题")
            log["got_drag"] = False
            with open(os.path.join(OUT, "log.json"), "w", encoding="utf-8") as f:
                json.dump(log, f, ensure_ascii=False, indent=2)
            return 1

        log.update({"got_drag": True, "prompt": prompt})
        cap = challenge.evaluate(GET_CANVAS)
        if not cap.get("ok"):
            print("canvas 抓取失败")
            return 1
        before = os.path.join(OUT, "before.png")
        save(cap["data"], before)
        bw, bh = cap["buf"]["w"], cap["buf"]["h"]
        print(f"canvas CSS {cap['rect']['w']}x{cap['rect']['h']} 缓冲 {bw}x{bh}")
        log["canvas"] = {"css": [cap["rect"]["w"], cap["rect"]["h"]], "buf": [bw, bh]}

        iframe_box = page.query_selector(CHALLENGE_IFRAME_SELECTOR).bounding_box()
        geom = CanvasGeometry(iframe_box, cap["rect"], (bw, bh))

        # 据截图：可拖拽方块在左侧面板，虚线轮廓在右侧
        sx, sy = bw * 0.135, bh * 0.40          # 左上那个方块
        tx, ty = bw * 0.600, bh * 0.53          # 左上那个虚线轮廓
        p1 = geom.to_page(sx, sy)
        p2 = geom.to_page(tx, ty)
        print(f"抓取点 缓冲({sx:.0f},{sy:.0f}) -> 页面({p1[0]:.0f},{p1[1]:.0f})")
        print(f"目标点 缓冲({tx:.0f},{ty:.0f}) -> 页面({p2[0]:.0f},{p2[1]:.0f})")
        log["source_xy"] = [round(sx), round(sy)]
        log["target_xy"] = [round(tx), round(ty)]

        # 1) 先点一下方块（有些实现需要先选中，截图里方块上有 Move 手柄）
        page.mouse.move(p1[0], p1[1])
        page.mouse.click(p1[0], p1[1])
        time.sleep(0.5)
        cap_sel = challenge.evaluate(GET_CANVAS)
        if cap_sel.get("ok"):
            sel = os.path.join(OUT, "after_select.png")
            save(cap_sel["data"], sel)
            r, n = diff(before, sel)
            log["select_diff_pixels"] = n
            print(f"点击方块后像素变化: {n} px ({r:.6f})")

        # 2) 按住并分两段移动，中途抓帧
        page.mouse.move(p1[0], p1[1])
        time.sleep(0.08)
        page.mouse.down()
        time.sleep(0.15)
        mid = (p1[0] + (p2[0] - p1[0]) * 0.55, p1[1] + (p2[1] - p1[1]) * 0.55)
        for i in range(1, 9):
            t = i / 8.0
            page.mouse.move(p1[0] + (mid[0] - p1[0]) * t, p1[1] + (mid[1] - p1[1]) * t)
            time.sleep(0.025)
        time.sleep(0.25)
        # 按住不放时抓帧
        cap_mid = challenge.evaluate(GET_CANVAS)
        middrag = os.path.join(OUT, "middrag.png")
        if cap_mid.get("ok"):
            save(cap_mid["data"], middrag)
            r_mid, n_mid = diff(before, middrag)
            log["middrag_diff_pixels"] = n_mid
            log["piece_follows_cursor"] = n_mid > 500
            print(f"★ 按住中途像素变化: {n_mid} px ({r_mid:.6f}) -> "
                  f"{'✅ 方块跟着光标走，手势被接收' if n_mid > 500 else '❌ 方块未动，手势未被接收'}")

        # 3) 完成拖拽并松手
        for i in range(1, 9):
            t = i / 8.0
            page.mouse.move(mid[0] + (p2[0] - mid[0]) * t, mid[1] + (p2[1] - mid[1]) * t)
            time.sleep(0.025)
        time.sleep(0.15)
        page.mouse.up()
        time.sleep(0.8)

        cap2 = challenge.evaluate(GET_CANVAS)
        if cap2.get("ok"):
            after = os.path.join(OUT, "after.png")
            save(cap2["data"], after)
            r2, n2 = diff(before, after)
            log["final_diff_pixels"] = n2
            log["drag_verified"] = n2 > 500
            print(f"松手后像素变化: {n2} px ({r2:.6f}) -> "
                  f"{'✅ 拖拽生效' if n2 > 500 else '❌ 未生效（可能松手回弹或目标不对）'}")

        try:
            challenge.evaluate(
                "() => { const b=document.querySelector('.button-submit'); if(b) b.click(); }")
            time.sleep(2)
            err = challenge.evaluate(
                "() => { const e=document.querySelector('.display-error');"
                " return e && e.offsetParent!==null ? e.innerText : null; }")
            log["hcaptcha_error"] = err
            print(f"提交回执: {err!r}")
        except Exception as e:
            log["hcaptcha_error"] = f"err: {e}"
    finally:
        try:
            solver.__exit__(None, None, None)
        except Exception:
            pass

    with open(os.path.join(OUT, "log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"产物: {OUT}/  (before/middrag/after.png, log.json)")
    return 0 if log.get("drag_verified") else 1


if __name__ == "__main__":
    raise SystemExit(main())
