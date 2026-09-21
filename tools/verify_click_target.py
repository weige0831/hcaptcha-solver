"""
校验「live 点击是否落在预期的格子上」

为什么必须单独测：`verify_pick_coords.py` 只验证了**几何换算**（格心算出来确实
落在格框内），但**从未验证过 live 里点下去的坐标是否真的被 hCaptcha 判给了那一格**。
两者是不同的东西 —— 后者错了，即使判据选对格子也会点偏。

线索（trial_33）：判据选的是 [[3,2],[1,3]]，我从图上按"每行重复同一角色"读出的
答案也是 [[1,3],[3,2]] —— **完全一致**，可 hCaptcha 仍回 'Please try again'。
既然排序没问题，就必须查"点没点对"。

方法（利用一个已知行为）：早先探查确认过 hCaptcha 会在**被点击的位置**画一个
选中标记（圆圈+叉）。所以：
  1. 取一道真格状题，记录 16 格几何
  2. 只点**第 1 个候选**格心，抓图
  3. 与前一张图做差分，找出标记出现的位置（变化区域的重心）
  4. 判断该重心落在**预期那一格**里，还是落在隔壁
这就能区分"点对了"与"点偏了"。

用法: python tools/verify_click_target.py
产物: research_out/verify_click_target/{before,after,diff}.png + log.json
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from PIL import Image

from hcaptcha_solver import (
    HCaptchaSolver, HTML_TEMPLATE, CHALLENGE_IFRAME_SELECTOR, REFRESH_SELECTOR,
)
from canvas_actions import CanvasGeometry
from research.grid_cells3 import crop_puzzle, find_blobs, build_cells

OUT = os.path.join(os.environ.get("RESEARCH_OUT", "research_out"), "verify_click_target")
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


def to_arr(u):
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(u.split(",", 1)[1]))).convert("RGB"))


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
        w = solver._wait_for_widget(page, URL, tries=12)
        w.evaluate("() => document.querySelector('#checkbox').click()")

        # 找一道真格状题（闸门会拒绝散点图）
        for rd in range(40):
            ch = None
            for _ in range(14):
                time.sleep(1.4)
                cf = solver._get_challenge_frame(page)
                if cf:
                    ch = cf
                    break
            if not ch:
                continue
            cap = ch.evaluate(GET_CANVAS)
            if not cap.get("ok"):
                continue
            arr = to_arr(cap["data"])
            rgb, top = crop_puzzle(arr)
            cells, _ = build_cells(rgb, find_blobs(rgb), 4, debug=False)
            if not cells or len(cells) < 16:
                # 散点图，刷新找下一道
                try:
                    ch.evaluate(f"() => {{ const r=document.querySelector('{REFRESH_SELECTOR}'); if(r) r.click(); }}")
                except Exception:
                    pass
                time.sleep(2.0)
                continue

            # 拿到真格状题：点第 (0,0) 格的格心（用最左上的格子，便于辨认）
            target = (0, 0)
            t = cells[target]
            before = rgb.copy()
            iframe = page.query_selector(CHALLENGE_IFRAME_SELECTOR).bounding_box()
            geom = CanvasGeometry(iframe, cap["rect"], (cap["buf"]["w"], cap["buf"]["h"]))
            px, py = geom.to_page(t["cx"], t["cy"] + top)
            print(f"第{rd+1}轮拿到格状题。点第 {target} 格 格心(缓冲)="
                  f"({t['cx']:.0f},{t['cy']+top:.0f}) -> 页面({px:.0f},{py:.0f})")
            log["target_cell"] = list(target)
            log["cell_boxtop"] = [t["x"], t["y"], t["w"], t["h"]]
            log["page_click"] = [round(px), round(py)]
            log["buf_click"] = [round(t["cx"]), round(t["cy"] + top)]

            page.mouse.move(px, py)
            page.mouse.click(px, py)
            time.sleep(1.0)
            cap2 = ch.evaluate(GET_CANVAS)
            if not cap2.get("ok"):
                print("二次抓图失败")
                return 1
            after = to_arr(cap2["data"])

            # 差分找标记重心（只在拼图区内）
            a = before[top:, :, :3].astype(np.int16)
            b = after[top:, :, :3].astype(np.int16)
            d = np.abs(a - b).sum(axis=2)
            mask = (d > 40).astype(np.uint8)
            n = int(mask.sum())
            log["changed_px"] = n
            print(f"点击后变化像素 {n}")
            if n < 30:
                print("=> 点击没有产生可见标记，无法判定落点")
                log["verdict"] = "no_marker"
            else:
                ys, xs = np.where(mask > 0)
                mx, my = float(xs.mean()), float(ys.mean())
                log["marker_centroid_buf"] = [round(mx), round(my)]
                tx, ty, tw, th = t["x"], t["y"], t["w"], t["h"]
                inside = (tx <= mx <= tx + tw) and (ty <= my <= ty + th)
                # 该重心属于哪一格？
                got = None
                for k, c2 in cells.items():
                    if (c2["x"] <= mx <= c2["x"] + c2["w"]) and (c2["y"] <= my <= c2["y"] + c2["h"]):
                        got = k
                        break
                log["marker_in_cell"] = list(got) if got else None
                print(f"标记重心(缓冲)=({mx:.0f},{my:.0f})  落在格 {got}  预期格 {target}")
                if got == target:
                    print("=> ✅ 点击落在预期格子内 —— 坐标映射在 live 中正确")
                    log["verdict"] = "correct_cell"
                elif got is None:
                    print("=> ❌ 标记落在格框之外（点偏了）")
                    log["verdict"] = "outside"
                else:
                    print(f"=> ❌ 点到了**隔壁格** {got}（预期 {target}）—— 坐标映射有偏移")
                    log["verdict"] = f"wrong_cell_{got}"

            Image.fromarray(before).save(os.path.join(OUT, "before.png"))
            Image.fromarray(after).save(os.path.join(OUT, "after.png"))
            cv2.imwrite(os.path.join(OUT, "diff.png"), mask * 255)
            break
        else:
            print("40 轮内未拿到真格状题")
            log["verdict"] = "no_grid_challenge"
    finally:
        try:
            solver.__exit__(None, None, None)
        except Exception:
            pass
    with open(os.path.join(OUT, "log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"日志: {OUT}/log.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
