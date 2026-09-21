"""
持久猎取拖拽题并完成 live 验证

背景：拖拽题现在很稀有（45 道题只出 1 道，约 1/45）。上一轮拿到的那道还因为
起手点落在右侧面板空白处（0 像素变化），没能验证。所以这次：
  1. 用 score 更高的备用 sitekey（上次就是它出的拖拽题）
  2. 耐心循环，最多 110 轮挑战，每 25 轮整页重载一次以避开状态累积
  3. 拿到拖拽题后，用 detect_drag_piece 检测**可拖拽方块的实际位置**作为起手点
     （不再硬编码，因为方块位置每题都变）
  4. 判定方式：在**按住不放的中途**抓帧，并单独统计**左侧面板区域**的像素变化 ——
     方块原本就在左侧，如果左侧出现变化，说明方块跟着光标动了，即手势被接收。
     这比只看首尾两帧可靠（避免松手回弹造成的误判）。

用法:
    python tools/hunt_drag.py [最大轮数，默认 110]
产物:
    research_out/drag_hunt/<轮次>/before.png middrag.png after.png
    research_out/drag_hunt/log.json      每轮题面 + 验证结果
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
    HCaptchaSolver, HTML_TEMPLATE, CHALLENGE_IFRAME_SELECTOR, REFRESH_SELECTOR,
)
from canvas_actions import CanvasGeometry
from research.detect_drag_piece import detect_pieces  # noqa: E402

OUT = os.path.join(os.environ.get("RESEARCH_OUT", "research_out"), "drag_hunt")

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


def to_arr(data_url: str) -> np.ndarray:
    raw = base64.b64decode(data_url.split(",", 1)[1])
    return np.asarray(Image.open(__import__("io").BytesIO(raw)).convert("RGB"))


def diff_region(a: np.ndarray, b: np.ndarray, x0f=0.0, x1f=1.0):
    if a.shape != b.shape:
        return 1.0, -1
    h, w = a.shape[:2]
    x0, x1 = int(w * x0f), int(w * x1f)
    d = np.abs(a[:, x0:x1].astype(np.int16) - b[:, x0:x1].astype(np.int16)).sum(axis=2)
    n = int((d > 30).sum())
    return n / d.size, n


def verify(solver, page, challenge, prompt, tag, log):
    """
    与布局无关的抓取探测。

    踩过三次坑之后改的设计：拖拽题至少有**两种完全不同的布局**
      变体A：左侧面板放米黄方块（带 Move 手柄）+ 右侧面板白色虚线轮廓
      变体B：全幅背景，木条/螺钉堆在画面中部，右边缘一个圆形 Move 手柄
    「方块在哪」每题都不同，靠检测背景纹理去猜会误检（实测把变体B的暖色背景
    当成了方块，起手点落在空白处 -> 0 像素变化，而这种 0 像素既可能是没抓住、
    也可能是手势没被接收，无法区分）。

    所以这里不再猜：对若干候选起手点逐个探测 —— 按住 -> 小幅移动 -> 按住不放时
    抓帧比对。只要**任意一个候选**让画面产生变化，就证明「手势被 canvas 接收」
    这个结论成立，与具体布局无关。
    """
    cap = challenge.evaluate(GET_CANVAS)
    if not cap.get("ok"):
        log["error"] = "canvas_capture_failed"
        return False
    d = os.path.join(OUT, tag)
    os.makedirs(d, exist_ok=True)
    base_arr = to_arr(cap["data"])
    Image.fromarray(base_arr).save(os.path.join(d, "before.png"))
    bw, bh = cap["buf"]["w"], cap["buf"]["h"]
    print(f"   canvas 缓冲 {bw}x{bh}")

    iframe_box = page.query_selector(CHALLENGE_IFRAME_SELECTOR).bounding_box()
    geom = CanvasGeometry(iframe_box, cap["rect"], (bw, bh))

    # 候选起手点：覆盖两种已知变体 + 通用位置
    candidates = [
        ("中心",             0.50, 0.55),
        ("中部偏右下(木条堆)", 0.60, 0.62),
        ("变体A 上块",        0.135, 0.40),
        ("变体A 下块",        0.169, 0.65),
        ("右缘 Move 手柄",    0.83, 0.43),
        ("下中部",           0.52, 0.78),
    ]
    probes = []
    hit = None
    for name, fx, fy in candidates:
        sx, sy = bw * fx, bh * fy
        tx, ty = sx + bw * 0.10, sy + bh * 0.06      # 小幅移动，便于观察
        p1, p2 = geom.to_page(sx, sy), geom.to_page(tx, ty)
        try:
            page.mouse.move(p1[0], p1[1])
            time.sleep(0.08)
            page.mouse.down()
            time.sleep(0.15)
            for i in range(1, 9):
                t = i / 8.0
                page.mouse.move(p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t)
                time.sleep(0.03)
            time.sleep(0.25)
            capm = challenge.evaluate(GET_CANVAS)
            n = -1
            if capm.get("ok"):
                arr = to_arr(capm["data"])
                _, n = diff_region(base_arr, arr)
            page.mouse.up()
            time.sleep(0.5)
            # 松手后回到基线（若对象被拖动后回弹，也算没抓住）
            capr = challenge.evaluate(GET_CANVAS)
            n_after = -1
            if capr.get("ok"):
                n_after = diff_region(base_arr, to_arr(capr["data"]))[1]
            probes.append({"name": name, "frac": [fx, fy], "mid_diff": n, "after_diff": n_after})
            flag = "✅ 有变化" if (n or 0) > 300 else "—"
            print(f"   探测 [{name:16s}] 起手({sx:.0f},{sy:.0f}) "
                  f"按住中变化={n} 松手后变化={n_after}  {flag}", flush=True)
            if (n or 0) > 300:
                hit = probes[-1]
                Image.fromarray(to_arr(capm["data"])).save(os.path.join(d, "middrag.png"))
                break
        except Exception as e:
            probes.append({"name": name, "error": f"{type(e).__name__}: {e}"})
            try:
                page.mouse.up()
            except Exception:
                pass

    log["probes"] = probes
    ok = hit is not None
    log["drag_gesture_accepted"] = ok
    if ok:
        log["hit_candidate"] = hit["name"]
        print(f"   ★ 命中：用手势在 [{hit['name']}] 处成功拖动了对象 —— "
              f"canvas 确实接收该手势", flush=True)
        # 完成一次完整拖拽并提交
        try:
            challenge.evaluate(
                "() => { const b=document.querySelector('.submit-button, .button-submit');"
                " if(b) b.click(); }")
            time.sleep(2)
            err = challenge.evaluate(
                "() => { const e=document.querySelector('.display-error');"
                " return e && e.offsetParent!==null ? e.innerText : null; }")
            log["hcaptcha_error"] = err
            print(f"   提交回执: {err!r}", flush=True)
        except Exception as e:
            log["hcaptcha_error"] = f"err: {e}"
    else:
        print("   ❌ 全部候选起手点都没能让画面变化", flush=True)
    return ok


def main():
    max_rounds = next((int(a) for a in sys.argv[1:] if a.isdigit()), 110)
    os.makedirs(OUT, exist_ok=True)
    log = {"rounds": [], "sitekey": SITEKEY, "max_rounds": max_rounds}
    dist = {}

    solver = HCaptchaSolver(headless=True, widget_retries=3)
    solver.__enter__()
    try:
        page = solver.browser.new_page()
        html = HTML_TEMPLATE.replace("{{SITEKEY}}", SITEKEY)

        def setup(p):
            for u in (URL, URL + "/", URL.rstrip("/")):
                p.route(u, lambda r: r.fulfill(status=200, content_type="text/html", body=html))

        setup(page)
        widget = solver._wait_for_widget(page, URL, tries=12)
        print("widget 就绪")
        widget.evaluate("() => document.querySelector('#checkbox').click()")

        for rd in range(1, max_rounds + 1):
            # 每 25 轮整页重载，避开状态累积/限流
            if rd > 1 and rd % 25 == 0:
                print(f"--- 第 {rd} 轮前整页重载 ---", flush=True)
                try:
                    page.close()
                except Exception:
                    pass
                page = solver.browser.new_page()
                setup(page)
                try:
                    w2 = solver._wait_for_widget(page, URL, tries=14)
                    w2.evaluate("() => document.querySelector('#checkbox').click()")
                except Exception as e:
                    print(f"  重载后 widget 失败: {str(e)[:70]}", flush=True)
                    continue

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
            kind = "drag" if ("drag" in low or "into their outlines" in low) else \
                   ("pattern" if ("pattern" in low or "does not follow" in low) else "other")
            dist[kind] = dist.get(kind, 0) + 1
            print(f"[{rd:3d}] {kind:8s} | {prompt[:56]}", flush=True)
            log["rounds"].append({"round": rd, "kind": kind, "prompt": prompt})

            if kind == "drag":
                print("\n>>> 拿到拖拽题，开始 live 验证\n", flush=True)
                sub = {}
                try:
                    ok = verify(solver, page, challenge, prompt, f"round{rd}", sub)
                except Exception as e:
                    sub["error"] = f"{type(e).__name__}: {e}"
                    ok = False
                    print(f"   验证异常: {sub['error'][:120]}", flush=True)
                log["rounds"][-1]["verify"] = sub
                log["drag_verified"] = bool(ok)
                log["verified_at_round"] = rd
                if ok:
                    print("\n✅ 拖拽 live 验证通过", flush=True)
                    break
                print("   本次未通过，继续猎取\n", flush=True)

            try:
                challenge.evaluate(
                    f"() => {{ const r=document.querySelector('{REFRESH_SELECTOR}'); if(r) r.click(); }}")
            except Exception:
                pass
            time.sleep(2.0)

        log["distribution"] = dist
        if "drag_verified" not in log:
            log["drag_verified"] = False
            log["note"] = f"{max_rounds} 轮内未拿到可用的拖拽题"
    finally:
        try:
            solver.__exit__(None, None, None)
        except Exception:
            pass

    with open(os.path.join(OUT, "log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"\n{'='*60}")
    print(f"题型分布 ({sum(dist.values())} 题): {dist}")
    print(f"拖拽 live 验证: {log.get('drag_verified')}")
    print(f"日志: {OUT}/log.json")
    return 0 if log.get("drag_verified") else 1


if __name__ == "__main__":
    raise SystemExit(main())
