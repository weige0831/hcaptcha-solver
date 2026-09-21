"""
拖拽题型 live 验证（多配置扫描）

问题：拖拽动作已通过机制级验证（自建 canvas，8/8），但「hCaptcha 的 canvas 是否
接受这个手势」一直没测到 —— 连续 28 道题全是 pattern，一次没遇到拖拽题。
题型由服务端决定，可能受浏览器指纹/行为影响（早先批次还是 4/10 是 drag）。

所以这里不赌单一配置，而是扫几组配置，看哪组能拿到拖拽题：
    A. headless + 主 sitekey            （基线，已知 0/28）
    B. headless + 主 sitekey + humanize （拟人化输入，可能影响判定）
    C. headless + 备用 sitekey          （换 sitekey 换个题目池）
    D. 有头模式 + 主 sitekey            （有头/无头的判定常不同）

每组最多刷新 N 次；一旦遇到拖拽题，就执行拖拽 -> 像素差分 -> 提交 -> 读回执。
同时统计各配置的题型分布（这个数据本身也有用：drag 占多少）。

用法:
    python tools/verify_drag_sweep.py [每组最多刷新次数]
产物:
    research_out/drag_sweep/report.json  各配置题型分布与拖拽验证结果
    research_out/drag_sweep/<配置>/before.png after.png
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
    REFRESH_SELECTOR, is_unsupported_challenge,
)
from canvas_actions import CanvasGeometry, drag_and_drop

OUT = os.path.join(os.environ.get("RESEARCH_OUT", "research_out"), "drag_sweep")

SITEKEY_MAIN = "a5f74b19-9e45-40e0-b45d-47ff91b7a6c2"
URL_MAIN = "https://accounts.hcaptcha.com/demo"
SITEKEY_ALT = "338af34c-7bcb-4c7c-900b-acbec73d7d43"
URL_ALT = "https://democaptcha.com/demo-form-eng/hcaptcha.html"

GET_CANVAS = r"""() => {
    try {
        const c = document.querySelector('canvas');
        const r = c.getBoundingClientRect();
        return {ok: true, data: c.toDataURL('image/png'),
                rect: {x: r.x, y: r.y, w: r.width, h: r.height},
                buf: {w: c.width, h: c.height}};
    } catch (e) { return {ok: false, err: String(e)}; }
}"""


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


def save_png(data_url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(base64.b64decode(data_url.split(",", 1)[1]))


def diff(before_p, after_p):
    a = np.asarray(Image.open(before_p).convert("RGB"), dtype=np.int16)
    b = np.asarray(Image.open(after_p).convert("RGB"), dtype=np.int16)
    if a.shape != b.shape:
        return 1.0, -1
    d = np.abs(a - b).sum(axis=2)
    n = int((d > 30).sum())
    return n / d.size, n


def run_config(label, sitekey, url, headless, humanize, max_refresh):
    """返回 {distribution, drag_verified, ...}"""
    rec = {"label": label, "headless": headless, "humanize": humanize,
           "sitekey": sitekey, "distribution": {}, "rounds": 0}
    print(f"\n{'='*60}\n配置 {label}: headless={headless} humanize={humanize} "
          f"sitekey={sitekey[:8]}...\n{'='*60}")

    solver = HCaptchaSolver(headless=headless, humanize=humanize,
                            widget_retries=3, on_unsupported=lambda p: None)
    solver.__enter__()
    try:
        page = solver.browser.new_page()
        html = HTML_TEMPLATE.replace("{{SITEKEY}}", sitekey)
        for u in (url, url + "/", url.rstrip("/")):
            page.route(u, lambda r: r.fulfill(status=200, content_type="text/html", body=html))

        # widget（有头模式下渲染更稳，重试次数给小一点）
        widget = None
        for attempt in range(1, 4):
            try:
                widget = solver._wait_for_widget(page, url, tries=12)
                break
            except Exception as e:
                print(f"  widget 重试 {attempt}: {str(e)[:60]}")
        if not widget:
            rec["error"] = "widget_not_rendered"
            print("  ❌ widget 未渲染")
            return rec
        print("  widget 就绪")

        try:
            widget.evaluate("() => document.querySelector('#checkbox').click()")
        except Exception as e:
            rec["error"] = f"checkbox_click: {e}"
            return rec

        for rd in range(1, max_refresh + 1):
            challenge = None
            for _ in range(14):
                time.sleep(1.5)
                cf = solver._get_challenge_frame(page)
                if cf:
                    challenge = cf
                    break
            if not challenge:
                print(f"  [{rd}] 挑战未出现")
                continue
            prompt = solver._read_prompt(challenge)
            kind = classify(prompt)
            rec["distribution"][kind] = rec["distribution"].get(kind, 0) + 1
            rec["rounds"] = rd
            print(f"  [{rd}] {kind:9s} | {prompt[:58]}")

            if kind == "drag":
                print("  >>> 拿到拖拽题，开始 live 验证")
                return verify_drag_live(solver, page, challenge, prompt, rec, label)

            try:
                challenge.evaluate(
                    f"() => {{ const r=document.querySelector('{REFRESH_SELECTOR}'); if(r) r.click(); }}")
            except Exception:
                pass
            time.sleep(2.5)

        print(f"  {max_refresh} 次刷新内未遇到拖拽题")
        rec["drag_verified"] = False
        rec["note"] = f"{max_refresh} 次刷新未遇到拖拽题"
        return rec
    finally:
        try:
            solver.__exit__(None, None, None)
        except Exception:
            pass


def verify_drag_live(solver, page, challenge, prompt, rec, label):
    cap = challenge.evaluate(GET_CANVAS)
    if not cap.get("ok"):
        rec["drag_verified"] = False
        rec["error"] = "canvas_capture_failed"
        return rec
    before = os.path.join(OUT, label, "before.png")
    save_png(cap["data"], before)
    bw, bh = cap["buf"]["w"], cap["buf"]["h"]
    print(f"   canvas CSS {cap['rect']['w']}x{cap['rect']['h']} 缓冲 {bw}x{bh}")

    iframe_box = page.query_selector(CHALLENGE_IFRAME_SELECTOR).bounding_box()
    geom = CanvasGeometry(iframe_box, cap["rect"], (bw, bh))

    # 拖拽题常见形态是「把 X 拖到 Y」，源在画面中部、目标在别处。
    # 做一次大跨度拖拽，覆盖画布上半到下半，便于观察是否被接收。
    x1, y1 = bw * 0.5, bh * 0.45
    x2, y2 = bw * 0.5, bh * 0.80
    p1, p2 = geom.to_page(x1, y1), geom.to_page(x2, y2)
    print(f"   拖拽 缓冲({x1:.0f},{y1:.0f})->({x2:.0f},{y2:.0f})  页面({p1[0]:.0f},{p1[1]:.0f})->({p2[0]:.0f},{p2[1]:.0f})")
    drag_and_drop(page, p1[0], p1[1], p2[0], p2[1])

    cap2 = challenge.evaluate(GET_CANVAS)
    if not cap2.get("ok"):
        rec["drag_verified"] = False
        return rec
    after = os.path.join(OUT, label, "after.png")
    save_png(cap2["data"], after)

    ratio, px = diff(before, after)
    rec["canvas_diff_pixels"] = px
    rec["canvas_diff_ratio"] = round(ratio, 6)
    rec["drag_verified"] = ratio > 0.0005
    print(f"   像素变化 {px} px ({ratio:.6f}) -> "
          f"{'✅ canvas 接收到了拖拽' if rec['drag_verified'] else '❌ 无变化，手势未被接收'}")

    try:
        challenge.evaluate(
            "() => { const b=document.querySelector('.button-submit'); if(b) b.click(); }")
        time.sleep(2)
        err = challenge.evaluate(
            "() => { const e=document.querySelector('.display-error');"
            " return e && e.offsetParent!==null ? e.innerText : null; }")
        rec["hcaptcha_error"] = err
        print(f"   提交回执: {err!r}")
    except Exception as e:
        rec["hcaptcha_error"] = f"submit_err: {e}"
    return rec


CONFIGS = {
    "A_headless_main":      (SITEKEY_MAIN, URL_MAIN, True,  False),
    "B_headless_humanized": (SITEKEY_MAIN, URL_MAIN, True,  True),
    "C_headless_altsite":   (SITEKEY_ALT,  URL_ALT,  True,  False),
    "D_headed_main":        (SITEKEY_MAIN, URL_MAIN, False, False),
}


def main():
    """
    只跑一个配置（配置名由命令行给出）。

    为什么一次只跑一个：Camoufox/Playwright 的 sync API 在同一个进程里反复
    __enter__/__exit__ 会抛 "Sync API inside the asyncio loop"（前面踩过），
    而不同配置需要不同的启动参数（headless/humanize），必须是不同浏览器实例。
    所以由外层脚本对每个配置各起一个进程。
    """
    args = [a for a in sys.argv[1:] if not a.isdigit()]
    max_refresh = next((int(a) for a in sys.argv[1:] if a.isdigit()), 14)
    label = args[0] if args else None
    if label not in CONFIGS:
        print(f"用法: python tools/verify_drag_sweep.py <配置名> [每组最多刷新次数]")
        print(f"可选配置: {', '.join(CONFIGS)}")
        return 2

    sk, url, hl, hum = CONFIGS[label]
    os.makedirs(OUT, exist_ok=True)
    try:
        rec = run_config(label, sk, url, hl, hum, max_refresh)
    except Exception as e:
        rec = {"label": label, "error": f"{type(e).__name__}: {e}"}
        print(f"  配置异常: {rec['error'][:150]}")

    with open(os.path.join(OUT, f"{label}.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    print(f"\n本配置结果: drag_verified={rec.get('drag_verified')} "
          f"dist={rec.get('distribution')}")
    print(f"写入 {OUT}/{label}.json")
    return 0 if rec.get("drag_verified") else 1


if __name__ == "__main__":
    raise SystemExit(main())
