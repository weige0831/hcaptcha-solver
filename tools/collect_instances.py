"""
采集多道 pattern 题，检验「无离群」这个结论是否可推广

为什么必须做：前面五份否定性结论（旋转掩膜、转角函数、相似度聚合、尺寸形状、
匹配滤波）**全部建立在同一张图上**（research_out/click/before.png 那枚火箭/星云题）。
单实例结论可能过拟合：万一那张图恰好是特别难的（星云高对比背景干扰分割与匹配），
我就会把"某些实例其实可解"误判成"整类不可解"。

所以这里做两件事：
  1. 连续采集 N 道 pattern 题的 canvas，各自存盘
  2. 对每张图跑一遍匹配滤波（最干净的那个检验：不做分割，直接在原始像素上
     旋转模板匹配），统计「稳健 z < -3.5 的图标数」

判读：
  - 若所有实例都是 0 个离群 => 否定结论可推广，CV 路线确实死透
  - 若某些实例恰好出现 2 个离群 => 说明部分题可解，值得深挖（这是重大发现）

用法:
    python tools/collect_instances.py 8
产物:
    research_out/instances/pat_*.png
    research_out/instances/report.json
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
from research.matched_filter import best_response_map, build_template  # noqa: E402
from canvas_strategy import detect_icons  # noqa: E402

OUT = os.path.join(os.environ.get("RESEARCH_OUT", "research_out"), "instances")
SITEKEY = "a5f74b19-9e45-40e0-b45d-47ff91b7a6c2"
URL = "https://accounts.hcaptcha.com/demo"

GET_CANVAS = r"""() => {
    try {
        const c = document.querySelector('canvas');
        return {ok: true, data: c.toDataURL('image/png')};
    } catch (e) { return {ok: false, err: String(e)}; }
}"""


def classify(p: str) -> str:
    p = (p or "").lower()
    if "drag" in p or "into their outlines" in p:
        return "drag"
    if "break the pattern" in p or "does not follow" in p or "follow the pattern" in p:
        return "pattern"
    return "other"


def analyze(path: str) -> dict:
    """对单张 canvas 跑匹配滤波离群检验（与 research/matched_filter.py 同法）"""
    arr = np.asarray(Image.open(path))
    rgb = arr[:, :, :3].copy()
    alpha = arr[:, :, 3] if arr.shape[2] == 4 else None
    top = 0
    if alpha is not None:
        rows = (alpha > 10).sum(axis=1)
        nz = np.where(rows > 0)[0]
        top = int(nz.min()) if len(nz) else 0
    rgb_p = rgb[top:, :].copy()
    gray = Image.fromarray(rgb_p).convert("L")
    gray = np.asarray(gray)

    icons = [ic for ic in detect_icons(rgb, alpha) if ic["cy"] - top > 10]
    if len(icons) < 4:
        return {"error": f"图标太少 ({len(icons)})", "n_icons": len(icons)}

    refs = sorted(icons, key=lambda d: -d["area"])[:3]
    acc = np.full(gray.shape, -1.0, np.float32)
    for r in refs:
        tpl = build_template(rgb_p, r["cx"], r["cy"] - top)
        if tpl is None:
            continue
        resp = best_response_map(gray, np.asarray(Image.fromarray(tpl).convert("L")))
        np.maximum(acc, resp, out=acc)

    h, w = gray.shape
    vals = []
    for ic in icons:
        cx, cy = int(ic["cx"]), int(ic["cy"] - top)
        x0, x1 = max(0, cx - 26), min(w, cx + 26)
        y0, y1 = max(0, cy - 26), min(h, cy + 26)
        vals.append(float(acc[y0:y1, x0:x1].max()))
    vals = np.array(vals)
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) or 1e-9
    rz = (vals - med) / (1.4826 * mad)
    low = int((rz < -3.5).sum())

    # 若恰好 2 个离群，给出它们的位置（供人工核验）
    picks = []
    if low == 2:
        order = np.argsort(rz)[:2]
        picks = [[round(icons[i]["cx"]), round(icons[i]["cy"])] for i in order]

    return {"n_icons": len(icons), "median": round(med, 3),
            "min": round(float(vals.min()), 3), "max": round(float(vals.max()), 3),
            "n_outliers_z3.5": low,
            "min_z": round(float(rz.min()), 2),
            "outlier_positions": picks}


def main():
    n_want = next((int(a) for a in sys.argv[1:] if a.isdigit()), 8)
    os.makedirs(OUT, exist_ok=True)
    report = {"instances": []}

    solver = HCaptchaSolver(headless=True, widget_retries=3)
    solver.__enter__()
    try:
        page = solver.browser.new_page()
        html = HTML_TEMPLATE.replace("{{SITEKEY}}", SITEKEY)
        for u in (URL, URL + "/", URL.rstrip("/")):
            page.route(u, lambda r: r.fulfill(status=200, content_type="text/html", body=html))
        widget = solver._wait_for_widget(page, URL, tries=12)
        print("widget 就绪", flush=True)
        widget.evaluate("() => document.querySelector('#checkbox').click()")

        got, rounds = 0, 0
        while got < n_want and rounds < n_want * 8:
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
            kind = classify(prompt)
            print(f"[{rounds:3d}] {kind:8s} | {prompt[:50]}", flush=True)

            if kind == "pattern":
                cap = challenge.evaluate(GET_CANVAS)
                if cap.get("ok"):
                    got += 1
                    p = os.path.join(OUT, f"pat_{got:02d}.png")
                    with open(p, "wb") as f:
                        f.write(base64.b64decode(cap["data"].split(",", 1)[1]))
                    print(f"      已存 {p}", flush=True)
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

    print(f"\n共采集 {got} 道 pattern 题，开始逐题分析...\n", flush=True)
    print(f"{'实例':<12}{'图标数':>6}{'中位响应':>10}{'最低z':>8}{'z<-3.5数':>10}")
    print("-" * 48)
    for i in range(1, got + 1):
        p = os.path.join(OUT, f"pat_{i:02d}.png")
        r = analyze(p)
        r["file"] = os.path.basename(p)
        report["instances"].append(r)
        if "error" in r:
            print(f"{r['file']:<12}  {r['error']}")
        else:
            print(f"{r['file']:<12}{r['n_icons']:>6}{r['median']:>10.3f}"
                  f"{r['min_z']:>8.2f}{r['n_outliers_z3.5']:>10}")

    ok = [r for r in report["instances"] if "error" not in r]
    counts = [r["n_outliers_z3.5"] for r in ok]
    report["summary"] = {
        "analyzed": len(ok),
        "outlier_counts": counts,
        "instances_with_exactly_2": sum(1 for c in counts if c == 2),
    }
    print(f"\n=== 判读 ===")
    print(f"  成功分析的实例: {len(ok)}")
    print(f"  各实例离群数: {counts}")
    if counts and all(c == 0 for c in counts):
        print("  => 所有实例都是 0 个离群：**否定结论可推广**，不是单张图的特例。")
    elif any(c == 2 for c in counts):
        print("  => 有实例恰好出现 2 个离群！CV 路线可能并未死透，值得深挖。")
    else:
        print("  => 结果不一致，需逐例查看。")

    with open(os.path.join(OUT, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告: {OUT}/report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
