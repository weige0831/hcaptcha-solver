"""
坐标级点击可行性验证

决定架构的关键问题：canvas 题型到底收不收「按坐标点击」？
  - 如果点得动（canvas 像素发生变化、出现选中标记），那坐标级操作这条路成立，
    只需要再解决「点哪里」的视觉推理问题。
  - 如果点不动（像素毫无变化），那要么需要更完整的指针事件序列（down/move/up），
    要么 hCaptcha 在做事件可信度校验。

做法：反复取题直到拿到「click」型题目（拖拽型用点击本来就无效，不能用来判断），
然后在拼图区域按网格采样点击，每次点击后重新抓 canvas 并做像素差分。

用法:
    python tools/verify_click.py
产物:
    research_out/click/*.png   各次点击前后的 canvas
    research_out/click_report.json
"""

import json
import os
import base64
import time
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

from hcaptcha_solver import (
    HCaptchaSolver,
    HTML_TEMPLATE,
    CHALLENGE_IFRAME_SELECTOR,
    REFRESH_SELECTOR,
)

SITEKEY = os.environ.get("HC_SITEKEY", "a5f74b19-9e45-40e0-b45d-47ff91b7a6c2")
URL = os.environ.get("HC_URL", "https://accounts.hcaptcha.com/demo")
OUT = os.path.join(os.environ.get("RESEARCH_OUT", "research_out"), "click")

GET_CANVAS = r"""() => {
    try {
        const c = document.querySelector('canvas');
        if (!c) return {ok: false, err: 'no canvas'};
        const r = c.getBoundingClientRect();
        return {ok: true, data: c.toDataURL('image/png'),
                rect: {x: r.x, y: r.y, w: r.width, h: r.height},
                buf: {w: c.width, h: c.height}};
    } catch (e) { return {ok: false, err: e.name + ': ' + e.message}; }
}"""

CLICK_REFRESH = f"() => {{ const r = document.querySelector('{REFRESH_SELECTOR}'); if (r) r.click(); }}"


def save_png(data_url, path):
    try:
        with open(path, "wb") as f:
            f.write(base64.b64decode(data_url.split(",", 1)[1]))
        return True
    except Exception:
        return False


def diff_ratio(path_a, path_b):
    """返回变化像素占比"""
    a = np.asarray(Image.open(path_a).convert("RGB"), dtype=np.int16)
    b = np.asarray(Image.open(path_b).convert("RGB"), dtype=np.int16)
    if a.shape != b.shape:
        return 1.0, "尺寸变化"
    d = np.abs(a - b).sum(axis=2)
    changed = int((d > 30).sum())
    return changed / d.size, f"{changed}px"


def main():
    os.makedirs(OUT, exist_ok=True)
    report = {"attempts": []}

    with HCaptchaSolver(headless=True) as solver:
        page = solver.browser.new_page()
        html = HTML_TEMPLATE.replace("{{SITEKEY}}", SITEKEY)
        for u in (URL, URL + "/", URL.rstrip("/")):
            page.route(u, lambda r: r.fulfill(status=200, content_type="text/html", body=html))

        widget = solver._wait_for_widget(page, URL)
        widget.evaluate("() => document.querySelector('#checkbox').click()")

        # 取一道 click 型题目
        challenge, prompt = None, ""
        for round_no in range(10):
            challenge = None
            for _ in range(16):
                time.sleep(1.5)
                cf = solver._get_challenge_frame(page)
                if cf:
                    challenge = cf
                    break
            if not challenge:
                print("挑战未出现，重试")
                continue
            prompt = solver._read_prompt(challenge)
            print(f"[{round_no+1}] 题面: {prompt!r}")
            if "click" in prompt.lower() and "drag" not in prompt.lower():
                break
            print("    不是 click 型，刷新")
            try:
                challenge.evaluate(CLICK_REFRESH)
            except Exception:
                pass
            time.sleep(3.5)
        else:
            print("未取到 click 型题目")
            return 1

        report["prompt"] = prompt

        # 基线 canvas
        cap = challenge.evaluate(GET_CANVAS)
        if not cap.get("ok"):
            print("canvas 抓取失败:", cap.get("err"))
            return 1
        before = os.path.join(OUT, "before.png")
        save_png(cap["data"], before)
        rect, buf = cap["rect"], cap["buf"]
        report["canvas"] = {"rect": rect, "buf": buf,
                            "scale": round(buf["w"] / rect["w"], 3)}
        print(f"canvas: CSS {rect['w']}x{rect['h']}, 缓冲 {buf['w']}x{buf['h']}, "
              f"缩放 {report['canvas']['scale']}x")

        iframe_box = page.query_selector(CHALLENGE_IFRAME_SELECTOR).bounding_box()

        # 拼图区域在 header 之下（实测 header 高 110 CSS px）
        header_h = 110
        poi = []
        for iy, fy in enumerate((0.35, 0.55, 0.75)):
            for ix, fx in enumerate((0.2, 0.5, 0.8)):
                css_x = rect["w"] * fx
                css_y = header_h + (rect["h"] - header_h) * fy
                poi.append((ix, iy, fx, fy, css_x, css_y))

        results = []
        for ix, iy, fx, fy, css_x, css_y in poi:
            abs_x = iframe_box["x"] + rect["x"] + css_x
            abs_y = iframe_box["y"] + rect["y"] + css_y
            try:
                page.mouse.move(abs_x, abs_y)
                page.mouse.click(abs_x, abs_y)
            except Exception as e:
                print(f"  点击异常: {e}")
            time.sleep(1.2)
            cap2 = challenge.evaluate(GET_CANVAS)
            after = os.path.join(OUT, f"after_g{iy}{ix}.png")
            if not cap2.get("ok"):
                results.append({"grid": [ix, iy], "err": cap2.get("err")})
                continue
            save_png(cap2["data"], after)
            ratio, detail = diff_ratio(before, after)
            results.append({"grid": [ix, iy], "frac": [fx, fy],
                            "abs": [round(abs_x), round(abs_y)],
                            "changed_ratio": round(ratio, 5), "detail": detail})
            flag = "✅ 有变化" if ratio > 0.0005 else "❌ 无变化"
            print(f"  第 {iy*3+ix+1} 点 ({fx:.2f},{fy:.2f}) -> {flag} 变化率={ratio:.5f} ({detail})")
            # 变化很大说明题目换了，停止
            if ratio > 0.5:
                print("    题目已切换，停止采样")
                break

        report["clicks"] = results
        any_change = any(r.get("changed_ratio", 0) > 0.0005 for r in results)
        report["clicks_register"] = any_change
        print(f"\n结论：坐标点击{'有效' if any_change else '无效'}")

        page.screenshot(path=os.path.join(OUT, "screen_final.png"), full_page=True)

    with open(os.path.join(os.environ.get("RESEARCH_OUT", "research_out"),
                           "click_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("报告已写入 click_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
