"""
canvas 题型结构探查脚本

目的：搞清楚 hCaptcha 的 canvas 题型到底是怎么渲染和接收点击的，
以便决定后续是接视觉推理模型，还是做坐标级操作。

抓三类信息：
  1. 像素   —— canvas 原图（toDataURL）+ 整屏截图
  2. 坐标   —— canvas 属性尺寸 / CSS 尺寸 / devicePixelRatio / 在页面中的位置
  3. 命中判定 —— 网格采样 elementsFromPoint，看点击落在哪个元素上；
                再装监听器做真实点击，观察选中状态怎么记录

用法:
    python tools/inspect_challenge.py
产物:
    research_out/canvas.png         canvas 原始像素
    research_out/screen.png         整屏截图
    research_out/report.json        结构报告
"""

import json
import os
import base64
import time
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hcaptcha_solver import (
    HCaptchaSolver,
    HTML_TEMPLATE,
    CHALLENGE_IFRAME_SELECTOR,
    PROMPT_SELECTOR_CANDIDATES,
    is_unsupported_challenge,
)

SITEKEY = os.environ.get("HC_SITEKEY", "a5f74b19-9e45-40e0-b45d-47ff91b7a6c2")
URL = os.environ.get("HC_URL", "https://accounts.hcaptcha.com/demo")
OUT = os.environ.get("RESEARCH_OUT", "research_out")

# 在挑战 iframe 内执行的探查 JS
PROBE_CANVAS = r"""() => {
    const c = document.querySelector('canvas');
    if (!c) return {found: false};
    const r = c.getBoundingClientRect();
    return {
        found: true,
        id: c.id, cls: c.className,
        attrW: c.width, attrH: c.height,
        cssW: Math.round(r.width * 100) / 100,
        cssH: Math.round(r.height * 100) / 100,
        rect: {x: r.x, y: r.y, w: r.width, h: r.height},
        dpr: window.devicePixelRatio,
        parentTag: c.parentElement ? c.parentElement.tagName : null,
        parentCls: c.parentElement ? c.parentElement.className : null,
        ctxAttrs: (() => { try { const x = c.getContext('2d');
            return x ? {alpha: x.getContextAttributes().alpha} : null; } catch(e) { return null; } })(),
    };
}"""

# canvas 原图；被跨域图片污染时会抛 SecurityError
PROBE_CANVAS_PNG = r"""() => {
    try {
        const c = document.querySelector('canvas');
        if (!c) return {ok: false, err: 'no canvas'};
        return {ok: true, data: c.toDataURL('image/png')};
    } catch (e) {
        return {ok: false, err: e.name + ': ' + e.message};
    }
}"""

# 网格采样命中判定 + 收集覆盖层元素
PROBE_HITTEST = r"""(samples) => {
    const c = document.querySelector('canvas');
    const cr = c.getBoundingClientRect();
    const seen = new Map();
    const grid = [];
    for (let iy = 0; iy < samples; iy++) {
        for (let ix = 0; ix < samples; ix++) {
            const x = cr.x + (cr.width  * (ix + 0.5) / samples);
            const y = cr.y + (cr.height * (iy + 0.5) / samples);
            const stack = document.elementsFromPoint(x, y);
            const sig = stack.map(e => e.tagName + '.' + (e.className || '').toString().split(' ')[0]).join(' > ');
            if (!seen.has(sig)) seen.set(sig, {count: 0, sample: {x: Math.round(x - cr.x), y: Math.round(y - cr.y)}});
            seen.get(sig).count++;
            grid.push({ix, iy, x: Math.round(x - cr.x), y: Math.round(y - cr.y), top: stack[0]
                ? stack[0].tagName + '.' + (stack[0].className || '').toString().split(' ')[0] : null,
                depth: stack.length});
        }
    }
    // 覆盖层：与 canvas 矩形有交集的非 canvas 元素
    const overlays = [];
    for (const el of document.querySelectorAll('.challenge-view *')) {
        if (el.tagName === 'CANVAS') continue;
        const r = el.getBoundingClientRect();
        if (r.width < 1 || r.height < 1) continue;
        const overlaps = !(r.right < cr.left || r.left > cr.right || r.bottom < cr.top || r.top > cr.bottom);
        if (!overlaps) continue;
        const cs = getComputedStyle(el);
        overlays.push({
            tag: el.tagName, cls: (el.className || '').toString(), id: el.id,
            role: el.getAttribute('role'),
            rect: {x: Math.round(r.x - cr.x), y: Math.round(r.y - cr.y),
                   w: Math.round(r.width), h: Math.round(r.height)},
            zIndex: cs.zIndex, pointerEvents: cs.pointerEvents, position: cs.position,
        });
    }
    return {
        canvasRect: {w: cr.width, h: cr.height},
        distinctStacks: Array.from(seen.entries()).map(([sig, v]) => ({sig, ...v})),
        grid,
        overlays: overlays.slice(0, 40),
    };
}"""

# 观察点选后的状态变化：收集带"选中"语义的类名
PROBE_STATE = r"""() => {
    const out = {canvasCls: null, selected: [], marked: []};
    const c = document.querySelector('canvas');
    if (c) out.canvasCls = (c.className || '').toString();
    document.querySelectorAll('[class*="select" i],[class*="mark" i],[class*="click" i],[class*="active" i],[class*="check" i]').forEach(e => {
        if (e.tagName === 'CANVAS') return;
        out.selected.push({tag: e.tagName, cls: (e.className || '').toString().slice(0, 70),
                           n: document.querySelectorAll(e.tagName + '.' + (e.className||'').toString().split(' ')[0]).length});
    });
    document.querySelectorAll('[aria-checked],[aria-pressed],[data-selected]').forEach(e => {
        out.marked.push({tag: e.tagName, cls: (e.className||'').toString().slice(0,50),
                         ariaChecked: e.getAttribute('aria-checked'), ariaPressed: e.getAttribute('aria-pressed'),
                         dataSelected: e.getAttribute('data-selected')});
    });
    return out;
}"""

# 挑战 iframe 里的可疑全局对象
PROBE_GLOBALS = r"""() => {
    const keys = [];
    for (const k of Object.keys(window)) {
        if (/^(on|webkit|moz|ms)/.test(k)) continue;
        const v = window[k];
        const t = typeof v;
        if (t === 'object' || t === 'function') {
            keys.push({key: k, type: t});
        }
    }
    return keys.slice(0, 120);
}"""


def save_data_url(data_url: str, path: str) -> bool:
    try:
        if "," not in data_url:
            return False
        raw = base64.b64decode(data_url.split(",", 1)[1])
        with open(path, "wb") as f:
            f.write(raw)
        return True
    except Exception as e:
        print(f"   保存失败: {e}")
        return False


def main():
    os.makedirs(OUT, exist_ok=True)
    report = {}

    with HCaptchaSolver(headless=True) as solver:
        page = solver.browser.new_page()
        html = HTML_TEMPLATE.replace("{{SITEKEY}}", SITEKEY)
        for u in (URL, URL + "/", URL.rstrip("/")):
            page.route(u, lambda r: r.fulfill(status=200, content_type="text/html", body=html))

        print("1) 加载 widget ...")
        widget = solver._wait_for_widget(page, URL)
        widget.evaluate("() => document.querySelector('#checkbox').click()")

        print("2) 等待挑战 ...")
        challenge = None
        for _ in range(16):
            time.sleep(1.5)
            cf = solver._get_challenge_frame(page)
            if cf:
                challenge = cf
                break
        if not challenge:
            print("   ✗ 挑战未出现")
            return 1

        prompt = solver._read_prompt(challenge)
        report["prompt"] = prompt
        report["is_canvas_type"] = is_unsupported_challenge(prompt)
        print(f"   题面: {prompt!r}")
        print(f"   是否为 canvas 题型: {report['is_canvas_type']}")

        # ---- 1. canvas 几何 ----
        print("3) 采集 canvas 几何 ...")
        geo = challenge.evaluate(PROBE_CANVAS)
        report["canvas_geometry"] = geo
        print(f"   {json.dumps(geo, ensure_ascii=False)}")

        # ---- 2. 像素 ----
        print("4) 采集像素 ...")
        png = challenge.evaluate(PROBE_CANVAS_PNG)
        report["toDataURL"] = {"ok": png.get("ok"), "err": png.get("err")}
        if png.get("ok"):
            ok = save_data_url(png["data"], os.path.join(OUT, "canvas.png"))
            print(f"   canvas.toDataURL 成功，已保存 canvas.png: {ok}")
        else:
            print(f"   canvas.toDataURL 失败（可能是跨域污染）: {png.get('err')}")
        page.screenshot(path=os.path.join(OUT, "screen.png"), full_page=True)
        print("   已保存 screen.png")

        # ---- 3. 命中判定 ----
        print("5) 网格采样命中判定 ...")
        hit = challenge.evaluate(PROBE_HITTEST, 10)
        report["hittest"] = {
            "distinctStacks": hit["distinctStacks"],
            "overlays": hit["overlays"],
        }
        print(f"   不同元素栈组合: {len(hit['distinctStacks'])} 种")
        for s in hit["distinctStacks"][:8]:
            print(f"     x{s['count']:>3}  {s['sig'][:110]}")
        print(f"   canvas 上方的覆盖元素: {len(hit['overlays'])} 个")
        for o in hit["overlays"][:10]:
            print(f"     {o['tag']}.{o['cls'][:40]} rect={o['rect']} z={o['zIndex']} pe={o['pointerEvents']}")

        # ---- 4. 全局对象 ----
        print("6) 收集挑战 iframe 全局对象 ...")
        globs = challenge.evaluate(PROBE_GLOBALS)
        interesting = [g for g in globs if not g["key"].startswith(("__", "Object", "Array", "Math", "JSON", "Reflect", "Promise"))]
        report["globals_sample"] = interesting[:60]
        print(f"   可疑全局: {[g['key'] for g in interesting][:25]}")

        # ---- 5. 真实点击，观察状态变化 ----
        print("7) 状态基线 ...")
        state_before = challenge.evaluate(PROBE_STATE)
        report["state_before_click"] = state_before
        print(f"   选中语义元素: {len(state_before['selected'])}, 标记属性: {len(state_before['marked'])}")

        iframe_box = page.query_selector(CHALLENGE_IFRAME_SELECTOR).bounding_box()
        clicks = []
        # 在 canvas 中部附近点两下，观察选中机制（不知道正确答案，只为看机制）
        for frac in ((0.30, 0.45), (0.70, 0.72)):
            cx = iframe_box["x"] + geo["rect"]["x"] + geo["rect"]["w"] * frac[0]
            cy = iframe_box["y"] + geo["rect"]["y"] + geo["rect"]["h"] * frac[1]
            print(f"   点击 ({frac[0]:.2f}, {frac[1]:.2f}) -> 页面坐标 ({cx:.0f}, {cy:.0f})")
            page.mouse.click(cx, cy)
            time.sleep(1.5)
            st = challenge.evaluate(PROBE_STATE)
            clicks.append({"frac": frac, "abs": [round(cx), round(cy)], "state": st})
        report["clicks"] = clicks
        page.screenshot(path=os.path.join(OUT, "screen_after_clicks.png"), full_page=True)

        after = clicks[-1]["state"]
        print(f"   点击后选中语义元素: {len(after['selected'])}, 标记属性: {len(after['marked'])}")
        if after["marked"]:
            print(f"   标记属性示例: {json.dumps(after['marked'][:3], ensure_ascii=False)}")

        # 点选后再抓一次 canvas，对比是否出现选中标记
        png2 = challenge.evaluate(PROBE_CANVAS_PNG)
        if png2.get("ok"):
            save_data_url(png2["data"], os.path.join(OUT, "canvas_after_clicks.png"))
            print("   已保存 canvas_after_clicks.png")

    with open(os.path.join(OUT, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 报告已写入 {OUT}/report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
