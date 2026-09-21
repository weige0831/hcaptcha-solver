"""
拖拽动作的机制级验证（不依赖 hCaptcha）

背景：拖拽代码写好后一直没能live验证 —— 实测 28 道题（10 次评测 + 18 次刷新）
全是 pattern 题型，服务端一次都没下发拖拽题。题型是服务端随机决定的，
碰不到就只能等，不能靠"跑过一批评测"就宣称拖拽可用。

但拖拽里真正有技术风险的是**指针事件序列**：canvas 上的拖拽通常由 pointermove
驱动，只发 click、或者从起点直接跳到终点，都不会被识别。这部分可以脱离
hCaptcha 验证 —— 自己起一个 canvas，记录收到的每一个指针事件，然后检查
drag_and_drop() 产生的事件流是否符合预期。

检查项:
    1. 恰好 1 次 pointerdown、1 次 pointerup
    2. pointermove 次数 >= 10（是分段移动，不是瞬移）
    3. down 落在起点、up 落在终点（容差内）
    4. 中间点单调推进（没有跳变）

用法:
    python tools/verify_drag_mechanism.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camoufox.sync_api import Camoufox
from canvas_actions import drag_and_drop

OUT = os.path.join(os.environ.get("RESEARCH_OUT", "research_out"), "drag_mech")

PAGE = """<!DOCTYPE html><html><head><meta charset="UTF-8"><title>drag mech</title></head>
<body style="margin:0">
<canvas id="c" width="600" height="400"
        style="width:600px;height:400px;display:block;background:#eee"></canvas>
<pre id="log" style="display:none"></pre>
<script>
// 注意：playwright 的 page.evaluate 跑在隔离世界里，读不到页面里定义的 JS 全局变量
// （window.__ev 永远是 undefined），所以事件日志写进 DOM，由外部读 DOM 取回。
var ev = [];
var c = document.getElementById('c');
function flush() { document.getElementById('log').textContent = JSON.stringify(ev); }
function rec(type, e) {
    var r = c.getBoundingClientRect();
    ev.push({type: type,
             x: Math.round((e.clientX - r.left) * 10) / 10,
             y: Math.round((e.clientY - r.top) * 10) / 10,
             buttons: e.buttons, pointerId: e.pointerId});
    flush();
}
['pointerdown','pointermove','pointerup'].forEach(function(t) {
    c.addEventListener(t, function(e) { rec(t, e); });
});
flush();
</script></body></html>"""


def main():
    os.makedirs(OUT, exist_ok=True)
    checks = {}

    with Camoufox(headless=True, humanize=False, i_know_what_im_doing=True) as browser:
        page = browser.new_page()
        # 用路由拦截在一个真实 URL 上提供服务。
        # 注意：不要用 page.set_content() —— 实测它只写入 DOM 而不执行内联脚本
        # （页面停在 about:blank，window.__ev 一直是 undefined）。
        TEST_URL = "https://drag-test.local/canvas"
        page.route(TEST_URL, lambda r: r.fulfill(status=200, content_type="text/html", body=PAGE))
        page.goto(TEST_URL, wait_until="load", timeout=30000)
        page.wait_for_selector("#c")
        if not page.evaluate("() => !!document.querySelector('#log')"):
            print("❌ 测试页未就绪，无法验证")
            return 1
        box = page.query_selector("#c").bounding_box()
        print(f"canvas 位置: x={box['x']:.0f} y={box['y']:.0f} "
              f"{box['width']:.0f}x{box['height']:.0f}")

        x1, y1 = box["x"] + 100, box["y"] + 120
        x2, y2 = box["x"] + 460, box["y"] + 300
        print(f"拖拽: ({x1:.0f},{y1:.0f}) -> ({x2:.0f},{y2:.0f})")
        drag_and_drop(page, x1, y1, x2, y2)

        ev = json.loads(page.evaluate("() => document.querySelector('#log').textContent") or "[]")
        with open(os.path.join(OUT, "events.json"), "w", encoding="utf-8") as f:
            json.dump(ev, f, ensure_ascii=False, indent=2)

        downs = [e for e in ev if e["type"] == "pointerdown"]
        moves = [e for e in ev if e["type"] == "pointermove"]
        ups = [e for e in ev if e["type"] == "pointerup"]
        print(f"\n收到事件: down={len(downs)} move={len(moves)} up={len(ups)}")

        checks["one_down"] = len(downs) == 1
        checks["one_up"] = len(ups) == 1
        checks["enough_moves"] = len(moves) >= 10

        lx1, ly1 = x1 - box["x"], y1 - box["y"]
        lx2, ly2 = x2 - box["x"], y2 - box["y"]
        if downs and ups:
            d0, u0 = downs[0], ups[-1]
            checks["down_at_start"] = abs(d0["x"] - lx1) < 3 and abs(d0["y"] - ly1) < 3
            checks["up_at_end"] = abs(u0["x"] - lx2) < 3 and abs(u0["y"] - ly2) < 3
            print(f"  down 落点 ({d0['x']:.0f},{d0['y']:.0f})  期望 ({lx1:.0f},{ly1:.0f})")
            print(f"  up   落点 ({u0['x']:.0f},{u0['y']:.0f})  期望 ({lx2:.0f},{ly2:.0f})")
        else:
            checks["down_at_start"] = checks["up_at_end"] = False

        # 按住期间 buttons 位应为 1。
        # 注意：mouse.down() 之前还有一次定位用的 pointermove，它 buttons=0 是正常的
        # （那是悬停，不属于拖拽），所以只检查 down 之后、up 之前的 move。
        i_down = next((i for i, e in enumerate(ev) if e["type"] == "pointerdown"), None)
        i_up = next((i for i, e in enumerate(ev) if e["type"] == "pointerup"), None)
        if i_down is not None and i_up is not None:
            held_moves = [e for e in ev[i_down:i_up + 1] if e["type"] == "pointermove"]
            checks["button_held"] = bool(held_moves) and all(
                (e.get("buttons") or 0) & 1 for e in held_moves)
            print(f"  拖拽期间 move 数: {len(held_moves)}  全部 buttons=1: "
                  f"{checks['button_held']}")
        else:
            checks["button_held"] = False

        # 中间点单调推进
        if len(moves) >= 2:
            xs = [e["x"] for e in moves]
            ys = [e["y"] for e in moves]
            checks["monotonic_x"] = all(b >= a - 1 for a, b in zip(xs, xs[1:]))
            checks["monotonic_y"] = all(b >= a - 1 for a, b in zip(ys, ys[1:]))
            print(f"  move 路径 x: {xs[0]:.0f} -> {xs[-1]:.0f} ({len(xs)} 步)")
        else:
            checks["monotonic_x"] = checks["monotonic_y"] = False

    print("\n=== 检查结果 ===")
    for k, v in checks.items():
        print(f"  {'✅' if v else '❌'} {k}")
    ok = all(checks.values())
    print(f"\n{'✅ 拖拽事件序列正确，机制可用' if ok else '❌ 拖拽机制有问题，需修'}")
    with open(os.path.join(OUT, "checks.json"), "w", encoding="utf-8") as f:
        json.dump(checks, f, ensure_ascii=False, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
