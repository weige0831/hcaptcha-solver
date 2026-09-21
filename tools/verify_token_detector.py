"""
校验「成功检测器」本身是否可靠

动机（这是个测量工具的有效性问题）
--------------------------------
本仓库所有"成功率 0%"的结论都来自一个判据：`HCaptchaSolver._get_token()`
能从页面里读到 token。但我**只在"没有 token"的情况下跑过它** ——
如果它本身写错了（字段名不对、选择器过时、隔离世界读不到），
那么即使某次真的解出来了，也会被记成失败，**我的 0% 就是错的**。

所以这里反过来测：构造一个"已经解出"的页面（把 token 写进
`h-captcha-response` / `g-recaptcha-response` 文本框，模拟 hCaptcha 成功回调），
然后检查 `_get_token()` 能否读到。

同时顺带确认一个已知限制：`page.evaluate` 跑在隔离世界，
**读不到页面定义的 JS 全局变量** —— 所以 `_get_token` 里那条
`window._token` 兜底路径是无效的。这里一并实测，避免把无效路径当保险。

产物: 终端输出（✅/❌），无需 key、无需网络。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camoufox.sync_api import Camoufox

FAKE = "TOKEN_" + "a" * 60          # 长度 > 10，符合 _get_token 的最低长度要求

# 与 hcaptcha_solver.HTML_TEMPLATE 同构，但把 widget 换成一个"立刻成功"的桩
PAGE_SOLVED = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><title>solved stub</title></head>
<body>
<textarea id="h-captcha-response" name="h-captcha-response"></textarea>
<textarea id="g-recaptcha-response" name="g-recaptcha-response"></textarea>
<pre id="log" style="display:none"></pre>
<script>
// 模拟：hCaptcha 成功回调把 token 写进两个文本框（这是真实成功路径）
function onSuccess(token) {{
    document.getElementById('h-captcha-response').value = token;
    document.getElementById('g-recaptcha-response').value = token;
}}
onSuccess("{FAKE}");
// 同时写一个全局变量，用来验证"隔离世界读不到页面全局"这一限制
window._token = "{FAKE}";
document.getElementById('log').textContent = String((window._token || '').length);
</script>
</body></html>"""

# 对照组：未解出（token 为空）
PAGE_UNSOLVED = """<!DOCTYPE html><html><head><meta charset="UTF-8"></head>
<body>
<textarea id="h-captcha-response" name="h-captcha-response"></textarea>
<textarea id="g-recaptcha-response" name="g-recaptcha-response"></textarea>
</body></html>"""


def main():
    from hcaptcha_solver import HCaptchaSolver

    checks = {}
    solver = HCaptchaSolver(headless=True, widget_retries=1)
    solver.__enter__()
    try:
        # ---- 场景 A：页面处于"已解出"状态 ----
        page = solver.browser.new_page()
        page.route("https://nomock.local/solved",
                   lambda r: r.fulfill(status=200, content_type="text/html", body=PAGE_SOLVED))
        page.goto("https://nomock.local/solved", wait_until="load", timeout=30000)

        # 先确认页面里的全局变量确实写进去了（用 DOM 日志旁证，绕开隔离世界）
        dom_len = page.evaluate("() => document.querySelector('#log').textContent")
        print(f"  页面内 window._token 长度（经 DOM 旁证）: {dom_len}")
        checks["page_script_ran"] = dom_len.isdigit() and int(dom_len) > 10

        # 关键检查：_get_token 能否读到
        try:
            tok = solver._get_token(page)
            checks["detects_solved"] = (tok == FAKE)
            print(f"  场景A _get_token -> {'读到了正确 token' if tok == FAKE else '读到了但内容不符'}"
                  f" (len={len(tok)})")
        except Exception as e:
            checks["detects_solved"] = False
            print(f"  ❌ 场景A _get_token 抛异常: {type(e).__name__}: {e}")

        # 顺带实测：隔离世界能否读到页面 JS 全局（预期：读不到）
        try:
            seen = page.evaluate("() => window._token || ''")
            checks["isolated_world_hides_globals"] = (seen == "")
            print(f"  page.evaluate 读 window._token -> {seen[:20]!r} "
                  f"({'符合预期：隔离世界读不到' if seen == '' else '意外：竟然读到了'})")
        except Exception as e:
            print(f"  读全局异常: {e}")
        page.close()

        # ---- 场景 B：对照组，未解出（应判为失败）----
        page2 = solver.browser.new_page()
        page2.route("https://nomock.local/unsolved",
                    lambda r: r.fulfill(status=200, content_type="text/html", body=PAGE_UNSOLVED))
        page2.goto("https://nomock.local/unsolved", wait_until="load", timeout=30000)
        try:
            solver._get_token(page2)
            checks["rejects_unsolved"] = False
            print("  ❌ 场景B（未解出）竟然也读到了 token —— 判据过松")
        except Exception:
            checks["rejects_unsolved"] = True
            print("  场景B（未解出）正确判为失败 ✅")
        page2.close()
    finally:
        try:
            solver.__exit__(None, None, None)
        except Exception:
            pass

    print("\n=== 检查结果 ===")
    for k, v in checks.items():
        print(f"  {'✅' if v else '❌'} {k}")
    ok = checks.get("detects_solved") and checks.get("rejects_unsolved")
    print(f"\n{'✅ 成功检测器可靠：能读到真 token，也能拒绝空 token' if ok else '❌ 成功检测器有问题，此前 0% 的结论不可信'}")
    if checks.get("isolated_world_hides_globals"):
        print("ℹ️ 已知限制确认：_get_token 里 `window._token` 那条兜底路径无效")
        print("   （隔离世界读不到页面全局）。主路径是读文本框，那条才是有效的。")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
