"""
hCaptcha 打码器
通过 sitekey + url 本地复现挑战并获取 token（使用路由拦截伪装原始 URL）

⚠️ 关于本题库能做到什么，请先读这里

实测结论（2026-09，两套 sitekey 共 18 次挑战）：hCaptcha 目前下发的挑战
**几乎全部是单 <canvas> 渲染的题型**，例如：

    Please click on the TWO icons that break the pattern
    Click on the TWO characters that do not follow the pattern
    Please drag the screw to the empty joint
    Drag the shapes into their outlines
    Click the broken spot in the chain

这些题型的 DOM 实测为 canvas=1 / img=0 / .task=0 —— 既没有可点击的图块元素，
也没有 <img> 子图可裁；题目考的是「规律推理」和「拖拽」，不是「逐格目标分类」。
所以下面的 YOLO / 零样本检测管线**解不了这些题**，这不是模型强弱的问题，
而是需要另一类能力（视觉推理 / 坐标级拖拽操作）。

因此本文件是分层实现：

  1. 平台流程骨架 —— 实测可用，可复用：
     widget 加载（含重试）、勾选框点击、挑战 iframe 接入、题面读取、
     token 提取、提交/刷新/报错处理。
  2. 照片网格题型 —— 代码路径完整（.task 图块 → 截图 → 检测 → 点击 → 提交），
     但当下的 hCaptcha 基本不下发该题型，因此**未做端到端验证**。
  3. canvas 题型 —— 明确判定为不支持：打印题面、触发 on_unsupported 回调、
     刷新重试，连续多次后带明确信息退出。接入自定义求解器只需传 on_unsupported。

请仅在你拥有或已获授权的环境（自建站点的 QA、自己的测试账号、
官方公开演示页）中使用。
"""

from camoufox.sync_api import Camoufox
from playwright.sync_api import Page
from PIL import Image
import io
import time
import os
import random
from typing import Optional, Set, Callable
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from vision import (
    DEBUG,
    ENABLE_IMAGE_PREPROCESSING,
    SCREENSHOT_DIR,
    GROUNDING_DINO_CONFIDENCE,
    GROUNDING_DINO_DEBUG,
    GROUNDING_PROMPTS,
    VisionModels,
    crop_image_from_bytes,
    ensure_dir,
    get_category_info,
    preprocess_image,
)

# --- hCaptcha 相关选择器（均经实测确认）---
HCAPTCHA_JS = os.environ.get("HCAPTCHA_JS", "https://js.hcaptcha.com/1/api.js")

# widget（勾选框）iframe：title 含 "checkbox"；挑战（弹层）iframe：title 固定
WIDGET_IFRAME_SELECTOR = "iframe[title*='checkbox' i]"
CHALLENGE_IFRAME_SELECTOR = "iframe[title='hCaptcha challenge']"

CHECKBOX_SELECTOR = "#checkbox"                  # widget 内的勾选框
PROMPT_SELECTOR_CANDIDATES = [".prompt-text", "#prompt-question"]
TASK_TILE_SELECTOR = ".task"                     # 照片网格题型的图块
SUBMIT_SELECTOR = ".button-submit"
REFRESH_SELECTOR = ".refresh.button"
ERROR_SELECTOR = ".display-error"

# 命中即判定为「当前检测管线不支持」的 canvas 题型关键词
UNSUPPORTED_CHALLENGE_KEYWORDS = [
    "break the pattern", "follow the pattern", "does not follow",
    "drag the", "into their outlines", "to the empty", "broken spot",
]

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>hCaptcha Solver</title>
    <script src="__HCAPTCHA_JS__" async defer></script>
    <style>
        body { font-family: Arial; display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; background: #f5f5f5; }
        .container { text-align: center; padding: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div id="h-captcha" class="h-captcha" data-sitekey="{{SITEKEY}}" data-callback="onSuccess"></div>
        <textarea id="h-captcha-response" name="h-captcha-response"></textarea>
        <textarea id="g-recaptcha-response" name="g-recaptcha-response"></textarea>
    </div>
    <script>
        var _token = "";
        function onSuccess(token) {
            _token = token;
            document.getElementById('h-captcha-response').value = token;
            document.getElementById('g-recaptcha-response').value = token;
        }
    </script>
</body>
</html>""".replace("__HCAPTCHA_JS__", HCAPTCHA_JS)


def is_unsupported_challenge(prompt_text: str) -> bool:
    """判断是否为当前检测管线无法求解的 canvas 题型"""
    low = (prompt_text or "").lower()
    return any(kw in low for kw in UNSUPPORTED_CHALLENGE_KEYWORDS)


class HCaptchaSolver:
    """hCaptcha 打码器"""

    def __init__(self, headless: bool = True, humanize: bool = False,
                 widget_retries: int = 6,
                 max_unsupported: int = 5,
                 on_unsupported: Optional[Callable[[str], None]] = None):
        """
        :param widget_retries: widget 渲染重试次数。
            实测 widget 约 50% 概率不渲染（iframe 已建立、title 正确，
            但 frame.url 为空且 #checkbox 不存在，也不会请求 newassets.hcaptcha.com），
            必须整页重载重试。
        :param max_unsupported: 连续遇到多少次不支持的题型后放弃
        :param on_unsupported: 遇到 canvas 题型时的回调，入参为题面文本，
            便于接入自定义求解器
        """
        self.headless = headless
        self.humanize = humanize
        self.widget_retries = widget_retries
        self.max_unsupported = max_unsupported
        self.on_unsupported = on_unsupported
        self.browser = None
        VisionModels.load()
        ensure_dir(SCREENSHOT_DIR)

    def __enter__(self):
        self.browser = Camoufox(
            headless=self.headless,
            humanize=self.humanize,
            i_know_what_im_doing=True,
            config={'forceScopeAccess': True},
            disable_coop=True,
        ).__enter__()
        return self

    def __exit__(self, *args):
        if self.browser:
            self.browser.__exit__(*args)

    # ------------------------------------------------------------------
    # 对外入口
    # ------------------------------------------------------------------
    def solve(self, sitekey: str, url: str, timeout: int = 120) -> str:
        """
        解决 hCaptcha 挑战

        :param sitekey: hCaptcha sitekey（形如 xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx）
        :param url: 原始页面 URL，决定 origin/host（用于伪装）
        :param timeout: 超时时间（秒）
        :return: token 字符串
        :raises Exception: 获取失败
        """
        page = self.browser.new_page()
        html_content = HTML_TEMPLATE.replace('{{SITEKEY}}', sitekey)

        def handle_route(route):
            route.fulfill(status=200, content_type='text/html', body=html_content)

        page.route(url, handle_route)
        if url.endswith('/'):
            page.route(url.rstrip('/'), handle_route)
        else:
            page.route(url + '/', handle_route)

        try:
            print(f"🌐 正在打开页面: {url}")
            widget = self._wait_for_widget(page, url)
            print("🖱️ 点击勾选框...")
            widget.evaluate(
                f"() => {{ const c = document.querySelector('{CHECKBOX_SELECTOR}'); if (c) c.click(); }}"
            )
            time.sleep(1)
            return self._solve_challenge(page, timeout)
        finally:
            page.close()

    # ------------------------------------------------------------------
    # widget 加载
    # ------------------------------------------------------------------
    def _find_widget(self, page):
        """定位 widget iframe 的 content_frame；未就绪返回 None"""
        el = page.query_selector(WIDGET_IFRAME_SELECTOR)
        if not el:
            return None
        cf = el.content_frame()
        if not cf:
            return None
        try:
            if cf.evaluate(f"() => !!document.querySelector('{CHECKBOX_SELECTOR}')"):
                return cf
        except Exception:
            pass
        return None

    def _wait_for_widget(self, page, url: str, tries: int = 20):
        """
        等待 widget 就绪，失败则整页重载。

        注意：hCaptcha 存在「iframe 已建立、title 正确，但内部为空」的状态，
        此时 frame.url 是空字符串、#checkbox 不存在 —— 必须重载页面，
        单纯等待不会恢复。
        """
        for attempt in range(1, self.widget_retries + 1):
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
            for _ in range(tries):
                time.sleep(1.5)
                cf = self._find_widget(page)
                if cf:
                    print(f"   ✅ widget 已就绪 (第 {attempt} 次加载)")
                    return cf
            print(f"   ⚠️ widget 未渲染，重试 ({attempt}/{self.widget_retries})")
            time.sleep(3)
        raise Exception("hCaptcha widget 始终未渲染（可能被限流或指纹被识别）")

    # ------------------------------------------------------------------
    # 挑战循环
    # ------------------------------------------------------------------
    def _get_challenge_frame(self, page, tries: int = 16):
        """等待挑战 iframe 出现并带有题面文本"""
        for _ in range(tries):
            time.sleep(1.5)
            el = page.query_selector(CHALLENGE_IFRAME_SELECTOR)
            if el:
                cf = el.content_frame()
                if cf:
                    try:
                        if cf.evaluate(
                            "() => !!document.querySelector('.prompt-text') || "
                            "!!document.querySelector('#prompt-question')"
                        ):
                            return cf
                    except Exception:
                        pass
        return None

    def _read_prompt(self, frame) -> str:
        for sel in PROMPT_SELECTOR_CANDIDATES:
            try:
                el = frame.locator(sel).first
                if el.count() > 0:
                    txt = el.inner_text(timeout=2000)
                    if txt:
                        return txt.strip()
            except Exception:
                continue
        return ""

    def _get_token(self, page) -> str:
        """
        从回调写入的 textarea / 输入框取 token。

        注意：**主路径是读 DOM**（textarea / input）。最后那条读 `window._token`
        的兜底路径实际**永远不会生效** —— playwright 的 `page.evaluate` 跑在隔离
        世界，读不到页面里定义的 JS 全局变量（已由 tools/verify_token_detector.py
        实测确认）。保留它只是无害的历史残留，不要当作保险。

        本方法的可靠性已单独校验：构造"已解出"的页面能读到正确 token，
        未解出的页面能正确判失败（见 tools/verify_token_detector.py）——
        这一条很关键，因为本仓库所有"成功率 0%"都出自这个判据。
        """
        try:
            for sel in ("textarea[name='h-captcha-response']",
                        "input[name='h-captcha-response']",
                        "textarea[name='g-recaptcha-response']",
                        "input[name='g-recaptcha-response']"):
                el = page.query_selector(sel)
                if not el:
                    continue
                tag = el.evaluate("e => e.tagName")
                val = el.input_value() if tag == "TEXTAREA" else el.get_attribute("value")
                if val and len(val) > 10:
                    return val
            # 无效兜底（隔离世界读不到页面全局），保留仅为兼容，勿依赖
            val = page.evaluate("() => window._token || ''")
            if val and len(val) > 10:
                return val
        except Exception:
            pass
        raise Exception("failed to get token")

    def _click_refresh(self, frame):
        try:
            frame.evaluate(
                f"() => {{ const r = document.querySelector('{REFRESH_SELECTOR}'); if (r) r.click(); }}"
            )
        except Exception:
            pass

    def _solve_challenge(self, page: Page, timeout: int) -> str:
        start_time = time.time()
        max_rounds = 30
        current_round = 0
        clicked_history: Set[int] = set()
        last_category = None
        unsupported_seen = 0

        while current_round < max_rounds and time.time() - start_time < timeout:
            current_round += 1
            print(f"\n🔄 --- 第 {current_round} 次循环检测 ---")

            # 1) token 是否已经出来
            try:
                token = self._get_token(page)
                print("✅ 验证成功！(token 已写入)")
                return token
            except Exception:
                pass

            # 2) 取挑战框；消失也可能意味着已通过
            challenge = self._get_challenge_frame(page)
            if not challenge:
                try:
                    return self._get_token(page)
                except Exception:
                    print("❓ 挑战窗口未找到，等待...")
                    time.sleep(1)
                    continue

            prompt = self._read_prompt(challenge)
            if not prompt:
                print("❓ 题面为空，等待...")
                time.sleep(1)
                continue
            print(f"📝 题目要求: {prompt}")

            # 3) canvas 题型：当前管线不支持
            if is_unsupported_challenge(prompt):
                unsupported_seen += 1
                print("   🚫 不支持的题型（canvas 渲染，非照片网格）")
                if self.on_unsupported:
                    try:
                        self.on_unsupported(prompt)
                    except Exception as e:
                        print(f"   ⚠️ on_unsupported 回调异常: {e}")
                if unsupported_seen >= self.max_unsupported:
                    raise Exception(
                        f"连续 {unsupported_seen} 次都是不支持的 canvas 题型，放弃。"
                        f"最后一题: {prompt!r}"
                    )
                self._click_refresh(challenge)
                time.sleep(3)
                continue

            # 4) 照片网格题型：读取图块
            tile_count = challenge.evaluate(
                f"() => document.querySelectorAll('{TASK_TILE_SELECTOR}').length")
            if not tile_count:
                print("   ⚠️ 未找到可点击图块，刷新")
                self._click_refresh(challenge)
                time.sleep(3)
                continue

            grid_side = 4 if tile_count == 16 else 3
            print(f"   📋 照片网格题型 ({grid_side}x{grid_side}, {tile_count} 块)")

            target_classes, use_zero_shot, category_name = get_category_info(prompt.lower())
            if not target_classes:
                print("   ⚠️ 题面类别无法映射，刷新")
                self._click_refresh(challenge)
                time.sleep(3)
                continue
            if category_name != last_category:
                clicked_history.clear()
                last_category = category_name

            # 5) 截图网格区域
            grid_box = challenge.evaluate(
                """() => {
                    const t = document.querySelector('.task');
                    if (!t) return null;
                    const p = t.parentElement.getBoundingClientRect();
                    return {x: p.x, y: p.y, width: p.width, height: p.height};
                }"""
            )
            if not grid_box:
                print("   ⚠️ 无法获取网格位置，刷新")
                self._click_refresh(challenge)
                time.sleep(3)
                continue

            iframe_box = page.query_selector(CHALLENGE_IFRAME_SELECTOR).bounding_box()
            if not iframe_box:
                print("   ⚠️ 无法获取挑战框位置，刷新")
                self._click_refresh(challenge)
                time.sleep(3)
                continue

            dpr = page.evaluate("window.devicePixelRatio")
            x1 = int((iframe_box['x'] + grid_box['x']) * dpr)
            y1 = int((iframe_box['y'] + grid_box['y']) * dpr)
            x2 = int((iframe_box['x'] + grid_box['x'] + grid_box['width']) * dpr)
            y2 = int((iframe_box['y'] + grid_box['y'] + grid_box['height']) * dpr)
            print(f"📐 坐标: dpr={dpr}, box=({x1},{y1},{x2},{y2})")

            time.sleep(1)
            full_screenshot = page.screenshot()
            if DEBUG:
                path = os.path.join(SCREENSHOT_DIR, f"hc_full_{current_round}.png")
                with open(path, "wb") as f:
                    f.write(full_screenshot)

            image_cp = crop_image_from_bytes(full_screenshot, (x1, y1, x2, y2))
            if not image_cp:
                print("   ⚠️ 裁剪失败")
                continue
            if DEBUG:
                path = os.path.join(SCREENSHOT_DIR, f"hc_crop_{current_round}.jpg")
                with open(path, "wb") as f:
                    f.write(image_cp)

            img_obj = Image.open(io.BytesIO(image_cp))
            if ENABLE_IMAGE_PREPROCESSING:
                try:
                    img_enhanced = preprocess_image(img_obj)
                    if DEBUG:
                        img_enhanced.save(
                            os.path.join(SCREENSHOT_DIR, f"hc_enhanced_{current_round}.jpg"), "JPEG")
                except Exception as e:
                    print(f"   ⚠️ 预处理失败: {e}")
                    img_enhanced = img_obj
            else:
                img_enhanced = img_obj

            img_w, img_h = img_obj.size
            tile_w = img_w / grid_side
            tile_h = img_h / grid_side

            # 6) 目标检测 -> 图块索引
            click_indices: Set[int] = set()

            def mark(box_xyxy, tag=""):
                bx1, by1, bx2, by2 = box_xyxy
                col = int(((bx1 + bx2) / 2) / tile_w)
                row = int(((by1 + by2) / 2) / tile_h)
                if 0 <= row < grid_side and 0 <= col < grid_side:
                    idx = row * grid_side + col
                    click_indices.add(idx)
                    if DEBUG:
                        print(f"      格子 [{row},{col}] {tag}")

            if use_zero_shot:
                grounding_prompt = None
                for keyword in GROUNDING_PROMPTS:
                    if keyword in prompt.lower() or (category_name and keyword in category_name.lower()):
                        grounding_prompt = GROUNDING_PROMPTS[keyword]
                        break
                if not grounding_prompt:
                    grounding_prompt = ". ".join(target_classes) + "."
                print(f"   🦖 零样本检测: {grounding_prompt}")
                for box_, score, _label in VisionModels.detect_zero_shot(img_obj, grounding_prompt,
                                                                        GROUNDING_DINO_CONFIDENCE):
                    mark(box_, f"置信度 {score:.3f} ✓")
            else:
                print(f"   🎯 YOLO 检测: {target_classes}")
                for cls_name, conf, box_ in VisionModels.detect_yolo(img_enhanced, target_classes):
                    mark(box_, f"{cls_name} conf={conf:.3f} ✓")

            sorted_indices = sorted(i for i in click_indices if i not in clicked_history)
            print(f"🎯 最终识别结果: 需点击网格 {sorted_indices}")

            # 7) 点击图块
            if sorted_indices:
                order = sorted_indices.copy()
                if len(order) > 2 and random.random() > 0.3:
                    random.shuffle(order)
                for idx in order:
                    try:
                        challenge.evaluate(
                            f"() => {{ const t = document.querySelectorAll('{TASK_TILE_SELECTOR}')[{idx}]; "
                            f"if (t) t.click(); }}"
                        )
                        clicked_history.add(idx)
                        time.sleep(0.15)
                    except Exception as e:
                        print(f"   ⚠️ 点击图块 {idx} 失败: {e}")
            else:
                print("   🤷 本轮未发现目标")

            # 8) 提交
            try:
                challenge.evaluate(
                    f"() => {{ const b = document.querySelector('{SUBMIT_SELECTOR}'); if (b) b.click(); }}"
                )
                print("🖱️ 点击提交按钮...")
                time.sleep(1.2)
                err = challenge.evaluate(
                    f"() => {{ const e = document.querySelector('{ERROR_SELECTOR}'); "
                    f"return e && e.offsetParent !== null ? e.innerText : null; }}"
                )
                if err:
                    print(f"   ❌ hCaptcha 提示: {err.strip()}")
            except Exception as e:
                print(f"⚠️ 提交异常: {e}")

        raise Exception("failed to solve hcaptcha")


def solve_hcaptcha(sitekey: str, url: str, headless: bool = True, timeout: int = 200,
                   on_unsupported: Optional[Callable[[str], None]] = None) -> str:
    """
    便捷函数：解决 hCaptcha 挑战

    :param sitekey: hCaptcha sitekey
    :param url: 原始页面 URL（用于伪装 origin/host）
    :param headless: 是否无头模式
    :param timeout: 超时时间（秒）
    :param on_unsupported: 遇到 canvas 题型时的回调，入参为题面文本
    :return: token 字符串

    示例:
        token = solve_hcaptcha(
            sitekey="a5f74b19-9e45-40e0-b45d-47ff91b7a6c2",
            url="https://accounts.hcaptcha.com/demo",
        )
    """
    with HCaptchaSolver(headless=headless, on_unsupported=on_unsupported) as solver:
        return solver.solve(sitekey, url, timeout)


# === 主程序测试：hCaptcha 官方演示页 ===
if __name__ == "__main__":
    test_sitekey = "a5f74b19-9e45-40e0-b45d-47ff91b7a6c2"
    test_url = "https://accounts.hcaptcha.com/demo"

    print(f"🔑 Sitekey: {test_sitekey}")
    print(f"🌐 URL: {test_url}")
    print("⏳ 正在解决 hCaptcha 挑战...")

    def _on_unsupported(prompt):
        print(f"   ℹ️ 可在 on_unsupported 中接入自定义求解器，题目: {prompt!r}")

    try:
        token = solve_hcaptcha(test_sitekey, test_url, headless=True,
                               on_unsupported=_on_unsupported)
        print(f"\n🎉 Token: {token}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
