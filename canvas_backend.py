"""
canvas 题型的「坐标生成」后端接口

设计意图：把「决定点哪里」这件事抽成可替换的后端，这样：
    - 现在没有 API key / GPU 时，管线其余部分（点击、提交、像素差分验证、
      评测）依然可以完整跑通并被测试（用 MockBackend）
    - 一旦有了多模态 LLM 的 key 或本地大模型，只要实现/选一个后端即可接入

证据表明可行的后端只有「大型多模态模型」这一类（见 README 最终结论）。
本地传统 CV 已被三轮实验证伪。

后端协议
--------
    solve(image: PIL.Image, prompt: str) -> dict
        返回 {"clicks": [(x, y), ...], "drag": [(x1,y1,x2,y2), ...]}
        坐标使用**图像像素坐标系**（与原图同尺寸），由调用方负责换算到页面坐标。

内置后端
--------
    MockBackend            固定返回，用于验证管线（不需要任何凭据）
    OpenAICompatBackend    任何 OpenAI 兼容的视觉 API（OpenAI / Gemini 兼容层 /
                           vLLM / Ollama 本地服务都走这个）
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from typing import List, Tuple

from PIL import Image

# 坐标统一用图像像素坐标系
Click = Tuple[float, float]
Drag = Tuple[float, float, float, float]


class CanvasBackend:
    """后端基类：子类实现 solve()"""

    name = "base"

    def solve(self, image: Image.Image, prompt: str) -> dict:
        raise NotImplementedError

    # -- 工具 --
    @staticmethod
    def _parse_json(text: str) -> dict:
        """从模型回复里抠出 JSON（容忍 markdown 代码块与前后废话）"""
        if not text:
            return {}
        m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
        if m:
            text = m.group(1)
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return {}
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}

    @staticmethod
    def _to_pil(image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image
        return Image.fromarray(image)


class MockBackend(CanvasBackend):
    """
    固定返回，用于端到端验证管线。

    默认给出整图中央附近的若干点 —— 只为验证「拿到坐标 -> 点击 -> 提交 ->
    像素差分确认」这条路是通的，不代表任何解题能力。
    """

    name = "mock"

    def __init__(self, n_points: int = 2):
        self.n_points = n_points

    def solve(self, image: Image.Image, prompt: str) -> dict:
        w, h = self._to_pil(image).size
        pts: List[Click] = []
        for i in range(self.n_points):
            frac = (i + 1) / (self.n_points + 1)
            pts.append((w * 0.5, h * (0.3 + 0.4 * frac)))
        return {"clicks": pts, "drag": []}


class OpenAICompatBackend(CanvasBackend):
    """
    任何 OpenAI 兼容的视觉接口。用环境变量配置，代码里不出现任何凭据。

        SOLVER_API_BASE   例如 https://api.openai.com/v1 或本地服务地址
        SOLVER_API_KEY    凭据（本地服务可留空）
        SOLVER_MODEL      模型名，例如 gpt-4o / gemini-... / qwen2-vl

    提示词把题目要求、坐标约定、输出格式一次讲清，并要求只输出 JSON。
    """

    name = "openai-compat"

    SYSTEM = (
        "You solve visual CAPTCHA puzzles. You are given the puzzle image and the "
        "instruction text. Respond with ONLY a JSON object, no prose.\n"
        "Coordinate system: origin is the TOP-LEFT of the image, x grows right, "
        "y grows down, units are pixels of the provided image.\n"
        'Schema: {"clicks": [[x, y], ...], "drag": [[x1, y1, x2, y2], ...], '
        '"confidence": 0..1, "reasoning": "one short sentence"}'
    )

    def __init__(self, base: str | None = None, key: str | None = None,
                 model: str | None = None, timeout: int = 60):
        self.base = (base or os.environ.get("SOLVER_API_BASE", "")).rstrip("/")
        self.key = key or os.environ.get("SOLVER_API_KEY", "")
        self.model = model or os.environ.get("SOLVER_MODEL", "")
        self.timeout = timeout
        if not self.base or not self.model:
            raise RuntimeError(
                "需要设置 SOLVER_API_BASE 与 SOLVER_MODEL（可用 SOLVER_API_KEY 提供凭据）")

    def _encode(self, image: Image.Image) -> str:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def solve(self, image: Image.Image, prompt: str) -> dict:
        import urllib.request

        img = self._to_pil(image)
        b64 = self._encode(img)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM},
                {"role": "user", "content": [
                    {"type": "text", "text":
                        f"Instruction shown to the user: {prompt!r}\n"
                        f"Image size: {img.size[0]}x{img.size[1]}.\n"
                        "Return the coordinates to click / drag."},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ]},
            ],
            "temperature": 0,
        }
        req = urllib.request.Request(
            f"{self.base}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {self.key}"} if self.key else {})},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            body = json.loads(r.read().decode())
        text = (body.get("choices") or [{}])[0].get("message", {}).get("content", "")
        data = self._parse_json(text)
        clicks = [tuple(map(float, p)) for p in data.get("clicks", []) if len(p) == 2]
        drag = [tuple(map(float, p)) for p in data.get("drag", []) if len(p) == 4]
        return {"clicks": clicks, "drag": drag,
                "confidence": data.get("confidence"),
                "reasoning": data.get("reasoning"), "raw": text[:2000]}


def get_backend(name: str | None = None) -> CanvasBackend:
    """按名字取后端：mock / openai（默认读环境变量，缺失则回退 mock 并告警）"""
    name = (name or os.environ.get("SOLVER_BACKEND", "")).lower()
    if name in ("openai", "openai-compat", "api"):
        return OpenAICompatBackend()
    if name == "mock" or not name:
        if not os.environ.get("SOLVER_API_BASE"):
            print("ℹ️ 未配置 SOLVER_API_BASE，使用 MockBackend（仅用于验证管线）")
        return MockBackend()
    raise ValueError(f"未知后端: {name}")
