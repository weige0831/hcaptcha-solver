"""
canvas 动作执行层

把后端给出的「坐标计划」变成真实的交互动作。与 canvas_backend.py 分工：
    canvas_backend   决定「点哪里 / 从哪拖到哪」（需要模型）
    canvas_actions   负责「怎么把这个动作做出来」（纯机械，不需要模型）

这样点击/拖拽这类机械部分可以脱离模型独立测试（用 MockBackend 喂坐标即可）。

坐标约定：传入的坐标一律是 **canvas 绘制缓冲坐标系**（例如 1000x940），
本模块负责换算到页面绝对坐标：
    页面坐标 = iframe 偏移 + canvas 在 iframe 内的偏移 + 缓冲坐标 / scale
其中 scale = 缓冲宽度 / CSS 宽度（实测 hCaptcha 是 2x 超采样）。
"""

from __future__ import annotations

import time
from typing import List, Optional, Sequence, Tuple

from playwright.sync_api import Page

# 拖拽参数：分段移动 + 逐步延时，模拟真实指针轨迹
DRAG_STEPS = 14
DRAG_STEP_DELAY = 0.025
DRAG_HOLD_MS = 120


class CanvasGeometry:
    """canvas 在页面中的几何关系，用于缓冲坐标 -> 页面坐标换算"""

    def __init__(self, iframe_box: dict, canvas_rect: dict, buf_size: Tuple[int, int]):
        self.ox = float(iframe_box["x"]) + float(canvas_rect["x"])
        self.oy = float(iframe_box["y"]) + float(canvas_rect["y"])
        self.scale = float(buf_size[0]) / float(canvas_rect["w"])

    def to_page(self, bx: float, by: float) -> Tuple[float, float]:
        """缓冲坐标 -> 页面绝对坐标"""
        return self.ox + bx / self.scale, self.oy + by / self.scale

    def to_buffer(self, px: float, py: float) -> Tuple[float, float]:
        """页面绝对坐标 -> 缓冲坐标"""
        return (px - self.ox) * self.scale, (py - self.oy) * self.scale


def click_at(page: Page, x: float, y: float, settle: float = 0.35) -> None:
    """在页面绝对坐标处点击（先移动再按下，避免部分实现忽略无移动的点击）"""
    page.mouse.move(x, y)
    page.mouse.click(x, y)
    time.sleep(settle)


def drag_and_drop(page: Page, x1: float, y1: float, x2: float, y2: float,
                  steps: int = DRAG_STEPS, hold_ms: int = DRAG_HOLD_MS) -> None:
    """
    在页面绝对坐标之间做一次拖拽。

    必须用完整的 pointer 事件序列（down -> 多次 move -> up）：
    canvas 上的拖拽通常由 pointermove 驱动，只做 click 或一次性跳变都不会被识别。
    """
    page.mouse.move(x1, y1)
    time.sleep(0.05)
    page.mouse.down()
    time.sleep(hold_ms / 1000.0)
    for i in range(1, steps + 1):
        t = i / float(steps)
        # ease-out，更接近人手
        e = 1 - (1 - t) ** 2
        page.mouse.move(x1 + (x2 - x1) * e, y1 + (y2 - y1) * e)
        time.sleep(DRAG_STEP_DELAY)
    time.sleep(0.08)
    page.mouse.up()
    time.sleep(0.25)


def execute_plan(page: Page, geom: CanvasGeometry, plan: dict,
                 drag_offset_y: float = 0.0) -> dict:
    """
    执行后端返回的计划。

    :param plan: {"clicks": [(bx,by), ...], "drag": [(bx1,by1,bx2,by2), ...]}
                 坐标是 canvas 缓冲坐标系；drag_offset_y 用于把「已裁掉顶部题面区」
                 的坐标补回完整 canvas 坐标系。
    :return: 实际执行的动作统计
    """
    done = {"clicks": 0, "drags": 0, "errors": []}

    for item in plan.get("clicks", []) or []:
        try:
            bx, by = float(item[0]), float(item[1]) + drag_offset_y
            px, py = geom.to_page(bx, by)
            click_at(page, px, py)
            done["clicks"] += 1
        except Exception as e:
            done["errors"].append(f"click {item}: {type(e).__name__}: {e}")

    for item in plan.get("drag", []) or []:
        try:
            bx1, by1, bx2, by2 = [float(v) for v in item[:4]]
            by1 += drag_offset_y
            by2 += drag_offset_y
            p1, p2 = geom.to_page(bx1, by1), geom.to_page(bx2, by2)
            drag_and_drop(page, p1[0], p1[1], p2[0], p2[1])
            done["drags"] += 1
        except Exception as e:
            done["errors"].append(f"drag {item}: {type(e).__name__}: {e}")

    return done
