"""
拖拽题「可拖拽方块」检测器（离线开发 + 自测）

背景：拖拽题布局是左侧面板放可拖拽方块（各带 Move 手柄）、右侧面板是白色虚线
轮廓。要 live 验证拖拽，必须把起手点落在方块上 —— 上次硬编码画面正中导致
0 像素变化（那里是右侧面板空白）。方块位置每道题都不同，所以要检测而不是写死。

特征（据 research_out/drag_sweep/C_headless_altsite/before.png 观察）：
    - 方块是米黄/卡其色块，R 明显大于 B
    - 左侧面板背景是浅灰白、右侧面板是彩色，方块只出现在左侧
    - 方块面积较大且成片

离线自测:
    python research/detect_drag_piece.py research_out/drag_sweep/C_headless_altsite/before.png
产物:
    research_out/drag_piece/annotated.png
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
from PIL import Image

OUT = "research_out/drag_piece"


def detect_pieces(rgb: np.ndarray, debug: bool = False):
    """
    返回 [{'cx','cy','x','y','w','h','area'}...]，按面积降序。
    只在画面左侧 40% 内找（右侧面板是虚线轮廓，不该被当成方块）。
    """
    h, w = rgb.shape[:2]
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)

    # 米黄/卡其：偏暖，R > B 明显，整体偏亮
    mask = ((r > 150) & (g > 130) & (r - b > 30) & (r - g > 5) & (r - g < 70)).astype(np.uint8)
    # 只保留左侧面板
    mask[:, int(w * 0.40):] = 0

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)

    pieces = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 800:            # 太小的是噪声
            continue
        if bw < 30 or bh < 30:
            continue
        pieces.append({"cx": float(cents[i][0]), "cy": float(cents[i][1]),
                       "x": int(x), "y": int(y), "w": int(bw), "h": int(bh),
                       "area": int(area)})
    pieces.sort(key=lambda d: -d["area"])
    if debug:
        print(f"  检测到 {len(pieces)} 个候选方块")
        for p in pieces[:4]:
            print(f"    area={p['area']:6d} 中心=({p['cx']:.0f},{p['cy']:.0f}) "
                  f"bbox=({p['x']},{p['y']},{p['w']},{p['h']})")
    return pieces, mask


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else \
        "research_out/drag_sweep/C_headless_altsite/before.png"
    os.makedirs(OUT, exist_ok=True)
    arr = np.asarray(Image.open(src))
    rgb = arr[:, :, :3].copy()
    print(f"图像 {rgb.shape[1]}x{rgb.shape[0]}")

    pieces, mask = detect_pieces(rgb, debug=True)
    cv2.imwrite(os.path.join(OUT, "mask.png"), mask * 255)

    vis = rgb.copy()
    for i, p in enumerate(pieces):
        x, y, w, h = p["x"], p["y"], p["w"], p["h"]
        color = (255, 0, 255) if i == 0 else (0, 200, 0)
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 3)
        cv2.circle(vis, (int(p["cx"]), int(p["cy"])), 6, (255, 0, 0), -1)
        cv2.putText(vis, f"#{i} {p['area']}", (x, max(16, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    Image.fromarray(vis).save(os.path.join(OUT, "annotated.png"))
    print(f"标注图: {OUT}/annotated.png  (品红=面积最大即起手方块)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
