"""
假设检验：异常项可能是「尺寸/形状差异」而不是「旋转差异」

为什么之前没测到：我做相似度匹配时按 sqrt(面积) 归一化，**这一步会把尺寸差异
主动抹掉**。如果题目考的是某个图标尺寸/形状不一样，那正好被我消掉了。
这是个方法学盲点。

两个独立线索都指向这个方向：
  1. 像素分析里顶部中央那枚 bbox 宽 139px，其余约 80px（近 2 倍）
  2. Moondream2 被问「哪个不一样」时独立答 "Top center"

关键量：**bbox 对角线 sqrt(w^2+h^2)**。对刚体形状来说它**与旋转无关**
（同一形状任意旋转，外接框对角线基本不变），所以它是个干净的「尺寸探针」：
如果所有图标是同尺寸同字形，对角线应该高度一致；跳出 2 个 => 它们尺寸不同。

同时对比：面积（描边像素数）、bpp 对角线、填充率。

用法:
    python research/size_outliers.py [canvas.png]
产物:
    research_out/size/annotated.png
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from canvas_strategy import detect_icons

OUT = "research_out/size"


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "research_out/click/before.png"
    os.makedirs(OUT, exist_ok=True)
    arr = np.asarray(Image.open(src))
    rgb = arr[:, :, :3].copy()
    alpha = arr[:, :, 3] if arr.shape[2] == 4 else None

    icons = detect_icons(rgb, alpha)
    print(f"检测到 {len(icons)} 个图标（沿用较松的分割，保证不漏检）")

    for ic in icons:
        ic["diag"] = float(np.hypot(ic["w"], ic["h"]))
        ic["fill"] = ic["area"] / float(ic["w"] * ic["h"])

    def report(key, fmt="{:.1f}"):
        vals = np.array([ic[key] for ic in icons], dtype=float)
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med))) or 1e-9
        # 稳健 z 分数：用 MAD 而不是标准差，避免离群点自己把尺度撑大
        rz = (vals - med) / (1.4826 * mad)
        order = np.argsort(-np.abs(rz))
        print(f"\n=== {key} ===")
        print(f"  中位 {fmt.format(med)}  区间 [{fmt.format(vals.min())}, {fmt.format(vals.max())}]")
        print(f"  离散度 (std/mean) = {vals.std()/vals.mean():.3f}")
        print("  稳健 z 最大的 5 个:")
        for i in order[:5]:
            print(f"    icon#{i:2d}  {key}={fmt.format(vals[i])}  z={rz[i]:+.2f}"
                  f"  ({'<== 离群' if abs(rz[i]) > 3.5 else ''})")
        return rz, order

    rz_diag, order_diag = report("diag")
    rz_area, order_area = report("area", "{:.0f}")
    report("fill", "{:.3f}")

    # 若对角线高度一致 -> 所有图标同尺寸，异常不是尺寸问题
    diag_vals = np.array([ic["diag"] for ic in icons])
    cv_diag = diag_vals.std() / diag_vals.mean()
    print(f"\n=== 判定 ===")
    print(f"  对角线离散度 {cv_diag:.3f}")
    if cv_diag < 0.10:
        print("  => 对角线高度一致：所有图标同尺寸，**异常不是尺寸差异**")
        print("     （之前 sqrt(面积) 归一化抹掉尺寸的担心可以排除）")
    else:
        print("  => 对角线差异较大，尺寸可能是异常维度，需看图确认是否为分割污染")

    top2 = list(order_diag[:2])
    print(f"  对角线 z 最大的两个: {top2}")
    for i in top2:
        print(f"    icon#{i}: bbox=({icons[i]['x']},{icons[i]['y']},"
              f"{icons[i]['w']},{icons[i]['h']}) 中心=({icons[i]['cx']:.0f},{icons[i]['cy']:.0f}) "
              f"diag={icons[i]['diag']:.0f} area={icons[i]['area']}")

    vis = rgb.copy()
    picked = set(int(i) for i in top2)
    for i, ic in enumerate(icons):
        x, y, w, h = ic["x"], ic["y"], ic["w"], ic["h"]
        out = i in picked
        cv2.rectangle(vis, (x, y), (x + w, y + h),
                      (255, 0, 255) if out else (0, 200, 0), 3 if out else 1)
        cv2.putText(vis, f"d={ic['diag']:.0f}", (x, max(14, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 255) if out else (0, 220, 0), 2 if out else 1, cv2.LINE_AA)
    Image.fromarray(vis).save(os.path.join(OUT, "annotated.png"))
    print(f"\n标注图: {OUT}/annotated.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
