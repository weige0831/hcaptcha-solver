"""
实验：干净分割 + 重跑离群分析

上一轮的相似度矩阵实验结论是"找到的是分割噪声"。但那是**测量被混淆**，
不等于方法本身死了。混淆源已查明：

    图标描边 = 完全中性灰  RGB 均值 [234,235,234]，通道极差中位数 0，S≈0
    星云亮核 = 暖白        RGB 类似 [213,204,201]，色相 13，S>12

之前用 `V>=140 & S<=70`，太松，星云亮核全都漏进来了 -> 污染图标掩膜 ->
面积失真 -> 归一化尺度错 -> 与谁都匹配不好。所以低分的那批是"分割差的"，
不是"题目里的异常项"。

这次改用紧阈值（V>=200 且 S<=12），并额外做两件事：
  1. 报告连通域面积的**离散程度** —— 分割干净的话，面积应该高度一致
  2. 每个图标记录一个"污染度"指标，把分割质量从离群得分里剥离出来看

产物:
    research_out/clean/annotated.png    干净分割 + 离群标注
    research_out/clean/pairs.txt        相似度矩阵与排名
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from canvas_strategy import _norm_patch, _similarity

SRC = sys.argv[1] if len(sys.argv) > 1 else "research_out/click/before.png"
OUT = "research_out/clean"

V_MIN, S_MAX = 200, 12
AREA_MIN, AREA_MAX = 200, 4000
ROT_STEP = 4


def segment(rgb, alpha):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    v, s = hsv[:, :, 2], hsv[:, :, 1]
    m = ((v >= V_MIN) & (s <= S_MAX)).astype(np.uint8)
    if alpha is not None:
        m &= (alpha > 10).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
    icons = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (AREA_MIN <= area <= AREA_MAX):
            continue
        if w < 14 or h < 14 or w > 220 or h > 220:
            continue
        patch = (labels[y:y + h, x:x + w] == i).astype(np.uint8) * 255
        # 污染度：填充率异常（正常描边图标填充率低且稳定）
        fill_ratio = area / float(w * h)
        icons.append({"cx": float(cents[i][0]), "cy": float(cents[i][1]),
                      "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                      "area": int(area), "mask": patch, "fill_ratio": float(fill_ratio)})
    return icons, m


def silhouette(patch, size=112, target=2600):
    """闭运算填成实心轮廓，再按质心+sqrt(面积)归一化"""
    p = cv2.morphologyEx(patch, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    p = np.zeros_like(p)
    cv2.drawContours(p, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
    ys, xs = np.where(p > 0)
    if len(xs) == 0:
        return None
    cy, cx = ys.mean(), xs.mean()
    sc = float(np.sqrt(target / len(xs)))
    ys2 = (ys - cy) * sc + size / 2
    xs2 = (xs - cx) * sc + size / 2
    out = np.zeros((size, size), np.float32)
    k = (ys2 >= 0) & (ys2 < size) & (xs2 >= 0) & (xs2 < size)
    out[ys2[k].astype(np.int32), xs2[k].astype(np.int32)] = 1.0
    return cv2.GaussianBlur(out, (0, 0), 1.2)


def variants(tmpl, use_mirror):
    out = []
    srcs = [tmpl] if not use_mirror else [tmpl, cv2.flip(tmpl, 1)]
    for s in srcs:
        for a in range(0, 360, ROT_STEP):
            M = cv2.getRotationMatrix2D((tmpl.shape[0] / 2,) * 2, a, 1.0)
            out.append(cv2.GaussianBlur(cv2.warpAffine(s, M, tmpl.shape,
                                                       flags=cv2.INTER_LINEAR), (0, 0), 1.2))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    arr = np.asarray(Image.open(SRC))
    rgb = arr[:, :, :3].copy()
    alpha = arr[:, :, 3] if arr.shape[2] == 4 else None

    icons, mask = segment(rgb, alpha)
    print(f"紧阈值分割 (V>={V_MIN}, S<={S_MAX}) -> {len(icons)} 个图标")
    areas = sorted(ic["area"] for ic in icons)
    fr = sorted(ic["fill_ratio"] for ic in icons)
    print(f"  面积: 最小 {areas[0]} 中位 {areas[len(areas)//2]} 最大 {areas[-1]}")
    print(f"  面积离散度 (std/mean) = {np.std(areas)/np.mean(areas):.3f}  (越小说明分割越一致)")
    print(f"  填充率: 中位 {fr[len(fr)//2]:.3f}  范围 [{fr[0]:.3f}, {fr[-1]:.3f}]")
    print(f"  面积明细: {areas}")
    cv2.imwrite(os.path.join(OUT, "mask.png"), mask * 255)

    # 归一化模板
    tmpls = []
    for ic in icons:
        t = silhouette(ic["mask"])
        tmpls.append(t)
    ok = [i for i, t in enumerate(tmpls) if t is not None]
    print(f"  可用轮廓: {len(ok)}/{len(icons)}")

    results = {}
    for use_mirror, tag in ((False, "rot"), (True, "mirror")):
        n = len(icons)
        cache = {j: variants(tmpls[j], use_mirror) for j in ok}
        M = np.zeros((n, n), np.float32)
        for i in ok:
            for j in ok:
                M[i, j] = 1.0 if i == j else max(
                    _similarity(tmpls[i], v) for v in cache[j])
        M = (M + M.T) / 2
        med = np.array([np.median(np.delete(M[i], i)) if i in ok else 0.0 for i in range(n)])
        order = [int(i) for i in np.argsort(med) if i in ok]
        print(f"\n=== {tag} ===")
        for r, idx in enumerate(order[:8]):
            print(f"  {r+1:2d}低 icon#{idx:2d} med={med[idx]:.4f} "
                  f"area={icons[idx]['area']:4d} fill={icons[idx]['fill_ratio']:.3f}")
        # 断层：第2与第3之间、第3与第4之间
        g23 = med[order[2]] - med[order[1]]
        g34 = med[order[3]] - med[order[2]]
        print(f"  断层 2-3={g23:.4f}  3-4={g34:.4f}")
        results[tag] = (M, med, order)

    M, med, order = results["mirror"]
    # 检查候选是否只是分割差：看它们的 area 是否偏离中位
    med_area = np.median([ic["area"] for ic in icons])
    print(f"\n=== 候选是否只是分割异常？(面积中位 {med_area:.0f}) ===")
    for idx in order[:4]:
        dev = abs(icons[idx]["area"] - med_area) / med_area
        print(f"  icon#{idx:2d} area={icons[idx]['area']:4d} 偏离中位 {dev:.1%} "
              f"fill={icons[idx]['fill_ratio']:.3f} med={med[idx]:.4f}")

    vis = rgb.copy()
    picked = set(order[:2])
    for i, ic in enumerate(icons):
        x, y, w, h = ic["x"], ic["y"], ic["w"], ic["h"]
        if i in picked:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 255), 3)
            cv2.putText(vis, f"OUT {med[i]:.3f} a={ic['area']}", (x, max(14, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (60, 200, 60), 1)
            cv2.putText(vis, f"{med[i]:.3f}", (x, max(12, y - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 200, 60), 1, cv2.LINE_AA)
    Image.fromarray(vis).save(os.path.join(OUT, "annotated.png"))

    with open(os.path.join(OUT, "pairs.txt"), "w", encoding="utf-8") as f:
        for i in range(len(icons)):
            f.write(" ".join(f"{M[i,j]:.3f}" for j in range(len(icons))) + "\n")
    print(f"\n产物: {OUT}/annotated.png  {OUT}/mask.png  {OUT}/pairs.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
