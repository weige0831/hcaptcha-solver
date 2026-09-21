"""
格状「找出不符合规律的字符」题型分析（新发现的题型，值得单独处理）

发现过程：采集 8 道 pattern 题做推广性检验时，发现同一句题面
"Click on the TWO characters that do not follow the pattern" 对应**两种截然不同的布局**：

  散点型（icons/rockets）：一堆同字形图标以不同旋转角散布在场景里
      -> 这是我前面五种 CV 思路全部失败的那种，信号是"旋转"，太弱
  格状型（characters）：4x4 白色方块格子，每格一个**完全不同的卡通角色**
      -> 角色之间色彩+外形差异巨大，信号很强！

重要的是：格状型天然适合算法求解，因为
  1. 格子由白色方块分隔，边界清晰，不需要在复杂背景上分割
  2. 角色之间差异大（颜色、轮廓都不同），"是否同一角色"是个强信号
  3. 规律很可能是「每一列/行重复同一角色」，于是异常 = 偏离本列多数的那两格
     —— 这是按位置分组做多数表决，不需要理解"规律"本身

所以本脚本对格状型做题格提取 + 两两相似度 + 按列多数表决。
与前面失败的五种思路的关键区别：**信号强度完全不同量级**。

用法:
    python research/grid_characters.py research_out/instances/pat_01.png
产物:
    research_out/grid/annotated.png
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
from PIL import Image

OUT = "research_out/grid"
CELL = 96


def crop_puzzle(arr: np.ndarray):
    rgb = arr[:, :, :3].copy()
    top = 0
    if arr.shape[2] == 4:
        rows = (arr[:, :, 3] > 10).sum(axis=1)
        nz = np.where(rows > 0)[0]
        top = int(nz.min()) if len(nz) else 0
    return rgb[top:, :].copy(), top


def find_characters(rgb: np.ndarray):
    """
    找每个格子里的角色。角色是高饱和色块，格子底是近白，页面底是浅色纹理。
    用"饱和度 + 与白色的差异"取前景，再取足够大的连通域。
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    s, v = hsv[:, :, 1].astype(np.int16), hsv[:, :, 2].astype(np.int16)
    # 高饱和（彩色角色）或 明显暗于背景（深色角色，如蓝色机器人）
    mask = ((s > 70) | (v < 140)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    blobs = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 1500:
            continue
        if w < 30 or h < 30:
            continue
        if w > 260 or h > 260:
            continue
        blobs.append({"cx": float(cents[i][0]), "cy": float(cents[i][1]),
                      "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                      "area": int(area)})
    return blobs


def group_to_grid(blobs, n_side=4):
    """
    把角色按位置聚成 n_side x n_side 网格。
    做法：用 1D k-means 把 x 坐标分 4 组、y 坐标分 4 组。
    """
    if len(blobs) < n_side:
        return None
    xs = np.array([b["cx"] for b in blobs], np.float32)
    ys = np.array([b["cy"] for b in blobs], np.float32)

    def kmeans1d(vals, k):
        lo, hi = vals.min(), vals.max()
        centers = np.linspace(lo, hi, k)
        for _ in range(60):
            idx = np.argmin(np.abs(vals[:, None] - centers[None, :]), axis=1)
            for j in range(k):
                m = idx == j
                if m.any():
                    centers[j] = vals[m].mean()
        return centers, idx

    cx, ix = kmeans1d(xs, n_side)
    cy, iy = kmeans1d(ys, n_side)
    grid = {}
    for b, i, j in zip(blobs, ix, iy):
        grid[(int(j), int(i))] = b       # (row, col)
    return grid, np.sort(cx), np.sort(cy)


def cell_patch(rgb: np.ndarray, b: dict, size: int = CELL) -> np.ndarray:
    """按角色外接框取块并归一化到固定尺寸（保留长宽比）"""
    x, y, w, h = b["x"], b["y"], b["w"], b["h"]
    pad = int(max(w, h) * 0.12)
    x0, x1 = max(0, x - pad), min(rgb.shape[1], x + w + pad)
    y0, y1 = max(0, y - pad), min(rgb.shape[0], y + h + pad)
    patch = rgb[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    ph, pw = patch.shape[:2]
    sc = (size * 0.8) / max(ph, pw)
    nw, nh = max(1, int(pw * sc)), max(1, int(ph * sc))
    rs = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.full((size, size, 3), 255, np.uint8)
    oy, ox = (size - nh) // 2, (size - nw) // 2
    out[oy:oy + nh, ox:ox + nw] = rs
    return out


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """颜色直方图 + 灰度结构 的加权相似度（角色差异大，这个组合足够）"""
    if a is None or b is None:
        return 0.0
    # 颜色直方图（HSV 的 H/S）
    ha = cv2.calcHist([cv2.cvtColor(a, cv2.COLOR_RGB2HSV)], [0, 1], None, [24, 16], [0, 180, 0, 256])
    hb = cv2.calcHist([cv2.cvtColor(b, cv2.COLOR_RGB2HSV)], [0, 1], None, [24, 16], [0, 180, 0, 256])
    cv2.normalize(ha, ha); cv2.normalize(hb, hb)
    hist = float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL))
    # 灰度结构
    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32).ravel()
    gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float32).ravel()
    ga -= ga.mean(); gb -= gb.mean()
    d = float(np.sqrt((ga * ga).sum()) * np.sqrt((gb * gb).sum()))
    corr = float((ga * gb).sum() / d) if d > 1e-6 else 0.0
    return 0.55 * max(hist, 0) + 0.45 * max(corr, 0)


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "research_out/instances/pat_01.png"
    os.makedirs(OUT, exist_ok=True)
    arr = np.asarray(Image.open(src))
    rgb, top = crop_puzzle(arr)
    print(f"图像 {rgb.shape[1]}x{rgb.shape[0]} (裁顶部 {top})")

    blobs = find_characters(rgb)
    print(f"检测到 {len(blobs)} 个角色候选")
    if len(blobs) < 8:
        print("角色太少，可能不是格状型")
        return 1

    res = group_to_grid(blobs, 4)
    if res is None:
        print("无法组织成网格")
        return 1
    grid, cx, cy = res
    print(f"网格占用: {len(grid)}/16 格")
    rows = sorted(set(r for r, _ in grid))
    cols = sorted(set(c for _, c in grid))
    print(f"行 {rows} / 列 {cols}")

    # 每格取块
    patches = {k: cell_patch(rgb, b) for k, b in grid.items()}

    # 按列做多数表决：与本列其他格相似度高的是"正常"，最低的是异常
    scores = {}
    for (r, c), p in patches.items():
        others = [patches[(rr, cc)] for (rr, cc) in patches if cc == c and rr != r]
        if not others:
            continue
        sims = [similarity(p, o) for o in others]
        scores[(r, c)] = float(np.mean(sims))

    ranked = sorted(scores.items(), key=lambda kv: kv[1])
    print(f"\n=== 按列多数表决：与本列其他格的平均相似度（越低越异常）===")
    for (r, c), s in ranked[:8]:
        print(f"  格({r},{c})  相似度 {s:.3f}")

    # 换行方向再算一遍，交叉验证
    scores_r = {}
    for (r, c), p in patches.items():
        others = [patches[(rr, cc)] for (rr, cc) in patches if rr == r and cc != c]
        if not others:
            continue
        scores_r[(r, c)] = float(np.mean([similarity(p, o) for o in others]))
    ranked_r = sorted(scores_r.items(), key=lambda kv: kv[1])
    print(f"\n=== 按行多数表决（交叉验证）===")
    for (r, c), s in ranked_r[:8]:
        print(f"  格({r},{c})  相似度 {s:.3f}")

    col_top2 = [k for k, _ in ranked[:2]]
    row_top2 = [k for k, _ in ranked_r[:2]]
    print(f"\n列方向候选: {col_top2}")
    print(f"行方向候选: {row_top2}")
    print(f"两方向一致: {'✅ 一致' if set(col_top2) == set(row_top2) else '⚠️ 不一致'}")

    # 标注
    vis = rgb.copy()
    picks = set(col_top2)
    for (r, c), b in grid.items():
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        out = (r, c) in picks
        cv2.rectangle(vis, (x, y), (x + w, y + h),
                      (255, 0, 255) if out else (0, 190, 0), 4 if out else 1)
        cv2.putText(vis, f"{scores.get((r,c),0):.2f}", (x, max(16, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 0, 255) if out else (0, 200, 0), 2 if out else 1, cv2.LINE_AA)
    Image.fromarray(vis).save(os.path.join(OUT, "annotated.png"))
    print(f"\n标注图: {OUT}/annotated.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
