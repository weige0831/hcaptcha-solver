"""
诊断三：用「方块边框」做几何锚点，检验格框对齐假设

诊断二把列极差从 0.0665 降到 0.0588（仅 12%），说明背景色调只是成因之一。
残余最可能来自格框本身的列相关偏移：我用 k-means on 块质心定格，
角色在格内有偏移（有的居中、有的偏左），于是**不同列的裁块对齐程度不同**，
这会直接压低某些列的相似度，表现就是"偏离度随列变化"。

检验思路：方块底之间有**灰色细边框**，那是与内容无关的几何锚点。
先由质心格阵估出格子区域与格距，再在该区域内沿列/行投影"暗度"，
找到 3 条内部边框的位置，用它们作为精确格界 —— 不再依赖角色位置。

判据：若格框对齐假设成立，用边框对齐后**各列偏离极差应明显下降**。
若基本不变，则该假设也被排除，需要另找残余来源。

用法: python research/diagnose_bias3.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from PIL import Image

from research.grid_cells3 import (
    crop_puzzle, find_blobs, build_cells, cell_patch, similarity,
)

INST = "research_out/instances"
CELL = 88


def lattice_block(rgb, debug=False):
    """由块质心格阵估出格子区域与格距"""
    blobs = find_blobs(rgb)
    cells, _ = build_cells(rgb, blobs, 4, debug=False)
    if not cells or len(cells) < 16:
        return None
    xs = [cells[k]["cx"] for k in cells]
    ys = [cells[k]["cy"] for k in cells]
    dx = (max(xs) - min(xs)) / 3.0
    dy = (max(ys) - min(ys)) / 3.0
    if debug:
        print(f"    格阵: x {min(xs):.0f}..{max(xs):.0f} (列距{dx:.0f})  "
              f"y {min(ys):.0f}..{max(ys):.0f} (行距{dy:.0f})")
    return {"x0": min(xs), "x1": max(xs), "y0": min(ys), "y1": max(ys),
            "dx": dx, "dy": dy}


def snap_boundaries(gray, block, axis, debug=False):
    """
    在预计的 3 条内部边框位置附近，沿该方向投影"暗度"并吸附到最暗处。
    返回 5 个边界（含两端）的坐标。
    """
    x0, x1 = block["x0"], block["x1"]
    y0, y1 = block["y0"], block["y1"]
    d = block["dx"] if axis == "col" else block["dy"]

    # 取一条横跨整个格阵的窄带做投影，避开格子外的背景
    if axis == "col":
        yy0, yy1 = int(y0 - d * 0.3), int(y1 + d * 0.3)
        sub = gray[max(0, yy0):yy1, :].astype(np.float32)
        prof = sub.mean(axis=0)              # 每列的亮度
        centers = [x0 + i * d for i in range(4)]
    else:
        xx0, xx1 = int(x0 - d * 0.3), int(x1 + d * 0.3)
        sub = gray[:, max(0, xx0):xx1].astype(np.float32)
        prof = sub.mean(axis=1)
        centers = [y0 + i * d for i in range(4)]

    # 边界 = 相邻格心的中点；在其附近找最暗列/行（灰边）
    bounds = [centers[0] - d / 2]
    for i in range(3):
        approx = (centers[i] + centers[i + 1]) / 2.0
        span = d * 0.22
        lo, hi = int(max(0, approx - span)), int(min(len(prof), approx + span))
        if hi <= lo:
            bounds.append(approx)
            continue
        seg = prof[lo:hi]
        # 用平滑后的最暗位置，避免单像素噪声
        seg_s = np.convolve(seg, np.ones(5) / 5, mode="same")
        bounds.append(lo + int(np.argmin(seg_s)))
    bounds = np.array(bounds + [centers[3] + d / 2], dtype=np.float32)
    if debug:
        print(f"    {axis} 边界(吸附后): {[int(v) for v in bounds]}")
    return bounds


def cells_from_bounds(bounds_x, bounds_y):
    cells = {}
    for r in range(4):
        for c in range(4):
            x0, x1 = float(bounds_x[c]), float(bounds_x[c + 1])
            y0, y1 = float(bounds_y[r]), float(bounds_y[r + 1])
            if x1 - x0 < 20 or y1 - y0 < 20:
                continue
            cells[(r, c)] = {"x": int(x0), "y": int(y0),
                             "w": int(x1 - x0), "h": int(y1 - y0),
                             "cx": (x0 + x1) / 2, "cy": (y0 + y1) / 2}
    return cells


def dev_scores(rgb, cells):
    patches = {k: cell_patch(rgb, t, size=CELL) for k, t in cells.items()}
    keys = sorted(patches)
    S = {k: {} for k in keys}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            v = similarity(patches[a], patches[b])
            S[a][b] = v
            S[b][a] = v

    def line_stats(axis):
        within, dev = [], {}
        for idx in range(4):
            line = [k for k in keys if (k[1] if axis == "col" else k[0]) == idx]
            bc, bv = None, -1
            for k in line:
                m = float(np.mean([S[k][o] for o in line if o != k]))
                if m > bv:
                    bv, bc = m, k
            within.append(bv)
            for k in line:
                dev[k] = 0.0 if k == bc else 1.0 - S[k][bc]
        return float(np.mean(within)), dev

    cc, cd = line_stats("col")
    rc, rd = line_stats("row")
    return (cd if cc >= rc else rd)


def collect(mode, verbose_files=False):
    cms, rms, n = [], [], 0
    for f in sorted(os.listdir(INST)):
        if not f.endswith(".png"):
            continue
        arr = np.asarray(Image.open(os.path.join(INST, f)))
        rgb, top = crop_puzzle(arr)
        gray = np.asarray(Image.fromarray(rgb).convert("L"))
        block = lattice_block(rgb)
        if not block:
            continue
        if mode == "lattice":
            blobs = find_blobs(rgb)
            cells, _ = build_cells(rgb, blobs, 4, debug=False)
        else:  # border
            bx = snap_boundaries(gray, block, "col")
            by = snap_boundaries(gray, block, "row")
            cells = cells_from_bounds(bx, by)
        if not cells or len(cells) < 16:
            continue
        dev = dev_scores(rgb, cells)
        if dev is None or len(dev) < 16:
            continue
        cm = [np.mean([v for k, v in dev.items() if k[1] == c]) for c in range(4)]
        rm = [np.mean([v for k, v in dev.items() if k[0] == r]) for r in range(4)]
        cms.append(cm); rms.append(rm); n += 1
    if not n:
        return None
    return np.array(cms).mean(axis=0), np.array(rms).mean(axis=0), n


def main():
    print("=== 诊断三：格框对齐（质心定格 vs 边框吸附）===\n")
    for mode, tag in (("lattice", "质心定格（当前实现）"), ("border", "边框吸附（本轮）")):
        out = collect(mode)
        if not out:
            print(f"{tag}: 无实例"); continue
        pc, pr, n = out
        print(f"{tag}  (n={n})")
        print(f"  各列平均偏离 {[round(float(v),4) for v in pc]}   极差 {pc.max()-pc.min():.4f}")
        print(f"  各行平均偏离 {[round(float(v),4) for v in pr]}   极差 {pr.max()-pr.min():.4f}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
