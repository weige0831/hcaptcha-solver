"""
格状题型的格子提取（第二版：从"检白色方块"入手，而不是检角色）

上一版的问题：先检角色块（30 个）再往 16 格里塞，结果一个角色被拆成多个连通域，
取到的块可能只是角色的一部分（比如一条胳膊），相似度被污染。

这一版的思路反过来：**格子的白方块是干净、规则、边界清晰的**，直接从白方块入手
就能拿到精确的 16 个格子框，格内的角色必然完整落在框内。
  1. 掩膜 = 高亮度 + 低饱和度（白方块底）
  2. 连通域 -> 过滤出尺寸相近的方形块
  3. 按位置聚成 4x4
  4. 每格取方块内部（内缩一点避开边框），即一个完整角色
  5. 两两相似度 -> 按列/行多数表决找异常

用法:
    python research/grid_cells.py research_out/instances/pat_01.png
产物:
    research_out/grid2/annotated.png   （框出 16 格 + 标相似度）
    research_out/grid2/cells.png       （16 个格内角色拼图，便于肉眼核验）
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
from PIL import Image

OUT = "research_out/grid2"
CELL = 88


def crop_puzzle(arr):
    rgb = arr[:, :, :3].copy()
    top = 0
    if arr.shape[2] == 4:
        rows = (arr[:, :, 3] > 10).sum(axis=1)
        nz = np.where(rows > 0)[0]
        top = int(nz.min()) if len(nz) else 0
    return rgb[top:, :].copy(), top


# 实测得到的阈值：格子内的白方块底 V 中位约 245，而页面背景 V 中位 204~217。
# V>235 & S<40 时背景仅 0.02% 通过（几乎不漏），白方块底保留 13%。
# 注意不能只看 S：方块里装着彩色角色，S 很高，所以"低饱和"不能单独作判据。
TILE_V_MIN = 235
TILE_S_MAX = 40


def find_tiles(rgb, gray=None, debug=True):
    """
    返回 {(row,col): tile}，两条路径：
      A. 白方块底被灰边分开 -> 得到 ~16 个独立连通域，按位置聚成 4x4
      B. 白方块底连成一整块 -> 取其包围盒，再按内部灰边切成 4x4
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    mask = ((v > TILE_V_MIN) & (s < TILE_S_MAX)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 4000:
            continue
        ar = w / float(h) if h else 0
        if not (0.5 < ar < 2.0):
            continue
        comps.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h),
                      "area": int(area), "cx": float(cents[i][0]), "cy": float(cents[i][1])})

    if debug:
        print(f"  白方块连通域 {len(comps)} 个，面积 {sorted(c['area'] for c in comps)[:4]}...")

    # 路径 A：足够多且尺寸相近 -> 直接当格子
    if len(comps) >= 12:
        med = np.median([c["area"] for c in comps])
        keep = [c for c in comps if 0.35 * med < c["area"] < 2.6 * med]
        keys = [(c["cy"], c["cx"]) for c in keep]
        xs = np.array([c["cx"] for c in keep], np.float32)
        ys = np.array([c["cy"] for c in keep], np.float32)
        _, ix = cluster_1d(xs, 4)
        _, iy = cluster_1d(ys, 4)
        cells = {}
        for c, i, j in zip(keep, ix, iy):
            key = (int(j), int(i))
            if key not in cells or c["area"] > cells[key]["area"]:
                cells[key] = c
        if debug:
            print(f"  路径A(独立方块) -> {len(cells)}/16 格")
        if len(cells) >= 12:
            return cells

    # 路径 B：连成整块 -> 按灰边切
    if comps and gray is not None:
        big = max(comps, key=lambda c: c["area"])
        if debug:
            print(f"  路径B(整块切分) 包围盒 x={big['x']} y={big['y']} "
                  f"{big['w']}x{big['h']}")
        cells = split_grid(gray, (big["x"], big["y"], big["w"], big["h"]), 4, debug=debug)
        return cells
    return None


def _snap(vals_profile, approx, span):
    """把边界吸附到附近最暗的位置（灰边），搜索范围 ±span"""
    lo = max(0, int(approx - span))
    hi = min(len(vals_profile), int(approx + span))
    if hi <= lo:
        return int(approx)
    seg = vals_profile[lo:hi]
    return lo + int(np.argmin(seg))


def split_grid(gray, bbox, n=4, debug=True):
    """
    把一个连成整块的格子区域切成 n x n。
    做法：先按几何等分，再把每条内部边界吸附到该处最暗的行/列（即灰边）。
    """
    x, y, w, h = bbox[:4]
    sub = gray[y:y + h, x:x + w].astype(np.float32)
    col_prof = sub.mean(axis=0)
    row_prof = sub.mean(axis=1)

    xs = [0]
    for k in range(1, n):
        xs.append(_snap(col_prof, w * k / n, span=w * 0.06))
    xs.append(w)
    ys = [0]
    for k in range(1, n):
        ys.append(_snap(row_prof, h * k / n, span=h * 0.06))
    ys.append(h)

    if debug:
        print(f"  列边界(相对): {xs}")
        print(f"  行边界(相对): {ys}")

    tiles = {}
    for r in range(n):
        for c in range(n):
            cw = xs[c + 1] - xs[c]
            ch = ys[r + 1] - ys[r]
            if cw < 20 or ch < 20:
                continue
            tiles[(r, c)] = {"x": x + xs[c], "y": y + ys[r], "w": cw, "h": ch,
                             "area": cw * ch, "cx": x + xs[c] + cw / 2.0,
                             "cy": y + ys[r] + ch / 2.0}
    return tiles


def cluster_1d(vals, k):
    lo, hi = vals.min(), vals.max()
    centers = np.linspace(lo, hi, k)
    idx = None
    for _ in range(80):
        idx = np.argmin(np.abs(vals[:, None] - centers[None, :]), axis=1)
        for j in range(k):
            m = idx == j
            if m.any():
                centers[j] = vals[m].mean()
    return centers, idx


def build_cells(tiles, n=4, debug=True):
    """把方块聚成 n x n，返回 {(row,col): tile}"""
    if len(tiles) < n * n * 0.7:
        return None
    xs = np.array([t["cx"] for t in tiles], np.float32)
    ys = np.array([t["cy"] for t in tiles], np.float32)
    _, ix = cluster_1d(xs, n)
    _, iy = cluster_1d(ys, n)
    # 同一格若落入多个块，取面积最大的
    best = {}
    for t, i, j in zip(tiles, ix, iy):
        key = (int(j), int(i))
        if key not in best or t["area"] > best[key]["area"]:
            best[key] = t
    if debug:
        print(f"  网格占用 {len(best)}/{n*n}")
    return best


def cell_patch(rgb, t, size=CELL, inset=0.10):
    """取方块内部（内缩避开边框），即一个完整角色"""
    x, y, w, h = t["x"], t["y"], t["w"], t["h"]
    dx, dy = int(w * inset), int(h * inset)
    x0, y0 = x + dx, y + dy
    x1, y1 = x + w - dx, y + h - dy
    patch = rgb[max(0, y0):y1, max(0, x0):x1]
    if patch.size == 0:
        return None
    ph, pw = patch.shape[:2]
    sc = (size * 0.92) / max(ph, pw)
    nw, nh = max(1, int(pw * sc)), max(1, int(ph * sc))
    rs = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.full((size, size, 3), 255, np.uint8)
    oy, ox = (size - nh) // 2, (size - nw) // 2
    out[oy:oy + nh, ox:ox + nw] = rs
    return out


def similarity(a, b):
    """颜色直方图 + 灰度结构 的加权相似度"""
    if a is None or b is None:
        return 0.0
    ha = cv2.calcHist([cv2.cvtColor(a, cv2.COLOR_RGB2HSV)], [0, 1], None, [24, 16], [0, 180, 0, 256])
    hb = cv2.calcHist([cv2.cvtColor(b, cv2.COLOR_RGB2HSV)], [0, 1], None, [24, 16], [0, 180, 0, 256])
    cv2.normalize(ha, ha); cv2.normalize(hb, hb)
    hist = float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL))
    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32).ravel()
    gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float32).ravel()
    ga -= ga.mean(); gb -= gb.mean()
    d = float(np.sqrt((ga * ga).sum()) * np.sqrt((gb * gb).sum()))
    corr = float((ga * gb).sum() / d) if d > 1e-6 else 0.0
    return 0.55 * max(hist, 0) + 0.45 * max(corr, 0)


def analyze(path, tag, verbose=True):
    arr = np.asarray(Image.open(path))
    rgb, top = crop_puzzle(arr)
    gray = np.asarray(Image.fromarray(rgb).convert("L"))
    cells = find_tiles(rgb, gray, debug=verbose)
    if not cells or len(cells) < 12:
        if verbose:
            print(f"  格提取不足（{len(cells) if cells else 0} 格），放弃")
        return None
    patches = {k: cell_patch(rgb, t) for k, t in cells.items()}

    # 按列 / 按行 多数表决
    def vote(axis):
        out = {}
        for (r, c), p in patches.items():
            others = [patches[q] for q in patches
                      if (q[1] == c if axis == "col" else q[0] == r) and q != (r, c)]
            if not others:
                continue
            out[(r, c)] = float(np.mean([similarity(p, o) for o in others]))
        return out

    sc, sr = vote("col"), vote("row")
    # 两方向都要低才算候选：取两方向分数的均值排序（更稳健）
    common = set(sc) & set(sr)
    comb = {k: (sc[k] + sr[k]) / 2.0 for k in common}
    ranked = sorted(comb.items(), key=lambda kv: kv[1])
    vals = np.array([v for _, v in ranked])
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))

    return {"path": path, "tag": tag, "n_cells": len(patches),
            "col": sc, "row": sr, "comb": comb, "ranked": ranked,
            "median": med, "mad": mad, "rgb": rgb, "cells": cells,
            "patches": patches, "ranked_list": ranked}


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "research_out/instances/pat_01.png"
    os.makedirs(OUT, exist_ok=True)
    print(f"=== {src} ===")
    res = analyze(src, os.path.basename(src))
    if not res:
        print("格提取失败")
        return 1

    print(f"\n  格子数 {res['n_cells']}  组合相似度中位 {res['median']:.3f}  MAD {res['mad']:.4f}")
    print("  最异常的 6 格（列+行两方向平均，越低越异常）:")
    for (r, c), v in res["ranked_list"][:6]:
        print(f"    格({r},{c})  列={res['col'][(r,c)]:.3f}  行={res['row'][(r,c)]:.3f}  均={v:.3f}")

    # 是否出现明显断层
    if len(res["ranked_list"]) >= 4:
        v = [x[1] for x in res["ranked_list"]]
        g23 = v[2] - v[1]
        g34 = v[3] - v[2]
        print(f"\n  断层 2-3={g23:.3f}  3-4={g34:.3f}   整体跨度={v[-1]-v[0]:.3f}")
        print(f"  => {'第2与第3间有明显断层，候选较明确' if g23 > 0.05 and g23 > 3*max(g34,1e-6) else '无显著断层，区分度不足'}")

    # 标注
    vis = res["rgb"].copy()
    picks = {k for k, _ in res["ranked_list"][:2]}
    for (r, c), t in res["cells"].items():
        x, y, w, h = t["x"], t["y"], t["w"], t["h"]
        out = (r, c) in picks
        cv2.rectangle(vis, (x, y), (x + w, y + h),
                      (255, 0, 255) if out else (0, 190, 0), 4 if out else 1)
        cv2.putText(vis, f"{res['comb'].get((r,c),0):.2f}", (x + 4, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 0, 255) if out else (0, 170, 0), 2 if out else 1, cv2.LINE_AA)
    Image.fromarray(vis).save(os.path.join(OUT, "annotated.png"))

    # 16 格拼图，便于肉眼核验"是否每个格取到了完整角色"
    keys = sorted(res["patches"])
    sheet = np.full((4 * CELL, 4 * CELL, 3), 255, np.uint8)
    for i, k in enumerate(keys[:16]):
        r, c = divmod(i, 4)
        p = res["patches"][k]
        if p is not None:
            sheet[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL] = p
    Image.fromarray(sheet).save(os.path.join(OUT, "cells.png"))
    print(f"\n标注图: {OUT}/annotated.png   格内角色拼图: {OUT}/cells.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
