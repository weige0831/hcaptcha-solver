"""
格状题型格子提取（第三版：由角色块质心反推格距，完全不碰亮度阈值）

前两版都靠"找白方块底"（亮度/饱和度阈值），三次都失败：
背景漏进来（包围盒变成整幅图）、或者阈值过严（包围盒只剩一个角色的一小块）。
阈值法在这里不成立，因为方块底与页面背景的亮度是重叠的。

这一版换几何思路，完全不用阈值找格子：

  1. 检角色块（高饱和 或 明显偏暗）—— 即使一个角色被拆成几块也没关系
  2. 把所有块的中心 x 聚成 4 簇、y 聚成 4 簇，得到 4 条列线与 4 条行线
  3. 列线 x 4 * 行线 x 4 => 16 个格心（笛卡尔积）
  4. 格边长取 `0.9 * min(列距, 行距)`，以格心为中心裁块

关键区别：用的是**块的位置**而不是**块的归属**，所以第 1 版那种
"一个角色拆成多块导致某格被覆盖"的问题不会出现 —— 碎片的位置仍在它所属的
行/列附近，只会轻微扰动簇心，不会造成整格错位。

用法:
    python research/grid_cells3.py research_out/instances/pat_01.png
产物:
    research_out/grid3/cells.png       16 格拼图（必须肉眼核验！）
    research_out/grid3/annotated.png   在原图上画出格心与裁块范围
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
from PIL import Image

OUT = "research_out/grid3"
CELL = 88

# 「真的是格状图吗」的闸门：格内白方块底占比的中位数下限（实测取 0.40）
GRID_BRIGHT_MIN = 0.40


def crop_puzzle(arr):
    rgb = arr[:, :, :3].copy()
    top = 0
    if arr.shape[2] == 4:
        rows = (arr[:, :, 3] > 10).sum(axis=1)
        nz = np.where(rows > 0)[0]
        top = int(nz.min()) if len(nz) else 0
    return rgb[top:, :].copy(), top


def find_blobs(rgb):
    """
    角色块。

    上一版用 `高饱和 OR 明显偏暗`，结果把页面背景的暗纹理线也检进来了（32 块），
    最外侧簇心被拽到画面边缘（x=112 / 898），导致格框整体偏移、切到空白或半个角色。

    这一版只认**强饱和**：角色都是鲜艳配色（绿/粉/青/紫/红），而背景纹理与
    灰阶线条都不满足。深色机器人那类可能漏检 1~2 个，但 16 格里够 14 个以上
    就足以定出 4 条行/列线。
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1].astype(np.int16)
    mask = (s > 90).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    blobs = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 2000 or w < 20 or h < 20 or w > 300 or h > 300:
            continue
        # 角色是实心块；细长的纹理碎片填充率低，滤掉
        if area / float(w * h) < 0.30:
            continue
        blobs.append({"cx": float(cents[i][0]), "cy": float(cents[i][1]),
                      "x": int(x), "y": int(y), "w": int(w), "h": int(h), "area": int(area)})
    return blobs


def kmeans1d(vals, k):
    lo, hi = float(vals.min()), float(vals.max())
    centers = np.linspace(lo, hi, k)
    for _ in range(100):
        idx = np.argmin(np.abs(vals[:, None] - centers[None, :]), axis=1)
        for j in range(k):
            m = idx == j
            if m.any():
                centers[j] = float(vals[m].mean())
    return np.sort(centers)


def build_cells(rgb, blobs, n=4, debug=True):
    if len(blobs) < n * 2:
        return None, None
    xs = np.array([b["cx"] for b in blobs], np.float32)
    ys = np.array([b["cy"] for b in blobs], np.float32)
    col_x = kmeans1d(xs, n)
    row_y = kmeans1d(ys, n)
    dx = np.median(np.diff(col_x)) if n > 1 else 0
    dy = np.median(np.diff(row_y)) if n > 1 else 0
    if dx <= 0 or dy <= 0:
        return None, None
    side = 0.90 * min(dx, dy)
    if debug:
        print(f"  列线 x: {[int(v) for v in col_x]}")
        print(f"  行线 y: {[int(v) for v in row_y]}")
        print(f"  列距 {int(dx)}  行距 {int(dy)}  取格边长 {int(side)}")

    cells = {}
    for r in range(n):
        for c in range(n):
            cx, cy = float(col_x[c]), float(row_y[r])
            x0, y0 = int(cx - side / 2), int(cy - side / 2)
            cells[(r, c)] = {"cx": cx, "cy": cy, "x": x0, "y": y0,
                             "w": int(side), "h": int(side)}

    # ---- 关键校验：这图像真的是「格状」吗？----
    # 踩过的坑：本函数会**强行**把 4x4 格阵套到任何图上有 >=8 个饱和块的图上。
    # 散点火箭图的背景（星云）本身就是高饱和块，于是被套出 16 个"格子"，
    # 判据再从这些空格子里挑 2 个点出去 —— 点的是空白背景，必然失败。
    # 实测判别依据：格状题每格是**白色方块底**（亮且低饱和占比高、暗占比极低），
    # 散点题是**深空背景**。实测值：
    #     pat_01 白底 0.579 / 暗 0.016      pat_04 白底 0.563 / 暗 0.026
    #     trial_01（实为散点）白底 0.201 / 暗 0.682
    # 故用「白底占比中位 > 0.40」作为闸门。
    bright = []
    for t in cells.values():
        p = cell_patch(rgb, t, size=48)
        if p is None:
            bright.append(0.0)
            continue
        hsv = cv2.cvtColor(p, cv2.COLOR_RGB2HSV)
        s, v = hsv[:, :, 1].astype(np.int32), hsv[:, :, 2].astype(np.int32)
        bright.append(float(((v > 200) & (s < 60)).mean()))
    med_bright = float(np.median(bright)) if bright else 0.0
    if debug:
        print(f"  白底占比中位 {med_bright:.3f} (格状应 >0.40)")
    if med_bright <= GRID_BRIGHT_MIN:
        if debug:
            print(f"  ✗ 判定为**非格状**（白底占比 {med_bright:.3f} <= "
                  f"{GRID_BRIGHT_MIN}），拒绝套格")
        return None, None
    return cells, (col_x, row_y, side)


def cell_patch(rgb, t, size=CELL):
    h, w = rgb.shape[:2]
    x0, y0 = max(0, t["x"]), max(0, t["y"])
    x1, y1 = min(w, t["x"] + t["w"]), min(h, t["y"] + t["h"])
    patch = rgb[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    ph, pw = patch.shape[:2]
    sc = (size * 0.94) / max(ph, pw)
    nw, nh = max(1, int(pw * sc)), max(1, int(ph * sc))
    rs = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.full((size, size, 3), 255, np.uint8)
    oy, ox = (size - nh) // 2, (size - nw) // 2
    out[oy:oy + nh, ox:ox + nw] = rs
    return out


def similarity(a, b):
    if a is None or b is None:
        return 0.0
    ha = cv2.calcHist([cv2.cvtColor(a, cv2.COLOR_RGB2HSV)], [0, 1], None, [24, 16], [0, 180, 0, 256])
    hb = cv2.calcHist([cv2.cvtColor(b, cv2.COLOR_RGB2HSV)], [0, 1], None, [24, 16], [0, 180, 0, 256])
    cv2.normalize(ha, ha); cv2.normalize(hb, hb)
    hist = max(float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL)), 0.0)
    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32).ravel()
    gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float32).ravel()
    ga -= ga.mean(); gb -= gb.mean()
    d = float(np.sqrt((ga * ga).sum()) * np.sqrt((gb * gb).sum()))
    corr = max(float((ga * gb).sum() / d), 0.0) if d > 1e-6 else 0.0
    return 0.55 * hist + 0.45 * corr


def analyze(path, verbose=True, save=True):
    arr = np.asarray(Image.open(path))
    rgb, top = crop_puzzle(arr)
    blobs = find_blobs(rgb)
    if verbose:
        print(f"  角色块 {len(blobs)} 个")
    cells, geom = build_cells(rgb, blobs, 4, debug=verbose)
    if not cells:
        return None
    patches = {k: cell_patch(rgb, t) for k, t in cells.items()}

    def vote(axis):
        out = {}
        for (r, c), p in patches.items():
            others = [patches[q] for q in patches
                      if (q[1] == c if axis == "col" else q[0] == r) and q != (r, c)]
            if others:
                out[(r, c)] = float(np.mean([similarity(p, o) for o in others]))
        return out

    sc, sr = vote("col"), vote("row")
    comb = {k: (sc[k] + sr[k]) / 2.0 for k in (set(sc) & set(sr))}
    ranked = sorted(comb.items(), key=lambda kv: kv[1])
    vals = np.array([v for _, v in ranked]) if ranked else np.array([0.0])

    if save:
        os.makedirs(OUT, exist_ok=True)
        vis = rgb.copy()
        picks = {k for k, _ in ranked[:2]}
        for (r, c), t in cells.items():
            out = (r, c) in picks
            cv2.rectangle(vis, (t["x"], t["y"]), (t["x"] + t["w"], t["y"] + t["h"]),
                          (255, 0, 255) if out else (0, 190, 0), 4 if out else 2)
            cv2.putText(vis, f"{comb.get((r,c),0):.2f}", (t["x"] + 5, t["y"] + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 0, 255) if out else (0, 170, 0), 2, cv2.LINE_AA)
        Image.fromarray(vis).save(os.path.join(OUT, "annotated.png"))

        sheet = np.full((4 * CELL, 4 * CELL, 3), 245, np.uint8)
        for i, k in enumerate(sorted(patches)):
            r, c = k
            p = patches[k]
            if p is not None and r < 4 and c < 4:
                sheet[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL] = p
        Image.fromarray(sheet).save(os.path.join(OUT, "cells.png"))

    return {"n": len(patches), "col": sc, "row": sr, "comb": comb,
            "ranked": ranked, "median": float(np.median(vals)),
            "mad": float(np.median(np.abs(vals - np.median(vals))))}


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "research_out/instances/pat_01.png"
    print(f"=== {src} ===")
    res = analyze(src)
    if not res:
        print("格提取失败")
        return 1
    print(f"\n  格子数 {res['n']}  组合相似度中位 {res['median']:.3f}  MAD {res['mad']:.4f}")
    print("  最异常的 6 格:")
    for (r, c), v in res["ranked"][:6]:
        print(f"    格({r},{c})  列={res['col'][(r,c)]:.3f}  行={res['row'][(r,c)]:.3f}  均={v:.3f}")
    if len(res["ranked"]) >= 4:
        v = [x[1] for x in res["ranked"]]
        print(f"\n  断层 2-3={v[2]-v[1]:.3f}  3-4={v[3]-v[2]:.3f}  跨度={v[-1]-v[0]:.3f}")
    print(f"\n格子拼图(必须核验): {OUT}/cells.png    标注: {OUT}/annotated.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
