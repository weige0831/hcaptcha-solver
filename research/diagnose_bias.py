"""
诊断：偏离度排序是否系统性偏向某些行列（位置偏差假设）

由来：live 测试里 3 道格状题的候选取都落在第 1~2 列，从未碰过第 0 或第 3 列
（picks=[[1,2],[0,2]] / [[2,2],[1,2]] / [[1,2],[1,1]]）。
若异常位置在题间随机，不该如此聚集 —— 提示排序的可能是**位置**而不是**内容**。

好在这一点不用跑 live 就能查：本地有 8 道已存实例（research_out/instances/），
对其中能提出 16 格的逐题算偏离度，然后看两件事：

  1. 各列/各行偏离度的**均值**是否随位置系统变化（真正的偏差会表现为
     某一列整体偏高，而不是个别格偏高）
  2. top-4 候选在列/行上的分布是否显著偏离均匀（16 格里每列各 4 格，
     若随机，top-4 落在某列的概率不高）

如果确认有位置偏差，就说明失败原因不是"判据不够聪明"，而是**度量里混进了
与内容无关的固定成分**（背景渐变 / 光照 / 格框裁切偏移），这会直接解释 0/3。

用法:
    python research/diagnose_bias.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

from research.grid_cells3 import (
    crop_puzzle, find_blobs, build_cells, cell_patch, similarity,
)

INST = "research_out/instances"


def deviation_scores(rgb_full):
    """
    返回 {(r,c): dev} —— 与 research/grid_cells3 的 axis 判据同源：
    先看哪个方向自洽，再算与线上"主流角色"的不相似度
    """
    rgb, top = crop_puzzle(rgb_full)
    blobs = find_blobs(rgb)
    cells, _ = build_cells(rgb, blobs, 4, debug=False)
    if not cells or len(cells) < 16:
        return None
    patches = {k: cell_patch(rgb, t) for k, t in cells.items()}
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
            best_c, best_v = None, -1
            for k in line:
                m = float(np.mean([S[k][o] for o in line if o != k]))
                if m > best_v:
                    best_v, best_c = m, k
            within.append(best_v)
            for k in line:
                if k != best_c:
                    dev[k] = 1.0 - S[k][best_c]
            dev.setdefault(best_c, 0.0)      # 主流角色自身偏离记为 0
        return float(np.mean(within)), dev

    cc, cd = line_stats("col")
    rc, rd = line_stats("row")
    return (cd if cc >= rc else rd), ("col" if cc >= rc else "row")


def main():
    files = sorted(f for f in os.listdir(INST) if f.endswith(".png"))
    per_col, per_row = [], []
    top4_cols, top4_rows = [], []
    used = []

    for f in files:
        arr = np.asarray(Image.open(os.path.join(INST, f)))
        out = deviation_scores(arr)
        if not out:
            print(f"{f:12s} 非格状型（提不出 16 格）")
            continue
        dev, axis = out
        used.append(f)
        vals = np.array([dev[k] for k in sorted(dev)])
        colmean = [np.mean([dev[k] for k in dev if k[1] == c]) for c in range(4)]
        rowmean = [np.mean([dev[k] for k in dev if k[0] == r]) for r in range(4)]
        per_col.append(colmean)
        per_row.append(rowmean)
        ranked = sorted(dev.items(), key=lambda kv: -kv[1])
        top4 = ranked[:4]
        top4_cols += [k[1] for k, _ in top4]
        top4_rows += [k[0] for k, _ in top4]
        print(f"{f:12s} axis={axis} 列均偏离={[round(v,3) for v in colmean]} "
              f"top4列={[k[1] for k,_ in top4]}")

    if not used:
        print("\n没有可用实例")
        return 1

    n = len(used)
    print(f"\n=== 汇总（{n} 道格状实例）===")
    pc = np.array(per_col).mean(axis=0)
    pr = np.array(per_row).mean(axis=0)
    print(f"各列平均偏离: {[round(v,4) for v in pc]}   极差 {pc.max()-pc.min():.4f}")
    print(f"各行平均偏离: {[round(v,4) for v in pr]}   极差 {pr.max()-pr.min():.4f}")

    # top-4 按列分布 vs 均匀期望（16 格每列 4 格，随机时每列期望 n*4*4/16 = n）
    print(f"\ntop4 候选的列分布: {[top4_cols.count(c) for c in range(4)]}  (随机期望每列 {n})")
    print(f"top4 候选的行分布: {[top4_rows.count(r) for r in range(4)]}  (随机期望每行 {n})")

    col_dev = max(abs(np.array([top4_cols.count(c) for c in range(4)]) - n)) / max(n, 1)
    print(f"\n最大列偏离比例: {col_dev:.0%}")
    if pc.max() - pc.min() > 0.03 or col_dev > 0.5:
        print("=> ⚠️ 存在**位置偏差**：偏离度随列/行系统变化，或 top 候选按位置聚集。")
        print("   这解释了 live 的 0/3 —— 排序里混进了与内容无关的固定成分")
        print("   （背景渐变 / 光照 / 格框裁切偏移），而不是判据'不够聪明'。")
    else:
        print("=> 未见明显位置偏差；分列/分行均值接近，top 候选分布接近均匀。")
    print(f"\n参与分析的实例: {used}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
