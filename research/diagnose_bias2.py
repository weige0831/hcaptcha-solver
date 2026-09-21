"""
诊断二：把格子裁块里的**背景**去掉后，位置偏差是否消失

诊断一确认了偏离度存在系统性位置成分（各列均值 0.099/0.102/0.152/0.165，极差 0.067）。
推测机制：页面背景本身有颜色渐变，而我的格块把方块底/背景也一起算了进去，
于是不同列的格子带上不同的背景色调 -> 色直方图系统性偏差异 -> 偏离度随位置变化。

验证方法：裁块后先把**背景抠掉**（只保留角色前景：强饱和 或 明显偏暗），
其余像素置为固定白，再重算偏离度与位置偏差。若极差显著下降，机制成立且可修。

用法: python research/diagnose_bias2.py
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, numpy as np
from PIL import Image
from research.grid_cells3 import crop_puzzle, find_blobs, build_cells, cell_patch, similarity

INST = "research_out/instances"

def fg_patch(rgb, t, size=88):
    """裁块后抠掉背景，只留角色前景，其余填白"""
    p = cell_patch(rgb, t, size=size)
    if p is None:
        return None
    hsv = cv2.cvtColor(p, cv2.COLOR_RGB2HSV)
    s, v = hsv[:,:,1].astype(np.int16), hsv[:,:,2].astype(np.int16)
    fg = (s > 60) | (v < 150)
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3,3), np.uint8)).astype(bool)
    out = np.full_like(p, 255)
    out[fg] = p[fg]
    return out

def scores(arr, use_fg):
    rgb, top = crop_puzzle(arr)
    blobs = find_blobs(rgb)
    cells, _ = build_cells(rgb, blobs, 4, debug=False)
    if not cells or len(cells) < 16:
        return None
    patches = {k: (fg_patch(rgb, t) if use_fg else cell_patch(rgb, t)) for k, t in cells.items()}
    keys = sorted(patches)
    S = {k: {} for k in keys}
    for i, a in enumerate(keys):
        for b in keys[i+1:]:
            v = similarity(patches[a], patches[b]); S[a][b] = v; S[b][a] = v
    def line_stats(axis):
        within, dev = [], {}
        for idx in range(4):
            line = [k for k in keys if (k[1] if axis=="col" else k[0]) == idx]
            bc, bv = None, -1
            for k in line:
                m = float(np.mean([S[k][o] for o in line if o != k]))
                if m > bv: bv, bc = m, k
            within.append(bv)
            for k in line:
                dev[k] = 0.0 if k == bc else 1.0 - S[k][bc]
        return float(np.mean(within)), dev
    cc, cd = line_stats("col"); rc, rd = line_stats("row")
    return (cd if cc >= rc else rd)

def col_row_means(dev):
    cm = [np.mean([v for k,v in dev.items() if k[1]==c]) for c in range(4)]
    rm = [np.mean([v for k,v in dev.items() if k[0]==r]) for r in range(4)]
    return cm, rm

for tag, use_fg in (("原始（含背景）", False), ("抠掉背景（只留角色）", True)):
    cms, rms, n = [], [], 0
    for f in sorted(os.listdir(INST)):
        if not f.endswith(".png"): continue
        arr = np.asarray(Image.open(os.path.join(INST, f)))
        dev = scores(arr, use_fg)
        if dev is None: continue
        cm, rm = col_row_means(dev); cms.append(cm); rms.append(rm); n += 1
    if not n:
        print(f"{tag}: 无可用实例"); continue
    pc = np.array(cms).mean(axis=0); pr = np.array(rms).mean(axis=0)
    print(f"{tag}  (n={n})")
    print(f"  各列平均偏离 {[round(v,4) for v in pc]}   极差 {pc.max()-pc.min():.4f}")
    print(f"  各行平均偏离 {[round(v,4) for v in pr]}   极差 {pr.max()-pr.min():.4f}")
