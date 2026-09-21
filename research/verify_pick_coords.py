"""
校验「候选坐标 -> 实际像素位置」是否落在预期的格子里

背景（链条里一处未验证的环节）：
我验证过"点击能被 canvas 接收"（像素发生变化），但**从没验证过点击是否落在
我想要的那一格上**。若格心坐标与格子实际位置之间有系统性偏移，
那么即使判据选对了格子，点出去也会点到隔壁 —— 直接解释 live 的 0/3。

这里把每道格状实例的 16 个格框 + 候选格心画到图上，供肉眼核对：
  · 绿框 = 16 个格子
  · 红十字 = 判据选出的 2 个候选的格心（即实际会点击的位置）
判据：红十字应落在被选中的格子内部（而不是压在格线上或落到隔壁）。
同时可顺带肉眼判断"选中的是否就是肉眼可见的异常项"。

用法: python research/verify_pick_coords.py
产物: research_out/verify_picks/<实例>_picks.png
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, numpy as np
from PIL import Image
from research.grid_cells3 import crop_puzzle, find_blobs, build_cells, cell_patch, similarity

INST = "research_out/instances"
OUT = "research_out/verify_picks"

def picks_for(rgb):
    rgb_c, top = crop_puzzle(rgb)
    blobs = find_blobs(rgb_c)
    cells, _ = build_cells(rgb_c, blobs, 4, debug=False)
    if not cells or len(cells) < 16:
        return None
    patches = {k: cell_patch(rgb_c, t) for k, t in cells.items()}
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
            for k in line: dev[k] = 0.0 if k == bc else 1.0 - S[k][bc]
        return float(np.mean(within)), dev
    cc, cd = line_stats("col"); rc, rd = line_stats("row")
    dev = cd if cc >= rc else rd
    ranked = sorted(dev.items(), key=lambda kv: -kv[1])
    return rgb_c, top, cells, ranked[:2]

def main():
    os.makedirs(OUT, exist_ok=True)
    n_ok = 0
    for f in sorted(os.listdir(INST)):
        if not f.endswith(".png"): continue
        arr = np.asarray(Image.open(os.path.join(INST, f)))
        r = picks_for(arr)
        if not r: 
            print(f"{f}: 非格状型"); continue
        rgb_c, top, cells, picks = r
        vis = rgb_c.copy()
        for k, t in cells.items():
            cv2.rectangle(vis, (t["x"], t["y"]), (t["x"]+t["w"], t["y"]+t["h"]), (0,190,0), 1)
        inside = []
        for k, _ in picks:
            t = cells[k]
            cx, cy = int(t["cx"]), int(t["cy"])
            # 判据：格心必须落在该格框内部（留 8px 容差，排除压在格线上的情况）
            ok = (t["x"]+8 <= cx <= t["x"]+t["w"]-8) and (t["y"]+8 <= cy <= t["y"]+t["h"]-8)
            inside.append(ok)
            cv2.drawMarker(vis, (cx, cy), (255,0,0), cv2.MARKER_CROSS, 26, 3)
            cv2.rectangle(vis, (t["x"], t["y"]), (t["x"]+t["w"], t["y"]+t["h"]), (255,0,255), 3)
        p = os.path.join(OUT, f.replace(".png","_picks.png"))
        Image.fromarray(vis).save(p)
        n_ok += all(inside)
        print(f"{f}: 候选={[list(k) for k,_ in picks]} 格心落在格内={inside} -> {p}")
    print(f"\n全部候选格心都在格内的实例数: {n_ok}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
