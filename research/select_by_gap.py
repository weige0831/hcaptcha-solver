"""
换「选点方式」：不看绝对偏离值，看偏离值的**断层**

诊断：CLIP 嵌入与手写特征给出完全相同的 3/4 命中 —— 说明瓶颈不在特征质量，
而在**选点方式**。具体推想：同一行内"同一角色的细微渲染差异"也会产生偏离值，
这些伪偏离与真实异常项的偏离**混在一起**，于是"取 top-2"可能取到噪声。

对照实验：选点方式改为
   A. top-2 by deviation（原做法）
   B. 按偏离值排序后，取**最大断层**之上的那些项（先找 gap，再取其上方）
并同时跑两种特征（手写 / CLIP），共 4 组，看哪组能同时答对两个已知真值的实例。

预登记的成功判据：**在 pat_01 与 pat_04 上都要 2/2**，才算真的改进
（只看一个实例容易被单例巧合骗到 —— 这是本次会话我已经犯过三次的错）。

用法: python research/select_by_gap.py
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from PIL import Image
from research.grid_cells3 import crop_puzzle, find_blobs, build_cells, cell_patch, similarity

INST = "research_out/instances"
GT = {"pat_01.png": {(0,2),(1,3)}, "pat_04.png": {(1,3),(2,2)}}

def sim_matrix(keys, patches, feat):
    if feat == "hand":
        return {(a,b): similarity(patches[a], patches[b]) for a in keys for b in keys}
    m, p = sim_matrix.clip
    imgs = [Image.fromarray(patches[k] if patches[k] is not None
            else np.full((88,88,3),255,np.uint8)).convert("RGB") for k in keys]
    with torch.no_grad():
        inp = p(images=imgs, return_tensors="pt")
        f = m.get_image_features(**inp)
        if not torch.is_tensor(f):
            for at in ("pooler_output","image_embeds","last_hidden_state"):
                v = getattr(f, at, None)
                if torch.is_tensor(v): f = v; break
            else: f = f[0]
        if f.dim()==3: f = f[:,0,:]
        f = f / f.norm(dim=-1, keepdim=True)
    return {(a,b): float((f[keys.index(a)]*f[keys.index(b)]).sum()) for a in keys for b in keys}

def deviations(keys, S):
    def ls(axis):
        within, dev = [], {}
        for i in range(4):
            line=[k for k in keys if (k[1] if axis=="col" else k[0])==i]
            bc,bv=None,-1
            for k in line:
                mm=float(np.mean([S[(k,o)] for o in line if o!=k]))
                if mm>bv: bv,bc=mm,k
            within.append(bv)
            for k in line: dev[k]=0.0 if k==bc else 1.0-S[(k,bc)]
        return float(np.mean(within)), dev
    cc,cd=ls("col"); rc,rd=ls("row")
    return (cd,"col") if cc>=rc else (rd,"row")

def sel_top2(dev):
    return [k for k,_ in sorted(dev.items(), key=lambda kv:-kv[1])[:2]]

def sel_gap(dev):
    """按偏离降序，找最大断层位置，取断层之上的项（至多 2 个）"""
    s = sorted(dev.items(), key=lambda kv:-kv[1])
    vals = [v for _,v in s]
    if len(vals) < 3: return [k for k,_ in s[:2]]
    gaps = [(vals[i]-vals[i+1], i) for i in range(len(vals)-1)]
    gi = max(gaps)[1]
    take = s[:gi+1]
    return [k for k,_ in take[:2]] if len(take)<=2 else [k for k,_ in s[:2]]

def main():
    from transformers import CLIPModel, CLIPProcessor
    m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
    p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
    m.eval(); sim_matrix.clip = (m,p)

    files = [f for f in sorted(os.listdir(INST)) if f.endswith(".png")]
    for feat in ("hand","clip"):
        for sname, sfn in (("top2", sel_top2), ("gap", sel_gap)):
            tot_hit = tot_gt = 0; detail=[]
            for f in files:
                arr = np.asarray(Image.open(os.path.join(INST,f)))
                rgb, top = crop_puzzle(arr)
                blobs = find_blobs(rgb)
                cells,_ = build_cells(rgb, blobs, 4, debug=False)
                if not cells or len(cells)<16: continue
                patches = {k: cell_patch(rgb,t) for k,t in cells.items()}
                keys = sorted(patches)
                S = sim_matrix(keys, patches, feat)
                dev, axis = deviations(keys, S)
                picks = sfn(dev)
                if f in GT:
                    hit = len(set(picks)&GT[f]); tot_hit+=hit; tot_gt+=len(GT[f])
                    detail.append(f"{f}:{hit}/2")
            verdict = "✅ 两例都全对" if tot_hit==tot_gt and tot_gt==4 else ""
            print(f"  特征={feat:5s} 选点={sname:5s} -> 命中 {tot_hit}/{tot_gt}  "
                  f"({' '.join(detail)}) {verdict}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
