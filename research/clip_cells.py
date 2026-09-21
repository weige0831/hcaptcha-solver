"""
用 CLIP 嵌入替换手写特征，重测格状判据

动机（直接针对已定位的瓶颈）
--------------------------
我诊断出的失败原因是「判据的判断力不够」：手写特征是**颜色直方图 + 灰度相关**，
对"形状不同但配色相近"的角色分辨不足。而 CLIP 的视觉嵌入是在大规模图文对上
训出来的语义特征，对"这是不是同一种东西"的分辨力本质上强得多。

所以这里**只换特征、不换算法** —— 保持完全相同的「先判方向自洽、再找偏离」
判据，把 cell_patch 的输出从"手写相似度"换成"CLIP 嵌入的余弦相似度"，
这样能干净地隔离出"特征质量"这一个变量。

对照基准（我肉眼核对过的两例）：
    pat_01  应为 (0,2)、(1,3)
    pat_04  应为 (1,3)、(2,2)   <- 手写特征在这里漏掉了 (2,2)

用法:
    python research/clip_cells.py
产物:
    research_out/clip/{实例}_picks.png   标注了选定格
    research_out/clip/summary.txt
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
from PIL import Image

from research.grid_cells3 import crop_puzzle, find_blobs, build_cells, cell_patch

INST = "research_out/instances"
OUT = "research_out/clip"
GROUND_TRUTH = {                      # 我肉眼核对出的答案（按"每行/每列重复同一角色"读法）
    "pat_01.png": {(0, 2), (1, 3)},
    "pat_04.png": {(1, 3), (2, 2)},
}


def load_clip():
    from transformers import CLIPModel, CLIPProcessor
    m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
    p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
    m.eval()
    return m, p


@torch.no_grad()
def embed(model, proc, patches):
    """批量取 16 个格子的 CLIP 图像嵌入（已 L2 归一化，点积即余弦）"""
    keys = sorted(patches)
    imgs = []
    for k in keys:
        p = patches[k]
        if p is None:
            p = np.full((88, 88, 3), 255, np.uint8)
        imgs.append(Image.fromarray(p).convert("RGB"))
    inputs = proc(images=imgs, return_tensors="pt")
    feats = model.get_image_features(**inputs)
    # 不同 transformers 版本返回形式不同：老版直接给张量，新版给 output 对象
    if not torch.is_tensor(feats):
        for attr in ("pooler_output", "image_embeds", "last_hidden_state"):
            v = getattr(feats, attr, None)
            if torch.is_tensor(v):
                feats = v
                break
        else:
            feats = feats[0]
    if feats.dim() == 3:          # (B, T, D) -> 取 [CLS]
        feats = feats[:, 0, :]
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return keys, feats


def pick(keys, feats):
    """与 grid_cells3 相同的 axis 判据，只把相似度换成嵌入余弦"""
    S = {}
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            S[(a, b)] = float((feats[i] * feats[j]).sum())

    def line_stats(axis):
        within, dev = [], {}
        for idx in range(4):
            line = [k for k in keys if (k[1] if axis == "col" else k[0]) == idx]
            bc, bv = None, -1
            for k in line:
                m = float(np.mean([S[(k, o)] for o in line if o != k]))
                if m > bv:
                    bv, bc = m, k
            within.append(bv)
            for k in line:
                dev[k] = 0.0 if k == bc else 1.0 - S[(k, bc)]
        return float(np.mean(within)), dev

    cc, cd = line_stats("col")
    rc, rd = line_stats("row")
    axis = "col" if cc >= rc else "row"
    dev = cd if axis == "col" else rd
    ranked = sorted(dev.items(), key=lambda kv: -kv[1])
    return axis, ranked


def main():
    os.makedirs(OUT, exist_ok=True)
    model, proc = load_clip()
    lines, hits_total, gt_total = [], 0, 0

    for f in sorted(os.listdir(INST)):
        if not f.endswith(".png"):
            continue
        arr = np.asarray(Image.open(os.path.join(INST, f)))
        rgb, top = crop_puzzle(arr)
        blobs = find_blobs(rgb)
        cells, _ = build_cells(rgb, blobs, 4, debug=False)
        if not cells or len(cells) < 16:
            lines.append(f"{f:12s} 非格状型")
            continue
        patches = {k: cell_patch(rgb, t) for k, t in cells.items()}
        keys, feats = embed(model, proc, patches)
        axis, ranked = pick(keys, feats)
        picks = [k for k, _ in ranked[:2]]

        gt = GROUND_TRUTH.get(f)
        mark = ""
        if gt:
            hit = len(set(picks) & gt)
            hits_total += hit
            gt_total += len(gt)
            mark = f"  真值={sorted(gt)} 命中={hit}/{len(gt)}"
        lines.append(f"{f:12s} axis={axis} picks={[list(k) for k in picks]}{mark}")
        print(lines[-1], flush=True)

        vis = rgb.copy()
        for k, t in cells.items():
            cv2.rectangle(vis, (t["x"], t["y"]), (t["x"] + t["w"], t["y"] + t["h"]),
                          (0, 190, 0), 1)
        for i, (k, _) in enumerate(ranked[:2]):
            t = cells[k]
            col = (255, 0, 255) if i == 0 else (0, 140, 255)
            cv2.rectangle(vis, (t["x"], t["y"]), (t["x"] + t["w"], t["y"] + t["h"]), col, 3)
            cv2.drawMarker(vis, (int(t["cx"]), int(t["cy"])), col,
                           cv2.MARKER_CROSS, 28, 3)
        Image.fromarray(vis).save(os.path.join(OUT, f.replace(".png", "_picks.png")))

    lines.append("")
    if gt_total:
        lines.append(f"在有真值的实例上：CLIP 嵌入特征命中 {hits_total}/{gt_total} 个单点")
        lines.append(f"对照：手写特征在同一批上为 3/4（在 pat_04 上漏掉 (2,2)）")
    with open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines[-3:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
