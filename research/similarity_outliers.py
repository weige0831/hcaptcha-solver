"""
实验：用「两两相似度矩阵」找离群图标（不依赖任何规律假设）

动机
----
单对匹配的判别余量太弱（掩膜法 0.0005、转角函数 0.0082，阈值需 >0.05），
但"弱信号"不等于"没信号"。每对比较的噪声若独立，把同一个图标与其余
19 个图标的相似度取统计量，信噪比可以提升 sqrt(19) ≈ 4.4 倍。

这个做法还有一个好处：**不需要知道规律是什么**。
"破坏规律的那两个"在定义上就是与其余图标整体最不一致的两个，
所以直接看「每个图标与其他所有图标的相似度中位数」，最低的两个即候选。

同时把「镜像模板」也纳入搜索：如果异常是左右翻转（而不是旋转），
旋转扫描不会发现它，但镜像模板会匹配得很好。

产物:
    research_out/similarity/matrix.txt     相似度矩阵
    research_out/similarity/ranking.txt    每个图标的离群得分
    research_out/similarity/annotated.png  标注图
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from canvas_strategy import detect_icons, _norm_patch, _similarity

ROT_STEP = 5
OUT_DIR = "research_out/similarity"


def build_variants(mask: np.ndarray, mirrored: bool = False):
    """生成某图标在所有旋转角（可选含镜像）下的归一化模板"""
    base = _norm_patch(mask)
    if base is None:
        return []
    out = []
    srcs = [base] if not mirrored else [base, cv2.flip(base, 1)]
    for s in srcs:
        for a in range(0, 360, ROT_STEP):
            M = cv2.getRotationMatrix2D((base.shape[0] / 2, base.shape[1] / 2), a, 1.0)
            r = cv2.warpAffine(s, M, base.shape, flags=cv2.INTER_LINEAR)
            out.append(cv2.GaussianBlur(r, (0, 0), 1.5))
    return out


def pair_similarity(icons, i: int, j: int, variants_cache, use_mirror: bool):
    """图标 i 与 j 在最优旋转（可选镜像）下的相似度"""
    key = (j, use_mirror)
    if key not in variants_cache:
        variants_cache[key] = build_variants(icons[j]["mask"], use_mirror)
    p = _norm_patch(icons[i]["mask"])
    if p is None or not variants_cache[key]:
        return 0.0
    return max(_similarity(p, v) for v in variants_cache[key])


def analyze(icons, use_mirror: bool, tag: str):
    n = len(icons)
    cache = {}
    M = np.zeros((n, n), np.float32)
    for i in range(n):
        for j in range(n):
            if i == j:
                M[i, j] = 1.0
                continue
            M[i, j] = pair_similarity(icons, i, j, cache, use_mirror)
    M = (M + M.T) / 2.0                       # 对称化

    # 每个图标与其余图标的相似度中位数：越低越"离群"
    med = np.array([np.median(np.delete(M[i], i)) for i in range(n)])
    order = np.argsort(med)                    # 升序：最离群的在前

    print(f"\n=== {tag} ===")
    print("离群得分（与其余图标相似度中位数，越低越离群）:")
    for rank, idx in enumerate(order):
        mark = "  <-- 候选" if rank < 2 else ""
        print(f"  第{rank+1:2d}低  icon#{idx:2d}  med={med[idx]:.4f}{mark}")

    # 第2名与第3名之间是否有明显断层？
    gap = med[order[2]] - med[order[1]] if n > 2 else 0.0
    spread = med.max() - med.min()
    print(f"  第2低={med[order[1]]:.4f}  第3低={med[order[2]]:.4f}  断层={gap:.4f}")
    print(f"  整体跨度={spread:.4f}  断层/跨度={gap/spread if spread>0 else 0:.2%}")
    return M, med, order, gap


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "research_out/click/before.png"
    os.makedirs(OUT_DIR, exist_ok=True)
    arr = np.asarray(Image.open(src))
    rgb = arr[:, :, :3].copy()
    alpha = arr[:, :, 3] if arr.shape[2] == 4 else None
    icons = detect_icons(rgb, alpha)
    print(f"检测到 {len(icons)} 个图标")

    results = {}
    for use_mirror, tag in ((False, "仅旋转"), (True, "旋转 + 镜像")):
        M, med, order, gap = analyze(icons, use_mirror, tag)
        results[tag] = (M, med, order, gap)
        with open(os.path.join(OUT_DIR, f"matrix_{'mirror' if use_mirror else 'rot'}.txt"),
                  "w", encoding="utf-8") as f:
            f.write(tag + "\n")
            for i in range(len(icons)):
                f.write(" ".join(f"{M[i,j]:.3f}" for j in range(len(icons))) + "\n")

    # 用断层最明显的那组做标注
    best_tag = max(results, key=lambda t: results[t][3])
    M, med, order, _ = results[best_tag]
    print(f"\n断层最明显: {best_tag}")

    vis = rgb.copy()
    picked = set(int(i) for i in order[:2])
    for i, ic in enumerate(icons):
        x, y, w, h = ic["x"], ic["y"], ic["w"], ic["h"]
        if i in picked:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 255), 3)
            cv2.putText(vis, f"OUT med={med[i]:.3f}", (x, max(14, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (60, 200, 60), 1)
            cv2.putText(vis, f"{med[i]:.3f}", (x, max(12, y - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 200, 60), 1, cv2.LINE_AA)
    Image.fromarray(vis).save(os.path.join(OUT_DIR, "annotated.png"))
    print(f"标注图: {OUT_DIR}/annotated.png")
    print(f"候选图标下标: {sorted(picked)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
