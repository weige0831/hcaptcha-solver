"""
canvas 题型求解策略

目前实现「找出不符合规律的图标」这一类（pattern 题型）：
    1. 从 canvas 像素里分割出所有图标（白色描边、低饱和、高亮度）
    2. 估计每个图标的旋转角（以其中一个为参考做旋转扫描匹配）
    3. 用「局部邻域角度的中位数偏差」找离群点 —— 假设同题内图标角度
       构成平滑场/规律序列，破坏规律的那几个局部偏差最大
    4. 返回需要点击的坐标（canvas 缓冲坐标系）

离线自测（在已保存的 canvas PNG 上跑，不出网）:
    python canvas_strategy.py research_out/click/before.png
产物:
    research_out/offline/annotated.png  标注了检测到的图标与角度
"""

from __future__ import annotations

import os
import sys
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# --- 可调参数 ---
ICON_V_MIN = 140          # 亮度下限
ICON_S_MAX = 70           # 饱和度上限
ICON_AREA_MIN = 120       # 连通域面积下限（缓冲像素）
ICON_AREA_MAX = 6000      # 上限，滤掉整片星云
ROT_SWEEP_DEG = 5         # 角度扫描步长
PATCH_SIZE = 96           # 归一化后的图标画布边长
OUTLIER_KNN = 5           # 局部邻域大小
OUTLIER_TOPK = 2          # 取偏差最大的前 K 个


# ----------------------------------------------------------------------
# 1. 图标分割
# ----------------------------------------------------------------------
def _opaque_bounds(alpha: np.ndarray) -> Tuple[int, int]:
    rows = (alpha > 10).sum(axis=1)
    nz = np.where(rows > 0)[0]
    if len(nz) == 0:
        return 0, alpha.shape[0]
    return int(nz.min()), int(nz.max()) + 1


def detect_icons(rgb: np.ndarray, alpha: Optional[np.ndarray] = None,
                 v_min: int = ICON_V_MIN, s_max: int = ICON_S_MAX,
                 area_min: int = ICON_AREA_MIN, area_max: int = ICON_AREA_MAX,
                 debug: bool = False) -> List[Dict]:
    """
    分割图标，返回 [{cx, cy, x, y, w, h, area, mask}]
    mask 是该图标在其 bbox 内的 0/1 图（uint8）
    """
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    v, s = hsv[:, :, 2], hsv[:, :, 1]

    mask = ((v >= v_min) & (s <= s_max)).astype(np.uint8)
    if alpha is not None:
        mask &= (alpha > 10).astype(np.uint8)

    # 闭运算把描边连成整体
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    icons = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if not (area_min <= area <= area_max):
            continue
        # 太扁/太长的多半是星云碎块或边框
        if bw < 12 or bh < 12:
            continue
        if bw > 260 or bh > 260:
            continue
        patch = (labels[y:y + bh, x:x + bw] == i).astype(np.uint8) * 255
        icons.append({
            "cx": float(cents[i][0]), "cy": float(cents[i][1]),
            "x": int(x), "y": int(y), "w": int(bw), "h": int(bh),
            "area": int(area), "mask": patch,
        })

    if debug:
        print(f"  分割: 阈值 v>={v_min} s<={s_max} -> {len(icons)} 个候选")
        for ic in sorted(icons, key=lambda d: -d["area"])[:6]:
            print(f"    area={ic['area']:5d} bbox=({ic['x']},{ic['y']},{ic['w']},{ic['h']})")
    return icons


# ----------------------------------------------------------------------
# 2. 旋转角估计
# ----------------------------------------------------------------------
def _norm_patch(patch: np.ndarray, target_area: int = 900,
                size: int = PATCH_SIZE) -> Optional[np.ndarray]:
    """
    把图标 mask 归一化：按质心居中、按 sqrt(面积) 缩放。

    注意：必须用「质心 + 面积」而不是 bbox —— bbox 会随形状旋转而改变，
    用它归一化会把旋转信息抹掉（这是之前匹配分数只有 0.2~0.4 的原因）。
    面积和质心都是旋转不变量。
    """
    ys, xs = np.where(patch > 0)
    if len(xs) == 0:
        return None
    cy, cx = ys.mean(), xs.mean()
    scale = float(np.sqrt(target_area / len(xs)))
    ys2 = (ys - cy) * scale + size / 2.0
    xs2 = (xs - cx) * scale + size / 2.0
    out = np.zeros((size, size), np.float32)
    keep = (ys2 >= 0) & (ys2 < size) & (xs2 >= 0) & (xs2 < size)
    out[ys2[keep].astype(np.int32), xs2[keep].astype(np.int32)] = 1.0
    # 描边很细，先模糊再比较，避免 1px 偏差就让相关性崩掉
    return cv2.GaussianBlur(out, (0, 0), 2.0)


def _similarity(a: np.ndarray, b: np.ndarray) -> float:
    """余弦相似度（对细描边比硬 IoU 稳健）"""
    na, nb = float(np.sqrt((a * a).sum())), float(np.sqrt((b * b).sum()))
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return float((a * b).sum() / (na * nb))


def estimate_rotations(icons: List[Dict], size: int = PATCH_SIZE,
                       step_deg: int = ROT_SWEEP_DEG) -> None:
    """
    以一个参考图标为基准，旋转扫描匹配出每个图标的相对角度。

    写入 icon['angle']（度）、icon['angle_score']（相似度 0~1）与
    icon['is_ref']。角度是「相对参考图标的旋转量」，同题内自洽，
    所以后续用相对差做离群检测不受参考选择影响。
    """
    if not icons:
        return
    if len(icons) < 3:
        for ic in icons:
            ic["angle"], ic["angle_score"], ic["is_ref"] = 0.0, 0.0, False
        return

    # 参考选面积最大的：描边最完整，匹配最稳
    ref_idx = int(np.argmax([ic["area"] for ic in icons]))
    ref = _norm_patch(icons[ref_idx]["mask"], size=size)
    if ref is None:
        return

    angles = list(range(0, 360, step_deg))
    rotated = []
    for a in angles:
        M = cv2.getRotationMatrix2D((size / 2.0, size / 2.0), a, 1.0)
        rotated.append(cv2.GaussianBlur(
            cv2.warpAffine(ref, M, (size, size), flags=cv2.INTER_LINEAR), (0, 0), 1.5))

    for i, ic in enumerate(icons):
        p = _norm_patch(ic["mask"], size=size)
        if p is None:
            ic["angle"], ic["angle_score"], ic["is_ref"] = 0.0, 0.0, (i == ref_idx)
            continue
        best_a, best_s = 0, -1.0
        for a, r in zip(angles, rotated):
            s = _similarity(p, r)
            if s > best_s:
                best_s, best_a = s, a
        ic["angle"] = float(best_a)
        ic["angle_score"] = float(best_s)
        ic["is_ref"] = (i == ref_idx)


# ----------------------------------------------------------------------
# 3. 离群点
# ----------------------------------------------------------------------
def _ang_diff(a: float, b: float) -> float:
    """角度差，取 [-180,180]"""
    d = abs((a - b) % 360)
    return min(d, 360 - d)


def find_outliers(icons: List[Dict], k: int = OUTLIER_KNN,
                  topk: int = OUTLIER_TOPK) -> List[int]:
    """
    用局部邻域角度的中位数偏差找离群点。

    思路：同题内图标角度构成平滑场（或规律序列），因此每个图标的角度
    应当接近其空间近邻的中位数；偏离最大的即为破坏规律者。
    返回按偏差降序的图标下标。
    """
    if len(icons) < 4:
        return []
    pts = np.array([[ic["cx"], ic["cy"]] for ic in icons], dtype=np.float32)
    angs = np.array([ic["angle"] for ic in icons], dtype=np.float32)

    dev = np.zeros(len(icons), dtype=np.float32)
    for i in range(len(icons)):
        d = np.linalg.norm(pts - pts[i], axis=1)
        idx = np.argsort(d)[1:k + 1]              # 排除自己
        if len(idx) == 0:
            continue
        neigh = angs[idx]
        # 邻域中位数（环形量：先对齐到 i 的角度再取中位）
        rel = np.array([_ang_diff(a, angs[i]) * (1 if ((a - angs[i]) % 360) <= 180 else -1)
                        for a in neigh])
        dev[i] = abs(np.median(rel))
    order = np.argsort(-dev)
    return [int(i) for i in order[:topk]]


def analyze(rgb: np.ndarray, alpha: Optional[np.ndarray] = None,
            debug: bool = False) -> Dict:
    """完整分析：分割 -> 角度 -> 离群点"""
    icons = detect_icons(rgb, alpha, debug=debug)
    estimate_rotations(icons)
    outlier_idx = find_outliers(icons)
    return {"icons": icons, "outliers": outlier_idx,
            "click_points": [(icons[i]["cx"], icons[i]["cy"]) for i in outlier_idx]}


# ----------------------------------------------------------------------
# 可视化 / 离线自测
# ----------------------------------------------------------------------
def visualize(rgb: np.ndarray, result: Dict, path: str) -> None:
    vis = rgb.copy()
    outliers = set(result["outliers"])
    for i, ic in enumerate(result["icons"]):
        is_out = i in outliers
        color = (255, 60, 60) if is_out else (60, 200, 60)
        x, y, w, h = ic["x"], ic["y"], ic["w"], ic["h"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        cx, cy = int(ic["cx"]), int(ic["cy"])
        ang = np.deg2rad(ic["angle"])
        cv2.line(vis, (cx, cy),
                 (int(cx + 40 * np.cos(ang)), int(cy + 40 * np.sin(ang))), color, 2)
        cv2.putText(vis, f"{int(ic['angle'])}", (x, max(12, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        if is_out:
            cv2.circle(vis, (cx, cy), int(max(w, h) * 0.7), (255, 0, 255), 3)
    Image.fromarray(vis).save(path)


def _main():
    if len(sys.argv) < 2:
        print("用法: python canvas_strategy.py <canvas.png> [out_dir]")
        return 1
    src = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "research_out/offline"
    os.makedirs(out_dir, exist_ok=True)

    im = Image.open(src)
    arr = np.asarray(im)
    rgb = arr[:, :, :3].copy()
    alpha = arr[:, :, 3] if arr.shape[2] == 4 else None

    res = analyze(rgb, alpha, debug=True)
    print(f"\n检测到 {len(res['icons'])} 个图标")
    print("角度分布:", sorted(int(i['angle']) for i in res['icons']))
    print("匹配分数 (IoU):", [round(i['angle_score'], 2) for i in res['icons']])
    print(f"离群点下标: {res['outliers']}")
    print("待点击坐标 (缓冲系):", [(round(x), round(y)) for x, y in res['click_points']])

    vis_path = os.path.join(out_dir, "annotated.png")
    visualize(rgb, res, vis_path)
    print("已保存标注图:", vis_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
