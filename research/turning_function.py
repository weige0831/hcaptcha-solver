"""
实验：用「转角函数」做图标旋转估计（替代失败的掩膜重叠法）

为什么换方法：掩膜重叠法在真实图标上判别余量只有 0.0002。根因是图标矮胖、
近似镜像对称，旋转后与自身重叠度几乎不变 —— 面积重叠对这类形状是弱信号。

转角函数（turning function）走的是另一条路：
    沿轮廓弧长 s 记录切线方向 theta(s)。
    同一形状旋转 phi 后，theta'(s) = theta(s) + phi（整条曲线平移一个常量）。
    因此用曲率 kappa(s) = dtheta/ds 做循环互相关，即可求出相对旋转。
它利用的是「特征沿轮廓出现的顺序」（比如火箭的头锥、尾翼），
而不是面积重叠，所以对矮胖形状也更敏感。

自检包含两部分：
    1. 合成旋转测试 —— 把参考图标旋转已知角度，看能否还原
    2. 真实图标判别余量 —— best 与次优的差距（需要 > 0.05 才算有信号）

用法:
    python research/turning_function.py research_out/click/before.png
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from canvas_strategy import detect_icons

N_SAMPLES = 256          # 轮廓重采样点数
SMOOTH = 5               # 曲率平滑窗口


def outer_contour(mask: np.ndarray) -> np.ndarray | None:
    """取最大外轮廓"""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    return c.reshape(-1, 2).astype(np.float32)


def resample(contour: np.ndarray, n: int = N_SAMPLES) -> np.ndarray:
    """按弧长等间隔重采样为 n 个点（闭合）"""
    p = np.vstack([contour, contour[:1]])
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 0:
        return np.zeros((n, 2), np.float32)
    targets = np.linspace(0, total, n, endpoint=False)
    out = np.zeros((n, 2), np.float32)
    for i, t in enumerate(targets):
        j = int(np.searchsorted(cum, t, side="right") - 1)
        j = max(0, min(j, len(seg) - 1))
        f = (t - cum[j]) / seg[j] if seg[j] > 1e-9 else 0.0
        out[i] = p[j] + f * (p[j + 1] - p[j])
    return out


def turning_function(mask: np.ndarray, n: int = N_SAMPLES):
    """
    返回 (kappa, theta)：
      theta(s) —— 切线方向（弧度），随形状旋转整体平移
      kappa(s) —— dtheta/ds，旋转不变、但随起点循环平移
    """
    c = outer_contour(mask)
    if c is None or len(c) < 8:
        return None, None
    pts = resample(c, n)
    d = np.roll(pts, -1, axis=0) - pts
    theta = np.arctan2(d[:, 1], d[:, 0])
    # 解缠绕，避免 ±pi 跳变
    theta = np.unwrap(theta * 2.0) / 2.0
    kappa = np.gradient(theta)
    k = np.ones(SMOOTH) / SMOOTH
    kappa = np.convolve(np.r_[kappa[-SMOOTH:], kappa, kappa[:SMOOTH]], k, mode="same")[SMOOTH:-SMOOTH]
    return kappa.astype(np.float32), theta.astype(np.float32)


def _norm_kappa(k: np.ndarray) -> np.ndarray:
    k = k - k.mean()
    s = k.std()
    return k / s if s > 1e-9 else k


def relative_rotation(k_ref, th_ref, k, th):
    """
    用曲率循环互相关求出起点对齐 s0，再返回相对旋转角（度）。
    """
    a, b = _norm_kappa(k_ref), _norm_kappa(k)
    n = len(a)
    # FFT 循环互相关
    corr = np.fft.irfft(np.fft.rfft(b) * np.conj(np.fft.rfft(a)), n=n)
    s0 = int(np.argmax(corr))
    # 对齐后 theta 的差值中位数 = 相对旋转
    th_b = np.roll(th, -s0)
    d = th_b - th_ref
    d = (d + np.pi) % (2 * np.pi) - np.pi
    phi = float(np.median(d))
    return np.degrees(phi), float(corr[s0]), s0


def curvature_margin(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """best 与次优相关峰的高度差 —— 衡量判别余量（越大越可信）"""
    ka, tha = turning_function(mask_a)
    kb, thb = turning_function(mask_b)
    if ka is None or kb is None:
        return 0.0
    a, b = _norm_kappa(ka), _norm_kappa(kb)
    n = len(a)
    corr = np.fft.irfft(np.fft.rfft(b) * np.conj(np.fft.rfft(a)), n=n)
    s = np.sort(corr)[::-1]
    norm = np.sqrt((a * a).sum() * (b * b).sum())
    return float((s[0] - s[1]) / norm) if norm > 0 else 0.0


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "research_out/click/before.png"
    arr = np.asarray(Image.open(src))
    rgb = arr[:, :, :3].copy()
    alpha = arr[:, :, 3] if arr.shape[2] == 4 else None
    icons = detect_icons(rgb, alpha)
    print(f"检测到 {len(icons)} 个图标")

    ref = max(icons, key=lambda d: d["area"])
    k_ref, th_ref = turning_function(ref["mask"])
    if k_ref is None:
        print("参考图标轮廓提取失败")
        return 1
    print(f"参考图标: 轮廓 {N_SAMPLES} 点，曲率范围 [{k_ref.min():.3f}, {k_ref.max():.3f}]")

    # ---- 1. 合成旋转测试 ----
    print("\n=== 合成旋转测试（把参考图标旋转已知角度）===")
    errs = []
    for true_rot in (15, 30, 60, 90, 135, 180, 225, 300):
        M = cv2.getRotationMatrix2D((ref["w"] / 2, ref["h"] / 2), true_rot, 1.0)
        rot = cv2.warpAffine(ref["mask"], M, (ref["w"], ref["h"]), flags=cv2.INTER_LINEAR)
        rot = (rot > 127).astype(np.uint8) * 255
        k2, th2 = turning_function(rot)
        if k2 is None:
            print(f"  {true_rot:3d}° -> 轮廓提取失败")
            continue
        phi, peak, s0 = relative_rotation(k_ref, th_ref, k2, th2)
        # 图像旋转 +true_rot（cv2 逆时针为正）-> 轮廓点顺时针转，theta 变化 -true_rot
        exp = (-true_rot) % 360
        got = phi % 360
        d = min(abs(got - exp), 360 - abs(got - exp))
        errs.append(d)
        print(f"  {true_rot:3d}° -> 估计 {got:6.1f}°  期望 {exp:6.1f}°  误差 {d:5.1f}°")
    if errs:
        print(f"平均误差 {np.mean(errs):.1f}°  最大 {max(errs):.1f}°")

    # ---- 2. 真实图标判别余量 ----
    print("\n=== 真实图标判别余量（曲率互相关 vs 之前的掩膜法）===")
    margins = []
    for ic in sorted(icons, key=lambda d: -d["area"])[1:8]:
        m = curvature_margin(ref["mask"], ic["mask"])
        margins.append(m)
    print(f"余量: {[round(m,4) for m in margins]}")
    print(f"中位: {np.median(margins):.4f}   最大: {max(margins):.4f}")
    verdict = "✅ 有信号（可继续）" if np.median(margins) > 0.05 else "❌ 仍无信号"
    print(f"判定: {verdict}")
    print(f"（参考：掩膜重叠法的中位余量是 0.0005~0.0016）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
