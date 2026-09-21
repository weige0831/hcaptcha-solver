"""
匹配滤波：绕开分割，直接在原始像素上做旋转模板匹配

为什么再试一次：前面四种思路全部失败，但有一个共同点 —— 它们都建立在
**分割掩膜**之上（先阈值化提取图标描边，再做比较）。而我已经量化证明
「离群得分与分割质量严格同序」，也就是瓶颈在分割本身（星云区图标描边被
抗锯齿混色，颜色法失效）。

那么合理的下一步是：**根本不做分割**。直接在原始像素上用旋转模板匹配
（匹配滤波 / matched filter）：

    1. 取一个干净图标作为模板
    2. 把它旋转到各个角度，在整幅图上做归一化互相关（cv2.matchTemplate
       TM_CCOEFF_NORMED，对亮度和对比度不变，因此不怕星云与深空背景不同）
    3. 对每个已知图标位置，取其邻域内**所有旋转中的最高响应**作为该图标得分
       —— 同字形同尺寸的图标应当都能找到高响应；异常项则对任何角度都匹配不上
    4. 用多个不同图标作模板重复，取最大值，避免"参考本身恰好是异常项"导致全体偏低

这个做法不依赖分割，所以之前的混淆源被绕开了。

用法:
    python research/matched_filter.py [canvas.png]
产物:
    research_out/matched/annotated.png
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from canvas_strategy import detect_icons

OUT = "research_out/matched"
ROT_STEP = 4
TPL_SIZE = 96          # 模板边长（缓冲像素）
NEIGH = 26             # 图标位置邻域半径，用于取最大响应


def build_template(rgb: np.ndarray, cx: float, cy: float, size: int = TPL_SIZE):
    h, w = rgb.shape[:2]
    x0, x1 = int(cx - size / 2), int(cx + size / 2)
    y0, y1 = int(cy - size / 2), int(cy + size / 2)
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None
    return rgb[y0:y1, x0:x1].copy()


def best_response_map(gray: np.ndarray, tpl_gray: np.ndarray) -> np.ndarray:
    """所有旋转角下 matchTemplate 的逐像素最大响应"""
    best = np.full(gray.shape, -1.0, np.float32)
    th, tw = tpl_gray.shape
    for a in range(0, 360, ROT_STEP):
        M = cv2.getRotationMatrix2D((tw / 2.0, th / 2.0), a, 1.0)
        rot = cv2.warpAffine(tpl_gray, M, (tw, th), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
        res = cv2.matchTemplate(gray, rot, cv2.TM_CCOEFF_NORMED)
        # 响应对应模板左上角；转成模板中心坐标再铺回全图尺寸
        pad = np.full(gray.shape, -1.0, np.float32)
        oy, ox = th // 2, tw // 2
        pad[oy:oy + res.shape[0], ox:ox + res.shape[1]] = res
        np.maximum(best, pad, out=best)
    return best


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "research_out/click/before.png"
    os.makedirs(OUT, exist_ok=True)
    arr = np.asarray(Image.open(src))
    rgb = arr[:, :, :3].copy()
    alpha = arr[:, :, 3] if arr.shape[2] == 4 else None

    # 只看拼图区（裁掉顶部透明题面区）
    top = 0
    if alpha is not None:
        rows = (alpha > 10).sum(axis=1)
        nz = np.where(rows > 0)[0]
        top = int(nz.min()) if len(nz) else 0
    rgb_p = rgb[top:, :].copy()
    gray = cv2.cvtColor(rgb_p, cv2.COLOR_RGB2GRAY)
    print(f"拼图区 {gray.shape[1]}x{gray.shape[0]} (裁掉顶部 {top}px)")

    icons = detect_icons(rgb, alpha)
    # 把图标坐标平移到拼图区坐标系
    for ic in icons:
        ic["py"] = ic["cy"] - top
    icons = [ic for ic in icons if ic["py"] > 10]
    print(f"检测到 {len(icons)} 个图标位置")

    # 选 3 个参考模板：面积最大的三个（描边最完整）
    refs = sorted(icons, key=lambda d: -d["area"])[:3]
    print("参考模板: " + ", ".join(f"({r['cx']:.0f},{r['py']:.0f})" for r in refs))

    acc = np.full(gray.shape, -1.0, np.float32)
    for k, r in enumerate(refs):
        tpl = build_template(rgb_p, r["cx"], r["py"])
        if tpl is None:
            continue
        resp = best_response_map(gray, cv2.cvtColor(tpl, cv2.COLOR_RGB2GRAY))
        print(f"  模板#{k} 响应: 最大 {resp.max():.3f} 中位 {np.median(resp[resp>0]):.3f}")
        np.maximum(acc, resp, out=acc)

    # 每个图标取其邻域内的最高响应
    h, w = gray.shape
    scores = []
    for ic in icons:
        cx, cy = int(ic["cx"]), int(ic["py"])
        x0, x1 = max(0, cx - NEIGH), min(w, cx + NEIGH)
        y0, y1 = max(0, cy - NEIGH), min(h, cy + NEIGH)
        s = float(acc[y0:y1, x0:x1].max())
        scores.append((s, ic))
    scores.sort(key=lambda t: t[0])

    vals = np.array([s for s, _ in scores])
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) or 1e-9
    rz = (vals - med) / (1.4826 * mad)
    print(f"\n=== 匹配响应得分（越高越像参考字形）===")
    print(f"  中位 {med:.3f}  区间 [{vals.min():.3f}, {vals.max():.3f}]")
    print(f"  离散度 (std/mean) = {vals.std()/vals.mean():.3f}")
    print("  得分最低的 6 个（最可能是异常项）:")
    for i in range(min(6, len(scores))):
        s, ic = scores[i]
        print(f"    #{i+1} 得分 {s:.3f}  z={rz[i]:+.2f}  "
              f"位置({ic['cx']:.0f},{ic['cy']:.0f}) area={ic['area']}"
              f"  {'<== 显著离群' if rz[i] < -3.5 else ''}")

    # 断层检查：第2低与第3低之间。
    # 注意：断层大不等于有离群 —— 分布是连续渐变时也会有"看似明显的断层"。
    # 判定必须以稳健 z 为准（下面单独统计），断层只作参考。
    if len(scores) >= 4:
        g23 = scores[2][0] - scores[1][0]
        g34 = scores[3][0] - scores[2][0]
        print(f"\n  断层 2-3 = {g23:.3f}   3-4 = {g34:.3f}  (仅参考，见下判定)")
    n_out = int((rz < -3.5).sum())
    print(f"\n  稳健 z < -3.5 的图标数: {n_out}  （题目要找 2 个）")
    if n_out == 0:
        print("  => 无任何图标达到离群阈值：**分布是连续渐变，没有干净的异常项**。")
        print("     即便绕开了分割（本方法直接在原始像素上匹配），也拿不到信号 ——")
        print("     说明异常不是「字形与其他明显不同」这种可由相关性捕获的差异。")
    elif n_out == 2:
        print("  => 恰好 2 个离群，与题目要求相符，值得进一步验证。")
    else:
        print(f"  => 离群数 {n_out} 与题目要求的 2 不符，阈值可能需调整。")

    vis = rgb_p.copy()
    picks = set(id(ic) for _, ic in scores[:2])
    for s, ic in scores:
        x, y, ww, hh = ic["x"], ic["y"] - top, ic["w"], ic["h"]
        out = id(ic) in picks
        cv2.rectangle(vis, (x, y), (x + ww, y + hh),
                      (255, 0, 255) if out else (0, 200, 0), 3 if out else 1)
        cv2.putText(vis, f"{s:.2f}", (x, max(14, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 255) if out else (0, 220, 0), 2 if out else 1, cv2.LINE_AA)
    Image.fromarray(vis).save(os.path.join(OUT, "annotated.png"))
    print(f"\n标注图: {OUT}/annotated.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
