"""
实验：本地 VLM（Moondream2）能否解「找出破坏规律的图标」

市场调研结论是：这个题型的可行解法是**大型多模态 LLM**（hcaptcha-challenger
项目已把默认模型换成 Gemini 3.7 Flash）。这里退而求其次，实测一个能在
本机 CPU 上跑起来的小模型（Moondream2，约 2B，带 pointing 能力）到底行不行。

要回答的问题：
  1. 模型能不能在我们这张 canvas 图上正常工作（先做基础能力自检）
  2. 让它"指出不符合规律的图标"，给出的坐标是否落在真正的异常图标上
  3. 如果不行，是能力不够还是接口用错

产物:
    research_out/vlm/points.png      把模型给出的点画在图上
    research_out/vlm/log.json        每次问答的原始输出与耗时
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

MODEL_DIR = "models/moondream2"
SRC = sys.argv[1] if len(sys.argv) > 1 else "research_out/click/before.png"
OUT = "research_out/vlm"


def load_moondream(model_dir: str):
    """
    加载 Moondream2。

    moondream2 的远程代码是按旧版 transformers 写的，在 transformers 5.x 上会缺
    若干内部属性（如 all_tied_weights_keys）。这里用「边试边补」的方式给
    远程类打补丁，避免为了一个实验去降级 transformers（降级会连带影响
    vision.py 里的 GroundingDINO）。
    """
    import warnings
    warnings.filterwarnings("ignore")
    from transformers import AutoModelForCausalLM
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    cls = get_class_from_dynamic_module("hf_moondream.HfMoondream", model_dir)
    patched = []

    def default_for(attr: str):
        # transformers 5.x 里 all_tied_weights_keys 是按 dict 用的（会调 .keys()），
        # 给成 list 会继续报 "'list' object has no attribute 'keys'"
        if "tied_weights_keys" in attr:
            return {}
        if attr.endswith("_no_split_modules") or attr.endswith("_keys"):
            return []
        return None

    for attempt in range(12):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_dir, trust_remote_code=True, local_files_only=True)
            return model, patched
        except AttributeError as e:
            msg = str(e)
            attr = None
            # 只处理 "<something> has no attribute 'X'" 这种「缺属性」的情形，
            # 不要被 "'list' object has no attribute 'keys'" 之类误导
            if "has no attribute" in msg:
                tail = msg.split("has no attribute")[-1].strip().strip("'\"")
                head = msg.split("has no attribute")[0]
                # 只处理「模型类缺属性」；builtin 类型缺属性说明是我上一个补丁
                # 给错了类型，不能再去补，要让它报出来
                builtin = ("'list'", "'dict'", "'str'", "'int'", "'float'",
                           "'tuple'", "'NoneType'", "'ndarray'", "'Tensor'")
                if tail and " " not in tail and not any(b in head for b in builtin):
                    attr = tail
            print(f"  [调试] 捕获: {msg[:110]}")
            if not attr or attr in patched:
                raise
            val = default_for(attr)
            setattr(cls, attr, val)
            patched.append(attr)
            print(f"  [补丁] {attr} = {type(val).__name__} (第 {attempt+1} 次)")
    raise RuntimeError("补丁尝试次数用尽")


def main():
    os.makedirs(OUT, exist_ok=True)
    log = {}

    arr = np.asarray(Image.open(SRC))
    rgb = arr[:, :, :3].copy()
    alpha = arr[:, :, 3] if arr.shape[2] == 4 else None

    # 裁掉顶部透明区（题面是 DOM 画的，不在 canvas 上）
    if alpha is not None:
        rows = (alpha > 10).sum(axis=1)
        nz = np.where(rows > 0)[0]
        top = int(nz.min()) if len(nz) else 0
    else:
        top = 0
    img = Image.fromarray(rgb[top:, :]).convert("RGB")
    print(f"图像: {img.size} (裁掉顶部 {top}px)")

    t0 = time.time()
    import torch

    print("加载 Moondream2 (CPU)...", flush=True)
    try:
        model, patched = load_moondream(MODEL_DIR)
        model.eval()
        log["patched_attrs"] = patched
    except Exception as e:
        print(f"❌ 加载失败: {type(e).__name__}: {e}")
        log["load_error"] = f"{type(e).__name__}: {e}"
        with open(os.path.join(OUT, "log.json"), "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
        return 1

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    except Exception:
        tokenizer = None
    log["load_seconds"] = round(time.time() - t0, 1)
    print(f"✅ 加载完成 ({log['load_seconds']}s)")

    def ask(kind: str, arg: str):
        tt = time.time()
        try:
            if kind == "query":
                res = model.query(img, arg)
            elif kind == "point":
                res = model.point(img, arg)
            elif kind == "detect":
                res = model.detect(img, arg)
            else:
                res = None
            return res, round(time.time() - tt, 1), None
        except Exception as e:
            return None, round(time.time() - tt, 1), f"{type(e).__name__}: {e}"

    # ---- 1. 基础能力自检 ----
    print("\n=== 基础能力自检 ===")
    for q in ("How many rocket icons are in this image?",
              "Describe this image briefly."):
        res, sec, err = ask("query", q)
        print(f"  Q: {q}\n  A: {res}   ({sec}s)  {('ERR ' + err) if err else ''}")
        log.setdefault("sanity", []).append({"q": q, "a": str(res), "seconds": sec, "err": err})

    # ---- 2. pointing：指出不符合规律的图标 ----
    print("\n=== pointing: 指出异常图标 ===")
    point_queries = [
        "rocket that does not follow the pattern",
        "rocket icon facing a different direction than the others",
    ]
    all_points = []
    for q in point_queries:
        res, sec, err = ask("point", q)
        print(f"  Q(point): {q}\n  A: {res}   ({sec}s)  {('ERR ' + err) if err else ''}")
        log.setdefault("points", []).append({"q": q, "a": str(res), "seconds": sec, "err": err})
        if isinstance(res, dict) and res.get("points"):
            for p in res["points"]:
                all_points.append((p.get("x"), p.get("y")))

    # ---- 3. VQA：直接问哪个不一样 ----
    print("\n=== VQA: 哪个图标不一样 ===")
    for q in ("Which rocket icon is different from the others?",
              "Are there any rocket icons that break the pattern? Answer yes or no."):
        res, sec, err = ask("query", q)
        print(f"  Q: {q}\n  A: {res}   ({sec}s)  {('ERR ' + err) if err else ''}")
        log.setdefault("vqa", []).append({"q": q, "a": str(res), "seconds": sec, "err": err})

    # ---- 画点 ----
    vis = img.copy()
    d = ImageDraw.Draw(vis)
    W, H = vis.size
    for (x, y) in all_points:
        if x is None or y is None:
            continue
        px, py = float(x) * W, float(y) * H
        d.ellipse([px - 18, py - 18, px + 18, py + 18], outline=(255, 0, 255), width=4)
    vis.save(os.path.join(OUT, "points.png"))
    log["n_points"] = len(all_points)
    log["points_raw"] = [str(p) for p in all_points]
    with open(os.path.join(OUT, "log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"\n模型给出 {len(all_points)} 个点 -> {OUT}/points.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
