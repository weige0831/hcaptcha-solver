"""
单次查询计时 + 输出质量：把图缩小到能在 CPU 上跑完的尺寸

注意：本脚本必须用隔离环境跑（`models/moondream2/config.json` 声明
`transformers_version: 4.52.4`，主环境的 5.17 会报
`'HfMoondream' object has no attribute 'all_tied_weights_keys'`）：

    .venv-vlm/Scripts/python.exe research/vlm_quick.py research_out/click/before.png 448

所以这里用 __main__ 保护 —— 否则仅仅 import 本模块就会去加载 3.6GB 模型
（并且因为 transformers 版本不匹配而崩溃）。
"""

import sys
import time
import warnings

warnings.filterwarnings("ignore")


def main():
    import numpy as np
    from PIL import Image
    from transformers import AutoModelForCausalLM

    src = sys.argv[1] if len(sys.argv) > 1 else "research_out/click/before.png"
    maxw = int(sys.argv[2]) if len(sys.argv) > 2 else 448

    arr = np.asarray(Image.open(src))
    rgb = arr[:, :, :3]
    if arr.shape[2] == 4:
        rows = (arr[:, :, 3] > 10).sum(axis=1)
        nz = np.where(rows > 0)[0]
        rgb = rgb[int(nz.min()):, :]
    img = Image.fromarray(rgb).convert("RGB")
    w, h = img.size
    if w > maxw:
        img = img.resize((maxw, int(h * maxw / w)), Image.LANCZOS)
    print(f"图像 {img.size} (原 {w}x{h})", flush=True)

    t = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        "models/moondream2", trust_remote_code=True, local_files_only=True)
    model.eval()
    print(f"加载完成 {time.time() - t:.1f}s", flush=True)

    for q in ["How many rocket icons are in this image?",
              "Which rocket icon is different from the others? Answer with its position."]:
        t = time.time()
        try:
            out = model.query(img, q)
            ans = out.get("answer") if isinstance(out, dict) else out
            ans = (ans or "").replace("\n", " ")[:300]
            print(f"Q: {q}\nA: {ans}\n   ({time.time() - t:.1f}s)", flush=True)
        except Exception as e:
            print(f"Q: {q}\nERR: {type(e).__name__}: {str(e)[:200]}"
                  f"  ({time.time() - t:.1f}s)", flush=True)

    t = time.time()
    try:
        out = model.point(img, "rocket that does not follow the pattern")
        pts = (out or {}).get("points", [])
        print(f"point -> {len(pts)} 个点, 前3: {pts[:3]}  ({time.time() - t:.1f}s)", flush=True)
    except Exception as e:
        print(f"point ERR: {type(e).__name__}: {str(e)[:200]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
