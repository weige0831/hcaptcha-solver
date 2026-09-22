"""
在隔离环境里用本地 VLM 回答「哪两格不同」（供主环境以子进程调用）

为什么单独一个文件：主环境是 transformers 5.17，而 Moondream2 的 config 声明
transformers_version=4.52.4，两者不兼容；所以 VLM 推理必须在 .venv-vlm 里跑。
主环境负责浏览器（camoufox 装在主环境），通过子进程把图交给这个脚本、拿回答案。

用法（由 tools/eval_grid_vlm.py 调用）:
    .venv-vlm/Scripts/python.exe research/vlm_grid_answer.py <canvas.png> <out.json>
输出 json:
    {"cells": [[r,c], [r,c]], "raw": "...", "seconds": 123.4}
"""

import json
import re
import sys
import time
import warnings

warnings.filterwarnings("ignore")


PROMPT = (
    "This image shows a 4x4 grid of cartoon characters. "
    "Rows are numbered 1 to 4 from top to bottom. "
    "Columns are numbered 1 to 4 from left to right. "
    "In each row, three characters are the same and one is different. "
    "Find the two rows whose different character is most clearly different, "
    "and report the (row, column) of those two different characters. "
    "Answer strictly in the form: (row,column) and (row,column)"
)


def parse_cells(text):
    """从模型回复里抠出 (row,column) 形式的格子编号，转为 0-based [r,c]"""
    out = []
    for m in re.finditer(r"\(\s*(\d)\s*[,，]\s*(\d)\s*\)", text or ""):
        r, c = int(m.group(1)), int(m.group(2))
        if 1 <= r <= 4 and 1 <= c <= 4:
            out.append([r - 1, c - 1])
    # 去重保序
    seen, uniq = set(), []
    for k in out:
        if tuple(k) not in seen:
            seen.add(tuple(k))
            uniq.append(k)
    return uniq[:2]


def main():
    canvas_path = sys.argv[1]
    out_path = sys.argv[2]
    t0 = time.time()
    result = {"cells": [], "raw": "", "seconds": 0.0}

    try:
        import numpy as np
        from PIL import Image
        from transformers import AutoModelForCausalLM

        img = Image.open(canvas_path).convert("RGB")
        # 裁掉顶部透明题面区（与主环境 crop_puzzle 同法）
        try:
            import numpy as np2
            arr = np2.asarray(Image.open(canvas_path))
            if arr.shape[2] == 4:
                rows = (arr[:, :, 3] > 10).sum(axis=1)
                nz = np2.where(rows > 0)[0]
                if len(nz):
                    img = img.crop((0, int(nz.min()), img.size[0], img.size[1]))
        except Exception:
            pass
        # 缩小以控制 CPU 推理时间
        w, h = img.size
        if w > 512:
            img = img.resize((512, int(h * 512 / w)), Image.LANCZOS)

        model = AutoModelForCausalLM.from_pretrained(
            "models/moondream2", trust_remote_code=True, local_files_only=True)
        model.eval()
        out = model.query(img, PROMPT)
        answer = out.get("answer") if isinstance(out, dict) else str(out)
        result["raw"] = (answer or "")[:500]
        result["cells"] = parse_cells(answer)
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)[:300]}"

    result["seconds"] = round(time.time() - t0, 1)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
