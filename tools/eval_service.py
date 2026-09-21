"""
商业打码服务路线的 10 次一组评测

与 tools/eval_harness.py 的区别：
    eval_harness    走浏览器，考的是「自建 canvas 求解」的能力（当前 0%）
    eval_service    走打码服务，考的是「服务商给不给得出 token」

后者是唯一有把握达到高成功率的路线，见 README「最终结论」一节。
本脚本不需要浏览器 —— 求解完全在服务商那边完成。

用法:
    export SOLVER_SERVICE_KEY=...
    export SOLVER_SERVICE_BASE=https://2captcha.com     # 可换成兼容镜像
    python tools/eval_service.py 10

产物:
    research_out/eval_service/summary.txt / report.json
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from token_service import get_service, ServiceError

SITEKEY = os.environ.get("HC_SITEKEY", "a5f74b19-9e45-40e0-b45d-47ff91b7a6c2")
URL = os.environ.get("HC_URL", "https://accounts.hcaptcha.com/demo")
OUT = os.path.join(os.environ.get("RESEARCH_OUT", "research_out"), "eval_service")


def main():
    n = next((int(a) for a in sys.argv[1:] if a.isdigit()), 10)
    os.makedirs(OUT, exist_ok=True)

    try:
        svc = get_service()
    except ServiceError as e:
        print(f"❌ 无法初始化服务后端: {e}")
        print("   请先设置 SOLVER_SERVICE_KEY（可选 SOLVER_SERVICE_BASE）")
        return 2

    print(f"服务: {svc.name}   base: {getattr(svc, 'base', '?')}")
    print(f"sitekey: {SITEKEY}   url: {URL}")
    print(f"试验次数: {n}\n")

    records = []
    for i in range(1, n + 1):
        t0 = time.time()
        rec = {"trial": i}
        try:
            token = svc.solve_hcaptcha(SITEKEY, URL)
            rec["ok"] = True
            rec["token_len"] = len(token)
            rec["token_head"] = token[:24] + "..."
            print(f"[{i:2d}/{n}] ✅ 拿到 token（长度 {len(token)}）"
                  f" {time.time()-t0:.1f}s", flush=True)
        except Exception as e:
            rec["ok"] = False
            rec["error"] = f"{type(e).__name__}: {e}"
            print(f"[{i:2d}/{n}] ❌ {rec['error'][:90]}  {time.time()-t0:.1f}s", flush=True)
        rec["seconds"] = round(time.time() - t0, 1)
        records.append(rec)

    ok = sum(1 for r in records if r.get("ok"))
    times = [r["seconds"] for r in records if r.get("ok")]
    summary = {
        "trials": n,
        "solved": ok,
        "success_rate": round(ok / n, 3),
        "avg_seconds": round(sum(times) / len(times), 1) if times else None,
        "min_seconds": min(times) if times else None,
        "max_seconds": max(times) if times else None,
        "service": svc.name,
        "sitekey": SITEKEY,
        "url": URL,
    }

    lines = [
        f"路线               : 商业打码服务 ({svc.name})",
        f"试验次数           : {n}",
        f"成功拿到 token     : {ok}/{n}  ({summary['success_rate']:.0%})",
    ]
    if times:
        lines.append(f"平均耗时           : {summary['avg_seconds']}s"
                     f"  (最快 {summary['min_seconds']}s / 最慢 {summary['max_seconds']}s)")
    else:
        lines.append("平均耗时           : 无成功样本")
    lines += [
        "",
        "注: 本路线不涉及 canvas 解析/坐标生成/拖拽手势 —— 这些都在服务商那边完成，",
        "    因此不受本仓库 pattern 题型五种 CV 思路被证伪的影响。",
    ]

    with open(os.path.join(OUT, "report.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f,
                  ensure_ascii=False, indent=2)
    with open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\n报告: {OUT}/report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
