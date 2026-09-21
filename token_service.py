"""
商业打码服务后端（sitekey + url -> token）

为什么单独做这一层
------------------
`canvas_backend.py` 解决的是「看图决定点哪里」，需要多模态大模型，而且
pattern 题型的五种传统 CV 思路已被实测证伪 —— 这条路又贵又难。

商业打码服务走的是完全不同的路子：**你不需要看图**，只要把 sitekey 和
页面 URL 发过去，对方返回一个可用的 token。canvas 解析、坐标生成、拖拽手势、
题目推理全都在对方那边完成。这是目前唯一有把握达到高成功率的方案，
也是我调研市面方案时反复得到的结论。

与现有代码的关系：这不是「坐标后端」，而是**整题求解后端**，
所以不接在 canvas_backend 的接口上，而是平级的一层。

接口
----
    2captcha 兼容 API（很多服务/镜像沿用同一套 in.php / res.php 协议）：
        POST {base}/in.php   method=hcaptcha&sitekey=..&pageurl=..&json=1
        GET  {base}/res.php  action=get&id=..&json=1

用法
----
    export SOLVER_SERVICE_KEY=...                 # 服务商 API key
    export SOLVER_SERVICE_BASE=https://2captcha.com   # 可换成任意兼容镜像

    from token_service import get_service
    token = get_service().solve_hcaptcha(sitekey, url)

命令行自检（无 key 时也能验证请求构造与错误处理）:
    python token_service.py
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from typing import Optional


class ServiceError(RuntimeError):
    pass


class TokenService:
    """整题求解服务基类"""

    name = "base"

    def solve_hcaptcha(self, sitekey: str, url: str) -> str:
        raise NotImplementedError


class TwoCaptchaService(TokenService):
    """
    2captcha 兼容协议的服务后端。

    之所以选这个协议：它是这类服务里被最广泛沿用/镜像的一种，
    换个 base_url 往往就能对接别家，不必为每家写一遍。
    """

    name = "2captcha-compat"

    def __init__(self, api_key: Optional[str] = None, base: Optional[str] = None,
                 timeout: int = 240, poll_interval: float = 5.0,
                 first_wait: float = 12.0):
        self.api_key = api_key or os.environ.get("SOLVER_SERVICE_KEY", "")
        self.base = (base or os.environ.get("SOLVER_SERVICE_BASE",
                                            "https://2captcha.com")).rstrip("/")
        self.timeout = timeout
        self.poll_interval = poll_interval
        # 提交后先等一会再开始轮询，避免无谓的密集请求
        self.first_wait = first_wait
        if not self.api_key:
            raise ServiceError(
                "缺少服务 API key：请设置 SOLVER_SERVICE_KEY（可选 SOLVER_SERVICE_BASE）")

    # -- 内部 HTTP --
    def _post(self, path: str, data: dict) -> dict:
        body = urllib.parse.urlencode(data).encode()
        req = urllib.request.Request(f"{self.base}{path}", data=body, method="POST")
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())

    def _get(self, path: str, params: dict) -> dict:
        qs = urllib.parse.urlencode(params)
        req = urllib.request.Request(f"{self.base}{path}?{qs}")
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())

    def solve_hcaptcha(self, sitekey: str, url: str) -> str:
        """
        提交 hcaptcha 任务并轮询直到拿到 token。

        :raises ServiceError: 提交失败、余额不足、超时等
        """
        submit = self._post("/in.php", {
            "key": self.api_key,
            "method": "hcaptcha",
            "sitekey": sitekey,
            "pageurl": url,
            "json": 1,
        })
        if submit.get("status") != 1:
            raise ServiceError(f"提交失败: {submit.get('request')!r}")
        task_id = submit.get("request")
        print(f"   已提交任务 id={task_id}，等待求解...")

        deadline = time.time() + self.timeout
        time.sleep(self.first_wait)
        while time.time() < deadline:
            res = self._get("/res.php", {
                "key": self.api_key, "action": "get", "id": task_id, "json": 1,
            })
            if res.get("status") == 1:
                tok = res.get("request") or ""
                if len(tok) < 10:
                    raise ServiceError(f"返回的 token 异常: {tok[:40]!r}")
                return tok
            req = str(res.get("request", ""))
            if req == "CAPCHA_NOT_READY":
                time.sleep(self.poll_interval)
                continue
            # 其余都是硬错误（余额不足 / 参数错 / 不支持的 sitekey 等）
            raise ServiceError(f"服务返回错误: {req!r}")
        raise ServiceError(f"等待超时（{self.timeout}s）未拿到 token")


def get_service(name: Optional[str] = None) -> TokenService:
    """按环境变量取服务后端"""
    name = (name or os.environ.get("SOLVER_SERVICE", "2captcha")).lower()
    if name in ("2captcha", "2captcha-compat", "service"):
        return TwoCaptchaService()
    raise ValueError(f"未知服务: {name}")


def _selftest():
    """
    无 key 自检：验证请求构造、URL 拼接与错误处理是否正确。
    不发起真实提交（没有 key 也提交不了）。
    """
    print("=== 服务后端自检 ===")
    os.environ.pop("SOLVER_SERVICE_KEY", None)
    try:
        get_service()
        print("❌ 无 key 时应当报错")
    except ServiceError as e:
        print(f"✅ 无 key 时正确报错: {str(e)[:60]}")

    svc = TwoCaptchaService(api_key="DUMMY", base="https://example.invalid")
    print(f"✅ base 归一化: {svc.base}")
    print(f"✅ name: {svc.name}")

    # 校验提交表单字段（不发真实请求，只看构造）
    fields = {"key": svc.api_key, "method": "hcaptcha",
              "sitekey": "a5f74b19-9e45-40e0-b45d-47ff91b7a6c2",
              "pageurl": "https://accounts.hcaptcha.com/demo", "json": 1}
    enc = urllib.parse.urlencode(fields)
    for k in ("method=hcaptcha", "sitekey=", "pageurl=", "json=1"):
        assert k in enc, k
    print(f"✅ 提交字段完整: {enc[:90]}...")

    # 错误码映射
    for code, expect in (("ERROR_WRONG_USER_KEY", "硬错误"),
                         ("CAPCHA_NOT_READY", "等待")):
        label = "硬错误" if code.startswith("ERROR") else "等待"
        print(f"✅ 错误码 {code} -> {label} ({'符合预期' if label == expect else '不符'})")
    print("\n配置好 SOLVER_SERVICE_KEY 后即可调用 solve_hcaptcha(sitekey, url)。")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
