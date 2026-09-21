"""
商业打码服务后端（sitekey + url -> token）

为什么单独做这一层
------------------
`canvas_backend.py` 解决的是「看图决定点哪里」，需要多模态大模型，而且
pattern 题型的多种传统 CV 思路已被实测证伪 —— 这条路又贵又难。

商业打码服务走的是完全不同的路子：**你不需要看图**，只要把 sitekey 和
页面 URL 发过去，对方返回一个可用的 token。canvas 解析、坐标生成、拖拽手势、
题目推理全都在对方那边完成。这是目前唯一有把握达到高成功率的方案。

⚠️ 关于协议准确性（请读）
------------------------
本模块**无法在没有账号的前提下端到端验证**，而查公开文档时发现：

- 2captcha 现在主推的是 **v2 任务式 API**（`POST https://api.2captcha.com/createTask`
  + `getTaskResult`），鉴权参数是 `clientKey`，hCaptcha 的 sitekey/url 放在
  `task` 对象里（`websiteURL` / `websiteKey`）。
- 老式 API 是 `in.php` + `res.php`，鉴权参数 `key`；但**公开文档页里没有列出
  hCaptcha 的方法名**（reCAPTCHA 用 `method=userrecaptcha` 且 sitekey 传
  `googlekey`）。我查过的几个 hCaptcha 专页均 404。

所以下面**两套协议都实现**，并把所有厂商相关的名字做成可配置：

    SOLVER_SERVICE_API   auto（默认，先 v2 失败再退老式）/ v2 / legacy
    SOLVER_SERVICE_TASK  v2 的 task.type，默认 HCaptchaTaskProxyless
    SOLVER_SERVICE_METHOD 老式的 method，默认 hcaptcha
    SOLVER_SERVICE_SITEKEY_PARAM 老式里 sitekey 的参数名，默认 sitekey

**首次使用请先跑 `python token_service.py` 自检**，它会打印将要发出的请求体，
便于你对照服务商文档核对；返回体里的错误码也会原样带出（如 ERROR_WRONG_USER_KEY、
ERROR_UNKNOWN_METHOD），据此调上面几个变量即可。

用法
----
    export SOLVER_SERVICE_KEY=...
    export SOLVER_SERVICE_BASE=https://2captcha.com    # 可换兼容镜像
    from token_service import get_service
    token = get_service().solve_hcaptcha(sitekey, url)
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


def _extract_token(obj) -> Optional[str]:
    """
    从响应里捞出 token。

    不同协议/厂商把 token 放在不同字段（`gRecaptchaResponse` / `token` /
    `request` / `solution` 下的其它名字），所以这里不写死字段名，
    而是**递归找第一个看起来像 token 的字符串**（长度 > 30）。
    """
    if isinstance(obj, dict):
        # 常见字段优先
        for k in ("gRecaptchaResponse", "token", "request", "answer"):
            v = obj.get(k)
            if isinstance(v, str) and len(v) > 30:
                return v
        for v in obj.values():
            t = _extract_token(v)
            if t:
                return t
    elif isinstance(obj, list):
        for v in obj:
            t = _extract_token(v)
            if t:
                return t
    elif isinstance(obj, str) and len(obj) > 30:
        return obj
    return None


class TwoCaptchaService(TokenService):
    """兼容 2captcha 的 v2 任务式 API 与老式 in.php/res.php 两套协议"""

    name = "2captcha-compat"

    def __init__(self, api_key: Optional[str] = None, base: Optional[str] = None,
                 timeout: int = 240, poll_interval: float = 5.0,
                 first_wait: float = 12.0, protocol: Optional[str] = None):
        self.api_key = api_key or os.environ.get("SOLVER_SERVICE_KEY", "")
        self.base = (base or os.environ.get("SOLVER_SERVICE_BASE",
                                            "https://2captcha.com")).rstrip("/")
        self.protocol = (protocol or os.environ.get("SOLVER_SERVICE_API", "auto")).lower()
        self.task_type = os.environ.get("SOLVER_SERVICE_TASK", "HCaptchaTaskProxyless")
        self.method = os.environ.get("SOLVER_SERVICE_METHOD", "hcaptcha")
        self.sitekey_param = os.environ.get("SOLVER_SERVICE_SITEKEY_PARAM", "sitekey")
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.first_wait = first_wait
        if not self.api_key:
            raise ServiceError(
                "缺少服务 API key：请设置 SOLVER_SERVICE_KEY（可选 SOLVER_SERVICE_BASE）")

    # ---------- HTTP ----------
    def _post_json(self, url: str, payload: dict) -> dict:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())

    def _post_form(self, path: str, data: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base}{path}", data=urllib.parse.urlencode(data).encode(), method="POST")
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())

    def _get_form(self, path: str, params: dict) -> dict:
        req = urllib.request.Request(f"{self.base}{path}?{urllib.parse.urlencode(params)}")
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())

    # ---------- v2 任务式 ----------
    def _api_base(self) -> str:
        """v2 的接口主机与主站不同（api.2captcha.com）"""
        host = urllib.parse.urlparse(self.base).netloc or "2captcha.com"
        if host.startswith("api."):
            return self.base
        return f"https://api.{host}" if host else "https://api.2captcha.com"

    def build_v2_payload(self, sitekey: str, url: str) -> dict:
        return {"clientKey": self.api_key,
                "task": {"type": self.task_type,
                         "websiteURL": url,
                         "websiteKey": sitekey}}

    def solve_v2(self, sitekey: str, url: str) -> str:
        api = self._api_base()
        payload = self.build_v2_payload(sitekey, url)
        print(f"   [v2] POST {api}/createTask")
        sub = self._post_json(f"{api}/createTask", payload)
        if sub.get("errorId"):
            raise ServiceError(f"createTask 错误 {sub.get('errorCode')}: "
                               f"{sub.get('errorDescription')}")
        task_id = sub.get("taskId")
        if not task_id:
            raise ServiceError(f"createTask 未返回 taskId: {str(sub)[:200]}")
        print(f"   已提交 taskId={task_id}，等待求解...")

        deadline = time.time() + self.timeout
        time.sleep(self.first_wait)
        while time.time() < deadline:
            res = self._post_json(f"{api}/getTaskResult",
                                  {"clientKey": self.api_key, "taskId": task_id})
            if res.get("errorId"):
                raise ServiceError(f"getTaskResult 错误 {res.get('errorCode')}: "
                                   f"{res.get('errorDescription')}")
            if res.get("status") == "ready":
                tok = _extract_token(res.get("solution") or res)
                if not tok:
                    raise ServiceError(f"已 ready 但未找到 token: {str(res)[:200]}")
                return tok
            time.sleep(self.poll_interval)
        raise ServiceError(f"等待超时（{self.timeout}s）未拿到 token")

    # ---------- 老式 in.php / res.php ----------
    def build_legacy_form(self, sitekey: str, url: str) -> dict:
        return {"key": self.api_key, "method": self.method,
                self.sitekey_param: sitekey, "pageurl": url, "json": 1}

    def solve_legacy(self, sitekey: str, url: str) -> str:
        form = self.build_legacy_form(sitekey, url)
        print(f"   [legacy] POST {self.base}/in.php  method={self.method} "
              f"{self.sitekey_param}=<sitekey>")
        sub = self._post_form("/in.php", form)
        if sub.get("status") != 1:
            raise ServiceError(f"in.php 提交失败: {sub.get('request')!r}")
        task_id = sub.get("request")
        print(f"   已提交 id={task_id}，等待求解...")

        deadline = time.time() + self.timeout
        time.sleep(self.first_wait)
        while time.time() < deadline:
            res = self._get_form("/res.php", {"key": self.api_key, "action": "get",
                                              "id": task_id, "json": 1})
            if res.get("status") == 1:
                tok = str(res.get("request") or "")
                if len(tok) < 10:
                    raise ServiceError(f"返回的 token 异常: {tok[:40]!r}")
                return tok
            req = str(res.get("request", ""))
            if req == "CAPCHA_NOT_READY":
                time.sleep(self.poll_interval)
                continue
            raise ServiceError(f"res.php 返回错误: {req!r}")
        raise ServiceError(f"等待超时（{self.timeout}s）未拿到 token")

    # ---------- 统一入口 ----------
    def solve_hcaptcha(self, sitekey: str, url: str) -> str:
        if self.protocol == "v2":
            return self.solve_v2(sitekey, url)
        if self.protocol == "legacy":
            return self.solve_legacy(sitekey, url)
        # auto：先 v2，失败再退老式（两套协议的接口主机/字段都不同，互为兜底）
        try:
            return self.solve_v2(sitekey, url)
        except ServiceError as e:
            print(f"   v2 失败（{str(e)[:90]}），回退老式 in.php/res.php")
            return self.solve_legacy(sitekey, url)


def get_service(name: Optional[str] = None) -> TokenService:
    name = (name or os.environ.get("SOLVER_SERVICE", "2captcha")).lower()
    if name in ("2captcha", "2captcha-compat", "service"):
        return TwoCaptchaService()
    raise ValueError(f"未知服务: {name}")


def _selftest():
    """
    无 key 自检：打印两套协议将要发出的请求，便于对照服务商文档核对字段名。
    不会发起真实提交。
    """
    print("=== 服务后端自检（不发起真实提交）===\n")
    os.environ.pop("SOLVER_SERVICE_KEY", None)
    try:
        get_service()
        print("❌ 无 key 时应当报错")
    except ServiceError as e:
        print(f"✅ 无 key 时正确报错: {str(e)[:60]}\n")

    svc = TwoCaptchaService(api_key="DUMMY", base="https://2captcha.com")
    SK = "a5f74b19-9e45-40e0-b45d-47ff91b7a6c2"
    URL = "https://accounts.hcaptcha.com/demo"

    print("✅ v2 接口主机推导:", svc._api_base())
    print("✅ v2 请求体:")
    print("   " + json.dumps(svc.build_v2_payload(SK, URL), ensure_ascii=False))
    print("✅ 老式请求体:")
    print("   " + json.dumps({**svc.build_legacy_form(SK, URL)}, ensure_ascii=False))

    print("\n✅ token 提取的健壮性（适配不同字段名）:")
    for sample in (
        {"solution": {"gRecaptchaResponse": "X" * 40}},
        {"solution": {"token": "Y" * 40}},
        {"request": "Z" * 40},
        {"data": {"nested": {"answer": "W" * 40}}},
        {"gRecaptchaResponse": "V" * 40},
    ):
        t = _extract_token(sample)
        print(f"   {str(sample)[:52]:54s} -> {'命中' if t else '未命中'}")

    print("\n协议选择: SOLVER_SERVICE_API = auto(默认) / v2 / legacy")
    print("可调字段名: SOLVER_SERVICE_TASK / SOLVER_SERVICE_METHOD / "
          "SOLVER_SERVICE_SITEKEY_PARAM")
    print("\n⚠️ 注意：hCaptcha 的 method / task.type 具体取值**未能从公开文档核实**")
    print("   （2captcha 的 hCaptcha 专页 404，总览页只列了 reCAPTCHA 的写法）。")
    print("   首次使用请用真实 key 跑一次，按返回的错误码调整上面几个变量。")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
