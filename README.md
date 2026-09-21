# hCaptcha 解题框架（hCaptcha Solver Framework）

一个基于 Camoufox + YOLO11x / GroundingDINO 的 hCaptcha 自动化框架。

**请先读「现状」一节再决定要不要用** —— 这个仓库的定位和名字可能给你的预期不一样。

---

## 现状：框架可用，但解不了当下的题

这个仓库是一个**流程框架**，不是开箱即用的打码服务。实测结论如下。

### ✅ 已经跑通、可复用的部分

整套平台流程骨架都经过实测验证：

| 环节 | 实现方式 |
|---|---|
| 伪装页面 origin | 用路由拦截在目标 URL 上返回本地 HTML，使 origin/host 与 sitekey 匹配 |
| widget（勾选框）定位 | `iframe[title*='checkbox' i]` → `content_frame()` → `#checkbox` |
| widget 加载重试 | 实测 widget 约 50% 概率不渲染，必须整页重载重试 |
| 挑战 iframe 接入 | `iframe[title='hCaptcha challenge']` → `content_frame()` |
| 题面读取 | `.prompt-text` / `#prompt-question` |
| 令牌提取 | `h-captcha-response`（hCaptcha 同时会写 `g-recaptcha-response`） |
| 交互 | `.button-submit` / `.refresh.button` / `.display-error` |

实测运行：widget 首次加载即就绪 → 点击勾选框成功 → 连续多题均正确读出题面文本。

### ❌ 解不了的部分：hCaptcha 现在不下发照片网格题

用两套 sitekey 跑了 **18 次挑战，0 次**出现经典照片网格
（"Please click each image containing a X"）。实际下发的是：

```
Please click on the TWO icons that break the pattern
Click on the TWO characters that do not follow the pattern
Please drag the screw to the empty joint
Drag the shapes into their outlines
Click the broken spot in the chain
```

DOM 实测：`canvas=1`、`img=0`、`.task=0`。

也就是说，**整道题绘制在一个 `<canvas>` 上**：

- 没有 `.task` 图块元素可以点击
- 没有 `<img>` 子图可以裁剪
- 题型考的是「规律推理」和「拖拽」，不是「逐格目标分类」

这正是 `hcaptcha_solver.py` 里检测管线的核心假设（截图 → 切网格 → 逐格判定 → 点图块），
所以**这条路径对当下的 hCaptcha 无效**。这不是换个更强的检测模型能补的：
需要的不是更准的分类器，而是另一类能力（视觉推理、或在 canvas 上做坐标级拖拽）。

`docs/hcaptcha-canvas-challenge.png` 是其中一道 canvas 题型的截图，可以直观看到长什么样。

### 📋 代码里的分层

| 层 | 状态 |
|---|---|
| 平台流程骨架 | ✅ 实测可用 |
| 照片网格解题路径（`.task` → 截图 → 检测 → 点击 → 提交） | ⚠️ 代码完整，但当下遇不到该题型，**未做端到端验证** |
| canvas 题型 | 🚫 明确判为不支持：打印题面 → 触发 `on_unsupported` 回调 → 刷新重试 → 连续多次后带明确信息退出 |

---

## 安装

```bash
pip install -r requirements.txt
python -m camoufox fetch      # 下载 Camoufox 浏览器（约 150 MB）
```

还需要 `yolo11x.pt` 权重（约 109 MB，未纳入仓库，见 `.gitignore`）：

```bash
# 官方地址
curl -L -o yolo11x.pt \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11x.pt

# 若 GitHub 不可达，可用代理前缀
curl -L -o yolo11x.pt \
  "https://ghproxy.net/https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11x.pt"
```

零样本检测模型（`IDEA-Research/grounding-dino-tiny`，约 659 MB）会在首次运行时
自动从 Hugging Face 下载；国内网络可设镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 使用

```python
from hcaptcha_solver import solve_hcaptcha

def on_unsupported(prompt: str):
    # 遇到 canvas 题型时会被调用，可以在这里接你自己的求解器
    print("拿到题面了:", prompt)

token = solve_hcaptcha(
    sitekey="a5f74b19-9e45-40e0-b45d-47ff91b7a6c2",   # hCaptcha 官方演示页
    url="https://accounts.hcaptcha.com/demo",
    headless=True,
    on_unsupported=on_unsupported,
)
print(token)
```

或直接跑内置的演示用例：

```bash
python hcaptcha_solver.py
```

### 可调参数

`HCaptchaSolver` 的构造参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `headless` | `True` | 无头模式 |
| `humanize` | `False` | Camoufox 的拟人化输入 |
| `widget_retries` | `6` | widget 未渲染时的整页重载次数 |
| `max_unsupported` | `5` | 连续遇到多少次不支持的题型后放弃 |
| `on_unsupported` | `None` | canvas 题型回调，入参为题面文本 |

环境变量：

| 变量 | 默认 | 说明 |
|---|---|---|
| `SOLVER_DEBUG` | `1` | 打印调试信息、保存截图 |
| `SOLVER_PREPROCESS` | `1` | 启用图像自适应增强 |
| `SOLVER_SCREENSHOT_DIR` | `debug_screenshots` | 截图目录 |
| `YOLO_WEIGHTS` | `yolo11x.pt` | YOLO 权重路径 |
| `HCAPTCHA_JS` | `https://js.hcaptcha.com/1/api.js` | hCaptcha 脚本地址 |

### 路径替换成自己的求解器

遇到不支持的 canvas 题型时，`on_unsupported` 会收到题面文本。
要真正解题，可以在这个回调里接一个能处理「规律推理 / 拖拽」的模块，
再把结论回写到挑战框（canvas 题型需要坐标级点击，而不是点图块元素）。

---

## 文件结构

```
hcaptcha_solver.py   # 主流程：widget 加载、勾选、挑战循环、令牌提取
vision.py            # 通用视觉层：图像质量评估、自适应预处理、类别映射、模型管理
docs/                # 文档配图
```

`vision.py` 不包含任何平台特定逻辑，可以直接复用（图像质量评估、
自适应增强、YOLO / 零样本检测封装）。

---

## 已知限制

1. **canvas 题型无法求解**（见上，这是主要限制）。
2. **照片网格路径未经端到端验证** —— 当下没有可复现的测试环境。
3. **widget 渲染不稳定** —— 约 50% 概率需要重载，`widget_retries` 是必需的，不是保险。
4. CPU 推理较慢 —— 本仓库在纯 CPU（torch 2.14.0+cpu）上验证，有 GPU 会快很多。
5. 频繁请求会被限流 —— 实测连续加载多次后会成片失败，需要间隔。

---

## 使用范围

请仅用于你拥有或已获明确授权的环境，例如：

- 你自己站点的验证码接入测试与 QA 自动化
- 安全研究 / 学术研究（评估验证码方案的健壮性）
- hCaptcha 官方公开演示页（`accounts.hcaptcha.com/demo`）

未经授权对第三方站点使用可能违反其服务条款，在部分司法辖区还可能涉及法律问题。
使用者需自行承担合规责任。

## 许可

MIT，见 `LICENSE`。
