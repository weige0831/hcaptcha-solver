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

## canvas 题型结构实测

用 `tools/inspect_challenge.py` 和 `tools/verify_click.py` 探查了两类 canvas 题目，
结论如下（这些是决定后续技术路线的依据）。

### 渲染结构

| 项目 | 实测值 |
|---|---|
| canvas 绘制缓冲 | **1000 x 940** |
| canvas CSS 显示尺寸 | 500 x 470 |
| `devicePixelRatio` | 1 |
| canvas 在 iframe 内的位置 | x=10, y=10 |

注意：缓冲是显示尺寸的 **2 倍**，而且这个 2x **不是** 来自 `devicePixelRatio`（它是 1）
—— hCaptcha 自己按 2x 超采样绘制。

`canvas.toDataURL()` **可用**（未被跨域图片污染），拿到的是 1000x940 的纯净题图：
没有浏览器边框、没有缩放损失、**也没有题面文字**（题面是 DOM 画在 canvas 上方的，
canvas 对应区域是透明的）。这是喂给视觉模型的理想输入。

### 坐标映射

```
canvas 缓冲坐标  ->  除以 2        ->  iframe 内 CSS 坐标
iframe 内 CSS 坐标 + canvas 在 iframe 内的偏移 + iframe 在页面中的偏移  ->  页面绝对坐标
```

实测按这套映射发出的点击，落点与预期完全一致。

### 命中判定：没有 DOM 图块

对 canvas 区域做 10x10 网格采样 `elementsFromPoint`，只出现 4 种元素栈，
且覆盖在 canvas 上方的 6 个元素**全部局限在顶部题面区域**（y=0..110 CSS），
拼图区域上方**没有任何 DOM 元素**。

也就是说：没有 `.task` 之类的图块元素可点，点击必须按坐标发。
点击由 canvas 元素接收，再由 hCaptcha 自己的 JS 做命中判定。

### 选中状态：DOM 里看不到

点选前后扫描 `[aria-checked]`、`[aria-pressed]`、`[data-selected]` 以及类名含
select/mark/active/check 的元素：**全部为 0 个**。选中状态完全保存在 JS 内部，
只体现在 canvas 重绘上。挑战 iframe 的 `window` 上也没有暴露可用的状态对象。

**所以判断「点中了没有」的唯一途径是重新抓 canvas 做像素比对。**

### 坐标点击有效（已验证）

`docs/canvas-click-challenge.png` 是一道 "Please click on the TWO icons that break the pattern"：
一堆不同旋转角度的火箭图标。"TWO" 只是个提示，实际选中上限更高。

在拼图区域按 3x3 网格依次点击，每次点击后重新抓 canvas 与基线做差分：

```
第 1 点 (0.20,0.35)  变化 2712px   ✅ 出现选中标记
第 2 点 (0.50,0.35)  变化 5418px   ✅ 又加一个（增量 ~2710px）
第 3 点 (0.80,0.35)  变化 8146px   ✅ 又加一个
第 4 点 (0.20,0.55)  变化 10864px  ✅ 又加一个
第 5~9 点            变化 10882px  ❌ 不再变化（选中已达上限）
```

`docs/canvas-click-selected.png` 是点完之后的画面：**4 个圆形叉号标记，
位置与 4 次点击的坐标逐一对应**。这证明：

1. 坐标点击被正常接收，落点精确；
2. 每次选中在 canvas 上留下约 2710 像素的固定大小标记；
3. 选中数量存在上限，达到后被静默忽略（后续点击不报错也不生效）。

### 结论：可以走坐标级操作，难点在「点哪里」

点击通道和反馈通道都打通了：按坐标点击有效，且能用「重新抓图 + 差分」确认选中。
真正剩下的是视觉推理本身——以那道火箭题为例，需要判断「哪两个图标的旋转角度
不符合整体规律」。

一个值得先试的方向：这类图标题的图标是**高对比度白色轮廓、形状相同、只有旋转不同**，
所以可以用传统 CV 而不是大模型来解——分割出每个图标 → 估计各自旋转角 →
找出偏离规律的那几个。这比接一个通用 VLM 便宜得多，也更可控。
但这个假设还没验证（规律到底是旋转序列、位置排列还是别的，需要再抓几道题确认）。

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
hcaptcha_solver.py        # 主流程：widget 加载、勾选、挑战循环、令牌提取
vision.py                 # 通用视觉层：图像质量评估、自适应预处理、类别映射、模型管理
tools/inspect_challenge.py # 探查 canvas 题型的渲染/命中判定/状态结构
tools/verify_click.py      # 验证坐标点击是否生效（抓图差分）
docs/                      # 文档配图
```

`vision.py` 不包含任何平台特定逻辑，可以直接复用（图像质量评估、
自适应增强、YOLO / 零样本检测封装）。

两个 `tools/` 脚本是上面「canvas 题型结构实测」所用的一次性探查工具，
换 sitekey / URL 时可复用：

```bash
python tools/inspect_challenge.py   # 产物在 research_out/
python tools/verify_click.py        # 产物在 research_out/click/
```

---

## 已知限制

1. **canvas 题型的「点哪里」未解决**（主要限制）—— 点击通道已验证可用，
   但还缺视觉推理来决定点击坐标。
2. **照片网格路径未经端到端验证** —— 当下没有可复现的测试环境。
3. **widget 渲染不稳定** —— 约 50% 概率需要重载，`widget_retries` 是必需的，不是保险。
4. **选中状态只能靠像素差分读** —— DOM 里没有任何标记，见上。
5. **选中数量有上限** —— 达到上限后点击被静默忽略，不会报错。
6. CPU 推理较慢 —— 本仓库在纯 CPU（torch 2.14.0+cpu）上验证，有 GPU 会快很多。
7. 频繁请求会被限流 —— 实测连续加载多次后会成片失败，需要间隔。

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
