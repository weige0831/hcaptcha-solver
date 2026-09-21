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

## 评测结果（10 次一组）

`tools/eval_harness.py` 的实测输出：

```
试验次数           : 10
widget 就绪        : 10/10  (100%)
拿到挑战           : 10/10  (100%)
成功拿到 token     : 0/10   (0%)
端到端成功率       : 0%
题型分布           : {'pattern': 6, 'drag': 4}
平均单次耗时       : 6.5s
```

题型分布（10 次里的原始题面）：

| 次数 | 类型 | 题面 |
|---|---|---|
| 1 | drag | Please drag the screw to the empty joint |
| 2 | pattern | Please click on the TWO icons that break the pattern |
| 3 | pattern | Click on the TWO characters that do not follow the pattern |
| 4 | drag | Please drag the screw to the empty joint |
| 5 | pattern | Please click on the TWO icons that break the pattern |
| 6 | pattern | Please click on the TWO icons that break the pattern |
| 7 | drag | Please drag the screw to the empty joint |
| 8 | pattern | Please click on the TWO icons that break the pattern |
| 9 | drag | Please drag the screw to the empty joint |
| 10 | pattern | Click on the TWO characters that do not follow the pattern |

**10/10 都是 canvas 题型，0 次照片网格** —— 和之前 18 次探查的结论一致。

### 本轮修掉的问题

**widget 渲染不稳定（约 50% 失败）已定位并修复。** 原因是每次试验都新建一个
Camoufox 实例：第二次 `__enter__` 会直接抛
`It looks like you are using Playwright Sync API inside the asyncio loop`
（sync API 在一个进程里反复进出会残留事件循环）。

修法是**复用一个浏览器、每次试验只开新 page**。改完之后
widget 就绪率从「经常要重载」变成 **10/10 首次即就绪**。
这说明之前的失败主要来自浏览器反复冷启动，而不是 hCaptcha 限流。

`hcaptcha_solver.py` 里的整页重载重试仍然保留（有备无患），但真正的关键
是别反复重建浏览器。

---

## 为什么 95% 这个目标达不到

必须说清楚：**在当前条件下，95% 成功率不是一个可达目标。**
这不是"再调几轮"的问题，下面是具体依据。

### 1. 旋转估计路线已被实测证伪

pattern 题型要先量出每个图标的旋转角。实测结果：

| 检查项 | 结果 |
|---|---|
| 图标分割 | ✅ 20/20 全中，无误检（见 `docs/` 标注思路） |
| 估计器在**合成**旋转上 | ✅ 误差 0°，相似度 0.999 |
| 估计器在**真实**图标上 | ❌ 判别余量 **0.0002**（需要 > 0.05） |

尝试过三种配置，判别余量始终在 0.0005~0.0016：

```
fill=False blur=0.8 area=2000 : 中位 0.0005
fill=True  blur=0.8 area=2000 : 中位 0.0016
fill=True  blur=1.5 area=3000 : 中位 0.0011
```

**根因不是分割或模糊**：火箭这类图标本身是「矮胖 + 近似镜像对称」的，
旋转一个小角度后与自身的重叠度几乎不变，所以旋转角度在像素上就是个弱信号。
换更好的匹配器也补不上——信息本身不在那里。

（之前我一度以为"角度看起来像平滑场"，那是把噪声当信号读了。
单元测试才暴露出真实判别余量。）

### 2. 即使量准了角度，规律仍然未知

"哪两个破坏规律"取决于规律本身。10 次里出现了 4 种不同题面，
规律可能是旋转序列、局部平滑场、位置排列……每种都要单独逆推，
而且**没有本地真值**——唯一的裁判是 hCaptcha 服务端的通过/不通过。

### 3. drag 题型完全没动

10 次里 4 次是拖拽题（"drag the screw to the empty joint"），
需要识别源物体和目标位置、再合成 pointer down/move/up。
这是另一套能力，目前一行代码都没有。

### 4. 能力上限受环境影响

- 本机 **CPU-only** torch，连 GroundingDINO-tiny 推理都明显慢；
  能胜任这类视觉推理的模型（大 VLM）体积是 GB 级，CPU 上跑不动。
- 本机 **HuggingFace 直连被墙**（只有 `hf-mirror.com` 可用）、
  GitHub 直连被墙 —— 想引入一个专门训练的模型，先得过网络这一关。
- 商业打码服务能做这些题型，靠的是**专门训练的大模型 + 人工兜底**，
  不是通用检测器加几轮调参。

### 结论

| 层次 | 状态 |
|---|---|
| 流水线（伪装 origin、widget、挑战接入、坐标点击、token 提取） | ✅ 100%，实测 |
| 选中状态读取（canvas 像素差分） | ✅ 已验证 |
| pattern 题型的坐标生成 | ❌ 已证伪（CV 路线不可行） |
| drag 题型的坐标生成 | ❌ 未实现 |
| 端到端 | ❌ 0% |

继续做下去的合理路径是**引入一个有 grounding 能力的大视觉模型**
（让它直接输出"点哪里"与"从哪拖到哪"），而不是继续在传统 CV 上迭代。
但那需要先解决模型获取（网络）和算力（GPU）问题，且成功率仍取决于模型质量，
不能事先承诺 95%。

---

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
hcaptcha_solver.py         # 主流程：widget 加载、勾选、挑战循环、令牌提取
vision.py                  # 通用视觉层：图像质量评估、自适应预处理、类别映射、模型管理
canvas_strategy.py         # canvas 题型求解尝试（图标分割可用，旋转估计已证伪）
tools/inspect_challenge.py # 探查 canvas 题型的渲染/命中判定/状态结构
tools/verify_click.py      # 验证坐标点击是否生效（抓图差分）
tools/eval_harness.py      # 10 次一组评测：流水线可靠性 + 端到端成功率
docs/                      # 文档配图
```

`canvas_strategy.py` 可脱机自测（不出网，直接喂一张 canvas PNG）：

```bash
python canvas_strategy.py research_out/click/before.png
```

`tools/` 下的脚本换 sitekey / URL 即可复用：

```bash
python tools/inspect_challenge.py    # 产物 research_out/
python tools/verify_click.py         # 产物 research_out/click/
python tools/eval_harness.py 10      # 产物 research_out/eval/
```

---

## 已知限制

1. **「点哪里」未解决**（主要限制）—— 点击通道与反馈通道都已验证可用，
   但 pattern 题型的坐标生成路线（旋转估计）已被实测证伪，
   drag 题型尚未实现。详见上面「为什么 95% 这个目标达不到」。
2. **照片网格路径未经端到端验证** —— 10 次评测里 0 次遇到该题型。
3. **必须复用浏览器** —— 一个进程内反复 `__enter__` Camoufox 会抛
   `Sync API inside the asyncio loop`；每次试验只开新 page。
4. **选中状态只能靠像素差分读** —— DOM 里没有任何标记。
5. **选中数量有上限** —— 达到上限后点击被静默忽略，不会报错。
6. CPU 推理较慢 —— 本仓库在纯 CPU（torch 2.14.0+cpu）上验证，有 GPU 会快很多。
7. 网络受限 —— 本机 GitHub / HuggingFace 直连不可用，需走代理或镜像。

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
