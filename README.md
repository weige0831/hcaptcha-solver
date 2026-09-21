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

### 5. 市面方案调研：结论是「必须用托管的大模型」

对最有代表性的开源项目 hcaptcha-challenger（QIN2DIM，一直在维护到 2026-08）
做了调研：

- 它的挑战类型分类是：`image_label_binary`（ResNet ONNX）、
  `image_label_area_select`（YOLOv8 ONNX 检测/分割）、
  `image_label_multiple_choice`（ViT ONNX 零样本）、`image_drag_drop`。
- 但**最近的提交已经把默认模型换成 Gemini 3.7 Flash**
  （"feat(models): update default models to Gemini 3.7 Flash"），
  另外还有 "Gemini 3"、"dynamic thinking budget"、"retry logic for spatial challenges"。
- 它自带的 ONNX 模型是 2 月的产物，针对的是旧版挑战分类。

也就是说：**这个领域里最专注、维护最久的项目，最后也把难题交给托管的大型
多模态 LLM 去解**。这和我本地三轮实验的结论一致。

### 6. 本地小模型（Moondream2）实测：已排除

第一次测出的是乱码，但那是我这边的环境问题，不能当作能力结论。这轮把它修好并
测出了真实数字。

**先修环境（记录以备复现）：**

| 问题 | 原因 | 处理 |
|---|---|---|
| 输出退化成复读（`1KeKeKeKe...`） | 模型 config 声明 `transformers_version: 4.52.4`，而主环境是 5.17.0 | 建隔离环境 `.venv-vlm` 装 transformers 4.52.4（不动主环境，避免影响 GroundingDINO） |
| `FileNotFoundError: transformers_modules/moondream2/layers.py` | 动态模块缓存只拷进去 9/17 个 .py 文件 | 把本地模型目录的 `*.py` 同步进 `~/.cache/huggingface/modules/transformers_modules/moondream2/` |

**修好之后的真实能力与耗时**（图缩到 448x313，CPU）：

| 查询 | 回复 | 耗时 |
|---|---|---|
| How many rocket icons are in this image? | "There are fifteen rocket icons in the image." | 106.1s |
| Which rocket icon is different from the others? | "Top center" | 115.5s |
| `point("rocket that does not follow the pattern")` | **返回 15 个点**（前3: 0.42/0.82, 0.50/0.71, 0.59/0.82） | 185.4s |

模型是**真的能看懂图**——能正确数出数量（15 个），说明之前那次乱码确实只是版本问题。

但两点让它无法用于本题：

1. **接口语义不对**。`point()` 是「实例定位」原语：给它一个名词，它返回该物体的
   **全部**实例——所以问「不符合规律的那个」它返回了全部 15 个点。
   它没有「推理出异常项」这个能力。真正的推理走 `query()`，而 `query()` 给出的是
   "Top center" 这种**模糊散文**，不是可点击的两组坐标。
2. **耗时不可接受**。448px 的小图单次 106~185s（CPU）。要组成「先定位 15 个图标、
   再两两比较」的策略，需要约 190 次查询 —— 完全不现实；即便单次查询，
   也可能撞上验证码的超时。

**一个有意思的交叉验证**：模型对「哪个不一样」答 "Top center"，
而我之前的像素分析里，偏离最大的也正是顶部中央那枚（bbox 139px vs 其余约 80px，
相似度中位 0.420）。两个完全不同的方法指向同一个图标，说明它确实视觉上显著。
但那只说明「它看起来不一样」，不等于「它就是题目要的那两个」——
而后者需要真值，本机没有。

（顺带记两个环境坑：`huggingface_hub` 的 `snapshot_download` 下 3.6GB 大文件会
卡死，改用 `curl -C -` 续传能跑到 45MB/s；`models/` 与 `.venv-vlm/` 都已加入
`.gitignore`。）

---

## 最终结论

**在「本机、CPU、无外网 API」的条件下，95% 成功率无法达成。** 依据是上面六项
彼此独立的证据，而不是单点失败：

| 尝试 | 结果 |
|---|---|
| 1. 掩膜重叠 + 旋转扫描 | ❌ 判别余量 0.0005 |
| 2. 转角函数（曲率互相关） | ❌ 判别余量 0.0082（阈值需 >0.05） |
| 3. 两两相似度矩阵聚合 | ❌ 找到的是分割噪声（参考图标 bbox 139px vs 其他 80px、星云区污染），不是题目异常 |
| 4. drag 题型 | ✅ 动作已实现并 live 验证（探测命中 62649px，方块随光标移动）；缺的同样只是坐标 |
| 5. 市面方案 | ❌ 最专注的开源项目已改用托管 LLM（Gemini 3.7 Flash） |
| 6. 本地小 VLM | ❌ 输出退化 + 80~180s/次，速度上不可行 |

### 能达到 95% 的唯一现实路径

**接入一个托管的大型多模态模型**（Gemini / GPT / Claude 等），让它直接输出
"点哪里"和"从哪拖到哪"。所需条件：

1. 一个多模态 LLM 的 **API key**（会产生费用）——本机没有，我无法代持；
2. 或者一块 **GPU** + 足够大的本地 VLM（7B 以上，且需解决上面那个
   transformers 兼容问题）；
3. 以及每道题超时的预算（云端 API 单次几秒，可接受；CPU 本地推理不可接受）。

框架这一侧我已经备好了：`canvas_strategy.py` 的接口就是"给一张 canvas 图，
返回要点击的坐标列表"，换成云端模型只需替换坐标生成部分，
点击、提交、像素差分验证、评测harness 都已实测可用。


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

1. **「点哪里 / 从哪拖到哪」未解决**（主要限制）—— 点击与拖拽的**执行**都已验证
   可用，缺的是决定坐标的后端。pattern 题型的传统 CV 路线已量化证伪；
   本地小模型已实测排除；drag 题型的动作已实现并做了机制级验证。
2. **drag 题型的 live 验证 ✅ 已完成** —— 动作已实测被 hCaptcha 的 canvas 接收。
   过程曲折，记录如下（含两次错误结论的更正）：

   - **拖拽题很稀有**：跨 4 组配置 + 两轮长扫描，共 61 题只有 1 道拖拽
     （58 pattern / 2 other / 1 drag），约 1.6%。所以复现一次需要跑上百轮刷新。
   - **踩过的坑一：起手点落在空白处**。第一次拿到拖拽题时从画面正中起手，
     得到 0 像素变化，我判断为"手势未被接收"——**这个结论是错的**。
     看 before.png 才发现那类题的布局是「左侧面板放方块 + 右侧面板虚线轮廓」，
     画面正中是右侧面板空白，根本没抓住东西。
   - **踩过的坑二：按颜色找方块不可靠**。改成检测米黄色方块后，遇到另一种布局
     （全幅场景、木条堆在中间、右缘圆形 Move 手柄），检测器把暖色**背景**当成了
     方块；后来又遇到方块是**浅蓝色**的版本。颜色/位置都不能写死。
   - **最终解法：与布局无关的抓取探测**。不猜方块在哪，而是对若干候选起手点
     逐个「按住 → 小幅移动 → 按住不放时抓帧比对」。只要任意一个候选让画面变化，
     就证明手势被接收。实测结果：

     ```
     探测 [中心]               起手(500,517) 按住中变化=0      松手后=0      —
     探测 [中部偏右下(木条堆)]  起手(600,583) 按住中变化=0      松手后=0      —
     探测 [变体A 上块]         起手(135,376) 按住中变化=62649  松手后=59081  ✅
     ```

     命中在轮次 61、题面 "Drag the shapes into their outlines"。变化区域
     x[46..333] y[253..525]，正好是方块原位加移动轨迹。对照图
     `research_out/drag_hunt/compare_round61.png` 可直观看到方块跟着光标移动、
     Move 手柄随动；松手后仍有 59081 px 变化，说明不是回弹。

   结论：**点击与拖拽的执行层（`canvas_actions.py`）均已 live 验证可用**。
   工具：`tools/hunt_drag.py`（持久猎取 + 探测验证）。
3. **照片网格路径未经端到端验证** —— 实测里 0 次遇到该题型。
4. **必须复用浏览器** —— 一个进程内反复 `__enter__` Camoufox 会抛
   `Sync API inside the asyncio loop`；每次试验只开新 page。
5. **选中状态只能靠像素差分读** —— DOM 里没有任何标记。
6. **选中数量有上限** —— 达到上限后点击被静默忽略，不会报错。
7. CPU 推理较慢 —— 大模型本地推理单次 106~185s（实测），不适用于实时解题。
8. 网络受限 —— 本机 GitHub / HuggingFace 直连不可用，需走代理或镜像。

### 两个环境坑（踩过，记录以免重复）

- **`page.evaluate` 跑在隔离世界**：能读 DOM，但读不到页面里定义的 JS 全局变量
  （`window.xxx` 恒为 `undefined`）。要传递数据请走 DOM 节点
  （例如把 JSON 写进某个元素的 `textContent` 再读）。
  这也意味着早先「`window.hcaptcha` 是 undefined」的观察不能作为
  「hCaptcha 没暴露 API」的证据。
- **`page.set_content()` 不执行内联脚本**：实测只写入 DOM，页面停在 `about:blank`。
  要执行脚本请用路由拦截在一个真实 URL 上提供服务。

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
