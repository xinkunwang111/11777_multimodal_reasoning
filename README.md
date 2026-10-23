# RQ1: Compositional Grounding in Qwen3-VL

> VLM 是否把属性/关系绑定到了正确的物体上？（"红色立方体 + 蓝色球 → 模型说立方体是蓝的"）

本仓库包含三部分：
1. **探针**（01–04）：从*注意力 → 表征 → 因果*三个层面分析 Qwen3-VL 的 grounding，并专门针对 Qwen3-VL 的两个对齐设计（**DeepStack**、**Interleaved-MRoPE**）做干预。
2. **Grounding Bench**（05）：一个严格的注意力 grounding 评测协议，回答三个问题：看哪些 query token、如何设阈值、怎样和 V\* GT 对应。
3. **方法**（06）：**Self-Grounded Zoom + MRoPE 对齐回注（SGZ）**。免训练，用探针找到的 grounding heads 自我定位，再放大回答。

> Qwen3-VL 没有 7B，稠密版本是 2B/4B/**8B**/32B。默认用 `Qwen/Qwen3-VL-8B-Instruct`；`--model` 可换成任意 Qwen3-VL。

---

## 1. Qwen3-VL 中与 grounding 相关的结构

| 设计 | 做了什么 | 对应的探针 / 方法 |
|---|---|---|
| ViT patch 16 + 2×2 merge | 每个 LLM 视觉 token = 32×32 像素，按 merged grid 光栅顺序排列 | bbox → token 覆盖率网格（所有实验的基础） |
| **DeepStack** | ViT 第 8/16/24 层特征经 merger 后，**加到** LLM 第 0/1/2 层输出的视觉 token 残差上 | 04 去掉全部/单层 DeepStack；02/03 早期层；记录注入范数比 |
| **Interleaved-MRoPE** | 图像 token 的 (t,h,w) 位置只进入 rotary 角度，不加到残差 | 04 collapse/shuffle/flip；03 位置探针；**06 对齐回注** |

---

## 2. 探针实验（01–04）

| # | 脚本 | 问题 | 方法 / 指标 |
|---|---|---|---|
| 0 | `00_make_synthetic.py` | 受控绑定场景 | 2–4 个形状，颜色和形状在图内唯一；**颜色互换孪生图**；left/right/above/below 关系题 |
| 1 | `01_attention_grounding.py` | 快速查看：看哪里？ | answer / 物体词 query 的逐层、逐头注意力热图（快速可视化；严格评测见 05） |
| 2 | `02_logit_lens.py` | 物体位置上的 token 说了什么？ | 逐层解码 bbox 内外 token：物体名 rank、正确属性词 vs 干扰选项 |
| 3 | `03_linear_probe.py` | 属性/位置信息存在哪？ | 线性探针，特征取 inbox/outbox/干扰物/物体词/answer token 和 ViT DeepStack 各层；另有位置 R² 探针 |
| 4 | `04_interventions.py` | 是否因果依赖？ | 目标区域均值消融 vs 随机消融；DeepStack 置零；MRoPE collapse/shuffle/**hflip**（`follow_rope`）；孪生图一致性 |

合成集上的错误归因：选了图中*另一个物体*的颜色记为 `binding`（RQ1 的目标现象）；选了图中没有的颜色记为 `hallucination`（更接近 RQ2）。

---

## 3. Grounding Bench（`05_grounding_bench.py`）

### 3.1 Query：不只看一个 token
单个 token 的热图噪声大，而且关系题里还分不清看的是哪个物体。bench 对三类 **query 集合** 取平均：

| query 集合 | token | 对应的 GT |
|---|---|---|
| `answer` | 输出选项字母的位置 | 所有目标框的并集 |
| `question` | 问题句的**全部** token | 所有目标框的并集 |
| `object` | 每个物体短语的**全部**子词，只在问题句内查找（不会匹配到选项里的同名词） | **该物体自己的框**；其他物体和干扰物作为 "other" |

`object` 对每个物体分别评测，得到组合性指标 `own_gt_other`：该物体 query 落在自己框上的注意力，是否多于落在另一个物体框上的。对关系题，就是 "cup" 是否看 cup 而不是 dog；对合成集，就是 "square" 是否看 square 而不是另一个颜色的物体。这直接对应 RQ1。

层/头的聚合方式有三种：`all`（全部层和头）、`best_layer`、`heads`（top-k **grounding heads**）。`best_layer` 和 `heads` 在**校准集**上选出，测试集上固定不变。

**Attention sink**：Qwen 系列会把大量注意力放在少数残差范数极大的视觉 token 上，与 query 无关。判定规则：残差范数大于中位数 k 倍（`--sink_k`，默认 5）的层数至少占一半，就视为 sink。带 `+ns` 后缀的 map 会去掉这些 token 再评测，两种结果都会报告。

此外提供 `grad`：对视觉 embedding（主路 + DeepStack）做 gradient×input，目标是*预测*选项的对数概率，不使用 GT。它作为非注意力的参照。

### 3.2 与 GT 对应：先不设阈值
GT 框映射到 token 网格后，得到每个 cell 的覆盖率 `cov`；`cov > 0` 的 cell 记为正样本。主要指标都**不需要阈值**：

| 指标 | 含义 | 随机水平 |
|---|---|---|
| pointing / pointing_tol | 热图峰值是否落在框内（容差 1 个 cell） | 框面积占比 |
| energy | 热图质量落在框内的比例 | 框面积占比 |
| **lift** = energy / 框面积占比 | 比均匀分布多看了几倍，不受框大小影响 | 1 |
| **AUROC** / AP | 用热图值给 cell 排序，区分框内和框外 | 0.5 |
| own_gt_other | 看自己的框多于看别的物体 | — |

### 3.3 阈值：多少算"重点在看"？
这个问题没有通用答案，所以 bench 做两件事：**同时扫描四种二值化规则**，并且**阈值只在校准集上选、在测试集上固定**。这是 WSOL 领域 MaxBoxAcc 的协议（Choe et al., CVPR 2020），避免在测试集上调阈值。

| 规则 | 含义 | 扫描范围 |
|---|---|---|
| `rel@t` | M ≥ t·max(M)（CAM 惯例） | t = 0.05…0.95 |
| `mass@p` | 覆盖 p 比例注意力质量的最少 cell 集合（"模型一半的注意力在哪"） | p = 0.1…0.9 |
| `uni@k` | M ≥ k / n_cells，即**比均匀分布多看 k 倍** | k = 1…50 |
| `otsu` | 每张图自适应，无参数 | — |

二值 mask 取**包含峰值的连通域**，算出它的外接框，再与 GT 比较：
- `box_iou`，以及 `box_acc@0.3` / `box_acc@0.5`
- `center_hit`：GT 中心落在 mask 内
- `recall`：GT 框落在预测框内的比例（裁剪需要的就是它）
- `tok_f1`：cell 级 F1
- `area`：区域占全图的比例（代价）

校准目标默认是 `tok_f1`，可用 `--calib_objective` 修改。`bench_table.csv` 同时给出两组结果：校准阈值下的测试集指标（`rule@cal:*`，诚实结果），以及在测试集上扫阈值得到的 `MaxBoxAcc(oracle)`（与文献对比用）。`threshold_sweep.png` 画出每种规则下各指标随阈值的变化。

> **V\* 的量化上限**：V\* 原图约 2246×1582，目标往往只有几十像素。在约 1M 像素的预算下，目标常常**不足一个 32px token**，此时 IoU≥0.5 在 token 分辨率下根本做不到。bench 报告 `oracle_iou`（GT 框贴合到 cell 后的 IoU）作为上限。对 V\* 来说，pointing、AUROC、lift、center_hit 和 recall 比 IoU 更有意义。这也正是要做 zoom 的原因。

### 3.4 用法
```bash
# 在 V* 内部按 qid 哈希划分：30% 校准，70% 测试
python scripts/05_grounding_bench.py --data_root data/vstar --out outputs/vstar/bench --grad
# 或者在合成集上校准、在完整 V* 上测试（没有泄漏）
python scripts/05_grounding_bench.py --data_root data/vstar --calib_dataset synthetic --calib_root data/synthetic \
       --calib_limit 150 --out outputs/vstar/bench
```
输出：
- `grounding_config.json`：grounding heads、best layer、各 map 各规则的阈值
- `bench_table.csv`、`bench_summary.json`：按类别和正确/错误分组
- `layer_{q}_{auroc,pointing,lift}.png`：raw 与去 sink 对比
- `heads_{q}_*.png`、`threshold_sweep.png`
- `vis/`：绿框是 GT，白框是校准阈值下的预测区域

---

## 4. 方法：Self-Grounded Zoom + MRoPE 对齐回注（`06_self_grounded_zoom.py`）

**动机**：探针预计会显示，模型往往*知道*目标在哪（grounding heads 的 pointing 高），但 V\* 的目标太小，在 token 分辨率下读不出属性。Qwen3-VL 的 MRoPE 又让我们能精确地告诉 LLM 一个新视图*来自原图的哪个位置*。

1. **自我定位**：在正常预算下做一次前向。读取**物体短语 token** 经**校准 grounding heads**、**去 sink** 后的注意力，按**校准阈值**二值化，取峰值连通域。关系题对每个物体各取一个区域，再取并集，保证关系的两端都在视野里。
2. **放大**：从**原始高分辨率图**裁出该区域，加 margin，并保证最小边长；小裁剪会上采样到至少一半预算。
3. **回注**：用 [全图, 裁剪图] 重新回答。普通做法里，裁剪图在 RoPE 看来是一张无关的新图。**`zoom_aligned`** 则把裁剪 token 的 (h, w) 设为它们在原图中对应区域的**分数坐标**，t 设为 t+1，即"同一位置的下一帧、更细的视图"。rotary 角度是连续的，所以分数位置有定义；测试验证了整数位置下结果与默认前向完全一致。

**对比与消融**（`summary.csv`，按类别给准确率和平均 token 数）：

| 变体 | 作用 |
|---|---|
| `base` / **`base_hi`** | 单图；`base_hi` 把全图放大到 N+M token，与所有 zoom 变体**等算力**，是最重要的对照 |
| `{src}:crop` / `:zoom` / `:zoom_aligned` | 只看裁剪图 / 双图默认 RoPE / 双图对齐 RoPE（ours） |
| 区域来源 `sgz` | 校准 heads + 物体 query + 去 sink（ours） |
| 区域来源 `attn_all` | answer token、全部层和头平均（类似 ViCrop 的 baseline） |
| 区域来源 `gt` / `random` / `grad` | GT 框（上界）/ 同尺寸随机框（对照）/ 梯度图 |

另外两个输出：
- `gating.csv`：只在 base 置信度低于 τ 时才放大。事后计算，不需要额外前向。
- `crop_quality.csv`：裁剪是否包含目标（center_in / recall / area），用来把准确率的提升归因到定位质量上。

```bash
python scripts/06_self_grounded_zoom.py --data_root data/vstar --config outputs/vstar/bench/grounding_config.json \
       --out outputs/vstar/sgz --max_pixels 1048576 --crop_pixels 1048576
```

**预期的主要结论形式**：
- `sgz:zoom_aligned` > `sgz:zoom` > `base_hi`：增益来自定位和对齐，而不是更多 token。
- `sgz` 接近 `gt`，远好于 `random`。
- bench 的定位指标（05）与 zoom 的增益（06）相关：探针结果直接转化为方法。

**创新点与相关工作的关系**（写 paper 时注意定位）：
- 用注意力裁剪做免训练放大：ViCrop（Zhang et al., ICLR 2025, *MLLMs Know Where to Look*）。
- 少数定位头：Kang et al., CVPR 2025, *Your LVLM Only Needs A Few Attention Heads for Visual Grounding*。
- 视觉 attention sink：Kang et al., ICLR 2025, *See What You Are Told*。
- 空间推理中的注意力调节：AdaptVis（Chen et al., ICML 2025）。

我们的差异在于：
- (i) **基于 Qwen3-VL 的 MRoPE 坐标对齐回注**：据我们所知是新的，而且与 04 的 `pos_hflip` 探针形成"探针 → 方法"的闭环；
- (ii) **按物体分别定位、再取并集**，服务于组合性/关系问题；
- (iii) 在**校准集上**选 heads 和阈值，并在等算力 baseline 下评测。

---

## 5. 可以验证的假设
- **H1 先定位后读取**：物体 query 在中间层 AUROC 最高；answer token 的 grounding 与答题正确率相关（05 的 correct/incorrect 分组）。
- **H2 绑定发生在视觉 token 上**：inbox 探针和 logit lens 准确率远高于 outbox；在 DeepStack 层（0–2）之后出现跃升。
- **H3 DeepStack 有助于细粒度定位**：`ds_zero_all` 会降低 direct_attributes 的准确率和注意力 grounding（04 `--attention`）。
- **H4 空间关系来自 MRoPE 还是内容**：`pos_hflip` 的 `follow_rope`；对应的方法问题是 `zoom_aligned` 是否优于 `zoom`。
- **H5 绑定错误 = 注意力错位**：binding 错误样本的 `own_gt_other` 更低。
- **H6 小目标瓶颈**：pointing 很高但答错的样本，经 SGZ 后被救回的比例最高。

---

## 6. 使用与代码结构
```bash
pip install -r requirements.txt     # transformers>=4.57；8B bf16 约 18GB 显存
python scripts/00_make_synthetic.py --out data/synthetic --n_images 300
python scripts/0{1,2,3,4}_*.py --data_root data/vstar --out outputs/vstar/<name>   # 探针
python scripts/05_grounding_bench.py ...    # bench + 校准
python scripts/06_self_grounded_zoom.py ... # 方法
pytest -q tests                             # 离线 CPU 测试（微型随机 Qwen3-VL）
```
常用参数：
- `--categories direct_attributes`
- `--limit 50`（调试用）
- `--max_pixels`：分辨率本身就是变量，建议扫 256/1024/4096 个 token

```
rq1grounding/
  data.py            V* / jsonl，prompt，bbox→token 覆盖率，选项关键词
  synthetic.py       合成绑定集（孪生图）
  model.py           加载、按预算缩放、多图输入、问题/物体 token 定位、选项打分
  hooks.py           注意力行重算（sdpa 下精确，已与 eager 对齐测试）、残差/范数记录、
                     视觉 token 消融、DeepStack 置零/记录、MRoPE 编辑
  maps.py            query 集合注意力、sink 检测、head/layer 聚合、grad×input
  grounding_eval.py  无阈值指标、四种二值化规则、连通域→框、MaxBoxAcc 校准
  zoom.py            SGZ 区域提议、裁剪、双图输入、MRoPE 对齐位置
  metrics.py / viz.py
```
测试覆盖：
- 重算注意力 = eager 注意力；logit lens 最后一层 = 模型 logits
- 消融、DeepStack、MRoPE 编辑都生效
- AUROC 与 sklearn 一致；完美 map 和均匀 map 的指标正确
- 物体 query 只在问题句内匹配
- 对齐位置正好落在区域内，且不改变后续文本的位置；整数位置复现默认前向
