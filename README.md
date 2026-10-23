# RQ1: Compositional Grounding in Qwen3-VL — 找到了，但读对了吗？

RQ1 关心的是：VLM 有没有把属性和关系**绑定到正确的物体**上。比如红方块旁边有个蓝球，模型却说方块是蓝的。

本项目把"答错"拆成两类：
- **没找到**物体（定位失败）；
- **找到了但读错了**（绑定 / 读取失败）。

探查部分回答三个问题："找到了吗？""在哪几层读？""Qwen3-VL 的 DeepStack 和 MRoPE 起什么作用？"。方法部分针对"找到了但读错"：**回答时把模型的注意力移到它自己框出的物体上**。整个方法免训练、不是 agent、不调用工具，每题固定两步推理。

> Qwen3-VL 没有 7B，稠密版本是 2B / 4B / **8B** / 32B。默认用 `Qwen/Qwen3-VL-8B-Instruct`，`MODEL=...` 可替换。

---

## 0. 一键运行

```bash
pip install -r requirements.txt
QUICK=1 bash run_all.sh     # 试跑：每个数据集 20 题，先确认整条流程能通
bash run_all.sh             # 全部实验，结果汇总在 outputs/Qwen3-VL-8B-Instruct/REPORT.md
```

| 步骤（`STEPS`） | 脚本 | 依赖 |
|---|---|---|
| `data` | `scripts/make_synthetic.py`；V\* 不存在时自动下载 | — |
| `p1` | `scripts/probe1_find.py` | — |
| `p2` | `scripts/probe2_read.py` | — |
| `p3` | `scripts/probe3_design.py` | p1 |
| `method` | `scripts/method_steer.py` | p1、p2 |
| `appendix`（默认不跑） | `scripts/appendix/{logit_lens,linear_probe}.py` | — |
| `report` | `scripts/collect_results.py` → `REPORT.md` | — |

每个步骤分别在 V\* 和合成集上跑一遍（`DATASETS="vstar synthetic"`）。常用环境变量：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `MAX_PIXELS` | 1048576 | 全图预算，即 1024 个视觉 token（每个 32×32 px） |
| `CROP_PIXELS` | 262144 | 裁剪对照组额外的 256 个 token |
| `CALIB_FRAC` | 0.3 | 校准集比例；或用 `CALIB_FROM_SYNTH=1` 让 V\* 用合成集校准 |
| `LIMIT` | 全部 | 每个数据集的题数上限 |
| `QUICK` | 0 | 1 = 试跑模式（每个数据集 20 题、小合成集） |
| `FORCE` | 0 | 1 = 忽略 `.done`，全部重跑 |
| `KEEP_GOING` | 1 | 0 = 遇到第一个失败就停 |
| `VERBOSE` | 0 | 1 = 日志同时打印到终端 |

完成的步骤会写 `.done` 标记，中断后重跑同一条命令即可续跑。日志在 `$OUT/logs/`。访问不了 HF 时：`export HF_ENDPOINT=https://hf-mirror.com`。

---

## 1. 数据与 GT 约定（`rq1grounding/data.py`, `geometry.py`）

**V\* Bench**（`craigwu/vstar_bench`）的每张图配一个 JSON：

```
{"target_object": [名字...], "bbox": [[x, y, w, h] 原图像素...], "question": ..., "options": [正确项, ...]}
```

| 场景 | 处理方式 |
|---|---|
| `direct_attributes` | 1 个物体 |
| `relative_position` | 2 个物体，顺序为 **[主体, 参照物]**，与问题 "Is the A on the left or right side of the B?" 一致 |
| 同名的多个 box | 视为同一物体的多个实例，评测时取最佳实例 |
| 名字和 box 数量对不上 | 记录到 `meta.gt_warning` |
| box 越界 / 退化 | 越界部分裁掉，退化的 box 丢弃 |
| 选项顺序 | 若存在 `test_questions.jsonl`，使用其中的**官方选项顺序和答案字母**（结果可与 lmms-eval 对比）；否则按 qid 确定性打乱（JSON 里正确项总是第一个） |

**统一坐标**：全项目只用"相对原图的归一化 xyxy"。转换都在 `geometry.py` 里：

| 来源 / 去处 | 格式 | 函数 |
|---|---|---|
| V\* 标注 | 原图像素 `[x, y, w, h]` | `from_xywh_px` |
| Qwen3-VL 输出 / 文本提示 | 0–1000 相对坐标 `[x1, y1, x2, y2]`（官方 cookbook 的格式） | `from_qwen` / `to_qwen` |
| 画框 | 模型输入图的像素 | `to_px` |

**token 网格**：输入图先按 32 的倍数缩放，处理器就不会再缩放，所以 token 网格是精确已知的。
- `coverage()` 计算每个 32 px cell 被 GT 覆盖的比例。
- 与 GT 有任何重叠的 cell 都算"目标 token"，所以比一个 token 还小的物体也至少占有它所在的那个 cell（V\* 中很常见）。

**合成绑定集**（`synthetic.py`）：
- 每张图 6–10 个小形状，形状和颜色会重复。目标的形状唯一，并且没有其他物体与它同色。
- 目标旁边紧贴一个异色的**邻居**，这是属性泄漏的经典设置。
- 4 个选项的来源都已知：target / neighbor（绑定错误） / in_image / absent（幻觉）。
- 每个场景有一张**颜色互换孪生图**：布局相同、选项相同，但答案必须改变。
- 关系题在两个唯一形状之间出题。
- GT 格式与 V\* 完全一致，同一张图的题目在划分时总落在同一侧。

---

## 2. 探查

### Probe 1：找到了吗？（`probe1_find.py`）

对问题里的每个物体分别测两种"找到"：
- **explicit（嘴上说的）**：在答题用的同一张图上，让模型 `Locate the <物体> ...`。解析它输出的 `bbox_2d`，计算框中心是否落在 GT 内，以及 IoU。
- **implicit（眼睛看的）**：问题中物体名的**全部 token** 对图像 token 的注意力（去掉 attention sink）。逐层、逐头计算 pointing（峰值是否落在 GT 内）、AUROC、lift，以及 own>other（看自己的框多于看别的物体）。
  - 在校准集上选出 k 个 **grounding heads**；
  - 同时校准一个"注意力图 → 框"的阈值：保留 ≥ t·峰值的、包含峰值的连通块，t 取使 IoU 最高的值。

然后报告：
- **found × correct 2×2 表**，其中关键数字是"答错的题里，有多大比例其实已经找到了物体"；
- **说与看是否一致**：注意力峰值是否落在模型自己输出的框里；
- 合成集上的错误类型（绑定 vs 幻觉）和孪生图一致性。

### Probe 2：在哪几层读物体？（`probe2_read.py`）

按层窗口（默认每 4 层一个窗口）切断文本 query 对三类 key 的注意力：
- **target**：GT 物体的 token；
- **control**：同形状、随机位置、避开所有物体的框，token 数相当；
- **image**：全部视觉 token。

记录 Δlog p(正确选项)。在某个窗口里，切 target 比切 control 伤得多得多，那么这个窗口就是"读取层"。方法会在这些层里做 steering。

query 分三组：`all`（图像之后的全部文本）、`last`（答案位置）、`object`（物体名）。合成集上还会记录：切断目标后，概率是否流向邻居的颜色（绑定泄漏）。

### Probe 3：DeepStack / MRoPE 帮的是"找"还是"读"？（`probe3_design.py`）

在以下条件下重测 Probe 1 的"找"（显式框和注意力）和答题准确率：
- 去掉全部或某一层 DeepStack；
- MRoPE collapse（去掉 2D 布局）；
- MRoPE 左右 / 上下镜像（像素不动）。

镜像条件下报告三个 follow_rope 比例：**模型输出的框**、**注意力峰值**、**左右答案**有多少跟着位置编码翻转。翻得多，说明模型的空间感来自 MRoPE；翻得少，说明来自视觉内容（ViT 自带的位置信息）。只在镜像后与原框不重叠的物体上统计。

附录（`scripts/appendix/`）：logit lens 和线性探针，回答"属性信息在不在物体 token 上"。

---

## 3. 方法：先框后看（`method_steer.py`）

每题固定两步：
1. **找**：模型自己框出问题里的物体（就是 Probe 1 的 explicit grounding）。
2. **答**：输入原图、问原问题，但在 Probe 2 找到的读取层里，把文本 token 的注意力移到框内的视觉 token 上。

**对照网格**（`summary.csv`；行 = 怎么告诉模型，列 = 框从哪来）：

| 告诉方式 \ 框来源 | self（模型自己框的） | attn（Probe 1 grounding heads） | gt（上界） |
|---|---|---|---|
| base（不给框） | | | |
| base_hi（全图放大到等 token 数，作为 crop 的等算力对照） | | | |
| text：坐标写进 prompt（Qwen3-VL 原生 0–1000 格式） | | | |
| draw：在图上画框，颜色避开所有选项，并且画在物体**外侧**，不盖住物体 | | | |
| crop：[全图, 原图高清裁剪]，输入侧放大，需要额外 token | | | |
| **steer（ours）**：不改输入，只改读取层的注意力 | | | |

怎么解读：
- "text 无效而 steer 有效"，说明模型文字端的坐标没有接到视觉读取上，这就是 cross-modal alignment gap，本身是一个 RQ1 结论。
- `by_found.csv` 把收益拆到"自己框对了 / 框错了"的题上，与 Probe 1 的 2×2 表对应。
- `box_quality.csv` 给出每种框来源的定位质量，以及没给出框的比例（这些题回退到 base 的答案）。

---

## 4. 注意力的"探针 / 放大 / 缩小"是怎么做的（`attention.py`）

**一个包装，两种用途**：
- 包装的是 LLM 层的注意力 kernel（sdpa / flash / eager 都适用）。它拿到的是 RoPE 之后的 q、k 和 v，**只对选中的 query 行用 fp32 重算**，其余行保持原 kernel 的结果。
- 不需要 eager，也不需要生成 L×L 的注意力矩阵（V\* 输入有 1–4k 个 token）。
- `AttentionRecorder` 读注意力（探针），`AttentionScaler` 改注意力（放大 / 缩小 / 切断）。

**两种编辑语义**（作用在每个 head、每个 query 行的 softmax 概率上）：

| 编辑 | 操作 | 适用场景 |
|---|---|---|
| `scale(a)` | 目标 key 上的权重 ×a 后重新归一化。a=0 是经典的 attention knockout，0<a<1 是缩小，a>1 是放大 | 副作用：图像总注意力和文本注意力也会随之改变 |
| `shift(λ)` | λ>0：把"图像其他区域"注意力的 λ 比例移到目标上；λ<0：把目标注意力的 \|λ\| 比例移回其他区域（λ=−1 是**保持图像总注意力不变的 knockout**） | 文本注意力和图像总注意力**完全不变**，目标内部的相对权重也保持不变 |

为什么方法用 `shift` 而不用 `scale`：
- `shift` 只改变"看图像的**哪里**"，不改变"**多**依赖图像"，所以效果可以干净地归因到位置上。
- 它对目标大小自适应：小目标原本的注意力质量接近 0，乘上固定的 a 几乎没有作用；`shift` 按比例移入质量，不受这个问题影响。
- λ 和层集合在**校准集**上选择。候选层集合包括 Probe 2 在校准集上算出的读取窗口、中间三分之一的层、全部层。

**测试验证**（`tests/test_model.py`）：
- 记录的注意力与 eager 的 `output_attentions` 逐元素一致；
- 全层 knockout 的结果与"用 4D mask 屏蔽同样位置的 eager 前向"一致（sdpa 和 eager 两种实现都测了）；
- `scale(1)` / `shift(0)` 不改变输出。

---

## 5. 防泄漏协议

所有需要"选"的东西都只在校准集上选，测试集上只评测：
- grounding heads、注意力 → 框的阈值（Probe 1）；
- steer 的层和 λ（方法）。读取层由方法自己从 Probe 2 的逐题结果里，**只用校准集的 qid** 重新计算。

划分方式是按 qid（孪生图按 pair_id）哈希，在所有脚本里一致。也可以用 `--calib_dataset synthetic` 换成用另一个数据集校准。

---

## 6. 代码结构

```
rq1grounding/
  data.py        Sample / GTObject，V* 加载（官方格式 + 官方选项顺序），合成集读取，划分，prompt
  geometry.py    坐标转换（像素 xywh / 归一化 xyxy / 0–1000），IoU，GT→token 网格，区域 token
  synthetic.py   绑定场景生成（邻居、孪生图、关系题）
  model.py       加载，按预算缩放，多图输入，问题 / 物体 token 定位，选项打分，手写贪心解码（可配合位置编辑）
  attention.py   AttentionRecorder / AttentionScaler（scale、shift）
  hooks.py       残差 / 范数记录与 sink 检测，DeepStack 消融与记录，MRoPE 位置编辑，logit lens
  grounding.py   显式定位（prompt + 解析）、隐式定位（query 组注意力、指标、选头、图→框）
  cues.py        告诉模型位置的四种方式：text / draw / crop / steer
  cli.py         参数、划分、Probe 1 / 2 结果的读取（供下游使用）
  viz.py
scripts/  make_synthetic.py  probe1_find.py  probe2_read.py  probe3_design.py  method_steer.py
          collect_results.py  appendix/{logit_lens,linear_probe}.py
tests/    test_core.py（无模型） test_model.py（微型 Qwen3-VL） test_pipeline.py（模拟 V* 端到端）
```

`pytest -q tests` 全部离线运行，CPU 上大约 1 分钟。

---

## 7. 相关工作与定位
- **ViCrop**（ICLR 2025）：模型答错时注意力往往仍然指对地方，并用注意力裁剪放大。这对应我们的 Probe 1 和 crop 对照组。
- **Kang et al.**（CVPR 2025）：少数注意力头就能完成定位。对应 grounding heads。
- **See What You Are Told**（ICLR 2025）：视觉 attention sink。对应 sink 去除。
- **PAI**（ECCV 2024）：全图统一放大图像注意力。
- **ControlMLLM**（NeurIPS 2024）：在用户给定的区域上，用测试时优化放大注意力。这是与本方法最接近的工作。
- **Cross-modal information flow**（CVPR 2025）：用 attention knockout 分析 MLLM 中的信息流。对应 Probe 2 的方法。

本方法的差异：
- 用**模型自己的框**；
- 只在**探查找出的读取层**编辑；
- 用 **shift**，不改变图像总注意力；
- 不需要优化；
- 在同一个对照网格里与 text / draw / crop 公平比较（crop 有等算力对照）。

## 8. 注意事项
- **分辨率**：V\* 原图约 2246×1582，目标常常不足一个 32 px token。分辨率本身就是变量，可以改 `MAX_PIXELS` 多跑几组。
- **sink 阈值**：`--sink_k` 默认 5，表示残差范数 > 5×中位数、且在至少一半层里成立。实际 sink 比例写在 Probe 1 的 summary 里。
- **测试用的是随机初始化的微型模型**，只验证代码正确性，测试里的数字没有实验意义。
