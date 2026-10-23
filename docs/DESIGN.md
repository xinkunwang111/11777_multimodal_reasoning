# 设计说明：RQ1 Compositional Grounding in Qwen3-VL

> 如何安装和运行见仓库根目录的 [README](../README.md)。本文档解释每个实验在做什么、为什么这样设计。

RQ1 关心的是：VLM 有没有把属性和关系**绑定到正确的物体**上。比如红方块旁边有个蓝球，模型却说方块是蓝的。

本项目的逻辑是两个探查加两个方法：

1. **探查 1：注意力找得到物体吗？答错的时候也找得到吗？**
   看问题里物体名对图像的注意力，是否落在 V\* 的 GT 框上，并分"答对 / 答错"两组统计。
2. **探查 2：这份注意力有用吗？**
   切断"文本 → 物体区域"的注意力，看答案会不会变差。同时切断一个同样大小的对照区域作比较。切物体伤得多，说明答案确实依赖它；伤得最多的那几层就是"读取层"。
3. **方法 A：放大这份注意力。**
   如果探查 1 显示"答错时注意力往往也指对了"，探查 2 显示"这份注意力被用到"，那么问题出在"看到了但用得不够"。于是在读取层里，把注意力往"注意力自己找到的区域"上再集中一些。整个方法免训练、不生成、不调工具，每题两次前向。
4. **方法 B：复现 VEA。**
   *Seeing but Not Believing*（ICLR 2026）报告了同样的现象，并给出一个免训练的方法：用一次前向读出模型在看哪里，把图像其余部分调暗，再让模型回答。我们在 Qwen3-VL 和 V\* 上完整复现它，作为已发表的对照方法。

方法 A 与两个探查是一一对应的：放大**哪里**来自探查 1（grounding heads 给出的区域），放大**哪几层**来自探查 2（读取层），为什么值得放大由两者共同说明。

> Qwen3-VL 没有 7B，稠密版本是 2B / 4B / **8B** / 32B。默认用 `Qwen/Qwen3-VL-8B-Instruct`。

---

## 1. 数据与 GT 约定

**V\* Bench**（`craigwu/vstar_bench`）的每张图配一个 JSON：

```
{"target_object": [名字...], "bbox": [[x, y, w, h] 原图像素...], "question": ..., "options": [正确项, ...]}
```

| 场景 | 处理方式（`data.py`） |
|---|---|
| `direct_attributes` | 1 个物体 |
| `relative_position` | 2 个物体，顺序为 [主体, 参照物]；每个物体的注意力只和**自己的框**比 |
| 同名的多个 box | 视为同一物体的多个实例 |
| box 越界 / 退化 | 越界部分裁掉，退化的 box 丢弃 |
| 名字和 box 数量对不上 | 记录到 `meta.gt_warning` |
| 选项顺序 | 若存在 `test_questions.jsonl`，使用官方选项顺序和答案字母（可与 lmms-eval 对比）；否则按 qid 确定性打乱 |

**统一坐标**（`geometry.py`）：全项目只用"相对原图的归一化 xyxy"。V\* 的像素 xywh、文本提示用的 Qwen 0–1000 坐标、画框用的像素坐标，都在这里转换。

**GT → token 网格**：输入图先按 32 的倍数缩放，token 网格就是精确已知的。与 GT 框有任何重叠的 32 px cell 都算目标 token，所以比一个 token 还小的物体（V\* 中常见）也至少占有一个 cell。

**合成绑定集**（`synthetic.py`）：
- 每张图 6–10 个小形状。目标的形状唯一，并且没有其他物体与它同色；目标旁边紧贴一个异色**邻居**。
- 4 个选项的来源都已知：target / neighbor（绑定错误） / in_image / absent（幻觉）。
- 每个场景配一张**颜色互换孪生图**，答案必须随之改变。
- 另有关系题。GT 格式与 V\* 一致。

---

## 2. 探查 1：注意力找得到物体吗？答错时也找得到吗？（`probe1_attention.py`）

**度量方式**：
- 对问题里每个物体，取其名字**全部 token** 对图像 token 的注意力，并去掉 attention sink（残差范数异常大、无论问什么都吸走注意力的 token）。
- 与该物体的 GT 框比较，计算：
  - **pointing**：注意力峰值是否落在物体上。这就是"注意力找到了物体"的定义，不需要阈值；
  - AUROC（0.5 = 随机）；
  - lift（框内注意力 ÷ 框面积占比，1 = 随机）；
  - own>other（看自己多于看别的物体）。
- 以上指标逐层、逐头都会计算。
- 在**校准集**上选出定位最好的 k 个 **grounding heads**，并校准"注意力图 → 区域"的阈值（保留 ≥ t·峰值、包含峰值的连通块，t 取使 IoU 最高的值）。方法 A 会用到这两样东西。

**关键输出**（测试集）：
- **注意力找到物体的比例：答对时 vs 答错时**；
- 2×2 表：注意力找到 / 没找到 × 答对 / 答错；
- 合成集上的绑定错误（答成了邻居的颜色）中，注意力其实落在目标上的比例；
- `layers.png`：逐层 pointing，分答对 / 答错两条曲线。

## 3. 探查 2：这份注意力有用吗？（`probe2_knockout.py`）

**做法**：
- 按层窗口（默认每 4 层一个窗口），切断图像之后全部文本 token 对三类 key 的注意力：
  - **目标**：GT 物体的 token；
  - **对照**：同形状、随机位置、避开所有物体的框；
  - **全图**：全部视觉 token。
- 记录 Δlog p(正确选项)。

**输出**：
- 切目标比切对照伤得多，说明答案确实在用"物体上的注意力"。summary 中的 `used` 字段给出读取窗口内两者的平均变化，以及"切目标伤得更多"的题目比例。
- 伤得最多的窗口就是**读取层**，方法 A 只在这里放大。
- 合成集上还会看：切断目标后，概率是否流向邻居的颜色。

---

## 4. 方法 A：放大注意力自己找到的区域（`method_steer.py`）

每题两次前向，不训练、不生成、不调工具：
1. **正常前向**：得到 base 答案，同时读出物体名的注意力（探查 1 的 grounding heads）→ 峰值所在的连通区域。
2. **放大前向**：输入完全相同，只在读取层（探查 2）里，把文本 token 看图像时"其他区域"注意力的 λ 比例移到这个区域上。

**对比**（`summary.csv`；行 = 怎么用这个区域，列 = 区域从哪来）：

| | attn（注意力自己找的，ours） | gt（GT 框，上界） |
|---|---|---|
| base（不用区域） | | |
| base_hi（全图放大到与 crop 相同的 token 数，等算力对照） | | |
| **steer（ours）**：只改读取层的注意力，不改输入 | | |
| crop：[全图, 该区域的高清裁剪]，即 ViCrop 式的输入放大，需要额外 token | | |
| 可选：text（区域坐标写进 prompt）、draw（在图上画框，颜色避开选项） | | |

- `by_found.csv` 把收益拆到"注意力找到 / 没找到物体"两组。按方法的逻辑，它应当主要在"找到了"的题上起作用；"没找到"的题它修不了，这正是方法的边界。
- `region_quality.csv` 给出区域的定位质量。

**层和 λ 只在校准集上选**：候选层集合是探查 2 在校准题上算出的读取窗口（取前 1 / 前 2 个），再加中间三分之一的层、全部层；λ ∈ {0.25, 0.5, 0.75}。

---

## 5. 方法 B：复现 VEA（`method_vea.py`）

**来源**：*Seeing but Not Believing*（Liu et al., ICLR 2026；代码 github.com/ZhiningLiu1998/VEA）。论文的核心发现就是探查 1 的前提：模型答错时，深层注意力仍然落在证据上。论文的方法 VEA（Visual Evidence Augmentation）据此把这份注意力"画回"输入图像。方法 A 改的是模型内部的注意力，VEA 改的是输入像素。

**步骤**：`rq1grounding/vea.py` 逐步对照官方代码实现，超参数全部用论文默认值。

| 步骤 | 做什么 | 论文默认值 |
|---|---|---|
| A 选层 | 在校准集上，逐层计算最后一个 prompt token 的注意力（各头平均）对 GT 证据 token 的 AUROC，保留最高的 10% 层 | 10%（8B 共 36 层 → 4 层） |
| B 归因 | 一次前向：取选中层的注意力平均，每个视觉 token 得到一个分数 | — |
| C 去噪 | 某个 token 比周围 8 个邻居都大 λ 倍以上，视为孤立尖峰，换成邻居的平均值 | λ = 10 |
| D 平滑 | 把分数画到输入图像的像素上，做高斯模糊，再 min-max 归一化到 [0, 1] | σ = 0.5 × 图像短边 |
| E 高亮 | 每个像素乘以 α + (1 − α) × 证据图：证据处保持原样，无证据处亮度降到 α | α = 0.5 |

模型随后在高亮图上作答，prompt 前加一句提示："Use the image, especially its highlighted region, to answer the question."

D 步的结果与官方实现（先画到像素上，再用 `scipy.ndimage.gaussian_filter`）一致，误差 < 1e-6，但这里用可分离的矩阵形式精确计算，快 20–60 倍。测试逐项对照了官方的计算方式。

**对照组**（`summary.csv`；与论文一致，另加一个 GT 上界）：

| 行 | 图像 | prompt | 作用 |
|---|---|---|---|
| `base` | 原图 | 普通 | 基线，与探查 1、方法 A 的 base 完全相同 |
| `inst` | 原图 | 带提示语 | 只加提示语：排除"只是 prompt 在起作用" |
| **`vea`** | 高亮图 | 带提示语 | **完整方法** |
| `vea_no_prompt` | 高亮图 | 普通 | 只看高亮本身的作用 |
| `vea_no_denoise` / `vea_no_smooth` / `vea_no_profiling` | 去掉 C / 去掉 D / 用全部层 | 带提示语 | 论文的三个消融 |
| `vea_gt` | 用 GT 框生成的高亮图（不去噪，否则只占一个 token 的物体会被抹掉） | 带提示语 | 证据图完全正确时的上界 |

`fixed` / `broken` 两列是相对 base 改对 / 改错的题数。

**同时复现论文的两项分析**（测试集，不需要额外前向）：
- `attribution.csv`：注意力分数对 GT 的 AUROC 和 pointing。比较固定深度区间（前 50% 层、后 50% 层、全部层）与选出的层（+ 去噪），对应论文的 attribution 表。
- `layers_rapt.png`：逐层计算证据 token 与非证据 token 的"每个 token 的注意力 ÷ 整个 prompt 的平均"，分答对 / 答错两组，对应论文的核心图。如果答错时证据曲线仍明显高于非证据曲线，就是 "seeing but not believing"。
- `summary.json` 的 `seeing_not_believing` 字段：后半层的证据 / 非证据比值，以及答对 / 答错两组的 AUROC 和峰值命中率。
- `by_found.csv`：按 VEA 的注意力峰值是否落在物体上，把准确率拆成两组。

**与原论文的差异**（都来自模型和数据，方法本身不变）：
- Qwen3-VL 的一个视觉 token 对应 32×32 px（Qwen2.5-VL 是 28×28）。
- 注意力由 `attention.py` 的 kernel 包装逐行重算，测试验证它与 eager 的 `output_attentions` 一致。因此不需要 eager 注意力，也就不需要论文提到的 80 GB 显存。
- V\* 是选择题，用选项字母的 logits 判对错；论文是生成答案后算 EM。
- 提示语改写成适合选择题的版本。论文原文 "Only use words from the picture, especially those in the highlighted region" 是为文字识别类 VQA 写的。
- 选层用同一数据集的校准集（V\* 的 30%）；论文用的是 100 道 TextVQA 题。

---

## 6. 注意力的"探针 / 放大 / 缩小"实现（`attention.py`）

**一个包装，读写两用**：
- 包装的是 LLM 层的注意力 kernel（sdpa / flash / eager 都适用），拿到 RoPE 之后的 q、k、v，**只对选中的 query 行用 fp32 重算**，其余行保持原结果。
- 不需要 eager，也不需要生成 L×L 矩阵。
- `AttentionRecorder` 读注意力（探查 1），`AttentionScaler` 改注意力（探查 2 的切断、方法 A 的放大）。VEA 的归因也用 `AttentionRecorder`。

**两种编辑语义**（作用在每个 head、每个 query 行的 softmax 概率上）：
- `scale(a)`：目标 key 上的权重 ×a 后重新归一化。a=0 是 knockout（探查 2 默认用这个），0<a<1 是缩小，a>1 是放大。
- `shift(λ)`：
  - λ>0：把"图像其他区域"注意力的 λ 比例移到目标上；
  - λ<0：把目标注意力移回其他区域；λ=−1 是图像总注意力不变的 knockout。
  - 文本注意力和图像总注意力**完全不变**，目标内部的相对权重也保持不变。

**方法 A 用 `shift` 的原因**：
- 只改变"看图像的哪里"，不改变"多依赖图像"，效果可以干净地归因到位置上；
- 它对目标大小自适应：小目标原本的注意力接近 0，乘一个固定倍数几乎没有作用。

**测试验证**：
- 记录的注意力与 eager 的 `output_attentions` 逐元素一致；
- 全层 knockout 等价于用 4D mask 屏蔽同样位置的 eager 前向（sdpa / eager 两种实现都测了）；
- `scale(1)` / `shift(0)` 不改变输出。

## 7. 防泄漏协议

- grounding heads、区域阈值、steer 的层和 λ、VEA 的层，都只在校准集上选。
- 读取层由方法 A 自己从探查 2 的逐题结果里，**只用校准题**重新计算。
- 划分方式是按 qid（孪生图和同图的关系题按 pair_id）哈希，在所有脚本里一致。

## 8. 代码结构

```
rq1grounding/
  data.py        Sample / GTObject，V* 加载（官方格式 + 官方选项顺序），合成集读取，划分，prompt
  geometry.py    坐标转换、IoU、GT→token 网格、区域 token
  synthetic.py   绑定场景生成
  model.py       加载，按预算缩放，(多图) 输入，问题 / 物体 token 定位，选项打分
  attention.py   AttentionRecorder / AttentionScaler（scale、shift）
  grounding.py   注意力 query 组、sink 去除、指标、grounding heads、注意力图→区域
  cues.py        区域的用法：steer / crop / text / draw
  vea.py         VEA 复现：选层、归因、去噪、平滑、高亮
  hooks.py       残差范数与 sink 检测、DeepStack 消融、MRoPE 位置编辑、logit lens（附录用）
  cli.py, viz.py
scripts/  check_setup.py  make_synthetic.py  probe1_attention.py  probe2_knockout.py  method_steer.py  method_vea.py
          collect_results.py
          appendix/{logit_lens, linear_probe, qwen3vl_design}.py
tests/    test_core.py  test_model.py  test_pipeline.py（模拟 V* 格式的端到端测试）
```

`pytest -q tests` 全部离线运行（CPU 即可）。

## 9. 相关工作与定位
- **Seeing but Not Believing / VEA**（ICLR 2026）：答错时深层注意力仍指对证据（与探查 1 的前提相同），据此调暗证据之外的像素再回答。我们完整复现它，作为方法 B。
- **ViCrop**（ICLR 2025）："MLLMs know where to look"，答错时注意力也常指对，并据此裁剪放大。我们在 Qwen3-VL 上检验这一前提（探查 1），并把它作为 crop 对照组。
- **Kang et al.**（CVPR 2025）：少数头即可定位。对应 grounding heads。
- **See What You Are Told**（ICLR 2025）：attention sink。
- **PAI**（ECCV 2024）：全图统一放大图像注意力。
- **ControlMLLM**（NeurIPS 2024）：在用户给定的区域上，用测试时优化放大注意力。这是与方法 A 最接近的工作。
- **Cross-modal information flow**（CVPR 2025）：attention knockout 方法。

方法 A 的差异：
- 区域来自**模型自己的注意力**，不需要用户给框，也不需要优化；
- 只在**探查确认的读取层**放大；
- 用 `shift`，不改变图像总注意力；
- 与 crop 做了等算力比较。
- 与 VEA 相比：VEA 用最后一个 token 的注意力，每题一张证据图，改的是输入像素；方法 A 用问题中物体名 token 的注意力，按物体给出区域，只改读取层内部的注意力。

## 10. 注意事项
- 两个方法是否值得用，都取决于探查 1 的结果。如果答错时注意力也大多没找到物体，放大或高亮就没有意义，这时瓶颈在定位或分辨率（crop 可能更有效）。这本身也是一个结论。
- V\* 原图约 2246×1582，目标常常不足一个 token。可以改 `MAX_PIXELS` 多跑几组，看分辨率如何影响"找到"和"答对"。
- VEA 的 σ = 0.5 × 图像短边非常宽。V\* 的目标常常不到一个 token，高亮会变成一大片渐变。如果 VEA 在 V\* 上效果不明显，可以用 `--sigma 0.1` 另跑一组做敏感性分析（换一个 `--out` 目录）。
- 测试用的是随机初始化的微型模型，只验证代码正确性。
