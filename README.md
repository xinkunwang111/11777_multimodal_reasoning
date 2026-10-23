# RQ1: Compositional Grounding in Qwen3-VL

> VLM 是否把属性/关系绑定到了正确的物体上？（"红色立方体 + 蓝色球 → 模型说立方体是蓝的"）

本仓库是 RQ1 的探针工具包：在 **V\* Bench**（高分辨率 visual search，带 GT bbox）和一个 **可控合成绑定数据集** 上，从 *注意力 → 表征 → 因果* 三个层面检查 Qwen3-VL 的 grounding，并专门针对 Qwen3-VL 的两个模态对齐设计（**DeepStack**、**Interleaved-MRoPE**）做干预。

> 注：Qwen3-VL 没有 7B，稠密版本是 2B/4B/**8B**/32B。默认用 `Qwen/Qwen3-VL-8B-Instruct`（最接近 7B）；`--model` 可换成任意 Qwen3-VL（包括 30B-A3B MoE）。

## 1. Qwen3-VL 中和 grounding 相关的结构

| 设计 | 做了什么 | 对应的探针 |
|---|---|---|
| ViT patch 16 + 2×2 merge | 每个 LLM 视觉 token = 32×32 像素，按 merged grid 光栅顺序排列 | bbox → token 覆盖率网格（所有探针的基础） |
| **DeepStack** | ViT 第 8/16/24 层特征经 merger 后，**加到** LLM 第 0/1/2 层输出的视觉 token 残差上 | 4: 去掉全部/单层 DeepStack；2/3: 早期层的 logit lens 和线性探针；记录注入范数比 |
| **Interleaved-MRoPE** | 图像 token 的 (t,h,w) 位置只进入 rotary 角度，不加到残差里 | 4: collapse/shuffle/hflip/vflip 位置编码；3: 位置线性探针 |

## 2. 实验设计

| # | 脚本 | 问题 | 方法 | 关键指标 |
|---|---|---|---|---|
| 0 | `00_make_synthetic.py` | 构造受控的绑定场景 | 2–4 个形状，形状和颜色在图内都唯一；每张图配一个**颜色互换的孪生图**；另有 left/right/above/below 关系题 | — |
| 1 | `01_attention_grounding.py` | 模型**看**哪里？ | 每层每头的 answer token 与问题里物体词 token 对图像 token 的注意力（在 sdpa 下用 hook 精确重算，无需 eager） | in-box 占比、in-box/面积比（1=随机）、pointing-game 命中率、熵；正确 vs 错误；目标 vs 干扰物；**grounding heads** 排名 |
| 2 | `02_logit_lens.py` | 物体位置上的视觉 token **说了**什么？ | 每层用 final norm + lm_head 解码 bbox 内 vs bbox 外的 token | 物体名的最佳 rank；**绑定准确率**：bbox 内 token 上正确选项词（如 "red"）是否高于干扰选项 |
| 3 | `03_linear_probe.py` | 属性/位置信息**存在哪**？ | 交叉验证线性探针（按图分组，孪生图不跨折）：特征取 inbox / outbox / 干扰物 / 物体词 token / answer token，以及 ViT 侧 DeepStack 各层和 merger 输出 | 各层探针准确率；位置探针 R²（MRoPE 下残差是否显式编码 2D 位置） |
| 4 | `04_interventions.py` | 是不是**因果**地依赖这些机制？ | 对目标区域视觉 token 做均值消融（对照：同数量随机 token）；去掉 DeepStack；MRoPE collapse/shuffle/flip | 准确率与 Δp(correct)；**follow_rope**（只翻转位置编码时，左右答案跟着翻的比例）；孪生图一致性；可选：各干预下的注意力 grounding |

**合成数据上的错误归因**：答错时，如果选的是图中*另一个物体*的颜色 → `binding` 错误（RQ1 要找的现象）；如果是图中不存在的颜色 → `hallucination`（更接近 RQ2）。`layer_ans_target_vs_distractor.png` 用来看绑定错误时注意力是否落在了干扰物上。

### 可以验证的假设（示例）
- **H1 先定位后读取**：物体词 token 在中间层指向目标；answer token 的 grounding 峰值层与正确率相关（`hit_rate_best_layer_correct_vs_incorrect`）。
- **H2 绑定发生在视觉 token 上**：inbox 探针/logit lens 准确率 ≫ outbox；DeepStack 层（0–2）之后出现跃升。
- **H3 DeepStack 有助于细粒度定位**：`ds_zero_all` 会降低 V\* direct_attributes 的准确率和注意力 in-box 占比，而 `ablate_random` 不会。
- **H4 空间关系来自 MRoPE 还是内容**：`pos_hflip` 的 follow_rope 很高 → 左右判断读的是 rotary 位置；接近 0 → 读的是 ViT 内容特征（ViT 自身带绝对位置和 2D RoPE）。
- **H5 绑定错误 = 注意力错位**：binding 错误的样本在干扰物 bbox 上的注意力高于目标。

## 3. 使用

```bash
pip install -r requirements.txt          # transformers>=4.57（Qwen3-VL），需要 GPU；8B bf16 约 18GB

# V*（不存在时自动从 HF 下载 craigwu/vstar_bench）
python scripts/01_attention_grounding.py --data_root data/vstar --out outputs/vstar/attn
python scripts/02_logit_lens.py         --data_root data/vstar --out outputs/vstar/lens
python scripts/03_linear_probe.py       --data_root data/vstar --out outputs/vstar/probe
python scripts/04_interventions.py      --data_root data/vstar --out outputs/vstar/interv --attention

# 合成绑定集
python scripts/00_make_synthetic.py --out data/synthetic --n_images 300
python scripts/01_attention_grounding.py --dataset synthetic --data_root data/synthetic --out outputs/synth/attn
# ... 其余脚本同上，加 --dataset synthetic
```

常用参数：`--categories direct_attributes`、`--limit 50`（快速调试）、`--max_pixels`（像素预算；默认 1024 个视觉 token ≈ 1M 像素。V\* 原图约 2246×1582，目标通常很小，**分辨率本身就是一个变量**，建议扫 256/1024/4096 个 token）。

### 输出
- `01`：`layer_*.png`（逐层曲线，按全部/正确/错误/类别划分）、`heads_*.png`（层×头热图）、`vis/*.png`（热力图叠加：各层 + top grounding heads + 物体词 query）、`summary.json`（含 top grounding heads）
- `02`：`layer_obj_rank.png`、`layer_binding_{acc,margin}_*.png`、`top_tokens.jsonl`（bbox 内 token 每 4 层解码出的 top 词）
- `03`：`probe_{attribute,relation,position}.png`、`probe_results.json`（可用 `--skip_extract` 只重新训练）
- `04`：`summary.csv`（类别 × 条件）、`summary.json`（孪生图一致性、DeepStack 注入强度）、`attention_by_condition.png`

## 4. 代码结构

```
rq1grounding/
  data.py       V* / jsonl 读取，prompt，bbox→token 覆盖率，选项关键词与镜像选项
  synthetic.py  可控绑定数据生成（含颜色互换孪生图）
  model.py      加载模型，按像素预算缩放（32 的倍数，token 网格精确已知），token 布局，选项打分
  hooks.py      AttentionRecorder（包装 apply_rotary_pos_emb 精确重算注意力行）
                HiddenStateRecorder + logit_lens
                VisualFeatureEditor（视觉 token 消融 / DeepStack 置零 / 记录）
                DeepStackRecorder（注入范数比、余弦）
                mrope_position_ids（collapse / shuffle / hflip / vflip）
  metrics.py    grounding 指标，grounding heads，绑定错误分类
  viz.py        叠加图与曲线
tests/          用随机初始化的微型 Qwen3-VL 在 CPU 上离线测试（注意力 hook 与 eager 注意力逐元素一致等）
```

`pytest -q tests` 会在 CPU 上构建一个微型随机 Qwen3-VL，验证：重算的注意力与 eager `output_attentions` 一致；logit lens 最后一层与模型 logits 一致；消融/DeepStack/MRoPE 编辑生效且不碰前面的文本位置；hflip 正好镜像宽度方向的位置 id。

## 5. 注意事项与扩展
- **Attention sink**：Qwen 系列会把大量注意力放在少数高范数视觉 token 上，热图可能被少数点主导。可以看 `in_box` 和 `ratio`，并在热图中观察；必要时去掉 attention mass 最高的 1% token 后重新计算。
- 注意力 ≠ 解释：01 的结论需要 04 的因果结果来印证（例如 grounding heads 的注意力高 ↔ 消融目标区域后准确率下降）。
- V\* 只有 191 题（115 属性 + 76 关系），线性探针统计功效有限，所以用合成集补足样本量。
- 可扩展：对 top grounding heads 做 head ablation / activation patching（干净图 ↔ 孪生图）；在 04 中加入 ViT 侧注意力；换 2B/4B/32B 做 scaling 对比。
