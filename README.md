# RQ1：Qwen3-VL 的 Compositional Grounding 探查与注意力放大

研究问题：Qwen3-VL 答错 V\* 这类视觉搜索题时，它的注意力是否其实已经落在了正确的物体上？如果是，在推理时把这份注意力放大，能否把题答对？

- **探查 1**：注意力能找到物体吗？答错时也能吗？
- **探查 2**：这份注意力有没有被答案用到？用在哪几层？
- **方法**：在这些层里，把注意力往"注意力自己找到的区域"上再集中一些。免训练，每题两次前向。

每个实验的设计细节见 [docs/DESIGN.md](docs/DESIGN.md)。本文档只讲怎么跑。

---

## 1. 获取代码

```bash
git clone -b rq1-grounding-probes https://github.com/xinkunwang111/11777_multimodal_reasoning.git
cd 11777_multimodal_reasoning
```

> **必须加 `-b rq1-grounding-probes`。** 仓库在 GitHub 上的默认分支目前是一个旧分支（`claude/brave-meitner-dxs9jb`，第一版代码），不加 `-b` 会拿到旧代码。建议在 GitHub 仓库的 Settings → General → Default branch 里把默认分支改成 `rq1-grounding-probes`，然后删掉旧分支。
>
> 如果仓库是私有的，需要先获得访问权限，然后用 `gh auth login` 登录，或改用 SSH 地址 `git@github.com:xinkunwang111/11777_multimodal_reasoning.git`。

## 2. 安装环境

```bash
conda create -n rq1 python=3.11 -y
conda activate rq1

# 先装与本机 CUDA 匹配的 PyTorch（以 CUDA 12.1 为例；其他版本见 https://pytorch.org/get-started/locally/）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
python -c "import transformers; print(transformers.__version__)"   # 必须 >= 4.57（Qwen3-VL 从这个版本开始支持）
```

**显卡要求**：
- 8B 模型的 bf16 权重约 17 GB，建议单卡显存 ≥ 24 GB（如 3090 / 4090 / A5000，40 GB 或 80 GB 更宽裕）。
- 有多张卡时，模型会自动切分到各卡上（`device_map="auto"`）。
- 显存不够时，可以换 4B / 2B 模型，或调小 `MAX_PIXELS`（见第 9 节）。

## 3. 下载 Qwen3-VL 权重

Qwen3-VL 没有 7B，稠密版本是 2B / 4B / 8B / 32B。本项目默认用 **8B Instruct**。

```bash
# 方式 A：Hugging Face
#   如果命令行里没有 hf，说明 huggingface_hub 版本较旧，把 `hf download` 换成 `huggingface-cli download`
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir checkpoints/Qwen3-VL-8B-Instruct

#   国内网络连不上 huggingface.co 时，先设置镜像再执行上面的命令
export HF_ENDPOINT=https://hf-mirror.com

# 方式 B：ModelScope（魔搭）
pip install modelscope
modelscope download --model Qwen/Qwen3-VL-8B-Instruct --local_dir checkpoints/Qwen3-VL-8B-Instruct
```

- 其他尺寸：把模型名换成 `Qwen/Qwen3-VL-4B-Instruct`、`Qwen/Qwen3-VL-2B-Instruct` 或 `Qwen/Qwen3-VL-32B-Instruct`。
- 之后所有命令都通过 `MODEL=checkpoints/Qwen3-VL-8B-Instruct`（一键脚本）或 `--model checkpoints/Qwen3-VL-8B-Instruct`（单独脚本）指向本地权重。
- 如果不指定，脚本会自动下载 `Qwen/Qwen3-VL-8B-Instruct` 到 Hugging Face 缓存目录。

## 4. 下载 / 生成数据

```bash
# V* Bench（191 题：direct_attributes 115 + relative_position 76），同样可以用 HF_ENDPOINT 镜像
hf download craigwu/vstar_bench --repo-type dataset --local-dir data/vstar

# 合成绑定集（300 张图，约 900 题；只用于补充分析，可以调小 --n_images）
python scripts/make_synthetic.py --out data/synthetic --n_images 300
```

下载后 `data/vstar/` 的结构应该是：

```
data/vstar/
  direct_attributes/   sa_xxxx.jpg + sa_xxxx.json   (target_object, bbox [x,y,w,h], question, options)
  relative_position/   ...
  test_questions.jsonl                              (官方选项顺序；有它时准确率可与 lmms-eval 对比)
```

一键脚本的 `data` 步骤会自动完成这两件事，这里手动执行也可以。

## 5. 启动前自检（约 1 分钟）

```bash
python scripts/check_setup.py --model checkpoints/Qwen3-VL-8B-Instruct --data_root data/vstar
```

这个脚本会：
- 打印 torch / transformers 版本和可用的 GPU；
- 加载模型，每个类别答 1 道题；
- 打印问题和选项、模型的答案、视觉 token 网格的大小、物体名 token 的注意力是否落在 GT 框上，以及切断这份注意力后答案概率的变化。

最后看到 `OK - setup works` 就说明环境、权重、数据和注意力工具都正常。

## 6. 运行实验

### 6.1 一键运行（推荐）

```bash
export MODEL=checkpoints/Qwen3-VL-8B-Instruct
QUICK=1 bash run_all.sh      # 先试跑：每个数据集 20 题，确认整条流程能通
bash run_all.sh              # 全量；结果汇总在 outputs/Qwen3-VL-8B-Instruct/REPORT.md
```

默认会依次执行下表中除 `appendix` 外的步骤，每一步都在 V\* 和合成集上各跑一遍：

| 步骤 | 做什么 | 脚本 | 依赖 |
|---|---|---|---|
| `data` | 生成合成集；V\* 不存在时自动下载 | `scripts/make_synthetic.py` | — |
| `p1` | 探查 1：注意力能找到物体吗，答错时也能吗？ | `scripts/probe1_attention.py` | — |
| `p2` | 探查 2：切断物体上的注意力，答案变差吗？在哪几层？ | `scripts/probe2_knockout.py` | — |
| `method` | 方法：在读取层放大注意力找到的区域，与 crop 和 GT 区域对比 | `scripts/method_steer.py` | p1、p2 |
| `appendix` | 附录（默认不跑）：DeepStack / MRoPE 消融、logit lens、线性探针 | `scripts/appendix/*.py` | p1 |
| `report` | 汇总成 `REPORT.md` | `scripts/collect_results.py` | — |

常用写法：

```bash
STEPS="p1 p2 method report" bash run_all.sh                 # 只跑部分步骤
STEPS="data p1 p2 method appendix report" bash run_all.sh   # 连附录一起跑
DATASETS="vstar" bash run_all.sh                            # 只跑 V*
CALIB_FROM_SYNTH=1 bash run_all.sh                          # V* 用合成集做校准（默认用 V* 自身 30% 的题）
MAX_PIXELS=4194304 OUT=outputs/8b_4096tok bash run_all.sh   # 换分辨率（4096 个视觉 token）另跑一组
```

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `MODEL` | `Qwen/Qwen3-VL-8B-Instruct` | 本地权重目录或 HF 模型名 |
| `OUT` | `outputs/<模型目录名>` | 输出目录 |
| `DATASETS` | `vstar synthetic` | 要跑的数据集 |
| `VSTAR_ROOT` / `SYNTH_ROOT` | `data/vstar` / `data/synthetic` | 数据位置 |
| `MAX_PIXELS` | 1048576 | 每张图的像素预算；1 个视觉 token = 32×32 px，所以默认是 1024 个 token |
| `CROP_PIXELS` | 262144 | crop 对照组额外使用的 256 个 token |
| `CALIB_FRAC` | 0.3 | 校准集比例（grounding heads、阈值、放大的层和强度都只在校准集上选） |
| `LIMIT` | 全部 | 每个数据集最多跑多少题 |
| `SYNTH_N` | 300 | 合成集图片数 |
| `QUICK` | 0 | 1 = 试跑（每个数据集 20 题） |
| `FORCE` | 0 | 1 = 忽略已完成标记，全部重跑 |
| `KEEP_GOING` | 1 | 0 = 遇到第一个失败就停 |
| `VERBOSE` | 0 | 1 = 日志同时打印到终端 |
| `PYTHON` | `python` | 使用的 Python 解释器 |

**断点续跑**：每完成一步会在输出目录里写一个 `.done` 文件。中断后重新执行同一条命令，已完成的步骤会被跳过。每一步的日志在 `$OUT/logs/<步骤>.log`，失败时终端会打印日志的最后几行。

**多卡并行**：先跑完 `p1`，然后其余步骤可以分到不同的卡上：

```bash
CUDA_VISIBLE_DEVICES=0 STEPS="p2 method" bash run_all.sh &
CUDA_VISIBLE_DEVICES=1 STEPS="appendix" bash run_all.sh &
```

### 6.2 单独运行某个实验

```bash
M=checkpoints/Qwen3-VL-8B-Instruct
D="--dataset vstar --data_root data/vstar"   # 换成合成集：--dataset synthetic --data_root data/synthetic

# 探查 1（会写出 config.json：grounding heads + 区域阈值，供方法使用）
python scripts/probe1_attention.py --model $M $D --out outputs/run1/vstar/p1_attention

# 探查 2（会写出逐题的切断结果，方法从中选出读取层）
python scripts/probe2_knockout.py --model $M $D --out outputs/run1/vstar/p2_knockout

# 方法（需要探查 1、2 的输出）
python scripts/method_steer.py --model $M $D --p1 outputs/run1/vstar/p1_attention \
    --p2 outputs/run1/vstar/p2_knockout --out outputs/run1/vstar/method

# 附录（可选）
python scripts/appendix/qwen3vl_design.py --model $M $D --p1 outputs/run1/vstar/p1_attention --out outputs/run1/vstar/appendix_design
python scripts/appendix/logit_lens.py     --model $M $D --out outputs/run1/vstar/appendix_logit_lens
python scripts/appendix/linear_probe.py   --model $M $D --out outputs/run1/vstar/appendix_linear_probe

# 汇总：参数是 <数据集名> 目录的上一级 -> outputs/run1/REPORT.md
python scripts/collect_results.py outputs/run1
```

如果想把这些结果和一键运行的结果放在一起，输出目录要沿用 `<OUT>/<数据集名>/<实验名>` 的结构，`collect_results.py` 按这个结构读取。

所有脚本的通用参数：

| 参数 | 说明 |
|---|---|
| `--model` | 本地权重目录或 HF 模型名 |
| `--dataset` / `--data_root` | `vstar` 或 `synthetic`（也可以直接给一个 samples.jsonl 路径），以及数据目录 |
| `--max_pixels` | 每张图的像素预算 |
| `--limit` | 最多跑多少题 |
| `--categories` | 只跑某些类别，如 `direct_attributes` |
| `--dtype` | 默认 `bfloat16` |
| `--out` | 输出目录 |

各脚本的主要专有参数（完整列表见 `python scripts/<脚本> -h`）：

| 脚本 | 参数 |
|---|---|
| `probe1_attention.py` | `--calib_frac 0.3`（或 `--calib_dataset synthetic --calib_root data/synthetic`）、`--top_heads 8`、`--sink_k 5`、`--n_vis 12` |
| `probe2_knockout.py` | `--window 4 --stride 4`（层窗口）、`--groups all`（也可以加 `last object`）、`--mode scale`（`shift` = 保持图像总注意力不变的切断） |
| `method_steer.py` | `--lambdas 0.25 0.5 0.75`（放大强度候选）、`--steer_mode shift`、`--modes steer crop`（还可加 `text draw`）、`--sources attn gt`、`--crop_pixels`、`--pad` |

## 7. 结果在哪里，先看什么

```
outputs/Qwen3-VL-8B-Instruct/
  REPORT.md                  <- 所有结果的汇总（表格 + 图），先看这个
  logs/                      <- 每个步骤的日志和本次运行的配置
  vstar/  synthetic/
    p1_attention/  summary.json  config.json  layers.png  heads.png  vis/*.png  per_sample.jsonl
    p2_knockout/   summary.json  knockout_all.png  per_sample.jsonl
    method/        summary.csv  by_found.csv  region_quality.csv  summary_by_category.csv  calibration.json
    appendix_*/    （如果跑了附录）
```

按研究逻辑，依次回答三个问题：

| 问题 | 看哪里 | 看什么 |
|---|---|---|
| 答错时注意力也找到了物体吗？ | `p1_attention/summary.json` → `attention_found_x_correct` | `found_rate_when_wrong`（答错时注意力落在物体上的比例），与 `found_rate_when_correct` 对比；`layers.png` 按答对 / 答错分开画了逐层曲线 |
| 这份注意力被用到了吗？ | `p2_knockout/summary.json` → `groups.all.used` | `delta_target` 是否明显比 `delta_control` 更负；`share_target_hurts_more` 是切物体比切对照伤得更多的题目比例；`window` 是读取层 |
| 放大有用吗？ | `method/summary.csv` | `steer` 行的 `attn` 列与 `base`、`base_hi`、`crop` 比较；`gt` 列是区域完全正确时的上界 |
| 收益来自哪里？ | `method/by_found.csv` | 收益应主要出现在"注意力找到了物体"的那组题上 |

## 8. 计算量（前向次数）

以 8B 模型（36 层）、默认参数估算：

| 实验 | 每道题的前向次数 | V\*（191 题） |
|---|---|---|
| 探查 1 | 1 | 191 |
| 探查 2 | 1 + 9 个层窗口 × 3 个区域 = 28 | 约 5.3k |
| 方法 | 校准题 1 + 12，测试题 6 | 约 1.5k（30% 校准） |
| 附录 DeepStack / MRoPE | 8 | 约 1.5k |

合成集默认约 900 题，耗时主要在探查 2。可以用 `SYNTH_N=100` 或 `LIMIT=...` 缩小。所有前向都不保留 KV cache，每次只处理一道题。

## 9. 常见问题

| 现象 | 解决办法 |
|---|---|
| `KeyError: 'qwen3_vl'` 或模型类型无法识别 | transformers 太旧：`pip install -U "transformers>=4.57"` |
| 下载很慢或失败 | `export HF_ENDPOINT=https://hf-mirror.com`，或改用 ModelScope 下载权重 |
| 显存不足（CUDA OOM） | 调小 `MAX_PIXELS`（如 `524288` = 512 个 token），或换 `Qwen3-VL-4B-Instruct`，或用多卡 |
| clone 下来没有 `scripts/probe1_attention.py` | clone 了旧的默认分支，执行 `git checkout rq1-grounding-probes` |
| 某一步失败 | 看 `$OUT/logs/<步骤>.log`；修好后重新执行同一条命令即可续跑 |
| 想重跑某一步 | 删掉该步骤输出目录里的 `.done`，或 `FORCE=1` 全部重跑 |

## 10. 离线测试（不需要 GPU 和网络）

```bash
pytest -q tests
```

测试会在 CPU 上构建一个随机初始化的微型 Qwen3-VL，验证以下内容：
- 注意力的记录和切断与模型自身的 eager 注意力完全一致；
- V\* 标注格式的读取正确；
- 整条流程能端到端跑通（用 V\* 格式的模拟数据）。

修改代码后先跑一遍测试。
