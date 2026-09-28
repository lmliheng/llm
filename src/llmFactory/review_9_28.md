# 训练笔记评审 · train_note_9_28（Qwen2.5-0.5B + LoRA SFT）

评审对象：`src/llmFactory/train_note_9_28.md`（838 B，实为训练参数 dump）、`train_note_9_28.png`（loss 曲线）、`dataset_info.json`（数据集注册）。
评审目的：确认这份笔记能否支撑一次"可复现 + 可判断"的实验复盘，并列出还缺什么。

---

## 0. 结论先讲（三句）

1. **这是一份"配置 dump"，不是"实验记录"。** 参数齐全（39 个键），但没有一行关于"结果"和"判断"的信息——而复盘要的恰恰是后者。
2. **loss 图我读出来了**（下面第 2 节有逐点数字），它能告诉我们：这次训练大约只跑了 **80 个优化器步、16 个日志点**。在这个粒度上，曲线只能证明"配置跑通了、loss 确实在降"，不能证明"模型变好了"。
3. **最要命的遗漏**：你切了 18% 验证集，但 `eval_steps: 100 > 总步数 ≈ 80`，**验证集一次都没被评估过**；`save_steps: 100` 同理，**中途一个 checkpoint 都没有**。也就是说这次训练留下了"一个最终 adapter"和"一张训练 loss 图"，其余为零。

---

## 1. 从笔记里能确认的部分（及我的解读）

| 参数 | 值 | 解读 / 风险 |
| --- | --- | --- |
| `stage` / `finetuning_type` | sft / lora | 路线正确：0.5B + 8GB 显存，LoRA 是唯一合理起点 |
| `lora_rank` / `lora_alpha` / `lora_dropout` | 8 / 16 / 0 | 缩放 α/r = 2（常规）。**dropout=0：数据只有 500 条，等于不设防地拟合噪声**，建议 0.05–0.1 |
| `lora_target` | all | 覆盖全部线性层。0.5B（hidden 896、24 层）按此配置算，可训练参数约 4.4M（≈总量的 0.9%）——**以启动日志里 `trainable params` 那一行为准**，不要用我的估算 |
| `learning_rate` / `lr_scheduler_type` / `warmup_steps` | 5e-5 / cosine / 0 | LoRA 常用 1e-4~2e-4；5e-5 偏保守。**warmup=0 且总步数只有 80**：cosine 在 80 步内就走完了，前 5 步没有热身 |
| `per_device_train_batch_size` × `gradient_accumulation_steps` | 2 × 8 | 有效批 16 条序列/步。每步梯度来自 16 条样本，噪声大——这正是曲线上"尖峰"的主要来源 |
| `num_train_epochs` | 3.0 | 500 条 × 3 epoch ≈ 1230 条序列过一遍，对 0.5B 是"够它记住"的规模 |
| `cutoff_len` | 2048 | **待验证**：中文 alpaca 样本通常远短于 2048，若 p99 远小于它，多余显存全花在 padding 上（第 4 节 C 会给出统计） |
| `val_size` / `eval_steps` / `eval_strategy` | 0.18 / 100 / steps | **配置自相矛盾**：切了 90 条验证集，却在 80 步的 run 里永远等不到第 100 步 |
| `save_steps` | 100 | 同上，中途无 checkpoint，只有训练结束时的最终保存 |
| `template` / `enable_thinking` | default / true | 两处都**需要日志确认**：`default` 会被解析成哪个模板？`enable_thinking` 是 Qwen3 系模板的概念，Qwen2.5 的模板里没有 thinking 段，你的版本是忽略它还是报警告？ |
| `bf16` / `optim` / `seed` | true / adamw_torch / 42 | 正常。单卡用 `adamw_torch` 即可 |
| `max_samples` | 100000 | 形同虚设（数据只有 500 条），无信息量 |
| `logging_steps` / `plot_loss` / `report_to` | 5 / true / none | **唯一的产物就是那张 PNG**；逐点数值必须从 output_dir 的 `trainer_log.jsonl` / `trainer_state.json` 拿 |
| — | — | **缺 `gradient_checkpointing`**：0.5B 在 8GB 上确实不必开，但请在日志里确认它没有被默认打开（会拖慢训练） |

顺带一个部署层面的问题：**没有 merge/export 记录，也没有推理记录**。adapter 装没装上、用哪个后端跑的，笔记里都没有。

---

## 2. loss 图反推出来的数字（证据）

`train_note_9_28.png` 是 matplotlib 输出：标题 `training loss of saves\Qwen2.5-0.5B\lora\train_2026-09-28-14-45-48`，图例两条线 `original`（浅蓝半透明，每步原始值）与 `smoothed`（深蓝实线，滑动平均）。横轴 `step`，刻度 10, 20, …, 80；纵轴 `loss`，刻度 0.70 / 0.75 / 0.80 / 0.85 / 0.90（**整张图只占 0.70–0.90 这个 0.2 宽的窗口**）。

我按像素坐标（x 每 6 px = 1 step，y 每 71 px = 0.05 loss）把曲线上的点还原成数值，误差约 ±0.005：

| step | original | smoothed |
| ---: | ---: | ---: |
| 5 | 0.901 | 0.902 |
| 10 | 0.866 | 0.875 |
| 15 | 0.836 | 0.847 |
| 20 | 0.809 | 0.820 |
| 25 | 0.784 | 0.797 |
| 30 | 0.755 | 0.767 |
| 35 | ~0.80 | 0.803 |
| 40 | 0.782 | 0.789 |
| 45 | 0.753 | 0.764 |
| 50 | **0.672** | 0.700 |
| 55 | 0.735 | 0.724 |
| 60 | 0.715 | 0.717 |
| 65 | ~0.78 | 0.793 |
| 70 | 0.714 | 0.722 |
| 75 | 0.723 | 0.723 |
| 80 | 0.733 | 0.729 |

（step 35 / 65 两行落在陡坡上，像素平均会失真，只标 ~。）

从这 16 个点能读出的东西：

- **前 50 步稳定下降**：0.90 → 0.67，说明 LoRA 确实在学，管线没问题。
- **50 步之后不再下降**：最低点 0.672（step 50，约第 2 个 epoch 中段），此后一路震荡回升，末点 0.733——**比最低点高 0.06（+9%）**。
- **两处尖峰**（≈step 33–35 与 63–68）：每次 5 步才记录一个点、每点由 8 个 micro-batch（16 条序列）平均，所以尖峰既可能是噪声、也可能是某一批长样本/格式异常样本。**现有粒度无法区分**。
- **末段是否在降：不是。** step 64–80 是平台 + 小幅震荡。
- **没有 eval 曲线**（整张图只有 train loss），这不是绘图问题，而是验证集从未被评估。

**能下的结论**：这次训练"能跑通、前 2 个 epoch 有收益、第 3 个 epoch 基本白跑"。**不能下的结论**：模型是否真的变好（没有任何验证证据）。

另一个细节对不上，需要你确认：按你的配置推算 `500 × (1−0.18) = 410` 条训练样本，`410 / 16 = 25.6 → 26 步/epoch`，3 epoch = **78 步**；但图上最后一个点落在 **step 80**。差 2 步。合理猜测是数据集实际条数不是 500（或者 val 划分方式和我算的不同）。请用 `trainer_state.json` 里的 `global_step` 和 `num_train_epochs` 给我定论——**"总步数"是把 eval/save 周期配对的第一步**。

---

## 3. 这份笔记缺的五类信息（以及缺了会怎样）

| # | 缺什么 | 缺了会怎样 |
| --- | --- | --- |
| 1 | **环境与版本**：LLaMA-Factory / torch / transformers / peft / accelerate 版本、GPU 型号与显存、CUDA/驱动、Windows 还是 WSL | 三个月后你自己都复现不了这次 run；模板行为、bf16、flash-attn 是否可用全部取决于版本 |
| 2 | **数据**：条数、来源、字段填充率、token 长度分布、截断率、是否有空 output/重复 | loss 是"数据 + 配置"的共同结果。数据不知道长什么样，loss 的高低就无法解释（0.73 是好是坏？取决于数据难度） |
| 3 | **逐点数值**：`trainer_log.jsonl`（每步 loss / lr / epoch）、`trainer_state.json`、`train_results.json` | 只剩一张图。图上 16 个点、0.2 宽的窗口，读出的是"形状"，不是"数字"，无法做任何定量对比 |
| 4 | **结果**：eval 指标、base vs LoRA 对照样例、merge/export 记录、推理后端与耗时 | **这是本次最大的空白**：没有一条模型输出，等于没有交付物。"训练完成"和"模型可用"是两件事 |
| 5 | **假设与决定**：为什么是 rank 8 / alpha 16 / lr 5e-5 / 3 epoch；这次实验想验证什么；下次打算改什么 | 没有假设就没有实验，只有"跑了一次"。参数只能解释为"抄的默认值" |

---

## 4. 需要你补的东西（一次问完，A→E）

### A. 环境（3 条命令，1 分钟）

在 LLaMA-Factory 目录下跑：

```bat
cd /d <你的 LLaMA-Factory 目录>
llamafactory-cli env > env_report.txt
pip list > pip_list.txt
nvidia-smi > nvidia_smi.txt
```

（若你的版本没有 `llamafactory-cli env` 这个子命令，就跳过它，给我后两个。）
**我只要这些**：`env_report.txt` 全文；`pip_list.txt` 里 `llamafactory / torch / transformers / peft / accelerate / datasets / trl` 七个包的版本；GPU 型号 + 显存 + 驱动版本 + CUDA 版本。

### B. 训练产物（output_dir 下的文本，不是图）

```bat
dir /b saves\Qwen2.5-0.5B\lora\train_2026-09-28-14-45-48
type saves\Qwen2.5-0.5B\lora\train_2026-09-28-14-45-48\trainer_state.json
type saves\Qwen2.5-0.5B\lora\train_2026-09-28-14-45-48\trainer_log_history.json
```

`trainer_log.jsonl`（每行一条 JSON）可以直接整份贴；嫌长就贴**前 5 行 + 每 5 行一条 + 末 5 行**。
**我最需要的字段**：`global_step`、`loss`、`learning_rate`、`epoch`、以及有没有 `eval_loss`。顺便把启动时控制台输出的这两行贴给我：`trainable params: ...`、`Using template: ...`（模板那行决定 `template: default` 到底解析成了什么）。

### C. 数据集体检（跑我这次提交的脚本）

```bat
python src\llmFactory\inspect_dataset.py <你的数据文件路径> --model D:\Models\models\Qwen2.5-0.5B --cutoff-len 2048
```

（`src\llmFactory\inspect_dataset.py` 已随本次提交入库。数据文件是 `alpaca_zh_500_9_27.json`，在你 LLaMA-Factory 的 `data\` 下。）
它会输出：条数、空 output/空 instruction 的条数、完全重复的样本数、字符长度分布、token 长度分布、**超过 2048 被截断的条数**、prompt 占样本长度的比例，以及 3 条随机样本。
另外用一句话告诉我：**这 500 条是从哪来的**（你自己写的 / 开源数据集 / 模型生成的），以及**你想让它学会做什么**。

### D. 结果对照（判断"有没有学到"的唯一证据）

同一批 5–10 个 prompt，分别用 **不加 adapter 的 base** 和 **加 adapter 的 LoRA** 各跑一遍，两份回答贴给我：

```bat
llamafactory-cli chat <你的 infer.yaml>     :: 或者直接跑 webui
```

同时回答：**有没有跑 `llamafactory-cli export` 合并权重？** 合并后的模型后来用什么跑的（transformers / vllm / ollama）？

### E. 显存（按 8GB 约束问，只有 4 个问题）

1. 训练时**峰值显存**多少？（跑训练时抓一次 `nvidia-smi`，或看任务管理器）
2. 有没有出现 **OOM** 或 **loss 变 nan**？
3. `cutoff_len` 定 2048 是**你真实需要的最长长度**，还是随手写的？（C 的统计出来我就能判断能不能砍；砍到 512/1024 省下的显存可以换成更大 batch 或更多数据）
4. 你用的是 **Windows 原生** 还是 **WSL**？flash-attn 在 Windows 上基本装不了，`flash_attn: auto` 大概率退化成了 SDPA——确认一下日志里有没有相关提示。

> 补充一条仓库层面的提醒：`train_note_9_28.png` 在仓库里是 **Git LFS 指针文件**（130 字节）。我这边没有 git-lfs，是靠 `media.githubusercontent.com` 的直链才读到真图的。建议以后**数字走文本**（`trainer_log.jsonl` + 上面这个体检脚本的输出），**图片只作展示**——这样任何时候回顾都不依赖 LFS 是否装好。

---

## 5. 拿到之后我会怎么分析

1. **复原真实曲线**：用 `trainer_log.jsonl` 标出 epoch 边界与 lr 曲线，判断"第 2 个 epoch 就开始平台"还是"一直在降但被噪声盖住"。
2. **定位尖峰**：对照数据体检结果（截断率、长度分布），判断那两个尖峰是噪声还是异常样本。
3. **判断过拟合/欠拟合**：只有加上 eval_loss 才能做——这也是下一版必须改 `eval_steps` 的原因。
4. **给下一版配置草案**（方向，不是结论）：`eval_strategy: epoch` 或 `eval_steps: 10`、`save_strategy` 同步、`cutoff_len` 按 p99 定、`lora_dropout: 0.05~0.1`、epoch 试 1–2、数据量优先扩到 2000+ 条再谈调参。
5. **对齐岗位要求**：能说清"我为什么这样切验证集、为什么这个 lr、为什么这个 rank"，比"我跑了 LoRA"值钱得多——这部分我们在下一轮对话里逐条过。

---

## 6. 检查性问题（先自己想，再回答我）

1. 这次 run 总共约 80 步，而 `eval_steps = save_steps = 100`——**这两个值各触发过几次？** 那么"切了 18% 验证集"这个操作，实际起了什么作用？
2. 如果你只能看那条 train loss 曲线，你能判断模型是**"没学会"**还是**"学过头了"**吗？为什么不能？**你需要哪个数字**才能区分？
3. 如果把 `cutoff_len` 从 2048 降到 512、显存空出来了一块，你会花在**更大 batch** 上还是**更多数据**上？两者的代价分别是什么？
