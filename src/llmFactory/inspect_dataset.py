# -*- coding: utf-8 -*-
"""数据集体检脚本 —— 针对 LLaMA-Factory 的 alpaca 格式 SFT 数据。

为什么需要它：
    训练日志里的 loss 只能告诉你"优化器把数字压下去了多少"，压不动的部分往往
    出在数据上（样本太短/太长被截断、output 为空、样本重复、监督信号太少）。
    这个脚本把"数据长什么样"变成一组可复现的数字，用来解释 loss 曲线。

用法（在你本地 LLaMA-Factory 目录下运行）：
    python inspect_dataset.py ..\\llm\\src\\llmFactory\\data\\alpaca_zh_500_9_27.json ^
        --model D:\\Models\\models\\Qwen2.5-0.5B --cutoff-len 2048

参数说明：
    数据文件路径   必须是 alpaca 格式（每条含 instruction / input / output）
    --model        本地 tokenizer 目录；不传就只做结构统计，不做 token 统计
    --cutoff-len   训练时用的 cutoff_len，用来统计"有多少条被截断"
    --samples      末尾随机打印几条样本，便于肉眼检查（默认 3，给 0 则不打印）

依赖：只用标准库 + transformers（你训练时已经装好）。
"""

import argparse
import json
import random
import statistics
import sys
from collections import Counter

# LLaMA-Factory 的 alpaca 字段名（对应 dataset_info.json 里的 columns 映射）
INSTRUCTION_KEYS = ("instruction", "prompt")
INPUT_KEYS = ("input", "query")
OUTPUT_KEYS = ("output", "response")


def load_records(path):
    """读入 JSON，统一成 list[dict]。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "records", "examples"):
            if isinstance(data.get(key), list):
                return data[key]
    raise SystemExit("无法识别的数据格式：顶层既不是 list，也没有 data/records/examples 列表")


def pick(record, keys):
    """按候选字段名取第一个存在的值，返回字符串。"""
    for k in keys:
        if k in record and record[k] is not None:
            return str(record[k])
    return ""


def build_prompt(instruction, extra_input):
    """复刻 LLaMA-Factory 的 alpaca 拼装：input 非空时用换行拼在 instruction 后面。"""
    if extra_input.strip():
        return instruction + "\n" + extra_input
    return instruction


def pct(values, q):
    """简单分位数（不引入 numpy）。"""
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_file")
    ap.add_argument("--model", default=None, help="本地 tokenizer/model 目录，用于统计 token 长度")
    ap.add_argument("--cutoff-len", type=int, default=2048)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    records = load_records(args.data_file)
    total = len(records)
    print("=" * 68)
    print("数据文件: %s" % args.data_file)
    print("总条数  : %d" % total)
    print("=" * 68)

    # ---------- 1. 结构统计：字段缺失、空值、重复 ----------
    missing_instr = 0
    empty_instr = 0
    empty_output = 0
    empty_input = 0
    instr_chars, input_chars, output_chars = [], [], []
    prompt_hash = Counter()

    for rec in records:
        if not isinstance(rec, dict):
            missing_instr += 1
            continue
        instruction = pick(rec, INSTRUCTION_KEYS)
        extra_input = pick(rec, INPUT_KEYS)
        output = pick(rec, OUTPUT_KEYS)

        if instruction == "" and not any(k in rec for k in INSTRUCTION_KEYS):
            missing_instr += 1
        if instruction.strip() == "":
            empty_instr += 1
        if output.strip() == "":
            empty_output += 1
        if extra_input.strip() == "":
            empty_input += 1

        instr_chars.append(len(instruction))
        input_chars.append(len(extra_input))
        output_chars.append(len(output))
        prompt_hash[build_prompt(instruction, extra_input).strip()] += 1

    dup_groups = {k: v for k, v in prompt_hash.items() if v > 1}
    dup_rows = sum(v - 1 for v in dup_groups.values())

    print("\n[1] 结构与质量")
    print("  缺 instruction 字段的条数 : %d" % missing_instr)
    print("  instruction 为空的条数    : %d" % empty_instr)
    print("  output(回答)为空的条数    : %d   <-- 训练时这些样本的监督信号几乎为 0" % empty_output)
    print("  input(补充输入)为空的条数 : %d   (%.1f%%，alpaca 里常见，不代表有问题)"
          % (empty_input, 100.0 * empty_input / max(1, total)))
    print("  prompt 完全重复的组数     : %d，多出来的重复条数: %d" % (len(dup_groups), dup_rows))

    print("\n[2] 字符长度分布 (instruction / input / output)")
    for name, vals in (("instruction", instr_chars), ("input", input_chars), ("output", output_chars)):
        if not vals:
            continue
        print("  %-12s min=%4d  p50=%4d  p90=%4d  max=%4d  mean=%.1f"
              % (name, min(vals), pct(vals, .5), pct(vals, .9), max(vals), statistics.mean(vals)))

    # ---------- 3. token 长度统计（可选，需要 tokenizer） ----------
    tok = None
    if args.model:
        try:
            from transformers import AutoTokenizer
        except Exception as e:  # noqa: BLE001
            print("\n[3] 跳过 token 统计：import transformers 失败 (%s)" % e)
        else:
            try:
                tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
                print("\n[3] token 长度分布   (tokenizer=%s, cutoff_len=%d)"
                      % (args.model, args.cutoff_len))
                print("    注意：这里用 tokenizer 自带的 chat template 近似 LLaMA-Factory 的模板，")
                print("    模板细节会有几个 token 的出入，用来判断量级足够。")
            except Exception as e:  # noqa: BLE001
                print("\n[3] 跳过 token 统计：加载 tokenizer 失败 (%s)" % e)

    if tok is not None:
        total_toks, prompt_toks, trunc, over_ratio = [], [], 0, []
        for rec in records:
            if not isinstance(rec, dict):
                continue
            instruction = pick(rec, INSTRUCTION_KEYS)
            extra_input = pick(rec, INPUT_KEYS)
            output = pick(rec, OUTPUT_KEYS)
            user_msg = build_prompt(instruction, extra_input)
            n_prompt = len(tok(user_msg, add_special_tokens=False)["input_ids"])
            try:
                text = tok.apply_chat_template(
                    [{"role": "user", "content": user_msg},
                     {"role": "assistant", "content": output}],
                    tokenize=False, add_generation_prompt=False)
                n_total = len(tok(text, add_special_tokens=False)["input_ids"])
            except Exception:
                n_total = n_prompt + len(tok(output, add_special_tokens=False)["input_ids"])
            total_toks.append(n_total)
            prompt_toks.append(n_prompt)
            if n_total > args.cutoff_len:
                trunc += 1
            over_ratio.append(n_prompt / max(1, n_total))

        print("  样本总长度 token : min=%d  p50=%d  p90=%d  p99=%d  max=%d"
              % (min(total_toks), pct(total_toks, .5), pct(total_toks, .9), pct(total_toks, .99), max(total_toks)))
        print("  其中 prompt 部分 : p50=%d  p90=%d" % (pct(prompt_toks, .5), pct(prompt_toks, .9)))
        print("  超过 cutoff_len(%d) 被截断的条数 : %d  (%.1f%%)   <-- 这些样本的答案尾部会丢"
              % (args.cutoff_len, trunc, 100.0 * trunc / max(1, len(total_toks))))
        print("  prompt 占样本长度的比例 p50=%.2f  p90=%.2f              <-- 越高，真正被监督的 token 越少"
              % (pct(over_ratio, .5), pct(over_ratio, .9)))
        print("  训练 1 个 epoch 的 token 总量约 : %d" % sum(total_toks))
        if pct(total_toks, .99) < args.cutoff_len / 2:
            print("  >> 提示：p99 远小于 cutoff_len，说明 2048 大部分时候是 padding，显存花得不值。")

    # ---------- 4. 抽样打印 ----------
    if args.samples > 0 and records:
        rng = random.Random(args.seed)
        print("\n[4] 随机抽样 %d 条（检查格式与内容质量）" % args.samples)
        for i, rec in enumerate(rng.sample(records, min(args.samples, len(records)))):
            if not isinstance(rec, dict):
                continue
            print("  --- 样本 %d ---" % (i + 1))
            print("  instruction: %s" % pick(rec, INSTRUCTION_KEYS)[:200].replace("\n", "\\n"))
            print("  input      : %s" % pick(rec, INPUT_KEYS)[:120].replace("\n", "\\n"))
            print("  output     : %s" % pick(rec, OUTPUT_KEYS)[:200].replace("\n", "\\n"))

    print("\n把上面整段输出贴给我即可。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
