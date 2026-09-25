import json
import os
from datasets import Dataset
from transformers import AutoTokenizer

DATA_PATH = "lora_finetune/data/alpaca_zh_52k_sample.json"
OUTPUT_PATH = "lora_finetune/data/tokenized_data"

# print(os.path.dirname(OUTPUT_PATH))
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# 加载数据
with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

print(f"共加载 {len(raw_data)} 条数据")

# 格式化指令数据
formatted_data = []
for item in raw_data:
    instruction = item.get("instruction", "")
    input_text = item.get("input", "")
    output = item.get("output", "")

    if input_text:
        text = f"指令：{instruction}\n输入：{input_text}\n回答：{output}"
    else:
        text = f"指令：{instruction}\n回答：{output}"

    formatted_data.append({"text": text})

# 转为 HuggingFace Dataset
dataset = Dataset.from_list(formatted_data)

# 加载分词器（用 Qwen2.5-0.5B 的分词器，稍后下载）
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    trust_remote_code=True
)

# 设置填充 token（Qwen 默认没有 pad_token）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 分词函数
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

# 对数据集进行分词
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 保存到磁盘
tokenized_dataset.save_to_disk(OUTPUT_PATH)
print(f"分词完成，已保存至 {OUTPUT_PATH}")
print(f"数据示例：")
print(tokenized_dataset[0]["input_ids"][:20])