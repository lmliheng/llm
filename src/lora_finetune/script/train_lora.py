import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType

# 配置
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DATA_PATH = "lora_finetune/data/tokenized_data"
OUTPUT_DIR = "lora_finetune/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 检查 GPU 是否可用
if not torch.cuda.is_available():
    raise RuntimeError("CUDA 不可用，请检查驱动或 NVIDIA 驱动安装")
print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# 1. 加载分词器
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. 加载模型（GPU，半精度）
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,       # 半精度，省显存
    device_map="auto",               # 自动分配到 GPU
    trust_remote_code=True
)

# 3. 配置 LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 4. 加载数据
tokenized_dataset = load_from_disk(DATA_PATH)

# 5. 添加 labels 字段
def add_labels(examples):
    examples["labels"] = examples["input_ids"].copy()
    return examples

tokenized_dataset = tokenized_dataset.map(add_labels)

# 6. 设置训练参数（GPU 优化版）
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,    # GPU 可以加大 batch
    gradient_accumulation_steps=2,    # 等效 batch size = 8
    save_steps=500,
    logging_steps=10,
    learning_rate=2e-4,
    warmup_steps=50,
    save_total_limit=2,
    remove_unused_columns=False,
    report_to="none",
    fp16=True,                        # 启用混合精度训练
    dataloader_pin_memory=True,       # GPU 下启用
    optim="adamw_torch",              # 使用 PyTorch 原生优化器
)

# 7. 数据整理器
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)

# 8. 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# 9. 开始训练
print("开始训练...")
print(f"训练数据量: {len(tokenized_dataset)} 条")
trainer.train()

# 保存Lora权重
model.save_pretrained(os.path.join(OUTPUT_DIR, "lora_adapter"))
# 保证tokenizer配置
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "lora_adapter"))
print(f"训练完成")