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

DATA_PATH = "C:/Users/Lenovo/Desktop/superModel/src/lora_finetune/data/tokenized_data"
OUTPUT_DIR = "C:/Users/Lenovo/Desktop/superModel/src/lora_finetune/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device=torch.device("cpu")
print(f"使用设备: {device}")

# 1. 加载分词器
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. 加载模型
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,  # CPU 用 float32，GPU 可改为 float16
    low_cpu_mem_usage=False,  # 低内存模式
    device_map=None,
    trust_remote_code=True
)

# 3. 配置 LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                    # LoRA 秩
    lora_alpha=32,          # 缩放系数
    lora_dropout=0.1,       # Dropout
    target_modules=["q_proj", "v_proj"],  # Qwen 的注意力投影层
)

# 应用 LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 打印可训练参数量

# 4. 加载数据
tokenized_dataset = load_from_disk(DATA_PATH)

# 5. 设置训练参数
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,     # CPU 用小 batch
    gradient_accumulation_steps=4,     # 梯度累积模拟更大 batch
    save_steps=500,
    logging_steps=10,
    learning_rate=2e-4,
    warmup_steps=100,
    save_total_limit=2,
    remove_unused_columns=False,
    report_to="none",                  # 不启用 wandb 等
    fp16=False,                        # CPU 不支持 fp16

    dataloader_pin_memory=False,
)

# 6. 数据整理器
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)

# 7. 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# 8. 开始训练
print("开始训练...")
trainer.train()

# 9. 保存模型
model.save_pretrained(os.path.join(OUTPUT_DIR, "lora_adapter"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "lora_adapter"))
print(f"训练完成！LoRA 适配器已保存至 {OUTPUT_DIR}/lora_adapter")