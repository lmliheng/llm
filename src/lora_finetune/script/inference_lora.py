import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-0.5B"
LORA_PATH = "lora_finetune/output/lora_adapter"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载基础模型和分词器
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    device_map="auto",
    trust_remote_code=True
)

# 加载 LoRA 权重
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

# 测试
test_prompts = [
    "指令：请介绍一下你自己。",
    "指令：用Python写一个斐波那契数列函数。",
    "指令：解释什么是机器学习。",
]

for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"输入: {prompt}")
    print(f"输出: {response}")
    print("=" * 60)