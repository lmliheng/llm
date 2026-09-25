"""
利用 Qwen2.5-0.5B 批量生成 Alpaca 格式训练数据
输出: lora_finetune/data/alpaca_data_500.json
"""

import json
import os
import random
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------- 配置 ----------
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
OUTPUT_PATH = "lora_finetune/data/alpaca_data_500.json"
NUM_SAMPLES = 500
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

random.seed(SEED)

# ---------- 种子指令池 ----------
SEED_INSTRUCTIONS = [
    # 知识问答类
    {"instruction": "解释什么是神经网络。", "category": "knowledge"},
    {"instruction": "什么是反向传播算法？", "category": "knowledge"},
    {"instruction": "解释L1和L2正则化的区别。", "category": "knowledge"},
    {"instruction": "什么是注意力机制？", "category": "knowledge"},
    {"instruction": "解释梯度消失和梯度爆炸。", "category": "knowledge"},
    {"instruction": "什么是卷积神经网络？", "category": "knowledge"},
    {"instruction": "解释循环神经网络的工作原理。", "category": "knowledge"},
    {"instruction": "什么是迁移学习？", "category": "knowledge"},
    {"instruction": "解释集成学习的原理。", "category": "knowledge"},
    {"instruction": "什么是生成对抗网络？", "category": "knowledge"},
    {"instruction": "解释主成分分析PCA的原理。", "category": "knowledge"},
    {"instruction": "什么是强化学习？", "category": "knowledge"},
    {"instruction": "解释偏差-方差权衡。", "category": "knowledge"},
    {"instruction": "什么是批归一化？", "category": "knowledge"},
    {"instruction": "解释dropout的原理和作用。", "category": "knowledge"},
    
    # 代码编写类
    {"instruction": "用Python写一个快速排序算法。", "category": "coding"},
    {"instruction": "用Python实现一个栈数据结构。", "category": "coding"},
    {"instruction": "写一个函数判断字符串是否是回文。", "category": "coding"},
    {"instruction": "用Python实现二分查找。", "category": "coding"},
    {"instruction": "写一个装饰器记录函数调用次数。", "category": "coding"},
    {"instruction": "用Python实现链表反转。", "category": "coding"},
    {"instruction": "写一个函数找出数组中的第k大元素。", "category": "coding"},
    {"instruction": "用Python实现广度优先搜索。", "category": "coding"},
    {"instruction": "写一个简单的HTTP服务器。", "category": "coding"},
    {"instruction": "用Python实现矩阵乘法。", "category": "coding"},
    
    # 创意写作类
    {"instruction": "写一首关于夏天的五言绝句。", "category": "creative"},
    {"instruction": "写一段关于人工智能未来的展望。", "category": "creative"},
    {"instruction": "写一个简短的科幻故事开头。", "category": "creative"},
    {"instruction": "用比喻的方式描述时间。", "category": "creative"},
    {"instruction": "写一段产品文案介绍智能手表。", "category": "creative"},
    {"instruction": "写一篇关于机器学习的打油诗。", "category": "creative"},
    {"instruction": "用拟人手法描述一次日出。", "category": "creative"},
    {"instruction": "写一段对话体现两个人性格不同。", "category": "creative"},
    {"instruction": "为一款咖啡写广告语。", "category": "creative"},
    {"instruction": "写一个关于坚持的励志短句。", "category": "creative"},
    
    # 实用技能类
    {"instruction": "如何用Git回退到上一个版本？", "category": "practical"},
    {"instruction": "如何在Linux中查找大文件？", "category": "practical"},
    {"instruction": "如何优化SQL查询性能？", "category": "practical"},
    {"instruction": "如何在Python中处理JSON数据？", "category": "practical"},
    {"instruction": "如何使用Docker部署应用？", "category": "practical"},
    {"instruction": "如何在Vim中批量替换文本？", "category": "practical"},
    {"instruction": "如何用正则表达式匹配邮箱地址？", "category": "practical"},
    {"instruction": "如何在命令行中压缩和解压文件？", "category": "practical"},
    {"instruction": "如何设置Python虚拟环境？", "category": "practical"},
    {"instruction": "如何在Jupyter中调试代码？", "category": "practical"},
    
    # 概念解释类
    {"instruction": "什么是面向对象编程？", "category": "concept"},
    {"instruction": "解释函数式编程的特点。", "category": "concept"},
    {"instruction": "什么是设计模式？举例说明。", "category": "concept"},
    {"instruction": "解释MVC架构模式。", "category": "concept"},
    {"instruction": "什么是RESTful API设计原则？", "category": "concept"},
    {"instruction": "解释数据库事务的ACID特性。", "category": "concept"},
    {"instruction": "什么是哈希表？如何解决冲突？", "category": "concept"},
    {"instruction": "解释编译器和解释器的区别。", "category": "concept"},
    {"instruction": "什么是微服务架构？", "category": "concept"},
    {"instruction": "解释CAP定理。", "category": "concept"},
]

# ---------- 模板 ----------
PROMPT_TEMPLATE = """你是一个AI训练数据生成器。请根据以下指令生成一个高质量的回答。

指令：{instruction}

要求：
1. 回答要准确、清晰、完整
2. 如果是代码问题，给出可直接运行的代码
3. 如果是概念问题，用通俗易懂的语言解释
4. 如果是创意问题，发挥想象力
5. 回答长度控制在50-300字之间

请直接输出回答，不要输出其他内容。"""

# ---------- 加载模型 ----------
print(f"加载模型: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()
print(f"模型加载完成，使用设备: {model.device}")

# ---------- 生成回答 ----------
def generate_response(instruction: str) -> str:
    prompt = PROMPT_TEMPLATE.format(instruction=instruction)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 只取生成的部分（去掉prompt）
    response = full_output[len(prompt):].strip()
    return response

# ---------- 生成数据集 ----------
def generate_dataset():
    # 先从种子指令开始
    data = []
    
    print(f"开始生成 {NUM_SAMPLES} 条数据...")
    
    # 第一轮：用种子指令生成
    for seed in tqdm(SEED_INSTRUCTIONS[:50], desc="生成种子数据"):
        try:
            output = generate_response(seed["instruction"])
            if len(output) > 10:  # 过滤太短的回复
                data.append({
                    "instruction": seed["instruction"],
                    "input": "",
                    "output": output
                })
        except Exception as e:
            print(f"生成失败: {seed['instruction']}, 错误: {e}")
    
    # 第二轮：用已有数据生成变体
    categories = list(set(item["category"] for item in SEED_INSTRUCTIONS))
    base_instructions = [item["instruction"] for item in SEED_INSTRUCTIONS]
    
    variation_templates = [
        "请详细解释{}",
        "用通俗的语言解释{}",
        "举例说明{}",
        "谈谈你对{}的理解",
        "写一篇关于{}的短文",
        "用Python实现{}的功能",
        "总结{}的核心要点",
        "从不同角度分析{}",
        "用类比的方式解释{}",
        "写一个教程讲解{}",
    ]
    
    pbar = tqdm(total=NUM_SAMPLES - len(data), desc="生成扩展数据")
    
    while len(data) < NUM_SAMPLES:
        # 随机选择一个基础指令
        base = random.choice(base_instructions)
        template = random.choice(variation_templates)
        
        # 生成变体指令
        new_instruction = template.format(base.lower())
        
        # 避免重复
        if any(d["instruction"] == new_instruction for d in data):
            continue
        
        try:
            output = generate_response(new_instruction)
            if len(output) > 10:
                data.append({
                    "instruction": new_instruction,
                    "input": "",
                    "output": output
                })
                pbar.update(1)
        except Exception as e:
            continue
    
    pbar.close()
    
    # 保存
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n生成完成！共 {len(data)} 条数据")
    print(f"保存路径: {OUTPUT_PATH}")
    
    # 统计类别分布
    category_count = {}
    for item in data:
        cat = item["instruction"][:10]  # 简化统计
    print(f"数据样例:")
    for i in range(min(3, len(data))):
        print(f"\n--- 样例 {i+1} ---")
        print(f"指令: {data[i]['instruction'][:50]}...")
        print(f"输出: {data[i]['output'][:100]}...")

if __name__ == "__main__":
    generate_dataset()