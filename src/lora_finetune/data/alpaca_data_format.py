import json

# 读取原始数据
with open("lora_finetune/data/alpaca_zh_500_9_27.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 统一格式化
formatted_data = []
for item in data:
    formatted_item = {
        "instruction": f"指令：{item['instruction']}\n输出：",
        "input": "",
        "output": item["output"]
    }
    formatted_data.append(formatted_item)

# 保存
with open("lora_finetune/data/alpaca_data_formatted.json", "w", encoding="utf-8") as f:
    json.dump(formatted_data, f, ensure_ascii=False, indent=2)

print(f"已格式化 {len(formatted_data)} 条数据")