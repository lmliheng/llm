from transformers import AutoTokenizer

# 加载一个轻量中文分词器（首次运行会自动下载），明显中文分词器对英文不好
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

# 待切分的句子
sentences = [
    "l love you , but I love deep-learn more",
    "我爱你",
    "我爱深度学习",
    "今天天气真好",
]

for s in sentences:
    tokens = tokenizer.tokenize(s)          # 切词结果（字符串）
    ids = tokenizer.convert_tokens_to_ids(tokens)  # 转成数字ID
    print(f"原文: {s}")
    print(f"Tokens: {tokens}")
    print(f"IDs: {ids}")
    print("-" * 40)