import torch
import torch.nn as nn
import json
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 1. 定义与训练完全一致的模型结构
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x



# 2. 加载配置（归一化参数）
with open("mnist_config.json", "r") as f:
    config = json.load(f)


# 3. 准备单张图片（从测试集随机取）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((config["mean"],), (config["std"],))
])
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
image, label = test_dataset[0]  # 取第0张，形状: [1, 28, 28]

# 4. 加载模型权重
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNN().to(device)
model.load_state_dict(torch.load("mnist_simple_net.pth", map_location=device))
model.eval()  # 关键：评估模式，关闭 Dropout

# 5. 推理（不加梯度计算）
with torch.no_grad():
    # 增加 batch 维度: [1, 28, 28] -> [1, 1, 28, 28]
    input_tensor = image.unsqueeze(0).to(device)
    output = model(input_tensor)
    _, predicted = torch.max(output, 1)
    pred_num = predicted.item()
    true_num = label

# 6. 可视化（展示原始图，不归一化）
plt.imshow(image.squeeze().numpy(), cmap='gray')
plt.title(f"真实值: {true_num} | 预测值: {pred_num}")
plt.axis('off')
plt.show()

print(f"推理完成！真实数字: {true_num}, 预测数字: {pred_num}")