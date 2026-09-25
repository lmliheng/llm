import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

train_losses = []
test_losses = []

# 1. 设置设备
# 优先用 GPU，没有就用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 2. 准备数据
# 定义数据预处理：转为张量并归一化到 [0, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 数据集的均值和标准差
])


# 下载并加载训练集和测试集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 创建数据加载器
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")

# 3. 定义模型
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # 输入是 28x28=784 维，输出是 10 类（数字0-9）
        self.fc1 = nn.Linear(784, 128)  # 第一层：784 -> 128
        self.fc2 = nn.Linear(128, 64)   # 第二层：128 -> 64
        self.fc3 = nn.Linear(64, 10)    # 输出层：64 -> 10
        self.relu = nn.ReLU()           # 激活函数
        self.dropout = nn.Dropout(0.2)  # 防止过拟合

    def forward(self, x):
        # 将 28x28 的图像展平为 784 维向量
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # 输出层不用激活函数，后面 CrossEntropyLoss 会处理
        return x


model = SimpleNN().to(device)
print(model)

# 4. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 适合多分类问题
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器

# 5. 训练循环
num_epochs = 5

for epoch in range(num_epochs):
    model.train()  # 切换到训练模式
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_train_loss = running_loss / len(train_loader)
    
    # 6. 测试
    model.eval()  # 切换到评估模式
    correct = 0
    total = 0
    test_loss = 0.0
    
    with torch.no_grad():  # 测试时不需要计算梯度
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_test_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total
    
    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Test Loss: {avg_test_loss:.4f} | "
          f"Accuracy: {accuracy:.2f}%")
    
    train_losses.append(avg_train_loss)
    test_losses.append(avg_test_loss)

# 绘图
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss', marker='o')
plt.plot(range(1, num_epochs+1), test_losses, label='Test Loss', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Testing Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_curve.png', dpi=150)
plt.show()
print("训练完成！")



# 保存模型权重
model_save_path = "mnist_simple_net.pth"
torch.save(model.state_dict(), model_save_path)
print(f"模型已保存至 {model_save_path}")

# 归一化参数也存下来，推理时必须用到
import json
config = {
    "mean": 0.1307,
    "std": 0.3081,
    "input_size": 784,
    "hidden1": 128,
    "hidden2": 64,
    "output": 10
}
with open("mnist_config.json", "w") as f:
    json.dump(config, f)
print("配置已保存至 mnist_config.json")