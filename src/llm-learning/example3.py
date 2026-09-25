import torch
import numpy as np
# 生成数据
x = torch.linspace(-np.pi, np.pi, 100).reshape(-1, 1)
y_true = torch.sin(x)
# ---------- 2. 模型 ----------
model = torch.nn.Sequential(
    torch.nn.Linear(1, 64),   # 先把 1 维映射到 64 维
    torch.nn.ReLU(),           # 非线性激活：ReLU(x) = max(0, x)
    torch.nn.Linear(64, 1)    # 再从 64 维映射回 1 维
)
# ---------- 3. 损失函数 ----------
loss_fn = torch.nn.MSELoss()

# ---------- 4. SGD ，learn rate取0.01 ----------
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epochs=10000

# ---------- 5. 训练循环 ----------
for epoch in range(epochs):
    # 前向：模型猜答案
    y_pred = model(x)

    # 计算 loss
    loss = loss_fn(y_pred, y_true)

    # 反向传播：算梯度
    optimizer.zero_grad()   # 清空上一次的梯度
    loss.backward()         # 自动算 ∂loss/∂w, ∂loss/∂b

    # 更新参数：w -= lr * grad_w, b -= lr * grad_b
    optimizer.step()

    # 每 50 轮打印一次
    if epoch % 50 == 0:
        print(f"epoch {epoch:3d} | loss: {loss.item():.6f}")

# ---------- 6. 测试 ----------
print("\n训练结束")
print(f"预测 x=1.5 → {model(torch.tensor([[1.5]])).item():.4f},预期是{np.sin(1.5)} ")


# 1. 修改激活函数
# 2. 修改线性函数的参数
# 3. 修改learning rate
# 4. 修改预测范围，如果超出训练数据的范围 出现分布外的泛化