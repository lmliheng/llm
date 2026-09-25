import torch
x = torch.tensor([[1.0], [2.0], [3.0]])
y_true = torch.tensor([[3.0], [5.0], [7.0]])

# ---------- 2. 模型 ----------
model = torch.nn.Linear(1, 1)   # y = wx + b，自动初始化 w 和 b

# ---------- 3. 损失函数 ----------
loss_fn = torch.nn.MSELoss()

# ---------- 4. 优化器（负责 step） ----------
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# ---------- 5. 训练循环 ----------
for epoch in range(500):
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
        w, b = model.weight.item(), model.bias.item()
        print(f"epoch {epoch:3d} | loss: {loss.item():.6f} | w: {w:.4f} | b: {b:.4f}")

# ---------- 6. 测试 ----------
print("\n训练结束")
print(f"最终模型: y = {model.weight.item():.4f}x + {model.bias.item():.4f}")
print(f"预测 x=4 → {model(torch.tensor([[4.0]])).item():.4f} (期望 9)")