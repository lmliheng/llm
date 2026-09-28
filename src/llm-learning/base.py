# 向前 计算损失 清理梯度 反向传播 更新参数

import torch
# 数据集
x = torch.tensor([[1.0], [2.0], [3.0]])
y_true = torch.tensor([[3.0], [5.0], [7.0]])
# 训练次数
epochs=1000
model=torch.nn.Linear(1,1) # 模型
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # 
for epoch in range(epochs):
    y=model(x) # 向前
    loss=torch.nn.MSELoss()(y,y_true) # 损失
    optimizer.zero_grad() # 清空梯度
    loss.backward() # 反向传播
    optimizer.step() # 更新参数
    if epoch % 50 == 0:
        w, b = model.weight.item(), model.bias.item()
        print(f"epoch {epoch:3d} | loss: {loss.item():.6f} | w: {w:.4f} | b: {b:.4f}")
print("\n训练结束")
print(f"最终模型: y = {model.weight.item():.4f}x + {model.bias.item():.4f}")
print(f"预测 x=4 → {model(torch.tensor([[4.0]])).item():.4f} (期望 9)")
