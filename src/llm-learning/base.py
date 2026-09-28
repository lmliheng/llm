# 向前 计算损失 清理梯度 反向传播 更新参数

import torch
# 数据集 torch.manual_seed(0)Linear 的 w/b 随机初始化，每次跑的轨迹都不同，任何超参对比都失去意义
torch.manual_seed(0)
x = torch.tensor([[1.0], [2.0], [3.0]])
y_true = torch.tensor([[3.0], [5.0], [7.0]])
# 训练次数
steps=1000
model=torch.nn.Linear(1,1) # 模型
loss_fn=torch.nn.MSELoss() # 当loss_fn=1/3)Σ(ŷ−y)²
optimizer = torch.optim.SGD(model.parameters(), lr=0.2) # 0.01 / 0.1 / 0.2(loss为nan，为什么上限是0.18)
# epoch：把整个数据集完整过一遍。
# step（iteration）：一次参数更新
for step in range(steps):
    y=model(x) # 向前
    loss=loss_fn(y,y_true) # 损失
    optimizer.zero_grad() # 清空梯度
    loss.backward() # 反向传播
    if step % 50 == 0:
       print(f" loss: {loss.item():.6f} ")
    optimizer.step() # 更新参数
    if step % 50 == 0:
        w, b = model.weight.item(), model.bias.item()
        print(f"step {step:3d} |  w: {w:.4f} | b: {b:.4f}")
print("\n训练结束")
print(f"最终模型: y = {model.weight.item():.4f}x + {model.bias.item():.4f}")
print(f"预测 x=4 → {model(torch.tensor([[4.0]])).item():.4f} (期望 9)")



