# -*- coding: utf-8 -*-
"""
base.py 的批注版（可直接替换 base.py 运行，行为等价、只是把细节补齐）。

你在 base.py 里手写的五步，顺序**完全正确**：

    1. 前向  y = model(x)                  拿预测
    2. 损失  loss = loss_fn(y, y_true)     量出"错得有多离谱"
    3. 清零  optimizer.zero_grad()         清掉上一轮残留的梯度   <-- 必须在 4 之前
    4. 反向  loss.backward()               算出 ∂loss/∂每个参数（只算，不改参数）
    5. 更新  optimizer.step()              用梯度改参数：w -= lr * grad_w

这个顺序是 PyTorch 训练的全部骨架，剩下的所有东西（DataLoader、验证集、
混合精度、分布式）都只是往这五步外面加壳。下面的 6 处改动都不改变骨架，
只是把"现在无害、上大模型必炸"的习惯补掉，改动处标了 [改动 N]。
"""

import torch
import torch.nn as nn

# [改动 1] 固定随机种子。
#   nn.Linear 的 w/b 是随机初始化的（kaiming_uniform_），不固定种子时每次运行
#   初始点都不同，收敛轨迹、最终 loss 全都不一样，实验之间没法对比。
#   调任何超参之前的第一件事，就是让"除被测因素外的一切"都可复现。
torch.manual_seed(0)


def main():
    # ---------- 1. 数据 ----------
    # 这是个标准答案已知的玩具问题：y = 2x + 1。
    # x = 1,2,3 → y_true = 3,5,7。训练完 w 应该逼近 2、b 应该逼近 1。
    # 形状 (3, 1) = (样本数, 特征数)；nn.Linear 吃的是"最后维是特征"的张量。
    x = torch.tensor([[1.0], [2.0], [3.0]])
    y_true = torch.tensor([[3.0], [5.0], [7.0]])

    # ---------- 2. 模型 / 损失 / 优化器 ----------
    model = nn.Linear(1, 1)  # y = wx + b，内部持有 weight 与 bias 两个 Parameter

    # [改动 2] 损失函数移到循环外。
    #   它是"配置"，不是"每步要重算的东西"。原来写成 torch.nn.MSELoss()(y, y_true)，
    #   等价于每轮新建一个 MSELoss 对象——1000 轮就分配 1000 次。这里开销小，
    #   但同样的写法放到 70B 模型里就是每步一次多余的对象分配。
    #   （你在 example2.py / example3.py 里本来是写在循环外的，这次凭记忆重写丢了。）
    loss_fn = nn.MSELoss()  # reduction 默认 'mean'：对 batch 内所有元素取平均

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # [改动 3] 循环变量的名字：epoch → step。
    #   在你当前这份代码里两者恰好相等，因为整个数据集只有一批（full-batch 梯度下降）：
    #       1 个 epoch（过完一遍数据）= 1 次参数更新 = 1 个 step
    #   一旦引入 DataLoader 和 batch_size，一个 epoch 就变成"多次"更新，名字会开始骗人。
    #   从现在起养成习惯：循环变量是"参数更新次数"，数据集过几遍是另一件事。
    total_steps = 1000

    # ---------- 3. 训练循环：五步 ----------
    for step in range(total_steps):
        # [改动 4] 显式声明训练模式。
        #   nn.Linear 上没有 Dropout / BatchNorm，写不写结果一样；但只要模型里出现
        #   Dropout 或 BatchNorm，漏掉 model.train() 会让它们按推理行为工作，
        #   训练/验证结果就开始对不上。习惯要从没有副作用的时候养成。
        model.train()

        y = model(x)                  # ① 前向
        loss = loss_fn(y, y_true)     # ② 损失。注意参数顺序是 (预测值, 真值)。
                                      #    MSELoss 对称所以写反了看不出问题，
                                      #    换 nn.CrossEntropyLoss() 就会静默训练错方向。
        optimizer.zero_grad()         # ③ 清零：PyTorch 的 .grad 是"累加"语义，不清会带上历史梯度
        loss.backward()               # ④ 反向：算出梯度写进 param.grad
        optimizer.step()              # ⑤ 更新：param.data -= lr * param.grad

        if step % 50 == 0:
            w, b = model.weight.item(), model.bias.item()
            print(f"step {step:4d} | loss: {loss.item():.6f} | w: {w:.4f} | b: {b:.4f}")

    # ---------- 4. 评估 ----------
    # [改动 5] 评估要关两样东西：
    #   model.eval()      —— 切换 Dropout/BatchNorm 的推理行为
    #   torch.no_grad()   —— 不建计算图。推理不需要梯度，省显存、也更快
    #   你原来直接 model(torch.tensor([[4.0]])) 会照建一张计算图再扔掉，纯浪费。
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor([[4.0]]))  # 期望 9.0（因为 2*4+1 = 9）

    print("\n训练结束")
    print(f"最终模型: y = {model.weight.item():.4f}x + {model.bias.item():.4f}")
    print(f"预测 x=4 → {pred.item():.4f} (期望 9)")


# [改动 6] if __name__ == "__main__" 保护 + 把逻辑收进 main()。
#   好处有两个：别的文件 import 这个文件时不会顺手训练一遍；后面要把它改成
#   多组超参的循环（见 base_review.md 里的 lr 实验）时，改一个函数就够。
if __name__ == "__main__":
    main()
