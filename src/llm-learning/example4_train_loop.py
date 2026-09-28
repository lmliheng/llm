# -*- coding: utf-8 -*-
"""
example4_train_loop.py —— 训练循环解剖：从「手算梯度」到「工业级循环」

对照仓库里已有的代码看，效果最好：
    example1.py        用 nn.Linear + SGD 学 y = 2x + 1（黑箱版，只知道结果）
    train_mnist.py     MNIST 的完整训练脚本（能跑通，但训练流程有两处坑，见 Stage 3）
本文件就是把那个黑箱打开。按 5 级递进，一级一级跑，别跳：

    Stage 0   环境自检：确认 PyTorch 认得你的 RTX 5060（Blackwell / sm_120）
    Stage 1   纯手写梯度下降：不用 autograd，自己推 dL/dw、dL/db
    Stage 2   改用 autograd 求梯度，但参数更新和梯度清零仍然手写
    Stage 2b  .grad 是「累加」不是「覆盖」——忘记清零到底会发生什么
    Stage 3   工业级循环：DataLoader + nn.Module + 验证集 + 最优 checkpoint

运行（在仓库根目录）：
    uv run python src/llm-learning/example4_train_loop.py

依赖已在 pyproject.toml 里（torch 2.7.1+cu128 / torchvision 0.22.1+cu128，正好是
RTX 50 系需要的 cu128 版本；具体版本以官方文档为准）。
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 全局只决定一次设备，后面所有模型和张量都往这里搬
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================================================
# Stage 0  环境自检
# ==========================================================================
def stage0_env_check():
    print("=" * 72)
    print("Stage 0  环境自检")
    print("=" * 72)
    print(f"torch 版本      : {torch.__version__}")
    print(f"torch 编译的 CUDA: {torch.version.cuda}")
    print(f"CUDA 是否可用    : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("!! 没检测到 CUDA，后面会退回 CPU 跑。先确认装的是 CUDA 版 PyTorch。")
        print(f"!! 当前用的设备：{DEVICE}")
        return

    props = torch.cuda.get_device_properties(0)
    major, minor = torch.cuda.get_device_capability(0)
    print(f"GPU 型号        : {props.name}")
    print(f"算力(sm)        : sm_{major}{minor}")
    print(f"显存            : {props.total_memory / 1024 ** 3:.1f} GiB")

    # PyTorch 的安装包是「预编译」好一批 GPU 架构内核的二进制。
    # 如果你的显卡算力不在这个列表里，说明它压根没编你的内核，一跑就报
    # "no kernel image is available for execution on the device"。
    arch_list = torch.cuda.get_arch_list()
    print(f"本版支持的算力   : {arch_list}")
    if f"sm_{major}{minor}" not in arch_list:
        print("!! 警告：你的算力不在支持列表里，跑起来一定报 no kernel image。")
        print("!! 解法：换带 cu128 的 PyTorch（RTX 50 系需要），以官方文档为准。")
    else:
        print("OK：算力匹配，可以开跑。")


# ==========================================================================
# 复用的数据：和 example1.py 一模一样
# ==========================================================================
def example1_data():
    """y = 2x + 1，三个点。真值 w=2, b=1，方便肉眼验收。"""
    x = torch.tensor([[1.0], [2.0], [3.0]])
    y = torch.tensor([[3.0], [5.0], [7.0]])
    return x, y


# ==========================================================================
# Stage 1  纯手写梯度下降 —— 一行 autograd 都不用
# ==========================================================================
def stage1_hand_written_gradient(lr=0.05, steps=200):
    print("\n" + "=" * 72)
    print("Stage 1  纯手写梯度下降（无 autograd）")
    print("=" * 72)

    x, y = example1_data()
    w = torch.tensor(0.0)   # 不给 requires_grad：梯度下面自己推
    b = torch.tensor(0.0)

    for step in range(1, steps + 1):
        y_hat = w * x + b                    # 1. 前向：模型猜答案
        loss = ((y_hat - y) ** 2).mean()     # 2. 损失：均方误差

        # 3. 反向：不调 loss.backward()，自己推解析梯度
        #     L = (1/N) * Σ (w*x + b - y)^2
        #     ∂L/∂w = (1/N) * Σ 2 * (y_hat - y) * x
        #     ∂L/∂b = (1/N) * Σ 2 * (y_hat - y) * 1
        err = y_hat - y
        dw = (2.0 * err * x).mean()
        db = (2.0 * err).mean()

        # 4. 更新：沿梯度反方向迈一小步
        with torch.no_grad():
            w -= lr * dw
            b -= lr * db

        if step <= 3 or step % 50 == 0:
            print(f"step {step:3d} | loss {loss.item():10.6f} | "
                  f"w {w.item():8.4f} | b {b.item():8.4f}")

    print(f"手写梯度结果： w = {w.item():.6f}, b = {b.item():.6f}   （真值 2 / 1）")
    return w.item(), b.item()


# ==========================================================================
# Stage 2  autograd 求梯度，但更新与清零还是手写
# ==========================================================================
def stage2_autograd_manual_update(lr=0.05, steps=200):
    print("\n" + "=" * 72)
    print("Stage 2  autograd 求梯度 + 手写更新 / 手写清零")
    print("=" * 72)

    x, y = example1_data()
    # requires_grad=True 是在告诉 autograd：这条计算链上的这个叶子节点要留梯度
    w = torch.tensor(0.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)

    for step in range(1, steps + 1):
        loss = F.mse_loss(w * x + b, y)   # 1. 前向 + 2. 损失
        loss.backward()                   # 3. 反向：把 ∂L/∂w、∂L/∂b 写进 w.grad / b.grad

        # 4. 更新 + 清零 —— 注意 autograd 只管「算梯度」，
        #    改参数（step）和清梯度（zero_grad）都是另外两件事，这里全手写出来。
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
            w.grad = None        # 等价于 optimizer.zero_grad()：清空，避免下一步累加
            b.grad = None

        if step <= 3 or step % 50 == 0:
            print(f"step {step:3d} | loss {loss.item():10.6f} | "
                  f"w {w.item():8.4f} | b {b.item():8.4f}")

    print(f"autograd 结果： w = {w.item():.6f}, b = {b.item():.6f}")
    return w.item(), b.item()


# ==========================================================================
# Stage 2b  .grad 是累加，不是覆盖
# ==========================================================================
def _run_linear_gd(x, y, lr, steps, zero_grad):
    """固定初值跑一次梯度下降，返回每一步的 (loss, 这一步实际用到的梯度)。"""
    w = torch.tensor(0.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    hist = []
    for _ in range(steps):
        loss = F.mse_loss(w * x + b, y)
        loss.backward()
        g_used = w.grad.item()          # 先记下来：这就是这一步真正参与更新的梯度
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
            if zero_grad:
                w.grad.zero_()
                b.grad.zero_()
        hist.append((loss.item(), g_used, w.item(), b.item()))
    return hist


def stage2b_grad_accumulation(lr=0.2, steps=40):
    print("\n" + "=" * 72)
    print("Stage 2b  忘记清零梯度会发生什么")
    print("=" * 72)

    # ---- (a) 最硬核的证据：同一个 w 连着 backward 两次 ----
    w = torch.tensor(1.0, requires_grad=True)
    ((w - 3.0) ** 2).backward()
    g1 = w.grad.item()
    ((w - 5.0) ** 2).backward()
    g2 = w.grad.item()
    print("(a) 对同一个 w 连续两次 backward：")
    print(f"    第 1 次后 w.grad = {g1:+.1f}   （2*(1-3) = -4.0）")
    print(f"    第 2 次后 w.grad = {g2:+.1f}  （2*(1-5) = -8.0，但实际是 -4 + -8 = -12）")
    print("    → 结论：.grad 永远是『加』不是『覆盖』。清零没有任何自动机制，必须手动做。")

    # ---- (b) 放进训练循环里看 ----
    # 学习率故意取大一点（0.2），让「更新量被放大」这件事在几十步内就暴露出来
    x, y = example1_data()
    good = _run_linear_gd(x, y, lr=lr, steps=steps, zero_grad=True)
    bad = _run_linear_gd(x, y, lr=lr, steps=steps, zero_grad=False)

    print(f"\n(b) 同一个循环跑两遍（lr={lr}，{steps} 步），只差一行清零：")
    print(f"{'step':>4} | {'清零: loss':>12} {'w.grad':>10} | "
          f"{'不清零: loss':>13} {'w.grad':>10}")
    show = {1, 2, 3, 5, 10, 20, 30, 40}
    for i in range(steps):
        if (i + 1) in show:
            lg, gg, wg, _ = good[i]
            lb, gb, wb, _ = bad[i]
            print(f"{i + 1:>4} | {lg:>12.6f} {gg:>10.4f} | {lb:>13.4f} {gb:>10.4f}")

    print(f"\n清零分支   最终 loss = {good[-1][0]:.8f}   （收敛到 0 附近，w={good[-1][2]:.4f}）")
    print(f"不清零分支 最终 loss = {bad[-1][0]:.8f}   （在最优解附近震荡，w={bad[-1][2]:.4f}）")
    print("机制是确定的：不清零时第 k 步用的是前 k 步梯度的『和』，")
    print("更新量 ≈ lr × (累计梯度)，等效学习率随步数增长，于是越过谷底来回震荡。")
    print("具体数值随数据和 lr 变化，但『不收敛』这一点不变；最坑的是它不报错。")


# ==========================================================================
# Stage 3  工业级训练循环（MNIST）
# ==========================================================================
class SimpleNN(nn.Module):
    """和 train_mnist.py 里的结构保持一致，方便你对照结果。"""

    def __init__(self):
        super().__init__()                      # 必须先调，否则参数注册不上
        self.fc1 = nn.Linear(784, 128)          # 28*28 = 784
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)            # 输出 logits，不是概率
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 784)                     # 展平：[B,1,28,28] -> [B,784]
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.fc3(x)                      # CrossEntropyLoss 内部自带 softmax


def build_mnist_loaders(batch_size=64):
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),   # MNIST 全局均值/标准差
    ])
    mnist_train = datasets.MNIST("./data", train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST("./data", train=False, download=True, transform=transform)

    # 关键：从训练集里切 6000 张当验证集。
    # 原来的 train_mnist.py 每个 epoch 直接对 test 集算指标，相当于边做题边看考卷，
    # 一旦你开始按这个指标调超参/提前停，测试分数就不可信了。
    n_val = 6000
    g = torch.Generator().manual_seed(42)
    train_ds, val_ds = torch.utils.data.random_split(
        mnist_train, [len(mnist_train) - n_val, n_val], generator=g
    )
    if DEVICE.type == "cpu":                # CPU 上少跑点，不然等到怀疑人生
        train_ds = torch.utils.data.Subset(train_ds, range(6000))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=2)
    test_loader = DataLoader(mnist_test, batch_size=512, shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()          # 打开 dropout 的训练行为；漏了它验证指标会莫名其妙地差
    total, correct, loss_sum = 0, 0, 0.0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)                  # 1. 前向
        loss = criterion(outputs, labels)        # 2. 算损失

        optimizer.zero_grad(set_to_none=True)    # 3. 清掉上一轮梯度（Stage 2b 讲过不清的后果）
        loss.backward()                          # 4. 反向传播
        optimizer.step()                         # 5. 更新参数

        # 累加 loss 一定要 .item()：否则每步的计算图都被拽住，显存会一直涨
        loss_sum += loss.item() * images.size(0)       # 按样本数加权，比「除以批数」准
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()           # 推理不建计算图：省显存、更快
def evaluate(model, loader, criterion):
    model.eval()           # 关掉 dropout
    total, correct, loss_sum = 0, 0, 0.0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_sum += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)
    return loss_sum / total, correct / total


def stage3_full_training_loop(epochs=5, batch_size=64, lr=1e-3):
    print("\n" + "=" * 72)
    print("Stage 3  工业级训练循环（MNIST）")
    print("=" * 72)

    torch.manual_seed(42)
    train_loader, val_loader, test_loader = build_mnist_loaders(batch_size)
    model = SimpleNN().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"设备 {DEVICE} | 参数量 {n_params:,} | 每轮批数 {len(train_loader)} | batch {batch_size}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        va_loss, va_acc = evaluate(model, val_loader, criterion)
        print(f"epoch {epoch}/{epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} "
              f"| val loss {va_loss:.4f} acc {va_acc:.4f} | {time.time() - t0:.1f}s")

        # 只存验证集最好的那一次：过拟合之后 val acc 会掉，存最后一个 epoch 就是存次品
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), "mnist_best.pt")
            print(f"    ↑ 新最优 val acc {va_acc:.4f}，已存 mnist_best.pt")

    # 训练全部结束、超参也定死了，才动测试集，只量一次
    model.load_state_dict(torch.load("mnist_best.pt", map_location=DEVICE))
    te_loss, te_acc = evaluate(model, test_loader, criterion)
    print(f"\n最终测试集: loss {te_loss:.4f} | acc {te_acc:.4f}   （这个网络大约 0.97~0.98）")
    if DEVICE.type == "cuda":
        print(f"本阶段峰值显存 {torch.cuda.max_memory_allocated() / 1024 ** 2:.1f} MiB"
              f"（8GB 卡跑这种小模型毫无压力，真正吃显存的是后面微调 LLM）")


def main():
    stage0_env_check()

    w1, b1 = stage1_hand_written_gradient()
    w2, b2 = stage2_autograd_manual_update()
    print(f"\n对照：手写梯度 (w={w1:.6f}, b={b1:.6f}) vs autograd (w={w2:.6f}, b={b2:.6f})")
    print("两者一致 → 说明 autograd 没有魔法，它算的就是你手推的那个导数。")

    stage2b_grad_accumulation()
    stage3_full_training_loop()

    print("\n全部跑通。下一单元：神经网络史 → 为什么 RNN 被淘汰 → 手写 RNN 与 Attention。")


if __name__ == "__main__":
    main()
