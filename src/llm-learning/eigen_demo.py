# -*- coding: utf-8 -*-
"""
eigen_demo.py —— 用闭式解复算 base.py 的训练结果（预测 vs 实测）

为什么值得跑：
    线性模型 + MSE 的 loss，对参数 (w,b) 是二次型 L = ½·eᵀHe（e = θ − θ*）。
    二次型可以按 H 的特征方向拆成两个**互不影响**的一维问题，各自按 (1 − lr·λ) 衰减。
    于是"屏幕上会打出什么数"可以被解析地算出来，不需要训练。
    这个脚本把闭式预测和真实训练并排打印，两者应当在 1% 以内吻合。

用法（在仓库根目录）：
    uv run python src/llm-learning/eigen_demo.py
"""

import torch
import torch.nn.functional as F

STEPS = 1000
LOG_AT = [0, 50, 100, 200, 250, 500, 950, 1000]
LRS = [0.01, 0.1, 0.2]

x = torch.tensor([[1.0], [2.0], [3.0]])
y_true = torch.tensor([[3.0], [5.0], [7.0]])
THETA_STAR = torch.tensor([2.0, 1.0], dtype=torch.float64)  # 数据真实规律 y = 2x + 1


def make_model():
    """固定种子后建模型：闭式预测和真实训练必须从同一个初始点出发。"""
    torch.manual_seed(0)
    return torch.nn.Linear(1, 1)


def hessian():
    """loss 对参数是二次的，它的二阶导矩阵只用数据就能写出来。"""
    n = x.shape[0]
    sxx = (x ** 2).sum().item()
    sx = x.sum().item()
    return torch.tensor([[2 * sxx / n, 2 * sx / n],
                         [2 * sx / n, 2.0]], dtype=torch.float64)


def main():
    H = hessian()
    lam, U = torch.linalg.eigh(H)          # 升序；U 的每一列是一个单位特征向量
    lam_max, lam_min = lam[1].item(), lam[0].item()

    print("=" * 82)
    print("第一步：地形长什么样")
    print("=" * 82)
    print("H = [[%.4f, %.4f], [%.4f, %.4f]]      # loss 对 w、b 的二阶导（曲率）"
          % (H[0, 0], H[0, 1], H[1, 0], H[1, 1]))
    print("λmax = %.4f    λmin = %.4f    条件数 κ = %.1f" % (lam_max, lam_min, lam_max / lam_min))
    print("SGD 上限  lr < 2/λmax = %.4f      # 这就是那个 0.18" % (2 / lam_max))
    print("慢方向 u_min = (%.4f, %.4f)   快方向 u_max = (%.4f, %.4f)"
          % (U[0, 0], U[1, 0], U[0, 1], U[1, 1]))

    # ---- 初始点与误差分解（与训练共用同一个种子）----
    m0 = make_model()
    w0, b0 = m0.weight.item(), m0.bias.item()
    loss0 = F.mse_loss(m0(x), y_true).item()
    e0 = torch.tensor([w0 - THETA_STAR[0].item(), b0 - THETA_STAR[1].item()], dtype=torch.float64)
    s0 = U.transpose(0, 1) @ e0            # 误差在两个特征方向上的分量
    share = lam * s0 * s0                  # 每个方向对初始 loss 的贡献

    print("\n初始点:  w0 = %.6f, b0 = %.6f    初始 loss = %.6f" % (w0, b0, loss0))
    print("初始误差 e0 = θ0 − θ* = (%.6f, %.6f),  ‖e0‖ = %.4f" % (e0[0], e0[1], e0.norm()))
    print("拆到两个特征方向上:  慢分量 = %+.6f   快分量 = %+.6f" % (s0[0], s0[1]))
    print("初始 loss 的构成:    慢方向 %.2f%%   快方向 %.2f%%"
          % (100 * share[0] / share.sum(), 100 * share[1] / share.sum()))
    print("→ 快方向先被吃掉（所以前 50 步 loss 掉得飞快），之后瓶颈是慢方向。")

    def predict(lr, k):
        """闭式解：两个方向各自乘 (1 − lr·λ)，跑 k 步。"""
        s = s0 * (1.0 - lr * lam) ** k
        loss = 0.5 * (lam * s * s).sum().item()
        theta = THETA_STAR + U @ s
        return loss, theta[0].item(), theta[1].item()

    def train(lr):
        """真实训练。注意 loss 在 optimizer.step() **之前**记录，
        这样同一行的 loss 和 w/b 对应的是同一步（打印位置放错会让两者差一次更新）。"""
        model = make_model()
        opt = torch.optim.SGD(model.parameters(), lr=lr)
        log = {}
        for k in range(STEPS + 1):
            loss = F.mse_loss(model(x), y_true)
            if k in LOG_AT:
                log[k] = (loss.item(), model.weight.item(), model.bias.item())
            if k == STEPS:
                break
            opt.zero_grad()
            loss.backward()
            opt.step()
        return log

    for lr in LRS:
        print("\n" + "=" * 82)
        print("lr = %-5s  每步衰减因子：慢方向 %.6f  快方向 %+.6f"
              % (lr, 1 - lr * lam_min, 1 - lr * lam_max))
        print("=" * 82)
        print("%6s | %-30s | %-30s" % ("step", "实测 (loss, w, b)", "闭式预测 (loss, w, b)"))
        print("-" * 82)
        real = train(lr)
        for k in LOG_AT:
            rl, rw, rb = real[k]
            pl, pw, pb = predict(lr, k)
            print("%6d | %-30s | %-30s"
                  % (k, "%.6g, %.4f, %.4f" % (rl, rw, rb), "%.6g, %.4f, %.4f" % (pl, pw, pb)))

    print("\n" + "=" * 82)
    print("自己动手（改完重跑，先预测再看数字）：")
    print("  1. lr 取 0.18（贴上限）和 0.19（越过上限），loss 的差别有多剧烈？")
    print("  2. x 换成 100/200/300，λmax 放大 1e4 倍，lr 上限变成多少？为什么这时必须做归一化？")
    print("  3. 把 data 换成 sin 曲线（example2.py）：为什么那时 H 会随训练过程变，")
    print("     'lr 上限' 不再是常数？（提示：模型不再对参数线性）")
    print("=" * 82)


if __name__ == "__main__":
    main()
