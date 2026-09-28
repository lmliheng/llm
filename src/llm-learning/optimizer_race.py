# -*- coding: utf-8 -*-
"""
optimizer_race.py —— 同一道题上赛跑 SGD / Momentum / Adam

承接 eigen_demo.py：那道玩具题的 Hessian 条件数 κ ≈ 46，是个"又长又扁的椭圆"。
扁椭圆上，普通梯度下降在陡方向来回震荡、在慢方向几乎挪不动。动量与 Adam 都是
针对这件事的药，但机制完全不同。这个脚本让它们从同一个初始点出发，看谁先到
loss < 1e-8，并把"理论每步衰减率"和"实测步数"并排打印出来。

用法（在仓库根目录）：
    uv run python src/llm-learning/optimizer_race.py
"""

import math

import torch
import torch.nn.functional as F

x = torch.tensor([[1.0], [2.0], [3.0]])
y_true = torch.tensor([[3.0], [5.0], [7.0]])

MAX_STEPS = 3000
TARGET = 1e-8                       # 认为"已收敛"的 loss 门槛
LOG_AT = [0, 50, 100, 200, 400, 800, 1600, 3000]


def make_model():
    """每个优化器都必须从同一个初始点出发，所以每次都用同一个种子重新建模型。"""
    torch.manual_seed(0)
    return torch.nn.Linear(1, 1)


def hessian_eigs():
    """loss 对参数是二次的，H 只用数据就能写出来（见 eigen_demo.py）。"""
    n = x.shape[0]
    sxx, sx = (x ** 2).sum().item(), x.sum().item()
    H = torch.tensor([[2 * sxx / n, 2 * sx / n],
                      [2 * sx / n, 2.0]], dtype=torch.float64)
    lam, _ = torch.linalg.eigh(H)   # 升序
    return lam[0].item(), lam[1].item()


LMIN, LMAX = hessian_eigs()
KAPPA = LMAX / LMIN

# 理论最优超参（由 λ 直接算出来，不需要试）
LR_SGD_BEST = 2 / (LMAX + LMIN)                             # 两个方向同速衰减
LR_MOM_BEST = 4 / (math.sqrt(LMAX) + math.sqrt(LMIN)) ** 2  # 重球法最优步长
BETA_MOM_BEST = ((math.sqrt(KAPPA) - 1) / (math.sqrt(KAPPA) + 1)) ** 2


def sgd_rate(lr):
    """普通 SGD 在单个特征方向上的每步衰减率 |1 - lr·λ|，取最慢的那个。"""
    return max(abs(1 - lr * LMIN), abs(1 - lr * LMAX))


def momentum_rate(lr, beta):
    """重球法：v ← βv + g，θ ← θ - lr·v。
    在单个特征方向上，误差满足 μ² - (1 + β - lr·λ)·μ + β = 0，取最慢的根。
    """
    worst = 0.0
    for lam in (LMIN, LMAX):
        c = 1 + beta - lr * lam
        disc = c * c - 4 * beta
        if disc < 0:
            worst = max(worst, math.sqrt(beta))          # 复根：|μ| = √β，与 λ 无关
        else:
            worst = max(worst,
                        abs((c + math.sqrt(disc)) / 2),
                        abs((c - math.sqrt(disc)) / 2))
    return worst


def run(make_optimizer):
    """跑一次训练。loss 在 optimizer.step() 之前取，保证 loss 与参数标注的步数一致。"""
    model = make_model()
    opt = make_optimizer(model)
    first_hit, log = None, {}
    for k in range(MAX_STEPS + 1):
        loss = F.mse_loss(model(x), y_true)
        value = loss.item()
        if first_hit is None and value < TARGET:
            first_hit = k
        if k in LOG_AT:
            log[k] = value
        if k == MAX_STEPS:
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
    return first_hit, log


def report(title, theory, make_optimizer):
    first_hit, log = run(make_optimizer)
    hit = ("第 %d 步" % first_hit) if first_hit is not None else "**未达到**"
    print("\n" + "-" * 84)
    print("%s   理论每步衰减 %.4f   首次 loss < 1e-8: %s" % (title, theory, hit))
    print("-" * 84)
    line = "  ".join("%d:%s" % (k, ("%.1e" % log[k]) if log[k] > 0 else "0") for k in LOG_AT)
    print("  " + line)


def main():
    print("=" * 84)
    print("这道题的地形")
    print("=" * 84)
    print("λmin = %.6f   λmax = %.6f   条件数 κ = %.2f" % (LMIN, LMAX, KAPPA))
    print("SGD 稳定上限        2/λmax            = %.4f" % (2 / LMAX))
    print("SGD 最优步长        2/(λmax+λmin)    = %.4f   → 每步衰减 %.4f"
          % (LR_SGD_BEST, sgd_rate(LR_SGD_BEST)))
    print("动量 稳定上限        2(1+β)/λmax      = %.4f (β=0.9)" % (2 * 1.9 / LMAX))
    print("动量 最优           lr = %.4f, β = %.4f → 每步衰减 %.4f"
          % (LR_MOM_BEST, BETA_MOM_BEST, momentum_rate(LR_MOM_BEST, BETA_MOM_BEST)))
    print("动量在复根区：两个方向的衰减率都变成 √β，与 λ 无关（β=0.9 → %.4f，β=%.3f → %.4f）"
          % (math.sqrt(0.9), BETA_MOM_BEST, math.sqrt(BETA_MOM_BEST)))
    print("→ 这就是动量治扁椭圆的机制：它把条件数“抹平”了，代价是衰减率被 β 卡住。")

    print("\n" + "=" * 84)
    print("赛跑（同一个初始点，同一批数据）")
    print("=" * 84)

    report("SGD        lr=0.01      （你原来的设置）",
           sgd_rate(0.01), lambda m: torch.optim.SGD(m.parameters(), lr=0.01))
    report("SGD        lr=0.1",
           sgd_rate(0.1), lambda m: torch.optim.SGD(m.parameters(), lr=0.1))
    report("SGD        lr=%.4f  （理论最优）" % LR_SGD_BEST,
           sgd_rate(LR_SGD_BEST), lambda m: torch.optim.SGD(m.parameters(), lr=LR_SGD_BEST))
    report("Momentum   lr=0.1,  β=0.9",
           momentum_rate(0.1, 0.9), lambda m: torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9))
    report("Momentum   lr=%.4f, β=%.4f （理论最优）" % (LR_MOM_BEST, BETA_MOM_BEST),
           momentum_rate(LR_MOM_BEST, BETA_MOM_BEST),
           lambda m: torch.optim.SGD(m.parameters(), lr=LR_MOM_BEST, momentum=BETA_MOM_BEST))
    report("Adam       lr=0.1",
           float("nan"), lambda m: torch.optim.Adam(m.parameters(), lr=0.1))
    report("Adam       lr=0.001     （PyTorch 默认量级）",
           float("nan"), lambda m: torch.optim.Adam(m.parameters(), lr=0.001))

    print("\n" + "=" * 84)
    print("自己动手：")
    print("  1. Momentum 的 β 从 0.55 扫到 0.99，看实测步数是不是呈 √β 那条曲线；")
    print("     β 太大（0.99）反而更慢——为什么？")
    print("  2. Adam 那两行：lr=0.1 快但抖、lr=0.001 稳但慢，最后都没到 1e-8。")
    print("     为什么 Adam 的步长不随梯度变小而变小？这对大模型训练意味着什么？")
    print("  3. 给 Adam 加 lr 衰减（每步乘 0.999 或 cosine 退火），看它能不能进 1e-8。")
    print("  4. 把 x 换成 100/200/300（同一道题放大 10^4 倍），看 SGD/动量的最优 lr")
    print("     是否按 1/λmax 缩小，而 Adam 的 lr 几乎不用改——这就是“尺度不变性”。")
    print("=" * 84)


if __name__ == "__main__":
    main()
