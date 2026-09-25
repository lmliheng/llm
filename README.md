|仓库|地址|亮点|
|---|---|---|
|**Hello-LLM-FineTuning**​|github.com/lailoo/Hello-LLM-FineTuning|2026年新出，中文全栈教程，覆盖 LoRA/QLoRA/Adapter/P-Tuning，理论+代码+部署，从零到进阶的完整体系|
|**llm-cookbook**​|github.com/datawhalechina/llm-cookbook|DataWhale 出品，吴恩达大模型课程中文复现，含模型微调章节|
|**Happy-LLM**​|github.com/datawhalechina/happy-llm|手写实现 Transformer，从零理解原理再动手微调|

|仓库|地址|适用场景|
|---|---|---|
|**LLaMA-Factory**​|github.com/hiyouga/LLaMA-Factory|一站式微调 UI+CLI，支持 LLaMA/Qwen/ChatGLM 等，LoRA/QLoRA 全搞定，小白友好|
|**Unsloth**​|github.com/unslothai/unsloth|极致显存优化，12GB 显存就能微调 7B 模型，速度快 2 倍|
|**Axolotl**​|github.com/OpenAccess-AI-Collective/axolotl|YAML 配置驱动，支持多 GPU、DeepSpeed，工程化训练首选|
|**TRL**​|github.com/huggingface/trl|HuggingFace 官方 RLHF/DPO/PPO 对齐库，做偏好优化必学|
|**PEFT**​|github.com/huggingface/peft|参数高效微调核心库，LoRA/Prefix Tuning/Adapter 等全收录|


| 仓库                              | 地址                                             | 亮点                                           |
| ------------------------------- | ---------------------------------------------- | -------------------------------------------- |
| **nn-zero-to-hero**​            | github.com/karpathy/nn-zero-to-hero            | 李飞飞弟子 Karpathy 出品，纯 NumPy 实现神经网络，配套视频，零框架学底层 |
| **d2l-zh / d2l-en**​            | github.com/d2l-ai/d2l-zh                       | 《动手学深度学习》配套 Notebook，每节可运行代码，伯克利教材级别         |
| **fastbook**​                   | github.com/fastai/fastbook                     | fastai 官方教材，自顶向下教学，先跑通再深挖                    |
| **DeepLearning-500-questions**​ | github.com/scutan90/DeepLearning-500-questions | 中文 500 问，覆盖数学基础到 CV/NLP，面试+复习利器              |

| 仓库                                  | 地址                                                 | 亮点                                               |
| ----------------------------------- | -------------------------------------------------- | ------------------------------------------------ |
| **handson-ml2**​                    | github.com/ageron/handson-ml2                      | 《Hands-On ML》配套代码，Scikit-Learn + Keras + TF，经典教材 |
| **pytorch/tutorials**​              | github.com/pytorch/tutorials                       | PyTorch 官方教程，迁移学习、NLP、CV 全覆盖                     |
| **introduction_to_ml_with_python**​ | github.com/amueller/introduction_to_ml_with_python | Scikit-Learn 作者亲自写的入门教程                          |

---

## 学习路径建议

> 根据你的目标选一条走完，比收藏 100 个仓库有用得多：

1. **纯新手**​ → `nn-zero-to-hero` → `d2l-zh` → `Hello-LLM-FineTuning`
2. **想快速出成果**​ → `llm-cookbook` → `LLaMA-Factory` 跑通微调 → `Unsloth` 优化显存
3. **做研究/深造**​ → `d2l-zh` → `pytorch/tutorials` → `TRL` + `PEFT` 深入对齐

如果是从零开始，建议先从 `nn-zero-to-hero` 理解神经网络本质，再用 `LLaMA-Factory` 跑一个 QLoRA 微调实战，基本就能上手了。需要我针对某个方向再深入展开吗？