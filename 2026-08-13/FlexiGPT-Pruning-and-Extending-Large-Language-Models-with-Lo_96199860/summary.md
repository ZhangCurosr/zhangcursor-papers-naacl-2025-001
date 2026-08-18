---
title: "FlexiGPT-Pruning-and-Extending-Large-Language-Models-with-Lo"
source: https://aclanthology.org/2025.naacl-long.31.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:01:40"
field: "大语言模型压缩与部署"
keywords: ["LLM 剪枝", "权重共享", "LoRA", "块级剪枝", "参数高效微调", "模型扩展"]
innovations: ["基于低秩 SVD 重构的块相似度度量与权重共享基块选择机制", "输出特征归一化小值初始化配合 SVD 差值 LoRA 初始化实现平滑剪枝恢复"]
benchmarks: ["ARC-e", "ARC-c", "PIQA", "WinoGrande", "HellaSwag", "MiniPile PPL"]
---

# 论文速读：FlexiGPT-Pruning-and-Extending-Large-Language-Models-with-Lo

## 一句话总结
本文提出 FlexiGPT，一种基于块级剪枝+低秩权重共享+LoRA适配器恢复的大语言模型压缩方法，在 LLaMA-2/3 和 OPT 系列上实现了 30%/40% 压缩率下的 SOTA 性能；同时可将小模型通过重复块扩展，仅用 0.3% tokens 的训练量显著提升下游表现。

## 研究问题与动机
1. **LLM 端侧部署的内存瓶颈**：大语言模型参数量庞大，难以部署在手机、边缘设备等内存受限平台。
2. **现有剪枝方法容量损失严重且恢复困难**：SliceGPT、ShortGPT、LLM Surgeon 等方法剪枝后性能骤降，虽然后续恢复（如 LoRA 微调）有一定效果，但仍有较大差距。
3. **块级剪枝缺乏有效替换策略**：ShortGPT 通过 Block Influence 分数剪掉低重要性块，但未解决"剪掉后用什么替代"的问题，导致容量不可逆损失。
4. **小模型性能提升成本高昂**：扩展小规模 LLM 通常需要大量额外预训练数据，资源开销巨大。

## 核心贡献（创新点）
1. **低秩权重共享机制**：为每个被剪块选取一个相似的非剪块作为共享基础，辅以 LoRA 适配器微调差异，本质上是一种"复用+微调"的容量恢复策略，区别于 ShortGPT 仅剪不补的纯删减范式。
2. **基于低秩 SVD 重构的块相似度度量**：提出用 rank-r SVD 近似计算块间距离来选择共享基块，避免所有剪块趋向选择同一候选块的退化现象，与朴素 Frobenius 范数相比能更好地保留块多样性。
3. **输出特征归一化初始化策略**：对共享块的 MHSA/MLP 输出施加 LayerNorm，并将缩放系数 γ 初始化为极小值，类比 LoRA 的 B=0 初始化，确保剪后 PPL 不出现剧烈跳变，实现平滑过渡学习。
4. **基于 SVD 差值的 LoRA 初始化**：将被剪块与共享基块的权重差做低秩 SVD 分解，将前 r 个奇异向量/值直接作为 LoRA A、B 矩阵的初始值，减少训练初期扰动；区别于 DoRA 在权重空间做分解，本方法在特征空间完成归一化。
5. **模型扩展能力**：将同一框架反向应用于小模型（TinyLLaMA 22 层→36 层），通过重复块+独立适配器+归一化，以仅 0.3% 额外 token 实现全维度性能提升，证明方法的双向通用性。

## 方法详解
- **Block Influence (BI) 剪枝打分**：$\mathrm{BI}_i = 1 - \mathbb{E}_{X,t} \frac{X_{i,t}^\top X_{i+1,t}}{\|X_{i,t}\|_2 \|X_{i+1,t}\|_2}$，衡量相邻块输出方向变化程度；BI 越低表示该块信息变换越小，越适合剪除。
- **权重共享基块选择**：$\hat{W}_i = (U_i\Sigma_i)[1:r](V_i[1:r])^\top$ 为低秩 SVD 重构；距离度量 $d(W_i, W_j) = \|\hat{W}_i - (\hat{W}_j + \Delta_{i-j})\|_F$，其中 $\Delta_{i-j}$ 是两块差的前 r 个奇异成分，取 argmin 即为共享基块；r=256。
- **输出特征归一化**：$h_{norm} = \frac{h - \mu(h)}{\sigma(h)} \times \gamma$，γ 初始化为极小值，沿 hidden dim 计算，使初始输出接近零，训练逐步放大。
- **LoRA 适配器与初始化**：共享块中插入 A、B 两个低秩矩阵，$A = (U_{i-j}\Sigma_{i-j})[1:r]$，$B = (V_{i-j}[1:r])^\top$，即 SVD 差值的前 r 成分，保证初始 $\Delta W$ 尽可能接近真实差异。
- **模型扩展模式**：Block 模式（逐块复制若干次）和 Sequential 模式（整段重复），每个重复块拥有独立适配器和归一化参数，首个重复块外均施加输出归一化。

## 实验与结果
- **数据集/基准**：剪枝用 SlimPajama 1B tokens 做恢复训练；评估用 ARC-e/c、PIQA、WinoGrande、HellaSwag 零样本 + MiniPile PPL；扩展实验用 TinyLLaMA 1.1B + SlimPajama 10B tokens 续训。
- **LLaMA-2 7B（30%）**：FlexiGPT PPL=6.55，超越 ShortGPT(22.76)/ShortGPT+LoRA(6.71)；平均零样本 62.68%，在 ARC-c/PIQA/WinoG/HellaS 四个基准上均获最高分。
- **LLaMA-2 7B（40%）**：PPL=7.35，在全部 6 个基准上均为最优。
- **LLaMA-3 8B（30%/40%）**：PPL 分别为 8.67 / 10.25，大幅优于 ShortGPT（1.4e4 / 9.1e4）；短 GPT+FT 在更大模型上几乎崩溃，凸显本方法对敏感模型的价值。
- **OPT 6.7B/1.3B**：30% 时 PPL 分别为 8.39 / 10.81，均优于 ShortGPT+FT（8.66 / 11.04）。
- **扩展实验**：TinyLLaMA 22 层→36 层，Block 模式 PPL=6.73（基线 6.84），全部 6 个零样本基准均有提升。
- **消融**：去掉高秩剪枝 PPL→6.77；去掉输出归一化 Start PPL 暴增至 8648.94，最终 PPL 6.68；去掉 LoRA 初始化 PPL→6.63；完整方法 PPL=6.55。
- **吞吐**：相较未剪枝模型 compute 增加约 5%，吞吐量略降约 5%；实现 self-speculative decoding 后较朴素 FlexiGPT 加速 30.11%。

## 相关工作脉络
1. **ShortGPT（Men et al., 2024）**：块级剪枝的开山工作，使用 BI 分数识别冗余块；本文在其剪枝基础上引入容量恢复机制，本质区别在于"剪而不舍"——剪掉的块由权重共享+适配器重建。
2. **SliceGPT（Ashkboos et al., 2024）**：对权重矩阵做正交投影后删除行/列，属于元素级结构化剪枝；本文针对 Transformer block 级粒度，并引入可恢复机制。
3. **LLM Surgeon（van der Ouderaa et al., 2024）**：迭代更新权重和结构，成本较高；本文在单次 BI 打分+剪枝流程内完成恢复，更高效。
4. **LoRA（Hu et al., 2021）**：经典 PEFT 方法；本文将其用于块级共享后的差异适配器，而非传统任务微调场景，应用范式不同。
5. **DoRA（Liu et al., 2024a）**：在权重空间分解幅度与方向进行微调；本文虽也有归一化环节，但在特征空间而非权重空间操作，服务于剪枝恢复目标。
6. **Subformer / MobileLLM**：早期权重共享工作，前者无适配器且共享比例固定，后者直接整体共享无剪枝；本文结合剪枝选择性共享+可学习适配器，灵活度更高。

## 局限性与未来方向
1. **计算效率未提升**：剪枝后存储参数量减少，但推理 FLOPs 与未剪枝模型基本持平，内存友好但计算不友好。
2. **需后剪枝恢复训练**：依赖 1B tokens 微调恢复性能，增加额外计算开销与时间，不属于 one-shot 剪枝方法。
3. **实验覆盖模型有限**：仅在 LLaMA-2/3、OPT、TinyLLaMA 上验证，其他架构（如 Mixtral、Qwen）的泛化性未检验。
4. **未来方向**：与 speculative decoding（如 Medusa、Jacobi）、early-exit 策略结合降低推理延迟；扩展至更多架构与极低压缩率（50%+）场景验证；探索自投机解码中的 draft 模型设计以进一步提升吞吐。

## 研究启发与可借鉴点
1. **SVD 差值初始化 LoRA 的思路可迁移**：将块间权重差做低秩 SVD 并直接作为适配器初值，比随机初始化更快收敛、更稳定，可推广至其他结构剪枝或层替换场景。
2. **输出特征归一化小值初始化策略**：类比 LoRA 的 B=0，将归一化缩放系数设为极小值以实现"渐进式激活"，是一种值得复用的稳定性技巧，可用于任何需要"软替换"模块的场景。
3. **低秩近似过滤噪声后再做相似度匹配**：先用 r 维 SVD 过滤高秩噪声再计算块距离，既保留功能关键分量又避免全量权重比较的计算与退化问题，可推广至任意模块相似度度量。
4. **剪枝+扩展双向统一框架**：同一套权重共享+适配器机制既能用于压缩也能用于扩展，说明其抽象层面捕捉了 Transformer 块间的函数相似性，可作为通用"块复用"范式的起点。
5. **self-speculative decoding 搭配剪枝模型**：在 draft 阶段使用未带共享替换层的轻量架构（等同于 ShortGPT），verification 阶段用完整模型，以几乎零额外参数代价获得 30% 解码加速，是工程落地的有效思路。

## 关键术语表
**Block Influence (BI)**：衡量 Transformer 相邻块输出方向差异的分数，越高表示块对信息的变换越大，越低越适合剪除。
**Low-Rank SVD Reconstruction**：对权重矩阵截取前 r 个奇异值及对应左右奇异向量，保留主成分以获取低秩近似，用于块相似度计算和适配器初始化。
**Weight Sharing with Adapters**：被剪块使用来自模型其他位置相似块的权重作为基础，并通过可学习的 LoRA 低秩适配器补充差异信息。
**Output Feature Normalization**：对共享块 MHSA/MLP 的输出施加 LayerNorm，并将缩放系数 γ 初始化为极小值，确保训练初期扰动最小化。
**Self-Speculative Decoding**：以轻量 draft 模型（不含权重共享层）生成候选 token，再用完整 FlexiGPT 模型验证，实现近似无损的解码加速。
**FlexiGPT Block/Sequential Extension**：两种小模型扩展模式，Block 指逐块复制，Sequential 指整段连续重复，均附带独立适配器和归一化参数。
**SlimPajama**：Cerebras 清理去重的 627B token 英文语料，本文用作剪枝恢复（1B tokens）和模型扩展（10B tokens）的训练数据。
**MiniPile**：The Pile 的一个子集挑战数据集，用于评估压缩模型的困惑度（PPL）。

## 可复现要素
- **数据集**：SlimPajama（Apache 2.0）、ARC（CC BY-SA 4.0）、PIQA（AFL 3.0）、HellaSwag（MIT）、WinoGrande（Apache 2.0）、MiniPile（MIT）均已公开；Pile 原数据公开但本文仅使用其 MiniPile 子集。
- **代码开源状态**：论文未明确声明 GitHub 仓库或开源链接，需自行实现；附录未附代码 URL。
- **模型**：LLaMA-2 7B、LLaMA-3 8B、OPT 1.3B/6.7B、TinyLLaMA 1.1B 均为公开模型，需分别按对应许可获取。
- **关键超参**：LoRA rank r=256；学习率 0.004，cosine decay；batch size=2/GPU，梯度累积至 total batch=480；恢复训练 1B tokens（剪枝）/10B tokens（扩展）；FSDP + FP16 混合精度；4×A100 80GB。
- **硬件**：4× NVIDIA A100 80GB GPU，约 192 GPU 小时/实验。
