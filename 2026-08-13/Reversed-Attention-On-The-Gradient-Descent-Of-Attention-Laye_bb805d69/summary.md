---
title: "Reversed-Attention-On-The-Gradient-Descent-Of-Attention-Laye"
source: https://aclanthology.org/2025.naacl-long.52.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:37:42"
field: "大语言模型可解释性"
keywords: ["Reversed Attention", "反向传播可解释性", "注意力层梯度", "Attention Patching", "因果中介", "mechanistic interpretability"]
innovations: ["发现注意力层 softmax 导数隐含构成逆向注意力矩阵，首次系统化分析 GPT 注意力反向传播动力学", "提出 Attention Patching 方法，在不更新权重的前提下将平均 RA 图注入前向传播以编辑模型预测"]
benchmarks: ["GPT2-xl ICL tasks", "Llama2-7B natural language tasks", "OPT-1.3B", "GPT-j"]
---

# 论文速读：Reversed-Attention-On-The-Gradient-Descent-Of-Attention-Laye

## 一句话总结
本文首次对 GPT 中注意力层反向传播的数学过程进行完整分析，发现反向传播隐含计算了一个类似正向注意力的下三角矩阵，命名为"逆向注意力（Reversed Attention, RA）"；利用 RA 可高效定位关键注意力头，并通过"注意力 patching"在不更新参数的情况下直接修改模型预测。

## 研究问题与动机
- Transformer 的注意力机制（前向传播）已被广泛研究，但反向传播过程中注意力层的梯度动力学几乎未被探索，是一个明确的文献空白。
- 前向注意力（Forward Attention, FA）已被证明无法可靠解释模型行为（Jain & Wallace, 2019），需要更精确的局部化方法。
- 现有可解释性主流方法如因果中介分析（Causal Mediation, CM）计算开销大（需对每个注意力头单独做前向传播），缺乏轻量替代方案。
- 如何在不更新权重的情况下，通过干预注意力图来修改语言模型的预测，仍缺乏系统研究。

## 核心贡献（创新点）
1. **提供了 GPT 注意力层反向传播的完整数学推导**：系统分析了输出投影矩阵 $\hat{W}_o$、值矩阵 $\hat{W}_v$、查询矩阵 $\hat{W}_q$ 和键矩阵 $\hat{W}_k$ 的 VJP 与梯度更新公式。
2. **提出"逆向注意力（Reversed Attention）"概念**：发现 softmax 导数在反向传播中隐含构成一个下三角注意力图 R，与正向注意力 A 在结构和计算方式上高度相似。
3. **利用 RA 实现高效的可解释性评估**：通过扰动实验（AUC 指标）证明 RA 在识别关键注意力头上可与 Causal Mediation（CM）竞争，且计算效率更高（仅需一次前向+一次反向）。
4. **提出"注意力 patching（Attention Patching）"新方法**：无需更新模型参数，直接将训练样本的平均 RA 图注入测试样本的前向注意力中，实现模型行为的定向编辑，效果可与多 shot ICL 提示媲美。

## 方法详解
- **GPT 注意力前向传播基础**：输入序列 $X \in \mathbb{R}^{n \times d}$，每个注意力头 $l$ 的查询、键、值分别为 $Q^l = X\hat{W}_q^l$、$K^l = X\hat{W}_k^l$、$V^l = X\hat{W}_v^l$，前向注意力图 $A^l = \mathrm{softmax}(Q^l K^{l\top}/\sqrt{d/h} + M)$，其中 $M$ 为因果下三角掩码。
- **$\hat{W}_o$ 的梯度**：VJP $\delta_o^j$ 直接从注意力块末端的残差流获得；单次 token 更新为 $\hat{W}_o \leftarrow \hat{W}_o - \eta \delta_o^j \times x_o^{j\top}$；该更新使得原输出沿 $\delta_o^j$ 方向偏移。
- **$\hat{W}_v$ 的 VJP**：基于因果性，每个 token $j$ 的 VJP 是后续所有 token 误差信号 $e^l = \delta_o^l \hat{W}_o^\top$ 的加权求和：$\delta_v^j = \sum_{l=j}^n A_{l,j} e^l$。
- **Softmax 导数与 RA 的推导**：先计算中间量 $\tilde{e}^j = \delta_o^j \hat{W}_o^\top V^\top \in \mathbb{R}^n$（类似于 logits），然后：$r^j = A_j \odot (\tilde{e}^j - \tilde{e}^j A_j \cdot \mathbf{1}) \sqrt{h/d}$。批量形式：$R = A \odot (\tilde{E}^\top - \mathrm{diag}(A\tilde{E}^\top))^\top \sqrt{h/d}$，其中 $R$ 即为**逆向注意力矩阵**。
- **$\hat{W}_q$ 与 $\hat{W}_k$ 的 VJP**：$\delta_q^j = R_j K$（RA 矩阵乘以正向键矩阵），$\delta_k^j = R_j^\top Q$（RA 的转置乘以正向查询矩阵）。
- **RA 作为注意力补丁的方法**：对训练集所有样本计算 RA，按注意力头取平均得到每个头的平均 RA 图；对测试样本，将平均 RA 乘以学习率 $\alpha$（实验中 $\alpha = 30$，使用负号方向）后注入各头的前向注意力图中，不更新任何权重。

## 实验与结果
- **模型与数据集**：GPT2-xl、OPT-1.3B、GPT-j、Llama2-7B；来自 Todd et al. (2023) 的 15 个 ICL 任务 + 来自 Hernandez et al. (2024) 的自然语言任务（country-capital、person-plays-pro-sport 等），共 21 个任务。
- **扰动测试（AUC 评估）**：将注意力头按不同方法排序后逐步解封，衡量模型准确率曲线下面积。关键结果：
  - **GPT2-xl ICL 任务**（Table 1）：RA 在多个任务上优于 FA 和随机排序，与 CM 竞争力强。例如 5-shot antonym：RA=0.32 vs CM=0.27；5-shot person-sport：RA=0.44 vs CM=0.34。
  - **自然语言任务**（Table 2）：RA 在 Llama2-7B 上 country-capital 5-shot 达到 0.43 vs CM 0.39；person-plays-pro-sport 5-shot 达到 0.48 vs CM 0.39。
  - 总体趋势：大模型且具备较高基础准确率时，RA 显著优于 CM；在少 shot ICL 失败的任务上 CM 表现较好。
- **注意力 Patching 实验**（Table 3）：
  - GPT2-xl capitalize 5-shot：原始 0.98 → RA patching 达 1.00；antonym 5-shot：原始 0.53 → RA patching 达 0.62。
  - OPT-1.3B capitalize 5-shot：原始 1.00 → RA patching 保持 1.00；antonym 5-shot：原始 0.42 → RA patching 达 0.59。
  - RA patching 在不提供示例（0-shot）的情况下即可接近甚至达到 5-shot ICL 提示的效果。
- **最强结果**：GPT2-xl capitalize 10-shot ICL 原始准确率 0.99，RA patching 达到 1.00；在多项 ICL 任务上，RA patching 将 0-shot 性能提升至接近 5-10 shot ICL 水平。

## 相关工作脉络
- **Vig et al. (2020)、Meng et al. (2022)**：因果中介分析（CM）是当前 LM 可解释性的主流方法，通过前向扰动注意力头输出测量间接效应；RA 作为更轻量的替代方案，仅需一次反向传播即可获得头部重要性排名。
- **Jain & Wallace (2019)、Serrano & Smith (2019)**：指出前向注意力图不可靠，不能作为模型行为的解释依据；本文在此基础上进一步利用反向传播信息构造更具判别力的 RA 图。
- **Zhang & Nanda (2023)、Todd et al. (2023)**：激活补丁（activation patching）通过将一模型的中间状态注入另一模型来研究可解释性；本文提出将"注意力分数图"本身作为可注入对象，无需修改权重。
- **Katz et al. (2024)**：发现梯度可被解释为 token 嵌入；本文进一步深入到注意力层内部，揭示 softmax 导数隐含的注意力图结构。
- **Dai et al. (2023)、Mahankali et al. (2023)**：在简化模型（单层线性注意力）上分析 GD 与 ICL 的关系；本文工作扩展到全规模多头注意力且无简化假设。

## 局限性与未来方向
- 数学推导仅针对不带 RoPE 和稀疏注意力等的解码器-only 模型，对于带旋转位置编码（如 Llama2、GPT-j）的模型，实验层面做了验证但理论框架未完全推广。
- 实验使用的 21 个任务相对简单，本文自评仅为概念验证（proof of concept），RA 在复杂推理或长上下文任务上的有效性尚待检验。
- 注意力 patching 要求样本具有相同长度和格式，限制了其在自由文本生成场景中的直接应用。
- 与 CM 的比较仅涉及基础实现，其他梯度-based 解释方法未纳入对比。
- 未来方向：探索更复杂的注意力图注入策略（无需固定长度）、将 RA 应用于长程 entity tracking 和因果电路发现、将 RA 思想推广至 encoder-decoder 和带 RoPE 的模型架构。

## 研究启发与可借鉴点
1. **"逆向注意力"的分析范式**可迁移至其他注意力变体（如 linear attention、sparse attention），用于理解不同架构中反向传播的隐式注意力结构。
2. **扰动排序 + AUC 评估**的实验设计是一种标准化、可复现的头部重要性量化方法，可广泛应用于各类模型的 mechanistic interpretability 研究。
3. **注意力 patching（无需权重的干预）**提供了一种介于 prompt engineering 和 fine-tuning 之间的新干预范式，可在不损害预训练知识的前提下定向调整模型行为，适合后续对齐/编辑研究。
4. **从 softmax 导数中提取结构化图**的思路可推广至其他非线性模块（如 LayerNorm 梯度、MLP 激活梯度），挖掘反向传播中隐含的可解释信号。
5. **RA 的稀疏性**（相比 FA 更易解读）提示：反向传播图可能天然具有更强的语义聚焦能力，值得在归因分析和因果发现中进一步探索。

## 关键术语表
- **Reversed Attention (RA)**：通过注意力层 softmax 导数在反向传播中隐式计算得到的下三角注意力图，反映梯度下降试图调整哪些查询-键对的关系。
- **Vector-Jacobian Product (VJP)**：反向传播中从损失向某一中间变量传递的错误信号向量，是计算该变量梯度的核心中间量。
- **Causal Mediation (CM)**：通过在前向传播中扰动模型组件输出并观测下游预测变化的因果中介分析方法，是 LM 可解释性的主流技术。
- **Attention Patching**：本文提出的新方法，将训练样本的平均 RA 图注入测试样本的前向注意力计算中，无需更新模型权重即可修改预测。
- **Perturbation Test (AUC)**：将注意力头按不同重要性排序后逐步解封，以模型准确率曲线下面积（AUC）量化排序方法的有效性。
- **In-Context Learning (ICL)**：通过在 prompt 中提供若干示例让模型学习任务模式，RA patching 可无需示例达到类似效果。

## 可复现要素
- **数据集**：来自 Todd et al. (2023) 的 ICL 任务和 Hernandez et al. (2024) 的自然语言任务（论文中附有完整任务列表）。
- **模型**：GPT2-xl、OPT-1.3B、GPT-j、Llama2-7B，均通过 HuggingFace transformers 获取。
- **代码**：论文已开源，地址 https://github.com/shacharKZ/Reversed-Attention。
- **关键超参**：attention patching 的注入学习率 $\alpha = 30$（对 RA 使用负方向）；FA patching 使用 $\alpha = 1$；每任务使用 25 个训练样本平均 RA 图。
- **硬件**：Nvidia A40 系列 GPU。
