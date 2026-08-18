---
title: "From-Redundancy-to-Relevance-Information-Flow-in-LVLMs-Acros"
source: https://aclanthology.org/2025.naacl-long.115.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:26:56"
field: "多模态大模型可解释性"
keywords: ["LVLM可解释性", "信息流分析", "LLaVA-CAM", "视觉token截断", "注意力分析", "多模态推理"]
innovations: ["提出注意力分数与LLaVA-CAM结合的双视角信息流分析方法", "发现并验证跨模型浅层收敛深层发散的信息流普遍模式", "定义信息流悬崖层并通过截断实验证实深层视觉token高度冗余"]
benchmarks: ["ScienceQA", "POPE", "TextVQA", "CHAIR"]
---

# 论文速读：From-Redundancy-to-Relevance-Information-Flow-in-LVLMs-Across-Reasoning-Tasks

## 一句话总结
本文提出了一种结合注意力分数与 LLaVA-CAM 的信息流分析方法，揭示了多类 LVLMs 在推理过程中视觉信息流的普遍规律：信息流在浅层/中层收敛后于深层发散，存在一个"信息流悬崖层"，此后图像 token 高度冗余且对答案贡献极低。

## 研究问题与动机
1. LVLMs 将图像转化为大量视觉 token 后，其与文本 token 的动态交互机制尚不明确，"黑箱"特性阻碍了对推理过程的深入理解。
2. 现有基于注意力分析的工作（如 OPERA、FastV）多关注幻觉检测或推理加速，对图像与文本间的信息流动规律缺乏系统性刻画。
3. 仅依赖注意力分数无法完整反映图像 token 的贡献——注意力分数来自前向传播，缺乏梯度信号，不能揭示模型决策过程；需要结合反向传播的梯度信息来补充分析。
4. 不同推理任务（复杂推理 vs. 通用推理 vs. 图像描述）对图像信息的依赖深度可能存在差异，亟需量化分析以确定各层的贡献变化。

## 核心贡献（创新点）
1. **提出结合注意力分数与 LLaVA-CAM 的信息流分析框架**：前者揭示前向传播中的相关区域，后者通过反向传播捕获梯度变化以量化图像特征对答案的贡献，二者互补。
2. **发现并验证了跨模型的"浅层收敛、深层发散"信息流模式**：在 LLaVA1.5、Qwen-VL、Intern-VL、Shikra、InstructBLIP2、LLaVA1.6 等多个 LVLM 上，视觉信息流均在浅层或中层汇聚，之后趋于稀疏。
3. **定义并定位了"信息流悬崖层"（i-f cliff layer）**：首次系统性地将信息流急剧下降的分层位置量化为可观察指标，并证明其随任务复杂度变化。
4. **通过图像 token 截断实验验证了该模式的泛化性**：在悬崖层之后完全截断图像 token，模型准确率保持不变甚至略有提升，证实了深层视觉 token 的高度冗余性。
5. **揭示了任务复杂度与信息流分布层数的关联规律**：复杂推理任务（如 ScienceQA）的信息流发散较早，而简单任务（如图像描述）则可维持更深的视觉注意力。

## 方法详解
1. **LLaVA-CAM 构建**：受 Smooth-CAM 启发，对 LLM 解码器的最后一层特征图 $A_k$ 计算 logit 梯度 $G_k = \partial z_c / \partial A_k$，经 ReLU 激活生成热力图 $Heat_{cam} = \text{ReLU}(\sum_k \alpha_k A_k)$，再通过添加高斯噪声生成多个扰动样本并取平均，最后做最大归一化并叠加到原图上，可视化图像区域对答案的正向贡献。
2. **注意力分数聚合定义**：将输出 token 对所有输入 token 的注意力按系统 token（$S$）、图像 token（$T$）、用户 token（$U$）三类聚合，计算图像 token 总注意力占比 $\lambda_{\text{img}}^j = \sum_{j \in T} A_{i,j}$。
3. **信息流悬崖层判定**：结合 LLaVA-CAM 热力图强度衰减与注意力分数下降，确定视觉信息贡献显著降低的首个深层位置即为悬崖层。
4. **注意力分数截断策略**：对第 $l$ 层注意力矩阵按头平均后，取图像 token 对应列段 $A_{\text{img}}$，用 $\arg\text{top}(A_{\text{img}}, k)$ 选出 top-$k$ 保留位置（$k=0$ 即完全截断），验证截断后模型输出变化。
5. **跨模型泛化验证**：对比不同 projector 类型（MLP、Cross-attention、Linear）下的截断实验，确认该收敛模式与 projector 无关，更多源于底层 LLM 的推理结构特性。

## 实验与结果
1. **数据集**：ScienceQA（复杂推理）、POPE（幻觉评估）、TextVQA（OCR 推理）、CHAIR（图像描述幻觉）。
2. **模型覆盖**：LLaVA1.5-7B、Qwen-VL-7B、Intern-VL-7B、Shikra、InstructBLIP2、LLaVA1.6，均基于 CLIP 视觉编码器 + 不同 projector + LLM。
3. **主要结果**：
   - LLaVA1.5 在 ScienceQA 上：基础 Acc 65.00%，悬崖层（第12层）截断后 Acc 66.48%（↑1.48）；POPE Acc 84.70→85.51（↑0.81）；TextVQA Acc 59.34→60.02（↑0.68）。
   - Qwen-VL 在 POPE 上：Acc 80.81→81.13（↑0.32）；F1 77.29→77.82（↑0.53）。
   - Intern-VL 在 POPE 上：Acc 94.01→94.58（↑0.57）；F1 86.31→87.21（↑0.90）。
   - InstructBLIP2 在 CHAIR 上：CHAIRs 24.50→22.80（↓1.70，幻觉减少）。
4. **核心结论**：所有模型在各自悬崖层之后截断图像 token，性能均等于或优于基线，验证了"从冗余到相关"的信息流模式具有跨模型泛化性；任务越复杂，悬崖层越靠前（ScienceQA 约第12层，TextVQA/POPE 约第18-24层，图像描述约第30层）。

## 相关工作脉络
1. **OPERA（Huang et al., 2024）**：通过注意力图检测 LVLM 幻觉，是首个可视化多模态幻觉的工作；本文与其区别在于不止关注特殊符号后的注意力异常，而是系统刻画全层的视觉信息流动态。
2. **FastV（Chen et al., 2024a）**：发现深层视觉 token 注意力计算效率极低并据此做 token 剪枝；本文进一步通过梯度+注意力双视角揭示了该现象的普遍规律并定义了悬崖层概念。
3. **Label Words are Anchors（Wang et al., 2023a）**：从信息聚合角度分析 ICL/CoT 中 label word 的锚定作用；本文将其思路迁移至多模态场景，刻画图像 token 与文本 token 间的贡献流动。
4. **DOPRA（Wei & Zhang, 2024）**：研究 Transformer 各层输出信息的累积与重分配；本文侧重视觉-文本跨模态信息流，而非纯文本层面的 over-accumulation 问题。
5. **CAM/Grad-CAM 系列（Omeiza et al., 2019）**：传统 CNN 可视化方法；本文将其适配到 LLM 架构，提出 LLaVA-CAM 以适配序列输入与多层 decoder 结构。

## 局限性与未来方向
1. **仅在前向/后向分数层面刻画信息流，未建立严格的因果度量**：注意力分数和梯度相关性不完全等价于信息因果贡献，后续需结合干预实验（如 neuron 删除、feature ablation）进一步验证。
2. **悬崖层定位依赖人工观察与实验调参**：当前通过 LLaVA-CAM 热力图和注意力分数下降趋势目视判定，缺乏自动化、精确的悬崖层检测算法。
3. **未在更大规模模型或多语言场景验证**：实验集中在 7B 级别英文模型，对百倍级模型、多语言指令微调场景的泛化性未做系统评估。
4. **截断策略仅考虑全截断（k=0），未探索选择性保留**：实际应用中可选保留关键图像 token 以在速度与精度之间取得更好平衡，本文未展开。
5. **对第32层"回顾层"（retrospective layer）的解释仍为推测**：需更多消融实验证明该层是否普遍存在于各类 LLM 中，以及其具体功能机制。

## 研究启发与可借鉴点
1. **前向注意力 + 反向梯度的双视角分析方法**可迁移至其他多模态模型（如视频语言模型、音频语言模型）的信息流可解释性研究。
2. **图像 token 截断实验的设计范式**可直接复用于 LVLM 推理加速，为开发轻量级推理管线（如 skip layers、early exit）提供理论依据。
3. **悬崖层随任务复杂度变化的发现**提示可在推理时根据任务类型动态选择保留深度，实现自适应计算分配。
4. **LLaVA-CAM 热力图叠加方法**可用于调试多模态模型的视觉聚焦偏差，辅助发现幻觉根源区域。
5. **跨 projector 类型的一致性验证策略**（MLP/Cross-attention/Linear 对比）可作为后续工作验证新方法普适性的标准实验设计。

## 关键术语表
**信息流（Information Flow）**：图像 token、用户 token 和系统 token 对答案 token 的贡献影响程度，通过注意力分数与梯度共同衡量。
**信息流悬崖层（i-f cliff layer）**：视觉信息流从收敛转为发散的临界层位置，此后图像 token 对最终输出贡献极低。
**LLaVA-CAM**：针对 LLaVA 架构改进的 Grad-CAM 方法，通过 LLM 最后一层特征图的梯度加权生成图像热力图，可视化图像区域对答案的正向贡献。
**注意力分数聚合（Attention-score Aggregation）**：将解码器各层注意力权重按 token 类型（系统/图像/用户）求和，量化各类输入对输出的注意力占比。
**推理加速截断（Truncation for Inference Acceleration）**：在识别出的悬崖层之后丢弃全部或部分图像 token，以降低计算开销而不损失准确率。
**回顾层（Retrospective Layer）**：论文推测的第32层左右，模型在此层重新聚焦图像和提示以完成最终答案生成的深层。

## 可复现要素
- **数据集**：ScienceQA、POPE、TextVQA、CHAIR（均为公开数据集）。
- **代码**：论文声明源码开源，地址 https://github.com/zhangbaijin/From-Redundancy-to-Relevance（链接在摘要末尾给出，论文未提供完整 URL）。
- **模型**：LLaVA1.5-7B、Qwen-VL-7B、Intern-VL-7B、Shikra、InstructBLIP2、LLaVA1.6（均为公开模型）。
- **关键超参**：高斯噪声标准差 $noise_s$（未明确给出具体数值）；扰动样本数 $N$（未明确给出）；截断实验中 $k=0$（全截断）。
- **硬件**：A100 GPU。
- **推理设置**：greedy search。
