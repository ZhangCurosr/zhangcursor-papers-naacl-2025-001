---
title: "Temporal-Aware-Soft-Prompt-Tuning-for-Automatic-Text-Dating"
source: https://aclanthology.org/2025.naacl-long.200.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:57:47"
field: "历史文本分析与时间感知NLP"
keywords: ["自动文本年代鉴定", "时间感知表示", "软提示微调", "语义演变", "参数高效微调", "历时语言建模"]
innovations: ["提出ATD-Bert解耦语义变异、不变性与时间上下文特征的时间感知文本表示", "将时间感知表示与三种PEFT软提示架构结合实现高效ATD", "设计基于时间距离指数惩罚的年代分类损失函数"]
benchmarks: ["Twenty-Four Histories", "Royal Society Corpus"]
---

# 论文速读：Temporal-Aware-Soft-Prompt-Tuning-for-Automatic-Text-Dating

## 一句话总结
本文提出 TASPT（Temporal-Aware Soft Prompt Tuning），一种将时间感知文本表示与参数高效软提示微调相结合的方法，用于解决自动文本年代鉴定任务。通过解耦语义变异、语义不变性和时间上下文特征，TASPT 在中文"二十四史"和英文皇家学会语料库上均超越现有所有基线方法。

## 研究问题与动机
1. **历史文本中词义随时间演变**：现有 ATD 方法多依赖静态词嵌入或预训练语言模型，无法有效捕捉跨越长时段的语义变化，例如"寺"在古代指政府机构，现代指佛教寺院。
2. **时间语义信息难以动态整合**：现有方法（如 TALM）采用流水线架构，存在级联误差传播风险；对 PLM 进行时段续训练则需要为每个时期维护独立模型。
3. **LLM 在 ATD 任务上表现不佳**：由于训练语料中历时文本占比有限，且存在时间幻觉（temporal hallucination）问题，开源 LLM 难以有效建立文本与时间标签的关联。

## 核心贡献（创新点）
1. **提出 ATD-Bert 时间感知历史文本表示模型**：通过语义变异学习、语义不变性学习、时间上下文感知学习三个子任务，将历史文本映射到保留时间特征的离散向量空间；区别于既往方法仅停留在词嵌入层面，本方法显式建模语义变化与不变的解耦表示。
2. **设计 TASPT 软提示参数高效微调框架**：将 ATD-Bert 输出的时间感知表示融入 P-tuning v1 LSTM / v2 / Pre-trained 三种变体，实现仅微调 1.4%–5.2% 参数的高效训练；与 TALM 的级联流水线方式本质不同，TASPT 在单一框架内联合优化。
3. **在双数据集上取得 SOTA 并揭示 LLM 局限**：在中文"二十四史"（F1=89.09%）和英文 RSC（F1=61.07%）上超越所有监督方法、LLM 及传统软提示方法，同时证明当前主流 LLM 在 ATD 任务上仍面临严重挑战。

## 方法详解
**1. ATD-Bert 时间感知文本表示（核心预训练）**

给定文本 $d$，经 ATD-Bert 编码得到 [CLS] 表示 $h \in \mathbb{R}^{768}$。ATD-Bert 包含三个子任务：

- **语义变异学习（SVL）**：对两篇文档表示 $h, h'$ 经 MLP 重参数化后，预测其时间间隔 $\Delta \hat{p}_{svl}$，使用 MSE 损失 $\mathcal{L}_{svl}$ 使 temporal gap 大的文本表示区分度更大。
- **语义不变性学习（SIL）**：引入判别器 $G_d$ 与梯度反转层（GRL），通过对抗学习目标迫使 ATD-Bert 提取跨时段共享的不变特征；GRL 在前向时恒等变换，反向时梯度乘以 $-\lambda$。
- **时间上下文感知学习（TCL）**：拼接两文档并执行 MLM，训练模型预测 [MASK] 位置词，借此学习跨时段上下文不变特征。

整体预训练损失：$\mathcal{L}_{\text{ATD-Bert}} = \mathcal{L}_{svl} + \mathcal{L}_{sil} + \mathcal{L}_{tcl}$

最终表示解耦为语义变异 $o_{sv}$ 和语义不变 $o_{si}$：$o_{si} = R(GRL(h)) \oplus \text{ATD-Bert}(d_{[\text{mask}]})$。

**2. 时间感知软提示微调**

将 $o = [\text{MLP}(o_{sv}); \text{MLP}(o_{si})]$ 分别接入三种 PEFT 架构：
- **TASPT Pre-trained**：$P = o \oplus e(B_v)$，虚拟 token 嵌入直接相加
- **TASPT v1 LSTM**：$P = \text{LSTM}(o)$，通过 LSTM 重参数化
- **TASPT v2**：$P = R(\text{MLP}([o; e(B_v)]))$，结合 prefix-tuning 思想

**3. 分类与年代损失**

软提示 $P$ 拼接于输入序列前，经冻结 PLM 后提取 [CLS] 表示，经 softmax 得到预测概率 $p_{td}$。定义新型年代损失：
$$\mathcal{L}_{td} = e^{\tanh((c - \hat{c})^2)} \times \frac{1}{N}\sum_i y_i^\top \log p_{td,i}$$
该损失对偏离真实年代越远施加指数级更大惩罚，捕捉"相近时期语义更相似"的先验。

## 实验与结果
**数据集**：中文"二十四史"（202 B.C.–1911 A.D.，13 个朝代）和英文 Royal Society Corpus（1660–1880，12 个 20 年区间），8:1:1 划分。

**基线**：涵盖传统方法（Bayesian、DPCNN、LSTM）、预训练模型（Hierarchical BERT、Longformer、SBERT、RoBERTa）、时序方法（TALM）、LLM（Qwen2-7B、Meta-Llama3.1-8B、GLM-4-9B）及软提示方法（PPT、P-tuning v1/v2）。

**主要结果**：
- **最强结果**：TASPT v1 LSTM 在"二十四史"上 F1=**89.09%**，在 RSC 上 F1=**61.07%**，全面超越 RoBERTa（87.52%/59.84%）。
- **相较软提示方法**：在"二十四史"上超越 PPT/P-tuning v2/P-tuning v1-LSTM 分别达 19.96%/19.99%/18.83%。
- **LLM 表现**：三大 LLM 在"二十四史"上 F1 仅 2.52%–6.34%，暴露时间幻觉与训练数据不足问题。
- **消融**：移除 SVL 模块 F1 下降最多（中文从 89.09%→47.20%，降幅 41.89%）；SIL 和 TCL 影响较小（分别下降约 0.9%）。

## 相关工作脉络
1. **早期统计/传统方法**：n-gram 语言模型（Jong et al., 2005）、图卷积网络（Vashishth et al., 2019）、LSTM（Yu & Huangfu, 2019）——依赖静态特征，未建模语义演变。
2. **预训练模型方法**：SBERT（Tian & Kübler, 2021）、RoBERTa（Li et al., 2022）——优化语义提取但未显式建模时间维度变化。
3. **TALM（Ren et al., 2023）**：引入时间对齐与适应模块，但为流水线架构，存在级联误差；本文方法在同一框架内联合优化，避免此问题。
4. **历时词嵌入研究**：跨时段对齐（Kulkarni et al., 2015）、全局初始化+时分微调（Di Carlo et al., 2019）——停留在嵌入层面，难以直接融入下游任务。
5. **时间感知预训练**：TALM 之前的工作多关注单词级别变化，本文引入 MLM 与对抗学习联合建模跨时段上下文不变性。
6. **LLM 在 ATD 上的局限**：本文揭示当前 LLM 在历时文本鉴定上存在时间幻觉（Qian et al., 2024）与训练数据偏差，为未来方向提供重要警示。

## 局限性与未来方向
1. **TASPT v2 性能不如 v1 LSTM**：表明将时间感知表示整合到 prefix-tuning 架构时可能存在信息损失，网络形状约束限制了表达力。
2. **SIL 和 TCL 模块影响低于预期**：尽管语言学研究表明语义不变性重要，但消融显示其贡献有限，未来需更深入优化这两类模块。
3. **LLM 在 ATD 上表现糟糕**：论文指出未来应重点改进 LLM 在此任务上的性能，暗示当前 LLM 的时间感知能力严重不足。
4. **数据集覆盖有限**：仅验证于中文史书和英文科学文献，方法的跨领域泛化能力有待进一步检验。

## 研究启发与可借鉴点
1. **语义变异/不变性解耦框架**可迁移至其他需要建模概念漂移的 NLP 任务（如风格分类、领域自适应），通过对比学习 + 对抗学习的组合捕捉动态与静态特征。
2. **梯度反转层（GRL）用于学习时间不变表示**的策略具有普适价值，可应用于跨时期/跨领域特征解耦的其他场景。
3. **Acc@K 误差惩罚损失**（$\mathcal{L}_{td}$ 中对远距预测施加指数惩罚）是一种设计精巧的监督信号，可借鉴于有序分类、时间预测等回归-分类混合任务。
4. **时间感知表示与软提示的结合**为 PEFT 领域开辟了新方向——如何将任务相关知识注入 prompt 而非全参微调，对低资源场景有重要参考价值。
5. **本团队可结合方向**：将 TASPT 的时间感知表征模块与本团队在历史文本分析、语义演变检测等工作结合，或在多语言 ATD、细粒度年代估计等任务上扩展验证。

## 关键术语表
**Automatic Text Dating (ATD)**：自动文本年代鉴定，利用计算模型推断无时间标记的历史文本所属年代的任务。
**Temporal-Aware Soft Prompt Tuning (TASPT)**：本文提出的方法，将时间感知文本表示与参数高效软提示微调相结合的 ATD 框架。
**Semantic Variance Learning (SVL)**：通过对比学习使不同时期文本表示拉开距离，捕捉词义随时间的演变。
**Semantic Invariance Learning (SIL)**：利用对抗学习（GRL+判别器）提取跨时期共享的稳定语义特征。
**Temporal Context-Aware Learning (TCL)**：通过跨文档 MLM 学习时间上下文不变性，增强模型对共现模式的理解。
**Gradient Reversal Layer (GRL)**：前向恒等变换、反向梯度取反的无参层，用于对抗域/时间不变性学习。
**Acc@K**：柔性评估指标，将误差在 ±K/2 范围内的预测视为正确，衡量误差严重程度。
**Parameter-Efficient Fine-Tuning (PEFT)**：仅微调少量参数（如软提示）而冻结主体模型的高效微调范式。

## 可复现要素
- **数据集**：中文"二十四史"（2,647卷）、Royal Society Corpus（9,779篇）；论文已声明代码开源。
- **代码**：https://github.com/coderlihong/TASPT（论文声明）
- **权重**：未明确提及公开权重，代码仓库应包含训练脚本。
- **关键超参**：ATD-Bert 输入长度 256，Embedding 维度 768，λ=1，学习率 3e-5；微调阶段输入长度 512，PLM 冻结，仅微调软提示参数（占总量 1.4%–5.2%）。
