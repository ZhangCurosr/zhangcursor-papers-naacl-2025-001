---
title: "Can-Unconfident-LLM-Annotations-Be-Used-for-Confident-Conclu"
source: https://aclanthology.org/2025.naacl-long.179.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:01:43"
---

# 论文速读：Can Unconfident LLM Annotations Be Used for Confident Conclusions?

## 一句话总结
提出 CONFIDENCE-DRIVEN INFERENCE 方法，利用 LLM 的零样本标注与口头置信度分数自适应选取少量人工标注进行统计估计，在保证置信区间有效性（≥90% coverage）的前提下，将所需人工标注预算降低 25% 以上。

## 研究问题与动机
1. LLM 虽在多项文本标注任务上接近人类水平，但存在人口统计学偏差、事实错误与输出不一致性，直接将其等同于人类数据用于下游统计推断可能产生误导性结论。
2. 计算社会科学（CSS）等研究高度依赖精确的统计估计（如流行率、回归系数），但完全依赖人工标注成本高昂，亟需在成本与推断有效性之间取得平衡。
3. 现有混合标注方法（如 Egami et al., 2024）采用均匀采样或固定权重融合，无法根据单条样本的标注难度/不确定性动态分配有限的人工预算。
4. 缺乏在 LLM 质量不可控时仍能提供严格统计保障（无偏性+有效置信区间）的实用框架，制约了 LLM 标注在科学实证研究中的规模化应用。

## 核心贡献（创新点）
1. 提出 CONFIDENCE-DRIVEN INFERENCE 框架，首次将 LLM 的口头置信度（verbalized confidence）与主动推断理论结合，实现人机标注成本的自适应优化分配。
2. 设计基于置信度-误差映射的采样策略，用轻量黑盒模型预测每条样本的预期标注错误率，据此设定 $\pi_i \propto \sqrt{\widehat{\text{err}}_i(C_i)}$ 的采样概率，优先覆盖 LLM 不确定的样本。
3. 引入幂调优（power tuning）参数 $\lambda$ 构造混合损失估计量，从理论上证明无论 LLM 质量高低，最终估计的均方误差永远不会低于纯人工基线，提供“利用辅助数据绝不更差”的安全边界。
4. 在 politeness、stance、political bias 三个 CSS 场景的五个估计任务上系统验证，方法在维持目标覆盖率的同时显著提升有效样本量，且对模型规模、提示形式与置信度校准噪声具备强鲁棒性。

## 方法详解
1. **两阶段 LLM 提示**：Stage 1 要求 LLM 零样本输出分类标注 $\hat{H}_i$；Stage 2 要求 LLM 为自身答案给出数值概率 $C_i \in [0,1]$，实验表明该口头置信度与相对人类标注的准确率呈良好校准关系。
2. **自适应采样策略**：采集到一批人工标注后，以 $\{ (C_j, (\hat{H}_j - H_j)^2) \}$ 为训练对拟合黑盒误差预测器 $\widehat{\text{err}}_i$（实验中使用 XGBoost），据此设定采样概率 $\pi_i$；对于回归系数估计任务，进一步乘以 $|X_i^\top h|$ 以反映协变量敏感度，并按预算 $n_{\text{human}}$ 归一化。
3. **置信驱动估计量**：构造目标函数 $\hat{\theta}^{\mathrm{conf}} = \arg\min_\theta \frac{1}{n}\sum_{i=1}^n \left(\lambda \hat{\ell}_{\theta,i} + (\ell_{\theta,i} - \lambda \hat{\ell}_{\theta,i})\frac{\xi_i}{\pi_i}\right)$，其中 $\hat{\ell}$ 为 LLM 损失、$\ell$ 为人工损失、$\xi_i$ 为采样指示变量；期望意义下该估计等价于全量人工估计，因而在大样本下对真实参数 $\theta^*$ 无偏。
4. **方差估计与置信区间**：基于 Active Inference 理论推导闭式方差估计 $\widehat{\Sigma}$，计算标准误 $\hat{\sigma}_{\mathrm{se}}$，最终构建 $C_{1-\alpha} = (\hat{\theta}^{\mathrm{conf}} \pm z_{1-\alpha/2}\hat{\sigma}_{\mathrm{se}})$；$\lambda$ 通过最小化 MSE 的闭式解自动选取，确保 $\lambda=0$（纯人工）始终是一个安全下界。

## 实验与结果
1. **数据集与任务**：Politeness（Stack Exchange 与 Wikipedia, $n=5,480$，估计 $\beta_{\mathrm{hedge}}, \beta_{\mathrm{1pp}}$）、Stance（新闻标题, $n=2,300$，估计 $O_{\mathrm{agreement}}$）、Political bias（新闻文本, 随机抽样 $n=2,000$，估计 $p_{\mathrm{left}}, p_{\mathrm{right}}$）；标注模型使用 GPT-4o 与 GPT-3.5 零样本调用。
2. **基线对比**：Human-only（仅人工）、Human+LLM (non-adaptive, $\lambda=1$ 且均匀采样)、LLM-only（直接当黄金数据使用）。
3. **有效样本量增益**：$n_{\mathrm{human}}=500$ 时，置信驱动方法在所有任务
