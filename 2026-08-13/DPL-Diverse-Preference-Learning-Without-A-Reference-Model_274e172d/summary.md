---
title: "DPL-Diverse-Preference-Learning-Without-A-Reference-Model"
source: https://aclanthology.org/2025.naacl-long.190.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:03:03"
---

# 论文速读：DPL-Diverse-Preference-Learning-Withou-A-Reference-Model

## 一句话总结
本文提出 DPL（Diverse Preference Learning），一种无需参考模型的离线偏好对齐框架，通过联合 SFT 项、对比式隐式奖励与非似然惩罚三项损失，并依据“基线期望值”与“相对偏好强度”双层分类动态分配权重，使模型能有效学习真实人类偏好数据中的多样性与弱信号。

## 研究问题与动机
- 现有直接对齐方法（DPO/KTO/ORPO/SIMPO）依赖 Bradley-Terry (BT) 模型，假设隐式奖励差即可表征真实偏好，但实际数据中存在大量弱偏好或“同等”评分对（如 Ultrafeedback 中约 17% 被 GPT-4 评定为 reward diff=0），导致策略过拟合或欠拟合。
- KTO 等进阶方法常将连续/细粒度偏好压平为二分类 desirable/undesirable 标签，损失了数据集内在的丰富标注粒度。
- 无参考模型方法虽免去维护 $\pi_{ref}$ 的开销，但在缺乏显式 KL 约束时，BT 模型对无限奖励差的追求易引发策略崩塌（policy degeneracy）与 token 退化。
- 现有算法未系统性利用偏好数据中的绝对质量分布（baseline desirability）与相对强度分布，限制了模型在诚实性、安全性等多维对齐目标上的表现。

## 核心贡献（创新点）
- 提出 DPL 框架，在参考模型自由设定下同时学习响应的基线期望值与相对偏好强度，从理论上规避 BT 模型在弱偏好下的过/欠拟合风险。
- 推导两种隐式奖励 formulation（odds-ratio 与 length-normalized），并构建包含 SFT 交叉熵、对比项与非似然惩罚的三组件联合损失，工程简洁且易于实现。
- 设计双层偏好分类与动态权重分配机制 $\mathcal{H}$，根据样本的基线质量与成对强度自动调制 $(\alpha, \gamma, \eta)$，使模型能区分强偏好、弱偏好与同等偏好信号。
- 在 Ultrafeedback 与 Reddit TL;DR 上完成系统化评测，首次在 Overall/Honesty/Helpfulness/Truthfulness/Safety 多轴向上证明 DPL 持续优于 SFT、ORPO、SIMPO 及 KTO，并在 AlpacaEval 2.0 保持领先。

## 方法详解
- **隐式奖励替代 KL 正则**：无参考模型时，采用 $r^*_{odds}(x,y)=\beta \log \frac{\pi_\theta(y|x)}{1-\pi_\theta(y|x)}$（来自 ORPO）或 $r^*_{norm}(x,y)=\beta \frac{\log \pi_\theta(y|x)}{|y|}$（来自 SIMPO）作为隐式奖励代理。
- **三组件损失**：
  1. $\alpha \cdot L_{SFT}$：对优选响应 $y_w$ 施加交叉熵，防止对劣响应过拟合，并支持使用未配对的优质样本。
  2. $\gamma \cdot \log[\sigma(r^*(x,y_w) - r^*(x,y_l))]$：BT 对比损失，学习成对响应的相对优劣。
  3. $\eta \cdot \log(1 - \pi_\theta(y_l|x))$：非似然损失，直接压制劣响应概率。
- **双层权重分配**：$\mathcal{H}(\mathcal{D}_{pref}) \to (\alpha, \gamma, \eta)$：
  - 第一层（基线期望值）：基于阈值 $T_a=7.0$、$T_p=4.0$（Ultrafeedback）或 expert confidence 百分位（TL;DR）划分 accepted / partially-accepted / rejected。
  - 第二层（相对强度）：分析配对内两样本分差，区分 accepted/rejected、accepted/partially-accepted、nondeterministic（同等偏好）等。
  - 权重调度示例（Table 1）：accepted 对仅启用 $\alpha=1$；accepted/rejected 对启用全三项 $(\alpha=1,\gamma=1,\eta=1)$；nondeterministic 对屏蔽对比与非似然项（$\gamma=0,\eta=0$），避免对无差别样本施加错误梯度。

## 实验与结果
- **数据集与模型**：指令跟随（Ultrafeedback）、新闻摘要（Reddit TL;DR / CNN Daily Mail）；统一使用 Phi-3-Mini-128k-Instruct（3.8B）。
- **评估协议**：GPT-4o Judge 的 FPP-style 多维度评分（Overall/HT/HL/TL/SF）、独立训练的 OPT-1.3B Reward Model 成对胜率、AlpacaEval 2.0 榜单胜率。
- **Ultrafeedback**：DPL-O 整体胜率 21.31%，领先 KTO（17.14%）、ORPO（13.34%）、SIMPO（13.23%）；各维度均夺冠。RM 胜率中 DPL-O 胜 SFT 92.1%、胜 KTO 60.5%、胜 ORPO 68.1%、胜 SIMPO 65.7%。
- **Reddit TL;DR**：DPL-O 整体胜率 22.92%，较 KTO（15.22%）提升约 7.7 点；DPL-O 在控制生成长度的摘要任务上显著优于 DPL-N。
- **AlpacaEval 2.0**：DPL-O 胜率 16.15%，较 KTO 提升约 2 点，较 ORPO/SIMPO 提升约 5 点。
- **消融**：剔除同等偏好样本（DPL-O(-)/DPL-N(-)）导致胜率平均下降 2-4 点，证明动态加权比直接丢弃弱信号更高效；t-test 验证多数对比 p<0.05。

## 相关工作脉络
- **RLHF / PPO**：依赖在线采样与独立 Reward Model，计算昂贵；DPL 属离线直接对齐，无需在线策略优化与外部奖励模型。
- **DPO / IPO / KTO**：基于参考模型 $\pi_{ref}$ 与 KL 约束；DPL 移除 $\pi_{ref}$，
