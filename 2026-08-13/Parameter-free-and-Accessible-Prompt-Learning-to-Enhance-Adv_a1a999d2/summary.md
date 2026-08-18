---
title: "Parameter-free-and-Accessible-Prompt-Learning-to-Enhance-Adv"
source: https://aclanthology.org/2025.naacl-long.33.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:32:13"
field: "多模态大模型安全与鲁棒性"
keywords: ["对抗鲁棒性", "Prompt Learning", "Vision-Language Model", "CLIP", "参数高效方法", "对抗防御"]
innovations: ["提出词级别离散 defense token 搜索框架 PFPT，无需额外参数即可增强 VLM 对抗鲁棒性", "设计两阶段粗到细搜索策略（STS + PTS），兼顾候选覆盖与组合联合优化", "引入一致性分数约束，确保附加 defense tokens 不显著偏离原始 prompt 语义分布"]
benchmarks: ["ImageNet", "ImageNet-V2", "ImageNet-Sketch", "ImageNet-R", "Caltech101", "OxfordPets", "Food101"]
---

# 论文速读：Parameter-free-and-Accessible-Prompt-Learning-to-Enhance-Adversarial-Robustness-for-Pre-trained-Vision-Language-Models

## 一句话总结
本文提出 **Parameter-Free Prompt Tuning (PFPT)**，通过在 CLIP 预训练词汇表中搜索离散的 defense tokens 并直接附加到文本 prompt 上，无需训练任何模型参数即可显著提升视觉-语言模型（VLM）的对抗鲁棒性。

## 研究问题与动机
- VLM（如 CLIP）易受对抗扰动误导，安全漏洞日益凸显。
- 现有对抗训练方法需对整个模型进行微调，计算成本高昂且可能损害零样本泛化能力。
- 参数高效方法（如 APT）优化连续文本嵌入，需访问模型 embedding 层、操作门槛高，且嵌入结果难以映射到真实自然语言词汇。
- 现有 prompt 学习方法几乎全部聚焦连续嵌入空间优化，缺乏**词级别（word-level）**的离散 prompt 搜索研究。

## 核心贡献（创新点）
- **提出 PFPT 框架**：从预训练固定词汇表中搜索 defense tokens，无需存储额外参数，所有模型权重保持冻结。
- **词级别搜索，直接可用人工可读**：学到的 tokens 天然存在于原生词汇中，可直接拼接到文本 prompt 使用，无需访问中间嵌入层。
- **两阶段粗到细搜索策略**：单 Token 搜索（STS）利用梯度筛选候选集，并行 Token 搜索（PTS）联合优化一致性分数与对抗准确率，避免陷入纯梯度下降或暴力枚举。
- **实验覆盖广泛**：在 11 个数据集、4 种 shot 配置（1/4/16/100-shot）、2 种扰动预算下验证，显著超越 HEP 和 APT。

## 方法详解
### 整体框架
CLIP 的图像编码器和文本编码器均冻结，defense tokens 来源于预训练词汇表 $\mathcal{V}$，以 suffix 形式附加到默认 prompt（如 `"a photo of a [CLASS]"`）末尾，形成新 prompt $T_j'$，其中 K 个 defense tokens 需要搜索。

### 对抗样本生成（Algorithm 1）
在训练过程中在线生成对抗样本，使用 PGD 攻击：
$$\arg\max_{\|\delta_i\|_p \leq \varepsilon} \mathcal{L}(\mathcal{E}_v(I_i + \delta_i), \mathcal{E}_t(T_i'), y_i)$$
使用 3 步 PGD，步长 $2\varepsilon/3$；评测时使用 100 步 PGD，步长 $\varepsilon/4$。

### 单 Token 搜索（STS，Algorithm 2）
对每个 defense token 位置 k，计算梯度导向的 defense 目标：
$$\mathcal{L}^{defense}(t_k) = -\nabla_{\mathbf{e}_{t_k}} \log p(\hat{y}=y | \mathbf{t}^{(template)} \oplus t_1 \oplus \cdots \oplus t_k \oplus \cdots \oplus t_K, I + \delta)$$
选取梯度目标最高的 top-L 个 token 作为该位置的候选集 $\mathcal{C}_{k,l}$。

### 并行 Token 搜索（PTS，Algorithm 3）
将各位置同排名 token 拼接为候选组合 $\mathcal{A}_l = \mathcal{C}_{1,l} \oplus \cdots \oplus \mathcal{C}_{K,l}$。

定义两个评分：
- **一致性分数**：用 KL 散度衡量附加 defense tokens 前后预测分布的差异
$$\mathcal{L}^{consist}(\mathbf{t}') = D_{KL}(q^1(\mathbf{t}'^{(defense)}=\mathbf{t}') \| q^2)$$
- **对抗准确率分数**：标记函数指示对抗样本分类是否正确

总评分：$\mathcal{L}^{total} = \mathcal{L}^{consist} + \lambda(1 - \mathcal{L}^{acc.})$，选取得分最低的组合。

## 实验与结果
**数据集**：ImageNet 及 10 个下游数据集（Caltech101, OxfordPets, StanfordCars, Flowers102, Food101, FGVC Aircraft, SUN397, DTD, EuroSAT, UCF101），4 种 shot 配置（1/4/16/100）。

**模型**：ViT-B/32，使用 TeCoA 预训练权重。

**核心结果（Table 1）**：

| 方法 | $\varepsilon=1/255$, 1-shot Acc. | Rob. | $\varepsilon=4/255$, 16-shot Acc. | Rob. |
|---|---|---|---|---|
| HEP | 44.92 | 31.93 | — | — |
| APT | 46.51 | 33.09 | 45.73 | 33.23 |
| **PFPT** | **49.41** | **37.51** | **50.19** | **38.01** |

- 平均较 HEP 提升 **+4.9%（准确率）** 和 **+5.8%（鲁棒性）**（$\varepsilon=1/255$）。
- 在 1-shot 和 4-shot 低数据场景下优势尤为显著，APT 在 4-shot 下甚至低于 HEP 基线。
- **泛化性（Table 2）**：在 ImageNet-V2、ImageNet-Sketch、ImageNet-R 及 10 个跨数据集测试中，PFPT 均取得最高准确率与鲁棒性，APT 跨数据集泛化明显退化。

**最强结果**：ImageNet 16-shot，$\varepsilon=1/255$，准确率 50.36%，鲁棒性 38.32%。

## 相关工作脉络
- **APT（Li et al., 2024b）**：优化连续文本嵌入以提升对抗鲁棒性，嵌入需映射回词表且依赖 embedding 层访问；PFPT 直接在词表上搜索离散 token，无需修改模型结构。
- **CoOp（Zhou et al., 2022）**：自动化 prompt 生成的开创性工作，优化连续 prompt embedding；PFPT 聚焦词级搜索，保证人类可读性与部署便捷性。
- **DefensePrefix（Azuma & Matsui, 2023）**：针对 CLIP 的 typo 攻击防御，使用人工设计的 prefix；PFPT 是数据驱动的自动搜索方法。
- **Subspace Prompt Tuning（Ma et al., 2023）**：缓解 VLM prompt 过拟合；PFPT 关注对抗鲁棒性而非语义保持。
- **Visual Prompting for Adversarial Robustness（Chen et al., 2023）**：在图像空间添加对抗视觉 prompt；PFPT 操作对象为文本 prompt，避免图像信息损失。
- **HEP（Radford et al., 2021）**：CLIP 原始手工设计 prompt 基线；PFPT 通过搜索自动学习替代方案。

## 局限性与未来方向
- 学到的 defense tokens 之间缺乏语义连贯性，以词为单位拼接，非连续自然语言句子，可尝试引入语言模型约束生成连贯表述。
- Defense tokens 为数据集特定（dataset-specific），尚未探索跨数据集的通用 defense token 搜索。
- 仅针对 CLIP 验证，未扩展到更大规模 VLM（如 BLIP-2、LLaVA）。

## 研究启发与可借鉴点
- **词级别离散搜索范式**：将 prompt 优化从连续嵌入空间迁移到离散词空间，是提升方法可用性和可解释性的有效思路，可迁移至其他 VLM 任务。
- **两阶段搜索设计**：先梯度筛选候选集再联合优化，兼顾搜索效率与组合效果，这一粗到细策略可推广至其他离散优化问题。
- **一致性约束（KL 散度）**：用原始 prompt 预测分布引导 defense token 选择，避免语义偏移，在对抗鲁棒性研究中具有通用参考价值。
- **低 shot 场景优势明显**：PFPT 在 1-shot/4-shot 下大幅领先，启示可将其与少样本学习场景结合，探索更多低资源任务。
- **评估协议完整**：同时报告 in-distribution 性能、分布偏移（ImageNet-V2/Sketch/R）和跨数据集泛化，可作为后续工作的标准评估范式参考。

## 关键术语表
**PFPT**：Parameter-Free Prompt Tuning，本文提出的无参数 prompt 搜索方法，从预训练词汇表中学习对抗防御 token。
**VLM**：Vision-Language Model，视觉-语言预训练模型（如 CLIP），共享图像和文本的嵌入空间。
**PGD**：Projected Gradient Descent，基于梯度的对抗攻击方法，通过多步迭代扰动图像。
**HEP**：Hand-Engineered Prompts，CLIP 原始手工设计的文本 prompt（如 "a photo of a [CLASS]"）。
**APT**：Adversarial Prompt Tuning（Li et al., 2024b），在连续 embedding 空间优化文本 prompt 以提升对抗鲁棒性的参数高效方法。
**STS**：Single Token Search，PFPT 第一阶段，逐位置利用梯度筛选 top-L 候选 token。
**PTS**：Parallel Tokens Search，PFPT 第二阶段，对所有候选组合按一致性+准确率联合评分并选出最优组合。
**CoOp**：Learning to Prompt for Vision-Language Models（Zhou et al., 2022），首个自动 prompt 学习方法，优化连续 prompt embedding。

## 可复现要素
- **数据集**：11 个公开数据集（ImageNet 等），均已公开。
- **代码/权重**：论文未明确说明代码开源情况；使用 TeCoA 预训练的 ViT-B/32 权重。
- **关键超参**：扰动预算 $\varepsilon = 1/255$ 或 $4/255$；训练 PGD 3 步、步长 $2\varepsilon/3$；评测 PGD 100 步、步长 $\varepsilon/4$；defense token 数量 K=16；候选集大小 L=4（参考 Table 1 脚注及正文描述）。
- **搜索位置**：实验中将 defense tokens 置于 prompt 后缀位置（suffix）。
