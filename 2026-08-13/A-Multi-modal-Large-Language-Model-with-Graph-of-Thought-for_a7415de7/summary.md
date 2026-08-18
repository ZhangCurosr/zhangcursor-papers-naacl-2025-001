---
title: "A-Multi-modal-Large-Language-Model-with-Graph-of-Thought-for"
source: https://aclanthology.org/2025.naacl-long.76.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:57:27"
field: "多模态推荐系统"
keywords: ["Graph-of-Thought", "Multi-modal Recommendation", "LLM for RecSys", "Text-graph Alignment", "Instruction Tuning", "LightGCN"]
innovations: ["首次将 Graph-of-Thought 提示技术应用于多模态推荐，解决 CoT 在图结构数据上图挖掘不足的问题", "提出 text-graph alignment + graph instruction tuning 两阶段方法，使 MLLM 能理解用户-物品交互图结构", "预训练推荐器辅助的自适应图截断策略，在 LLM token 限制下最大化高潜力候选信息融入"]
benchmarks: ["Amazon Clothing", "Amazon Baby", "HM Fashion", "Amazon Pantry", "Amazon Electronics", "Amazon Sports"]
---

# 论文速读：A Multi-modal Large Language Model with Graph-of-Thought for Effective Recommendation

## 一句话总结
本文提出 **GollaRec**，将 Graph-of-Thought (GoT) 提示技术首次应用于多模态推荐任务，通过将用户-物品交互图结构化为包含视觉与文本"思考"的提示，解决了传统 CoT 在图结构数据上"图挖掘不足"的问题，在 6 个数据集上均优于 12 个 SOTA 基线。

## 研究问题与动机
1. **CoT 线性局限**：Chain-of-Thought 在文本线性推理上有效，但无法原生处理推荐系统中核心的用户-物品交互图结构，导致"图挖掘不足（insufficient graph mining）"。
2. **MLLM 难以直接用于推荐**：现有 MLLM（如 CLIP、LLaVA）虽具备多模态理解能力，但缺乏将用户行为序列与图结构相结合的有效提示机制，难以在推荐场景中利用丰富的交互信号。
3. **Token 长度约束**：MLLM 输入 token 上限（如 LLaVA 为 2048）限制了可融入的用户-物品交互信息量，576 token 已分配给视觉数据，剩余空间不足以编码完整图结构。
4. **多模态语义不一致**：项目图文内容需要统一语义空间对齐，否则不同模态嵌入无法有效协同提升推荐效果。

## 核心贡献（创新点）
1. **首次将 GoT 引入多模态推荐**：提出 Graph-of-Thought 提示技术，将用户-物品交互图以图结构化的方式嵌入 MLLM 提示，区别于仅用线性 CoT 或纯文本的现有工作。
2. **Text-graph alignment + Graph instruction tuning**：通过对比学习对齐图节点嵌入与文本嵌入，并基于 GraphGPT 思想进行指令微调，使 MLLM 能理解图结构模式，解决 C3 挑战。
3. **Text-image alignment（ITC loss）**：利用图像-文本对比预训练统一多模态语义空间，使图文嵌入在 GoT 中形成一致表征，解决 C2 挑战。
4. **Adaptive graph truncation**：用预训练推荐器（LightGCN）生成初始候选列表，按需截断以在 token 限制内最大化高潜力候选信息的融入，解决 C1 挑战。
5. **Graph adaptor（LightGCN）融合交互信息**：将 MLLM 输出的嵌入通过 LightGCN 传播用户-物品交互结构，补充个性化排序信号。

## 方法详解

### 整体架构
GollaRec 由三部分组成：(1) MLLM（LLaVA Vicuna-7B）+ GoT 提示；(2) Text-graph / Text-image alignment 预训练模块；(3) LightGCN graph adaptor。

### GoT 提示构建（Algorithm 1）
- 先用预训练 LightGCN 生成初始候选排名；
- 保留视觉 token（576），剩余空间用于追加候选 item 描述；
- 自适应截断直至达到 2048 token 上限。

### Text-image Alignment（ITC Loss）
$$\mathcal{L}_{\mathrm{ITC}} = -\frac{1}{B}\sum_{p=1}^{N}\log\frac{\exp(\sin(v_p, t_p)/\tau)}{\sum_{q\neq p}\exp(\sin(v_p, t_q)/\tau)}$$
其中 $v_p, t_p$ 为第 $p$ 个 item 的视觉和文本嵌入，$\sin$ 为相似度函数，$\tau$ 为温度参数，目标是将图文对在统一空间中对齐。

### Text-graph Alignment（两步法）
1. **Grounding**：用 BERT 编码 item 描述得文本嵌入 $z_2$，用 Graph Transformer 编码图得节点嵌入 $z_1$，用对比损失对齐二者。
2. **Instruction Tuning**：将图节点嵌入经 MLP 投影为图 token $\hat{z}_1 = \text{MLP}(z_1)$，构造指令序列 `<graph_start>, <graph_token>_1, ..., <graph_token>_l, <graph_end>`，与对应文本 token 进行匹配任务，用交叉熵优化：
$$\psi(x_o|\hat{z}_1, z_3) = \prod_{j=1}^{l}\psi_{\theta_2}(x_j|\hat{z}_1, z_3)$$

### Graph Adaptor
使用 LightGCN 对 MLLM 输出的 item embedding 进行传播：
$$h_u = \sum_{i \in \mathcal{N}_u}\frac{h_i}{\sqrt{|\mathcal{N}_i||\mathcal{N}_u|}}$$
得到最终用户和 item 嵌入用于 top-k 排序。

## 实验与结果
- **数据集**：3 个通用推荐数据集（HM、Clothing、Baby）+ 3 个多域目标数据集（Pantry、Electronics、Sports），共 6 个；源域为 Food、Home、Clothing、Office。
- **基线**：12 个模型，涵盖 General（LightGCN）、Multi-modal（VBPR、MMGCL、BM3）、MLLM（CLIP、BEiT-3、LLaVA）、Language-based（P5、LMRecSys、TALLRec）、Multi-domain（MOME、PLE、MGFN）。
- **指标**：Recall@20、NDCG@20，8:1:1 划分。

**主要结果**：
| 任务 | 数据集 | GollaRec Recall@20 | 最佳基线 BM3 Recall@20 | 提升幅度 |
|------|--------|-------------------|----------------------|---------|
| 通用推荐 | Clothing | **0.0932** | 0.0797 | +16.9% |
| 通用推荐 | Baby | **0.0958** | 0.0863 | +11.0% |
| 通用推荐 | HM | **0.1880** | 0.1711 | +9.9% |
| 多域推荐 | Sports | **0.1112** | 0.0970 | +14.6% |
| 多域推荐 | Pantry | **0.1213** | 0.0932 | +30.2% |
| 多域推荐 | Electronics | **0.0681** | 0.0638 | +6.7% |

- **平均提升**：相比最强基线 BM3 平均提升 **12.7%**，最高达 30.2%（Pantry）。
- **消融实验**（Table 3）：移除 GoT、Adaptor、Text-image/Text-graph alignment 均导致显著下降，验证各组件有效性。
- **冷启动分析**：GollaRec 在冷启动用户上相对 BM3 提升 11.4%~32.1%，优于普通用户，说明 GoT 能有效利用 MLLM 世界知识弥补稀疏交互。
- **t-SNE 可视化**：GollaRec 图文嵌入平均 MSE 显著低于 BM3（Clothing: 1.66 vs 12.23；Sports: 0.12 vs 5.87），验证多模态语义一致性。
- **LLaVA 变体实验**：Vicuna-7B 与 Vicuna-13B 性能相当，表明当前推荐数据集规模下更大模型未必带来增益。

## 相关工作脉络
1. **LLM for Recommendation**：P5、LMRecSys、TALLRec 将推荐任务重构为文本生成，但未有效利用图结构和多模态内容；GollaRec 在此基础上引入 GoT 图文对齐与图适配器。
2. **Chain-of-Thought Prompting**：CoT 在数学推理等领域成功，但 Wang et al. (2024) 指出其在图相关任务上表现不佳；GollaRec 的 GoT 是对此的直接改进。
3. **GraphGPT / StructGPT / GPT4Graph**：通用图结构化 LLM 方法，但未针对推荐场景设计多模态对齐与交互图适配；GollaRec 首次将 GoT 专门化到多模态推荐。
4. **Multi-modal Recommendation**：BM3、MMGCL、VBPR 等方法融合多模态特征但缺乏 LLM 推理能力；GollaRec 通过 GoT 将 MLLM 的推理能力引入推荐。
5. **LLM 在推荐中的 Token 约束问题**：Ren et al. (2024) 指出 LLM 输入长度限制影响推荐效果；GollaRec 用预训练推荐器自适应截断解决此问题。

## 局限性与未来方向
1. **Prompt demonstration 手动设计**：当前 GoT 中的示例步骤需人工编写，虽对比自动生成的 LLaMA3 prompt 无明显性能差异，但仍需探索自动化的 prompt 生成策略。
2. **初始候选列表依赖 LightGCN**：使用 LightGCN 生成初步排序，未来可探索更先进的排序模型或隐空间相似度度量来优化候选选择。
3. **未探索对话式推荐**：目前仅验证通用和多域推荐，未测试在 multimodal conversational recommendation 中的效果，可作为未来方向。
4. **大规模数据集下的 scaling 问题**：当前实验数据集规模有限，更大模型（13B）未带来明显增益，未来需在更大推荐数据集上验证 scaling law 是否成立。

## 研究启发与可借鉴点
1. **GoT 提示范式可迁移**：图结构信息融入 LLM 提示的新思路，可推广至社交推荐、知识图谱问答等图密集型任务。
2. **Text-graph alignment + Instruction tuning 的组合策略**：两步法（grounding → instruction tuning）让 MLLM 理解图结构，技术路线清晰且可复用到其他图+LLM 任务。
3. **预训练推荐器辅助提示构建**：用轻量级推荐模型（LightGCN）做候选筛选以突破 token 限制，是一种实用且高效的工程策略。
4. **冷启动优势验证**：论文展示了 GoT 在冷启动场景下的显著增益（最高 32%），提示我们关注多模态 MLLM 的世界知识对稀疏交互用户的有效补充作用。
5. **多模态一致性评估**：用 t-SNE + MSE 量化图文嵌入对齐程度，作为多模态推荐模型的有效评估维度，值得在其他工作中复用。

## 关键术语表
**Graph-of-Thought (GoT)**：将图结构信息以步骤化提示形式融入 MLLM 的推理链，区别于线性 CoT，适用于用户-物品交互图等结构化数据。

**Text-graph Alignment**：通过对比学习将图节点嵌入与文本嵌入对齐到统一语义空间，使 MLLM 能理解图结构信息。

**Graph Instruction Tuning**：将图节点投影为图 token 后，构造匹配任务对 MLLM 进行指令微调，使其学会理解图结构模式。

**Adaptive Graph Truncation**：基于预训练推荐器生成候选列表，在固定 token 预算下自适应截断以保留最多高潜力 item 描述的输入策略。

**ITC Loss（Image-Text Contrastive Loss）**：最大化图文正对相似度、最小化负对相似度，实现多模态统一嵌入空间的对齐损失。

**Graph Adaptor（LightGCN）**：在 MLLM 输出嵌入之上叠加图神经网络传播，整合用户-物品交互结构以生成最终排序嵌入。

## 可复现要素
- **数据集**：Amazon Review（Clothing、Baby、Food、Home、Office、Pantry、Electronics、Sports）+ HM Fashion（公开数据集）
- **代码/权重**：开源，GitHub: https://github.com/zxy-ml84/GollaRec
- **关键超参**：LLaVA Vicuna-7B 为基础模型；最大输入长度 2048；视觉 token 占 576；batch size 8；Learning rate 1e-4~3e-3（依数据集）；Training steps 50k~60k；Warmup ratio 0.01~0.03；Weight decay 0~0.01；温度参数 τ 依 ITC loss 设定
