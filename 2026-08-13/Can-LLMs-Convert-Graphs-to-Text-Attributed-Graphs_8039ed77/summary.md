---
title: "Can-LLMs-Convert-Graphs-to-Text-Attributed-Graphs"
source: https://aclanthology.org/2025.naacl-long.65.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:01:15"
field: "图表示学习与大模型交叉"
keywords: ["text-attributed graphs", "graph neural networks", "large language models", "cross-graph learning", "feature alignment", "topology-aware generation"]
innovations: ["首次利用LLM将任意图自动转换为文本属性图(TAGs)，无需原始文本", "将五维拓扑属性注入LLM提示以生成节点角色感知描述", "统一处理文本丰富/有限/自由三类图场景的跨图特征对齐框架"]
benchmarks: ["Cora", "Pubmed", "USA Airport Network", "Europe Airport Network", "Brazil Airport Network", "ogbn-products"]
---

# 论文速读：Can-LLMs-Convert-Graphs-to-Text-Attributed-Graphs

## 一句话总结
本文提出了TANS（Topology-Aware Node description Synthesis），首次利用大语言模型将现有图数据自动转换为文本属性图（TAGs），通过注入拓扑信息引导LLM生成节点文本描述，解决了跨图学习中不同特征空间对齐的核心难题。

## 研究问题与动机
- **GNN跨图学习的根本障碍**：消息传递GNN的输入维度固定，无法直接处理不同特征维度的多个图联合训练，严重制约了迁移学习、域适应和图基础模型的发展。
- **现有对齐方法的局限**：SVD仅能对齐特征维度而无法保证语义一致性，且不适用于无节点特征的图；TAGs方法虽能通过文本编码器对齐语义，但高度依赖高质量节点文本描述，现实中难以获取。
- **LLM增强方法的文本依赖性**：TAPE和KEA等利用LLM增强节点描述的方法，均要求节点已有原始文本，无法处理纯拓扑的文本自由图（text-free graphs）。
- **拓扑信息未充分利用**：现有方法在处理无文本图时仅依赖单一拓扑指标（如度、特征向量），忽略了多源拓扑信息的综合表达能力。

## 核心贡献（创新点）
- **首次提出利用LLM将任意图自动转换为TAGs**，区别于SVD/TAPE/KEA等方法必须依赖已有文本或节点特征的前提，实现了从无文本到文本属性的跨越。
- **引入五维拓扑属性作为LLM提示的关键辅助知识**，使LLM能够理解节点在图中的角色与地位，而非简单拼接手工特征。
- **统一框架覆盖三类图场景**（文本丰富、文本有限、文本自由），而TAPE/KEA仅适用于前两类，这是方法适用范围的本质扩展。
- **生成的节点文本描述在特征空间中自然对齐**，无论原始图是citation network还是airport network，均可通过同一文本编码器和GNN进行跨图联合学习。

## 方法详解
TANS由四个步骤构成：

**Step 1 — 计算拓扑属性**：选取五个节点级拓扑性质——度中心性（Degree Centrality）、介数中心性（Betweenness Centrality）、接近中心性（Closeness Centrality）、聚类系数（Clustering Coefficient）和方聚类系数（Square Clustering Coefficient），用以刻画节点在局部和全局层面的重要性。

**Step 2 — 生成基础节点描述**：将以下四个组件拼接为Prompt输入LLM：（1）图类型/节点类型/边类型的前缀信息；（2）可选的原始节点文本；（3）可选的k=5个邻居节点的文本描述；（4）各拓扑属性的具体数值及其在全图中的排名百分比。这一设计使LLM获得"节点角色"的全景认知。

**Step 3 — LLM推理**：使用GPT-4o-mini，Prompt后缀要求模型输出Top-k（k=3或1）潜在节点类别及推理依据（不超过200词）。输出被设计为通用、可解释的描述，确保不同图的嵌入在特征空间中接近。

**Step 4 — 生成最终节点描述**：将LLM输出作为新节点文本描述，追加到原有文本（若有）之后，统一使用文本编码器（如MiniLM）生成节点嵌入，再送入GNN进行分类。

## 实验与结果
**数据集**：Cora（2,708节点，7类，text-rich/-limit）、Pubmed（19,717节点，3类）、USA/Europe/Brazil机场网络（无文本，按活跃度分4类）。

**单图学习（text-rich，Cora低标签）**：TANS+GCN达 **81.26±1.48%**，对比Raw Text（79.19%）、+TAPE（79.64%）、+KEA（80.08%），平均提升约 **1.6-2.1个百分点**；MLP backbone上提升最显著（72.47% vs 66.18%）。

**单图学习（text-free，GCN）**：TANS在Brazil上达 **80.61±12.14%**，远超Node Degree（71.82%）、Eigenvector（62.42%）、Random Walk（69.70%）；USA上达61.08%，Europe上达56.87%。

**域适应（text-free airport graphs）**：TANS平均 **58.40%**，第二名Node Degree为54.21%，**提升约4.2个百分点**，在6组实验中5组取得最优。

**迁移学习（Cora↔Pubmed）**：TANS在Cora→Pubmed上达76.14%，Pubmed→Cora上达80.05%，均超过所有基线。

## 相关工作脉络
- **SVD对齐方法**（Yu et al., 2024; Zhao et al., 2024）：仅对齐特征维度，无法处理无特征图，且语义一致性差；TANS通过文本自然编码实现语义级对齐。
- **TAGs方法**（Yan et al., 2023; Wang et al., 2024c）：依赖已有高质量文本，TANS首次实现从无文本到有文本描述的自动生成。
- **TAPE**（He et al., 2024）：利用LLM推理节点类别并增强文本，但要求节点已有原始描述；TANS无需任何原始文本即可工作。
- **KEA**（Chen et al., 2024a）：通过LLM解释关键术语增强嵌入，同样依赖已有文本；TANS从拓扑出发生成全新描述。
- **OFA**（Liu et al., 2024a）：基于模板的图基础模型，不进行额外文本生成；TANS可与其结合使用，实验证明TANS+OFA仍优于单纯模板方法。
- **手工拓扑特征方法**（Ribeiro et al., 2017; Dwivedi et al., 2022, 2023）：将度/特征向量等作为节点特征，仅支持单图训练；TANS生成的文本使跨图联合学习成为可能。

## 局限性与未来方向
- **大规模图不适用**：当前实验未包含超过10万节点的图，对每个节点查询GPT的成本和耗时限制了可扩展性，未来需探索更高效的模板设计。
- **LLM能力限制**：仅使用GPT-4o-mini，未比较更大规模模型（如GPT-4o）的效果差异。
- **Prompt设计灵活性不足**：不同图类型（社交网络vs引文网络）可能偏好不同的拓扑属性组合，当前固定五项属性可能非最优。
- **未扩展至边/图级任务**：仅验证了节点分类任务，边缘级和图级任务的适用性有待探索。
- **潜在伦理风险**：LLM生成内容可能引入偏见，需更审慎的提示设计。

## 研究启发与可借鉴点
- **拓扑属性注入LLM提示的设计范式**：将结构性先验知识以排名百分比形式融入Prompt，是一种可迁移至其他图任务的"结构感知提示"策略，可用于边预测、子图匹配等任务。
- **邻居文本辅助生成**：在text-limit场景下，1-hop邻居文本的加入使Cora性能从72.92%提升至76.21%（+3.3%），验证了局部上下文在LLM图推理中的价值，值得在其他生成型图任务中验证。
- **统一文本描述促进跨图对齐**：LLM生成的节点描述天然处于同一语义空间，为构建"一个GNN处理所有图"的基础模型提供了新的数据预处理路径。
- **文本编码器鲁棒性验证**：在四种不同文本编码器（MiniLM/ALBERT/RoBERTa/MPNet）下TANS均最优，说明生成文本质量稳定，可推广至其他下游编码器。
- **将结构化属性图（如Cora的1433维one-hot）转换为TAGs**的潜在方向：利用属性维度本身的语义（如关键词）生成文本描述，扩展了TANS的适用范围。

## 关键术语表
- **Text-Attributed Graph (TAG)**：每个节点附有文本描述的图结构，可通过文本编码器映射到统一特征空间以实现跨图学习。
- **Topology-Aware Node description Synthesis (TANS)**：本文提出的方法，利用拓扑属性引导LLM自动生成节点文本描述。
- **Message Passing GNN**：通过聚合邻居信息更新节点嵌入的图神经网络框架，其固定输入维度是跨图学习的主要障碍。
- **Domain Adaptation（域适应）**：在源图训练、目标图测试的跨图设置下评估模型泛化能力。
- **Pretrain & Finetune（预训练+微调）**：先在源图预训练模型参数，再在目标图上微调的跨图学习设置。
- **Degree/Betweenness/Closeness Centrality**：分别衡量节点的直接连接数、最短路径中介能力和全局可达性的图论中心性指标。
- **Clustering Coefficient**：衡量节点邻居之间相互连接紧密程度的局部聚集性指标。

## 可复现要素
- **数据集**：Cora、Pubmed（公开）、USA/Europe/Brazil机场网络（公开，Ribeiro et al., 2020）；ogbn-products子图实验也使用公开数据。
- **代码与数据**：已开源，地址 https://github.com/Zehong-Wang/TANS
- **关键超参**：文本编码器使用MiniLM；GNN backbone为GCN/GAT/MLP；隐藏维度{8, 16, 32, 64, 128, 256}；层数{1, 2, 3}；学习率{5e-2, 1e-2, 5e-3, 1e-3}；dropout{0.0, 0.1, 0.5, 0.8}；每个实验运行30次随机种子取平均。
- **LLM接口**：GPT-4o-mini（public API）。
- **邻居节点数**：k=5。
- **每节点生成文本长度限制**：≤200词。
- **低标签设置**：20节点/类训练，30节点/类验证，剩余测试；Brazil小图：10节点/类训练，20验证。
