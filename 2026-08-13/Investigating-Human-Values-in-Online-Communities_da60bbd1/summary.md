---
title: "Investigating-Human-Values-in-Online-Communities"
source: https://aclanthology.org/2025.naacl-long.77.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:04:56"
field: "计算社会科学与NLP交叉"
keywords: ["Schwartz values", "value polarity", "Reddit", "online communities", "computational social science", "DeBERTa", "multi-label classification", "stance detection"]
innovations: ["双阶段相关性+极性分类框架用于社区级价值观分析", "大规模自动标注12k子reddit九百万帖子的Schwartz十值数据集"]
benchmarks: ["ValueNet", "ValueArg", "Schwartz & Cieciuch 2022国家问卷"]
---

# 论文速读：Investigating-Human-Values-in-Online-Communities

## 一句话总结
论文提出了一种基于Schwartz价值理论的计算方法，训练了价值相关性与极性两个监督分类器，自动分析并标注了Reddit上12k个子社区、超过900万帖子的十类人类价值观及其态度极性，既验证了既有社会科学研究发现，也揭示了如Vegan社区对Conformity强烈负面态度等新洞察。

## 研究问题与动机
1. 传统社会科学研究人类价值观依赖自陈式问卷调查，样本量小、代表性有限，难以外推至更广泛人群（Gerlach & Eriksson, 2021; Bašináková et al., 2016）。
2. 社交媒体蕴含海量自然语言数据，能反映真实世界的人类思想与态度表达，但目前缺乏可大规模计算分析价值观的工具。
3. 现有NLP价值提取工作（如van der Meer et al., 2023）仅检测价值是否存在，忽视了对价值态度的极性（支持/反对），限制了可获得的洞察深度。
4. Reddit等平台按主题/子社区组织讨论，天然适合研究"社区层面"的价值观分布及其异同，而非个体层面。

## 核心贡献（创新点）
1. **双阶段"相关性+极性"分类框架**：先预测文本是否表达某一Schwartz价值（多标签），再预测对该价值的态度极性（正/负/中性），与仅检测存在性的前人工作本质不同。
2. **社区级价值观聚合方法**：将子社区内所有帖子/评论的价值概率向量取平均，得到代表整个社区的价值画像，而非个体用户标签。
3. **大规模自动标注数据集**：对11,616个热门子reddit、约900万条内容自动标注Schwartz十值相关性与极性，并公开发布，填补了无公开大规模社区价值观资源的空白。
4. **跨层次效度验证**：通过美国各州传统价值观与保守主义调查的Spearman R=0.55~0.63相关，验证了在线社区数据与传统问卷在宏观层面的对齐性，同时指出国家层面的在线vs离线差异（ρ=-0.03），为方法论定位提供了边界。

## 方法详解
**数据收集与清洗**：使用Pushshift API下载2022年1~8月的Reddit内容，过滤字数<10或点赞<10的噪音；移除NSFW、订阅者<5000、内容<250条的子reddit；大子社区随机下采样至1000条，最终得到11,616个子reddit。

**价值相关性分类器**：采用DeBERTa（优于RoBERTa的性能与速度），在多标签设置下为十个Schwartz值各自输出独立概率。训练数据为ValueNet（Qiu et al., 2022）与ValueArg（Kiesel et al., 2022）拼接集；输入构造为`[CLS] v [SEP] s [SEP]`，其中v为价值名，s为文本；宏平均F1为0.76。

**价值极性分类器**：仅在ValueNet非中性标签上训练，预测文本对某价值的正向/负向态度（原-1/1合并为单一极性）；F1为0.72，Cohen's κ为0.47（中等一致性，反映主观任务难度）。

**推理流程**：对每条内容c，先用相关性模型得到向量`u_rel ∈ [0,1]^10`；对概率>0.5的价值k，再用极性模型得到`u_stance^k ∈ [-1,1]`，否则为Null。

**社区聚合**：对子社区S，按公式计算均值向量：
- `u_rel(S) = (1/|S|) Σ u_rel(c_i)`
- `u_stance(S)^k = (1/|S^k|) Σ u_stance^k(c_i)`，其中S^k为极性非Null的内容子集。

**相似度分析**：定义价值观相似度σ_val为拼接后向量的余弦相似度；语义相似度σ_sem用sentence transformer嵌入子reddit描述计算；用户相似度σ_usr为用户重叠系数。相似子reddit的期望σ_val=0.81远高于随机0.64（z=73.2/74.4，p<0.001）。

## 实验与结果
**数据集规模**：11,616个子reddit，平均每个765条内容、62词/条；总量8,888,535条帖子、558,327,230词（Table 3）。

**模型评估**：
- 相关性模型：宏观F1=0.76；Spearman ρ=0.51；NDCG@1=0.87（高置信预测高度准确）。
- 极性模型：F1=0.72；Cohen's κ=0.47（中等一致）。

**主要发现**：
- 宗教类子reddit对Tradition高度相关但态度偏负；r/atheism、r/religion均有此模式。
- r/Vegan对Conformity态度极负（挑战现状）；r/carnivore略正，符合Holler et al. (2021)的系统综述结论。
- r/AbolishTheMonarchy对Conformity极负、对Benevolence偏负；r/monarchism对Tradition高相关但态度负、对Power/Achievement/Security偏正。
- 美国保守州（红色州）在Tradition值上显著高于自由州（蓝色州），与Pew调查Spearman R=0.55（保守意识形态）、R=0.63（宗教程度），p<0.0001。
- 国家层面Reddit数据与Schwartz & Cieciuch (2022)问卷数据的Spearman ρ=-0.03，无显著相关，说明线上社区不宜直接作为线下文化价值观代理。

**最强结果**：NDCG@1=0.87表明高置信相关性预测几乎全部命中目标价值；R=0.63（传统值vs宗教调查）展现了在线数据与传统方法的最好对齐。

## 相关工作脉络
1. **Ponizovskiy et al. (2020)** 发布Schwartz价值词典，基于词汇匹配，无法建模语境与极性；本文改为端到端神经网络并引入极性分类。
2. **van der Meer et al. (2023)** 训练价值存在性分类器分析个体用户分歧，但忽略极性；本文在其基础上增加极性阶段，并将分析单位从个体转向社区。
3. **Weld et al. (2023)** 构建子reddit社区价值分类法但依赖小规模自陈问卷，存在选择偏差；本文用大规模数字痕迹数据克服该限制。
4. **Trager et al. (2022)** 发布Moral Foundations Reddit语料库（16k评论），聚焦道德基础而非Schwartz十值；两者互补但价值框架不同。
5. **Roy et al. (2021)** 分析政治推文中的道德框架；本文扩展至更广泛的Schwartz框架及社区级聚合。
6. **Havaldar et al. (2024)** 用词汇方法在地理定位Twitter上研究价值，发现与问卷相关性低；本文同样在国家层面未发现相关，支持"线上≠线下"的边界判断。

## 局限性与未来方向
1. **任务固有主观性**：价值观标注存在不可避免的aleatoric uncertainty，模型预测偶有噪声与不一致。
2. **聚合信息损失**：将子reddit内所有内容压缩为单一向量，丢失了个体差异与社区内部动态。
3. **解释力不足**：模型只能识别"是什么"，无法解释"为什么"，需要与心理学/社会学专家协作开展深度定性研究。
4. **国家层面效度有限**：Reddit数据与国家问卷数据无显著相关，表明该工具更适合研究"在线社区"而非"国家文化"。
5. **未来方向**：个体层级分析、社区内部动态建模、与其他价值框架（如道德基础理论）的结合、跨语言扩展等。

## 研究启发与可借鉴点
1. **"相关性+极性"双阶段架构**可迁移至其他抽象概念提取任务（如意识形态、道德观、文化维度），先判断概念存在性再判断态度。
2. **社区级向量聚合方法**简洁有效，为"群体画像"提供了可复用的计算范式。
3. **用NDCG@1和Spearman ρ共同评估排序质量**的做法比单纯看F1更能反映实际应用效果，值得借鉴。
4. **将在线数据与传统调查做交叉验证**（如美国各州案例）的方法论，为计算社会科学的效度评估提供了模板。
5. **明确界定工具的互补定位**（而非替代传统方法）并如实报告边界（如国家层面无效），体现了负责任的AI研究姿态。

## 关键术语表
**Schwartz Values**：由Shalom Schwartz提出的十类基本人类价值观框架，包括Power、Achievement、Hedonism、Stimulation、Self-direction、Universalism、Benevolence、Tradition、Conformity、Security。
**Value Relevance Classifier**：预测文本是否与某一Schwartz价值相关的多标签分类器，输出每个价值的独立概率。
**Value Stance Classifier**：在确认价值相关后，预测文本对该价值的态度极性（正向/负向/中性）的分类器。
**DeBERTa**：Microsoft提出的预训练语言模型，采用解耦注意力与增强掩码解码器，在本题中优于RoBERTa。
**ValueNet**：Qiu et al. (2022)发布的以价值观驱动对话的数据集，含十类Schwartz价值标注。
**ValueArg**：Kiesel et al. (2022)发布的论证价值数据集，用于训练相关性分类器。
**Pushshift API**：提供Reddit历史数据下载的API接口，是本研究数据获取的主要工具。
**NDCG@1**：Normalized Discounted Cumulative Gain at rank 1，衡量模型最高置信预测的排序准确性。

## 可复现要素
- **数据集**：Reddit 2022年1~8月数据，经Pushshift API获取；11,616个子reddit、约900万条内容的标注结果已公开（论文 footnote 1 提及）。
- **代码/权重**：论文未明确提供开源链接，但提到在单个Titan RTX GPU上训练约5小时（相关性）和2小时（极性）。
- **关键超参**：学习率5e-5，batch size 32，AdamW优化器，线性学习率调度器，10个epoch+early stopping。
- **训练数据**：ValueNet + ValueArg拼接集；极性模型仅在ValueNet非中性标签上训练。
- **评估集**：研究者自行标注的10,000帖子+10,000评论（相关性）和200条样本（极性）。
