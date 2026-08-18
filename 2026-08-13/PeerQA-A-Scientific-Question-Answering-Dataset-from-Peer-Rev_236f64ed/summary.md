---
title: "PeerQA-A-Scientific-Question-Answering-Dataset-from-Peer-Rev"
source: https://aclanthology.org/2025.naacl-long.22.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:32:35"
field: "科学文献问答"
keywords: ["scientific QA", "peer review", "evidence retrieval", "long-context", "RAG", "answerability"]
innovations: ["首个从同行评审中提取问题、由作者标注答案的科学QA数据集", "验证去上下文化+标题前置策略对长文档检索的普适增益", "建立证据检索+可答性分类+自由生成三项任务的完整基线体系"]
benchmarks: ["PeerQA", "QASPER", "QASA", "BioASQ"]
---

# 论文速读：PeerQA-A-Scientific-Question-Answering-Dataset-from-Peer-Rev

## 一句话总结
论文提出了 **PeerQA**，一个从同行评审（peer reviews）中提取问题、由论文原作者标注答案的科学文档级问答数据集（579个QA对，208篇论文），支持证据检索、答案可答性分类与自由形式答案生成三项任务，并为三项任务建立了基线系统。

## 研究问题与动机
1. **科学文献爆炸式增长**，研究者与审稿人需要自动化工具辅助论文阅读与信息抽取，而现有QA系统受限于科学领域高质量数据的匮乏。
2. **现有科学QA数据集的问题**：问题来源多为annotators（仅读摘要或泛读），导致问题"不自然"（generic、非真实审稿场景）、答案标注者缺乏领域深度；而自然产生的问题（如搜索日志）在科学领域难以获取。
3. **同行评审是天然的高质量QA来源**：审稿人是领域专家、已深度阅读论文，提出的问题具有真实性和专业性，但此前无人系统性地将其转化为结构化QA数据集。
4. **长上下文建模需求**：科学论文平均长度约12k tokens，需要面向长文档QA的评测基准，现有数据集要么过短（如ConditionalQA平均1.5k tokens）要么非科学领域。

## 核心贡献（创新点）
1. **首个基于同行评审的科学QA数据集**：PeerQA问题源于真实审稿意见，答案由论文原作者标注，与QASPER/QASA等由"实践者/学生"创建的问题和答案有本质区别——问题更自然、标注者更具权威性。
2. **同时支持三项任务（证据检索 + 可答性分类 + 自由生成）**：不同于多数数据集仅聚焦单一任务，PeerQA完整覆盖科学QA系统开发的三个关键环节。
3. **提出并验证了"去上下文化+标题前置"的检索增强策略**：发现即使简单的去上下文化方法（prepend title）也能在所有架构上稳定提升检索性能，这是针对长科学文档检索的重要经验发现。
4. **开放了带标注的579个QA对 + 12k未标注问题（2623篇论文）**，为后续Few-shot学习、无监督预训练和数据扩充提供了丰富资源。

## 方法详解
### 数据收集流程
1. **数据来源**：从NLPeer（含ARR 2022、COLING 2020、ACL 2017、CoNLL 2016、F1000）及Geoscience期刊（ESD、ESurf）、OpenReview（ICLR 2022/2023、NeurIPS 2022 D&B Track）获取论文与评审。
2. **问题提取**：提取同行评审中以问号结尾的句子，初始获得17,910个问题。
3. **问题预处理（清洗与去上下文化）**：
   - 使用 **InstructGPT**，结合前三句评审上下文，生成独立可理解的去上下文化问题，同时修正拼写/语法错误。
   - 使用** constituency parser**检测根级连词，识别复合/跟进问题，再用InstructGPT拆分为独立问题。
   - 人工过滤30%的问题（修辞性问题、无关QA问题等），最终保留12,546个问题。
4. **答案标注（四步流程）**：
   - (1) 作者审核/修改/删除问题；(2) 在PDF中高亮标注**答案证据（Answer Evidence）**；(3) 提供**自由形式答案（Free-Form Answer）**；(4) 标记**不可答（Unanswerable）**问题（如审稿问题在rebuttal中已答但未入正文）。
5. **答案增强**：因作者答案质量方差大，用**GPT-4**对自由形式答案进行改写以提升一致性和完整性。

### 实验任务与设置
- **证据检索**：构建为信息检索问题，以段落/句子为检索单元，评估模型包括 MiniLM-L12-v2、Contriever、Dragon+、GTR-XL、ColBERTv2、BM25、SPLADEv3，指标为MRR和Recall@10。
- **答案可答性**：二分类任务（Answerable vs. Unanswerable），使用指令微调LLM，以macro-F1评估类别不平衡。
- **答案生成（RAG设置）**：以检索到的段落（Top-k=10/20/50/100）或全文作为上下文，使用 Llama-3-8B-Instruct、Command-R、Mistral-7B、GPT-3.5-Turbo、GPT-4o 生成答案；评估指标包括 Rouge-L、AlignScore（基于RoBERTa-large的事实一致性）、Prometheus-2（LLM-as-judge，1-5分正确性）。

## 实验与结果
### 证据检索（Table 3）
- **最佳模型**：**SPLADEv3**（MRR=0.4536，Recall@10=0.6661，段落级），MiniLM-L12-v2次之（MRR=0.4839 含标题）。
- **段落 > 句子**：所有模型在段落级别均优于句子级别。
- **去上下文化（+Title）**： prepend标题 consistently 提升检索性能（Mistral除外），但在句子级别效果不明显（标题占比过大干扰表示）。
- **dense模型 vs. sparse vs. cross-encoder**：SPLADEv3（sparse）和 MiniLM（cross-encoder）表现最佳，Contriever（dense）较弱。

### 答案可答性（Figure 3 / Table 11）
- **类别偏差明显**：Mistral和Command-R倾向于预测"可答"（高可答recall、低不可答recall）；Llama和GPT倾向于预测"不可答"。
- **最佳macro-F1**：**Command-R**（0.5694，RAG-20）和 **GPT-4o**（0.5712，RAG-50）取得最佳平衡。
- **Gold evidence setting**：所有模型几乎完美分类（precision=1.0），说明任务难度主要在检索环节。

### 答案生成（Figure 4 / Table 12）
- **Gold evidence = 上界**：所有模型在提供标注证据时表现最佳，证明高质量证据信号的重要性。
- **RAG > Full-text**：大多数模型在检索到少量相关段落（Top-10/20）时优于全文输入，说明 retriever 过滤比依赖LLM内部机制更有效。
- **GPT-4o例外**：随上下文增大表现稳定甚至提升，反映最强模型的长上下文能力。
- **检索-生成相关性**：Retriever Recall 与生成指标呈正相关（r≤0.42），印证检索质量对生成的影响。
- **错误分析（Table 4）**：GPT-3.5最低分80例中，43.75%为评估误差（模型答对但指标给分低），12.5%部分正确，10%推理错误，11.25%证据不充分需更多上下文。

## 相关工作脉络
1. **QASPER**（Dasigi et al., 2021）：NLP从业者基于摘要提问，问题较generic，标注者为实践者；PeerQA问题源自真实审稿人、答案由作者标注，质量和自然度更高。
2. **QASA**（Lee et al., 2023）： annotators 可读全文，但仍是外部人员创建问题；PeerQA直接利用审稿环节产生的问题，无需额外招聘标注者。
3. **BioASQ**（Tsatsaronis et al., 2015）：生物医学专家自拟问题，但答案仅 grounded 在摘要层面，问题不涉及单篇论文上下文；PeerQA聚焦单篇论文内问答，问题针对性更强。
4. **Singh et al. (2024) / SciDQA**：并发工作同样从同行评审提取问题，但答案来自 rebuttal（非正式发表版本），且证据标注采用半自动映射；PeerQA答案由作者在最终发表版中人工标注，更准确可靠。
5. **NarrativeQA / QuALITY**：长上下文QA数据集，但分别面向小说/剧本和通用文章；PeerQA专攻科学论文，支持更专业的证据检索和可答性判断任务。
6. **PubMedQA / SciDefinition**：大规模半自动生成数据集，问题质量依赖模板/蒸馏；PeerQA为人工专家标注，小但精，适合作为严格评测基准。

## 局限性与未来方向
1. **数据集规模有限**（579个标注QA对），虽发布12k未标注问题供扩充，但训练数据仍偏少，模型需依赖few-shot或无监督泛化。
2. **领域覆盖不均**：80%+来自ML/NLP，Geoscience和Public Health仅占约10%，限制了跨学科通用性。
3. **仅支持英语**：框架可扩展至其他语言，但当前检索器和生成器均为英语模型。
4. **自由形式答案质量方差**：作者答案有的简洁、有的冗长，虽有GPT-4增强但仍无法完全统一；答案中可能包含论文正文之外的作者内部知识。
5. **PDF提取损失**：GROBID偶尔遗漏段落，导致部分证据无法映射（Table 5显示约31个QA对的证据映射失败），影响可评估样本量。
6. **长形式QA评估仍不成熟**：现有Rouge/AlignScore/Prometheus各有局限，需更好的人类对齐评测方法。

## 研究启发与可借鉴点
1. **同行评审作为QA数据源的范式**：将学术出版流程中自然产生的互动（审稿-答辩-发表）转化为结构化数据，避免了人工标注的成本和质量问题，该思路可迁移至其他专业领域（如法律、医疗）。
2. **"去上下文化+标题前置"的简单有效策略**：对长文档检索而言，仅需在passage前prepend标题即可稳定提升MRR和Recall，实现成本低，值得在类似任务中复现验证。
3. **RAG中"少而精"的上下文优于全文**：即使LLM支持128k context，检索Top-k段落仍比直接输入全文效果更好（GPT-4o除外），提示在科学QA场景中使用retriever是关键设计。
4. **答案可答性分类中的类别偏差问题**：不同模型表现出截然相反的bias（有的过预测可答、有的过预测不可答），提示在构建可答性评测时需关注模型校准，而非仅看macro-F1。
5. **多指标+多ground truth的评估策略**：同时使用Rouge-L、AlignScore、Prometheus-2，并与"标注证据段落"和"自由形式答案"两种ground truth对比，能更全面揭示模型能力，避免单一指标误导。

## 关键术语表
- **PeerQA**：论文提出的科学文档级QA数据集，问题来源于同行评审，答案由论文作者标注，共579个标注QA对。
- **Evidence Retrieval（证据检索）**：给定问题和论文，从论文中检索与问题相关的句子或段落，是科学QA系统的第一步。
- **Answerability Classification（答案可答性分类）**：判断问题能否在当前论文中找到足够信息回答的二分类任务。
- **Decontextualization（去上下文化）**：将通过解析工具从评审语境中提取的、依赖上下文的问题，改写为独立可理解的句子。
- **RAG（Retrieval-Augmented Generation）**：先检索相关段落，再将其与问题一起输入生成模型回答的框架。
- **Free-Form Answer（自由形式答案）**：由作者以自然语言撰写的非抽取式答案，区别于仅标注证据片段的extractive答案。
- **Unanswerable Question（不可答问题）**：论文中标记为无法从正文中找到足够依据回答的问题，常见于审稿人在rebuttal中被解答但未收入正文的情况。
- **SPLADEv3**：本次实验中证据检索表现最佳的语言模型，基于稀疏表示的语义检索模型。

## 可复现要素
- **数据集**：PeerQA标注数据（579 QA对）已公开发布（CC-BY-NC-SA 4.0）；12k未标注问题（2623篇论文）亦已发布。
- **代码**：论文提供了数据处理脚本（下载和处理ICLR/NeurIPS数据）和annotation interface代码（详见附录）。
- **检索基线模型**：MiniLM-L12-v2、Contriever、Dragon+、GTR-XL、ColBERTv2、BM25、SPLADEv3——均为开源模型，可用HuggingFace加载复现。
- **生成基线模型**：Llama-3-8B-Instruct（8k/32k）、Command-R（34B）、Mistral-7B-Instruct-v0.2、GPT-3.5-Turbo-0613、GPT-4o-0806——部分为API调用，需Azure/OpenAI账号。
- **关键超参**：检索Top-k=10/20/50/100；greedy decoding；SPLADEv3用于RAG检索；Llama-3-32k使用dynamic rope-scaling扩展上下文。
- **评估工具**：pytrec_eval（MRR/Recall）、scikit-learn（F1）、rouge-score、AlignScore（RoBERTa-large nli_sp）、Prometheus-2（7B-v2.0）。
