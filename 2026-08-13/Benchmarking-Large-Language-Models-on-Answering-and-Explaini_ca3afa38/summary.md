---
title: "Benchmarking-Large-Language-Models-on-Answering-and-Explaini"
source: https://aclanthology.org/2025.naacl-long.182.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:59:30"
field: "医疗自然语言处理"
keywords: ["medical question answering", "large language models", "clinical reasoning", "explainable AI", "benchmark dataset", "chain-of-thought prompting", "model explanation evaluation"]
innovations: ["构建含专家解释的两个挑战性医疗QA数据集（JAMA Clinical Challenge和Medbullets）", "系统评估LLM在复杂临床问答和解释生成上的能力并揭示自动指标与人工判断的低相关性", "分析CoT提示在医疗QA中新引入的错误类型（None/Made-up/Multiple）"]
benchmarks: ["MedQA", "JAMA Clinical Challenge", "Medbullets"]
---

# 论文速读：Benchmarking-Large-Language-Models-on-Answering-and-Explaini

## 一句话总结
本文构建了两个包含专家撰写解释的挑战性医疗QA数据集（JAMA Clinical Challenge 和 Medbullets），系统评估了七个LLM在复杂临床病例问答与解释生成上的能力；实验表明新数据集显著高于现有基准的难度，且现有自动评估指标与人工判断的关联性较弱。

## 研究问题与动机
- 现有医疗LLM评测多依赖医学执照考试（如USMLE）或教科书知识型问题，难以反映真实临床决策的复杂性。
- 已有benchmark普遍缺乏参考解释，无法有效评估模型推理过程的可解释性——而这正是辅助医生决策的核心需求。
- 部分新近临床病例benchmark规模小、覆盖领域窄，或未提供专家级解释；MedQA等虽含类似题型，但发布时间较早且不含解释。
- 评估体系上，自动指标（ROUGE/BERTScore等）与人类对解释质量的判断相关性较弱，需探索更适合可解释医疗QA的评测方法。

## 核心贡献（创新点）
1. **构建两个高质量医疗QA数据集**：JAMA Clinical Challenge（1524条真实临床案例）和 Medbullets（308条USMLE Step 2&3风格题目），均配有专家撰写的详细解释，填补了现有benchmark无解释的空白。
2. **提出含解释生成能力的统一评测框架**：除预测准确率外，系统性地评估模型在 XY*→R 提示下的解释质量，兼顾自动指标与人工评估，首次将"解释生成"纳入医疗LLM评测的核心维度。
3. **揭示现有自动指标在医疗解释评估中的局限性**：发现CTC Consistency和BARTScore+会误判低质量解释，且多数自动指标与人工评分的相关性极弱（Pearson接近零），为后续评测方法改进提供了明确方向。
4. **系统性分析不同提示策略的效果差异**：对比X→Y、X→RY（CoT）和XY*→R三种策略，发现few-shot和CoT对部分模型（GPT-3.5/PaLM 2/Llama 2）几乎无效，并揭示了CoT引入的新错误类型（如"None of the above"、"Made-up"选项）。

## 方法详解
- **数据集构建**：JAMA Clinical Challenge 来自 JAMA Network 临床挑战档案（2013.07–2023.10），覆盖13个医学领域，共1524条文本案例；Medbullets 来自 X（Twitter）公开推送的 Medbullets Step 2/3 资源（2022.04–2023.12），共308条。两数据集均已排除图像，适配纯文本LLM。
- **提示策略**：
  - X→Y：直接要求模型选择答案；
  - X→RY（CoT）：两阶段提示，第一阶段生成逐步推理（"Let's think step by step and walk through all the choices in detail."），第二阶段基于推理输出最终答案；
  - XY*→R：给定输入和正确答案，要求模型解释为何该选项最优，其他选项为何错误。
- **Few-shot 设置**：采用 leave-one-out 交叉验证，分别测试0/2/5-shot效果，示例格式与zero-shot保持一致，仅将模型输出替换为gold答案或解释。
- **评估指标**：
  - 预测：准确率（Accuracy）；
  - 自动解释评估：ROUGE-L（表面形式相似度）、BERTScore、BLEURT、BARTScore+/++（语义相似性）、CTC三指标（Consistency/Relevance/Preservation）、G-Eval三指标（Coherence/Consistency/Relevance，以GPT-4o为backbone，温度=1，max tokens=5，均取5次运行均值）；
  - 人工评估：3名具硕士/博士学位的医学/健康领域crowdworker对30个样本进行三维度评分（Completeness/Correctness/Relevance，Likert 1–5分），每位标注者独立打分后取均值，报告 inter-annotator agreement（S系数：Correctness=0.60，Relevance=0.65）。

## 实验与结果
- **预测准确率（X→Y，零样本）**：
  - GPT-4 在 MedQA-4 上为 78.63%，降至 Medbullets-4 的 66.23%（-12.4%）、JAMA 的 67.32%；其余模型普遍下降 5%–12%。
  - JAMA 与 Medbullets-4 难度相当（GPT-3.5/4、PaLM 2、Llama 3 在两数据集表现相近），MedAlpaca（13B）和 Meerkat（7B）在JAMA上显著下滑至36.48%和45.99%。
  - MedQA Step 1 与 Step 2/3 难度相近，Medbullets 更难的原因主要来自数据新近性而非题型差异。
- **Few-shot 效果**：GPT-4 和 Llama 3 从 few-shot 中获益；GPT-3.5/PaLM 2 几乎无提升；Llama 2/MedAlpaca 在 JAMA 上随 shot 数增加反而下降（MedAlpaca 受512 token 输出限制）；Meerkat 整体受损。
- **CoT 效果**：在 MedQA 和 Medbullets 上普遍有提升（GPT-4：78.63%→82.64%），但在 JAMA 上仅 GPT-3.5（48.69%→50.13%）和 Meerkat（45.99%→49.86%）改善；CoT 引入新错误类型："None of the above"（最常见）、"Made-up"新选项、"Multiple"多选。
- **解释生成自动评估**：GPT-3.5/GPT-4/Llama 3/Meerkat 在多数指标上表现相近；Meerkat 在 CTC Relevance 上最优；G-Eval 偏好 Llama 3 和 GPT-4；MedAlpaca 因输出含输入片段导致 CTC Consistency 异常偏高。
- **人工评估（Medbullets-5，GPT-4 vs PaLM 2）**：GPT-4 在 Completeness（3.35 vs 2.67）、Correctness（4.45 vs 4.35）、Relevance（4.61 vs 4.53）三项均优于 PaLM 2。
- **自动-人工相关性**：Pearson 相关系数极低，多数指标接近零甚至为负，仅 G-Eval Consistency 与 Correctness 呈弱正相关（0.26），凸显了医疗解释自动评测指标的不足。

## 相关工作脉络
- **MedQA**（Jin et al., 2021）：含 USMLE 风格题库，但无解释且数据较旧（2021年3月采集）；本文 Medbullets 数据更新且含解释，难度更高。
- **MedMCQA**（Pal et al., 2022）：面向印度医学入学考试的多选题集，侧重知识检索而非临床推理，且无解释。
- **MeEval**（He et al., 2023）：多任务多领域评测，但主要关注分类和文本生成，未聚焦复杂临床推理与解释。
- **K-QA**（Manes et al., 2024）、**CLAUDE-MED**（Wang et al., 2024）：收集真实世界对话数据贴近临床实践，但同样缺少专家级解释。
- **MedExQA**（Kim et al., 2024b）：提供多个解释，但基于模拟题/在线考试，非真实临床病例，解释质量不及本文专家撰写版本。
- **ExplainCPE**（Li et al., 2023a）：面向中国药师考试，解释对象和语言不同；本文聚焦英文临床病例和专科医生解释。

## 局限性与未来方向
- 未采用更复杂的提示策略（如集成方法、动态 few-shot 选择），可能限制模型性能上限；未来可探索基于参考解释的 few-shot CoT 提示（需重构解释为思维链格式）。
- 数据已排除图像（X光、病理切片等），忽略了多模态临床信息；未来将在完整数据集上评估 GPT-4V/Gemini 等多模态模型。
- 自动评估指标与人工判断相关性弱，亟需开发专门针对医疗解释质量的新评估体系。
- JAMA 数据因版权限制无法公开共享，限制了社区复现与扩展。

## 研究启发与可借鉴点
- **高质解释数据的价值**：专家撰写解释作为 gold standard 可显著提升医疗LLM评测的区分度，未来可尝试构建更多带专家解释的垂直领域benchmark。
- **CoT 新错误类型分析框架**：对"None of the above"、"Made-up"、"Multiple"三类错误的系统性统计为推理类任务的错误分析提供了可复用的分类模板。
- **自动-人工相关性检验作为必要基线**：任何新评估指标的设计都应报告与人工评分的相关系数，本文的弱相关结果为后续指标改进提供了明确参照。
- **医学背景crowdworker评估模式**：通过 Prolific 招募具硕士/博士学位的医学健康领域标注者，保障了评估的专业性，该人群筛选策略可直接迁移至其他专业领域评测。
- **医疗小模型 vs 大模型差距分析**：MedAlpaca（13B）和 Meerkat（7B）在复杂临床任务上显著落后于70B开源模型，提示医学LoRA微调仍不足以弥补基础模型规模差异，未来可在数据增强（如合成CoT）和推理链设计方面寻找突破。

## 关键术语表
- **JAMA Clinical Challenge**：取自《美国医学会杂志》临床挑战档案的真实复杂病例题库，共1524条，每道题含病例描述、4个选项及专家撰写解释。
- **Medbullets**：来自 X 平台公开推送的 USMLE Step 2&3 风格模拟题库，共308条，每道题含5个选项及详细解释。
- **X→Y / X→RY / XY*→R**：三种提示策略，分别表示直接答题、思维链推理后答题、给定答案后生成解释。
- **Chain-of-Thought (CoT)**：通过引导模型逐步推理（如"Let's think step by step"）来提升复杂推理任务准确性的提示技术。
- **CTC (Compression, Transduction, Creation)**：统一的NLG评估框架，本文采用其三个子指标：Consistency（一致性）、Relevance（相关性）、Preservation（信息保留度）。
- **G-Eval**：以LLM（本文用GPT-4o）为backbone，基于CoT+表单填充范式评估生成质量的框架，涵盖Coherence/Consistency/Relevance三维度。
- **数据污染（Data Contamination）**：评估集数据混入模型训练集导致的性能虚高；本文用 TS-Guessing 方法验证新数据集污染率≤10%。
- **鲁棒性（Robustness）**：指模型性能对选项顺序的敏感性；本文通过随机打乱选项验证数据集不受顺序偏差影响。

## 可复现要素
- **数据集**：Medbullets 已公开可下载；JAMA Clinical Challenge 因版权限制不公开，作者提供URL及爬虫脚本，需通过机构订阅获取原始内容。
- **代码**：论文未明确提及代码仓库链接。
- **权重**：实验模型均为 API 调用或公开权重（Llama 2/3、MedAlpaca、Meerkat），GPT-3.5/4/PaLM 2 为闭源API。
- **关键超参**：GPT系列 temperature=1, top_p=1；Llama 2 temperature=0.8, top_p=0.95, repetition penalty=1.1, max output=1024；Llama 3 temperature=0.85, top_p=0.95；PaLM 2 temperature=0.8；MedAlpaca float16量化, max output=512, repetition penalty=1.1, beam=2；Meerkat greedy解码, float16, max output=1024, temperature=0.7, repetition penalty=1.0。所有结果为单次运行（single run）。
