---
title: "Adaptive-Prompting-Ad-hoc-Prompt-Composition-for-Social-Bias"
source: https://aclanthology.org/2025.naacl-long.122.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-08-18 16:27:57"
field: "大语言模型提示工程与公平性评估"
keywords: ["Adaptive Prompting", "Social Bias Detection", "Prompt Composition", "Shapley Values", "In-context Learning", "LLM Evaluation"]
innovations: ["实例级动态预测最优离散提示技术组合替代静态试错", "将Shapley值与交互值系统引入提示组件归因分析", "构建含互斥与时序约束的64种组合空间并验证跨模型偏好差异"]
benchmarks: ["StereoSet", "SBIC", "CobraFrames"]
---

# 论文速读：Adaptive-Prompting-Ad-hoc-Prompt-Composition-for-Social-Bias

## 一句话总结
论文提出**自适应提示（Adaptive Prompting）**框架，针对社会偏见检测任务，实时为每个输入实例预测最优的离散提示技术组合，替代传统的固定组合或单技术试错，在 StereoSet、SBIC 与 CobraFrames 三个基准上显著超越静态最优与单一技术基线。

## 研究问题与动机
- **现有方法碎片化**：自动提示优化研究多聚焦单一技术（如仅调 persona 或仅加示例），忽视多种技术的组合效应及其对输入内容的依赖性。
- **提示叠加引入噪声**：简单堆砌更多提示技术并非总是有效，LLM 无法同等注意力地处理所有信息，需动态选取最优子集。
- **社会偏见检测的适配需求**：该任务高度依赖上下文语义理解与社会世界知识，是检验提示组合自适应能力的理想场景。
- **研发成本高**：当前有效 prompt 的发现仍依赖人工试错，缺乏系统化、实例级的自动路由机制。

## 核心贡献（创新点）
- **实例级自适应组合预测**：构建编码器+多标签分类头，按输入实时输出最优提示组合概率分布，区别于静态 auto-prompt 或单技术优化。
- **结构化 64 种组合空间设计**：定义 5 类提示技术并施加互斥/时序约束（如同一变体互斥、定义必须先于示例），将组合爆炸控制在可计算范围。
- **Shapley 博弈论可解释分析**：引入 SV 与 SI 量化各提示技术的独立贡献与协同/冗余关系，为组合选择提供因果归因依据。
- **跨模型×跨数据集的系统对比**：在 3 款架构差异显著的 LLM 与 3 个社会偏见基准上验证，揭示“模型-数据集-提示”三元耦合特性。

## 方法详解
- **三步流水线**：① 标签收集：对全部定义组合在训练集上运行 LLM 并标注最优组合；② 模型训练：用标签微调编码器学习输入→最优组合映射；③ 自适应应用：推理时预测组合并调用 LLM 生成偏见分类。
- **提示技术池（5 类）**：
  - **Personas**：指定评估者视角
  - **Definitions**：显式定义社会偏见
  - **In-context Demonstrations**：3 种变体（Random / Similarity via SBERT / Category）
  - **Directional Stimulus**：提供数据集特定偏见类型线索
  - **Reasoning Step Instructions**：Chain-of-prompts 流水线将任务分解为子问题
- **组合空间公式**：$|C| = 2^{|T_1|} \cdot \prod_{t \in T_2} (|t| + 1)$，约束下实际组合数为 **64**。
- **预测模型架构**：**DeBERTA-v3-large** 编码器 + sigmoid 回归头 + binary cross-entropy loss，输出 C 维概率向量；每个数据集 × LLM 训练独立模型，5 个随机种子取平均。
- **Shapley 分析模块**：SV 计算单一技术的一阶边际贡献，SI 捕获技术对间的协同/冗余；SV/SI 可直接构建规则组合作为 adaptive prompting 的对照。

## 实验与结果
- **评估设置**：LLMs 为 Mistral-7B-Instruct-v0.2、C4AI Command-R v01 (35B)、Meta Llama 3 (70B)；使用 constrained decoding 强制二分类输出。
- **数据集**：StereoSet（刻板印象）、SBIC（隐性攻击性语言）、CobraFrames（社会情境与权力动态）。
- **基线**：Random/Majority、Self-Diagnosis (Schick et al., 2021)、DeBERTa fine-tuned、Ensemble、Best on Val/Test、Oracle。
- **StereoSet 核心结果**：Adaptive prompting 在 Mistral/Command-R/Llama 3 上分别达到 **0.809 / 0.781 / 0.853**，均超越 Best on Test（0.800 / 0.706 / 0.817）与单次技术最优（如 Similar demonstrations: 0.761 / 0.701 / 0.798）。
- **组合偏好洞察**：Mistral 极度偏好 `Dir. stim. + In-cont.(rand.)`；Command-R 倾向多组件完整组合；Llama 3 对 Persona 敏感；`In-cont.(cat.)` 在第三数据集上呈现跨模型一致性；Base composition 在多数数据集上低频被选；中等复杂度（2–4 组件）组合表现最优，复杂度非越大越好。

## 相关工作脉络
- **Self-Diagnosis (Schick et al., 2021)**：早期 GPT-2 XL 监督微调用于提示诊断，本文将其作为静态监督基线，但转向离散组合预测与多 LLM 验证。
- **DeBERTa fine-tuned**：作为纯监督微调上界参考，本文强调零样本/少样本提示组合能在接近该上界的同时保留大模型世界知识。
- **In-context Learning / Few-shot prompting**：本文将示例策略细分为随机/相似度/类别变体，并通过自适应路由取代固定 shot 数量设计。
- **Prompt tuning / P-Tuning**：区别于连续向量软提示优化，本文聚焦离散提示技术的组合选择，强调可解释性与工程可控性。
- **Shapley-based XAI**：首次将 SV/SI 系统引入提示工程组件归因，为后续 prompt 可解释性研究提供方法论参照。
- **定位差异**：从“单一技术优化/静态固定组合”转向“实例级动态组合路由+博弈论归因”，填补了社会偏见检测中提示组合自适应选择的空白。

## 局限性与未来方向
- **组合空间规模限制**：当前仅覆盖 64 种约束组合，提示技术池扩展后搜索空间可能再次面临指数增长。
- **任务域局限**：仅在 3 个社会偏见/公平性基准上验证，尚未测试至事实核查、法律推理或长文本理解等复杂下游任务。
- **预测器容量瓶颈**：DeBERTA-v3-large 作为轻量路由头，可能在极端领域偏移或罕见输入上出现概率分布平滑过度。
- **未来方向**：① 动态扩展提示技术池并引入分层/树状搜索控制组合空间；② 探索跨数据集共享路由表示或元学习初始化；③ 将 SV/SI 归因与在线反馈结合实现持续优化。

## 研究启发与可借鉴点
- **自适应路由范式可迁移**：将“输入→最优工具/提示子集”的预测逻辑迁移至 RAG 路由、多 agent 协作选择、领域自适应指令微调等场景。
- **Shapley 归因用于提示工程**：SV/SI 分析可为团队提供组件级贡献量化报告，指导 prompt 模板裁剪与冗余剔除。
- **约束驱动的组合空间设计**：互斥/时序约束（如定义先于示例）是控制组合爆炸的有效工程手段，值得在其他自动化 prompt 研究中复用。
- **跨模型偏好图谱构建**：不同架构 LLM 对同一提示技术响应差异显著，可启发“模型指纹”知识库建设，用于生产环境自动选型。

## 关键术语表
- **Adaptive Prompting**：根据输入实例实时预测并动态组装最优离散提示技术组合的框架。
- **In-context Demonstrations**：在提示中嵌入参考示例，本文细分为随机、SBERT 相似度与类别三种变体。
- **Directional Stimulus**：提供与数据集强相关的偏见维度线索，引导模型聚焦特定判断标准。
- **Shapley 值 (SV)**：基于合作博弈论的边际贡献度量，用于量化单个提示技术的独立影响力。
- **Shapley 交互值 (SI)**：捕获两个及以上提示技术协同或冗余程度的二阶归因指标。
- **Constrained Decoding**：通过解码约束强制 LLM 仅输出预设类别标签，确保分类格式合规。
- **StereoSet / SBIC / CobraFrames**：分别评估刻板印象倾向、隐性攻击性语言与社会情境权力动态的 3 个主流偏见基准。

## 可复现要素
- **数据集**：StereoSet、SBIC、CobraFrames（均为公开开源数据集，论文未声明自建私有数据）
- **代码/权重**：论文未明确声明开源仓库与模型权重
- **关键超参**：DeBERTA-v3-large 作为编码器；sigmoid 回归头 + BCE loss；64 种组合空间；5 个随机种子；constrained decoding 强制二分类输出；每数据集×LLM 独立训练路由模型。
