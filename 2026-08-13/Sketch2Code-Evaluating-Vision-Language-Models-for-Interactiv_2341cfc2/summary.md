---
title: "Sketch2Code-Evaluating-Vision-Language-Models-for-Interactiv"
source: https://aclanthology.org/2025.naacl-long.198.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:56:50"
field: "多模态人机交互与代码生成"
keywords: ["Vision-Language Models", "Sketch-to-Code", "Interactive UI Generation", "Multi-turn Evaluation", "Web Design Automation"]
innovations: ["首个 sketch-to-code 多轮交互评估基准", "提出 IoU-based 布局相似度度量指标", "发现用户偏好与模型能力的关键差距"]
benchmarks: ["Sketch2Code", "Visual Similarity", "Layout Similarity (IoU-based)"]
---

# 论文速读：Sketch2Code-Evaluating-Vision-Language-Models-for-Interactiv

## 一句话总结
本文提出了 Sketch2Code 基准，用于评估视觉语言模型（VLMs）将低 fidelity 线框草图转换为网页 HTML 代码的能力，并设计了多轮交互式评估框架（反馈跟随与主动提问两种范式），揭示了当前 VLMs 在布局理解上的不足及其与用户期望之间的差距。

## 研究问题与动机
1. **现有方法依赖高 fidelity 输入**：当前 UI/UX 自动化研究多使用 Figma 设计稿或详细截图作为输入，门槛高且耗时长，不适合早期设计迭代阶段。
2. **草图作为设计入口的价值被忽视**：草图是设计师低 fidelity、易上手的概念表达工具，但自动化的 sketch-to-code 工作多基于传统模式匹配，缺乏端到端的 VLM 评估基准。
3. **单轮生成不足以处理草图歧义**：草图缺少样式、文本等细节信息，单轮直接生成难以准确还原设计意图，需要多轮交互澄清。
4. **VLMs 的交互式 UI 生成能力未被系统评估**：现有工作（如 Si et al., 2024; Laurençon et al., 2024）仅关注截图到代码的单轮生成，缺乏对用户交互范式的评估。

## 核心贡献（创新点）
1. **提出首个 sketch-to-code 交互式评估基准**：收集了 731 张来自 484 个真实网页的手绘草图，建立了完整的 benchmark 数据集。
   与已有工作的区别：不同于仅评估截图到代码的 Si et al. (2024)，本文聚焦低 fidelity 草图输入，更贴近真实设计工作流。

2. **设计双范式多轮交互评估框架**：引入"反馈跟随"（agent 被动接收用户指令）和"主动提问"（agent 主动澄清设计细节）两种多轮对话场景。
   与已有工作的区别：首次将交互范式纳入 UI 代码生成的评估，模拟真实人机协作流程。

3. **发现模型能力与用户期望的关键差距**：用户研究揭示专家更偏好主动提问模式，尽管当前模型在该模式下表现不佳；而反馈跟随模式虽更有效，却不符合同意用户期望。
   与已有工作的区别：不仅评估技术性能，还通过用户研究验证了人机协作偏好的错位。

4. **提出 IoU-based 布局相似度度量**：针对草图缺乏样式信息的特性，设计了仅关注视觉组件空间重叠的 layout similarity 指标。
   与已有工作的区别：弥补了传统视觉相似度（依赖文本和颜色）在草图场景下的不足。

## 方法详解
1. **数据集构建**：从 Si et al. (2024) 的 484 个真实网页出发，招募 21 名具有 UI 设计背景的 annotators（通过 Prolific），按照标准 wireframing 规范绘制草图（含 731 张），其中 18.0% 的网页由 2 位设计师绘制，16.5% 由 3+ 位设计师绘制以覆盖风格多样性。

2. **任务定义**：
   - **Direct Generation**：给定草图（可选文本内容），直接生成 HTML+CSS 代码。
   - **Feedback Following**：agent 每轮生成原型，LLM 模拟用户比较当前实现与参考网页后提供反馈，agent 在下一轮修正。
   - **Question Asking**：agent 每轮主动提出澄清问题，模拟用户回答，agent 基于答案更新实现。

3. **评估指标**：
   - **Visual Similarity**：沿用 Si et al. (2024) 的 Block Match、Text、Position、Color、CLIP 五类指标平均。
   - **Layout Similarity（IoU-based）**：定义七类视觉组件（text blocks, images, video containers, navigation bars, forms/tables, buttons, dividers），计算各类组件边界框总面积的 IoU：
     $$IoU(c) = \frac{A_c' \cap A_c}{A_c' \cup A_c}$$
     总体相似度为加权平均：
     $$Sim_{Layout} = \sum_{c \in C} \frac{A_c' + A_c}{\sum_{c' \in C}(A_{c'}' + A_{c'})} \times IoU(c)$$

4. **提示策略**：采用 direct prompting 和 text-augmented prompting（提供网页文本内容），后者在多轮场景中表现更稳定。

5. **模拟用户设计**：基于 GPT-4o (temperature=0) 构建，feedback following 任务中仅提供草图+截图进行视觉对比；question asking 任务中额外提供 HTML 代码以保证回答准确性，限制回答不超过一句。

## 实验与结果
1. **数据集与模型**：731 张草图，测试 10 个模型（8 个商业模型：GPT-4o/GPT-4o mini/Claude 3.5 Sonnet/Claude 3 Opus/Sonnet/Haiku/Gemini 1.5 Pro/Flash；2 个开源模型：InternVL2-8b/Llava-1.6-8b）。

2. **直接生成结果（Table 1）**：
   - 商业模型显著优于开源模型：Claude 3.5 Sonnet 在 direct prompting 下达到最佳 layout similarity 21.64%，GPT-4o 为 19.20%。
   - 开源模型 InternVL2-8b 和 Llava-1.6-8b 的 layout similarity 均低于 10%（分别为 10.08% 和 6.68%）。
   - Text-augmented prompting 对较小商业模型（如 GPT-4o Mini）提升显著（+4.76%），但对强模型（Claude 3.5 Sonnet）提升有限（+0.62%）。
   - Human satisfaction 与 IoU layout similarity 呈强相关（$r^2$=0.87, Kendall's Tau=0.72）。

3. **多轮交互结果（Figure 3, Table 4-5）**：
   - **Feedback Following**：所有模型均有明显提升，最佳商业模型在五轮内视觉相似度提升最高 7.1%，布局相似度提升 2.7%。
   - **Question Asking**：所有模型均难以提出有效问题，性能提升不显著甚至下降；多数改进发生在前两轮，3-4 轮后性能 plateau 或退化。

4. **模拟用户有效性（Table D.5）**：
   - 83.3% 的模拟反馈能准确指出当前实现与参考实现的差异。
   - 86.7% 的模拟回答忠实于参考网页。

5. **用户研究（Section 5）**：
   - 8 位 UI/UX 专家均认可 sketch-to-code agent 的实用价值。
   - 7/8 专家强烈偏好 question-asking 模式，认为 agent 应承担更多认知负荷。
   - 人类专家提出的问题比模型生成的问题更有效：每轮提问使视觉相似度提升 0.58% vs. 模型提问的 -1.12%。

## 相关工作脉络
1. **前端 UI 代码生成**：Pix2Code (Beltramelli, 2017) 开创 CNN+RNN 端到端 UI 转码，但受限于复杂视觉编码；Azure Sketch2Code (2018) 和 Robinson (2019) 基于传统 CV 方法生成 HTML，支持有限；Soselia et al. (2023) 使用视觉 critic 进行 fine-tuning，但仅覆盖简单元素。本文首次在 sketch 输入下系统评估 VLMs 能力。

2. **截图到代码的 VLM 研究**：Si et al. (2024) 和 Laurençon et al. (2024) 证明 VLMs 可将截图转为 HTML，但截图作为输入在实际设计流程中不现实。本文填补了低 fidelity 草图输入的空白。

3. **草图到代码的早期探索**：Zhang et al. (2024a) 展示了 VLM 在 sketch-to-code 的应用，但缺乏综合 benchmark 和框架设计。本文首次提供完整基准和多轮交互评估。

4. **多轮 Agent 交互评估**：SWE-bench (Jimenez et al., 2023, 2024) 评估 LLM 解决 GitHub issue 的能力，但聚焦代码修复而非 UI 设计。本文将其思想引入设计领域。

5. **VLM 的代码生成能力**：Codex (Chen et al., 2021)、StarCoder (Li et al., 2023b) 等代码专用模型在 HumanEval/MBPP 上表现优异，但忽略了网页实现等真实任务。本文揭示了 open-source VLMs（如 InternVL2）在代码生成上的显著差距。

## 局限性与未来方向
1. **开源模型仅评估 8B 参数级别**：更大参数规模的开源 VLMs 可能具备更强的多轮交互能力，但未纳入评估。
2. **多轮评估计算成本高**：单样本需 40,000-160,000 输入 tokens 和约 10,000 输出 tokens，限制了大规模评估。
3. **输入/输出模态单一**：用户研究反映专家更偏好鼠标点击、拖拽等确定性交互，以及导出到 Figma 等工具，当前仅支持自然语言+HTML。
4. **潜在滥用风险**：可能被用于生成有害网页或逆向工程受版权保护的设计。
5. **未来方向**：(i) 训练开源模型支持多轮 UI 生成；(ii) 开发更主动、具备认知推理能力的 agentic 框架；(iii) 构建端到端 UI/UX AI 应用，提升设计师生产力。

## 研究启发与可借鉴点
1. **多轮交互范式设计**：feedback following 与 question asking 双范式可直接迁移至其他 UI/UX 自动化任务（如图标生成、配色建议）的评估。

2. **IoU-based 布局度量**：针对低 fidelity 输入（草图、线框图）的场景，可借鉴此度量思想，剥离样式噪声，专注于结构和布局评估。

3. **模拟用户（Simulated User）设计**：用 LLM 模拟用户进行多轮交互评估，可降低人工标注成本。本文的 prompt 设计（限制回答长度、禁止 code leakage）值得复用。

4. **人类专家 vs. 模型提问有效性对比**：揭示当前模型的提问能力不足，但人类专家提问显著提升性能——这一发现启示可探索"human-in-the-loop"或"expert-guided"的多轮交互微调策略。

5. **合成数据生成管道**：附录 H 提供的自动 sketch 生成工具（Canny edge + Bezier curves 模拟手绘风格）可用于大规模数据合成，支持模型训练和评估。

## 关键术语表
**Sketch2Code**：将手绘 UI 线框草图自动转换为网页 HTML/CSS 代码的任务。
**Vision-Language Models (VLMs)**：能够同时处理视觉和文本输入的多模态大语言模型。
**Layout Similarity (IoU-based)**：基于边界框交集并集比的布局相似度度量，专注于组件空间重叠而非样式。
**Feedback Following**：多轮交互范式，agent 根据用户反馈逐步改进生成结果。
**Question Asking**：多轮交互范式，agent 主动提问以澄清设计歧义。
**Simulated User**：基于 LLM 构建的虚拟用户，用于自动化多轮交互评估。
**Wireframing Conventions**：UI 设计的线框图规范，如用带 X 的方框表示图片、波浪线表示文本。
**Text-Augmented Prompting**：在草图输入基础上额外提供网页文本内容作为上下文的提示策略。

## 可复现要素
- **数据集**：731 张草图 + 484 个参考网页（基于 ODC-By 授权），来源于 Si et al. (2024)。论文未提供明确下载链接，但附录 H 包含合成 sketch 生成代码。
- **代码/权重**：论文未开源代码和权重；商业模型通过 API 调用（GPT-4o、Claude 3 系列、Gemini 1.5 系列）。
- **关键超参**：temperature=0.0, max tokens=4096, top p=1.0；开源模型额外使用 repetition penalty=1.1, temperature=0.5, best-of-3 sampling。
- **评估设置**：direct prompting 与 text-augmented prompting；多轮交互最多 5 轮，基于 50 个样本的子集测试。
