---
title: "IrokoBench-A-New-Benchmark-for-African-Languages-in-the-Age"
source: https://aclanthology.org/2025.naacl-long.139.pdf
model: agnes-2.5-flash
chunks: 6
summarized_at: "2026-08-18 16:07:16"
---

# 论文速读：IrokoBench-A-New-Benchmark-for-African-Languages-in-the-Age

## 一句话总结
本文提出了 **IrokoBench**，首个针对非洲语言的综合性大语言模型评测基准，系统评估了开源与闭源模型在自然语言推理、常识知识与数学推理任务上的多语言表现，揭示了低资源非洲语言在当前主流 LLM 中的严重能力短板、提示模板敏感性以及翻译引入的额外误差。

## 研究问题与动机
- 现有主流多语言 LLM 评测基准（如 MMLU、XNLI、CMU-GSM8K）主要覆盖高资源语言，缺乏针对非洲语言的系统性评测工具。
- 非洲语言（如沃洛夫语、瓦伊语等）在预训练语料中占比极低，但使用者众多，其真实语言能力未被充分摸底。
- 当前多语言评测往往忽略提示模板差异与“翻译链路”误差，导致跨任务、跨语言性能对比不可靠。
- 商业闭源模型与开源模型在非洲语言上的性能鸿沟缺乏大规模、标准化的对比数据，制约了社区后续研究投入与基线选取。

## 核心贡献（创新点）
- 构建了 **IrokoBench** 基准，涵盖 AfriXNLI（自然语言推理）、AfriMMLU（常识知识）与 AfriMGSM（数学推理）三个子任务，首次系统化覆盖 17 种非洲语言及英法双语。
- 提出了统一的 **5 种提示模板（t1–t5）** 并进行系统敏感性分析，揭示了同一模型在不同 prompt 下性能波动显著，指出多语言评测必须标准化 prompt 策略。
- 引入了 **In-language（母语作答）** 与 **Translate-test（翻译后作答）** 的双轨评测协议，量化了机器翻译引入的误差，证明母语直接推理显著优于翻译链路。
- 建立了以 **AfroXLMR-76L** 为基线的对比框架，证实非洲语言专用的 MLM 预训练能带来最大 **+16.1 分** 的绝对提升，明确了“预训练语言覆盖”是非洲语言能力的核心瓶颈。

## 方法详解
- **基准构建**：基于 MADLAD 清洗语料统计各语言 Web 单语数据量（MB），覆盖 17 种非洲语言 + 英语/法语；分别以 XNLI、MMLU、CMU-GSM8K 为原型构建 AfriXNLI、AfriMMLU、AfriMGSM。
- **评估工具与协议**：采用 **EleutherAI LM Evaluation Harness（lm-eval）** 作为标准化工具，支持 log-likelihood、perplexity 与 generation 三种评估模式；闭源模型因 API 限制无法获取 log probabilities，通过 **verbalizer** 提取答案，开源模型直接使用 log-likelihood 请求。
- **Prompt 模板设计**：每个任务设计 5 种 prompt 模板，进行敏感性实验，表格中标注 **"Best prompt in Gray"** 并报告各模板得分及平均得分（ave），避免单一 prompt 导致的性能误判。
- **双轨数学推理设置**：AfriMGSM 设定三种模式：In-language（母语直接作答）、In-languagem（另一种母语纯文本设置）、Translate-test（先将题目翻译为目标语言再作答），用于分离语言理解与翻译误差。
- **翻译质量辅助评估**：使用 **AfriCOMET** 自动评估翻译质量，发现阿姆哈拉语、约鲁巴语、祖鲁语相关性较好；林加拉语、特维语、沃洛夫语因底模型 AfroXLMR-large 未覆盖，自动评估相关性极低（Pearson 仅 0.279）。
- **基线选择机制**：通过 XNLI 跨语言迁移实验对比 XLM-R-large、Serengeti、AfroXLMR-base 与 AfroXLMR-76L，最终以覆盖全部 17 种语言的 **AfroXLMR-76L** 作为所有 LLM 的性能对照基线。

## 实验与结果
- **评测范围**：涵盖 12+ 开源模型（LLaMA 3/3.1 8B/70B、Gemma 2 9B/27B、Aya-101、BLOOMZ 7B、CommandR 等）与 5+ 闭源 API 模型（GPT-3.5/4/4o、Gemini-1.5-pro、Claude Opus）。
- **AfriXNLI 结果**：`gpt-4o-2024-08-06` 平均 **54.8** 分领先，`gemini-pro-1.5` 为 **53.6**；开源最佳 `Gemma 2 27B` 仅 **39.2**，`Aya-101` 约 **29.6**。英语/法语普遍 >75，部分非洲语言（`xho`, `yor`, `wol`）仅 20~40。
- **AfriMMLU 结果**：`gpt-4o` 平均 **48.7**，`gemini-pro 1.5` 为 **46.6**；开源最佳 `LLaMa 3.1 70B` 为 **38.1**。低资源语言 `vai=3.4`、`wol=26.4` 成为显著短板。
- **AfriMGSM 结果**：In-language 设置下 `gpt-4o` 最佳（**64.3**），`afro-xlm
