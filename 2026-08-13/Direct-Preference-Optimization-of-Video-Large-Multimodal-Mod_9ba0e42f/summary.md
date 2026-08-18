---
title: "Direct-Preference-Optimization-of-Video-Large-Multimodal-Mod"
source: https://aclanthology.org/2025.naacl-long.30.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:03:44"
---

# 论文速读：Direct-Preference-Optimization-of-Video-Large-Multimodal-Mod

## 一句话总结
本文提出了一种基于语言模型奖励的直接偏好优化（DPO）框架，利用高质量详细视频字幕作为视频内容的轻量代理，使 ChatGPT 能够低成本地生成带有思维链的事实对齐评分；该方法首次将 DPO 成功应用于视频大多模态模型（LMM）对齐，在开放视频问答任务上较监督微调（SFT）基线提升 8.1% 准确率。

## 研究问题与动机
1. **视频 LMM 对齐缺乏可扩展的奖励信号**：尽管 DPO 与 RL 在纯 LLM 领域已成熟，但在视频指令跟随任务中，生成内容往往涉及复杂的时空事实与幻觉，现有方法难以获得高质量、可扩展的偏好反馈。
2. **人工标注与 GPT-4V 奖励成本过高**：LLaVA-RLHF 采集 1 万条人工偏好数据需花费 \$3000；而直接使用 GPT-4V 对视频帧打分面临计算昂贵、推理延迟高、API 访问受限等问题，难以支撑大规模视频对齐。
3. **SFT 模型自评估的事实准确性不足**：现有利用 SFT 模型进行自我评估（Self-evaluation）的方案在判断答案与视频内容的事实一致性方面效果不确定，容易放大生成幻觉。
4. **细粒度视频字幕数据稀缺**：现有视频-文本数据集多为简短句或关键词（如 ActivityNet、WebVid 原有标注），缺乏涵盖时间动态、空间关系、世界知识与 OCR 信息的详细描述，制约了下游模型的细粒度理解训练。

## 核心
