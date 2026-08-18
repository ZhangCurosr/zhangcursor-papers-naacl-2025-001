---
title: "CFinBench-A-Comprehensive-Chinese-Financial-Benchmark-for-La"
source: https://aclanthology.org/2025.naacl-long.40.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:00:28"
---

# 论文速读：CFinBench-A-Comprehensive-Chinese-Financial-Benchmark-for-La

## 一句话总结
本文提出了CFinBench，这是目前规模最庞大、分类最系统的中文金融大语言模型评测基准，涵盖99,100道题与43个二级类别；实验显示当前顶尖模型在该基准上的最高平均准确率仅为66.02%，表明中文金融专业知识的掌握仍存在显著瓶颈。

## 研究问题与动机
- **现有中文金融基准规模小、题型单一：** 如FinEval（4,661题仅含单选）和FinanceIQ（7,173题仅含单选）难以有效区分先进LLM的真实能力上限。
- **分类逻辑与职业路径脱节：** 传统基准多按经济学/金融学静态学科划分，未覆盖中国金融从业者“学科基础→资格认证→实务岗位→法律合规”的能力进阶链条。
- **公开题库数据污染风险高：** 大量题目源自网络公开模拟卷与年报，极易混入训练集导致验证/测试分数虚高，缺乏系统的去重与防泄漏机制。
- **英文基准无法直接移植：** FLUE/FLARE等英文金融评测侧重基础NLP任务或时间序列推理，缺失中国特有的法规体系、考试制度与实务语境。

## 核心贡献（创新点）
- **构建职业轨迹驱动的四维评测框架：** 首次以Financial Subject、Qualification、Practice、Law为一级分类，43个二级类别精准映射中国金融从业者的真实成长路径。
- **推出迄今最大的中文金融专项数据集：** 汇总99,100道题目（单选/多选/判断），题量约为现有同类基准的20倍以上，大幅提升评测区分度。
- **设计多重抗污染与质量增强流水线：** 融合MinHash内外去重、GPT-4语义改写、最远选项交换（farthest option swapping）与三轮人类专家交叉校验，有效抑制选项位置记忆与数据泄漏。
- **提供全尺度模型基线评测与归因分析：** 覆盖<5B至>65B共二十余款模型，系统揭示模型规模、领域预训练、对齐策略与COT提示对金融知识表现的影响规律。

## 方法详解
- **分层Taxonomy设计：** 一级类别对应“学科基础→职业资格→实务操作→法律法规”四阶段；二级共43类，如政治经济学、注册会计师、初级银行从业、证券法、税法I/II等。
- **多源数据采集与格式标准化：** 来源包括互联网模拟考题库、公开经济学/法学教材、上市公司公告与财报、金融新闻资讯；使用PyMuPDF与EbookLib解析PDF/EPUB，统一单选题为4选项、多选题为4或5选项。
- **清洗与去重：** 先用fastText过滤非中文段落，再通过正则与启发式规则剔除乱码、无效标点与连续空白；内部去重采用MinHash，外部去重覆盖BBT-CFLEB、FinEval、CFLUE等已有基准。
- **防污染与多样性增强：** 随机打乱选项顺序，并执行“最远选项交换”（将正确答案与距离最远的错误答案互换）；随后调用GPT-4基于新顺序重写题干，保持语义一致的同时打破位置规律。
- **人机协同校验：** >10名金融背景专家分三轮验证：首轮2个月筛除约2万低质题，次月精修产出约3万高质量题，末轮10天抽样复审；最终保留约9.9万题。
- **评估设置与指标：** 基于OpenCompass框架，temperature=1.0、top_p=1.0、贪婪解码；输入截断至2048 tokens，输出限制64 tokens；最终得分按 $final = 0.4 \times \text{single} + 0.4 \times \text{multiple} + 0.2 \times \text{judgment}$ 加权计算。

## 实验与结果
- **数据集切分：** 每二级分类内随机划分开发集（3例用于few-shot）、验证集与测试集（比例2:8），测试集共99,100题。
- **评测基线：** 涵盖GPT-4、ChatGPT及Llama/Qwen/ChatGLM/Baichuan/InternLM/Yi/XuanYuan/FinMA/Phi/TigerBot/Skywork/Gemma/Mistral/DISC-FinLLM/CFGPT等系列，参数规模跨<5B至>65B。
- **核心结果：** 在3-shot Answer-Only设置下，**Qwen2.5-72B以66.02%**平均准确率领跑；GPT-4为56.77%，Yi1.5-34B为60.16%。10B~20B区间Qwen2.5-14B达60.12%、InternLM2.5-20B为55.82%；5B~10B区间Qwen2.5-7B与中国垂直模型YunShan-7B均达57.36%，性能超越部分70B级通用模型。
- **关键发现：**
  - 多数模型3-shot优于0-shot，但Phi3-14B-Instruct因训练方式与示例不匹配导致零样本更优。
  - 领域专属预训练/微调（如YunShan、XuanYuan注入高质量金融语料）显著提升表现。
  - COT提示对多选题无增益，Qwen-14B与ChatGLM3反而下降，验证了事实型/规则型任务中CoT的有效性边界。
  - 与FinEval交叉对比显示，同一模型在FinEval可达86%+，而在CFinBench仅约60%，证明本基准区分度更高、难度更强。
  - 数据污染实验表明：用验证集60%数据LoRA微调InternLM2.5-1.8B后，测试集平均准确率从33.00%升至42.85%，凸显开源评测数据防泄漏的必要性。

## 相关工作脉络
- **FLUE/FLARE：** 英文金融基础NLP与时间序列推理基准，CFinBench转向中文语境并补齐职业资格、实务岗位与法律法规维度。
- **BBT-CFLEB / CFBenchmark / CGCE：** 早期中文金融评测多侧重基础理解或体量过小（CGCE仅150题），CFinBench聚焦高阶知识与复杂推理，规模提升近一个数量级。
- **FinEval / FinanceIQ：** 同为中文金融知识评测，但仅含单选且题量不足5k/7k；本文证明自身难度显著更高，且题型与分类体系更贴近实战。
- **CFLUE / Ant-Fin-Eva：** 覆盖应用任务但缺乏职业晋升主线；CFinBench以从业者成长路径为轴重构分类，体系更系统。
- **金融垂类模型（BloombergGPT/FinMA/DISC-FinLLM等
