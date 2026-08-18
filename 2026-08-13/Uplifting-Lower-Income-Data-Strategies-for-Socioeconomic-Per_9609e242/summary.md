---
title: "Uplifting-Lower-Income-Data-Strategies-for-Socioeconomic-Per"
source: https://aclanthology.org/2025.naacl-long.106.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:59:39"
---

# 论文速读：Uplifting-Lower-Income-Data-Strategies-for-Socioeconomic-Per

## 一句话总结
本文通过设计融合语言、地理位置与社会经济属性的提示词（Prompt）策略，探索如何在多模态大模型（LMM）中提升对低收入/非西方图像数据的检索性能，并证实地理与社会经济后缀能有效触发模型“视角偏移”，从而显著改善欠代表性数据的召回表现。

## 研究问题与动机
- **核心问题：** 主流 LMM（如 CLIP、NLLB-CLIP-SigLIP）因训练语料中高收入/西方国家样本占绝对主导，导致对低收入家庭及非西方国家图像的图文对齐与检索性能显著偏低，进一步加剧 AI 应用的“数字鸿沟”。
- **现有方法不足：** 依赖网络爬取+LMM 自动过滤（如 LAION-5B）会继承并放大模型既有偏见；单纯将英文 Prompt 机器翻译为当地语言无法弥合性能差距，且低资源语言翻译质量本身有限。
- **研究空白：** 缺乏系统评估非英文语言、地理属性（国家后缀）与社会经济属性（收入等级后缀）作为 Prompt 组件时，对 LMM 跨收入/跨地区图像召回率的影响机制。
- **目标：** 提出并评估一系列属性集成 Prompt 策略，识别其在不同国家与经济层级下的有效边界，为缓解 LMM 表现不平等提供低成本、无需重新训练的后处理方案。

## 核心贡献（创新点）
1. **实证揭示“翻译无效”陷阱：** 首次系统证明将英文主题 Prompt 直译为图像来源国主流非英语语言，反而普遍降低 LMM 对低收入图像的召回率，且最优提示语言往往并非该国母语。
2. **提出地理与社会经济属性集成 Prompt 框架：** 设计国家后缀（Country Suffix）与收入等级后缀（Income Suffix，含 poor/rich/neutral）策略，无需微调模型即可定向拉升欠代表图像的检索召回。
3. **定位策略生效的最优上下文：** 通过细粒度分析指出，地理后缀对非洲低收入图像提升最大（+15.7%），且模型召回偏好会向与后缀经济层级匹配的图像倾斜，清晰刻画了 LMM 的“标准表征”偏移机制。
4. **多模型泛化验证与开源：** 在 NLLB-CLIP-SigLIP 主模型上完成全面实验，并在 OpenAI CLIP ViT-B/32 与 Sentence Transformers MCLIP 上验证结论一致性，附完整评估代码与流水线。

## 方法详解
- **数据集与分层：** 使用 Dollar Street 数据集（38,479 张 Household item 图像，覆盖 63 个国家、4 大洲）。按购买力平价调整后的月收入四分位数划分为 `poor`（$26.9–95.0）、`low-mid`（$195.4–685.0）、`up-mid`（$694.0–1,998.0）、`rich`（$2,001.0–19,671.0），前两类统称为 lower-income。
- **Prompt 策略设计：**
  - *Default English Topic Prompt：* 基础基线，模板为 “This is a photo of {topic}”。
  - *Native Translated Prompt：* 使用 NLLB-200-distilled-600M 将英文 Prompt 翻译为 40 种非英语主流语言，与图片来源国配对。
  - *Country Suffix Topic Prompt：* 在基础 Prompt 后附加具体国家名，如 “...from Cameroon”，共 63 个模板。
  - *Income Suffix Topic Prompt：* 附加社会经济属性后缀，包含贫困类（poor/impoverished）、富裕类（rich/wealthy）及中性类（a country/a home），并使用同义词增强鲁棒性。
- **评估流程：** 计算图像嵌入与文本嵌入的余弦相似度生成对齐分数；对每个 Topic 选取 Top-N 图像（N 为 Ground Truth 数量）计算 Recall，按国家、大洲、收入层级分组统计。
- **核心模型：** 主模型为 NLLB-CLIP-SigLIP（SigLIP 图像编码器 + NLLB 文本编码器，支持 Flores-200 的 201 种语言），辅以 CLIP ViT-B/32 与 MCLIP 进行交叉验证。
- **统计检验：** 采用 Wilcoxon Signed Rank Test（阈值 p < 0.05）评估干预与基线的差异显著性。

## 实验与结果
- **数据集与基线：** Dollar Street（63 国，272 个客观 Topic），基线为 Default English Prompt。
- **RQ1 翻译 Prompt 结果：** 35/39 个国家中，母语翻译 Prompt 劣于英文基线；仅 Burkina Faso、Nigeria、Pakistan、Tanzania 4 国略优。跨 40 种语言的平均 Recall 在所有收入层级均下降，rich 与 up-mid 下降最大（分别 -7.5%、-7.8%）。
- **RQ2 国家后缀结果：** 38/42 个拥有低收入数据的国家中，Country Suffix 显著提升 Recall。低/中低收入国家的后缀对 poor 图像召回提升最大（+9.7%）；高收入国家后缀反而拉低高收入图像召回。大洲维度上，非洲低收入图像
