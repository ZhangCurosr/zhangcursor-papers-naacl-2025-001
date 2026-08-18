---
title: "AgentMove-A-Large-Language-Model-based-Agentic-Framework-for"
source: https://aclanthology.org/2025.naacl-long.61.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:57:59"
field: "时空数据挖掘与移动预测"
keywords: ["next location prediction", "large language models", "mobility prediction", "agentic framework", "zero-shot learning", "spatial-temporal memory", "geospatial bias"]
innovations: ["首个将LLM-based agentic框架系统应用于零样本下一位置预测的框架", "设计时空记忆模块实现个体多尺度移动模式的结构化存储与读取", "提出世界知识生成器与集体知识提取器，分别建模城市结构与人群共享模式"]
benchmarks: ["Foursquare Check-in (12 cities)", "ISP GPS Trajectory (Shanghai)"]
---

# 论文速读：AgentMove: A Large Language Model based Agentic Framework for Zero-shot Next Location Prediction

## 一句话总结
本文提出了 **AgentMove**，首个将 LLM-based agentic 框架系统应用于零样本下一位置预测工作的框架。通过设计时空记忆、世界知识生成器和集体知识提取器三个模块，充分挖掘 LLM 的序列建模能力与全球地理空间知识，在 12 个城市的数据集上全面优于最强基线（提升 3.33%–8.57%）。

## 研究问题与动机
1. **核心问题**：如何在零样本（zero-shot）设定下实现具有泛化能力的下一位置预测，避免深度学习方法对大规模私有轨迹数据的依赖。
2. **现有 DL 方法不足**：需要大量训练数据；难以直接迁移到新城市（缺乏零样本能力）；序列建模能力有限，且缺乏对日常行为与城市结构的常识理解。
3. **现有 LLM-based 方法不足**：Wang et al. (2023)、Beneduce et al. (2024) 等方法仅将轨迹转为文本并用 LLM 直接生成输出，缺乏系统性的任务分解，未能有效捕捉个体复杂移动模式、未建模城市结构影响、未挖掘人群间的共享模式。
4. **LLM 潜力未被充分利用**：LLM 具备丰富的地理空间知识与序列推理能力，但现有工作未将其与人类移动的领域知识（如返回/探索行为、多尺度城市结构）系统结合。

## 核心贡献（创新点）
1. **首个 LLM-based agentic 框架用于移动预测**：将规划、记忆、世界知识、外部工具与推理模块完整引入移动预测领域，区别于仅用简单 prompt 的 LLM-Mob/LLM-ZS。
2. **时空记忆模块（Spatial-Temporal Memory）**：设计了短期记忆、长期记忆与用户画像三个子单元，显式存储与读取结构化轨迹信息，使 LLM 具备"经验学习"能力，区别于无记忆直接推理的方法。
3. **世界知识生成器（World Knowledge Generator）**：通过文本地址对齐 LLM 内部地理知识与轨迹空间，利用多尺度（区→街区→道路→POI）生成探索候选，区别于仅用坐标或离散 ID 表示位置的做法。
4. **集体知识提取器（Collective Knowledge Extractor）**：基于 NetworkX 构建全局位置转移图，利用 LLM 调用图工具挖掘相似用户的共享转移模式，填补了个体建模与群体模式间的空白。
5. **系统性消融与多城市验证**：在 12 个城市的双源数据（Foursquare + ISP）上验证，证明模块组合的有效性与跨城市低地理偏差的鲁棒性。

## 方法详解
**整体架构**：AgentMove 包含五个核心组件——任务分解模块、时空记忆模块、世界知识生成器、集体知识提取器、最终推理模块。

**任务分解**：将下一位置预测分解为三个子任务：
- 个体移动模式挖掘（personal mobility pattern mining）
- 共享移动模式发现（collective mobility pattern discovery）
- 城市结构建模（urban structure modelling）

**时空记忆模块**（Section 3.2）：
- **Memory Organization**：三类单元——用户画像（User Profile，动态总结）、长期记忆（Long-term Memory，捕获规律趋势）、短期记忆（Short-term Memory，捕获最近行为）。所有记忆以 key-value 形式存储在中央记忆池。
- **Memory Writing**：从历史轨迹 $\mathcal{H}_u$ 提取长期信息（地点-类别映射、top-k 活跃时间/地点、访问频率、转移矩阵）；从上下文停留 $\mathcal{C}_u$ 提取短期信息（最近访问时序、短期频率、最近访问详情）。
- **Memory Reading**：分别生成用户画像 prompt、长期记忆 prompt、短期记忆 prompt，整合为空间-时间摘要注入 LLM 输入。

**世界知识生成器**（Section 3.3）：
- **地址对齐**：利用 OpenStreetMap 逆向地理编码获取原始地址，再用 LLM 提取结构化地址信息（行政区、街区、道路、POI），解决 LLM 不擅长理解精确坐标的问题。
- **多尺度生成**：分四层级（district → block → street → POI name）逐步生成潜在探索候选，降低幻觉并提高可用性。

**集体知识提取器**（Section 3.4）：
- 利用 NetworkX 聚合多用户轨迹构建全局无向加权位置转移图（节点含地址、功能属性，边为 1-hop 转移频率）。
- LLM 调用 NetworkX 工具查询当前位置的 k-hop 邻居，或在多个最近位置邻居中筛选最相关候选，捕获人群共享模式。

**最终推理与预测**（Section 3.5）：汇总三类模块的输出，LLM 进行链式推理，按格式要求输出 JSON（包含 top-5 预测位置 ID 及理由）。

## 实验与结果
**数据集**：
- **Foursquare**：全球 415 城市签到数据（2012.4–2013.9），选取 Tokyo、Sao Paulo、Moscow 等 12 个城市，按 7:1:2 划分。
- **ISP GPS**：上海移动网络轨迹（2016.4，325,215 条记录），7:1:2 划分。该数据在 LLM 训练期之后公开，避免数据泄露。

**评估基线**：FPMC、RNN、DeepMove、LSTPM、GETNext、STHGCN（深度学习方法）；LLM-Mob、LLM-ZS、LLM-Move（LLM 方法）。指标：Acc@1、Acc@5、NDCG@5。

**主要结果**：
- AgentMove（GPT-4o-mini 为基础）在 4 城市共 12 个指标中 **8 个达到最优**。
- 相比最强深度学习方法（GETNext/STHGCN），提升幅度为 **-9.76% ~ +40.63%**，大部分指标正提升。
- 相比最强 LLM 基线（LLM-Move），提升幅度为 **-11.11% ~ +8.57%**，ISP@Shanghai 上 Acc@1 提升 **8.57%**、Acc@5 提升 **40.63%**、NDCG@5 提升 **31.08%**。
- 消融实验（Table 2）：完整 AgentMove 相比 base prompt 在 SaoPaulo 上 Acc@1 提升 **45.45%**、NDCG@5 提升 **17.99%**，在上海上 Acc@1 提升 **11.76%**。
- WKG 模块使模型更倾向于探索新位置（Table 3）：有 WKG 时位置返回率从 87.8% 降至 73.2%（上海），表明探索行为被有效激发。
- 跨 12 城分析（Figure 4）：AgentMove 在多数城市表现最优，且 Acc@5 的方差最小，**地理偏差最低**。
- LLM 尺寸影响（Figure 5）：Llama3.1-405B 显著优于 Llama3-8B，但并非越大越好（Tokyo 上 405B 略逊于 70B）。

## 相关工作脉络
1. **FPMC / DeepMove / GETNext / STHGCN**：传统深度学习轨迹预测方法，依赖大规模训练数据与区域特定编码，无法零样本迁移；AgentMove 以 LLM 为核心实现跨城市泛化。
2. **LLM-Mob (Wang et al., 2023)**：首次将 GPT-3.5 用于位置预测，仅做简单 prompt 转换，缺乏记忆与知识模块；AgentMove 通过系统化 agentic 设计补足这一缺陷。
3. **LLM-ZS (Beneduce et al., 2024)**：测试多种 LLM 在零样本移动预测的表现，提出基础 prompt 框架；AgentMove 在此基础上增加了三大专门模块，性能显著提升。
4. **LLM-Move (Feng et al., 2024c)**：使用 RAG 检索附近 POI 辅助 LLM 预测；AgentMove 进一步引入集体知识图与多尺度城市结构建模，覆盖更广的推理视角。
5. **GeoLLM (Manvi et al., 2023)**：证明 LLM 蕴含地理知识但需地址文本而非坐标激活；本文沿用此洞见，用 OSM 地址对齐作为世界知识生成器的输入基础。
6. **TrajAgent (Du et al., 2024)**：统一轨迹建模的 agent 框架；本文专注于下一位置预测这一具体任务，在模块设计（如 WKG 的多尺度探索）上更具针对性。

## 局限性与未来方向
1. **鲁棒性与幻觉问题**：LLM 输出不可完全控制，简单解析器偶有失败；可能生成现实中不存在的虚假地址，实际部署时需设计有效性验证机制。
2. **高 API 调用成本**：受限于费用，实验仅覆盖 12 个城市且测试集较小（每城 200 样本）；大规模部署需要更高效的 LLM（小模型、剪枝、蒸馏等）。
3. **地理偏差无法完全消除**：尽管 AgentMove 通过多模块设计减轻了偏差，但 LLM 训练数据的固有偏差仍会导致部分城市（如开普敦、内罗毕）表现偏低；未来可通过融合更多外部知识进一步缓解。
4. **扩展方向**：作者计划探索更有效的 LLM 世界知识提取方式，并将框架扩展至轨迹分类与轨迹生成等其他挖掘任务。

## 研究启发与可借鉴点
1. **任务分解 + 模块化 agent 设计**：将复杂预测任务分解为个体模式、共享模式、环境结构三个子任务，分别用专门模块处理再聚合推理，这一思路可迁移至其他时空预测任务（如客流预测、交通流预测）。
2. **地址文本对齐激活 LLM 地理知识**：用 OSM 地址替代坐标作为 LLM 输入，有效激活预训练中的地理常识，是连接 LLM 与空间数据的通用技巧，可在城市计算多项任务中复用。
3. **多尺度生成降低幻觉**：从区到 POI 逐级生成候选的策略，既利用 LLM 的全局知识又约束搜索空间，可借鉴于任何需要 LLM 生成结构化空间信息的场景。
4. **集体知识图的 LLM 推理**：用 NetworkX 构建全局转移图并由 LLM 调用工具进行图推理，实现了符号结构与神经推理的结合，为"LLM + 图"范式提供了移动预测领域的完整范例。
5. **跨城市地理偏差评估范式**：在多城市数据上分析性能方差作为偏差度量，并对比不同方法的分布稳定性，可作为 LLM 空间应用评测的标准实践。

## 关键术语表
**Zero-shot Next Location Prediction**：无需目标城市训练数据，直接利用 LLM 的预训练知识预测用户下一个访问位置的任务设定。
**Spatial-Temporal Memory**：AgentMove 中模拟人类记忆的模块，包含短期记忆（近期行为）、长期记忆（惯用模式）和用户画像三部分，以 key-value 形式存储与检索。
**World Knowledge Generator**：从 LLM 内部地理知识中提取多尺度城市结构信息（区→街区→道路→POI），用于建模人类移动中的"探索"行为。
**Collective Knowledge Extractor**：基于 NetworkX 构建全局位置转移图，通过 LLM 图推理挖掘人群中共享的移动模式。
**Location Return Rate**：衡量模型倾向于返回最近访问过位置的比例，越低表示探索行为越强，用于评估 WKG 模块的有效性。
**Geographical Bias**：LLM 因训练数据分布不均导致对某些地区（如欧美大城市）预测更准、对其他地区（如非洲城市）预测较差的偏差现象。

## 可复现要素
- **数据集**：Foursquare Check-in（公开，https://www.microsoft.com/en-us/research/project/location-data-service/）；ISP GPS 轨迹（公开，https://github.com/jiexhu/DPLink，2024年6月开源）。
- **代码**：已开源，https://github.com/tsinghua-fib-lab/AgentMove
- **关键超参**：LLM 推理 temperature=0（确定性输出），最大输入 token=2000，最大输出 token=1000；Foursquare 会话窗口 72h，最短 session≥4 stays；ISP 夜间（20:00–08:00）停留被过滤，时间窗口 2h 内相同地点合并。
- **基线实现**：FPMC/RNN/DeepMove/LSTPM 使用 LibCity；GETNext/STHGCN 使用官方代码；LLM 通过 OpenAI API（GPT-4o-mini）、DeepInfra、SiliconFlow 访问。
