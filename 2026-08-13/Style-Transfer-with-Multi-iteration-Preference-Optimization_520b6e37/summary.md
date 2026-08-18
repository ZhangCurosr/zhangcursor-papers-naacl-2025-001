---
title: "Style-Transfer-with-Multi-iteration-Preference-Optimization"
source: https://aclanthology.org/2025.naacl-long.135.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:56:47"
field: "文本风格迁移"
keywords: ["文本风格迁移", "偏好优化", "多迭代训练", "期望-恐惧采样", "动态加权奖励"]
innovations: ["多迭代对比偏好优化框架（多轮CPO+参考模型更新）", "期望-恐惧采样构造偏好对（同时考虑模型概率与任务奖励两极）", "动态加权奖励聚合防止多目标退化"]
benchmarks: ["CDS（Corpus of Diverse Styles）", "GYAFC（Grammarly Yahoo Answers Formality Corpus）"]
---

# 论文速读：Style-Transfer-with-Multi-iteration-Preference-Optimization

## 一句话总结
本文提出 STAMP（Style TrAnsfer with Multi-iteration Preference optimization），一个基于多迭代对比偏好优化的文本风格迁移训练框架，借鉴统计机器翻译"调优"（tuning）思想，引入期望-恐惧采样与动态加权奖励聚合，在 CDS 和 GYAFC 两个主流数据集上均超越 SOTA。

---

## 研究问题与动机

1. **风格迁移缺乏平行数据**：文本风格迁移（Style Transfer）目标是将输入文本改写为目标风格同时保留原意，但高质量平行风格数据稀缺，现有方法依赖伪平行数据生成。
2. **多目标优化易失衡**：风格迁移需同时优化三个独立目标——目标风格强度（TSS）、意义相似度（MS）和流畅度（F），简单乘积聚合易导致"逆向单目标分数"（reversed single-objective scores），即某些目标严重退化。
3. **离线偏好优化（PO）不足**：DPO/CPO 等单次离线 PO 方法无法持续探索改进，模型容易过拟合初始偏好对。
4. **启发于 MT 调优传统**：Och (2003) 的 MERT、Chiang (2012) 的在线调优等 MT 方法已证明多迭代生成-优化范式有效，本文将其适配到风格迁移场景。

---

## 核心贡献（创新点）

1. **多迭代对比偏好优化框架**：在 SFT 模型基础上进行多轮 CPO 训练，每轮更新参考模型并重构偏好对；与 ASTRAPOP/STEER 的单次 PO 本质不同，持续探索模型能力边界。
2. **期望-恐惧采样（Hope-and-Fear Sampling）构建偏好对**：从 k_PO 个候选中按 $\mathcal{M}^{\tau_\mathcal{M}} + \mathcal{R}$（希望）选获胜样本、按 $\mathcal{M}^{\tau_\mathcal{M}} - \mathcal{R}$（恐惧）选失败样本；避免危险低质量生成，鼓励可达的高分生成，区别于随机采样或次高分采样。
3. **端到端伪平行数据生成方法**：将两阶段 paraphrase→inverse-paraphrase 改为一步到位生成 $D_{trf}$，并在选择时引入温度参数 $\tau_{ms}>1$ 强调 MS，保证 SFT 阶段的梯度连通性；与 STRAP 的两步断裂相比，为 PO 提供更强的起始模型。
4. **动态加权奖励聚合（Dynamic Weighted Reward Aggregation）**：根据每轮偏好对中各目标的逆向分数数量自动调节 $\alpha, \beta, \gamma$，防止任一目标退化；简单乘积（$\alpha=\beta=\gamma=1$）在 CDS 上导致 MS 严重退化（图 2），加权版本则三目标收敛均衡。

---

## 方法详解

### 整体框架：两阶段 STAMP

```
SFT 阶段：非平行数据 → 端到端伪平行数据生成 → SFT 训练 f_SFT
PO 阶段：多迭代 CPO，每轮生成偏好对 → 更新参考模型 → f_PO（最终模型）
```

### SFT 阶段：端到端伪平行数据

1. 用 ParaNMT 训练的 paraphraser $f_{para}$ 对每个文本 $x_i$ 生成 $k_{para}$ 个释义，选 MS 最高的 $p_i$。
2. 对每个目标风格 $s$，用逆 paraphraser $f_{inv}^s$ 在 $(p_i \to x_i)$ 对上训练。
3. 关键改进：生成 $k_{sft}$ 个转移结果 $t_i = f_{inv}^s(f_{para}(x_i))$，按 $\text{F} \cdot \text{MS}^{\tau_{ms}} \cdot \text{TSS}$ 选择最优（$\tau_{ms}>1$ 强调意义保留），构建端到端数据 $D_{trf}$。
4. 统一模型（unified model）直接以 control code 控制目标风格：$p(t|x,s) = \prod p(t[i]|x, t[<i], s)$。

### PO 阶段：多迭代 CPO

**迭代逻辑**：设 $f_{ref}^1 = f_{SFT}$，第 $i$ 轮以 $f_{ref}^i$ 生成偏好对 $D_{PO}^i$，用 CPO 训练得 $f_{PO}^i$，令 $f_{ref}^{i+1} = f_{PO}^i$，直到验证 TSS 开始下降。

**期望-恐惧采样**：对每个文本 $x_i$（原始风格 $s_i \neq s$），用 $f_{ref}$ 生成 $k_{PO}$ 个改写，选：
- 获胜样本 $t_i^w$：最高 $\mathcal{M}^{\tau_\mathcal{M}} + \mathcal{R}$
- 失败样本 $t_i^l$：最高 $\mathcal{M}^{\tau_\mathcal{M}} - \mathcal{R}$

其中 $\mathcal{M}$ 为模型平均 token 概率，$\mathcal{R}$ 为奖励函数。

**奖励函数与动态加权**：
$$\mathcal{R} = \text{TSS}^\alpha \cdot \text{MS}^\beta \cdot \text{F}^\gamma$$

$\alpha, \beta, \gamma$ 每轮动态调整：依次找到最小正整数 $\alpha$（使 $r_{TSS} < r_{MS}, r_{TSS} < r_F$），最大正整数 $\beta$（使 $r_{MS} > r_{TSS}$），最大正整数 $\gamma$（使 $r_F > r_{TSS}, r_F > r_{MS}$），上限 $\tau_{max}$。

### 实验结果

| 模型 | CDS Agg. | GYAFC Agg. |
|------|----------|------------|
| STEER（最佳基线） | 0.395 | 0.686 |
| **STAMP** | **0.474** (+20%) | **0.828** (+21%) |

CDS 上 TSS=0.746、MS=0.801；GYAFC 上 TSS=0.958、MS=0.921，均大幅超越 SOTA。STAMP 在 CDS 上只需 22.2h×4A40s 即可超越 STEER 最终性能（52h×4A40s），训练效率更高。

---

## 相关工作脉络

1. **STRAP（Krishna et al., 2020）**：最早将风格迁移建模为释义任务的 SFT 方法，两步断裂设计，无 PO 优化；STAMP 在其基础上引入端到端 SFT + 多迭代 PO。
2. **STEER（Hallinan et al., 2023）**：结合专家/反专家引导的伪平行生成与 RL（Quark）优化；STAMP 用 CPO 替代 RL，用期望-恐惧采样替代随机偏好对。
3. **ASTRAPOP（Liu et al., 2024）**：将 PO 应用于作者风格迁移，使用 DPO/CPO/PPO 三种优化算法之一；STAMP 明确采用多迭代 CPO + 动态加权，针对通用文本风格迁移优化。
4. **DPO（Rafailov et al., 2023）**：单次离线对比偏好优化；STAMP 继承 CPO 并扩展为多迭代，利用 CPO 的负对数似然项提升多轮稳定性（Pang et al., 2024）。
5. **MT 调优传统**：Och (2003) MERT、Chiang (2008) 在线调优、Chiang (2012) 期望-恐惧采样；本文首次将其系统适配到风格迁移的偏好优化场景。

---

## 局限性与未来方向

1. **重复与幻觉**：部分生成文本存在重复和目标风格过拟合（peakiness），原因可能是 PO 训练放大高频 token 概率；需发展对 peakiness 鲁棒的 PO 算法或更完善的奖励模型。
2. **动态加权在简单任务上无效**：在 GYAFC（相对简单的正式度迁移）上，未加权版本的逆向分数比例已很低，动态加权帮助有限；需增加控制机制判断何时启用加权。
3. **伦理风险**：框架可被用于生成冒犯性或未经授权的风格模仿（如作者匿名化），但作者声明仅限研究用途。

---

## 研究启发与可借鉴点

1. **多迭代 PO 的通用价值**：将 CPO 从离线扩展到多迭代（参考模型逐轮更新），已在推理任务（Self-play Fine-tuning, Chen et al., 2024）和风格迁移中验证有效，可迁移到其他 Seq2Seq 任务。
2. **期望-恐惧采样的配对构造思路**：同时考虑模型概率（可达性）和任务奖励（质量）两极，避免随机采样导致的低质量偏好对；可直接借鉴到 RLHF/PO 的 online 阶段数据构建。
3. **动态多目标加权机制**：根据逆向分数比例自适应调整权重，比固定超参乘积更稳定；可推广到任何多目标优化场景（如摘要+连贯性+事实性）。
4. **端到端伪平行生成 vs 两步断裂**：保留梯度连通性对后续 PO 阶段至关重要，设计数据生成 pipeline 时应优先考虑下游训练需求的连通性。

---

## 关键术语表

- **Preference Optimization（偏好优化，PO）**：无需显式奖励模型的策略优化方法，通过对比偏好对直接优化模型；包含 DPO、CPO 等变体。
- **Hope-and-Fear Sampling（期望-恐惧采样）**：从候选集中选最高 $\mathcal{M}+\mathcal{R}$ 为"希望"样本、最高 $\mathcal{M}-\mathcal{R}$ 为"恐惧"样本，构造强对比偏好对（Chiang, 2012）。
- **TSS / MS / F**：目标风格强度（Target Style Strength）、意义相似度（Meaning Similarity）、流畅度（Fluency），风格迁移三目标评估指标。
- **Reversed Single-Objective Score（逆向单目标分数）**：聚合奖励最高的样本在某单一目标上反而较低的现象，由多目标独立性导致。
- **CPO（Contrastive Preference Optimization）**：CPO 相比 DPO 额外加入负对数似然项，在多迭代场景中更稳定有效（Xu et al., 2024a）。
- **Unified Model（统一模型）**：单模型通过 control code 控制所有目标风格，而非为每个风格单独训练模型（Hallinan et al., 2023）。
- **Agg.（Aggregate Score）**：$\text{Agg.} = \text{TSS} \cdot \text{MS} \cdot \text{F}$，三目标的几何平均聚合分数。

---

## 可复现要素

- **数据集**：CDS（Corpus of Diverse Styles，MIT 许可，开源）、GYAFC（Grammarly Yahoo Answers Formality Corpus，自定义研究许可，需申请）、ParaNMT-75k（Paraphrase 训练数据）。
- **代码**：作者以 MIT 许可证开源 STAMP 代码（论文 Appendix C.2）。
- **基础模型**：LLaMA-2-7B（Meta MIT 许可）、RoBERTa-large、SBERT all-mpnet-base-v2。
- **关键超参**：$k_{para}=20, k_{sft}=90, k_{PO}=10, N_{iter}=10, \tau_{ms}=8, \tau_{max}=6$；SFT learning rate=5e-5，CPO learning rate=2e-6，$\beta=0.1$；LoRA rank=16, $\alpha=32$（详见附录 Table 23–32）。

---
