---
title: "Stealthy-Jailbreak-Attacks-on-Large-Language-Models-via-Beni"
source: https://aclanthology.org/2025.naacl-long.88.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:57:03"
---

# 论文速读：Stealthy-Jailbreak-Attacks-on-Large-Language-Models-via-Beni

## 一句话总结
本文提出 ShadowBreak，一种通过良性数据蒸馏构建本地镜像模型来提升白盒对抗提示向黑盒目标模型迁移能力的隐身越狱方法。该方法在准备阶段完全不提交恶意查询，在 AdvBench 上以平均仅 1.5~3.1 次可检测 API 调用实现了 80%~92% 的攻击成功率，显著优于传统黑盒迭代攻击。

## 研究问题与动机
- **黑盒迭代攻击易被检测拦截**：PAIR、TAP 等主流方法需反复向目标模型提交含恶意指令的查询以获取反馈，极易触发在线内容审核、频次限制或行为异常报警，隐身性差。
- **传统迁移攻击成功率偏低**：直接在开源模型（如 Llama 2）上搜索的对抗后缀迁移至商业黑盒模型时成功率往往大幅下降，且新模型版本的安全对齐不断削弱迁移性（Meade et al., 2024）。
- **缺乏系统化的隐身评估体系**：现有工作普遍以 ASR 为核心指标，未对攻击准备阶段与攻击阶段的查询内容风险进行量化建模，难以客观比较方法的实际可探测性。
- **安全对齐数据对迁移的影响未被充分探索**：既往研究多忽略白盒与黑盒模型在安全域的行为对齐价值，或为提升迁移性而使用有害内容，牺牲了攻击隐蔽性。

## 核心贡献（创新点）
- **提出 ShadowBreak 良性数据镜像框架**：通过仅含无害指令的蒸馏数据在本地对齐目标模型行为，将对抗提示搜索完全移至本地进行，本质区别在于用“良性代理对齐”替代“恶意查询试探”。
- **形式化定义黑盒越狱隐身性量化指标**：引入总查询数 $Q$ 与可检测越狱查询数 $Q^!$，并将评估拆分为准备阶段与攻击阶段，填补了现有评测体系缺乏风险度量维度的空白。
- **揭示对齐数据构成对迁移效果的系统性影响**：发现纯良性数据覆盖面最广，混合良性+安全数据在多数类别上表现最佳，而纯安全数据会引发对齐偏差导致目标模型 ASR 全面下滑。
- **实证暴露当前商业 LLM 防御机制的脆弱性**：在 GPT-3.5 Turbo 上以 1.5 次平均可检测查询实现 80% ASR（AutoDAN），以 3.1 次实现 92% ASR（GCG），均大幅优于 PAIR（27.4 次可检测查询达 84% ASR）。

## 方法详解
- **攻击目标与评估指标**：给定目标黑盒模型 $\mathcal{M}_T$、有害指令集 $I$ 与判别器 $\mathcal{I}$，寻找 $I_i'$ 使 $\mathcal{I}(\mathcal{M}_T(I_i'))=1$。采用精确匹配判别器 $\mathcal{I}_M$ 与语义分类判别器 $\mathcal{I}_C$（Llama Guard 3）分别计算 $\mathrm{ASR_M}$ 与 $\mathrm{ASR_C}$；隐身性通过 $Q = |\text{Queries}|/|I|$ 与 $Q^! = |\{\text{Query}_i \mid \mathcal{I}(\text{Query}_i)=1\}|/|I|$ 量化，使用 Prompt-Guard-86M 作为越狱检测器。
- **镜像模型构建（Mirror Model Construction）**：从通用指令集 $\mathcal{D}$ 中筛选经 $\mathcal{I}_C$ 判定为良性的指令 $I_i$，向目标模型发起查询构建 $\mathcal{D}_{\mathcal{I}} = \{(I_i, \mathcal{M}_T(I_i)) \mid \mathcal{I}_C(I_i)=0\}$。以 Llama 3 8B Instruct 为底座，采用 SFT 或 DPO 损失最小化镜像模型 $\mathcal{M}_S$ 与目标输出的差异，完成行为与安全策略对齐。
- **对齐迁移攻击（Aligned Transfer Attack）**：在本地镜像模型 $\math
