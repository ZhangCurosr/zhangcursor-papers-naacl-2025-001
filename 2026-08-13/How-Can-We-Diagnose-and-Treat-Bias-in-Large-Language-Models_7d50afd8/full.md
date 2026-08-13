# How Can We Diagnose and Treat Bias in Large Language Models for Clinical Decision-Making?

Kenza Benkirane<sup>1</sup>, Jackie Kay<sup>1,2</sup>, Maria Perez-Ortiz<sup>1</sup> <sup>1</sup>AI Centre, Dept. of Computer Science, University College London (UCL), UK <sup>2</sup>Google DeepMind, UK Correspondence: kenza.benkirane.23@ucl.ac.uk

## Abstract

Recent advancements in Large Language Models (LLMs) have positioned them as powerful tools for clinical decision-making, with rapidly expanding applications in healthcare. However, concerns about bias remain a significant challenge in the clinical implementation of LLMs, particularly regarding gender and ethnicity. This research investigates the evaluation and mitigation of bias in LLMs applied to complex clinical cases, focusing on gender and ethnicity biases. We introduce a novel Counterfactual Patient Variations (CPV) dataset derived from the JAMA Clinical Challenge <sup>1</sup>. Using this dataset, we built a framework for bias evaluation, employing both Multiple Choice Questions (MCQs) and corresponding explanations. We explore prompting with eight LLMs and fine-tuning as debiasing methods. Our findings reveal that addressing social biases in LLMs requires a multidimensional approach as mitigating gender bias can occur while introducing ethnicity biases, and that gender bias in LLM embeddings varies significantly across medical specialities. We demonstrate that evaluating both MCQ response and explanation processes is crucial, as correct responses can be based on biased reasoning. We provide a framework for evaluating LLM bias in real-world clinical cases, offer insights into the complex nature of bias in these models, and present strategies for bias mitigation.

## 1 Introduction

Despite LLMs offering promising potential for text generation across various domains, recent studies have shown that these models are prone to exhibiting social biases inherited from their training data (Sheng et al., 2021; Navigli et al., 2023). Bias in this context refers to a model’s systematic tendency to unfairly discriminate against certain individuals or groups in favour of others (Friedman and Nissenbaum, 1996). This can manifest as lower prediction accuracy for certain demographic groups or as disparities in the quality of generated content across different populations (Baker and Hawn, 2022).

![](images/10c88bd6888f21b3d30b95a3c21836b7d1e0c81b47dd83a86eb184a8d8acd5d5.jpg)  
Figure 1: Illustration of our experimental setup for evaluating bias in LLMs for clinical cases using Counterfactual Patient Variations (CPVs). The example shows how changing demographic attributes (gender and ethnicity) in otherwise identical clinical cases can lead to different model outputs.

In healthcare, such biases may exacerbate health disparities and unfairly impact certain patient groups, posing significant risks where discriminatory outputs could lead to disparities in patient care and health outcomes (He et al., 2023; Lee et al., 2023; Singh et al., 2023; Harrer, 2023; Singh et al., 2023). For example, a recent study from (Zack et al., 2024) revealed that GPT-4 exhibited a 9% lower likelihood of recommending advanced imaging for Black patients and an 8% lower likelihood of rating stress testing as highly important for female patients compared to male patients.

Current approaches to evaluating LLMs in medical contexts primarily rely on Multiple Choice Questions (MCQ) from standardised exams like the United States Medical Licensing Examination (USMLE) (Nori et al., 2023). While some models have achieved scores comparable to or surpassing those of human medical professionals (Cascella et al., 2024), excelling at multiple-choice questions does not necessarily equate to superior reasoning skills needed for real-world clinical practice, as highlighted by (Saab et al., 2024; Homolak, 2023; Harris, 2023; Kanjee et al., 2023). At the same time, researchers have called for more comprehensive and clinically relevant benchmarks (Longhurst et al., 2024; Nickel et al., 2024).

In response to these concerns and to address the need for more clinically relevant evaluation methods, (Chen et al., 2024) introduced the JAMA dataset, comprising complex clinical cases that test decision-making skills in realistic clinical scenarios. Our work builds on this challenge, using the JAMA Clinical Challenge dataset, which provides real-world, complex medical cases along with MCQs and explanations (XPLs), allowing us to evaluate the decision-making rationale behind clinical-decision making with LLMs.

We implement Counterfactual Patient Variations (CPVs) to evaluate bias in LLMs across clinical scenarios (see Figure 1). Our research explores prompt engineering and fine-tuning for bias mitigation, as well as a real-world evaluation without multiplechoice labels given. Our framework incorporates a wide array of metrics for bias quantification, including accuracy comparisons, statistical measures, feature importance analysis, and embedding-based assessments. We address three main research questions: RQ1: Extent of LLM bias in CPV across gender and ethnicity in complex clinical scenarios. RQ2: Effectiveness of prompt and fine-tuning strategies in mitigating bias. RQ3: Fairness differences between structured MCQ and open-ended clinical explanations.

We find that LLMs exhibit pervasive gender and ethnicity biases in outcomes and reasoning, with discrepancies between MCQ performance and XPL quality revealing persistent biases despite apparent balanced accuracy. Fine-tuning can mitigate some biases but may introduce new ones, particularly across ethnic categories. Prompt engineering alone is insufficient for comprehensive debiasing, with effectiveness varying across models and demographics. Gender bias in LLM embeddings varies considerably across medical specialities, necessitating domain-specific debiasing strategies.

Our main contributions are:

a) A novel CPV framework enabling systematic

evaluation of bias in clinical cases.

b) A comprehensive bias evaluation in clinical LLMs, incorporating both MCQ performance and explanation quality metrics.

c) Insights into the complex nature of bias in clinical LLMs explanations from their embeddings, including the variability across medical specialities and the discrepancy between MCQ performance and explanation biases.

d) Evaluation of various prompting and finetuning strategies for bias mitigation, highlighting their strengths and limitations.

## 2 Dataset creation: JAMA Clinical Challenges with Counterfactual Patient Variations

Dataset scope and sources This study uses the JAMA Clinical Challenge, a collection of clinical cases extracted from the Journal of the American Medical Association (JAMA) Clinical Challenge archive, focusing on complex cases: cases that pose significant diagnostic challenges, encouraging readers to engage in critical thinking and apply their clinical knowledge. Each case comprises a detailed patient description (250 words), a specific clinical question, four answer options, the correct answer index, a discussion (500-600 words) elaborating on the preferred option, and a medical speciality classification. Appendix A.1 provides a representative sample, as well as a description of JAMA specialities. This dataset takes its value not only from the double evaluation of multiple-choice questions (MCQ) and the associated explanation, but also from the real-world unstructured clinical vignettes covering a wide range of medical topics, intentionally challenging and often requiring careful analysis of clinical findings. We extracted data in two phases: an initial extraction following (Chen et al., 2024)’s instructions, resulting in the JAMA\_Chen2024 dataset (1,522 cases), and a subsequent extraction on 10 August 2024, creating the JAMA\_CPV dataset (1,734 cases, July 2013 - August 2024), enabling access to 212 additional cases. To the best of our knowledge, this work represents the first analysis of the JAMA Clinical Challenge dataset for bias evaluation in LLMs and is the first to use the 212 additional cases. While (Chen et al., 2024) introduced the initial dataset, our study extends its application significantly in the context of bias evaluation and mitigation.

Clinical case feature extraction To facilitate gender swapping, identify questions asked, and gain insights into the patient population, we conducted extensive preprocessing of the dataset. This process began with a thorough human analysis of numerous clinical cases, which prompted the development of a rule-based system for feature extraction and case exclusion. This preliminary analysis helped identify the gender of cases in the dataset, which were Male, Female and Neutral. Preprocessing steps included extracting patient demographics (age, gender, ethnicity) using regex-based pattern matching; detecting gender-specific medical conditions (e.g., pregnancies, women’s health issues) for appropriate case exclusion; normalising clinical questions into three standardized formats; and implementing answer option randomisation to mitigate potential selection biases (Zheng et al., 2023). The rule-based system was iteratively refined based on human evaluation of its performance on a subset of cases. More details for these processes are available in Appendix A.2.

Creating Counterfactual Patient Variations (CPVs) To create tailored subsets for each experiment, we applied a systematic filtration and variation methodology. Filtration criteria included condition (excluding cases related to pregnancies and women’s health issues), ethnicity (removing cases with explicitly mentioned original ethnicities), medical speciality, and publication year. After filtration, we applied systematic variations, creating male, female, and gender-neutral versions of each case, and introducing diverse ethnic backgrounds (Arab, Asian, Black, Hispanic, White)<sup>2</sup>. This method preserves the initial structure of the text, without using LLMs, and remains bi-dimensional by modifying both the gender and ethnicity of patients simultaneously.

## 3 Methodology

Model selection We selected a diverse range of LLMs for our experiments, including GPT-3.5 (gpt-3.5-turbo-0301), GPT-4o (gpt-4o-2024-05-13),

GPT-4 Turbo (gpt-4-turbo-2024-04-09), Haiku (Claude3 Haiku), Sonnet (Claude 3.5 Sonnet), Gemini (Gemini 3.5 Flash), Llama3 (LLama3- 70B), Llama3.1 (Llama3.1-403B) for inference, as well as GPT-4o mini for fine-tuning.

Inference and prompts We developed multiple prompting strategies to evaluate different approaches to bias mitigation, based on initial work by (Chen et al., 2024) and prompting guidelines from (Liu et al., 2023), (Ganguli et al., 2023), and (Parrish et al., 2021). For the Exploratory CPV experiment, we enhanced the prompt by incorporating Chain-of-Thought (CoT) reasoning (Wei et al., 2022) and follow-up questions about gender and ethnicity relevance. For the prompt bias mitigation evaluation experiment, we implemented three distinct prompts: a baseline question (Q), a debiasing prompt adding Instruction Following (Q+IF), and a combination of debiasing instructions with Chain-of-Thought (CoT) reasoning (Q+IF+CoT), a framework based on (Ganguli et al., 2023). Finally, the ablation study without multiple-choice used a modified version of the prompt mitigation’s baseline prompt adapted not to provide the MCQ options. All the prompts are reported in Appendix G. To ensure consistent and deterministic outputs across all experiments, we set the temperature parameter to 0 for deterministic generation (Wang et al., 2023).

Fine-tuning For the fine-tuning experiment, we employed two task-specific paradigms: MCQ (Multiple Choice Question) and XPL (eXPLanation). For the MCQ task, we fine-tuned models on a dataset with case descriptions and options, outputting only the answer, while for the XPL task, we fine-tuned on a dataset with cases, options, and solutions, outputting only the explanation. We used OpenAI’s fine-tuning platform with GPT-4o mini. The datasets for both tasks were carefully curated to ensure a balanced representation across genders and ethnicities, with the MCQ dataset containing 1,409 training examples and the XPL dataset containing 4,044 training examples. For the MCQ task, we trained for 2 epochs with a batch size of 32 and a learning rate multiplier of 0.8. The XPL task was trained for 3 epochs with a batch size of 2 and a learning rate multiplier of 1.8. These hyperparameters were selected based on multiple iterations and performance on the validation set, balancing between model performance and generalisation.

## Metrics for bias quantification

By combining accuracy comparisons, statistical methods, SHAP analysis, and embedding-based measures, we provide a holistic view of bias manifestation, offering insights into performance disparities, underlying model behaviours, and latent biases in language representations.

Accuracy Comparison We calculated accuracy scores across dimensions like gender, ethnicity, model type, and prompt variations. To quantify performance disparities, we evaluate the Accuracy Delta, defined as $\Delta ( i , j ) = A _ { i } - A _ { j }$ for categories i and j with accuracies $A _ { i }$ and $A _ { j \cdot } \operatorname { A }$ positive value indicates higher accuracy for category i compared to j, providing a quantitative measure of potential bias.

Statistical Methods We employed statistical metrics to quantify bias: i) The Equality of Odds (EO) metric was used to assess whether the model’s performance is consistent across different demographic groups for both positive and negative outcomes. Additionally, we used ii) the SkewSize metric (Albuquerque et al., 2024) to quantify the distribution of bias-related effect sizes across different classes in our prediction task. The Skew-Size metric provides insight into the magnitude and direction of bias that may not be apparent from accuracy measures. We also calculated iii) the Coefficient of Variation (CV) to measure the relative variability of these effect sizes. The CV is defined as the ratio of the standard deviation to the mean.

SHAP Analysis To interpret feature contributions in model predictions, we employed SHAP (SHapley Additive exPlanations) values (Lundberg et al., 2017). Our implementation used the prompt text as input features and the binary MCQ performance (correct or incorrect) as the output prediction, enabling us to identify which aspects of the prompts were most predictive of the model’s success in answering multiple-choice questions.

Embeddings calculation We evaluated the models’ explanations through their sentence embeddings. We used the SBERT (Sentence-BERT, Bidirectional Encoder Representations from Transformers) model (Reimers and Gurevych, 2019), which is built on BERT for Natural Language Inference (NLI) and employs max pooling for discretisation. For our implementation, we used SentenceTransformer <sup>3</sup>, a flexible Python framework that allows easy transitions between language models without extra installations. This choice aligns with (Dolci et al., 2023), though we excluded names from our gender direction definition. We used the all-distilroberta-v1 model<sup>4</sup> instead of the legacy bert-base-nli-max-token. To analyse long text sequences exceeding the 512-token limit, we implemented a token-based sliding window approach (Perea and Harer, 2015) that preserves semantic integrity. Details are in Appendix E.

Gender bias We employed gender bias, adapting and extending the approach from (Bolukbasi et al., 2016; Garg et al., 2018), as proposed by (Dolci et al., 2023). To establish the gender direction, we collected 100 sentence pairs from the POM (Park et al., 2014), MELD (Poria et al., 2019), and SST (Socher et al., 2013) datasets, excluding proper names. Each pair comprises an original sentence and its gender-swapped counterpart. We computed difference vectors between the embeddings of original and gender-swapped sentences, and then performed Principal Component Analysis (PCA) on these vectors. The first principal component, explaining 73% of the variance, represents the primary gender direction ${ \vec { g } } .$ For each case C, we compute the gender bias score as: Gender $\begin{array} { r } { B i a s ( C ) { \bf \bar { \Psi } } = \frac { { \vec { e } } \cdot { \vec { g } } } { | { \vec { g } } | } } \end{array}$ , where ⃗e is the case embedding. This method captures subtle differences between male and female embeddings at the sentence level, providing a nuanced view of gender bias that may not be captured by more general performance metrics.

As a reference, Table1 displays the gender bias of a few example sentences with our model.

<table><tr><td>Object ↓/ Subject → someone father mother</td><td></td><td></td><td></td></tr><tr><td>quarterback</td><td>-0.07</td><td>-0.17</td><td>0.16</td></tr><tr><td>nurse</td><td>0.22</td><td>-0.08</td><td>0.26</td></tr></table>

Table 1: Gender bias values for sentences of the form “[Subject] is a [Object]”  
Blue indicates masculine-leaning bias (negative values), red indicates feminine-leaning bias (positive values).

Bias Score We use the bias score from (Dolci et al., 2023) to estimate gender bias in sentence embeddings. For a case C, we calculate: $\begin{array} { c c c } { { B i a s S c o r e ( C ) } } & { { = } } & { { \sum _ { w \in C } \cos ( \vec { e _ { w } } , \vec { g } ) \times } } \end{array}$

$I _ { w }$ , where $\vec { e _ { w } }$ is the word vector, $\vec { g }$ is the gender direction, and $I _ { w }$ is word importance. We compute the Median BiasScore as MB = $\begin{array} { r } { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \dot { \frac { B i a s S c o r e { _ M } ( C ) _ { i } + B i a s S c o r e { _ F } ( C ) _ { i } } { 2 } } } \end{array}$ , following (Dolci et al., 2023)’s methodology for word importance and gender word list.

As a reference, Table 2 displays the Bias Score of some examples.

<table><tr><td>Object ↓ / Subject →</td><td>they</td><td>he</td><td>she</td></tr><tr><td>sick</td><td>0.00</td><td>-0.14</td><td>0.22</td></tr><tr><td>nurse</td><td>0.73</td><td>-0.18</td><td>0.42</td></tr><tr><td>CEO</td><td>-0.05</td><td>-0.26</td><td>0.44</td></tr></table>

Table 2: Median Bias Scores for sentences of the form “[Subject] is/are [Object]”.  
Blue indicates masculine-leaning bias (negative values), red indicates feminine-leaning bias (positive values).

## 4 Experiments

Our experiments use a system-and-user prompt structure to query LLMs about clinical cases, evaluating their responses for potential biases. Each experiment prompted the models to provide both an MCQ response and an accompanying explanation, allowing us to assess bias in both decisionmaking and explanation, in a predict-then-explain framework (Siegel et al., 2024). Detailed dataset statistics per experiment are available in Appendix A.

We conducted four main experiments to evaluate and mitigate bias:

Exploratory CPVs We aimed to assess the extent of bias in LLMs when presented with CPV across gender and ethnicity: we evaluate how introducing intersectionality through gender and ethnicity CPV may reveal complex bias patterns in LLMs that may not be apparent when examining gender or ethnicity in isolation. The prompt used incorporated Chain-of-Thought reasoning and follow-up questions about gender or ethnicity relevance.

Bias mitigation with prompt engineering We sought to evaluate the effectiveness of targeted debiasing prompting strategies. The prompts used included an open-ended baseline without explicit debiasing instructions, and two debiasing prompts inspired by (Ganguli et al., 2023), including a moral correction-style prompt focusing on fairness (Ouyang et al., 2022).

Bias mitigation with fine-tuning This experiment explored the effectiveness of fine-tuning using CPVs for ethnicity representation in mitigating bias, aiming at compensating for a possible lack of representativity in training sets of our foundation models. We used two task-specific paradigms: MCQ, fine-tuned on case descriptions and options, outputting only the answer; and XPL, fine-tuned on cases, options, and solutions, outputting only the explanation.

Ablation study without multiple options We aimed to assess LLM performance across social attributes in a real-world context, where open questions would be presented without multiple options. The approach used a modified version of the baseline prompt for Bias mitigation with prompt engineering, adapted for scenarios without multiplechoice. Detailed results of this ablation study are available in Appendix C.

## 5 Results

<table><tr><td>Metric</td><td>GPT-3</td><td>GPT-40 GPT-4 Turbo</td></tr><tr><td>Gender CPV</td><td></td><td></td></tr><tr><td>∆(Female, Neutral)</td><td>+1.00% -0.50%</td><td>0.00%</td></tr><tr><td>∆(Male, Neutral)</td><td>0.00% -2.00%</td><td>-0.50%</td></tr><tr><td>Gender-x-Ethnicity CPV</td><td></td><td></td></tr><tr><td>∆(Female, Neutral)</td><td>+0.60%</td><td>-1.26% -1.59% -1.26%</td></tr><tr><td>∆(Male, Neutral)</td><td>+3.77%</td><td>-1.19%</td></tr><tr><td>∆(Asian, No ethnicity)</td><td>-0.46%</td><td>-0.93% -0.46%</td></tr><tr><td>∆(Black, No ethnicity)</td><td></td><td></td></tr><tr><td></td><td>-2.31%</td><td>-1.85%</td></tr><tr><td></td><td>-1.39%</td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td>∆(White, No ethnicity)</td><td>-2.31% +1.85%</td><td>-0.93%</td></tr></table>

Table 3: Exploratory CPVS | Comparative accuracies, across gender and gender-cross-ethnicities CPVs. This table shows that introducing ethnicity as a variable led to changes in gender-related disparities, with varying effects across models. It also reveals the introduction of ethnic biases, with Asian cases consistently showing the best performance.

red indicates lower values, green indicates higher values.

Intersectionality and prioritisation in bias mitigation Table 3 shows the results of the bias evaluation in our two CPV datasets, examining the impact of gender-only and gender-x-ethnicity CPV strategies on MCQ performance and explanation (XPL) quality. The introduction of ethnicity as a variable led to changes in gender-related disparities, with varying effects across models. For GPT-3.5, the gap between female and neutral cases narrowed from 1.00% to 0.60%, while the gap between male and neutral cases increased from 0.00% to 3.77%. Despite the reduction in gender-related disparities,

![](images/6d1e46ab6f26bf7fbe530e8375878e208e1642692703131850393e79e05f36a4.jpg)  
Figure 2: Exploratory CPVs | Top 5 features and their importance with regards to MCQ performance. This figure illustrates that ethnicity features became highly influential when introduced, often surpassing gender features in importance. It demonstrates how the introduction of ethnicity shifted rather than eliminated bias patterns.

Ethnicity features take prominence in the GxE CPV experiment over the Gender features. Grey indicates Other features.

gender terms remained among the top influential features for all models: "man" and "woman" appeared in the top 5 SHAP features for GPT-3 and GPT-4o in both experiments, as displayed in Figure 2. We also observed the introduction of ethnicity biases: GPT-3.5 and GPT-4 Turbo consistently underperformed on ethnicity-varied cases compared to the no ethnicity case, with Asian cases systematically showing the best performance (-0.46% for both models). The SHAP feature analysis revealed that ethnicity terms became highly influential when introduced. For instance, "white" became the most important feature for GPT-4o (0.74), while "black" became the most negatively influential feature for GPT-4 Turbo (-0.60). The introduction of ethnicity appeared to shift rather than eliminate bias patterns, as reflected in the changing importance and direction of influence for demographic terms. For example, "white" shifted from contributing to incorrect predictions (-0.45) to strongly favouring correct predictions (0.74) for GPT-4o. These findings underscore the need for comprehensive debiasing strategies that address both gender and ethnic dimensions in outcomes and reasoning processes.

Effectiveness of Fine-Tuning in mitigating with CPV for bias mitigation Our fine-tuning experiments showed interesting results across MCQ (Table 4) and XPL (Figure 3) GPT-4o mini models. For the MCQ model, the fine-tuning process demonstrated success in mitigating gender bias, reducing performance disparities between male and female categories. The Gender SkewSize metric decreased from 0.25 to 0.02, while the Equality of Odds (EO) decreased from 0.02 to 0.01, indicating a more balanced performance across gender categories relative to the neutral case.

<table><tr><td>Metric</td><td>Baseline Fine-tuned</td><td></td></tr><tr><td>∆(Female, Neutral) ∆(Male, Neutral)</td><td>+2.49% +0.93%</td><td>-2.49% -3.49%</td></tr><tr><td></td><td></td><td></td></tr><tr><td>Gender SkewSize Gender EO</td><td>-0.25 0.02</td><td>-0.02 0.01</td></tr><tr><td>∆(Arab, No ethnicity)</td><td>-0.98%</td><td>+5.48%</td></tr><tr><td>∆(Asian, No ethnicity)</td><td>-3.47%</td><td>+2.51%</td></tr><tr><td>∆(Black, No ethnicity)</td><td>+2.48%</td><td>-2.44%</td></tr><tr><td>∆(Hispanic, No ethnicity)</td><td>-1.49%</td><td>+2.51%</td></tr><tr><td>∆(White, No ethnicity)</td><td>-3.47%</td><td>+1.52%</td></tr><tr><td>Ethnicity SkewSize</td><td>-0.49</td><td>0.60</td></tr><tr><td>Ethnicity EO</td><td>0.06</td><td>0.08</td></tr></table>

Table 4: Bias mitigation withfine-tuning | Model performance differences across models This table shows that fine-tuning successfully mitigated gender bias in MCQ performance but led to more complex changes in ethnicity-related performance, with improvements for some ethnicities and declines for others. Values show percentage differences in accuracy compared to the neutral or no-ethnicity baseline. Positive values indicate higher accuracy and negative values indicate lower accuracy. Green highlights improvements, red highlights declines.

However, the ethnicity bias presented a more nuanced picture. The SkewSize increased from 0.49 to 0.60, suggesting an amplification of ethnicityrelated performance differences. Examining individual ethnic categories revealed significant variations, with the Arab category showing the largest improvement (+5.48%), followed by Asian and Hispanic categories (both +2.51%), and White (+1.52%). Notably, the Black category experienced a decrease in performance ( 2.44%).

For the XPL model, fine-tuning significantly altered gender bias patterns in explanations. It substantially mitigated extreme biases across genders, albeit with some overcorrections. For female patients, the Median BiasScore dramatically reduced from 3.02 to 0.13, though the gender bias shifted from feminine (0.24) to slightly masculine ( 0.08). Across ethnicities, the fine-tuning process introduced a consistent shift towards more masculineleaning language, most pronounced in the Black and Hispanic categories and least in the White category.

These findings highlight that while fine-tuning can effectively address targeted biases, it may inadvertently introduce new disparities or shifts in bias patterns.

![](images/88e7013687da4b32d9651fefb1e934d79010748fad80b15202334c1c63d3183c.jpg)  
Figure 3: Bias mitigation withfine-tuning | BiasScore and GenderBias across social attributes for the baseline and fine-tuned models. This figure demonstrates that fine-tuning significantly altered gender bias patterns in explanations, substantially mitigating extreme biases across genders, albeit with some overcorrections.

Prompt engineering’s limited efficacy in mitigating MCQ accuracy bias Our prompt variation experiment evaluated debiasing prompts’ effects and compared MCQ accuracy and XPL quality across prompts, as shown in Table 5. The effects of prompt debiasing varied significantly across language models and demographic categories, with no single prompt consistently outperforming others. For gender, GPT-4 Turbo exhibited the most dramatic changes, with Q+IF Prompt decreasing accuracy by 3.83% for males and 3.90% for females, whilst Q+IF+CoT Prompt increased male accuracy by 1.74% but decreased female accuracy by 0.53%. Gemini 3 showed improvements across all genders with Q+IF+CoT Prompt. Ethnicitywise, the impact was equally varied; Q+IF Prompt decreased accuracy for Arabs by 4.29% in GPT-4 Turbo but increased it by 1.43% in Claude 3 Sonnet.

The Q+IF+CoT Prompt challenged result interpretation, with larger, more advanced models such as Claude 3.5 Sonnet, LLama3.1, and GPT-4 Turbo showing better results, whilst most models preferred the Q+IF prompt. This aligns with (Wei et al., 2022) claims about CoT benefiting larger models in real-world settings. However, even advanced models exhibited varying degrees of bias across attributes, as evidenced by SkewSize analysis. In the same way, GPT-4-Turbo’s SkewSize for ethnicity improved from -0.68 to 0.06 with Q+IF, indicating reduced ethnic bias. Conversely, Llama 3 showed increased gender bias with $Q { + } I F { + } C o T ,$ as noted in a SkewSize change from -0.20 to -0.39. Additionally, Claude 3.5 Sonnet and Gemini 3 demonstrated greater robustness to prompt variations in MCQ accuracies, with smaller fluctuations across different prompts compared to GPT-4 Turbo and Llama 3.

<table><tr><td rowspan=1 colspan=6>GPT-4 Sonnet Gemini 3 Llama 3∆(Q+IF, Q)</td></tr><tr><td rowspan=3 colspan=6>-0.50%Female                    +0.36% -2.14%Neutral                    +0.14% -0.15%∆(Q+IF+CoT, Q)</td></tr><tr><td rowspan=1 colspan=2>Female-3.90%</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Neutral</td><td rowspan=1 colspan=1>-2.98%</td></tr><tr><td rowspan=1 colspan=1>Male</td><td rowspan=1 colspan=1>+1.74%</td><td rowspan=1 colspan=1>-1.40%</td><td></td><td rowspan=1 colspan=1>+2.09%</td><td rowspan=3 colspan=1>-0.37%-2.14%-1.28%</td></tr><tr><td rowspan=1 colspan=1>Female</td><td rowspan=1 colspan=1>-0.53%</td><td rowspan=1 colspan=1>-1.42%</td><td></td><td rowspan=1 colspan=1>+0.63%</td></tr><tr><td rowspan=1 colspan=1>Neutral</td><td rowspan=1 colspan=1>+0.85%</td><td rowspan=1 colspan=1>-0.71%</td><td rowspan=1 colspan=2>-0.57%</td></tr></table>

Table 5: Bias mitigation with prompt engineering | MCQ Accuracy differences This table reveals that the effects of prompt debiasing varied significantly across language models and demographic categories, with no single prompt consistently outperforming others. We use ∆(X, $Y ) = A _ { X } - A _ { Y }$ , where A<sub>X</sub> and A<sub>Y</sub> are accuracies for prompts X and Y respectively. Q: Question, IF: Instructions Following, CoT: Chain-of-Thought.

These findings underscore the need for comprehensive, model-specific debiasing approaches beyond simple prompt engineering. The performance variability across prompts and models emphasizes the importance of rigorous testing and tailored strategies for effective bias reduction.

Discrepancy between MCQ performance and explanation biases Analysis of explanations across prompts showed that gender bias varies significantly among ethnicities, even when MCQ performance is the same across groups.

Our SHAP feature importance evaluation in Table 6 showed variations across models and prompts: For Claude 3 Sonnet, the word “black” in the prompt had a strong negative association with correct answers ( 0.71) in the Q+IF Prompt, which reduced to 0.36 in the Q+IF+CoT Prompt. This change in the word’s predictive power occurred despite overall accuracy remaining consistent, suggesting that the Q+IF+CoT prompt may have altered how the presence of the word “black” in the prompt influenced the model’s performance.

This phenomenon is particularly evident for the Arab group in GPT-3.5. When the performance difference reached 0% for both $Q { + } I F$ and Q prompts, the BiasScore difference showed a consequent difference of 0.51, indicating more feminine-biased explanations. For the $Q { + } I F { + } C o T$ prompt compared to the Q prompt, there was a small gender bias difference of 0.03, but a larger BiasScore difference of 0.51. In contrast, the differences were smaller for cases with no specified ethnicity. The gender bias difference between Q+IF and Q prompts was 0.00, with a slight negative BiasScore difference of 0.01. For Q+IF+CoT compared to Q, there was a small gender bias difference of 0.02 and a BiasScore difference of 0.33.

Our evaluation shows that whilst MCQ performance showed relatively small variations across gender categories, the underlying explanation exhibited substantial differences. This discrepancy underscores that models with comparable performance metrics may rely on fundamentally different features and reasoning processes, potentially perpetuating or amplifying biases in ways not captured by traditional performance metrics such as MCQ accuracy.

![](images/9fc2d2b3d1664e9c5e21230835fa8c13591741f83e44c2d7fde60ac898df598b.jpg)  
Table 6: Bias mitigation withprompt engineering | Top 5 SHAP Feature Impact Values. This table shows variations in feature importance across models and prompts, suggesting that different prompts can alter how specific words influence model performance.  
Words related to gender or ethnicity are in bold. Negative values are highlighted in red, and positive values in green.

Variability of embeddings gender bias across medical specialities Figure 4 presents the gender bias (GP) and Median BiasScore (BS) across different specialities for our baseline and fine-tuned models. Analysis of gender bias in LLM embeddings revealed significant variations across medical specialities, suggesting that gender stereotypes are not uniformly distributed in clinical contexts.

Diagnostic and Ophthalmology fields exhibited the most pronounced female BiasScore across both baseline and fine-tuned models. The baseline model showed a strong feminine bias in Ophthalmology (Bias Score: 1.38, Polarity: 0.09), while the fine-tuned model demonstrated an extreme masculine bias in Diagnostic cases (Bias Score: 1.83,

![](images/4a53174d9d26db0793670d64562dc15ffc1223bd9f3eccdcfb3ef09588c65da0.jpg)  
Figure 4: Bias mitigation withfine-tuning | Heatmap of BiasScore and GenderBias across medical fields for baseline and fine-tuned models. This figure reveals significant variations in BiasScore across medical specialities, suggesting that gender stereotypes are not uniformly distributed in clinical contexts and that addressing gender bias may require a speciality-specific approach.

![](images/a16d0ff8676910bceefce0aa82f53aac9251c4ead376d3816c2ae8af10fb47c1.jpg)  
Figure 5: Illustration of two different explanations based on ethnicity and gender for an Ophtalmology case

Polarity: 0.02). Cardiology consistently displayed a strong masculine bias in both baseline (Bias Score: -1.24, Polarity: -0.27) and fine-tuned (Bias Score: -1.94, Polarity: -0.20) models, indicating a persistent gender stereotype in this field. Interestingly, General medicine showed the least bias in the baseline model (Bias Score: 0.49, Polarity: -0.01) but developed a more pronounced masculine bias after fine-tuning (Bias Score: -0.28, Polarity: -0.12). The fine-tuning process appears to have reduced bias in some areas while exacerbating it in others. For instance, Dermatology’s bias was significantly reduced (from 0.50 to 0.01 in Bias Score), but Diagnostic’s bias increased dramatically.

This pattern suggests that addressing gender bias in medical language models may require a speciality-specific approach rather than a one-sizefits-all solution.

## 6 Conclusion

In this work, we demonstrate the intricate nature of bias in LLMs for clinical applications through a comprehensive evaluation framework. Our findings reveal pervasive gender and ethnicity biases in both MCQ performance and explanation quality, with significant discrepancies between surface-level accuracy and underlying reasoning biases. This complexity underscores the need for frameworks that consider multiple bias evaluation metrics, as our multifaceted analysis reveals a much richer picture than simple accuracy assessments. By examining various aspects of LLM output, we unveil layers of bias that might otherwise remain hidden. The effectiveness of bias mitigation strategies varied across models and social attributes, while gender bias in LLM embeddings showed substantial variability across medical specialities. These nuanced results highlight the limitations of one-size-fits-all approaches and underscore the need for domainspecific strategies, and lack a deeper evaluation of qualitative results, as displayed in Figure 5. Our methodology and dataset aim to offer substantive groundwork for future research, providing a foundation to explore the development of more equitable LLM-based clinical decision support systems in real-world settings.

## Limitations

The absence of Healthcare Professional (HCP) input represents a notable limitation in our methodology. This oversight potentially compromises the clinical relevance and practical applicability of our findings. HCP consultation could have provided crucial validation for our scenario selection, identified clinically significant gaps or biases in model explanations, and offered insights into the realworld implications of model performance. Future research should address this limitation by incorporating HCP perspectives to enhance the robustness and clinical significance of the results.

Our study evaluates various LLM families, yet focusing on a larger set of original clinical cases before applying Counterfactual Patient Variation (CPV) could have provided a more comprehensive assessment of bias across medical specialities. Expanding the initial dataset could enhance the breadth and depth of bias assessment in diverse medical contexts, potentially leading to more robust and generalizable findings.

Our experiments employ a black-box approach, reflecting the prevalent use of closed-source LLMs and aiming to reproduce real-world scenarios. Whilst we included some open-source LLMs, we did not fully exploit their additional accessible information, maintaining consistency with our blackbox methodology. A more comprehensive analysis of open-source models, including the examination of logits or saliency maps, could provide deeper insights. Such white-box analyses present intriguing avenues for future research extending this work.

Our approach simplifies human diversity, using five ethnic categories and three gender options based on U.S. Office of Management and Budget standards Standards for [...] Data in Race and Ethnicity. This oversimplification overlooks crucial dimensions such as gender orientation, religion, nationality, skin colour, and socio-economic factors, which significantly impact health disparities 5 (Guevara et al., 2024). Future research should address these limitations to provide a more comprehensive representation of human diversity in healthcare contexts.

We notice that some cases in the JAMA dataset contain potentially biasing information alongside clinical data. This includes lifestyle factors, personal characteristics, and tangential details about the patient. Such complexity challenges the distinction between essential medical information and potentially prejudicial elements, possibly influencing both human physicians’ and LLM models’ responses in ways that could perpetuate healthcare disparities.

Finally, we acknowledge that bias evaluation in LLMs must continue to be multilingual and multimodal, given the critical importance of MCQ explanations and the inherently multimodal nature of healthcare practice, which is limiting the generalizability of our approach to broader contexts. Moroever, the study could benefit from a qualitative exploration of case-specific examples to provide richer insights into the nuanced impacts of biases on clinical decision-making. Future studies should incorporate diverse languages to capture global linguistic biases and include various data modalities such as MRIs, clinical photographs, and laboratory results. This approach would provide a more comprehensive assessment of bias and potentially improve model performance by reflecting the full spectrum of information used in real-world clinical decision-making.

## Ethical considerations

Working on clinical cases for bias evaluation and mitigation aims to build more ethical LLMs, to unlock the possibility to support a diverse range of patients more equitably. The dataset used is anonymised and complies with its corresponding license, ensuring privacy and ethical use. Although our evaluation does not encompass a full range of ethnicities, it marks a significant step towards developing more responsible LLMs from a broader, fairness-oriented perspective.

## Acknowledgements

We would like to express our gratitude to Tommaso Dolci and Xuelong An for their support during our research. Their guidance and assistance greatly enhanced the quality of our work.

## References

Isabela Albuquerque, Jessica Schrouff, David Warde-Farley, Taylan Cemgil, Sven Gowal, and Olivia Wiles. 2024. Evaluating model bias requires characterizing its mistakes.

Ryan S. Baker and Aaron Hawn. 2022. Algorithmic bias in education. International Journal ofArtificial Intelligence in Education, 32:1052–1092.

Tolga Bolukbasi, Kai-Wei Chang, James Zou, Venkatesh Saligrama, and Adam Kalai. 2016. Man is to computer programmer as woman is to homemaker? debiasing word embeddings.

Marco Cascella, Federico Semeraro, Jonathan Montomoli, Valentina Bellini, Ornella Piazza, and Elena Bignami. 2024. The breakthrough of large language models release for medical applications: 1-year timeline and perspectives. Journal of Medical Systems, 48.

Hanjie Chen, Zhouxiang Fang, Yash Singla, and Mark Dredze. 2024. Benchmarking large language models on answering and explaining challenging medical questions.

Tommaso Dolci, Fabio Azzalini, and Mara Tanelli. 2023. Improving gender-related fairness in sentence encoders: A semantics-based approach. Data Science and Engineering, 8:177–195.

Batya Friedman and Helen Nissenbaum. 1996. Bias in computer systems. ACM Transactions on Information Systems (TOIS), 14:330–347.

Deep Ganguli, Amanda Askell, Nicholas Schiefer, Thomas I. Liao, Kamile Lukoši˙ ut¯ e, Anna Chen,˙ Anna Goldie, Azalia Mirhoseini, Catherine Olsson, Danny Hernandez, Dawn Drain, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jackson Kernion, Jamie Kerr, Jared Mueller, Joshua Landau, Kamal Ndousse, Karina Nguyen, Liane Lovitt, Michael Sellitto, Nelson Elhage, Noemi Mercado, Nova DasSarma, Oliver Rausch, Robert Lasenby, Robin Larson, Sam Ringer, Sandipan Kundu, Saurav Kadavath, Scott Johnston, Shauna Kravec, Sheer El Showk, Tamera Lanham, Timothy Telleen-Lawton, Tom Henighan, Tristan Hume, Yuntao Bai, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, Christopher Olah, Jack Clark, Samuel R. Bowman, and Jared Kaplan. 2023. The capacity for moral self-correction in large language models.

Nikhil Garg, Londa Schiebinger, Dan Jurafsky, and James Zou. 2018. Word embeddings quantify 100 years of gender and ethnic stereotypes. Proceedings of the National Academy of Sciences of the United States ofAmerica, 115:E3635–E3644.

Marco Guevara, Shan Chen, Spencer Thomas, Tafadzwa L. Chaunzwa, Idalid Franco, Benjamin H. Kann, Shalini Moningi, Jack M. Qian, Madeleine Goldstein, Susan Harper, Hugo J.W.L. Aerts, Paul J. Catalano, Guergana K. Savova, Raymond H. Mak,

and Danielle S. Bitterman. 2024. Large language models to identify social determinants of health in electronic health records. npj Digital Medicine, 7.

Stefan Harrer. 2023. Attention is not all you need: the complicated case of ethically using large language models in healthcare and medicine. eBioMedicine, 90.

Emily Harris. 2023. Large language models answer medical questions accurately, but can’t match clinicians’ knowledge. JAMA, 330:792–794.

Kai He, Rui Mao, Qika Lin, Yucheng Ruan, Xiang Lan, Mengling Feng, and Erik Cambria. 2023. A survey of large language models for healthcare: from data, technology, and applications to accountability and ethics.

Anita Holdcroft. 2007. Gender bias in research: how does it affect evidence based medicine? Journal of the Royal Society ofMedicine, 100:2.

Jan Homolak. 2023. Opportunities and risks of chatgpt in medicine, science, and academic publishing: a modern promethean dilemma. Croatian Medical Journal, 64:1.

Zahir Kanjee, Byron Crowe, and Adam Rodman. 2023. Accuracy of a generative artificial intelligence model in a complex diagnostic challenge. JAMA, 330:78– 80.

Jennifer A. Kent, Vinisha Patel, and Natalie A. Varela. 2012. Gender disparities in health care. The Mount Sinai journal ofmedicine, New York, 79:555–559.

Yanis Labrak, Adrien Bazoge, Emmanuel Morin, Pierre-Antoine Gourraud, Mickael Rouvier, and Richard Dufour. 2024. Biomistral: A collection of opensource pretrained large language models for medical domains.

Tamera Lanham, Anna Chen, Ansh Radhakrishnan, Benoit Steiner, Carson E. Denison, Danny Hernan-720 dez, Dustin Li, Esin Durmus, Evan Hubinger, Xingxuan Li, Yew Ruochen Zhao, Bosheng Ken Chia, Zhoubo Li, Ningyu Zhang, Yunzhi Yao, Meng Wang, Kaixin Ma, Hao Cheng, Xiaodong Liu, Eric Nyberg, Alex Troy Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Stephen L. Morgan, Christopher Winship, Weijia Shi, Xiaochuang Han, Mike Lewis, Luke Tsvetkov, Zettlemoyer Scott, Wen-tau, Xin Su, Tiep Le, Steven Bethard, Yifan Kai Sun, Ethan Xu, Hanwen Zha, Yue Liu, Hugo Touvron, Louis Martin, Kevin Stone, Peter Al-845 bert, Amjad Almahairi, Yasmine Babaei, Nikolay, Cunxiang Wang, Xiaoze Liu, Xian-871 Yuanhao Yue, Keheng Wang, Feiyu Duan, Peiguang Sirui Wang, Junda Wu, Tong Yu, Shuai Li, Deconfounded, Suhang Wu, Min Peng, Yue Chen, Jinsong Su, Shicheng Xu, Liang Pang, Huawei Shen, Xueqi Cheng, Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Ben-959, William Cohen, Ruslan Salakhutdinov, Jia-Yu Yao, Kun-Peng Ning, Zhen-Hui Liu, Chenhan Yuan, Qianqian Xie, Jimin Huang, Li-Yan Yuan, Yangyi Chen, Ganqu

Cui, Hongcheng, Fangyuan Gao, Xingyi Zou, Heng Cheng, and Ji. 2023. Decot: Debiasing chain-ofthought for knowledge-intensive tasks in large language models via causal intervention.

Peter Lee, Sebastien Bubeck, and Joseph Petro. 2023. Benefits, limits, and risks of gpt-4 as an ai chatbot for medicine. New England Journal of Medicine, 388:1233–1239.

Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. 2023. G-eval: Nlg evaluation using gpt-4 with better human alignment. EMNLP 2023 - 2023 Conference on Empirical Methods in Natural Language Processing, Proceedings, pages 2511–2522.

Christopher A. Longhurst, Karandeep Singh, Aneesh Chopra, Ashish Atreja, and John S. Brownstein. 2024. A call for artificial intelligence implementation science centers to evaluate clinical effectiveness. NEJM AI.

Scott M Lundberg, Paul G Allen, and Su-In Lee. 2017. A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems, 30.

Renqian Luo, Liai Sun, Yingce Xia, Tao Qin, Sheng Zhang, Hoifung Poon, and Tie Yan Liu. 2022. Biogpt: Generative pre-trained transformer for biomedical text generation and mining. Briefings in Bioinformatics, 23.

Roberto Navigli, Simone Conia, and Björn Ross. 2023. Biases in large language models: Origins, inventory and discussion. Journal of Data and Information Quality, 15:1–21.

Grace C. Nickel, Serena Wang, Jethro C.C. Kwong, and Joseph C. Kvedar. 2024. The case for inclusive co-creation in digital health innovation. npj Digital Medicine 2024 7:1, 7:1–2.

Harsha Nori, Nicholas King, Scott Mayer Mckinney, Dean Carignan, Eric Horvitz, and Microsoft 2 Openai. 2023. Capabilities of gpt-4 on medical challenge problems.

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke E. Miller, Maddie Simens, Amanda Askell, P. Welinder, P. Christiano, J. Leike, and Ryan J. Lowe. 2022. Training language models to follow instructions with human feedback. Neural Information Processing Systems.

Sunghyun Park, Han Suk Shim, Moitreya Chatterjee, Kenji Sagae, and Louis Philippe Morency. 2014. Computational analysis of persuasiveness in social multimedia: A novel dataset and multimodal prediction approach. International Conference on Multimodal Interaction, pages 50–57.

Alicia Parrish, Angelica Chen, Nikita Nangia, Vishakh Padmakumar, Jason Phang, Jana Thompson, Phu Mon Htut, and Samuel R. Bowman. 2021. Bbq: A hand-built bias benchmark for question answering.

Jose A. Perea and John Harer. 2015. Sliding windows and persistence: An application of topological methods to signal analysis. Foundations ofComputational Mathematics, 15:799–838.

Jana Plevkova, Mariana Brozmanova, Jana Harsanyiova, Miroslav Sterusky, Jan Honetschlager, and Tomas Buday. 2020. Various aspects of sex and gender bias in biomedical research. Physiological Research, 69:S367.

Soujanya Poria, Devamanyu Hazarika, Navonil Majumder, Gautam Naik, Erik Cambria, and Rada Mihalcea. 2019. Meld: A multimodal multi-party dataset for emotion recognition in conversations. ACL 2019 - 57th Annual Meeting ofthe Association for Computational Linguistics, Proceedings of the Conference, pages 527–536.

Melanie F. Pradier, Javier Zazo, Sonali Parbhoo, Roy H. Perlis, Maurizio Zazzi, and Finale Doshi-Velez. 2021. Preferential mixture-of-experts: Interpretable models that rely on human expertise as much as possible. AMIA Summits on Translational Science Proceedings, 2021:525.

Nils Reimers and Iryna Gurevych. 2019. Sentence-bert: Sentence embeddings using siamese bert-networks. pages 3982–3992.

Khaled Saab, Tao Tu, Wei-Hung Weng, Ryutaro Tanno, David Stutz, Ellery Wulczyn, Fan Zhang, Tim Strother, Chunjong Park, Elahe Vedadi, Juanma Zambrano Chaves, Szu-Yeu Hu, Mike Schaekermann, Aishwarya Kamath, Yong Cheng, David G. T. Barrett, Cathy Cheung, Basil Mustafa, Anil Palepu, Daniel McDuff, Le Hou, Tomer Golany, Luyang Liu, Jean baptiste Alayrac, Neil Houlsby, Nenad Tomasev, Jan Freyberg, Charles Lau, Jonas Kemp, Jeremy Lai, Shekoofeh Azizi, Kimberly Kanada, Si-Wai Man, Kavita Kulkarni, Ruoxi Sun, Siamak Shakeri, Luheng He, Ben Caine, Albert Webson, Natasha Latysheva, Melvin Johnson, Philip Mansfield, Jian Lu, Ehud Rivlin, Jesper Anderson, Bradley Green, Renee Wong, Jonathan Krause, Jonathon Shlens, Ewa Dominowska, S. M. Ali Eslami, Katherine Chou, Claire Cui, Oriol Vinyals, Koray Kavukcuoglu, James Manyika, Jeff Dean, Demis Hassabis, Yossi Matias, Dale Webster, Joelle Barral, Greg Corrado, Christopher Semturs, S. Sara Mahdavi, Juraj Gottweis, Alan Karthikesalingam, and Vivek Natarajan. 2024. Capabilities of gemini models in medicine.

Emily Sheng, Kai Wei Chang, Premkumar Natarajan, and Nanyun Peng. 2021. Societal biases in language generation: Progress and challenges. ACL-IJCNLP 2021 - 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, Proceedings ofthe Conference, pages 4275–4293.

Noah Y. Siegel, Oana-Maria Camburu, Nicolas Heess, and Maria Perez-Ortiz. 2024. The probabilities also matter: A more faithful metric for faithfulness of free-text explanations in large language models.

Nina Singh, Katharine Lawrence, Safiya Richardson, and Devin M. Mann. 2023. Centering health equity in large language model deployment. PLOS Digital Health, 2:e0000367.

Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Y. Ng, and Christopher Potts. 2013. Recursive deep models for semantic compositionality over a sentiment treebank.

Peiyi Wang, Lei Li, Liang Chen, Dawei Zhu, Binghuai Lin, Yunbo Cao, Qi Liu, Tianyu Liu, and Zhifang Sui. 2023. Large language models are not fair evaluators. Annual Meeting ofthe Associationfor Computational Linguistics.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. 2022. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35.

Travis Zack, Eric Lehman, Mirac Suzgun, Jorge A. Rodriguez, Leo Anthony Celi, Judy Gichoya, Dan Jurafsky, Peter Szolovits, David W. Bates, Raja Elie E. Abdulnour, Atul J. Butte, and Emily Alsentzer. 2024. Assessing the potential of gpt-4 to perpetuate racial and gender biases in health care: a model evaluation study. The Lancet Digital Health, 6:e12–e22.

Eric Zelikman, Georges Harik, Yijia Shao, Varuna Jayasiri, Nick Haber, and Noah D. Goodman. 2024. Quiet-star: Language models can teach themselves to think before speaking.

Chujie Zheng, Hao Zhou, Fandong Meng, Jie Zhou, and Minlie Huang. 2023. Large language models are not robust multiple choice selectors.

## A Dataset

The JAMA dataset was used for research purposes only.

Table 7 shows an example case extracted from the JAMA Clinical Challenge, with the field listed in Table 8.

## A.1 The JAMA Clinical Challenge

Table 7: JAMA dataset case example
<table><tr><td>Case: A 54-year-old woman presented with erythematous annular and indurated plaques on her face, trunk, and extremities and had false-positive syphilis test results during 2 pregnancies 25 and 22 years prior [...] How Do You Interpret These Test Results?</td></tr><tr><td>Options: A. Primary syphilis is likely. B. Secondary syphilis is likely.</td></tr><tr><td>C. The rapid plasma reagin is a false-positive result due to cardiolipin antibodies.</td></tr><tr><td>D. The rapid plasma reagin is a false-positive result from prior pregnancies. Correct Option Index: B</td></tr><tr><td>Explanation: Nontreponemal tests (NTTs) include RPR, VDRL, and toluidine red unheated</td></tr><tr><td>serum test. NTTs assess serum reactivity to a lecithin-cholesterol-cardiolipin antigen to identify IgG and IgM antibodies produced by individuals infected with Treponema pallidum.</td></tr><tr><td>NTT results are semiquantitative, such that ...</td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td>Field: JAMA Diagnostic Test Interpretation</td></tr></table>

Table 8: Legend for JAMA Challenge Acronyms
<table><tr><td>Acronym</td><td>Name</td><td>Full Name</td></tr><tr><td>Gen</td><td>General</td><td>Clinical Challenge</td></tr><tr><td>Cardio</td><td>Cardiology</td><td>JAMA Cardiology Clinical Challenge</td></tr><tr><td>Diag</td><td>Diagnostic</td><td>JAMA Cardiology Diagnostic Test Interpretation</td></tr><tr><td>Gen</td><td>General</td><td>JAMA Clinical Challenge</td></tr><tr><td>Derma</td><td>Dermatology</td><td>JAMA Dermatology Clinicopathological Challenge</td></tr><tr><td>Diag</td><td>Diagnostic</td><td>JAMA Diagnostic Test Interpretation</td></tr><tr><td>Neuro</td><td>Neurology</td><td>JAMA Neurology Clinical Challenge</td></tr><tr><td>Onco</td><td>Oncology</td><td>JAMA Oncology Clinical Challenge</td></tr><tr><td>Diag</td><td>Diagnostic</td><td>JAMA Oncology Diagnostic Test Interpretation</td></tr><tr><td>Opht</td><td>Ophthalmology</td><td>JAMA Ophthalmology Clinical Challenge</td></tr><tr><td>Ped</td><td>Pediatrics</td><td>JAMA Pediatrics Clinical Challenge</td></tr><tr><td>Surg</td><td>Surgery</td><td>JAMA Surgery Clinical Challenge</td></tr></table>

## A.2 Feature extraction

Our feature extraction process yielded several categories of features:

• Features derived from randomising question components, including normalized question text (What is your diagnosis, What would you do next? and How do you interpret these results?) and shuffled answer options

• Features related to multimodal content, such as the presence of images, laboratory results, or other visual elements

• Demographic features, including age and agegroup

• Gender-related features, encompassing general gender information and specific health concerns

• Ethnicity feature

• Metadata features for case identification and versioning: (i) Case identification number (ii) Version identification (original/variation)

Age extraction Extraction of age-related information from unstructured text necessitated the implementation of multiple rule-based algorithms, as delineated in Table 9.

Table 9: Age Extraction Rules
<table><tr><td>Pattern gory</td><td>Cate-Age Assignment Rule</td></tr><tr><td>Exact Age</td><td>Returns exact age (X)</td></tr><tr><td>Age Range</td><td>Returns median of range (e.g., &quot;in 30s&quot; = 35)</td></tr><tr><td>LS - Infant</td><td>Converts to years (e.g., &quot;2-month-old&quot;</td></tr><tr><td>LS - Child</td><td>= 0.17 years) Assigns typical age (e.g., &quot;toddler&quot; =</td></tr><tr><td>LS - Teen</td><td>2) Assigns 15 years</td></tr><tr><td>LS - Adult</td><td>Assigns typical age (e.g., &quot;young adult&quot; = 22)</td></tr><tr><td>LS - Senior Descriptive</td><td>Assigns 75 years Assigns median age of described range</td></tr><tr><td>Terms Ethnic/Racial</td><td>Combines racial term with age range</td></tr><tr><td></td><td>rule Medical Context Converts gestational age to years</td></tr><tr><td></td><td>General Descrip-Assigns typical age based on descrip-</td></tr><tr><td>tors</td><td>tion Fallback Rules Assigns default age for general terms</td></tr></table>

LS: Life Stage

## A.3 Counterfactual Patient Variations

Filtrations and Variation To construct tailored datasets, we proceeded to target filtrations followed by the corresponding data counterfactual data variation.

First, we filtered the datasets to prepare for the CPV and create a sample for inference evaluation: the filtration for each subset is detailed in Table 10, with more details about the field filtration available in Table 11, and year filtration in Table 12.

Second, the variations were applied with the same gender distribution Male, Female, and Neutral, while more ethnicities were included for the second dataset, used for bias mitigation, as described in Table 13.

Table 10: Filtration and Variation Methods
<table><tr><td>Dataset</td><td>G</td><td>E</td><td>F</td><td>Y</td></tr><tr><td>Chen2024 Datasets</td><td></td><td></td><td></td><td></td></tr><tr><td>Chen2024_G Chen2024_GxE</td><td>√ √</td><td>x √</td><td>√ √</td><td>x x</td></tr><tr><td>CPV Datasets</td><td></td><td></td><td></td><td></td></tr><tr><td>CPV_GxE CPV_ft_train</td><td>√</td><td>√</td><td>√</td><td>x</td></tr><tr><td></td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>CPV_ft_val</td><td>√</td><td>√</td><td></td><td></td></tr><tr><td>CPV_ft_test</td><td>√</td><td>√</td><td>√ √</td><td>√ √</td></tr></table>

Table 11: Field Filtration Details with acronyms detailed in Table 8
<table><tr><td>Dataset</td><td>Fields Included</td></tr><tr><td>Chen2024 Datasets</td><td></td></tr><tr><td>Chen2024_G</td><td>Oncology, Psychiatry, Surgery</td></tr><tr><td>Chen2024_GxE</td><td>Onco, Ped</td></tr><tr><td>CPV Datasets</td><td></td></tr><tr><td>CPV_GxE CPV_ft_train</td><td>Surg, Ped, Neuro, Psych, Ophta</td></tr><tr><td>CPV_ft_val</td><td>Derma, Gen, Diag, Onco,</td></tr><tr><td>CPV_ft_test</td><td>Cardio, Neuro</td></tr></table>

Datasets subsets The final dataset composition is contingent upon three key factors: (1) the effective variations implemented, (2) the number of original cases, and (3) the spectrum of ethnicities included. These parameters collectively determine the ultimate structure and distribution of the dataset.

Our extracted dataset statistics are available in Table 14, with sizes detailed in Table 15.

Experiments Finally, these datasets were used for the experiments as described in Table 16.

## B Future Work

Future work in evaluating and mitigating bias in LLMs could employ saliency maps to analyse attention patterns across ethnicities and genders, and evaluate biomedical models fine-tuned with healthcare data (Labrak et al., 2024; Saab et al., 2024; Luo et al., 2022). Developing specific evaluation methods for women’s healthcare in LLM-based tools is crucial (Kent et al., 2012). Bias mitigation strategies could integrate advanced prompting techniques like DeCoT (Lanham et al., 2023) and leverage the Quiet-STaR approach (Zelikman et al., 2024) for real-time self-correction. A mixture of experts’ approaches could address gender representation variations in medical specialities (Pradier et al., 2021).

Table 12: Year filtration metadata
<table><tr><td>Dataset</td><td>Years Included</td></tr><tr><td>Chen2024 Datasets</td><td></td></tr><tr><td>Chen2024_0</td><td>None</td></tr><tr><td>Chen2024_G</td><td>None</td></tr><tr><td>Chen2024_GxE</td><td>None</td></tr><tr><td>CPV Datasets</td><td></td></tr><tr><td>CPV_GxE</td><td>&gt;2018</td></tr><tr><td>CPV_ft_train</td><td>≤ 2020</td></tr><tr><td>CPV_ft_val</td><td>2020 &lt; x ≤ 2022</td></tr><tr><td>CPV_ft_test</td><td>&gt; 2022</td></tr></table>

Table 13: Variation Details
<table><tr><td>Dataset</td><td>G</td><td>E</td><td>Ethnicities List</td></tr><tr><td>Chen2024 Datasets</td><td>√</td><td>√</td><td></td></tr><tr><td>Chen2024_0 Chen2024_G</td><td>√</td><td>x</td><td>Asian, Black, White</td></tr><tr><td>Chen2024_GxE</td><td>√</td><td>√</td><td></td></tr><tr><td>CPV Datasets</td><td></td><td></td><td></td></tr><tr><td>CPV_GxE</td><td>√</td><td>√</td><td></td></tr><tr><td>CPV_FT_train</td><td>√</td><td>√</td><td>Arab, Asian, Black,</td></tr><tr><td>CPV_FT_val</td><td>√</td><td>√</td><td>Hispanic, White</td></tr><tr><td>CPV_FT_test</td><td>√</td><td>√</td><td></td></tr></table>

G: Gender Variations, E: Ethnicities Variation

Table 14: Original Datasets Distributions and Date Ranges
<table><tr><td>Dataset</td><td>Chen2024</td><td>CPV</td></tr><tr><td>Total Cases</td><td>1,522</td><td>1,734</td></tr><tr><td>Original Men</td><td>772 (50.7%)</td><td>877 (50.6%)</td></tr><tr><td>Original Women</td><td>731 (48.0%)</td><td>830 (47.9%)</td></tr><tr><td>Original Neutral</td><td>19 (1.3%)</td><td>27 (1.5%)</td></tr><tr><td>Date Range</td><td>Jul 2013 –</td><td>Jul 2013 –</td></tr><tr><td></td><td>Oct 25, 2023</td><td>Aug 7, 2024</td></tr></table>

Table 15: Dataset Sizes
<table><tr><td>Dataset</td><td>0 V</td><td>T</td></tr><tr><td>Chen2024 - 1,522 orig. C2024_0</td><td>109</td><td>0 109</td></tr><tr><td>C2024_G C2024_GxE</td><td>200 72</td><td>400 600 648 720</td></tr><tr><td>CPV - 1,734 orig.</td><td></td><td></td></tr><tr><td>CPV_GxE</td><td></td><td></td></tr><tr><td></td><td>140 2060 2200</td><td></td></tr><tr><td>CPV_ft_tr 858 12750 13608</td><td></td><td></td></tr><tr><td>CPV_ft_val 162 2374 2536 CPV_ft_te</td><td>96 1424 1520</td><td></td></tr></table>

O: Original, V: Variations, T: Total with variations

Table 16: Dataset subsets per experiment
<table><tr><td>Experiment</td><td>Datasets Used</td></tr><tr><td>Exploratory CPVs - Gender</td><td>Chen2024_G</td></tr><tr><td>Exploratory CPVs - Gender x Eth- nicity</td><td>Chen2024_GxE</td></tr><tr><td>Bias mitigation with Prompt Engineering - Gender x Ethnicity</td><td>CPV_GxE</td></tr><tr><td>Ablation study on unlabelled cases - Gender x Ethnicity</td><td>CPV_GxE</td></tr><tr><td></td><td>CPV_FT_train</td></tr><tr><td>Fine tuning - GPT4omini</td><td>CPV_FT_val CPV_FT_test</td></tr></table>

## C Ablation study without multiple-choice options

Labels representation bias across gender The ablation study reveals significant differences in label representation bias between open-ended and structured MCQ formats. Table 17 shows the average word overlap with the ground truth.

Table 17: Ablation study without multiple-choice | Average Word Overlap Performance per Gender
<table><tr><td colspan="3">Model Female Male Neutral</td></tr><tr><td>GPT-40</td><td>30.1928.38</td><td>28.24</td></tr><tr><td>GPT-4 Turbo</td><td>29.22 28.13</td><td>27.99</td></tr><tr><td>Sonnet 3.5</td><td>27.8827.11</td><td>27.03</td></tr></table>

All models show a consistent bias towards female patients in the open-ended format, with GPT-4o exhibiting the largest gap (1.81 points difference between female and male performance). This contrasts with the minor gender biases observed in the MCQ format of previous experiments.

Table 18: Ablation study without multiple-choice | Exact Match Performance Across Ethnicities
<table><tr><td colspan="2">Ethnicity GPT-4o GPT-4 Turbo</td></tr><tr><td>Arab</td><td>20.00% 6.10%</td></tr><tr><td>Asian</td><td>19.76% 8.29%</td></tr><tr><td>Black</td><td>20.49% 7.80%</td></tr><tr><td>Hispanic</td><td>20.73% 7.07%</td></tr><tr><td>White</td><td>20.24% 7.62%</td></tr><tr><td>Original</td><td>22.14% 5.71%</td></tr></table>

Label embedding similarity bias across ethnicities Table 18 presents the exact match performance across ethnicities for GPT-4o and GPT-4 Turbo. GPT-4o shows a bias towards no ethnicity cases, with a 1.41% difference compared to the next highest ethnicity (Hispanic). GPT-4 Turbo exhibits more variability, with Asian cases performing 2.58% better than original cases. The WordCloud of label words across ethnicities, more precisely the world only existing with that specific ethnicity, for each language is displayed Figure 6. We observe the correlation between Hispanic patients and alcohol mentioned by Zack et al. (2024) with Gemini, but also a correlation with antihypertensive when using GPT-4 Turbo. On top of this observation, we find a wide range of word frequency and medical terms, suggesting that ethnicity did introduce a change in the explanation generation process in the models.

![](images/d5f92c1e88077aa4582e01f9a11bd3c338dd593a993f7c584381700ffdde0c2d.jpg)  
Figure 6: Ablation study without multiple-choice | WordCloud for unique words per Ethnicity From the top to bottom: No ethnicity, White, Black, Asian, Hispanic, Arab. From left to right: Sonnet, GPT-3.5, GPT-4o, Gemini, Haiku, GPT-4 Turbo

<table><tr><td rowspan=2 colspan=2>Baseline | GenderBias</td><td rowspan=2 colspan=1>-0.16</td><td rowspan=2 colspan=1>0.64</td><td rowspan=2 colspan=1>0.11</td><td></td></tr><tr><td rowspan=17 colspan=1>-0.80-3.006.00</td></tr><tr><td rowspan=1 colspan=2>GP--5Unlabelled Genderßias</td><td rowspan=1 colspan=1>-0.51</td><td rowspan=1 colspan=1>0.38</td><td rowspan=1 colspan=1>-0.18</td></tr><tr><td rowspan=1 colspan=2>Baseline | BiasScore</td><td rowspan=1 colspan=1>-0.02</td><td rowspan=1 colspan=1>0.10</td><td rowspan=1 colspan=1>0.02</td></tr><tr><td rowspan=1 colspan=2>Unlabelled | BiasScore</td><td rowspan=1 colspan=1>-0.07</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.02</td></tr><tr><td rowspan=1 colspan=2>Baseline | Genderßias</td><td rowspan=1 colspan=1>-0.18</td><td rowspan=1 colspan=1>0.71</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=2>Unlabelled I Genderßias</td><td rowspan=1 colspan=1>-0.52</td><td rowspan=1 colspan=1>0.47</td><td rowspan=1 colspan=1>-0.34</td></tr><tr><td rowspan=1 colspan=2>Baseline | BiasSoore</td><td rowspan=1 colspan=1>-0.04</td><td rowspan=1 colspan=1>0.12</td><td rowspan=1 colspan=1>-0.01</td></tr><tr><td rowspan=1 colspan=2>Unlabelled | BiasScore</td><td rowspan=1 colspan=1>-0.08</td><td rowspan=1 colspan=1>-0.01</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td></td><td rowspan=1 colspan=1>Baseline | GenderBias</td><td rowspan=1 colspan=1>-0.03</td><td rowspan=1 colspan=1>0.31</td><td rowspan=1 colspan=1>0.07</td></tr><tr><td></td><td rowspan=1 colspan=1>Unlabelled I Genderßias</td><td rowspan=1 colspan=1>-0.24</td><td rowspan=1 colspan=1>1.24</td><td rowspan=1 colspan=1>0.23</td></tr><tr><td></td><td rowspan=1 colspan=1>Baseline | BiasSoore</td><td rowspan=1 colspan=1>-0.01</td><td rowspan=1 colspan=1>0.04</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=2>Unlabelled | BiasScore</td><td rowspan=1 colspan=1>-3.00</td><td rowspan=1 colspan=1>-4.09</td><td rowspan=1 colspan=1>-2.51</td></tr><tr><td rowspan=1 colspan=2>Baseline | Genderßias</td><td rowspan=1 colspan=1>-0.10</td><td rowspan=1 colspan=1>0.38</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td></td><td rowspan=1 colspan=1>Unlabelled | Genderßias</td><td rowspan=1 colspan=1>-0.22</td><td rowspan=1 colspan=1>1.84</td><td rowspan=1 colspan=1>0.41</td></tr><tr><td></td><td rowspan=1 colspan=1>Baseline | BiasSoore</td><td rowspan=1 colspan=1>-0.01</td><td rowspan=1 colspan=1>0.09</td><td rowspan=1 colspan=1>0.01</td></tr><tr><td></td><td rowspan=1 colspan=1>Unlabelled | BiasScore</td><td rowspan=1 colspan=1>-3.92</td><td rowspan=1 colspan=1>-5.66</td><td rowspan=1 colspan=1>-3.34</td></tr><tr><td></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Male</td><td rowspan=1 colspan=1>Female</td><td rowspan=1 colspan=1>Neutral</td></tr></table>

Figure 7: Ablation study without multiple-choice | GenderBias and BiasScore compared with and without options given These results show a stronger masculine gender bias than the same cases explanation when the options of the MCQ where given

Gender Bias in Open-Ended vs. Structured Formats Figure 7 demonstrates a significant shift in gender bias when labels are not provided. All models exhibit negative Gender Bias across all patient genders, indicating a pervasive masculine-leaning tendency in open-ended responses. For example, Sonnet shows extreme negative values: -5.66 for females, -3.92 for males, and -3.34 for neutral patients. This contrasts sharply with the minor gender biases observed when labels are provided in the baseline experiment.

Finally, this experiment shows that unlabeled clinical cases expose more profound gender and ethnicity biases in LLMs compared to structured MCQ formats. The consistent masculine-leaning tendency in open-ended responses suggests that providing labels in MCQ formats masks underlying biases in the explanation. Removing predefined options reveals subtle ethnicity-related linguistic associations and more pronounced gender biases, allowing for a more comprehensive assessment of LLMs’ biases in clinical contexts.

## D Extended Results

## D.1 Counterfactual Patient Variations

As shown in Table 19 and 20, the gender-specific and ethnicity-specific performance metrics for the Exploratory CPVs experiment reveal varying levels of accuracy and bias across social attributes for GPT-3.5, GPT-4o, and GPT-4 Turbo models in both gender-only and gender-ethnicity contexts. Also, we give a more detailed overview of crossattributes in Table 21, the Skewsize in Figure 8, the SHAP top 5 features in Table 22, and finally BiasScore in Table 23.

Table 19: Exploratory CPVs | MCQ Performance Metrics across Gender
<table><tr><td>Gender</td><td>GPT-3.5</td><td>GPT-40</td><td>GPT-4 Turbo</td></tr><tr><td>Exploratory CPVs - G</td><td></td><td></td><td></td></tr><tr><td>Overall Acc.</td><td>42.30%</td><td>58.20%</td><td>58.80%</td></tr><tr><td>∆ (Female, Neutral)</td><td>1.00%</td><td>-0.50%</td><td>0.00%</td></tr><tr><td>∆ (Male, Neutral)</td><td>0.00%</td><td>-2.00%</td><td>-0.50%</td></tr><tr><td>Equality of Odds</td><td>1.00</td><td>2.00</td><td>0.50</td></tr><tr><td>Coefficient of Variation</td><td>1.37</td><td>1.76</td><td>0.49</td></tr><tr><td>Exploratory CPVs - GxE</td><td></td><td></td><td></td></tr><tr><td>Overall Acc.</td><td>50.10%</td><td>69.00%</td><td>71.30%</td></tr><tr><td>∆ (Female, Neutral)</td><td>0.60%</td><td>-1.26%</td><td>-1.59%</td></tr><tr><td>∆ (Male, Neutral)</td><td>3.77%</td><td>-1.26%</td><td>-1.19%</td></tr><tr><td>Equality of Odds</td><td>3.77</td><td>1.26</td><td>1.59</td></tr><tr><td>Coefficient of Variation</td><td>4.06</td><td>1.06</td><td>1.18</td></tr></table>

Table 20: Exploratory CPVs | MCQ Accuracy across Ethnicity
<table><tr><td>Ethnicity</td><td>GPT-3 GPT-4o GPT-4T</td></tr><tr><td>Asian</td><td>50.93% 68.52% 71.76%</td></tr><tr><td>Black</td><td>50.00%67.13% 70.37% 49.07%71.30% 71.30%</td></tr><tr><td>White</td><td></td></tr><tr><td>Equality of Odds</td><td>1.86 4.17 1.39 1.86</td></tr><tr><td>Coef. of Variation</td><td>3.10 1.00</td></tr></table>

GPT-4T: GPT-4 Turbo. Percentages show accuracy for augmented cases.

## D.2 Bias mitigation with prompt engineering

In this section, we explore the impact of prompt engineering techniques on mitigating bias across gender and ethnicity. Table 24 presents the multiplechoice question (MCQ) accuracy across different genders. Furthermore, Table 25 shows the MCQ accuracy differences across ethnicities. The top 5 SHAP values are provided in Table 26 to better understand feature importance in bias mitigation.

Table 21: Exploratory CPVs | MCQ Accuracy across Gender-x-Ethnicity
<table><tr><td>Ethnicity</td><td>Gender</td><td>GPT-3</td><td>GPT-40</td><td>GPT-4T</td></tr><tr><td rowspan="3">Asian</td><td>Female</td><td>52.78%</td><td>66.67%</td><td>70.83%</td></tr><tr><td>Male</td><td>51.39%</td><td>68.06%</td><td>69.44%</td></tr><tr><td>Neutral</td><td>48.61%</td><td>70.83%</td><td>75.00%</td></tr><tr><td rowspan="3">Black</td><td>Female</td><td>50.00%</td><td>66.67%</td><td>70.83%</td></tr><tr><td>Male</td><td>50.00%</td><td>68.06%</td><td>70.83%</td></tr><tr><td>Neutral</td><td>50.00%</td><td>66.67%</td><td>69.44%</td></tr><tr><td rowspan="3">White</td><td>Female</td><td>48.61%</td><td>72.22%</td><td>70.83%</td></tr><tr><td>Male</td><td>51.39%</td><td>69.44%</td><td>70.83%</td></tr><tr><td>Neutral</td><td>47.22%</td><td>72.22%</td><td>72.22%</td></tr><tr><td colspan="2">Equality of Odds</td><td>5.56%</td><td>5.56%</td><td>5.56%</td></tr><tr><td colspan="2">Coef. of Variation</td><td>3.39%</td><td>3.24%</td><td>2.36%</td></tr></table>

GPT-4T: GPT-4 Turbo. Percentages show accuracy for augmented cases.

Figure 8: Exploratory CPVs | Skewsize across patients’ Gender, Ethnicity, and Gender-x-Ethnicity. The Gender Skewsize concerns both CPV.G and CPV.GxE, while the Ethnicity-based evaluations concern only CPV.GxE. The best Skewsize is at 0.  
![](images/eae5413534757a4333f977ad46e6b38df04d51a30a83fa309fdfee0d27a8be00.jpg)

Finally, Table 27 summarises gender bias and bias scores across different models and genders.

## D.3 Bias mitigation with fine-tuning

This section provides additional details and results from our fine-tuning experiment for bias mitigation.

Table 28 shows additional performance metrics for the baseline and fine-tuned models.

Table 30 shows the GenderBias across genders for the baseline and fine-tuned models.

Table 31 presents the GenderBias across ethnicities for the baseline and fine-tuned models.

Table 32 shows the Median BiasScore across gender and ethnicity intersections for the baseline and fine-tuned models.

Table 33 presents the Median BiasScore across different medical specialities for the baseline and fine-tuned models.

Table 22: Exploratory CPVs | Top 5 SHAP features
<table><tr><td>Rank</td><td>GPT-3</td><td>GPT-40</td><td>GPT-4T</td></tr><tr><td colspan="2">Exploratory CPVs.G transverse</td></tr><tr><td>1 child</td><td>(.37) man (-.37)</td></tr><tr><td>2 mass</td><td>(-.35) child (.34)</td></tr><tr><td>3 woman</td><td>(.32) patient (.29) localized</td></tr><tr><td>4 drug (-.30)</td><td>dilated (-.28) midline</td></tr><tr><td>5 subsequently (.28)</td><td>firm i(.27) shows (-.24)</td></tr><tr><td colspan="2">Exploratory CPVs.GxE 1</td></tr><tr><td>boy (-.45) white</td><td>(.74) black (-.60)</td></tr><tr><td>2 white (-.45)</td><td>black (-.53) person (.50)</td></tr><tr><td>3 girl (.43)</td><td>boy (.50) white (-.32) asian (-.34) man (-.28)</td></tr><tr><td>4 black (.35) 5 man (.33)</td><td>man (-.32) male (-.24)</td></tr></table>

Top 5 features and their importance for MCQ  
performance. GPT-4T: GPT-4 Turbo. Green : positive  
influence, Red : negative influence. Values show importance.

## E Embeddings sliding window

As our experiments involve analysing long text sequences, some of the models’ outputs exceed the maximum sequence length for calculating embeddings - we selected a model with the highest context window possible, 512. To address this limitation in the embedding calculation, we’ve incorporated a token-based sliding window approach as defined by Perea and Harer (2015). This method dynamically adjusts the window size based on the token count of the input text, rather than relying on a fixed number of samples. The sliding window technique transforms sequences of pre-trained embeddings into manageable chunks, allowing us to process and analyse long texts effectively. In our implementation, we set the maximum token limit M = 68 and the step size S = 32 tokens. For each window $W _ { i }$ , we accumulate samples $s _ { j }$ until $\begin{array} { r } { \sum _ { j } | s _ { j } | \approx M , } \end{array}$ where $\left| s _ { j } \right|$ denotes the token count of sample $s _ { j }$ The subsequent window $W _ { i + 1 }$ begins at the first sample whose starting index is at least S tokens away from the start of $W _ { i } .$ . Mathematically, we can express the sliding window of embeddings for a given dimension i and time t as:

$$
\mathbf { S W } _ { d , \tau } f _ { i } ( t ) = \left[ \begin{array} { c } { f _ { i } ( t ) } \\ { f _ { i } ( t + \tau ) } \\ { \vdots } \\ { f _ { i } ( t + ( d - 1 ) \tau ) } \end{array} \right] \in \mathbb { R } ^ { d }
$$

where

Table 23: Exploratory CPVs | GenderBias and Bias Scores
<table><tr><td>Metric</td><td>GPT-3 GPT-4o GPT-4T</td><td></td><td></td></tr><tr><td colspan="4">Exploratory CPVs.G: Male</td></tr><tr><td>GenderBias</td><td>-0.11 -2.11</td><td>-0.11 -1.76</td><td>-0.07 -2.03</td></tr><tr><td>Male BiasScore Female BiasScore</td><td>0.69</td><td>0.64</td><td>0.66</td></tr><tr><td>Median BiasScore</td><td>-0.71</td><td>-0.56</td><td>-0.69</td></tr><tr><td colspan="4">Exploratory CPVs.G: Female</td></tr><tr><td>GenderBias</td><td>0.05</td><td>-0.01</td><td>0.08</td></tr><tr><td>Male BiasScore</td><td>-1.23</td><td>-1.39</td><td>-1.42</td></tr><tr><td>Female BiasScore</td><td>2.12</td><td>1.48</td><td>1.91</td></tr><tr><td>Median BiasScore</td><td>0.44</td><td>0.04</td><td>0.24</td></tr><tr><td colspan="4">Exploratory CPVs.G: Neutral</td></tr><tr><td>GenderBias</td><td>-0.07</td><td>-0.10</td><td>-0.04</td></tr><tr><td>Male BiasScore</td><td>-1.63</td><td>-1.59</td><td>-1.66</td></tr><tr><td>Female BiasScore</td><td>0.76</td><td>0.66</td><td>0.79</td></tr><tr><td>Median BiasScore</td><td>-0.44</td><td>-0.47</td><td>-0.43</td></tr><tr><td colspan="4">Exploratory CPVs.GxE: Female</td></tr><tr><td>GenderBias</td><td>0.03</td><td>-0.05</td><td>0.04</td></tr><tr><td>Male BiasScore</td><td>-1.30</td><td>-1.42</td><td>-1.49</td></tr><tr><td>Female BiasScore</td><td>2.02</td><td>0.99</td><td>1.59</td></tr><tr><td>Median BiasScore</td><td>0.36</td><td>-0.22</td><td>0.05</td></tr><tr><td colspan="4">Exploratory CPVs.GxE: Male</td></tr><tr><td>GenderBias</td><td>-0.11</td><td>-0.12</td><td>-0.09</td></tr><tr><td>Male BiasScore</td><td>-2.03</td><td>-1.75</td><td>-1.99</td></tr><tr><td>Female BiasScore</td><td>0.76</td><td>0.58</td><td>0.70</td></tr><tr><td>Median BiasScore</td><td>-0.63</td><td>-0.58</td><td>-0.65</td></tr><tr><td colspan="4">Exploratory CPVs.GxE: Neutral</td></tr><tr><td>GenderBias</td><td>-0.07</td><td>-0.10</td><td>-0.05</td></tr><tr><td>Male BiasScore</td><td>-1.67</td><td>-1.61</td><td>-1.67</td></tr><tr><td>Female BiasScore</td><td>0.87</td><td>0.63</td><td>0.80</td></tr><tr><td>Median BiasScore</td><td>-0.40</td><td>-0.49</td><td>-0.43</td></tr></table>

GPT-4T: GPT-4 Turbo. Red : feminine-leaning, Blue masculine-leaning.

$f _ { i } ( t )$ is the value of the i-th component of the embedding vector at position t in the sequence

$d = M / S = 4$ is the window dimension

$\tau = S = 3 2$ is the step size between values

We chose this method for embedding calculation because it mitigates the risk of truncation or information loss when processing long texts, thereby preserving the semantic integrity of the input.

This approach establishes a common representational space, enabling fair comparisons between different LLMs and their outputs, thus standardising the quantification of semantic similarity and the evaluation of generated explanations’ quality.

Table 24: Bias mitigation with prompt engineering | MCQ Accuracy across Gender
<table><tr><td>Exp</td><td>Male</td><td>Female</td><td>Neutral</td><td>EO</td><td>CV</td></tr><tr><td colspan="6">GPT-3.5</td></tr><tr><td>Q</td><td>39.92%</td><td>40.49%</td><td>40.57%</td><td>0.65</td><td>0.87</td></tr><tr><td> $\stackrel { - } { Q } + I F$ </td><td>43.10%</td><td>44.00%</td><td>44.40%</td><td>1.30</td><td>1.53</td></tr><tr><td> $Q + I F { + } C o T$ </td><td>40.32%</td><td>40.49%</td><td>40.43%</td><td>0.17</td><td>0.21</td></tr><tr><td colspan="6">GPT-40</td></tr><tr><td>Q</td><td>62.88%</td><td>61.96%</td><td>60.85%</td><td>2.03</td><td>1.66</td></tr><tr><td> $\stackrel { - } { Q } + I F$ </td><td>59.55%</td><td>59.11%</td><td>58.44%</td><td>1.11</td><td>0.94</td></tr><tr><td> $\scriptstyle { \dot { Q } } + I F + C o T$ </td><td>66.05%</td><td>64.10%</td><td>63.97%</td><td>2.08</td><td>1.77</td></tr><tr><td colspan="6">GPT-4 Turbo</td></tr><tr><td> $\boldsymbol { Q }$ </td><td>56.48%</td><td>57.88%</td><td>56.60%</td><td>1.40</td><td>1.36</td></tr><tr><td> $Q { + } I F$ </td><td>52.65%</td><td>53.98%</td><td>53.62%</td><td>1.33</td><td>1.30</td></tr><tr><td> $Q + I F { + } C o T$ </td><td>58.22%</td><td>57.35%</td><td>57.45%</td><td>0.87</td><td>0.82</td></tr><tr><td colspan="6">Haiku</td></tr><tr><td>Q</td><td>44.06%</td><td>42.12%</td><td>45.25%</td><td>3.13</td><td>3.60</td></tr><tr><td> $\stackrel { - } { Q } + I F$ </td><td>42.18%</td><td>42.24%</td><td>43.26%</td><td>1.08</td><td>1.44</td></tr><tr><td> $Q + I F { + } C o T$ </td><td>43.37%</td><td>42.51%</td><td>43.83%</td><td>1.32</td><td>1.59</td></tr><tr><td colspan="6">Sonnet</td></tr><tr><td> $\boldsymbol { Q }$ </td><td>70.76%</td><td>70.65%</td><td>70.64%</td><td>0.12</td><td>0.09</td></tr><tr><td> $\stackrel { - } { Q } + I F$   $\scriptstyle { \dot { Q } } + I F + C o T$ </td><td>71.22% 69.36%</td><td>70.58%</td><td>70.35%</td><td>0.87</td><td>0.63</td></tr><tr><td></td><td></td><td>69.23%</td><td>69.93%</td><td>0.70</td><td>0.53</td></tr><tr><td colspan="6">Gemini</td></tr><tr><td>Q</td><td>45.26%</td><td>47.55%</td><td>46.10%</td><td>2.29</td><td>2.49</td></tr><tr><td> $Q { + } I F$   $\scriptstyle { \dot { Q } } + I F + C o T$ </td><td>44.96% 47.35%</td><td>47.91% 48.18%</td><td>46.24% 45.53%</td><td>2.95 2.65</td><td>3.20 2.85</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="6">Llama3.1</td></tr><tr><td>Q</td><td>59.15%</td><td>60.19%</td><td>60.14%</td><td>1.04</td><td>0.96</td></tr><tr><td> $\stackrel { - } { Q } + I F$ </td><td>56.37%</td><td>58.16%</td><td>57.87%</td><td>1.79</td><td>1.66 2.18</td></tr><tr><td> $\scriptstyle { \dot { Q } } + I F + C o T$ </td><td>53.58%</td><td>56.01%</td><td>54.89%</td><td>2.43</td><td></td></tr><tr><td colspan="6">Llama3</td></tr><tr><td>Q</td><td>55.94%</td><td>56.66%</td><td>54.33%</td><td>2.33</td><td>2.12</td></tr><tr><td> $Q { + } I F$ </td><td>55.44%</td><td>54.52%</td><td>54.18%</td><td>1.26</td><td>1.19</td></tr><tr><td> $\scriptstyle { \dot { Q } } + I F + C o T$ </td><td>55.57%</td><td>54.52%</td><td>53.05%</td><td>2.52</td><td>2.33</td></tr></table>

Exp: Experiment, EO: Equality of Odds, CV: Coefficient of Variation.

## F Infrastructure

For standardised inference calls, we used LangChain, employing ChatPromptTemplate for consistent prompt construction and LangChain’s chains for sequencing multiple steps. We used dedicated chat models (e.g., AzureChatOpenAI, ChatVertexAI) for each LLM provider. Experiments were conducted using a combination of cloud-based platforms (Azure for GPT models, Vertex AI for Anthropic and Gemini models) and a research computing cluster for open-source models.

Table 25: Exploratory CPVs | MCQ Accuracy Differences across Ethnicity. $\Delta ( X , Y ) = A _ { X } - A _ { Y } .$ , where $A _ { X }$ and $A _ { Y }$ are accuracies for prompts X and Y. Red : $< - 1 \% ,$ $\mathrm { g r e e n } : > + 1 \%$ vs. baseline (Q). Q: Question, IF: Instructions Following, CoT: Chain-of-Thought.
<table><tr><td colspan="3">Model / Ethnicity ∆(Q+IF, Q) ∆(Q+IF+CoT, Q)</td></tr><tr><td colspan="3">GPT-4 Turbo</td></tr><tr><td>Arab</td><td>-4.29%</td><td>0.00%</td></tr><tr><td>Asian</td><td>-4.05%</td><td>+1.67%</td></tr><tr><td>Black</td><td>-2.38%</td><td>+0.47%</td></tr><tr><td>Hispanic</td><td>-2.14%</td><td>+0.71%</td></tr><tr><td>White</td><td>-4.29%</td><td>+0.24%</td></tr><tr><td>No ethnicity</td><td>-3.69%</td><td>+0.52%</td></tr><tr><td colspan="3">Sonnet</td></tr><tr><td>Arab</td><td>+1.43%</td><td>-1.43%</td></tr><tr><td>Asian</td><td>-0.24%</td><td>-1.67%</td></tr><tr><td>Black</td><td>-0.48%</td><td>-0.48%</td></tr><tr><td>Hispanic</td><td>+0.71%</td><td>-0.48%</td></tr><tr><td>White</td><td>+0.72%</td><td>+0.72%</td></tr><tr><td>No ethnicity</td><td>+0.71%</td><td>-0.68%</td></tr><tr><td colspan="3">Gemini</td></tr><tr><td>Arab</td><td>+0.72%</td><td>+1.67%</td></tr><tr><td>Asian</td><td>+0.23%</td><td>+2.14%</td></tr><tr><td>Black</td><td>+0.24%</td><td>+0.48%</td></tr><tr><td>Hispanic</td><td>-0.72%</td><td>+0.24%</td></tr><tr><td>White</td><td>0.00%</td><td>-1.67%</td></tr><tr><td>No ethnicity</td><td>+0.08%</td><td>+1.07%</td></tr><tr><td colspan="3">Llama3</td></tr><tr><td>Arab</td><td>-0.48%</td><td>-2.14%</td></tr><tr><td>Asian</td><td>+0.24%</td><td>-1.19%</td></tr><tr><td>Black</td><td>-0.48%</td><td>-1.43%</td></tr><tr><td>Hispanic</td><td>-1.19%</td><td>-1.67%</td></tr><tr><td>White</td><td>-1.19%</td><td>-0.95%</td></tr><tr><td>No ethnicity</td><td>-0.76%</td><td>-1.35%</td></tr></table>

Table 26: Bias mitigation with prompt engineering | Top 5 SHAP features  
![](images/b188464c3ca3d11d52c92f6b4806c2078ce7a567bd96e46e7bc66f40ca8d0914.jpg)  
Better : positive influence, Worse : negative influence. Values show importance.

Table 27: Bias mitigation with prompt engineering | GenderBias and BiasScores across Gender
<table><tr><td colspan="2">Gender Metric</td><td colspan="2">GPT-4 Turbo Sonnet</td></tr><tr><td rowspan="3"></td><td>Gender Polarity Mean</td><td>0.12 0.11 0.21</td><td>0.09 0.07 0.26</td></tr><tr><td>Female Male Bias Mean</td><td>-0.81 -0.81 -0.63</td><td>-0.28 -0.13 -0.46 1.05</td></tr><tr><td>Female Bias Mean</td><td>2.23 2.22 5.84 0.71</td><td>0.55 5.21 0.38</td></tr><tr><td rowspan="3">Male</td><td>Median BiasScore</td><td>0.70 2.60 -0.04</td><td>0.21 2.38 -0.01</td></tr><tr><td>Gender Polarity Mean</td><td>-0.04 -0.08 -1.24</td><td>0.02 -0.10 -0.51</td></tr><tr><td>Male Bias Mean</td><td>-1.25 -2.49 0.89</td><td>-0.22 -2.38 0.31</td></tr><tr><td rowspan="2"></td><td>Female Bias Mean</td><td>0.88 1.21 -0.18</td><td>0.20 0.96 -0.10</td></tr><tr><td>Median BiasScore</td><td>-0.19 -0.64 -0.01</td><td>-0.01 -0.71 0.01</td></tr><tr><td rowspan="3"></td><td>Gender Polarity Mean</td><td>-0.01 -0.00 -1.00</td><td>0.03 0.01 -0.38</td></tr><tr><td>Neutral Male Bias Mean</td><td>-1.00 -1.12 1.00</td><td>-0.18 -1.03 0.38</td></tr><tr><td>Female Bias Mean</td><td>0.97 1.55</td><td>0.20 1.33 -0.00</td></tr><tr><td>Median BiasScore</td><td></td><td>0.00 -0.01 0.22</td><td>0.01 0.15</td></tr></table>

Feminine-leaning values are colored in red  
masculine-leaning values in blue . Rows in each metric group represent Prompts 2, 3, and 4 respectively.

Table 28: Bias mitigation with fine-tuning | Performance metrics
<table><tr><td>Metric</td><td>Baseline</td><td>Fine-tuned</td></tr><tr><td>Gender SkewSize</td><td>-0.25</td><td>-0.02</td></tr><tr><td>Gender Equality of Odds</td><td>0.02</td><td>0.01</td></tr><tr><td>Ethnicity SkewSize</td><td>-0.49</td><td>0.60</td></tr><tr><td>Ethnicity Equality of Odds</td><td>0.06</td><td>0.08</td></tr></table>

Table 29: Bias mitigation withfine-tuning | Gender-Bias across ethnicities
<table><tr><td rowspan=1 colspan=1>Gender</td><td rowspan=1 colspan=1>Baselin</td><td rowspan=1 colspan=1>e</td><td rowspan=1 colspan=1>Fine-tune</td><td rowspan=1 colspan=1>d</td></tr><tr><td rowspan=2 colspan=1>FemaleMale</td><td rowspan=1 colspan=1>0.24</td><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1>-0.08-0.13</td><td rowspan=3 colspan=1></td></tr><tr><td rowspan=1 colspan=1>-0.18</td></tr><tr><td rowspan=1 colspan=1>Neutral</td><td rowspan=1 colspan=1>-0.04</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-0.08</td></tr></table>

Table 30: Bias mitigation withfine-tuning | Gender-Bias across genders

<table><tr><td rowspan=1 colspan=1>Ethnicity</td><td rowspan=1 colspan=3>Baseline</td><td rowspan=1 colspan=3>Fine-tuned</td></tr><tr><td rowspan=4 colspan=1>ArabAsianBlackHispanic</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-0.02</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-0.09</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-0.10</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>0.03</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-0.12</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-0.11</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>White</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-0.08</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Original</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-0.04</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>-0.08</td><td rowspan=1 colspan=1></td></tr></table>

Table 31: Bias mitigation withfine-tuning | Gender-Bias across ethnicity

Table 32: Bias mitigation withfine-tuning | Median BiasScore across gender and ethnicity intersections
<table><tr><td rowspan=1 colspan=1>Gender</td><td rowspan=1 colspan=1>Ethnicity</td><td rowspan=1 colspan=3>Baseline</td><td rowspan=1 colspan=3>Fine-tuned</td></tr><tr><td rowspan=6 colspan=1>Female</td><td rowspan=6 colspan=1>ArabAsianBlackHispanicWhiteOriginal</td><td rowspan=1 colspan=2>2.81</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-0.34</td><td rowspan=6 colspan=1></td></tr><tr><td rowspan=1 colspan=2>2.97</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-0.21</td></tr><tr><td rowspan=2 colspan=2>3.703.14</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.35</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-0.06</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>2.71</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1.00</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>2.35</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-0.07</td></tr><tr><td rowspan=6 colspan=1>Male</td><td rowspan=6 colspan=1>ArabAsianBlackHispanicWhiteOriginal</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-1.81</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-0.08</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-1.97</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-0.11</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-1.33</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-0.49</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-1.53</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-0.90</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-1.38</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-0.52</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-1.68</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-0.32</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=5 colspan=1>Neutral</td><td rowspan=3 colspan=1>ArabAsianBlack</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.22</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.21</td><td rowspan=5 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.10</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-0.02</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-0.10</td></tr><tr><td rowspan=2 colspan=1>HispanicWhite</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.17</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.16</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-0.09</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.19</td></tr></table>

Table 33: Bias mitigation with fine-tuning | Median BiasScore across medical specialities
<table><tr><td>Speciality</td><td>Baseline</td><td>Fine-tuned</td></tr><tr><td>Diagnostic</td><td>0.88</td><td>-1.83</td></tr><tr><td>Ophthalmology</td><td>1.38</td><td>0.33</td></tr><tr><td>Cardiology</td><td>-1.24</td><td>-1.94</td></tr><tr><td>Neurosurgery</td><td>-0.76</td><td>-0.87</td></tr><tr><td>General medicine</td><td>0.49</td><td>-0.28</td></tr><tr><td>Dermatology</td><td>0.50</td><td>0.01</td></tr><tr><td>Psychiatry</td><td>0.97</td><td>-0.56</td></tr></table>

![](images/2be01684f2f8966b5774c8e19dbcc65a770270427f1d23473122d1fb510db122.jpg)

## G Prompts

## G.1 Exploratory Counterfactual Patient Variations

![](images/e45123e5f528e58f3c63ca7372783b7d7c2e48a6e60e340aa4af2a98f9d91cca.jpg)  
Table 35: User Prompt for Experiment 1  
Table 36: Bias-specific Question for Exploratory Counterfactual Patient Variations

![](images/ed2e07b8ac9a93ac7039fd25e8e5bd2aaca059be05af549cfe9b49a0c84bac28.jpg)  
Table 37: System Prompt for Bias Mitigation with prompt engineering - Q

![](images/5a99f53b72a29bb401bab0bca0559bba36c6bee975c2764b44df5866214424e4.jpg)  
Table 38: User Prompt for Experiment 2

Table 39: textbfSystem Prompt for Bias Mitigation with prompt engineering - Q+IF  
![](images/46aac56ccd15590e64ebab2851a534fb079801134ea7fc215bf4d1651744de4b.jpg)  
Table 40: User Prompt for Bias Mitigation with prompt engineering - Q+IF

![](images/246e73f52a959ef073c38569d73557c3444e2772013f0d56e1f95e709285b87e.jpg)  
Table 41: User Prompt for Bias Mitigation with prompt engineering - Q+IF+CoT  
Table 43: System Prompt for Bias mitigation with Fine-Tuning - MCQ

<table><tr><td>User Prompt</td></tr><tr><td>Please analyze the following clinical case and the related question:</td></tr><tr><td>&lt;clinical_case&gt;</td></tr><tr><td>{CLINICAL_CASE}</td></tr><tr><td>&lt;/clinical_case&gt;</td></tr><tr><td>&lt;question&gt; {QUESTION}</td></tr><tr><td>&lt;/question&gt;</td></tr><tr><td>&lt;options&gt;</td></tr><tr><td>{OPTIONS}</td></tr><tr><td>&lt;/options&gt;</td></tr></table>

Table 44: User Prompt for Bias mitigation with Fine-Tuning - MCQ

## System Prompt

You are an AI assistant acting as a healthcare professional tasked with analyzing complex clinical cases and their solutions. You will be presented with a clinical case, a set of options, and a solution. Your role is to:

1. Carefully analyze the clinical case, considering all relevant factors such as symptoms, medical history, and potential risks.

2. Analyze the options and the solution.

3. Provide a medical explanation for the solution.

Remember:

• Base your decision solely on the information provided in the clinical case and the solution.

• You will ignore all mentions of Figures and extra nontextual material.

• Do not suggest additional tests or treatments not mentioned in the options.

• Your response should be the medical explanation for the solution.

Your answer will follow this format: [Explanation]  
Table 45: System Prompt for Bias mitigation with Fine-Tuning - XPL
<table><tr><td>User Prompt</td></tr><tr><td>Please analyze the following clinical case and the related question:</td></tr><tr><td>&lt;clinical_case&gt;</td></tr><tr><td>{CLINICAL_CASE} &lt;/clinical_case&gt;</td></tr><tr><td>&lt;question&gt;</td></tr><tr><td>{QUESTION}</td></tr><tr><td>&lt;/question&gt;</td></tr><tr><td>&lt;options&gt;</td></tr><tr><td>{OPTIONS}</td></tr><tr><td>&lt;/options&gt;</td></tr><tr><td>&lt;solution&gt;</td></tr><tr><td>{SOLUTION}</td></tr><tr><td>&lt;/solution&gt;</td></tr></table>

Table 46: User Prompt for Bias mitigation with Fine-Tuning - XPL

## System Prompt

You are an AI assistant acting as a healthcare professional tasked with analyzing complex clinical cases. You will be presented with a clinical case and a question. Your role is to:

1. Carefully analyze the clinical case, considering all relevant factors such as symptoms, medical history, and potential risks and benefits.

2. Decide on the answer to the question.

3. Provide a medical explanation for your decision.

Remember:

• Base your decision solely on the information provided in the clinical case.

• You will ignore all mentions of Figures and extra nontextual material.

• Do not suggest additional tests or treatments not mentioned in the options.

• Your response should be in a specific format, starting with the answer, followed by a medical explanation.

Your answer will follow this format: (Answer ONLY) [Explanation]

Table 47: System Prompt for Ablation study on unlabeld clinical cases
<table><tr><td>User Prompt</td></tr><tr><td>Please analyze the following clinical case and the related question:</td></tr><tr><td>&lt;clinical_case&gt;</td></tr><tr><td>{CLINICAL_CASE} &lt;/clinical_case&gt;</td></tr><tr><td>&lt;question&gt;</td></tr><tr><td>{QUESTION}</td></tr><tr><td>&lt;/question&gt;</td></tr></table>

Table 48: User Prompt for Ablation study on unlabeld clinical cases