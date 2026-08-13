# Specializing Large Language Models to Simulate Survey Response Distributions for Global Populations

Yong Cao<sup>1</sup>, Haijiang Liu<sup>2</sup>, Arnav Arora<sup>3</sup>, Isabelle Augenstein<sup>3</sup>,

Paul Röttger<sup>4</sup>, Daniel Hershcovich<sup>3</sup>

<sup>1</sup>University of Tübingen, Tübingen AI Center <sup>2</sup>Wuhan University of Science and Technology <sup>3</sup>University of Copenhagen, <sup>4</sup>Bocconi University yong.cao@uni-tuebingen.de, dh@di.ku.dk

## Abstract

Large-scale surveys are essential tools for informing social science research and policy, but running surveys is costly and time-intensive. If one could accurately simulate group-level survey results, this could be very valuable to social science research. Prior work has explored the use of large language models (LLMs) for simulating human behaviors, mostly through prompting. In this paper, we are the first to specialize LLMs for the task of simulating survey response distributions. As a testbed for this task, we use country-level results from two global cultural surveys. We devise a fine-tuning method based on first-token probabilities to minimize divergence between predicted and actual response distributions for a given question. Then, we show that this method substantially outperforms other methods and zero-shot classifiers, even on unseen questions, countries, and a completely unseen survey. While even our best models struggle with the task, especially on unseen questions, our results demonstrate the benefits of specialization for simulation, which may accelerate progress towards sufficiently accurate simulation in the future.

## 1 Introduction

Humans are diverse, and they hold diverse opinions. This is why surveys are essential tools for informing decision-making in policy and industry as well as social science research. Running large-scale surveys, however, is often costly and time-intensive.

Large language models (LLMs) have demonstrated promising potential for simulating human behaviors across groups and individuals (Argyle et al., 2023; Aher et al., 2023; Manning et al., 2024, inter alia). LLM simulations of survey responses, if accurate towards the corresponding populations, could accelerate social science research and aid in more informed policy decisions. Out of the box, however, LLMs are known to generate erroneous, stereotypical, or overconfident answers, especially in culturally diverse contexts (Yang et al., 2024), which limits their usefulness for survey simulations. Prior work has at most tried to improve simulation accuracy through prompting strategies (Kwok et al., 2024; Manning et al., 2024; Sun et al., 2024).

![](images/dbe33a4a0b9c6548cc0e3ca94ea605d403b8b756e6cc6362e7c282e4918599f1.jpg)  
Figure 1: Overview of our proposed survey response distribution simulation framework (right panel) versus direct answer prediction (left panel), highlighting a novel perspective on cultural simulation with LLMs.

In this study, our goal is instead to specialize LLMs for survey simulation and gain a better understanding of how good LLMs can be at simulating survey responses when trained to do so, rather than how good they are when prompted out of the box. As our main testbed, we use countrylevel response distributions from the widely-used World Values Survey (WVS; Haerpfer et al., 2022). When prompted with a survey question (e.g., “In your opinion, should the use of nuclear power in Japan be reduced, maintained at its current level, or increased?”), corresponding answering options (e.g. “Reduced”, “Maintained at current level”, “Increased”) and a target country (e.g. “Japan”), we want our model to predict the distribution over the answering options for that target country. To specialize LLMs for this task, we devise a fine-tuning method based on first-token probabilities, where the goal is to minimize divergence between predicted and actual country-level response distributions for a given survey question.

As shown in Figure 1, we train models on one set of questions and countries from the WVS, then evaluate on both seen and unseen countries and questions as well as another completely unseen survey. Across seven LLMs from three model families, our fine-tuning method substantially boosts prediction accuracy on seen and unseen WVS countries and questions. These results also hold for a completely unseen survey, i.e. the Pew Global Attitudes Survey. Simultaneously, we find the performance of even the best-fine-tuned models to be far from perfect, especially on unseen questions. We also find that all LLMs we tested, whether fine-tuned or not, are less diverse in their predictions across countries than the actual human survey data.

In summary, we make following three main contributions:

• We introduce group-level survey response distribution prediction as a simulation task, and share three datasets adapted for training and testing models on this task: two in English and one in Chinese.

• We propose a fine-tuning method based on first-token probabilities of multi-choice question answering, and show that this method performs best among the methods tested for our simulation task, which demonstrates the benefits of specialization for simulation.

• We contextualize these positive results with evidence of systematic inaccuracies in even the best-performing simulations, thus cautioning against the use of LLMs, specialized or not, for simulating survey response distributions today.<sup>1</sup>

## 2 Related Work

LLM Simulations Collecting human response data is one of the most challenging and costly aspects of social science research (Argyle et al., 2023; Hewitt et al., 2024). Consequently, much prior work has investigated the extent to which LLMs can accurately simulate human responses in surveys and experimental settings. Most prominently, Argyle et al. (2023), Horton (2023) and Aher et al.

(2023) all found evidence of LLMs providing reasonably accurate group-level simulations in behavioral science and economics experiments as well as for US political surveys. Some follow-up work has highlighted biases and conceptual challenges in such simulations (Bisbee et al., 2023; Bail, 2024; Park et al., 2024; Kozlowski and Evans, 2024). Other work has explored prompting strategies and frameworks for improving simulation accuracy (Kwok et al., 2024; Manning et al., 2024; Sun et al., 2024). Relatedly, several works have used survey questionnaires, including the WVS used in this paper, with the goal of evaluating values and opinions reflected in LLMs rather than simulating human survey responses (Benkler et al., 2023; Arora et al., 2023; Cao et al., 2023; AlKhamissi et al., 2024; Zhao et al., 2024; Wright et al., 2024, inter alia). In contrast to these efforts, our focus is on specializing LLMs to simulate group-level survey response distribution, which could aid in survey data collection. While we do measure the performance of LLMs out of the box (zero-shot), our main goal is to investigate the extent to which we can improve LLM performance on simulating group-level survey response distributions through fine-tuning, and explore their potential as specialized tools for social science research.

Distribution Simulation as Calibration. Calibration is aligning classifier predictive probabilities with the classification uncertainty. While most work focuses on majority class accuracy (Li et al., 2024; He et al., 2024), this is problematic when human label variation is substantial (Baan et al., 2022, 2024), and human calibration should consider the full human judgment distribution. Our task can therefore be viewed as human calibration for multiple-choice surveys, as opposed to many previous studies, which focused on accuracy measured against the majority answer (Arora et al., 2023; Cao et al., 2023; AlKhamissi et al., 2024).

## 3 Cultural Survey Simulation Dataset

## 3.1 Data Source

We use the 2017-2022 wave of the World Values Survey (WVS)<sup>2</sup> to construct our main simulation dataset. WVS was conducted across 66 countries with over 80,000 respondents. This extensive survey captures societal attitudes on various cultural dimensions, includingfamily, regional values, education, moral principles, corruption, accountability, etc. Our analysis includes all countries with more than 1,000 respondents to ensure robust crosscultural representation, resulting in a set of 65 countries (Northern Ireland did not qualify).

<table><tr><td>Instruction</td><td>How would someone from Andorra an- swer the following question:</td></tr><tr><td>Input</td><td>How interested would you say you are in politics? Here are the options:</td></tr><tr><td>Options</td><td>(A) Very interested (B) Somewhat interested (C) Not very interested</td></tr><tr><td>Format</td><td>If had to select one of the options, my answer would be ([A/B/...])</td></tr><tr><td>Options Distribution</td><td>(A) 15.16% (B) 29.02% (C) 28.31% (D) 27.51%</td></tr></table>

Table 1: Example entry from our formatted WVS dataset for the country of Andorra.

For our analysis, we use the original questions and answers, excluding validity-check options such as “not applicable” and “refuse to answer”, given their infrequent occurrence in human-collected responses. We conduct experiments using the English and Chinese versions of the datasets obtained from the official source, enabling the analysis of cross-linguistic differences in this task<sup>3</sup>.

## 3.2 Prompt Settings

We preserve the original questions and response options, adhering to the GlobalOpinionQA template for consistency (Durmus et al., 2023). As shown in Table 1, the model input consists of fields for instruction, input, options, and format, while the target for alignment is the distribution of the options. Note that theformat field is used to restrict the vocabulary of the first token to valid options.

## 3.3 Dataset Split

We use the first 259 questions of the WVS to construct our dataset, excluding demographics and notes for interviewers. We divide them into three parts based on topics: $Q _ { 1 }$ (questions 1-163), $Q _ { 2 }$ (questions 164-198), and $Q _ { 3 }$ (questions 199-259). Additionally, we divide the countries into three groups to ensure they are challenging to generalize between: $C _ { 1 }$ (all countries that are not included in the following two sets), $C _ { 2 }$ (the 8 surveyed countries that are in $\mathrm { { A f r i c a } } ^ { 4 } )$ , and $C _ { 3 }$ (medium-GDP countries sampled from each continent<sup>5</sup>).

<table><tr><td></td><td>Countries Description</td><td>N</td></tr><tr><td> $C _ { 1 }$ </td><td>All WVS countries not in  $C _ { 2 }$  or  $C _ { 3 }$ </td><td>46</td></tr><tr><td> $C _ { 2 }$ </td><td>African countries</td><td>8</td></tr><tr><td> $C _ { 3 }$ </td><td>Medium-GDP countries</td><td>11</td></tr><tr><td>Questions Description</td><td></td><td>N</td></tr><tr><td> $Q _ { 1 }$ </td><td>All WVS questions not in  $Q _ { 2 }$  or  $Q _ { 3 }$ </td><td>150</td></tr><tr><td> $Q _ { 2 }$ </td><td>Q&#x27;s about religious and ethical values</td><td>35</td></tr><tr><td> $Q _ { 3 }$ </td><td> $\mathrm { Q } \mathrm { \ ' } \mathrm { s }$  about political interest and culture</td><td>59</td></tr></table>

Table 2: Country and question splits that we use in our experiments with WVS data. Splits seen during training are highlighted in bold. For additional descriptive statistics on the dataset, see Appendix B.

We split training, validation, and test sets for the aforementioned questions and country subsets. The split and statistical information of the dataset are presented in Table 2 and Table 3 respectively. The test set comprises five subsets, designed to evaluate the performance of models in answering unseen value questions, unseen regional countries, and representative medium-GDP countries.

## 3.4 Unseen Survey Dataset

To evaluate generalization to a completely unseen survey, we use an additional subset of GlobalOpinionQA (Durmus et al., 2023), the Pew Global Attitudes Survey (Pew), which maintains a similar format to the WVS but includes different cultural questions. We compile two sets of countries for this test set: $C _ { 1 } ^ { \prime }$ and $C _ { 3 }$ . For $C _ { 1 } ^ { \prime }$ , we sample ten countries from $C _ { 1 }$ to maintain geographical and GDP-level diversity and then use the GlobalOpinionQA data specifically for these countries for evaluation (see Appendix A). For $C _ { 3 } ,$ we include the same medium-GDP countries as in $C _ { 3 }$

## 4 Methodology

To address the challenges of simulating culturally diverse survey responses, we introduce a framework for first-token alignment fine-tuning for distribution prediction, designed to improve generalization across populations and survey questions.

<table><tr><td>Split</td><td></td><td>Train |Valid</td><td colspan="5">Test</td></tr><tr><td>Countries Questions</td><td> $C _ { 1 }$   $Q _ { 1 }$ </td><td> $C _ { 1 }$   $Q _ { 2 }$ </td><td> $C _ { 1 }$  Q3</td><td> $C _ { 2 }$   $Q _ { 1 }$ </td><td> $C _ { 2 }$   $Q _ { 3 }$ </td><td> $C _ { 3 }$   $Q _ { 1 }$ </td><td> $C _ { 3 }$   $Q _ { 3 }$ </td></tr><tr><td>Entries</td><td></td><td></td><td>6,841 | 1,586 | 2,719</td><td>1,179 471</td><td></td><td> $1 { , } 6 4 4$ </td><td>660</td></tr></table>

Table 3: WVS dataset statistics across the country (C) and question (Q) splits we use in our experiments. Splits seen during training are highlighted in bold. Number of entries is not necessarily $C \times Q$ because some entries are missing from survey results.

## 4.1 Probability Distribution Simulation

Unlike most existing studies that directly prompt LLMs with multiple-choice questions to assess their cultural knowledge or behavior (Arora et al., 2023; Cao et al., 2023; AlKhamissi et al., 2024), we propose a novel task that focuses on simulating the distribution of response options for given questions rather than predicting single answers.

Specifically, let Q denote a multiple-choice question and $O = \{ o _ { 1 } , o _ { 2 } , . . . , o _ { n } \}$ be the corresponding set of response options, where n is the total number of options, which can vary between questions. The objective is to simulate the option distribution $P ( O | Q )$ to match human response distribution from a particular group (e.g., country). Consequently, models are evaluated by comparing the alignment of observed vs. predicted distributions rather than focusing on majority response options.

## 4.2 First-Token Probability Alignment

Using the dataset introduced in §3, we present firsttoken alignment fine-tuning, to align model outputs with the observed response distributions of specific population groups (e.g., countries).

The processed question $Q$ is used as input into LLMs. The model outputs logits $\{ z _ { 1 } , z _ { 2 } , \dots , z _ { n } \}$ for the first token of each question’s corresponding options O. The probability distribution for the first token is obtained by applying the softmax function to normalize the indexed option logits:

$$
P _ { \mathrm { L L M } } ( o _ { i } | Q ) = { \frac { e ^ { z _ { i } } } { \sum _ { j = 1 } ^ { n } e ^ { z _ { j } } } }
$$

For the training optimization objective, we employ Kullback-Leibler Divergence loss $( \mathrm { L o s s } _ { \mathrm { K L } } )$ to align the LLM’s first-token probability distribution with the human response distribution:

$$
\mathrm { L o s s } _ { \mathrm { K L } } = \sum _ { i = 1 } ^ { n } P _ { \mathrm { h u m a n } } \left( o _ { i } | Q \right) \log \left( \frac { P _ { \mathrm { h u m a n } } \left( o _ { i } | Q \right) } { P _ { \mathrm { L L M } } \left( o _ { i } | Q \right) } \right)
$$

where $P _ { \mathrm { h u m a n } } ( o _ { i } | Q )$ is the probability of option $o _ { i }$ based on human survey data, and $P _ { \mathrm { L L M } } ( o _ { i } | Q )$ is the probability output by the LLM.

To improve the efficiency of the fine-tuning process, we implement Low-Rank Adaptation (LoRA; Hu et al., 2022), a parameter-efficient method specifically designed for optimizing LLMs.

## 5 Experimental Setup

## 5.1 Models

We fine-tune seven models across three model families using our processed dataset: Vicuna1.5 (Chiang et al., 2023) in its 7B and 13B parameter versions, Llama3 (AI@Meta, 2024) in its 8B Base and Instruct versions, and Deepseek-Distilled-Qwen (Guo et al., 2025) in 7B, 14B, and 32B. Vicuna1.5 is a version of Llama2 (Touvron et al., 2023) fine-tuned on user conversations with Chat-GPT, whereas Llama3 is a stronger model. As Vicuna1.5 is less powerful than recent LLMs, it is chosen to evaluate the effect of fine-tuning as an equalizer despite zero-shot performance differences. DeepSeek is a state-of-the-art model series known for its strong performance across diverse benchmarks. We use the DeepSeek-Distilled-Qwen models, which are distilled from the DeepSeek-R1, the current state-of-the-art open weights model. For further details on all models and our inference setup, see Appendix C.

## 5.2 Baselines

In the main body of this paper, we compare our fine-tuning method (FT) to a zero-shot prompting (ZS) baseline, which is the default method explored in prior work. ZS involves directly querying the models with the country context and questions. As an additional control setting, for both ZS and FT, we replace countries in the queries with other countries randomly selected from among the full set of countries in the WVS. This approach is designed to investigate the sensitivity of the LLMs to the specific country given in the context vs. the prior distributions of response options. We denote this control setting as “[ctrl]”. In Appendix E, we show additional baselines such as K-Nearest Neighbors, which generally perform worse than FT.

## 5.3 Metrics

To measure the alignment of predicted response distributions with country-level reference distributions, we adopt two metrics for evaluation: i) 1-

<table><tr><td rowspan="2">Model</td><td rowspan="2">Methods</td><td colspan="6">(1–JSD) ↑</td><td colspan="6">EMD↓</td></tr><tr><td> $C _ { 1 } { - } Q _ { 3 }$ </td><td> $C _ { 2 } – Q _ { 1 }$ </td><td> $C _ { 2 } { - } Q _ { 3 }$ </td><td> $C _ { 3 } – Q _ { 1 }$ </td><td> $C _ { 3 } – Q _ { 3 }$ </td><td>Avg.</td><td> $C _ { 1 } { - } Q _ { 3 }$ </td><td> $C _ { 2 } – Q _ { 1 }$ </td><td> $C _ { 2 } { - } Q _ { 3 }$ </td><td> $C _ { 3 } – Q _ { 1 }$ </td><td> $C _ { 3 } – Q _ { 3 }$ </td><td> $A \nu g .$ </td></tr><tr><td rowspan="4">Vicuna1.5-7B</td><td>ZS [ctrl] ZS</td><td>0.732</td><td>0.754</td><td>0.748</td><td>0.761</td><td>0.747</td><td>0.748</td><td>0.095</td><td>0.108</td><td>0.095</td><td>0.106</td><td>0.086</td><td>0.098</td></tr><tr><td></td><td>0.732</td><td>0.754</td><td>0.749</td><td>0.761</td><td>0.749</td><td>0.749</td><td>0.095</td><td>0.107</td><td>0.096</td><td>0.106</td><td>0.085</td><td>0.098</td></tr><tr><td>FT [ctrl]</td><td>0.754</td><td>0.829</td><td>0.754</td><td>0.842</td><td>0.765</td><td>0.789</td><td>0.089</td><td>0.078</td><td>0.092</td><td>0.072</td><td>0.080</td><td>0.082</td></tr><tr><td>FT</td><td>0.766</td><td>0.859</td><td>0.767</td><td>0.875</td><td>0.775</td><td>0.808</td><td>0.087</td><td>0.071</td><td>0.091</td><td>0.062</td><td>0.078</td><td>0.078</td></tr><tr><td rowspan="4">Vicuna1.5-13B</td><td>ZS [ctrl]</td><td>0.735</td><td>0.755</td><td>0.743</td><td>0.775</td><td>0.747</td><td>0.751</td><td>0.095</td><td>0.102</td><td>0.096</td><td>0.096</td><td>0.086</td><td>0.095</td></tr><tr><td>ZS</td><td>0.738</td><td>0.756</td><td>0.744</td><td>0.779</td><td>0.752</td><td>0.754</td><td>0.094</td><td>0.103</td><td>0.097</td><td>0.097</td><td>0.085</td><td>0.095</td></tr><tr><td>FT [ctrl]</td><td>0.760</td><td>0.820</td><td>0.761</td><td>0.830</td><td>0.762</td><td>0.787</td><td>0.083</td><td>0.080</td><td>0.085</td><td>0.074</td><td>0.077</td><td>0.080</td></tr><tr><td>FT</td><td>0.781</td><td>0.869</td><td>0.781</td><td>0.882</td><td>0.781</td><td>0.819</td><td>0.079</td><td>0.063</td><td>0.084</td><td>0.057</td><td>0.073</td><td>0.071</td></tr><tr><td rowspan="4">Llama3-8B-Base</td><td>ZS [ctrl]</td><td>0.748</td><td>0.766</td><td>0.757</td><td>0.779</td><td>0.768</td><td>0.764</td><td>0.097</td><td>0.114</td><td>0.093</td><td>0.109</td><td>0.088</td><td>0.100</td></tr><tr><td>ZS</td><td>0.749</td><td>0.768</td><td>0.759</td><td>0.781</td><td>0.770</td><td>0.765</td><td>0.097</td><td>0.116</td><td>0.097</td><td>0.109</td><td>0.087</td><td>0.101</td></tr><tr><td>FT [ctrl]</td><td>0.756</td><td>0.823</td><td>0.751</td><td>0.837</td><td>0.770</td><td>0.787</td><td>0.084</td><td>0.082</td><td>0.087</td><td>0.073</td><td>0.077</td><td>0.081</td></tr><tr><td>FT</td><td>0.770</td><td>0.858</td><td>0.773</td><td>0.877</td><td>0.781</td><td>0.812</td><td>0.081</td><td>0.073</td><td>0.087</td><td>0.061</td><td>0.074</td><td>0.075</td></tr><tr><td rowspan="4">Llama3-8B-Instruct</td><td>ZS [ctrl]</td><td>0.574</td><td>0.626</td><td>0.563</td><td>0.644</td><td>0.587</td><td>0.599</td><td>0.130</td><td>0.147</td><td>0.135</td><td>0.145</td><td>0.136</td><td>0.139</td></tr><tr><td>ZS</td><td>0.585</td><td>0.650</td><td>0.589</td><td>0.657</td><td>0.584</td><td>0.613</td><td>0.130</td><td>0.139</td><td>0.126</td><td>0.145</td><td>0.141</td><td>0.136</td></tr><tr><td>FT [ctrl]</td><td>0.756</td><td>0.826</td><td>0.751</td><td>0.833</td><td>0.764</td><td>0.786</td><td>0.077</td><td>0.077</td><td>0.082</td><td>0.073</td><td>0.071</td><td>0.076</td></tr><tr><td>FT</td><td>0.777</td><td>0.881</td><td>0.783</td><td>0.890</td><td>0.784</td><td>0.823</td><td>0.073</td><td>0.060</td><td>0.080</td><td>0.053</td><td>0.067</td><td>0.067</td></tr><tr><td rowspan="4">Distil-Qwen-7B</td><td>ZS[ctrl]</td><td>0.586</td><td>0.641</td><td>0.642</td><td>0.698</td><td>0.682</td><td>0.650</td><td>0.084</td><td>0.096</td><td>0.138</td><td>0.100</td><td>0.123</td><td>0.108</td></tr><tr><td>ZS</td><td>0.583</td><td>0.645</td><td>0.639</td><td>0.701</td><td>0.671</td><td>0.648</td><td>0.084</td><td>0.095</td><td>0.138</td><td>0.100</td><td>0.126</td><td>0.109</td></tr><tr><td>FT[ctrl]</td><td>0.747</td><td>0.764</td><td>0.791</td><td>0.811</td><td>0.817</td><td>0.786</td><td>0.075</td><td>0.079</td><td>0.080</td><td>0.078</td><td>0.110</td><td>0.085</td></tr><tr><td>FT</td><td>0.756</td><td>0.781</td><td>0.803</td><td>0.833</td><td>0.834</td><td>0.801</td><td>0.073</td><td>0.076</td><td>0.077</td><td>0.072</td><td>0.107</td><td>0.081</td></tr><tr><td rowspan="4">Distil-Qwen-14B</td><td>ZS[ctrl]</td><td>0.650</td><td>0.704</td><td>0.749</td><td>0.711</td><td>0.743</td><td>0.712</td><td>0.088</td><td>0.082</td><td>0.089</td><td>0.115</td><td>0.141</td><td>0.103</td></tr><tr><td>ZS</td><td>0.654</td><td>0.716</td><td>0.756</td><td>0.731</td><td>0.746</td><td>0.721</td><td>0.088</td><td>0.081</td><td>0.086</td><td>0.110</td><td>0.136</td><td>0.100</td></tr><tr><td>FT[ctrl]</td><td>0.751</td><td>0.784</td><td>0.799</td><td>0.807</td><td>0.833</td><td>0.795</td><td>0.066</td><td>0.074</td><td>0.084</td><td>0.083</td><td>0.103</td><td>0.082</td></tr><tr><td>FT</td><td>0.777</td><td>0.816</td><td>0.827</td><td>0.851</td><td>0.861</td><td>0.826</td><td>0.061</td><td>0.068</td><td>0.077</td><td>0.070</td><td>0.099</td><td>0.075</td></tr><tr><td rowspan="4">Distil-Qwen-32B</td><td>ZS[ctrl]</td><td>0.649</td><td>0.671</td><td>0.721</td><td>0.674</td><td>0.743</td><td>0.691</td><td>0.089</td><td>0.105</td><td>0.099</td><td>0.131</td><td>0.126</td><td>0.110</td></tr><tr><td>ZS</td><td>0.658</td><td>0.683</td><td>0.719</td><td>0.695</td><td>0.749</td><td>0.701</td><td>0.088</td><td>0.105</td><td>0.101</td><td>0.129</td><td>0.125</td><td>0.110</td></tr><tr><td>FT[ctrl]</td><td>0.776</td><td>0.789</td><td>0.799</td><td>0.794</td><td>0.821</td><td>0.796</td><td>0.066</td><td>0.071</td><td>0.074</td><td>0.087</td><td>0.111</td><td>0.082</td></tr><tr><td>FT</td><td>0.800</td><td>0.821</td><td>0.815</td><td>0.830</td><td>0.846</td><td>0.822</td><td>0.062</td><td>0.067</td><td>0.073</td><td>0.078</td><td>0.110</td><td>0.078</td></tr></table>

Table 4: Main results for predicting country-level survey response distributions on the WVS data. We test all models with zero-shot prompting (ZS) and our proposed fine-tuning approach (FT). [ctrl] indicates a control setup, where we randomly replace countries in test prompts with other countries, to evaluate country context sensitivity. We report Jensen-Shannon Divergence (1 JSD , ) and Earth Mover Distance (EMD, ).

JSD, where JSD is Jensen-Shannon divergence, also employed by Durmus et al. (2023), is a symmetric measure of the similarity between two probability distributions, with higher values indicating greater similarity; ii) Earth Mover Distance (EMD; Rubner et al., 1998), also known as the Wasserstein distance, quantifies the minimum amount of work required to transform one distribution into another, with lower values indicating greater similarity in distribution. Both the 1-JSD and EMD metrics range from 0 to 1.

## 6 Results

We investigate two primary research questions:

RQ1 How effectively does the proposed alignment method improve the distribution simulation of the model on unseen countries and questions?

RQ2 What is necessary to perform well on the task—how much is dependent on modeling the prior distribution, and how much on context sensitivity?

We present comprehensive experimental results to address the two research questions.

## 6.1 RQ1: Generalization Performance

To address RQ1 (how effectively FT improves distribution simulation on unseen countries and questions), we train the selected models using our proposed simulation methods and evaluate their performance on unseen countries and questions. Table 4 presents the evaluation scores across models of varying sizes and types.

Zero-shot [ZS] vs. Fine-tuning [FT]. Across all model sizes and types, we observe that zero-shot prompting (ZS) consistently yields worse scores compared to fine-tuned models (FT), indicating that while ZS is capable of addressing unseen countries and questions, it lacks the adaptability needed for effective distribution simulation. In contrast, fine-tuned models show improved performance, with higher 1 JSD and lower EMD scores (e.g., a 34.3% 1 JSD increase and 0.069 EMD decrease for Llama3-8B-Instruct Avg.), demonstrating their enhanced ability to align with real-world distributions. This suggests that fine-tuning enables models to internalize more detailed patterns and relationships, making them more effective for simulation.

Unseen Countries vs. Unseen Questions. The generalization capabilities are revealed by evaluation on unseen attributes. Across all models and settings, unseen questions $\left( Q _ { 3 } \right)$ tend to present a greater challenge than unseen countries $( C _ { 2 } \thinspace \mathrm { o r } C _ { 3 } ) .$ as indicated by slightly worse scores for unseen questions (e.g., 0.781 vs. 0.886 1 JSD for Llama3- Instruct). This suggests that while the models are reasonably robust in handling new country distributions, they struggle more with generating accurate distributions for questions that were not encountered during training.

Comparison Across Models. Our method demonstrates consistent performance improvements across three representative model families and various model sizes, highlighting the effectiveness of our proposed first-token alignment approach. Specifically, while there are notable performance differences in the zero-shot setting—for example, Llama3-Base achieves a higher 1 JSD value (0.765) compared to Llama3-Instruct (0.613), suggesting that the base model better predicts the option token distribution—our fine-tuning procedure significantly bridges this gap. After finetuning, Llama3-Instruct not only closes the initial performance disparity but even surpasses Llama3- Base (34.3% 1 JSD increase vs. 6.1% increase). Similarly, experiments with Distilled Qwen models of different sizes reveal no clear scaling trends. Moreover, although Vicuna1.5 is generally considered weaker compared to Llama and Qwen, it surprisingly delivers similarly competitive results on this task. Overall, analysis across all models further revealing the finding that, regardless of the starting model, our fine-tuning approach consistently produces models with similar strong performance on our task.

Correlation Between First-Token Distribution and Response Accuracy. Beyond the alignment of first-token distributions, we further evaluate of the mode accuracy of models in responding to questionnaire items following fine-tuning. We consider the options with the highest probabilities as (single, argmax) predictions and calculate accuracy against the survey majority choice per country (Arora et al., 2023; Cao et al., 2023; AlKhamissi et al., 2024). As depicted in Figure 2, our findings reveal a significant increase in accuracy across all models and test subsets, highlighting the effectiveness of the fine-tuning process. Notably, performance improvements are particularly significant in unseen countries, as demonstrated by $C _ { 2 } – Q _ { 1 }$ and $C _ { 3 } – Q _ { 1 }$ subsets. These results suggest that our proposed method not only improves the simulation of option distributions but also strengthens the models’ alignment with the correct responses, underscoring the interdependence of distribution alignment and answer accuracy.

![](images/cfcc66962bc1b40d1e507c2bfe621b9fcb7dba537526888524feee95c872a9ca.jpg)  
Figure 2: Option prediction accuracy for cultural questions using the Llama3-8B-Instruct. The final option is simulated by selecting the option with the highest probabilities, compared against human majority choice.

## 6.2 RQ2: Variation Sensitivity

To address RQ2 (contribution of modeling the prior distribution vs. context sensitivity), we compare the control setting (with countries randomly replaced), analyze diversity changes of models, and explore shifts in response accuracy for unseen countries.

ZS[ctrl] vs. FT[ctrl]. In the control setting, ZS[ctrl] shows a smaller performance drop than ZS, while FT[ctrl] sees a larger decline, with a 16.7% avg. (1 JSD) drop between FT[ctrl] and FT across seven models, compared to 3.7% for ZS[ctrl] and ZS. This indicates that fine-tuned models (FT) are more sensitive to the country context and not just the prior distribution of responses (which FT[ctrl] is trained to simulate), suggesting they have become more specialized in capturing cultural nuances during training. In contrast, the smaller gap between ZS and ZS[ctrl] implies that zero-shot models maintain a more generalized understanding of cultural contexts, making them less affected by random permutations of country data. This difference highlights the fine-tuned models’ improved capability in simulating response distributions for specific countries.

Country Diversity in Model Outputs. We define diversity as the divergence across countries of response distributions given the same question.

![](images/c1223df12c6e29bc1821e5c8254d86979eddd1ad2a19d0f613c1a4873609a81e.jpg)  
(a) Llama-Base (ZS)

![](images/fb0498a18a1c78583c46b946b8f54653c9d4cae8f569c54e9a131e788c198de6.jpg)  
(b) Llama-Base (FT)

![](images/32d72409ec60c6f9adc2b586e4a1a3749ff21aba0d201bdc901e798b4f3d4f0a.jpg)  
(e) Country Accuracy on C2-Q1

![](images/1fe9a532dad61c762acf5099d57929ba1f9b140aae7610fcdea54e69c71eb242.jpg)  
(c) Llama-Instruct (ZS)

![](images/af77f925980bc70c73aa2745920c831d2240ca0a537a24c6a1f4733f4ea2a321.jpg)  
(d) Llama-Instruct (FT)

![](images/a14fa8f697d63d56fb1e395ea8c68b048579c25128699a7fb5c3cd95ad4f2b70.jpg)  
(f) Country Accuracy on $C 2 \mathrm { - } \mathrm { Q } 3$

Figure 3: Model Diversity and Country Accuracy Analysis. (a)-(d) denotes the comparison of 1 JSD scores across countries for specific questions $( C _ { 2 } \mathrm { - } Q _ { 3 }$ and $C _ { 3 } – Q _ { 3 } )$ . The blue-shaded area represents diversity changes, with the lower boundary indicating the mean of survey response scores and the upper boundary representing the mean of model outputs. (e)-(f) denotes the accuracy of options on African countries in both $C _ { 2 } – Q _ { 1 }$ and $C _ { 2 } – Q _ { 3 }$
<table><tr><td rowspan="2">Base Model</td><td rowspan="2">Methods</td><td colspan="6">(1–JSD)↑</td><td colspan="6">EMD↓</td></tr><tr><td> $C _ { 1 } { - } Q _ { 3 }$ </td><td> $C _ { 2 } – Q _ { 1 }$ </td><td> $C _ { 2 } { - } Q _ { 3 }$ </td><td> $C _ { 3 } – Q _ { 1 }$ </td><td> $C _ { 3 } – Q _ { 3 }$ </td><td>Avg.</td><td> $C _ { 1 } { - } Q _ { 3 }$ </td><td> $C _ { 2 } – Q _ { 1 }$ </td><td> $C _ { 2 } – Q _ { 3 }$ </td><td> $C _ { 3 } – Q _ { 1 }$ </td><td> $C _ { 3 } – Q _ { 3 }$ </td><td>Avg.</td></tr><tr><td rowspan="2">Llama3-8B-Instruct</td><td>ZS</td><td>0.603</td><td>0.654</td><td>0.601</td><td>0.659</td><td>0.613</td><td>0.626</td><td>0.125</td><td>0.140</td><td>0.131</td><td>0.145</td><td>0.132</td><td>0.135</td></tr><tr><td>FT</td><td>0.777</td><td>0.852</td><td>0.789</td><td>0.870</td><td>0.791</td><td>0.816</td><td>0.081</td><td>0.078</td><td>0.087</td><td>0.065</td><td>0.073</td><td>0.077</td></tr><tr><td rowspan="2">Distil-Qwen-7B</td><td>ZS</td><td>0.605</td><td>0.706</td><td>0.712</td><td>0.696</td><td>0.693</td><td>0.682</td><td>0.084</td><td>0.091</td><td>0.086</td><td>0.087</td><td>0.190</td><td>0.108</td></tr><tr><td>FT</td><td>0.764</td><td>0.779</td><td>0.771</td><td>0.791</td><td>0.851</td><td>0.791</td><td>0.067</td><td>0.085</td><td>0.081</td><td>0.082</td><td>0.130</td><td>0.089</td></tr></table>

Table 5: Model Performance on Chinese World Values Survey. Performance comparison between zero-shot prompting (ZS) and supervised fine-tuning (SFT) on two model families.

To assess whether FT can enhance model output diversity, we calculate 1 JSD for Llama3-Base and Llama3-Instruct (ZS and FT) output across countries for each question. A lower 1 JSD score suggests greater diversity in responses between countries, whereas a higher score indicates greater similarity in distributions. Results are shown in Figure 3, where the scores of survey responses are compared with those of the model predictions.

This visualization provides several interesting insights. Firstly, the outputs of Base models exhibit a high degree of uniformity across countries, indicating a limited sensitivity to national variations when addressing cultural values questions. After fine-tuning, there is a slight reduction in 1 JSD, suggesting an enhancement in the responsiveness to diverse cultural contexts. Secondly, although the Instruct model, having undergone alignment finetuning, initially produces a more varied distribution of answers, this diversity diminishes following distribution simulation fine-tuning. Thirdly, we observe no consistent correlation between the diversity of generated responses and the accuracy of simulated distributions. Lastly, the post-finetuning diversity of responses from both models converges, indicating that our fine-tuning approach improves the sensitivity to national differences.

Unseen Country Shifts. We visualize the option accuracy of African countries as observation objects in Figure 3e-3f. For seen questions, we find that all models show a relatively high performance across most countries, with Ethiopia and Nigeria displaying the highest accuracies close to 80%. For unseen questions, the performance decreases greatly compared to seen questions, particularly in countries like Egypt and Tunisia. Besides, the Instruct models maintain a higher level of accuracy relative to the Base models in most cases, showing the relative robustness of Instruct models in both seen and unseen scenarios across the African countries, which is consistent with results in Figure 2.

<table><tr><td rowspan="2">Methods</td><td colspan="2"> $C _ { 1 } ^ { \prime }$ </td><td colspan="2"> $C _ { 3 }$ </td></tr><tr><td>(1−JSD)↑</td><td>ACC↑</td><td>(1–JSD)↑</td><td>ACC ↑</td></tr><tr><td>Vicuna1.5 (ZS)</td><td>0.690</td><td>0.360</td><td>0.668</td><td>0.346</td></tr><tr><td>Vicuna1.5 (FT)</td><td>0.725</td><td>0.442</td><td>0.709</td><td>0.456</td></tr><tr><td>Llama3 (ZS)</td><td>0.617</td><td>0.472</td><td>0.613</td><td>0.446</td></tr><tr><td>Llama3 (FT)</td><td>0.767</td><td>0.562</td><td>0.755</td><td>0.568</td></tr></table>

Table 6: Evaluation results on GlobalOpinionQA Pew dataset for Llama3-Instruct and Vicuna1.5-7B. (1 JSD) and option accuracy scores are reported.

## 6.3 Robustness Analysis

In this section, we explore the robustness of the models on survey language and unseen survey.

Impact of Survey Language on Results. We fine-tuned the Llama3-8B-Instruct and Distil-Qwen-7B models on the official Chinese translation dataset, and the results are presented in Table 5. While the performance of both models in Chinese is marginally lower than in English both on (1-JSD) and EMD metrics, the difference is not significant, suggesting that current LLMs exhibit limited sensitivity to language differences in this task. Besides, while the Distilled-Qwen model demonstrates a better performance than Llama3 on most benchmarks, it does not outperform Llama3 in this task.

Generalization to a New Survey. We use Pew introduced in §3.4 to test the generalization of our fine-tuned models. Table 6 presents the results in both 1 JSD scores and accuracy. Notably, all metric scores show significant improvements for both models after fine-tuning. Additionally, across both $C _ { 1 } ^ { \prime }$ and $C _ { 3 } .$ , Llama3 outperforms Vicuna1.5 in terms of accuracy, particularly in the fine-tuned setting, where Llama3 (FT) achieves 19.1% and 27.4% improvement for two datasets, respectively. The consistent improvements prove its capability to generalize well to unseen surveys.

## 6.4 Ablation Studies

We conduct ablation studies to analyze the impact of the training loss function and option ordering.

Loss Function. As shown in Table 7, KL Loss is the most effective loss function for our task. However, Wasserstein (WA) Loss Jensen-Shannon (JS), and Cross-Entropy (CE) Loss also improve over zero-shot prompting.

<table><tr><td>Methods</td><td> $C _ { 1 } { - } Q _ { 3 }$ </td><td> $C _ { 2 } { - } Q _ { 1 }$ </td><td> $C _ { 2 } { - } Q _ { 3 }$ </td><td> $C _ { 3 ^ { - } } Q _ { 1 }$ </td><td> $C _ { 3 } – Q _ { 3 }$ </td></tr><tr><td>KL (Orig)</td><td>0.777</td><td>0.881</td><td>0.783</td><td>0.890</td><td>0.784</td></tr><tr><td>WA Loss</td><td>0.733</td><td>0.774</td><td>0.744</td><td>0.782</td><td>0.749</td></tr><tr><td>JS Loss</td><td>0.745</td><td>0.790</td><td>0.756</td><td>0.799</td><td>0.763</td></tr><tr><td>CE Loss</td><td>0.746</td><td>0.809</td><td>0.772</td><td>0.807</td><td>0.756</td></tr><tr><td>Shuffled</td><td>0.753</td><td>0.820</td><td>0.761</td><td>0.815</td><td>0.761</td></tr></table>

Table 7: Results of our ablation studies. We compare different loss functions (WA, JS and CE) to our KL loss setup. We also evaluate on a test set with shuffled option orders. All results are for Llama3-8B-Instruct.

Option Ordering. Dominguez-Olmedo et al. (2023) observed an A-bias effect, where models tend to disproportionately select the answer choice labeled “A”. To assess this bias, we re-evaluate our fine-tuned model on the same dataset where the answer options are shuffled. As shown in Table 7 (“Shuffled”), there is a reduction in performance, but it is small compared to the effect of fine-tuning or model choice. This indicates that the option ordering is not a major concern in our experiments.

## 7 Conclusion

In this paper, we explored the task of specialising LLMs to simulate survey response distributions across diverse countries and questions. For this task, we devised a fine-tuning method based on first-token probabilities. Our experiments demonstrate that fine-tuning models substantially improves response simulation prediction compared to zero-shot models, for both seen and unseen countries and questions. Further, fine-tuned models also show improved generalization to an entirely new survey dataset. Despite these improvements, our results also highlight systematic limitations of the models, particularly when simulating responses to unseen questions. We also observed that the models, whether fine-tuned or not, were less diverse in their predictions compared to the human survey response data, raising questions about their utility.

While our results provide clear evidence for the benefits of specializing LLMs for survey simulation tasks, they also underscore the need for caution when using LLMs for this task, as even the best-performing models exhibited systematic inaccuracies, especially in culturally diverse contexts.

## Limitations

While we proved the effectiveness of our proposed method, several limitations remain in our work.

Scope. Our trained models are highly specialized and can only be used for the specific task of predicting the distribution of answers to a given survey question from a specified human population. Future work will investigate whether the fine-tuning approach also results in less biased or more aligned models in general-purpose applications, but this cannot be claimed only based on our study.

Language and Countries Coverage. Our study only uses English prompts for experiments and uses countries to represent specific cultures, consistent with existing studies (Cao et al., 2023; AlKhamissi et al., 2024). While this approach offers some valuable insights, it may limit the applicability of our findings to non-English LLMs and diverse finegrained cultural contexts. We hope that future research could benefit from exploring broader languages and countries to enhance the robustness of the proposed framework.

Model Choice. Due to computational resource consideration, we did not fine-tune LLMs with more than 32B parameters and instead selected a limited number of models for validation. Despite this limitation, in future work, we aim to cover a range of powerful models of varying sizes, which will allow us to uncover interesting observations regarding their performance. We believe that the insights we draw will still contribute to future research, encouraging further exploration of larger models to better understand their capabilities in simulating cultural diversity.

## Ethics Statement

This research adheres to strict ethical standards, ensuring that all datasets, large language models, and prompt settings used are sourced from openaccess repositories and are properly licensed to their original creators.

While our proposed framework does not involve any inherently risky operations, we acknowledge that the deployment of LLMs carries inevitable potential ethical implications. Therefore, users interacting with our models are strongly encouraged to consider safety and ethical factors, remaining aware of the potential risks and harms that may arise from misuse or misinterpretation of the generated content. Through this work, we aim to contribute positively to a better understanding of cultural diversities and promote responsible practices in the simulation of cultural contexts.

## Acknowledgments

We thank anonymous reviewers for their valuable comments. This research was co-funded by a DFF Sapere Aude research leader grant under grant agreement No 0171-00034B, and supported by the Pioneer Centre for AI, DNRF grant number P1. Yong Cao was supported by a VolkswagenStiftung Momentum grant.

## References

Gati V Aher, Rosa I Arriaga, and Adam Tauman Kalai. 2023. Using large language models to simulate multiple humans and replicate human subject studies. In International Conference on Machine Learning, pages 337–371. PMLR.

AI@Meta. 2024. Llama 3 model card.

Badr AlKhamissi, Muhammad ElNokrashy, Mai Alkhamissi, and Mona Diab. 2024. Investigating cultural alignment of large language models. In Proceedings ofthe 62nd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 12404–12422. Association for Computational Linguistics.

Lisa P Argyle, Ethan C Busby, Nancy Fulda, Joshua R Gubler, Christopher Rytting, and David Wingate. 2023. Out of one, many: Using language models to simulate human samples. Political Analysis, 31(3):337–351.

Arnav Arora, Lucie-aimée Kaffee, and Isabelle Augenstein. 2023. Probing pre-trained language models for cross-cultural differences in values. In Proceedings ofthe First Workshop on Cross-Cultural Considerations in NLP (C3NLP), pages 114–130. Association for Computational Linguistics.

Joris Baan, Wilker Aziz, Barbara Plank, and Raquel Fernandez. 2022. Stop measuring calibration when humans disagree. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 1892–1915. Association for Computational Linguistics.

Joris Baan, Raquel Fernández, Barbara Plank, and Wilker Aziz. 2024. Interpreting predictive probabilities: Model confidence or human label variation? In Proceedings ofthe 18th Conference ofthe European Chapter ofthe Associationfor Computational Linguistics (Volume 2: Short Papers), pages 268–277. Association for Computational Linguistics.

Christopher A Bail. 2024. Can generative ai improve social science? Proceedings of the National Academy ofSciences, 121(21):e2314021121.

Noam Benkler, Drisana Mosaphir, Scott Friedman, Andrew Smart, and Sonja Schmer-Galunder. 2023. Assessing LLMs for moral value pluralism. arXiv preprint arXiv:2312.10075.

James Bisbee, Joshua D Clinton, Cassy Dorff, Brenton Kenkel, and Jennifer M Larson. 2023. Synthetic replacements for human survey data? the perils of large language models. Political Analysis, pages 1– 16.

Yong Cao, Li Zhou, Seolhwa Lee, Laura Cabello, Min Chen, and Daniel Hershcovich. 2023. Assessing cross-cultural alignment between ChatGPT and human societies: An empirical study. In Proceedings ofthe First Workshop on Cross-Cultural Considerations in NLP (C3NLP), pages 53–67. Association for Computational Linguistics.

Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. 2023. Vicuna: An opensource chatbot impressing GPT-4 with 90%\* Chat-GPT quality.

Ricardo Dominguez-Olmedo, Moritz Hardt, and Celestine Mendler-Dünner. 2023. Questioning the survey responses of large language models. arXiv preprint arXiv:2306.07951.

Esin Durmus, Karina Nyugen, Thomas I Liao, Nicholas Schiefer, Amanda Askell, Anton Bakhtin, Carol Chen, Zac Hatfield-Dodds, Danny Hernandez, Nicholas Joseph, et al. 2023. Towards measuring the representation of subjective global opinions in language models. arXiv preprint arXiv:2306.16388.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948.

C. Haerpfer, R. Inglehart, A. Moreno, C. Welzel, K. Kizilova, J. Diez-Medrano, M. Lagos, P. Norris, E. Ponarin, and B. Puranen. 2022. World values survey wave 7 (2017-2022) cross-national data-set.

Guande He, Peng Cui, Jianfei Chen, Wenbo Hu, and Jun Zhu. 2024. Investigating uncertainty calibration of aligned language models under the multiple-choice setting.

Luke Hewitt, Ashwini Ashokkumar, Isaias Ghezae, and Robb Willer. 2024. Predicting results of social science experiments using large language models.

John J Horton. 2023. Large language models as simulated economic agents: What can we learn from homo silicus? Technical report, National Bureau of Economic Research.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations.

Austin Kozlowski and James Evans. 2024. Simulating subjects: The promise and peril of ai stand-ins for social agents and interactions.

Louis Kwok, Michal Bravansky, and Lewis D Griffin. 2024. Evaluating cultural adaptability of a large language model via simulation of synthetic personas. arXiv preprint arXiv:2408.06929.

Wangyue Li, Liangzhi Li, Tong Xiang, Xiao Liu, Wei Deng, and Noa Garcia. 2024. Can multiple-choice questions really be useful in detecting the abilities of LLMs? In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 2819–2834. ELRA and ICCL.

Benjamin S Manning, Kehang Zhu, and John J Horton. 2024. Automated social science: Language models as scientist and subjects. Technical report, National Bureau of Economic Research.

Peter S Park, Philipp Schoenegger, and Chongyang Zhu. 2024. Diminished diversity-of-thought in a standard large language model. Behavior Research Methods, pages 1–17.

Yossi Rubner, Carlo Tomasi, and Leonidas J Guibas. 1998. A metric for distributions with applications to image databases. In Sixth international conference on computer vision (IEEE Cat. No. 98CH36271), pages 59–66. IEEE.

Seungjong Sun, Eungu Lee, Dongyan Nan, Xiangying Zhao, Wonbyung Lee, Bernard J. Jansen, and Jang Hyun Kim. 2024. Random silicon sampling: Simulating human sub-population opinion using a large language model based on group-level demographic information. Preprint, arXiv:2402.18144.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.

Dustin Wright, Arnav Arora, Nadav Borenstein, Srishti Yadav, Serge Belongie, and Isabelle Augenstein. 2024. Revealing fine-grained values and opinions in large language models.

Yahan Yang, Soham Dan, Dan Roth, and Insup Lee. 2024. On the calibration of multilingual question answering llms. Preprint, arXiv:2311.08669.

Wenlong Zhao, Debanjan Mondal, Niket Tandon, Danica Dillion, Kurt Gray, and Yuling Gu. 2024. Worldvaluesbench: A large-scale benchmark dataset for

multi-cultural value awareness of language models. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 17696–17706.

## A Pew Data Distribution

The detailed statistics for the C2 and C3 split of the cultural survey simulation and the C3 split of GlobalOpinionQA data mentioned are available in Table 8. The sampled countries, along with the sampling reason of $\mathrm { P e w } – C _ { 1 } ^ { \prime }$ , are in Table 9.

<table><tr><td colspan="2">C2</td><td colspan="3">C3</td></tr><tr><td>Country</td><td>CSS-N</td><td>Country</td><td>CSS-N</td><td>PWE-N</td></tr><tr><td>Egypt</td><td>157</td><td>Malaysia</td><td>150</td><td>516</td></tr><tr><td>Ethiopia</td><td>185</td><td>Thailand</td><td>150</td><td>319</td></tr><tr><td>Kenyā</td><td>185</td><td>Czechia</td><td>150</td><td>212</td></tr><tr><td>Libya</td><td>185</td><td>Greece</td><td>150</td><td>648</td></tr><tr><td>Morocco</td><td>184</td><td>Nigeria</td><td>210</td><td>1044</td></tr><tr><td>Nigeria</td><td>185</td><td>Morocco</td><td>209</td><td>450</td></tr><tr><td>Tunisia</td><td>184</td><td>Peru</td><td>146</td><td>545</td></tr><tr><td>Zimbabwe</td><td>185</td><td>Colombia</td><td>150</td><td>369</td></tr><tr><td colspan="2"></td><td>Mexico</td><td>150</td><td>890</td></tr><tr><td colspan="2"></td><td>Puerto Rico</td><td>150</td><td>221</td></tr><tr><td colspan="2"></td><td>New Zealand</td><td>149</td><td>274</td></tr></table>

Table 8: Number of Entries Used in C2 and C3 Split of Cultural Survey Simulation and C3 Split of GlobalOpinionQA Data.

<table><tr><td>Country</td><td>Continent</td><td>GDP-level</td><td>N</td></tr><tr><td>Germany</td><td>Europe</td><td>High</td><td>1130</td></tr><tr><td>Japan</td><td>Asia</td><td>High</td><td>891</td></tr><tr><td>Brazil</td><td>South America</td><td>Upper-middle</td><td>922</td></tr><tr><td>Australia</td><td>Oceania</td><td>High</td><td>627</td></tr><tr><td>India</td><td>Asia</td><td>Lower-middle</td><td>932</td></tr><tr><td>Nigeria</td><td>Africa</td><td>Lower-middle</td><td>1044</td></tr><tr><td>United States</td><td>North America</td><td>High</td><td>1104</td></tr><tr><td>Vietnam</td><td>Asia</td><td>Lower-middle</td><td>471</td></tr><tr><td>Chile</td><td>South America</td><td>Upper-middle</td><td>542</td></tr><tr><td>Ukraine</td><td>Europe</td><td>Lower-middle</td><td>661</td></tr></table>

Table 9: Sampled countries for Pew-C′ regarding geographical and GDP-level diversity. The dataset consists of 8,324 entries in total.

## B WVS data Distribution

Figure 4 illustrates the sample distribution across various countries in World Values Survey. For the purpose of balanced representation, in our formatted dataset, we only include countries with more than 1,000 samples, ensuring consistency in data distribution across countries and minimizing sample size discrepancies in the analysis.

<table><tr><td>Set</td><td>Cultural Dimension</td><td>Number</td></tr><tr><td> $Q _ { 1 }$ </td><td>Social values, attitudes and stereotypes Societal well-being Social capital, trust and organizational membership Economic values Corruption Migration Post-materialist index Science and technology</td><td>45 12 49 6 9 10 21</td></tr><tr><td> $Q _ { 2 }$ </td><td>Religious values Security</td><td>6 6 12</td></tr><tr><td> $Q _ { 3 }$ </td><td>Ethical values and norms Political interest and political participation Political culture and political regimes</td><td>23 36 25</td></tr><tr><td> $C _ { 1 }$ </td><td>Andorra, Argentina, Australia, Bangladesh, Armenia, Bolivia, Brazil, Myanmar, Canada, Chile, China, Tai- wan ROC, Cyprus, Ecuador, Germany, Guatemala, Hong Kong SAR, Indonesia, Iran, Iraq, Japan, Kazakhstan, Jordan, South Korea, Kyrgyzstan, Lebanon, Macao SAR, Maldives, Mongolia, Netherlands, Nicaragua, Pak- istan, Philippines, Romania, Russia, Serbia, Singapore, Slovakia, Vietnam, Tajikistan, Turkey, Ukraine, Great</td><td>46</td></tr><tr><td> $C _ { 2 }$ </td><td>Britain, United States, Uruguay, Venezuela Egypt, Ethiopia, Kenya, Libya, Morocco, Nigeria, Tunisia, Zimbabwe</td><td>8</td></tr><tr><td> $C _ { 3 }$ </td><td>Malaysia, Thailand, Czechia, Greece, Nigeria, Morocco, Peru, Colombia, Mexico, Puerto Rico, New Zealand</td><td>11</td></tr></table>

Table 10: Cultural dimensions and question ids of WVS. Question 82-223 and 94-106 are excluded in $Q _ { 1 }$ as they are demo graphical questions. Demo graphical questions are excluded as they are related to individual attribute regarding participants and does not have relevance to group culture.

Regarding dataset split, we split WVS by cultural dimensions and countries vis multiple sets, which is visualized in Figure 5 and the corresponding cultural dimensions are detailed in Table 10, offering a comprehensive view of the overall distribution in the cultural survey simulation.

## C Hyperparameter Settings

Training is performed on a single A100 GPU with a batch size of 16 for Llama3 and Vicuna1.5-7B, and 4 for Vicuna1.5-13B. Both models use the AdamW optimizer with a learning rate of 1e-4 and implement Fully Sharded Data Parallel along with a mixed precision strategy to enhance computational efficiency. In our experiments, we employ LoRA with a rank of 8, a scaling factor lora\_alpha set to 32, and a dropout rate of 0.05.

Model Download We list all used models here:

• Vicuna1.5-7B: https://huggingface.co/ lmsys/vicuna-7b-v1.5

• Vicuna1.5-13B: https://huggingface.co/ lmsys/vicuna-13b-v1.5

Country  
![](images/44c679692e48e217fc0e2606b69aac6ba64709adfdb5377a6f8b047b4002621c.jpg)

Figure 4: Sample distribution of the World Values Survey across countries, including only countries with more than 1,000 samples for balanced representation.  
![](images/ef456cdc86c15978fd7fca605c94625cefe25a0f43ce43e74667f505f7670e87.jpg)  
(a) Country distribution across training, validation, and testing sets.  
(b) Cultural dimension distribution.  
Figure 5: Visualization of country and cultural dimension divisions of WVS. Countries are categorized into three groups, and questions are divided based on selected cultural dimensions.

• Distil-Qwen-7B: https:// huggingface.co/deepseek-ai/ DeepSeek-R1-Distill-Qwen-7B

• Distil-Qwen-14B: https:// huggingface.co/deepseek-ai/ DeepSeek-R1-Distill-Qwen-14B

• Distil-Qwen-32B: https:// huggingface.co/deepseek-ai/ DeepSeek-R1-Distill-Qwen-32B

• Llama3-8B-Base: https://huggingface. co/meta-llama/Meta-Llama-3-8B

• Llama3-8B-Instruct: https: //huggingface.co/meta-llama/ Meta-Llama-3-8B-Instruct

• Qwen-7B: https://huggingface.co/ Qwen/Qwen-7B

## D More Option Prediction Analysis

Option Prediction Accuracy. In line with Figure 3e-3f, we present the remaining two sets of

Llama3 option prediction accuracy for C3-Q1 and C3-Q3 here (see Figure 6).

The results indicate that the model performs significantly better on seen questions (C3-Q1) compared to unseen ones (C3-Q3), with Llama3 Instruct models consistently outperforming Base models, and fine-tuned approaches demonstrating superior accuracy over zero-shot methods across most countries. Specifically, the model achieves its highest accuracy in Greece, Nigeria, and Peru, while showing weaker performance in Thailand and Mexico. Despite the challenges in predicting unseen questions, fine-tuning through simulation leads to noticeable improvements, highlighting the potential of fine-tuned models to enhance generalization in novel scenarios.

Country Distributions. Figure 8 visualizes the distribution of 1 JSD scores for the Llama3-Base and Llama3-Instruct models across unseen cultural questions (C<sub>1</sub>-Q<sub>3</sub>) in both zero-shot (ZS) and finetuned (FT) modes. In ZS mode, both models perform poorly across many regions, with lower scores indicating limited cultural sensitivity. This suggests that without additional tuning, both models struggle to effectively generalize to simulate culturally diverse distributions. Furthermore, the Instruct model performs slightly worse, likely due to the reinforcement learning and safety strategies, which may inadvertently reduce its sensitivity to cultures in this scenario.

![](images/892b45536bca038d8caac59e12d02e63c9223b569ea6c66732ef267eb59a7ceb.jpg)

(a) Country Accuracy on C3-Q1.  
![](images/b1bda6471e1df0090a55eb49cd8ce088c60535a932c9d15bebdfb576f5c02226.jpg)  
(b) Country Accuracy on C3-Q3.  
Figure 6: Llama3 Accuracy of options on African countries in both seen C3-Q1 and unseen C3-Q3 questions.

However, after FT, both models show improvement across all countries, particularly significant in regions previously displaying poor performance in the ZS setting. Additionally, the performance of both the fine-tuned Base model and the Instruct model is observed to be very similar across different countries, demonstrating that our methods can effectively align the Base and Instruct models.

![](images/dc0218e0a8ca4c5f531a256c9249982afd21f517270fdaaae70a7674513bb9f9.jpg)  
Figure 7: Training loss comparsion of three models: Qwen-7B, Llama-8B-Instruct, Distill-Qwen-32B(Deepseek)

## E More Baseline Comparsion

To further assess the effectiveness of our proposed approach, we compare it against several other baseline methods as follows:

• KNN: This method identifies the most similar training question and country (top-1, k = 1) using BERT embeddings and returns the corresponding option distributions as predictions.

• Avg\_Culture: We computes the mean option distribution across all training countries for each known question and adopts a uniform random distribution for unknown questions.

• JSON-ZS: This approach prompts the models to directly generate option distributions in JSON format without additional fine-tuning.

Table 11 presents the 1-JSD scores for various baseline methods and our proposed approach. The results indicate that all baseline methods perform substantially worse than our fine-tuned (FT) models. Among the baselines, the JSON-ZS method demonstrates superior performance compared to KNN and Avg\_Culture; however, it remains less effective than both zero-shot (ZS) and fine-tuned (FT) approaches. Notably, fine-tuning consistently yields the highest scores across all evaluated settings, underscoring its effectiveness in improving model performance.

![](images/cb6eacee3ff80c63f0392952ced31ff61bcf4d54c68ef2e2264724bf42d88e20.jpg)  
(a) Llama3\_Base\_ZS

![](images/7e445525d2388e861d6be81fe17ba32dda4434310938890aa5a850ebabb1c957.jpg)

![](images/503b890a194fdfcb081cef67e44b419ecf98c9c5bffc149238202b49eb19fd9a.jpg)  
(c) Llama3\_Instruct\_ZS

(b) Llama3\_Base\_FT  
![](images/531ad4dc3847fb8aaf0dbb33b57baaeaae64f82fe9d35d28a529e1d9d690d75c.jpg)  
(d) Llama3\_Instruct\_FT

Figure 8: Distribution of 1 JSD Global Scores for the Model on Unseen Cultural Questions $( C _ { 1 }  – Q _ { 3 } )$ . The Instruct model exhibits a more distinct improvement compared to the Base model.
<table><tr><td>Methods</td><td>C1-Q3</td><td>C2-Q1</td><td>C2-Q3</td><td>C3-Q1</td><td>C3-Q3</td><td> $\mathbf { A v 8 \cdot }$ </td></tr><tr><td>KNN</td><td>0.381</td><td>0.518</td><td>0.371</td><td>0.541</td><td>0.384</td><td>0.439</td></tr><tr><td>Avg_Culture</td><td>0.360</td><td>0.509</td><td>0.348</td><td>0.518</td><td>0.368</td><td>0.421</td></tr><tr><td colspan="7">Llama3-8B-Base</td></tr><tr><td>JSON-ZS</td><td>0.754</td><td>0.728</td><td>0.581</td><td>0.729</td><td>0.751</td><td>0.709</td></tr><tr><td>ZS</td><td>0.749</td><td>0.768</td><td>0.759</td><td>0.781</td><td>0.770</td><td>0.765</td></tr><tr><td>FT</td><td>0.770</td><td>0.858</td><td>0.773</td><td>0.877</td><td>0.781</td><td>0.812</td></tr><tr><td colspan="7">Llama3-8B-Instruct</td></tr><tr><td>JSON-ZS</td><td>0.735</td><td>0.728</td><td>0.747</td><td>0.729</td><td>0.751</td><td>0.738</td></tr><tr><td>ZS</td><td>0.585</td><td>0.650</td><td>0.589</td><td>0.657</td><td>0.584</td><td>0.613</td></tr><tr><td>FT</td><td>0.777</td><td>0.881</td><td>0.783</td><td>0.890</td><td>0.784</td><td>0.823</td></tr></table>

Table 11: Comparison of 1-JSD scores across different baseline methods and our approach. Higher values indicate better alignment with the ground-truth distribution.

![](images/e79cda85528afda263377e87d95bd06004a1cf6c5c5ae93d724f463011b1d469.jpg)  
Figure 9: Comparsion of different languages in 1-JSD score on random selected 100 samples.

## F More Model Evaluation

In this section, we present additional experiment results on original Qwen model. We observe that the original Qwen models present challenges in training for our task. As shown in Figure 7, we compare the training progress of three models. The loss of the Qwen-7B model exhibits difficulty in converging compared to Llama-8B-Instruct and Distill-Qwen-32B. This difficulty may stem from the extensive safety alignment or policy alignment incorporated into Qwen, which could introduce additional constraints or optimization challenges during training.

Secondly, we visualized the 1-JSD score distribution of the model for English ZS, FT, and Chinese FT, as shown in Figure 9, with the red dashed line indicating the average value. The results show that language differences have a smaller impact on the model than training levels. Additionally, we calculate the Pearson correlation coefficient between English ZS and FT which (0.349) is lower than that between English FT and Chinese FT (0.579). While both correlations are positive, the stronger correlation observed between English FT and Chinese FT suggests that training level exerts a greater influence than language differences.