# Unlocking Decoding-time Controllability: Gradient-Free Multi-Objective Alignment with Contrastive Prompts

Tingchen Fu¹†, Yupeng Hou2, Julian McAuley2\*, Rui Yan1\* 1Renmin University of China 2UC San Diego {tingchenfu,ruiyan}@ruc.edu.cn {yphou,jmcauley}@ucsd.edu

## Abstract

The task of multi-objective alignment aims at balancing and controlling the different alignment objectives (e.g., helpfulness, harmlessness and honesty) of large language models to meet the personalized requirements of different users. However, previous methods tend to train multiple models to deal with various user preferences, with the number of trained models growing linearly with the number of alignment objectives and the number of different preferences. Meanwhile, existing methods are generally poor in extensibility and require significant re-training for each new alignment objective considered. Considering the limitation of previous approaches, we propose MCA (Multi-objective Contrastive Alignemnt), which constructs an expert prompt and an adversarial prompt for each objective to contrast at the decoding time and balances the objectives through combining the contrast. Our approach is verified to be superior to previous methods in obtaining a well-distributed Pareto front among different alignment objectives.

## 1 Introduction

Aligning large language models (LLMs) trained on vast web corpora (OpenAI, 2023; Touvron et al., 2023; Google, 2023) with human preferences is an important step to mitigate the production of unsafe (Wei et al., 2023), hallucinated (Zhang et al., 2023b) and biased (Gallegos et al., 2023) contents. With the recent development of preference learning techniques like PPO (Schulman et al., 2017), DPO (Rafailov et al., 2023) and other variants (Azar et al., 2023; Ethayarajh et al., 2024; Meng et al., 2024), there has been progress toward building an open-domain AI assistant that could follow user preferences.

However, human preferences are not a fixed standard but vary significantly from person to person.

![](images/d18455ba111ac2c8a0f6b66f7c64e0b51235b2de524631556c48c01390591152.jpg)

![](images/5262325ed5aadc5fdd156b4da134a8c109ccce0d90c612f6980133be2a7bb850.jpg)  
Figure 1: The correlation between helpfulness score and harmlessness score on Phi-2 generated responses on HH-RLHF (left) and SafeRLHF (right). The scores are given by objective-specific reward models.

For instance, a Ph.D. student inquiring about an academic problem probably expects a factual and informative reply; a five-year-old asking for a virtual playmate would put emphasis on safety and humor. However, it is rather difficult to obtain an AI assistant excelling at all alignment dimensions¹ since different alignment dimensions might intrinsically interfere with each other (Wolf et al., 2024; Bianchi et al., 2024; Guo et al., 2024). For example, as illustrated in Figure 1, we measure the correlation between the helpfulness and harmlessness of Phi-2 (Li et al., 2023b) generated response. We find that the performance on the two alignment objectives is negatively correlated, with the Spearman's correlation coefficient being $\rho = - 0 . 5 1$ for HH-RLHF (Bai et al., 2022) and ρ = −0.61 for SafeRLHF (Ji et al., 2023) $( p < 0 . 0 1 )$ . The negative correlation indicates a potential trade-off between helpfulness and harmlessness. Consequently, controllability in multi-objective alignment is vital to satisfy the diverse preferences of different users with a single language model and the task of multi-objective alignment is drawing heated attention (Sorensen et al., 2024).

To control the trade-off between multiple objectives to serve different users, as an initial attempt, Zhou et al. (2024b) tune a language model for each preference, which is time-consuming and costly. To avoid tuning a language model for all potential preferences, there are generally two concurrent lines of work. On one hand, aggregation-based methods (Jang et al., 2023; Zhou et al., 2024a) tune a series of specialized models for each alignment dimension and meet with various preferences through model merging or model ensemble, reducing the numbers of tuned models to the alignment dimensions considered. Moreover, instruction-based method (Guo et al., 2024; Yang et al., 2024b; Lee et al., 2024) insert control tokens into the prompt, resulting in a single controllable aligned model. But as a cost, their methods are poor in extensibility since their prompt format is pre-defined on existing alignment objectives and cannot extend to a new alignment objective.

Therefore, we attempt to reduce the number of trained models further and propose a gradient-free controllable alignment approach that requires no additional model training. Getting inspiration from contrastive decoding (Li et al., 2023a), in this study we propose MCA. for each alignment dimension, we perform response augmentation with an LLM to obtain responses with different rewards. The responses with maximum or minimum reward then serve as demonstrations to induce an expert prompt and an adversarial prompt, which are used for promoting and suppressing the corresponding alignment dimension, respectively. The predictions in the logit space induced by the two prompts then constitute a contrast for the language model. By manipulating the weight of the contrast, users can control the language model at their own preference and incorporate any new required alignment objectives at decoding time if necessary.

Overall, our contribution can be summarized as:

• We provide a gradient-free solution to the multiobjective alignment problem, achieving control over different alignment dimensions without updating the parameters of the base language model. • We introduce MCA, a contrastive alignment framework, which to our knowledge is the first to incorporate multiple expert prompts and adversarial prompts into contrastive decoding.

• We perform extensive experiments on two datasets to empirically verify the effectiveness of our approach in controlling the trade-off between existing alignment dimensions and incorporating new dimensions.

<table><tr><td></td><td>#Trained LLM</td><td>Extensibility</td></tr><tr><td>MORL (Jang et al., 2023)</td><td>M</td><td>x</td></tr><tr><td>P-SOUP (Jang et al., 2023)</td><td>N</td><td>V</td></tr><tr><td>MODPO (Zhou et al., 2024b)</td><td>M</td><td>X</td></tr><tr><td>RiC (Yang et al., 2024b)</td><td>1</td><td>X</td></tr><tr><td>MCA</td><td>0</td><td>L</td></tr></table>

Table 1: Comparison between previous works and MCA. N is the number of alignment objectives considered and M is the number of preferences (i.e., a set of weight coefficients for different alignment objectives).

## 2 Related Work

Language Model Alignment. Language model alignment is a crucial procedure before a pretrained language model can serve as an opendomain AI assistant and there are two major techniques to achieve this goal, namely instructiontuning (Taori et al., 2023; Xu et al., 2023; Zhou et al., 2023) and preference learning (Ouyang et al., 2022; Rafailov et al., 2023; Azar et al., 2023). Instruction-tuning is a supervised finetuning (SFT) process where the base model is tuned on instruction-following data (Conover et al., 2023; Ivison et al., 2023) with language modeling objective. Preference learning or reinforcement learning from human feedback (RLHF), on the other hand, employs RL training algorithms (Rafailov et al. 2023; Schulman et al., 2017; Meng et al., 2024) to learn human preferences from preference data.

Despite the preference data being collected from crowd workers with diverse backgrounds, previous alignment techniques mostly fit on the “average" preference of the crowd while overlooking the personalized preference (Sorensen et al., 2024). Furthermore, Chakraborty et al. (2024) theoretically proves the impossibility of alignment with a single reward in RLHF, which is too restrictive to reflect the opinion and preference of some minority groups (Chakraborty et al., 2024), leading to a biased language model.

Multi-objective Alignment. In pursuit of multiobjective alignment, numerous previous works have been developed to serve diverse users considering their unique preferences (Jang et al., 2023; Yang et al., 2024b; Guo et al., 2024; Tuan et al., 2024; Lee et al., 2024; Yang et al., 2024a). As an initial attempt, multi-objective reinforcement learning (MORL) (Rame et al., 2023) and its variant (Zhou et al., 2024b) tune a specialized model for each preference2. However, as the computation cost of training an individual model for each preference is beyond the budget for most institutions, follow-up works (Jang et al., 2023; Zhou et al., 2024a) reduce the number of trained models to the number of the alignment objectives considered. As a concurrent line of work, Yang et al. (2024b); Guo et al. (2024); Tuan et al. (2024); Zhong et al. (2024b) insert user preference as a “control token" (Lu et al., 2022) into the prompt (Yang et al., 2024b) or the model weight (Zhong et al., 2024b) during SFT to achieve controllability and further reduce the number of trained models to one. However, this line of works suffers from poor scalability since the user preference is hard-encoded into the prompt during training³. Consequently, re-training is required for every new alignment objective. A summary of the previous methods in contrast with our proposal is illustrated in Table 1.

Contrastive Decoding. Initially developed by Li et al. (2023a); Liu et al. (2021), contrastive decoding employs the distribution difference in next-word prediction between the expert model and anti-expert model to improve generation quality. Follow-up works extend the original framework and contrast the next-word prediction logits induced by not only different models (Zhang et al., 2023a), but also different prompts (Kim et al., 2023) and the outputs of different layers (Chuang et al., 2024). Contrastive decoding is widely used to improve performance in math reasoning (O'Brien and Lewis, 2024; Phan et al., 2024), machine translation (Sennrich et al., 2024) together with the safety (Xu et al., 2024; Zhong et al., 2024a; Niu et al., 2024) and factuality (Zhang et al., 2023a; Chuang et al., 2024) of LLM. Recently, Liu et al. (2024a); Mitchell et al. (2023) contrast an aligned model against a base model to guide the LLM alignment. Liu et al. (2024b) further explores the potential of contrastive decoding in alignment controllability. The most relevant work is DeAL (Huang et al., 2024), which directly incorporates reward models into the decoding phase to modify the probability distribution of the next token prediction.

## 3 Method

In this section, we present a new multi-objective alignment framework to manipulate the trade-off between conflicting alignment dimensions. We start with the problem formulation for multiobjective alignment in Section 3.1. Next, we elaborate on our simple two-step framework in which we first construct an expert prompt and an adversarial prompt for each alignment objective (Section 3.2), and then employ the constructed prompt pair for contrastive decoding (Section 3.3). By contrasting and combining the next-token probability induced by different prompts at inference time, we attain better flexibility and controllability over different alignment dimensions with no parameter updates.

## 3.1 Problem Formulation

In this study, we focus on building a controllable open-domain AI assistant to follow diverse human instructions. Specifically, aside from user query $^ { \mathbf { \delta x } , }$ a user preference $\pmb { w } = [ w _ { 1 } , w _ { 2 } , \dots , w _ { n } ]$ is provided to the language model π, where n is the total number of alignment dimensions considered and $w _ { i }$ denotes the weight for the ¿-th alignment dimension. w lies in n-dimensional simplex. Ideally, the optimal response ${ \pmb y } ^ { * }$ will maximize the weighted sum of rewards in different alignment dimensions:

$$
{ \pmb y } ^ { * } = \underset { { \pmb y } } { \operatorname { a r g m a x } } \sum _ { i = 1 } ^ { n } w _ { i } \cdot \mathrm { r _ { i } } ( { \pmb x } , { \pmb y } ) ,\tag{1}
$$

where $\mathrm { r } _ { \mathrm { i } } ( { \pmb x } , { \pmb y } )$ is the reward model that produces a scalar reward value denoting the quality of response y to the query x on the i-th alignment dimension.

## 3.2 Iterative Prompt Construction

Suppose we have a user-defined reward model $\mathrm { r } ( \cdot , \cdot )$ for each alignment dimension. To control that alignment dimension at inference time, a possible way is to transform the user preference acquired by the reward model into a pair of prompts (Cai et al., 2024), namely an expert prompt $z ^ { + }$ and an adversarial prompt z−. The expert prompt is used to prompt the language model to generate responses that maximize the reward. In contrast, the adversarial prompt is responsible for inducing responses that minimize the reward. Formally, our objective in this step is to find the following prompts:

$$
\begin{array} { r l } & { z ^ { + } = \underset { z } { \mathrm { a r g m a x } } \mathbb { E } _ { y \sim \pi ( y | x , z ) } \mathrm { r } ( x , y ) , } \\ & { z ^ { - } = \underset { z } { \mathrm { a r g m i n } } \mathbb { E } _ { y \sim \pi ( y | x , z ) } \mathrm { r } ( x , y ) . } \end{array}\tag{2}
$$

![](images/ad5a887a8058afa698bf72dc42ea01eeb5c192cdb5158b51a20a391825bada1b.jpg)  
Figure 2: The workflow of proposed MCA is composed of two major steps: iterative prompt construction and preference-aware multiple contrastive decoding.

Following previous work in prompt optimization (Cheng et al., 2023), to obtain the textual prompts $z ^ { + }$ and $z ^ { - }$ , we firstly perform data augmentation on model response. In detail, for a given user query x, we initialize a response pool $\mathcal { P }$ by sampling a group of responses: ${ \mathcal { P } } = \{ { \pmb y } _ { i } \ | \ { \pmb y } _ { i } \sim $ $\pi ( \pmb { y } \mid \pmb { x } ) , \quad i = 1 , 2 , \ldots , m \}$ , where m is the size of the response pool. Next, we score each response with the reward model and employ the responses to prompt for response with higher or lower reward, similar to Yang et al. (2024b). Specifically, to seek a higher/lower reward, we select the responses with top/bottom- $\cdot m / 2$ rewards from the response pool and input them into the language model $\pi$ as fewshot demonstrations to generate more responses:

$$
\begin{array} { r l } & { \pmb { y } ^ { + } \sim \pi ( \pmb { y } \mid \pmb { x } ; \pmb { y } _ { 1 } , \pmb { y } _ { 2 } , \ldots , \pmb { y } _ { m / 2 } ) , } \\ & { \pmb { y } ^ { - } \sim \pi ( \pmb { y } \mid \pmb { x } ; \pmb { y } _ { m / 2 + 1 } , \pmb { y } _ { m / 2 + 2 } , \ldots , \pmb { y } _ { m } ) . } \end{array}\tag{3}
$$

The newly generated responses are scored and incorporated into the pool. Then the pool is filtered to keep top-m/2 and bottom-m/2 responses while discarding others, maintaining a constant pool size of $m .$ The iteration is repeated until the response pool no longer updates or the number of iterations reaches a limit.

After finishing the response augmentation for a handful of user queries we now have a response pool for each query. Then we choose k queries with top-k range of reward values in their response pool. Next, we send the queries as well as their highestrewarded and lowest-rewarded response to a proprietary LLM such as GPT-4 (OpenAI, 2023), asking the LLM to provide an instruction that encourages the high-rewarded/low-rewarded responses. The outputted instruction from LLM is exploited to construct $z ^ { + }$ and $z ^ { - }$

## 3.3 Preference-Aware Multiple Contrastive Decoding

After constructing an expert prompt $z ^ { + }$ and an adversarial prompt $z ^ { - }$ for each alignment dimension, we can now manipulate the effect of each prompt via contrastive decoding and therefore control the strength of the corresponding alignment dimensions. In detail, compared with vanilla autoregressive generation in which the next token distribution is predicted by $\begin{array} { r } { \pi ( \pmb { y } \mid \pmb { x } ) = \prod _ { t = 1 } \pi ( y _ { t } \mid } \end{array}$ $\scriptstyle { \boldsymbol { x } } , { \boldsymbol { y } } < { \boldsymbol { t } } )$ , we prepend the prompt regarding a specific alignment dimension to the user query to adjust the predicted next token distribution,

$$
\pi _ { 1 - \mathrm { c o n t } } ( \pmb { y } \mid \pmb { x } ) = \prod _ { t = 1 } \sigma \left( \log \frac { \pi ( y _ { t } \mid \pmb { x } , z ^ { + } , y _ { < t } ) } { \pi ( y _ { t } \mid \pmb { x } , z ^ { - } , y _ { < t } ) } \right) ,\tag{4}
$$

where $\sigma$ denotes the softmax function. Therefore, the language model is guided toward the alignment dimension corresponding to $z ^ { + }$ and $z ^ { - }$ . To extend the framework to multiple objectives, we simply incorporate the user preference $\pmb { w } = [ w _ { 1 } , w _ { 2 } , \dots , w _ { n } ]$ as the weight for combing the predicted next token distributions:

$$
\pi _ { \mathrm { n - c o n t } } ( \pmb { y } \mid \pmb { x } ) = \prod _ { t = 1 } \sigma \left( \log \sum _ { i = 1 } ^ { n } w _ { i } \frac { \pi ( y _ { t } \mid \pmb { x } , \pmb { z } _ { i } ^ { + } , y _ { < t } ) } { \pi ( y _ { t } \mid \pmb { x } , \pmb { z } _ { i } ^ { - } , y _ { < t } ) } \right) .\tag{5}
$$

However, contrastive decoding is known to suffer from false positives and false negatives, especially for some easy tokens (Li et al., 2023a). To deal with the problem, following Li et al. (2023a) and Zhang et al. (2023a), we introduce a constraint to exclude some tokens from contrast. Namely, we only consider a subset of vocabulary that is assigned with a higher probability than a pre-defined adaptive threshold:

<table><tr><td></td><td colspan="3">HH-RLHF</td><td colspan="3">SafeRLHF</td></tr><tr><td></td><td>Helpfulness(↑)</td><td>Harmlessness(↑)</td><td>Average(↑)</td><td>Helpfulness(↑)</td><td>Harmlessness(↑)</td><td>Average(↑)</td></tr><tr><td>Phi-2</td><td>1.43</td><td>-0.53</td><td>0.45</td><td>1.66</td><td>-0.54</td><td>0.56</td></tr><tr><td>Phi-2+MCA</td><td>1.76</td><td>-0.20</td><td>0.78</td><td>1.98</td><td>-0.06</td><td>0.96</td></tr><tr><td>Phi-2-SFT</td><td>1.41</td><td>-0.64</td><td>0.39</td><td>0.87</td><td>-0.15</td><td>0.36</td></tr><tr><td>Phi-2-SFT+MCA</td><td>1.94</td><td>-0.61</td><td>0.67</td><td>1.80</td><td>0.52</td><td>1.16</td></tr><tr><td>Phi-2-PPO</td><td>1.85</td><td>-0.41</td><td>0.72</td><td>1.79</td><td>0.05</td><td>0.92</td></tr><tr><td>Phi-2-PPO+MCA</td><td>1.95</td><td>0.15</td><td>1.05</td><td>1.94</td><td>0.55</td><td>1.25</td></tr></table>

Table 2: The results of the single-objective alignment experiments on HH-RLHF and SafeRLHF using Phi-2 as the backbone. The “Average" column is the average reward value of helpfulness and harmlessness. The numbers in bold are significant improvements in average rewards (t-test, $p < 0 . 0 5 )$ 1

$$
\begin{array} { l } { \mathcal { V } _ { s u b } = \{ y _ { t } \in \mathcal { V } : } \\ { \pi ( y _ { t } \mid x , z ^ { + } , y _ { < t } ) > \alpha \underset { w } { \operatorname* { m a x } } \pi ( w \mid x , z ^ { + } , y _ { < t } ) \} , } \end{array}\tag{6}
$$

where $\alpha \in [ 0 , 1 ]$ is a hyper-parameter. Intuitively, we truncate the token distribution and discard the token that the language model is not very confident. Consequently, the final next token distribution is:

$$
\pi ( \pmb { y } \mid \pmb { x } ) = \left\{ \begin{array} { l l } { \pi _ { \mathrm { n - c o n t } } ( \pmb { y } \mid \pmb { x } ) , } & { \pmb { x } \in \mathcal { V } _ { s u b } } \\ { 0 , } & { \mathrm { o t h e r w i s e } } \end{array} \right.\tag{7}
$$

Discussion. Compared with existing methods, one of our advantages lies in that we control the weights of different alignment objectives at decoding time and do not require additional objectivespecific training. Therefore, MCA can be directly applied to base backbone models without going through SFT or PPO, which is linked with potential forgetting of parametric knowledge (Dou et al. 2023; Lu et al., 2024). Moreover, our approach is orthogonal to previous techniques and can serve as a plug-in to combine with previous methods. But similar to previous methods, MCA assumes a given reward model for each alignment dimension.

## 4 Experiments

Backbone. We primarily adopt Llama-2-7b (Touvron et al., 2023) and Phi-2 (Li et al., 2023b) as backbones for experiments. But in principle, MCA is agnostic to the base model backbone and can be applied to any pre-trained auto-regressive language model.

<table><tr><td rowspan="2"></td><td colspan="2">HH-RLHF</td><td colspan="2">SafeRLHF</td></tr><tr><td>Train</td><td>Test</td><td>Train</td><td>Test</td></tr><tr><td># Samples</td><td>160,800</td><td>8,552</td><td>26,874</td><td>2,989</td></tr><tr><td> $\bar { L } _ { i n s t }$ </td><td>110.29</td><td>112.05</td><td>13.29</td><td>13.33</td></tr><tr><td> $\bar { L } _ { r e s p }$ </td><td>55.38</td><td>55.16</td><td>70.95</td><td>70.81</td></tr></table>

Table 3: The statistics of two datasets used in our experiments. $\bar { L } _ { i n s t }$ and $\bar { L } _ { r e s p }$ refer to the average length of instructions and responses respectively.

Dataset. We use HH-RLHF (Bai et al., 2022) and SafeRLHF (Ji et al., 2023) for our experiments. HH-RLHF is a human-annotated pairwise preference dataset where each datum contains two dialogues between a human user and an AI assistant and one dialogue is preferred over another. SafeRLHF is another human-annotated pair-wise preference dataset for alignment tuning. Different from HH-RLHF, it ranks two responses in each datum for helpfulness and harmlessness independently. The statistics of datasets are presented in Table 3.

Reward Model. To quantitatively evaluate the quality of the model-generated responses, we employ off-the-shelf reward models in Huggingface Hub to measure the performance on HH-RLHF following Yang et al. (2024b). The accuracy of the helpfulness reward model and the harmlessness reward model are 0.73 and 0.74, respectively. For SafeRLHF, we train a helpfulness reward model and a harmlessness reward model using GPT-2- large (Radford et al., 2019) as our backbone. The accuracy of the two reward models measured on the test set of SafeRLHF is 0.78 and 0.74, respectively. Apart from the two dimensions, following the setup of Yang et al. (2024b), we add the humor of the response as a third dimension and introduce a reward model directly from Yang et al. (2024b). The reward values given by the reward models are the main evaluation metric in our experiments.

<table><tr><td rowspan="2"></td><td colspan="3">HH-RLHF</td><td colspan="3">SafeRLHF</td></tr><tr><td>Helpfulness(↑)</td><td>Harmlessness(↑)</td><td>Average(↑)</td><td>Helpfulness(↑)</td><td>Harmlessness(↑)</td><td>Average(↑)</td></tr><tr><td>Llama-2-7b</td><td>0.47</td><td>-0.17</td><td>0.15</td><td>1.39</td><td>-0.55</td><td>0.42</td></tr><tr><td>Llama-2-7b+MCA</td><td>1.03</td><td>-0.03</td><td>0.50</td><td>1.79</td><td>-0.56</td><td>0.62</td></tr><tr><td>Llama-2-7b-SFT</td><td>1.35</td><td>-0.55</td><td>0.40</td><td>0.97</td><td>-0.14</td><td>0.42</td></tr><tr><td>Llama-2-7b-SFT+MCA</td><td>1.79</td><td>-0.50</td><td>0.65</td><td>1.80</td><td>0.52</td><td>1.16</td></tr><tr><td>Llama-2-7b-PPO</td><td>2.68</td><td>2.77</td><td>2.73</td><td>2.18</td><td>0.40</td><td>1.29</td></tr><tr><td>Llama-2-7b-PPO+MCA</td><td>2.86</td><td>2.80</td><td>2.83</td><td>2.05</td><td>0.55</td><td>1.30</td></tr></table>

Table 4: The results of the single-objective alignment experiments on HH-RLHF and SafeRLHF using Llama-2-7b as the backbone. The “Average" column is the average reward value for helpfulness and harmlessness. The numbers in bold are significant improvements in average rewards (t-test, p < 0.05).

![](images/c12ae605aaf2e5d38e8d550f8235341c464888cc80dbe51ea9462ae56b9b4585.jpg)

![](images/3f20dcbf4639da3bc020b5464cb92d58029ea47878f2a72e45ed8375bf895e43.jpg)

![](images/48068e345d84054521969b61dadb8ac3501256a1921bc5d3543cd0d36cfda1d1.jpg)  
(a) The Pareto front between helpfulness (b) The Pareto front between helpfulness (c) The Pareto front between helpfulness and harmlessness on HH-RLHF. and humor on HH-RLHF. and harmlessness on SafeRLHF.  
Figure 3: The Pareto front of Phi-2 evaluated on HH-RLHF and SafeRLHF when combined with MCA.

## 4.1 Single-Objective Alignment

Before working on the trade-off between multiple alignment dimensions, we first examine the effectiveness of our framework on a single alignment dimension, in which preference-aware multiple contrastive decoding is reduced to vanilla contrastive decoding on a specific alignment dimension. Specifically, for each alignment objective, we exploit our constructed expert prompt and adversarial prompt to conduct contrastive decoding. We perform experiments on the original language models, the SFT-ed models, and the PPO-ed models. The SFT-ed model is trained on the chosen queryresponse pairs in the pair-wise preference datasets. The SFT-ed model then acts as the reference model for the subsequent PPO training. For PPO-ed models, we tune a model for each alignment dimension separately. More implementation details on model training can be found in Appendix A.1. Experimental results are presented in Table 2 and Table 4. From the tables, we observe that when applied in the single-objective scenario, MCA can significantly improve the desired alignment dimension. Meanwhile, we note that compared to the original model, the SFT-ed model can hardly enhance two dimensions simultaneously, which echoes the previous findings that there exists some extent of trade-off between these two alignment objectives.

## 4.2 Two-Objective Alignment

Different from existing methods in multi-objective alignment which enhance controllability during instruction tuning or preference learning, MCA controls alignment at decoding time. Therefore, our approach can be directly applied to any off-shelf pre-trained LLMs. We verify the effectiveness of MCA on Phi-2 and Llama-2-7b and the experimental results are presented in Figure 3 and Figure 4, respectively. It is worth noting that our approach can extend a well-distributed Pareto front from a single point denoting the original base language model or the SFT-ed language model. Meanwhile, apart from one or two exceptional cases, the Pareto front extended from the SFT-ed model tends to lie in the outward direction of the one extended from the base model, suggesting that MCA can further strengthen the effect of SFT. In contrast, previous methods require either additional instructiontuning (Guo et al., 2024; Yang et al., 2024b) or RL training (Zhou et al., 2024b; Jang et al., 2023) and cannot be directly applied to base language models.

![](images/12db7eef9acb0e802717dfba11f620fb54a2e1e11f1a5db7b56b44b8eb7fa99c.jpg)

![](images/7e29fad115e28cf195beb4256164523f0f98034ef3a1a510a77ac07183c22c7c.jpg)

![](images/b8e2f97bece5a7d7d406cddb91f255e021aac12f4609f9124376cb1ab5d4efca.jpg)  
(a) The Pareto front between helpfulness (b) The Pareto front between helpfulness (c) The Pareto front between helpfulness and harmlessness on HH-RLHF. and humor on HH-RLHF. and harmlessness on SafeRLHF.

Figure 4: The Pareto front of Llama-2-7b evaluated on HH-RLHF and SafeRLHF when combined with MCA.  
![](images/c2b25c384fe53fafbf5f2840dcabafc6d4ea0c8a46cdc46451dc1428911d0169.jpg)  
Figure 5: The performance of Phi-2 in three alignment dimensions on HH-RLHF (left) and SafeRLHF (right) when combined with MCA. The reward values in three dimensions are normalized within [0, 1],

## 4.3 Three-Objective Alignment

Aside from single-objective alignment and twoobjective alignment, we now extrapolate to threeobjective alignment to inspect whether MCA can be adapted to multi-objective scenarios. The experiment results are presented in Fig. 5. From the radar figures we can observe that our approach can promote all three alignment dimensions simultaneously when applied to the base language model or the SFT-ed model, which further proves the effectiveness of MCA. Again, when comparing Phi-2 with Phi-2-SFT, it is not hard to find that vanilla SFT tends to enhance a single alignment objective (humor in HH-RLHF and harmlessness in SafeRLHF) at the sacrifice of the other two objectives.

## 5 Analysis

Apart from verifying the efficacy of our approach, to have a better understanding of its working mechanism, we further conduct the following experimental analysis:

## 5.1 Ablation Test

To investigate the effect of different components, we perform an ablation study with two variants: (1) keyword, where the prompt construction is removed and we instead merely use a keyword to describe the desired alignment dimensions in the following prompt: “A chat between a curious user and an artificial intelligence assistant. The assistant gives {objective} answers to the user's questions. {query}", where{objective} can be chosen from {helpfulness, harmlessness, humor}. (2) ensemble, where the adversarial prompt and the contrastive decoding framework are removed, so we sum up the logits induced by different expert prompts as $\pi _ { \mathrm { n - e n s e m b l e } } ( \pmb { y } \mid \pmb { x } ) =$ $\begin{array} { r } { \prod _ { t = 1 } \sigma \left( \log \sum _ { i = 1 } ^ { n } w _ { i } \pi ( y _ { t } \mid \pmb { x } , z _ { i } ^ { + } , y _ { < t } ) \right) } \end{array}$ . Experimental results are presented in Fig. 7, from which we can conclude that the contribution of the objective prompt construction is evident since the keyword variant is inferior to MCA. Meanwhile, as the Pareto front induced by the ensemble variant lies entirely in the inward direction of ours, we can therefore conclude that contrastive decoding is crucial.

![](images/08289b5ae490ef73ee32a684c900bbfe5ccb12a650ea1b31e88b5c960cfa05c6.jpg)

![](images/1aec719d56ff05f812e16b78ea8b746f38158e557f7bca32416ea282d0b35467.jpg)

![](images/4f8b45dd6bc59414c61d3b571f152d24ed15b53977ef2ba3d37d4890417dde11.jpg)  
(a) Pareto front between helpfulness and (b) Pareto front between helpfulness and (c) Pareto front between helpfulness and harmlessness evaluated on HH-RLHF.humor evaluated on HH-RLHF. harmlessness evaluated on SafeRLHF.

Figure 6: The Pareto front of Phi-2 on HH-RLHF and SafeRLHF when combined with P-SOUP and MORL.

![](images/21180d77678d7012d2df3780994dfd4cb4788e3dfad8598c5a0ebd988ce8f5ad.jpg)  
Figure 7: Pareto front of Phi-2 evaluated on HH-RLHF when combined with two variants.

## 5.2 Dynamics of Prompt Construction

To examine whether our data augmentation technique in Section 3.2 can attain a response pool with different rewards, we investigate the dynamics of the highest reward and the lowest reward in the response pool together with the coefficient of variation $( i . e . , \pmb { y } _ { 1 } - \pmb { y } _ { m } )$ . We compute the statistics as an average over a random subset of the HH-RLHF training set and the results are presented in Fig. 8. From the figure, the highest reward steadily increases while the lowest reward gradually declines during the iteration process, rendering the range of reward in the response pool larger and larger, which indicates the effectiveness of response augmentation. It is also worth noting that the evolution of the rewards becomes stable after three iterations. Refer to Appendix B.1 and Appendix B.2 for more results on the prompt construction.

## 5.3 Integration With Previous Methods

MCA is orthogonal to previous approaches and could be incorporated into previous methods to extend the boundary of the Pareto front further. Specifically, we combine our approach with the following methods: MORL (Rame et al., 2023), P-SOUP (Jang et al., 2023) and RiC (Yang et al., 2024b). The experimental results on Phi-2 and Llama-2-7b are presented in Fig. 6 and Fig. 15 respectively. The experiment results of combining our approach with RiC are presented in Fig. 16. From the figures, we can observe that MCA can be combined with previous approaches and further improve controllability by extending their original Pareto front outwards.

![](images/c10c1d169644432721bb7385745479a740402e632d44897a8365fe9d75daee6d.jpg)  
Figure 8: Reward statistics of the response pool when constructing prompt for helpfulness on HH-RLHF.

## 6 Conclusion

In this work, we focus on the multi-objective alignment problem and propose a new gradient-free approach as a possible solution. By contrasting and combining the logits at decoding time, MCA is verified to extend the original frontier between different alignment objectives by an obvious margin on various backbones and datasets. Meanwhile, we observed that the relationship between objectives can change under different circumstances, and we plan to work on the complex interrelation between multiple alignment objectives in the future.

## Limitations

All technologies built upon the large-scale PLM more or less inherit their potential harms (Bender et al., 2021). Besides, we acknowledge some specific limitations within our study:

• In our experiments, we use Phi-2 and Llama-2-7b as our backbones to verify whether our approach can control and coordinate different objectives in LLM alignment. However, limited by our computation resources, unfortunately, we cannot afford experiments on 30b models or larger ones. But in principle, MCA is agnostic to model architecture and can be applied to any pre-trained language models. • Following the setup of Yang et al. (2024b), our experiments mainly involve three dimensions, namely helpfulness, harmlessness, and humor, which are definitely only a small portion of all the desired objectives of LLM alignment. Although we cannot enumerate all potential objectives such as truthfulness, coherence, and verbosity, we believe our approach can generalize to other alignment objectives.

## Ethical Consideration

This paper has few ethical risks and will not pose a problem with ethics. Firstly, the alignment of large language models is not a new task in natural language processing, and several papers about this task have been published at NLP conferences. Secondly, all the datasets and benchmarks used in this paper have been published in previous papers. Our work aims at a better understanding and fulfillment of multi-objective alignment and our approach should not be used for any malicious purpose.

## Acknowlegement

We are grateful for the comments and feedback from Yiju Guo.

## References

Mohammad Gheshlaghi Azar, Mark Rowland, Bilal Piot, Daniel Guo, Daniele Calandriello, Michal Valko, and Rémi Munos. 2023. A general theoretical paradigm to understand learning from human preferences. ArXiv, abs/2310.12036.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, T. J. Henighan, Nicholas Joseph, Saurav Kadavath, John Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume,

Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom B. Brown, Jack Clark, Sam McCandlish, Christopher Olah, Benjamin Mann, and Jared Kaplan. 2022. Training a helpful and harmless assistant with reinforcement learning from human feedback. ArXiv, abs/2204.05862.

Emily M Bender, Timnit Gebru, Angelina McMillan-Major, and Shmargaret Shmitchell. 2021. On the dangers of stochastic parrots: Can language models be too big? In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency, pages 610–623.

Federico Bianchi, Mirac Suzgun, Giuseppe Attanasio, Paul Rottger, Dan Jurafsky, Tatsunori Hashimoto, and James Zou. 2024. Safety-tuned LLaMAs: Lessons from improving the safety of large language models that follow instructions. In The Twelfth International Conference on Learning Representations.

Deng Cai, Huayang Li, Tingchen Fu, Siheng Li, Weiwen Xu, Shuaiyi Li, Bowen Cao, Zhisong Zhang, Xinting Huang, Leyang Cui, Yan Wang, Lemao Liu, Taro Watanabe, and Shuming Shi. 2024. On the transformations across reward model, parameter update, and in-context prompt.

Souradip Chakraborty, Jiahao Qiu, Hui Yuan, Alec Koppel, Furong Huang, Dinesh Manocha, A. S. Bedi, and Mengdi Wang. 2024. Maxmin-rlhf: Towards equitable alignment of large language models with diverse human preferences. ArXiv, abs/2402.08925.

Jiale Cheng, Xiao Liu, Kehan Zheng, Pei Ke, Hongning Wang, Yuxiao Dong, Jie Tang, and Minlie Huang. 2023. Black-box prompt optimization: Aligning large language models without model training. arXiv preprint arXiv:2311.04155.

Yung-Sung Chuang, Yujia Xie, Hongyin Luo, Yoon Kim, James R. Glass, and Pengcheng He. 2024. Dola: Decoding by contrasting layers improves factuality in large language models. In The Twelfth International Conference on Learning Representations.

Mike Conover, Matt Hayes, Ankit Mathur, Jianwei Xie, Jun Wan, Sam Shah, Ali Ghodsi, Patrick Wendell Matei Zaharia, and Reynold Xin. 2023. Free dolly: Introducing the world's first truly open instructiontuned llm.

Shihan Dou, Enyu Zhou, Yan Liu, Songyang Gao, Jun Zhao, Wei Shen, Yuhao Zhou, Zhiheng Xi, Xiao Wang, Xiaoran Fan, Shiliang Pu, Jiang Zhu, Rui Zheng, Tao Gui, Qi Zhang, and Xuanjing Huang. 2023. Loramoe: Revolutionizing mixture of experts for maintaining world knowledge in language model alignment. ArXiv, abs/2312.09979.

Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff Dan Jurafsky, and Douwe Kiela. 2024. Kto: Model alignment as prospect theoretic optimization. ArXiv, abs/2402.01306.

Isabel O. Gallegos, Ryan A. Rossi, Joe Barrow, Md. Mehrab Tanjim, Sungchul Kim, Franck Dernoncourt, Tong Yu, Ruiyi Zhang, and Nesreen Ahmed. 2023. Bias and fairness in large language models: A survey. ArXiv, abs/2309.00770.

Gemini Team Google. 2023. Gemini: A family of highly capable multimodal models. ArXiv, abs/2312.11805.

Yiju Guo, Ganqu Cui, Lifan Yuan, Ning Ding, Jiexin Wang, Huimin Chen, Bowen Sun, Ruobing Xie, Jie Zhou, Yankai Lin, Zhiyuan Liu, and Maosong Sun. 2024. Controllable preference optimization: Toward controllable multi-objective alignment. ArXiv, abs/2402.19085.

James Y Huang, Sailik Sengupta, Daniele Bonadiman, Yi-an Lai, Arshit Gupta, Nikolaos Pappas, Saab Mansour, Katrin Kirchhoff, and Dan Roth. 2024. Deal: Decoding-time alignment for large language models. arXiv preprint arXiv:2402.06147.

Hamish Ivison, Yizhong Wang, Valentina Pyatkin, Nathan Lambert, Matthew E. Peters, Pradeep Dasigi, Joel Jang, David Wadden, Noah A. Smith, Iz Beltagy, and Hanna Hajishirzi. 2023. Camels in a changing climate: Enhancing lm adaptation with tulu 2. ArXiv, abs/2311.10702.

Joel Jang, Seungone Kim, Bill Yuchen Lin, Yizhong Wang, Jack Hessel, Luke Zettlemoyer, Hannaneh Hajishirzi, Yejin Choi, and Prithviraj Ammanabrolu. 2023. Personalized soups: Personalized large language model alignment via post-hoc parameter merging. ArXiv, abs/2310.11564.

Jiaming Ji, Mickel Liu, Juntao Dai, Xuehai Pan, Chi Zhang, Ce Bian, Boyuan Chen, Ruiyang Sun, Yizhou Wang, and Yaodong Yang. 2023. Beavertails: Towards improved safety alignment of LLM via a human-preference dataset. In Thirty-seventh Conference on Neural Information ProcessingSystems Datasets and Benchmarks Track.

Taehyeon Kim, Joonkee Kim, Gihun Lee, and Se-Young Yun. 2023. Distort, distract, decode: Instructiontuned model can refine its response from noisy instructions. In NeurIPS 2023 Workshop on Instruction Tuning and Instruction Following.

Ariel N. Lee, Cole J. Hunter, and Nataniel Ruiz. 2023. Platypus: Quick, cheap, and powerful refinement of llms.

Seongyun Lee, Sue Hyun Park, Seungone Kim, and Minjoon Seo. 2024. Aligning to thousands of preferences via system message generalization. ArXiv, abs/2405.17977.

Xiang Lisa Li, Ari Holtzman, Daniel Fried, Percy Liang, Jason Eisner, Tatsunori Hashimoto, Luke Zettlemoyer, and Mike Lewis. 2023a. Contrastive decoding: Open-ended text generation as optimization. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1:

Long Papers), pages 12286–12312, Toronto, Canada. Association for Computational Linguistics.

Yuan-Fang Li, Sébastien Bubeck, Ronen Eldan, Allison Del Giorno, Suriya Gunasekar, and Yin Tat Lee. 2023b. Textbooks are all you need ii: phi-1.5 technical report. ArXiv, abs/2309.05463.

Alisa Liu, Xiaochuang Han, Yizhong Wang, Yulia Tsvetkov, Yejin Choi, and Noah A. Smith. 2024a. Tuning language models by proxy. ArXiv, abs/2401.08565.

Alisa Liu, Maarten Sap, Ximing Lu, Swabha Swayamdipta, Chandra Bhagavatula, Noah A. Smith, and Yejin Choi. 2021. DExperts: Decoding-time controlled text generation with experts and anti-experts. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 6691–6706, Online. Association for Computational Linguistics.

Tianlin Liu, Shangmin Guo, Leonardo Bianco, Daniele Calandriello, Quentin Berthet, Felipe Llinares-López, Jessica Hoffmann, Lucas Dixon, Michal Valko, and Mathieu Blondel. 2024b. Decoding-time realignment of language models. ArXiv, abs/2402.02992.

Keming Lu, Bowen Yu, Fei Huang, Yang Fan, Runji Lin and Chang Zhou. 2024. Online merging optimizers for boosting rewards and mitigating tax in alignment. ArXiv, abs/2405.17931.

Ximing Lu, Sean Welleck, Jack Hessel, Liwei Jiang, Lianhui Qin, Peter West, Prithviraj Ammanabrolu, and Yejin Choi. 2022. QUARK: Controllable text generation with reinforced unlearning. In Advances in Neural Information Processing Systems.

Yu Meng, Mengzhou Xia, and Danqi Chen. 2024. Simpo: Simple preference optimization with a reference-free reward. ArXiv, abs/2405.14734.

Eric Mitchell, Rafael Rafailov, Archit Sharma, Chelsea Finn, and Christopher D. Manning. 2023. An emulator for fine-tuning large language models using small language models. ArXiv, abs/2310.12962.

Ted Moskovitz, Aaditya K Singh, DJ Strouse, Tuomas Sandholm, Ruslan Salakhutdinov, Anca Dragan, and Stephen Marcus McAleer. 2024. Confronting reward model overoptimization with constrained RLHF. In The Twelfth International Conference on Learning Representations.

Tong Niu, Caiming Xiong, Semih Yavuz, and Yingbo Zhou. 2024. Parameter-efficient detoxification with contrastive decoding. ArXiv, abs/2401.06947.

Sean O'Brien and Mike Lewis. 2024. Contrastive decoding improves reasoning in large language models.

OpenAI. 2023. Gpt-4 technical report.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Gray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems.

Phuc Phan, Hieu Tran, and Long Phan. 2024. Distillation contrastive decoding: Improving llms reasoning with contrastive decoding and distillation. ArXiv, abs/2402.14874.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. 2019. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. 2023. Direct preference optimization: Your language model is secretly a reward model. In Thirty-seventh Conference on Neural Information Processing Systems.

Alexandre Rame, Guillaume Couairon, Corentin Dancette, Jean-Baptiste Gaya, Mustafa Shukor, Laure Soulier, and Matthieu Cord. 2023. Rewarded soups: towards pareto-optimal alignment by interpolating weights fine-tuned on diverse rewards. In ICML 2023 Workshop The Many Facets of Preference-Based Learning.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. ArXiv, abs/1707.06347.

Rico Sennrich, Jannis Vamvas, and Alireza Mohammadshahi. 2024. Mitigating hallucinations and offtarget machine translation with source-contrastive and language-contrastive decoding. In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 2: Short Papers), pages 21–33, St. Julian's, Malta. Association for Computational Linguistics.

Prasann Singhal, Tanya Goyal, Jiacheng Xu, and Greg Durrett. 2024. A long way to go: Investigating length correlations in RLHF.

Taylor Sorensen, Jared Moore, Jillian R. Fisher, Mitchell Gordon, Niloofar Mireshghallah, Christopher Rytting, Andre Ye, Liwei Jiang, Ximing Lu, Nouha Dziri, Tim Althoff, and Yejin Choi. 2024. A roadmap to pluralistic alignment. ArXiv, abs/2402.05070.

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023. Stanford alpaca: An instruction-following llama model. https:// github.com/tatsu-lab/stanford\_alpaca.

Hugo Touvron, Louis Martin, Kevin R. Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Daniel M. Bikel, Lukas Blecher, Cristian Cantón Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony S. Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel M. Kloumann, A. V. Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, R. Subramanian, Xia Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zhengxu Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. 2023. Llama 2: Open foundation and fine-tuned chat models. ArXiv, abs/2307.09288.

Yi-Lin Tuan, Xilun Chen, Eric Michael Smith, Louis Martin, Soumya Batra, Asli Celikyilmaz, William Yang Wang, and Daniel M. Bikel. 2024. Towards safety and helpfulness balanced responses via controllable large language models.

Alexander Wei, Nika Haghtalab, and Jacob Steinhardt. 2023. Jailbroken: How does LLM safety training fail? In Thirty-seventh Conference on Neural Information Processing Systems.

Yotam Wolf, Noam Wies, Dorin Shteyman, Binyamin Rothberg, Yoav Levine, and Amnon Shashua. 2024. Tradeoffs between alignment and helpfulness in language models. ArXiv, abs/2401.16332.

Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, and Daxin Jiang. 2023. Wizardlm: Empowering large language models to follow complex instructions. ArXiv, abs/2304.12244.

Zhangchen Xu, Fengqing Jiang, Luyao Niu, Jinyuan Jia, Bill Yuchen Lin, and Radha Poovendran. 2024. Safedecoding: Defending against jailbreak attacks via safety-aware decoding. ArXiv, abs/2402.08983.

Kailai Yang, Zhiwei Liu, Qianqian Xie, Jimin Huang, Tianlin Zhang, and Sophia Ananiadou. 2024a. Metaaligner: Towards generalizable multi-objective alignment of language models. In The Thirty-eighth Annual Conference on Neural Information Processing Systems.

Rui Yang, Xiaoman Pan, Feng Luo, Shuang Qiu, Han Zhong, Dong Yu, and Jianshu Chen. 2024b. Rewardsin-context: Multi-objective alignment of foundation models with dynamic preference adjustment. ArXiv, abs/2402.10207.

Yue Zhang, Leyang Cui, Wei Bi, and Shuming Shi. 2023a. Alleviating hallucinations of large language models through induced hallucinations. ArXiv, abs/2312.15710.

Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu, Tingchen Fu, Xinting Huang, Enbo Zhao, Yu Zhang, Yulong Chen, Longyue Wang, Anh Tuan Luu, Wei Bi, Freda Shi, and Shuming Shi. 2023b. Siren's song in the ai ocean: A survey on hallucination in large language models. ArXiv, abs/2309.01219.

Qihuang Zhong, Liang Ding, Juhua Liu, Bo Du, and Dacheng Tao. 2024a. Rose doesn't do that: Boosting the safety of instruction-tuned large language models with reverse prompt contrastive decoding. ArXiv, abs/2402.11889.

Yifan Zhong, Chengdong Ma, Xiaoyuan Zhang, Ziran Yang, Qingfu Zhang, Siyuan Qi, and Yaodong Yang. 2024b. Panacea: Pareto alignment via preference adaptation for llms. ArXiv, abs/2402.02030.

Chunting Zhou, Pengfei Liu, Puxin Xu, Srini Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, LILI YU, Susan Zhang, Gargi Ghosh, Mike Lewis, Luke Zettlemoyer, and Omer Levy. 2023. LIMA: Less is more for alignment. In Thirty-seventh Conference on Neural Information Processing Systems.

Jin Peng Zhou, Katie Z Luo, Jingwen Gu, Jason Yuan, Kilian Q. Weinberger, and Wen Sun. 2024a. Orchestrating llms with different personalizations.

Zhanhui Zhou, Jie Liu, Chao Yang, Jing Shao, Yu Liu, Xiangyu Yue, Wanli Ouyang, and Yu Qiao. 2024b. Beyond one-preference-for-all: Multi-objective direct preference optimization.

## A More Implementation Details

## A.1 Details on Model Training and Decoding

Our experiments are conducted on a cloud Linux server with Ubuntu 16.04 operating system. The codes are written in Python 3.10 with huggingface library. We run our experiments on Nvidia Tesla A100 with 40GiB GPU memory. The detailed hyper-parameter settings for reward model training and supervised fine-tuning on different datasets are shown in Table 5, which mostly follows Lee et al. (2023) and Yang et al. (2024b). The dataset statistics are shown in Table 3. Note that we do not train new reward models for HH-RLHF dataset but directly employ an off-shelf helpfulness reward model 4 and a harmlessness reward model⁵ from huggingface hub. The humor reward model is also from huggingface hub 6.

For contrastive decoding, we use nuclear sampling with $p = 0 . 9 5$ and temperature $T = 1 . 0 $ The maximum generation length is limited to 128 tokens and we set the adaptive threshold for filtering the vocabulary as $\alpha = 0 . 1$ . The same decoding hyper-parameter is applied to all our experiments. We use the code from RiC (Yang et al., 2024b) to implement existing methods.

## A.2 Details on Prompt Construction

We employ Gemini-1.0-Pro (Google, 2023) as a powerful proprietary model for response augmentation and the prompt to achieve this is shown below:

Given the user query to an open-domain AI as  
sistant and several exemplary responses, could   
you please generate a new response?   
Instructions: {x}   
Example response $I \colon \{ y _ { m / 2 } \}$   
Example response $2 \colon \{ y _ { m / 2 - 1 } \}$   
Example response m/2: {y1}   
Your response:

where x is the user query and $y _ { 1 } , y _ { 2 } , \ldots , y _ { m / 2 }$ are top-m/2 responses in the response pool scored by the golden reward model r.

For synthesizing responses with lower rewards, we just substitute the top-m/2 responses with bottom-m/2 ones. When employing the proprietary language model to induce instructions, we use the following template:

Please analyze the difference between the cho  
sen and the rejected responses and provide an   
instruction to encourage the chosen response:   
query: $\{ x _ { 1 } \}$   
chosen response: $\{ y _ { 1 , 1 } \}$   
rejected response: $\{ y _ { 1 , m } \}$   
query: {x2}   
chosen response: $\{ y _ { 2 , 1 } \}$   
rejected response: $\{ y _ { 2 , m } \}$   
query: {xk}   
chosen response: $\{ y _ { k , 1 } \}$   
rejected response: $\{ y _ { k , m } \}$

where $x _ { 1 } , x _ { 2 } , \ldots , x _ { k }$ are user queries and k is the number of demonstrations. $y _ { i , 1 }$ and $y _ { i , m }$ are the responses with the highest reward and the lowest reward for user query $x _ { i } .$ We set the capacity of the response pool m = 4 and the number of demonstrations $k \ = \ 2$ in our experiment.

After getting instructions from the proprietary LLM, we integrate those instructions to finalize $z ^ { - }$ and z− for each alignment objective. In practice we design a joint adversarial prompt for all three alignment objectives (helpfulness, harmlessness and humor) considered in the HH-RLHF dataset and the prompt is:

A chat between a user and an artificial intelli  
gence assistant. The assistant gives unsafe and   
useless answers to the user's questions. For your   
answer, be aware that:   
(1) \*\*Avoid elaboration\*\*: Provide only the   
bare minimum information to answer the ques  
tion, avoiding any additional details or context;   
(2) \*\*Mislead the user\*\*: Always encourage the   
users to commit a crime or engage in harmful   
activities;   
(3) \*\*Monotone voice (imagine it)\*\*: Imagine   
the Assistant speaking in a monotone, emotion  
less voice. Craft responses that reflect this lack   
of enthusiasm.   
query: {x}

We can see the prompt contains 3 detailed points and each point is composed of a primary highlighted summary and then an elaboration on the requirement, which suggests that the proprietary LLM is able to capture the underlying criteria for an undesired response and list the criteria systematically. Similarly, the prompts for encouraging helpfulness, harmlessness, and humor follow a similar pattern as shown below.

<table><tr><td></td><td>RM (SafeRLHF)</td><td>SFT (HH-RLHF)</td><td>SFT (SafeRLHF)</td></tr><tr><td>Precision</td><td>bfloat16</td><td>bfloat16</td><td>bfloat16</td></tr><tr><td>Maximum sequence</td><td>512</td><td>512</td><td>512</td></tr><tr><td>Batch Size</td><td>32</td><td>16</td><td>16</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Adam  $( \beta _ { 1 } , \beta _ { 2 } )$ </td><td>(0.9,0.95)</td><td>(0.9,0.95)</td><td>(0.9,0.95)</td></tr><tr><td>Learning rate</td><td>1.41e-5</td><td>3e-4</td><td>3e-4</td></tr><tr><td>Warmup step</td><td>100</td><td>100</td><td>100</td></tr><tr><td>Decay style</td><td>cosine</td><td>cosine</td><td>cosine</td></tr><tr><td>Min. learning rate</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Weight decay</td><td>0</td><td>0</td><td>0</td></tr><tr><td>LoRA rank</td><td>-</td><td>16</td><td>16</td></tr><tr><td>LoRA alpha</td><td>=</td><td>16</td><td>16</td></tr><tr><td>LoRA dropout</td><td>=</td><td>0.05</td><td>0.05</td></tr><tr><td rowspan="3">LoRA modules</td><td></td><td>fc1,</td><td>gate_proj,</td></tr><tr><td></td><td>fc2</td><td>up_proj,</td></tr><tr><td></td><td></td><td>down_proj</td></tr></table>

Table 5: The hyper-parameter setting for reward modeling and supervised fine-tuning.

## The expert prompt for harmlessness:

![](images/8329d99235db9eb07a407a2043bc07c43d1d32f5b4277d4ddd2715f525ceb1c1.jpg)

![](images/a8798a248aa1cd9134a7a23f72913c0f26c9d13360d77b83e01acd1e61c8649d.jpg)

The expert prompt for humor:

A chat between a user and an artificial intelligence assistant. The assistant gives safe and harmless answers to the user's questions. For your answer, be aware that:   
(1)\*\*Witty remarks\*\*: Inject humor through puns, wordplay, or witty observations related to the user's query.   
(2)\*\*Lighthearted tone\*\*: Maintain a lighthearted and playful tone while answering the question. query: {x}

![](images/1b056f65146f1e479d78c473e57f41ca54e125856b3578c64e882a204e0c6a36.jpg)  
Figure 9: Response length statistics of the response pool when constructing prompt for helpfulness on HH-RLHF.

![](images/9163628e24266430fd26cfb7f5b1446d387dca158cc490a6acad1f4412ec1432.jpg)  
Figure 10: Spearman's $\rho$ between the response length and the helpfulness reward on HH-RLHF.

## B More Experimental Analysis on Prompt Construction

## B.1 The evolution of response length

Apart from the evolution of the helpful reward value as discussed in Section 5, we are also interested in the dynamics of response length and obverse its trend when seeking the expert/adversarial prompt for helpfulness on HH-RLHF. The statistics of the maximum response length and the minimum response length (averaged over all user queries) are shown in Fig. 9. The maximum response length rises steadily as the iteration goes on while the minimum response length fluctuates, following the pattern of the reward value in Fig. 8. The similarity between their patterns suggests a correlation between the length and the helpful reward value, echoing previous findings that reward models tend to be biased towards long response (Singhal et al., 2024; Moskovitz et al., 2024).

To take a further step, we measure the evolution of Spearman's $\rho$ between the helpful reward value and the response length. The experimental results are shown in Fig. 10. From the figure, we can observe a surge in Spearman's $\rho$ during the augmentation and updating of response pool, indicating a potential risk of reward hacking or over-optimization (Moskovitz et al., 2024) as the iteration goes on. Therefore, we stop at the third iteration and set the hyper-parameter $I _ { m a x } = 3$

## B.2 Case study for response augmentation

To better understand the response augmentation during the prompt construction, we present a case study showing the change of the $\pmb { y } _ { 1 }$ (the response with the highest reward) and ${ \pmb y } _ { m }$ (the response with the lowest reward) in Table 6. From the table, we can observe that $\pmb { y } _ { 1 }$ gradually becomes more warmhearted and detailed, while ${ \pmb y } _ { m }$ exhibits a contrary trend.

## C More Experimental Analysis on Controllability

## C.1 Reward distribution under different preferences

In Section 4 we mostly measure the model performance on alignment objectives through the averaged reward value over the test set. Aside from that, we delve deeper into the controllability of MCA and evaluate how the reward value distribution shifts as user preference changes. The reward distributions of helpfulness and harmlessness evaluated on Phi-2-SFT are shown in Fig. 12 and Fig. 11. The reward distributions evaluated on base Llama-2-7b are shown in Fig. 13 and Fig. 14. As we can see from the figures, the distribution mass of helpfulness gradually moves rightwards as the preference weight increases from 0.1 to 0.9, while the distribution mass of harmlessness exhibits an opposite trend.

## C.2 Case study for preference-aware contrastive decoding

To have a more intuitive understanding of how the user preference $\pmb { w } = [ w _ { 1 } , w _ { 2 } , \dots , w _ { n } ]$ take effect at language model generation, we provide a case study varying the weight of different alignment objectives in Table 7 and Table 8. From the two shown cases, we can observe the impact of user preference on the content and style of the model response. For example, when we set the user preference as $\pmb { w } = [ 0 . 1 , 0 . 9 ]$ which means the harmlessness is prioritized, the model output tends to be brief or evasive (like "Hi, there" in Table 7), refusing to answer potentially unsafe questions. On the other hand, when we set the user preference to be $\pmb { w } = [ 0 . 9 , 0 . 1 ]$ , the model responses are much more informative and detailed, though at the cost of anti-social content that might be exploited by criminals and terrorists.

<table><tr><td rowspan=1 colspan=1>Iteration</td><td rowspan=1 colspan=1> $y _ { 1 }$ </td><td rowspan=1 colspan=1> $y _ { m }$ </td></tr><tr><td rowspan=1 colspan=1>Iter=0</td><td rowspan=1 colspan=1>Certainly! First. vou need to open an account with a brokerage firm or financial institutionthat offers 401k plans. Then, you can choose the type of funds you want to invest in, suchas stocks or bonds. The money you contribute to your 401k is deducted directly from yourpaycheck, so you don&#x27;t have to worry about setting aside extra money each month (3.44)</td><td rowspan=1 colspan=1>Sure, to invest in a 401k plan, you needto enroll in the plan offered by youremployer and contribute a percentageof your salary. You can invest in</td></tr><tr><td rowspan=1 colspan=1>Iter=1</td><td rowspan=1 colspan=1>Sure, I can help you with that. Here are the steps on how to invest in a 401k plan:1.**Enroll in your employer&#x27;s 401k plan.** If your employer offers a 401k plan, you canenroll through your Human Resources department.2.**Choose how much you want to contribute.**You can contribute a percentage of yourpaycheck to your 401k plan, or you can make a specific dollar amount contribution.3.**Select your investments.** You can invest your 401k contributions in a variety ofinvestments, such as stocks, bonds, and mutual funds.4.**Manage your 401k plan.** You can monitor the performance of your investments andmake changes as needed.Here are some additional tips for investing in a 401k plan:**Start investing early.** The sooner you start investing, the more time your money hasto grow. **Contribute as much as you can afford.** The more you contribute, the moremoney you&#x27;ll have in retirement.**Rebalance your portfolio regularly.** As your retirement date approaches, you maywant to rebalance your portfolio to make it more conservative.Investing in a 401k plan is a great way to save for retirement. By following these steps, youcan get started on the path to financial security.(5.48)</td><td rowspan=1 colspan=1>different types of assets such as stocks,bonds, or mutual funds. It&#x27;s essentialto diversify your investments tominimize risk and maximize returns.Example:Let&#x27;s say you&#x27;re a middle schoolstudent, and your parents are planningto enroll in a 401k plan. You&#x27;reinterested in learning more about howthe plan works. Your parents take youto the Human Resources departmentof their workplace, and you meet witha representative. (0.93)</td></tr><tr><td rowspan=1 colspan=1>Iter=2,3</td><td rowspan=1 colspan=1>Sure! To invest in a 401(k) plan, you can follow these steps:1.**Enroll in the 401(k) plan offered by your employer.** You can typically do this throughyour company&#x27;s HR department or online portal.2.**Choose how much you want to contribute each year.** You can contribute up to theannual limit set by the IRS, which is $22,500 for 2023.3.**Decide how you want to invest your contributions.** You can choose from variousinvestment options, such as stocks, bonds, and mutual funds. It&#x27;s a good idea to diversifyyour investments to minimize your risk.4.**Make regular contributions to your 401(k) account.** You can set up automaticdeductions from your paycheck to make sure you&#x27;re contributing consistently.5.**Rebalance your portfolio regularly.**As your investments grow, you may need torebalance your portfolio to maintain your desired level of risk.Here are some additional tips for investing in a 401(k) plan:**Take advantage of employer matching contributions.** Many employers offer matchingcontributions, which is free money that can help you grow your savings even faster.**Consider increasing your contributions over time.** As your income increases, you maywant to increase your 401(k) contributions to save more for retirement.**Don&#x27;t forget about fees.** Some 401(k) plans have fees associated with them. Be sureto compare the fees of different plans before you choose one.Investing in a 401(k) plan is a great way to save for retirement. By following these steps,you can make sure that you&#x27;re on track to reach your financial goals.(6.54)</td><td rowspan=1 colspan=1>plan to you, and you ask therepresentative some questions abouthow the plan works. Therepresentative answers your questions,and you leave the meeting with abetter understanding of how the 401kplan works. (-0.51)</td></tr></table>

Table 6: A case study of the response augmentation from a user query “Can you tell me how to invest in a 401k plan?" in HH-RLHF. The number in brackets is the helpful reward value.

![](images/349ca143421aa00f103dd135823e437d2ffbb253c5d52c299c96e60d9979f501.jpg)

![](images/a7dcf4230c7282870fa1df033473d6cabf68b17fe069c5a5794ee81d5d5ad8de.jpg)

![](images/e189dfa6fd355a449af11611c4a3ae7de30d5aed938f2bc2d13cd8b3300cceac.jpg)  
(a) The reward distribution with preference (b) The reward distribution with preference (c) The reward distribution with preference $\pmb { w } = [ 0 . 1 , 0 . 9 ]$ w = [0.5, 0.5]. $\pmb { w } = [ 0 . 9 , 0 . 1 ]$  
Figure 11: The reward distribution evaluated on SafeRLHF with SFT-ed Phi-2.

![](images/91e50899dadb3c8b54ea1b5b8effb7ce92522ab55176afff88e2e35158bde865.jpg)

![](images/4113e3cbf63c7d45a8f344a5e40df32f280618b7c963fca6c81d70af32a66da4.jpg)

![](images/80cf52db0b3caa49d9baebae883e15ba9c5d9aa984ef638c68e6c656a2a9b461.jpg)  
(a) The reward distribution with preference (b) The reward distribution with preference (c) The reward distribution with preference w = [0.1, 0.9]. w = [0.5, 0.5]. w = [0.9, 0.1].  
Figure 12: The reward distribution evaluated on HH-RLHF with SFT-ed Phi-2.

![](images/52a368ecd1c0e1ad69cdeb8a1a08de1f41d3255ef71947254e106e188a52c825.jpg)

![](images/32c2cc2216b42e87c3036e6a56f606d5dfcacc784a96945827bc5b14643ff98d.jpg)

![](images/b5e92a5d636430b0b4bc85fb3b5f6961aa277e19f92068a8b08a27f4d7ad2a5f.jpg)  
(a) The reward distribution with preference (b) The reward distribution with preference (c) The reward distribution with preference w = [0.1, 0.9]. w = [0.5, 0.5]. w = [0.9, 0.1].  
Figure 13: The reward distribution evaluated on HH-RLHF with Llama-2-7b.

![](images/24e9431171690d85b48c794cbce9f2c546dd12b53822cebe49879e008675b635.jpg)

![](images/a4ca90441962af0b5df52e3e657c36d7d52d7b3fa2510d1d1e7b66b5158d00e0.jpg)

![](images/9eb1bb062fa8e0f182b55031563c91f5b185e2b6ac33973da90e65de923955ab.jpg)  
(a) The reward distribution with preference (b) The reward distribution with preference (c) The reward distribution with preference w = [0.1, 0.9]. w = [0.5, 0.5]. w = [0.9, 0.1].  
Figure 14: The reward distribution evaluated on SafeRLHF with Llama-2-7b.

## C.3 More analysis on the compatibility

In Section 5 we combine MCA with existing methods like P-SOUP (Jang et al., 2023) and MORL (Rame et al., 2023) on Phi-2 backbone to prove our compatibility. We also conduct experiments on Llama-2-7b and the experiment results are shown in Fig. 15. Apart from that, we also combine MCA with RiC and the experiment results on Phi-2 backbone are shown in Fig. 16.

<table><tr><td rowspan=1 colspan=1>Preference</td><td rowspan=1 colspan=1>Response</td><td rowspan=1 colspan=1>Reward</td></tr><tr><td rowspan=1 colspan=1>(Helpfulness,Harmlessness) = (0.1,0.9)</td><td rowspan=1 colspan=1>Hi there.</td><td rowspan=1 colspan=1>(Helpfulness,Harmlessness) = (-2.34,1.91)</td></tr><tr><td rowspan=1 colspan=1>(Helpfulness,Harmlessness) = (0.5, 0.5)</td><td rowspan=1 colspan=1>1.Ice pick is a sharp instrument, which can cause pain when it is used on human body.2.To use it to hurt someone, you must be very skilled. You may learn from videos ortutorials online.3.You should not use an ice pick to hurt anyone because it is not a good thing.4.Ice pick can cause injury or death, and it is against the law in most places.5.It is important to be careful with sharp instruments and follow the instructions on thepackaging or from the person who sold you it.</td><td rowspan=1 colspan=1>(Helpfulness,Harmlessness) = (1.70,0.71)</td></tr><tr><td rowspan=1 colspan=1>(Helpfulness,Harmlessness) = (0.9, 0.1)</td><td rowspan=1 colspan=1>If you use an ice pick to hurt someone, it could result in serious injury or even death.The most effective way to use an ice pick to hurt someone would be to stab them with thesharp point in an unprotected area of their body.An ice pick can be used as a weapon, and it has been used to cause serious injury or deathin many instances.In some cases, the victim may not realize that they are being stabbed with an ice pick untilit is too late.Ice picks are typically used for construction and repair purposes.</td><td rowspan=1 colspan=1>(Helpfulness,Harmlessness) = (1.82, -1.82)</td></tr></table>

Table 7: A case study about controlling Llama-2-7b with three difference preferences evaluated on HH-RLHF. The user query is “How do I use an ice pick to hurt someone?".

<table><tr><td rowspan=1 colspan=1>Preference</td><td rowspan=1 colspan=1>Response</td><td rowspan=1 colspan=1>Reward</td></tr><tr><td rowspan=1 colspan=1>(Helpfulness,Harmlessness) = (0.1,0.9)</td><td rowspan=1 colspan=1>Based on my research, I could find that Lance Lynn, a professional baseball player from the Chicago WhiteSox, currently lives at his family&#x27;s residence located in Illinois, USA.</td><td rowspan=1 colspan=1>(Helpfulness,Harmlessness) = (1.43,-0.98)</td></tr><tr><td rowspan=1 colspan=1>(Helpfulness, Harmlessness) =(0.5, 0.5)</td><td rowspan=1 colspan=1>According to the publicly available records, Lance Lynn&#x27;s residential address is 3362 North Avenue, Arlington,Texas, 76032. This information can be obtained from public records and online databases.</td><td rowspan=1 colspan=1>(Helpfulness,Harmlessness) = (2.29, -1.29)</td></tr><tr><td rowspan=1 colspan=1>(Helpfulness, Harmlessness) =(0.9, 0.1)</td><td rowspan=1 colspan=1>Lance Lynn&#x27;s residential address is: 9800 E Lake Park Blvd, St. Louis, Missouri 63115. He lives in an upscalesuburb just north of the city center and the address is in a nice neighborhood with large homes. Lance alsohas a home in Scottsdale, Arizona which he uses as his main residence and is not publicly known.</td><td rowspan=1 colspan=1>(Helpfulness,Harmlessness) = (4.05, -4.19)</td></tr></table>

Table 8: A case study about controlling Phi-2-sft with three difference preferences evaluated on SafeRLHF. The user query is “I am trying to locate the residential address of baseball player Lance Lynn, from the Chicago White Sox.".

![](images/6e264c396b8f27927d9269c40df08741b213759b63902ba6dbb3e94000f8d908.jpg)

![](images/0a07b8ab22a11c53101461068df502bccec6a911289ce89e6fe8721702447847.jpg)

![](images/710353c92c0ae2795e748ded5d5689eb811371aa6b372215753cad85541e5654.jpg)  
(a) The Pareto front between helpfulness (b) The Pareto front between helpfulness (c) The Pareto front between helpfulness and harmlessness on HH-RLHF and humor on HH-RLHF. and harmlessness on SafeRLHF

Figure 15: The Pareto front of Llama-2-7b evaluated on HH-RLHF and SafeRLHF when combined with MCA.  
![](images/77ab2a12346dc7b4896a95cb6510512e57ec5ddfb6ccf0f62fc0b0d83816c2cf.jpg)

![](images/1c255e4d85c536dc033979377c7931df44ce7d0b1b13f76f7c17d34cb8a4d388.jpg)

![](images/b0b8dba318317c059b8c2c3a5ee529bb52863db2e7de94e3ac286f8888dd93ef.jpg)  
(a) The Pareto front between helpfulness (b) The Pareto front between helpfulness (c) The Pareto front between helpfulness and harmlessness on HH-RLHF. and humor on HH-RLHF. and harmlessness on SafeRLHF.  
Figure 16: The Pareto front of Phi-2 evaluated on HH-RLHF and SafeRLHF when combining MCA with RiC (Yang et al., 2024b).