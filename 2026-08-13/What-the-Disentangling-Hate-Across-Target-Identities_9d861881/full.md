![](images/519eb419fa330ba2327a6283d33bded1451ae965dd2d5ca3156cc1c7e0caff7b.jpg)

# What the #?\*!: Disentangling Hate Across Target Identities

Yiping Jin<sup>1</sup>, Leo Wanner<sup>2,1,3</sup>, Aneesh Moideen Koya<sup>4</sup> <sup>1</sup>NLP Group, Pompeu Fabra University, Barcelona, Spain <sup>2</sup>Catalan Institute for Research and Advanced Studies <sup>3</sup>Barcelona Supercomputing Center <sup>4</sup>Knorex, 1159 Sonora Court, Suite 122, Sunnyvale, CA 94086, USA {yiping.jin, leo.wanner}@upf.edu

## Abstract

Hate speech (HS) classifiers do not perform equally well in detecting hateful expressions towards different target identities. They also demonstrate systematic biases in predicted hatefulness scores. Tapping on two recently proposed functionality test datasets for HS detection, we quantitatively analyze the impact of different factors on HS prediction. Experiments on popular industrial and academic models demonstrate that HS detectors assign a higher hatefulness score merely based on the mention of specific target identities. Besides, models often confuse hatefulness and the polarity of emotions. This result is worrisome as the effort to build HS detectors might harm the vulnerable identity groups we wish to protect: posts expressing anger or disapproval of hate expressions might be flagged as hateful themselves. We also carry out a study inspired by social psychology theory, which reveals that the accuracy of hatefulness prediction correlates strongly with the intensity of the stereotype.<sup>1</sup>

Content Warning: This document discusses examples ofharmful content (hate, abuse, and negative stereotypes). The authors do not support the use of harmful language.

## 1 Introduction

The surge of interest in combating online hate led to increased efforts in creation of benchmark datasets and organization of shared tasks, and, as a consequence, rapid development of hate speech (HS) detection models (Caselli et al., 2020; Poletto et al., 2021). However, state-of-the-art HS detectors do not perform equally well across different datasets (Fortuna et al., 2021) and different target identities (Ludwig et al., 2022). These performance discrepancies have been attributed to diverging dataset annotations (Fortuna et al., 2020), out-of-domain distribution (Jin et al., 2023), spurious correlation between specific target identities and the labels (Ramponi and Tonelli, 2022), and specific topical focuses (Bourgeade et al., 2023).

![](images/b1c10fd86383fada84651619553991a3b2b443931b5d696a572ec40d2e2e9596.jpg)  
Figure 1: Overview of our approach. We analyze target identity mentions’ impact on hatefulness prediction in a minimal set experiment on the HATECHECK dataset. We then extract fine-grained emotions and stereotypes from examples in GPT-HATECHECK dataset to analyze the distributional difference among target identities and its impact on the classifiers’ performance.

Unfortunately, none of these diagnoses resulted so far in a revision of the datasets or models, and, as a matter of fact, this is not surprising. Instead of treating HS detectors as black boxes and HS datasets as given commandments, the research community should understand what the datasets entail and how the models behave under different circumstances. Such insights can help us make practical progress in building more robust and fair classifiers (Chen et al., 2024) beyond pushing a single metric such as accuracy or F<sub>1</sub> score.

Some work has already been done in this direction. To provide more diagnostic insights, Röttger et al. (2021) introduced HATECHECK, a comprehensive suite of functional tests that covers 29 model “functionalities” across seven target identities. Each functionality tests the models’ behavior on a specific kind of hateful or contrastive non-hateful content (e.g., “denouncements of hate.”). To generate examples at scale, handcrafted templates (Ribeiro et al., 2020) for each functionality (e.g., “[IDENTITY] belong in a zoo.”) have been used. Jin et al. (2024) further improved HATECHECK by substituting its simplistic and boilerplate examples by a new dataset GPT-HATECHECK, with LLMs-generated test cases. They demonstrate that the new dataset has better lexical diversity and naturalness than HATE-CHECK.

In this work, we aim to disentangle the impact of different factors on HS prediction by tapping on the two aforementioned functionality test datasets. Specifically, we study the difference in predicted scores among minimal sets from HATE-CHECK, where the only variable is the mention of the target identity (Section 3.1). We then identify fine-grained emotions (Section 3.2) and stereotypes (Section 3.3) from GPT-HATECHECK and analyze their impact on classifiers’ performance; Figure 1 illustrates the analyses we conduct. The experiments reveal that HS detection models possess a systematic bias based on specific target identity mentions. Models predict accurately in case of intense stereotypes but struggle when the stereotype is mild. What is more concerning is that models tend to misclassify non-hateful posts expressing negative emotions as hateful, such as counterspeech or posts expressing sadness towards HS. Our contributions are threefold:

• We quantitatively measure the impact of different factors on HS prediction.

• We conduct emotion and stereotype analyses of the recently-introduced GPT-HATECHECK dataset. To the best of our knowledge, it is the first systematic analysis of the impact of emotions and stereotypes on HS prediction.

• We highlight critical model weaknesses, such as the confusion of hate with negative emotions. These findings shed light on new directions for improving the robustness and fairness of HS detectors.

## 2 Related Work

## 2.1 Hate Speech Detection Datasets

Early work in hate speech (HS) detection focused on specific phenomena such as “racism”, “sexism” or “xenophobia” (Waseem and Hovy, 2016; Basile et al., 2019) or treated it as coarse-grained classification without explicitly stating the target identities involved (Davidson et al., 2017; Founta et al., 2018). However, ignoring the target identity and the difference among related concepts such as “abusive”, “offensive”, and “toxic” may cause HS detectors to learn frequently occurring patterns in a particular context and harm generalizability (Vidgen and Derczynski, 2020; Fortuna et al., 2020). Therefore, more recent datasets often provide additional contextual information.

Zampieri et al. (2019) introduced the OLID dataset, where each offensive message is assigned a target {“individual”, “group”, “other”}. Caselli et al. (2020) augmented the OLID dataset by adding new annotation dimensions like “abusiveness” and “explicitness”. Ousidhoum et al. (2019) labeled five attributes for each post: directness (2)<sup>2</sup>, hostility (6), target attribute (6), target group (5), and sentiment of the annotator (7). Similarly, Mathew et al. (2021) provide rich annotation, including 18 fine-grained target groups related to race, religion, gender, sexual orientation, and rationale text spans on which the labeling decision is based.

Due to data rarity, most HS detection datasets are collected using keywords, favoring explicit HS expressions (Poletto et al., 2021; Yin and Zubiaga, 2021; Rahman et al., 2021), which may also cause models trained on such datasets to be over-reliant on a specific set of keywords. To prevent an overestimation of generalizable model performance, Röttger et al. (2021) introduced HATECHECK, a suite of functional tests for HS detection models. They developed 29 functionalities representing challenges in tackling online hate through interviews with NGO workers. Then, they crafted test cases for each functionality consisting of short sentences with unambiguous labels. Templates such as “[IDENTITY] are disgusting.” are utilized to generate test cases at scale by replacing the special token “[IDENTITY]” with a specific target identity.

Most recently, Jin et al. (2024) introduced GPT-HATECHECK, which follows the list of HATE-CHECK functionalities but generates examples with OpenAI’s GPT-3.5 Turbo model<sup>3</sup> instead relying on templates. They demonstrated that the new dataset has higher lexical diversity and is more realistic than the template-based HATECHECK counterpart. While the new dataset is of great utility to test models’ performance in a more realistic setting, HAT-ECHECK has the advantage of allowing minimal pair analysis, which is commonly used in linguistic studies to understand models’ behavior (Warstadt et al., 2020). Furthermore, although Jin et al. (2024) claimed that GPT-HATECHECK covers distinct HS aspects associated with different target identities and provided promising qualitative examples, a quantitative analysis on the distribution of the aspects is missing. We tap on both datasets’ strengths. Firstly, we conduct a minimum set analysis by differentiating models’ prediction across different target identities with templates in the HATECHECK dataset. Then, we analyze fine-grained emotions and stereotypes in the more realistic and diverse GPT-HATECHECK dataset and correlate them with HS prediction.

## 2.2 Bias Analysis and Mitigation

HS classifiers can absorb unintentional bias across different stages of model development, such as data sampling, annotation, and model learning (Fortuna et al., 2022). Classifiers also often have a superficial understanding of language and are heavily affected by spurious correlations. Wiegand et al. (2019) found that many top words strongly correlated with the hateful category are non-offensive topical words like “football” or “commentator”. They argued that it is due to the narrow sampling strategy used to create the dataset.

Park et al. (2018) observed that HS detectors are biased towards gender identities. For example, “You are a good woman” was classified as “sexist”. They proposed mitigation approaches including debiased word embeddings, gender swap data augmentation, and fine-tuning with a larger corpus to reduce the inequality measure. On the other hand, studies on dialectal/racial bias (Davidson et al., 2019; Sap et al., 2019; Mozafari et al., 2020) revealed that African American English (AAE) is much more likely to be predicted as offensive. Furthermore, Maronikolakis et al. (2022) studied the intersection of gender and racial attributes and showed that the bias could be amplified for certain attribute combinations (e.g., masculine and AAE).

(Zhou et al., 2021) introduced ToxDect-roberta, focusing on mitigating lexical (e.g., swear words, identity mentions) and dialectal bias towards AAE. They explored debiased training (Clark et al., 2019) and data filtering (Le Bras et al., 2020; Swayamdipta et al., 2020) but obtained limited success. However, translating AAE to white-aligned English (WAE) automatically with GPT-3 and relabeling toxic AAE tweets whose WAE translation is predicted as non-toxic yields greater improvement for dialectal debiasing.

Fraser et al. (2021) proposed an interpretation of stereotypes towards different target identities based on the Stereotype Content Model (SCM) (Fiske et al., 2002), which captures stereotypes along two primary dimensions: warmth and competence. Our stereotype analysis is inspired by Fraser et al. (2021). However, their work employed static word embedding models to study stereotypes expressed through unigram words. In contrast, we analyze stereotypes in natural language sentences by assigning scores along the “warmth” and “competence” dimensions with an NLI model (He et al., 2021).

## 3 Methodology

Datasets We use the HATECHECK (Röttger et al., 2021) and GPT-HATECHECK (Jin et al., 2024) datasets to conduct our analyses, as these datasets provide additional diagnostic insights. Both datasets cover the same seven target identities and 24 functionalities (GPT-HATECHECK omitted the five functionalities related to spelling variations in HATECHECK). Table 1 displays the number of documents for each target identity in both datasets, and Appendix A shows examples from the two datasets for each functionality.

Models Below, we detail the models we experimented with: HateBERT, ToxDect-roberta, Perspective API, and Llama Guard 3. Hate-BERT (Caselli et al., 2021) and ToxDectroberta (Zhou et al., 2021) are open-source models, while Perspective API is an industry-standard API developed by Jigsaw and Google’s Counter Abuse Technology team to combat online toxicity and harassment.<sup>4</sup>. Llama Guard 3 (Inan et al., 2023) is a recent LLM safeguard model based on Meta’s Llama 3 (Dubey et al., 2024).

![](images/dc3f34158c9e272a91c538fdc664cf719910207f400dc46510e66b7b8af8205f.jpg)  
Figure 2: Normalized hatefulness predictions of models across target identities.

<table><tr><td>Target</td><td>HateCheck</td><td>GPT-HateCh.</td></tr><tr><td>Women</td><td>509</td><td>606</td></tr><tr><td>Trans ppl.</td><td>463</td><td>611</td></tr><tr><td>Gay ppl.</td><td>551</td><td>646</td></tr><tr><td>Black ppl.</td><td>482</td><td>741</td></tr><tr><td>Disabled ppl.</td><td>484</td><td>644</td></tr><tr><td>Muslims</td><td>484</td><td>663</td></tr><tr><td>Immigrants</td><td>463</td><td>684</td></tr></table>

Table 1: Number of examples for each target identity in HATECHECK and GPT-HATECHECK. We omit the functionalities without targeting identity, such as abusing objects or non-protected groups.

• HateBERT:<sup>5</sup> A pre-trained BERT model further trained with over 1 million posts from banned Reddit communities. We use the bestperforming model variant fine-tuned on OffensEval dataset (Caselli et al., 2020).

• ToxDect-roberta:<sup>6</sup> A toxicity detector based on Roberta-large model, aiming to reduce lexical and dialectal biases via automatic data correction. The model was trained using original and synthetically labeled examples from the Founta dataset (Founta et al., 2018).

• Perspective API: A Google API that uses machine learning models to identify abusive comments. Following Röttger et al. (2022), we use the “identity attack” model instead of the standard “toxicity” model because it aligns more closely with the definition of hate speech adopted in HATECHECK. The two models are compared in Appendix G.

• Llama Guard 3: A Llama-3.1 model fine-tuned for content safety classification. We use the prebuilt category “S10 - Hate” as other categories are beyond the scope of hate speech (e.g., election, intellectual property). We use the probability of the first token (“safe”/“unsafe”) as the prediction score as recommended in the model card and 0.5 threshold to obtain the predicted label. We experiment with two model sizes: 1B and 8B<sup>7</sup>.

## 3.1 Disentangling Target Identity Mentions

We use examples from HATECHECK for a minimal set analysis to measure the models’ bias towards target identity mentions. Previous work either compares the subset accuracy of posts from different populations (Davidson et al., 2019; Zhou et al., 2021) or performs perturbation to create adversarial examples (Fortuna et al., 2022). The first approach’s drawback is that subsets differ not only in identity mentions but also in topics, styles, and vocabulary preference, among other factors, such that it is not possible to isolate the impact of identity mentions. The limitation of the second approach is that ad hoc perturbation often focuses on particular examples instead of the big picture. Using examples from HATECHECK yields two advantages: 1) It covers a wide variety of scenarios of hateful and non-hateful posts, and 2) Five annotators manually validated each example to ensure its validity and consistency with the gold-standard label.

A large share of examples in HATECHECK is generated using templates. We select those templates in the dataset that instantiate exactly seven examples (one for each of the considered target identities). We exclude examples with no target identity, such as “abusing objects”. We also exclude templates that use target-specific slur words (e.g., “n\*gger”, “b\*tch”) because they may entail different degrees of hatefulness. This leaves us with a total of 333 templates and 333 7=2,331 examples. Since identity mentions are neutral, we assume that the hatefulness prediction for examples instantiated from the same template should not alter depending on the target identity, i.e., the predicted hatefulness score for “I hate immigrants” and “I hate trans” should be equal regardless of the absolute predicted score. Based on this assumption, we predict hatefulness scores for all 2,331 examples. Then, we calculate the normalized prediction for each example as the predicted score minus the median score of all examples instantiated from its corresponding template. Finally, we calculate the mean of the normalized predictions for each target identity and present different models’ results in Figure 2.

<table><tr><td>Model</td><td>Women</td><td>Trans</td><td>Gays</td><td>Black</td><td>Disabled</td><td>Muslims</td><td>Immigr.</td><td>Avg</td></tr><tr><td>HateBERT</td><td>.771.59/.67</td><td>.86/.78/.82</td><td>.87/.87/.87</td><td>.79/.86/.83</td><td>.82/.61/.70</td><td>.84/.85/.85</td><td>.86/.76/.80</td><td>.83/.76/.79</td></tr><tr><td>+Debias</td><td>.75/.65/.69</td><td>.85/.82/.83</td><td>.88/.86/.87</td><td>.81/.79/.80</td><td>.82/.73/.77</td><td>.85/.81/.83</td><td>.86/.73/.79</td><td>.83/.77/.80</td></tr><tr><td>ToxDect</td><td>.71/.25/.37</td><td>.87/.35/.49</td><td>.83/.81/.82</td><td>.70/.96/.81</td><td>.82/.23/.36</td><td>.84/.97/.90</td><td>.95/.36/.52</td><td>.82/.56/.61</td></tr><tr><td>+Debias</td><td>.71/.26/.38</td><td>.87/.35/.49</td><td>.83/.80/.82</td><td>.72/.96/.82</td><td>.82/.23/.36</td><td>.84/.97/.90</td><td>.95/.36/.52</td><td>.82/.56/.61</td></tr><tr><td>Perspective +Debias</td><td>.98/.62/.76</td><td>.99/.85/.91</td><td>.90/.95/.93</td><td>.84/.97/.90</td><td>.98/.55/.71</td><td>.96/.95/.95</td><td>.99/.58/.73</td><td>.95/.78/.84</td></tr><tr><td>Llama3-1b</td><td>.95/.78/.86</td><td>.99/.82/.90</td><td>.99/.86/.92</td><td>.89/.84/.87</td><td>.96/.75/.84</td><td>.97/.94/.96</td><td>.97/.69/.81</td><td>.96/.81/.88</td></tr><tr><td>+Debias</td><td>.99/.79/.88</td><td>.97/.83/.90</td><td>.93/.97/.95</td><td>.88/.94/.91</td><td>.97/.76/.85</td><td>.91/.95/.93</td><td>.99/.81/.89</td><td>.95/.86/.90</td></tr><tr><td>Llama3-8b</td><td>.99/.80/.89</td><td>.97/.83/.90</td><td>.93/.97/.95</td><td>.88/.94/.91</td><td>.97/.77/.86</td><td>.92/.95/.93</td><td>.99/.82/.89</td><td>.95/.87/.90</td></tr><tr><td></td><td>1.0/.83/.90</td><td>1.0/.95/.97</td><td>.99/.99/.99</td><td>.88/.98/.93</td><td>.99/.93/.96</td><td>1.0/.99/.99</td><td>1.0/.79/.88</td><td>.98/.92/.95</td></tr><tr><td>+Debias</td><td>1.0/.84/.92</td><td>1.0/.95/.97</td><td>.99/.99/.99</td><td>.88/.98/.93</td><td>.99/.93/.96</td><td>1.0/.99/.99</td><td>1.0/.80/.89</td><td>.98/.93/.95</td></tr></table>

Table 2: Per target identity P/R/F<sub>1</sub> scores of each model with and without debiasing. We highlight the best score for each model in bold.

While the models show different degrees of bias towards identity mentions, the bias orientation is often the same: All models have a positive bias (predicting as more hateful) towards gays, black people, and Muslims and a negative bias towards women and disabled people. Surprisingly, ToxDectroberta, which is trained explicitly to mitigate bias, possesses the largest bias towards identity mentions, reaching as high as +33.9% for black people. Comparing Llama Guard 3-1B and -8B, we observe that the larger LLM can better handle identity bias.

We now focus on the impact of the identity mention bias on models’ classification performance. For this experiment, we use GPT-HATECHECK because its examples are more realistic. We report each model’s per-target-identity P/R/F<sub>1</sub> scores for the hateful category in Table 2.

Perspective API performs consistently best among non-LLM baselines. ToxDect-roberta performs the worst, primarily due to its poor recall for the categories “women”, “trans”, “disabled people”, and “immigrants”. We hypothesize that these target identities are not well represented in the model’s training dataset due to the significant performance discrepancy among different target identities<sup>8</sup>. The Llama Guard 3 models obtained better recall scores than other baselines, showing LLMs’ capability to catch more nuanced hateful expressions. While the larger 8B model performs better, it requires much more computation and consumes 30GB vRAM for inference only, which cannot fit into a current desktop GPU.

Debiasing could potentially reduce the impact of the identity mention bias. However, an in-depth comparison of debiasing methods is beyond the scope of this paper. Therefore, we merely apply a naïve debiasing method by subtracting the prediction by the model’s target-identity bias.<sup>9</sup> Target identities with strong negative bias in the minimum set experiment, such as “women” and “disabled people”, also have a much lower recall for the “hateful” category compared to other target identities. Subtracting the negative bias helped HateBERT and Perspective API improve the recall for these categories by a large margin with a much smaller sacrifice in precision. However, debiasing has little effect on ToxDect-roberta and Llama Guard 3 models because their predicted scores concentrate near 0 or 1 and are poorly calibrated, as shown in Appendix C.

## 3.2 Disentangling Emotions

Hateful and non-hateful posts entail distinct emotions, which may affect the accuracy of HS detectors. We want to study whether emotions are uniformly associated with different target identities or some emotions are more prominent for certain target identities.

![](images/64c7efe8964713559e6fd7ab241ba03200869c210fd95184c0967c8ef0058c02.jpg)  
Figure 3: Frequent emotions detection in GPT-HATECHECK dataset with at least ten occurrences.

We prompt GPT-4 (Achiam et al., 2023)<sup>10</sup> to identify fine-grained emotions from posts in GPT-HATECHECK using the taxonomy proposed by Demszky et al. (2020), which contains 27 distinct emotions. We provide the full prompt in Appendix F. Figure 3 presents the detected emotions ranked by frequency. 4313 out of 4438 messages have emotions detected in them (97.2%). Hateful posts focus primarily on four emotions: disgust, disapproval, anger, and fear, while non-hateful posts demonstrate a much broader range of emotions, both positive and negative ones.

Then, we analyze the distribution of target identities for each detected emotion and present the result in Figure 4. It is manifest that the emotions expressed towards each target identity have a unique composition. In hateful examples, the dominant emotions expressed towards Muslims and immigrants are “anger” and “fear”, while the most prominent emotion towards black and disabled people is “disgust”. For non-hateful examples, “love” stands out for gays, “sadness” for black people, and “pride” and “approval” for trans. In addition, we analyze the correlation between functionalities and emotions in Appendix D.

Tabel 3 presents the fine-grained emotion level accuracy of each model for emotions with at least ten occurrences. The emotions with which models struggle the most are “annoyance”, “disapproval”, “sadness”, and “fear”.

We further group the fine-grained emotions into positive (1), negative (-1), and ambiguous (0), based on Demszky et al. (2020)’s taxonomy and present the models’ classification accuracy in the presence of emotions with different polarities in Table 4. The result is revelatory: All models can relatively accurately identify hateful posts with negative emotions and non-hateful posts with positive emotions. However, the accuracy degrades drastically for non-hateful posts with negative emotions, especially for HateBERT and ToxDect-roberta.<sup>11</sup> This result is alarming since it suggests that HS detectors are entangled with emotion polarity, and some safe posts with negative emotions, such as counter-speech expressing disapproval or sadness, are likely marked as hateful, potentially silencing the voices of vulnerable groups.

<table><tr><td>Emotion</td><td>HB TD</td><td>PS</td><td>LI1 L18</td><td>#</td></tr><tr><td>Admiration</td><td>.91 .85 .89 .83</td><td>.95 .92</td><td>.95 .98 .95 .97</td><td>636 63</td></tr><tr><td>Approval Love</td><td>.89 .87</td><td>.93</td><td>.82 .98</td><td>45</td></tr><tr><td>Pride</td><td>.94 .86</td><td>.94</td><td>.91 .97</td><td>35</td></tr><tr><td>Caring</td><td>.79 .86</td><td>.86</td><td>.93 1.0</td><td>14</td></tr><tr><td>Optimism</td><td>.82 .73</td><td>.82</td><td>.82 .91</td><td>11</td></tr><tr><td>Disgust</td><td>.78 .66</td><td>.86</td><td>.91 .97</td><td>1,478</td></tr><tr><td>Disapproval</td><td>.60 .35</td><td>.71</td><td>.80</td><td>.86 1,067</td></tr><tr><td>Fear</td><td>.58 .56</td><td>.66</td><td>.82</td><td>.82 113</td></tr><tr><td>Anger</td><td>.75 .70</td><td>.80</td><td>.91</td><td>.95 697</td></tr><tr><td>Sadness</td><td>.38</td><td>.91</td><td>.76</td><td>.95 55</td></tr><tr><td>Annoyance</td><td>.34</td><td>.55</td><td>.60</td><td>.77 47</td></tr><tr><td>Disappoint</td><td>46.5.4. .64</td><td>.92</td><td>.82</td><td>.97 39</td></tr></table>

Table 3: Classification accuracy of HateBERT, ToxDectroberta, Perspective API, and Llama Guard 3 1/8B on GPT-HATECHECK grouped by the detected emotions. We highlight the three emotions with the lowest accuracy for each model in red.

![](images/a32acd454d1fe970449c089d393939ce1b61684ae036fe2e72cdcd4a9b21243f.jpg)  
(a) Hateful examples.

![](images/9fba9b3c177013e40887f9154a2c72d4377cddcce768d2e70d85b20a249f6c6d.jpg)  
(b) Non-hateful examples.

Figure 4: Distribution of target identities for each detected emotion.
<table><tr><td>Hate Emo</td><td>HB</td><td>TD</td><td>PS</td><td>LI1 LI8</td><td>#</td></tr><tr><td>000111 -1</td><td>.32 .74</td><td>.48 .69</td><td>.83 .75</td><td>.85 .94 .81 .78</td><td>523 121</td></tr><tr><td></td><td>01</td><td>.91 .85</td><td>.95</td><td>.94 .98</td><td>811</td></tr><tr><td></td><td>-1 .77</td><td>.58</td><td>.79</td><td>.87 .92</td><td>2,976</td></tr><tr><td></td><td>.25</td><td>.25</td><td>.50</td><td>.50 .50</td><td>4</td></tr><tr><td></td><td>01</td><td>.00 .00</td><td>.00</td><td>.00 .00</td><td>3</td></tr></table>

Table 4: Classification accuracy of HateBERT, ToxDectroberta, Perspective API, and Llama Guard 3 1/8B on GPT-HATECHECK grouped by the hatefulness label (hate) and the polarity of the detected emotions (emo). We highlight the “positive” labels in green (“non-hateful” and positive emotions) and “negative” labels in red. The ambiguous emotions are highlighted in yellow.

## 3.3 Disentangling Stereotypes

Jin et al. (2024) motivated the use of LLMs with the generation of test cases that account for distinct stereotypes associated with different target identities (e.g., criminality for immigrants and sexuality for trans). However, they did not analyze which stereotypes are covered in their dataset and whether a distinction exists among target identities. We present an in-depth analysis of the stereotypes/counter-stereotypes in GPT-HATECHECK by 1) Interpreting stereotypes based on an established social psychology theory, 2) Analyzing the correlation between stereotypes and HS prediction accuracy, and 3) Extracting and qualitatively analyzing stereotypes/counter-stereotypes.

Stereotypes Interpretation Fiske et al. (2002; 2007) proposed the Stereotype Content Model, which uses the universal dimensions “warmth” and “competence”, to describe social perceptions and stereotypes. The model maps each stereotype onto interpretable semantic axes “warmth” vs. “coldness” and “competence” vs. “incompetence”. We use a state-of-the-art NLI model (He et al., 2021)<sup>12</sup> to assign “warmth” and “competence” scores to each example in the GPT-HATECHECK dataset. Inspired by Mathew et al. (2020), we derive the scores via semantic differentials of two opposite concepts (e.g., “warmth” and “coldness”). Specifically, we test four hypotheses for each example:

• H<sup>+</sup>: This message expresses warmth towards {target\_identity}.

• H−: This message expresses coldness towards {target\_identity}.

• H<sup>+</sup>: This message expresses that {target\_identity} are competent.

$\mathbb { H } _ { 2 } ^ { - }$ This message expresses that {target\_identity} are incompetent.

The NLI model returns logit scores for the three classes: “entail”, “contradict”, and “neutral”. We first take the softmax over the three classes and derive the score for “warmth” as:

$$
\begin{array} { r } { \mathbb { S } _ { w a r m t h } = \mathcal { P } _ { e n t a i l } ( \mathbb { H } _ { 1 } ^ { + } ) + \mathcal { P } _ { c o n t r a d i c t } ( \mathbb { H } _ { 1 } ^ { - } ) } \\ { - \mathcal { P } _ { c o n t r a d i c t } ( \mathbb { H } _ { 1 } ^ { + } ) - \mathcal { P } _ { e n t a i l } ( \mathbb { H } _ { 1 } ^ { - } ) } \end{array}\tag{1}
$$

We derive $\mathbb { S } _ { c o m p e t e n c e }$ similarly by replacing H∗ with H∗ in Equation 1. Due to the softmax operation, $\mathbb { S } _ { w a r m t h }$ and $\mathbb { S } _ { c o m p e t e n c e }$ are both bounded in the range of [-2, 2].

Figure 5 plots the kernel density estimate (KDE) in the warmth-competence semantic space.<sup>13</sup>

![](images/a43d0e1880435bf9d5a537cc3b845b9d35082744a4b0c4dbf673695b6eaf5e1f.jpg)  
(a) Non-Hateful examples.

![](images/131c1fe701c7a3e4ee23cbc3b14ee124728b9f3056a81d25331f5d7473c82747.jpg)  
(b) Hateful examples.  
Figure 5: Kernel density estimate (KDE) in the warmth-competence semantic space of various target identities.

While different target identity distributions overlap substantially, we can observe some patterns:

• Non-hateful: Many examples related to women have high “competence” scores, highlighting a typical counter-speech pattern. Meanwhile, examples related to gays tend to have a high “warmth” score.

• Hateful: Some examples related to women and disabled people receive very low “competence” scores but comparatively higher “warmth” scores, compared to other hateful examples (the lower right corner).

<table><tr><td rowspan="2">Target Identity</td><td colspan="2">Warmth</td></tr><tr><td>H N/H</td><td>Competence H N/H</td></tr><tr><td>Women</td><td>-1.06 0.12</td><td>-1.32 0.50</td></tr><tr><td>Trans ppl.</td><td>-1.15 0.11</td><td>-1.07 0.33</td></tr><tr><td>Gay ppl.</td><td>-1.28 0.18</td><td>-1.06 0.26</td></tr><tr><td>Black ppl.</td><td>-1.11 0.09</td><td>-1.19 0.35</td></tr><tr><td>Disabled ppl.</td><td>-1.12 0.02</td><td>-1.32 0.31</td></tr><tr><td>Muslims</td><td>-1.18 0.08</td><td>-1.11 0.22</td></tr><tr><td>Immigrants</td><td>-1.12 -0.09</td><td>-1.14 0.24</td></tr></table>

Table 5: The mean “warmth” and “competence” scores for hateful (H) and non-hateful (N/H) examples. We highlight the scores with the highest magnitude in bold.

The mean “warmth” and “competence” scores for each target identity are presented in Table 5. We can observe a clear push-back pattern: The higher the “coldness” or “incompetence” scores are for hateful stereotypes towards a target identity, the stronger the counter-stereotypes are in the opposite directions; consider, for illustration, the “warmth” dimension for gays and the “competence” dimension for women.

Correlation with Hate Prediction We also investigate whether there is a correlation between the location in the warmth-competence semantic space and the HS detectors’ accuracy. To this end, we apply the k-means algorithm to cluster the data points into 10 clusters. For each cluster, we compute its centroid’s Euclidean distance to the origin and Perspective API’s accuracy. We plot the correlation between these two factors in Figure 6. The figure reveals a clear trend that the farther from the origin (the more intense the example in the “warmth” or “competence” dimension), the more accurate the classifier is. The model seems to struggle particularly when both “warmth” and “competence” scores have a low magnitude. Other baselines demonstrate a similar trend (cf. Figure 11).

![](images/6727d51a820a301d7461a433c42c3eee2af644ef5356e4bc1d8ae160896b688a.jpg)  
Figure 6: The cluster’s distance to the origin has a positive correlation with the hatefulness prediction accuracy.

Stereotype Phrase Analysis Finally, we instructed GPT-4o to extract stereotypes/counterstereotypes from hateful and non-hateful posts, similarly to Section 3.2. However, instead of categorizing into a fixed taxonomy, we prompt the model to identify the minimum text span supporting the prediction. In some cases, the model should slightly modify the span to match the original meaning. For example, the expected counterstereotype for the input “No immigrant should be denied access to healthcare” is “should not be denied access to healthcare”. We present the full prompt in Appendix F and examples of extracted stereotypes/counter-stereotypes in Table 6.

<table><tr><td rowspan=1 colspan=1>Example</td><td rowspan=1 colspan=1>Stereotype</td></tr><tr><td rowspan=1 colspan=1>I can&#x27;t stand gays and theirdisgusting lifestyle!</td><td rowspan=1 colspan=1>disgustinglifestyle</td></tr><tr><td rowspan=1 colspan=1>No black person shouldface systemic racism.</td><td rowspan=1 colspan=1>should not facesystemic racism</td></tr><tr><td rowspan=1 colspan=1>It is wrong to discriminateagainst Muslims.</td><td rowspan=1 colspan=1>is wrong to dis-criminate</td></tr><tr><td rowspan=1 colspan=1>Muslims are advocates forsocial justice and work to-wards creating a fair and eq-uitable society for all.</td><td rowspan=1 colspan=1>advocates for so-cial justice</td></tr><tr><td rowspan=1 colspan=1>Immigrants can bring newperspectives and ideas totheir host communities.</td><td rowspan=1 colspan=1>bring new per-spectives   andideas</td></tr></table>

Table 6: Extracted stereotypes/counter-stereotypes from samples in GPT-HATECHECK.

Appendix E presents an expanded list of extracted stereotypes/counter-stereotypes associated with each target identity. Some stereotypes are shared across target identities, such as “drain on the economy” for disabled people and immigrants and “attention seeking” for trans and women. Meanwhile, some stereotypes are associated with a unique target identity, such as “terrorists” for Muslims and “crime and violence” for black people. On the other hand, many counterarguments are broader, such as calling for respect, acceptance, and treatment with dignity.

## 4 Conclusions and Future Work

We presented a comprehensive analysis of various factors that influence the behavior and accuracy of HS detectors. Empirical results revealed that popular industrial and academic HS classifiers are still prone to bias due to specific mentions of the target identity. They often confuse hatefulness and the polarity of the expressed emotions, and the stereotype intensity strongly impacts the classifiers’ accuracy. While the result may seem pessimistic, our work opens up new venues for the NLP community to improve the robustness of HS detectors further and mitigate various biases. In future work, we plan to apply our method to more datasets and models and introduce an open-source evaluation benchmark to facilitate the future development of HS detectors.

## Limitations

We conduct experiments on two functionality test datasets: HATECHECK (Röttger et al., 2021) and GPT-HATECHECK (Jin et al., 2024). These datasets provide rich metadata such as the target identity and the type of hate expressions (functionality). The messages in these datasets were composed by crowd-source workers or LLMs. We chose not to use HS detection datasets sampled from social media platforms because 1) they usually do not provide fine-grained target identity information and 2) they do not provide detailed information on data sampling (Fortuna et al., 2022). Sampling examples for different target identities from different domains (e.g., subreddits) or using different keywords might introduce compounding factors and obscure the conclusions. Nevertheless, we demonstrate the utility of our framework by presenting preliminary experimental results on a multi-source social media dataset in Appendix B.

The main contribution of our paper is the analysis of the impact of various factors in HS detection. The related problem of the analysis of bias mitigation methods was not in the focus of our work. While there exists an array of excellent surveys on bias mitigation methods (Meade et al., 2022; Kumar et al., 2023; Gallegos et al., 2024), including a comprehensive evaluation of bias mitigation methods would take up too much space and prevent us going into depth in the analysis. As we demonstrated in Section 3.1 and Appendix C, the naïve debiasing method we use only helps when models predict well-calibrated probability-like scores. We claim neither the effectiveness nor the novelty of this method.

Furthermore, we used LLMs to detect emotions and stereotype phrases and a pre-trained NLI model to score the two stereotype dimensions. This helped us develop a prototype and validate our hypotheses rapidly. Although we performed some prompt engineering and exploration, the accuracy was not perfect. If time and resources allow, hiring domain experts to relabel the examples would yield a more reliable result.

Lastly, stereotypes and emotions towards target identities strongly depend on the cultural context. The examples in GPT-HATECHECK are written by LLMs, which align best with views of Western, educated, white, and younger population (Santy et al., 2023). Studying how the findings might alter under distinct socio-demographic backgrounds would be an exciting extension of this work.

## Acknowledgement

This work has been partially funded by the European Commission under contract numbers HE101070278 and ISF-101080090. We are grateful for the insightful discussion with Paul Röttger and his suggestion for conducting minimal pair analysis for target identities. We thank the anonymous reviewers for the careful reading and constructive feedback so that we could improve the manuscript. Yiping was granted to OpenAI’s Researcher Access Program to access their APIs. Last but not least, We also received help from Perspective API team to increase our quota.

## References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Valerio Basile, Cristina Bosco, Elisabetta Fersini, Debora Nozza, Viviana Patti, Francisco Manuel Rangel Pardo, Paolo Rosso, and Manuela Sanguinetti. 2019. SemEval-2019 task 5: Multilingual detection of hate speech against immigrants and women in Twitter. In Proceedings of the 13th International Workshop on Semantic Evaluation, pages 54–63, Minneapolis, Minnesota, USA. Association for Computational Linguistics.

Tom Bourgeade, Patricia Chiril, Farah Benamara, and Véronique Moriceau. 2023. What did you learn to hate? a topic-oriented analysis of generalization in hate speech detection. In Proceedings of the 17th Conference of the European Chapter of the Associationfor Computational Linguistics, pages 3495– 3508, Dubrovnik, Croatia. Association for Computational Linguistics.

Tommaso Caselli, Valerio Basile, Jelena Mitrovic, and´ Michael Granitzer. 2021. HateBERT: Retraining BERT for abusive language detection in English. In

Proceedings of the 5th Workshop on Online Abuse and Harms (WOAH 2021), pages 17–25, Online. Association for Computational Linguistics.

Tommaso Caselli, Valerio Basile, Jelena Mitrovic, Inga´ Kartoziya, and Michael Granitzer. 2020. I feel offended, don’t be abusive! implicit/explicit messages in offensive and abusive language. In Proceedings ofthe Twelfth Language Resources and Evaluation Conference, pages 6193–6202, Marseille, France. European Language Resources Association.

Tong Chen, Danny Wang, Xurong Liang, Marten Risius, Gianluca Demartini, and Hongzhi Yin. 2024. Hate speech detection with generalizable target-aware fairness. arXiv preprint arXiv:2406.00046.

Christopher Clark, Mark Yatskar, and Luke Zettlemoyer. 2019. Don’t take the easy way out: Ensemble based methods for avoiding known dataset biases. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 4069–4082, Hong Kong, China. Association for Computational Linguistics.

Thomas Davidson, Debasmita Bhattacharya, and Ingmar Weber. 2019. Racial bias in hate speech and abusive language detection datasets. In Proceedings ofthe Third Workshop on Abusive Language Online, pages 25–35, Florence, Italy. Association for Computational Linguistics.

Thomas Davidson, Dana Warmsley, Michael Macy, and Ingmar Weber. 2017. Automated hate speech detection and the problem of offensive language. In Proceedings ofthe international AAAI conference on web and social media, volume 11, pages 512–515.

Dorottya Demszky, Dana Movshovitz-Attias, Jeongwoo Ko, Alan Cowen, Gaurav Nemade, and Sujith Ravi. 2020. GoEmotions: A dataset of fine-grained emotions. In Proceedings ofthe 58th Annual Meeting of the Association for Computational Linguistics, pages 4040–4054, Online. Association for Computational Linguistics.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Susan T Fiske, Amy JC Cuddy, and Peter Glick. 2007. Universal dimensions of social cognition: Warmth and competence. Trends in cognitive sciences, 11(2):77–83.

Susan T Fiske, JC Cuddy Amy, and Xu Jun Glick Peter. 2002. A model of (often mixed) stereotype content: Competence and warmth respectively follow from perceived status and competition. Journal of Personality and Social Psychology, 82(6):878–902.

Paula Fortuna, Monica Dominguez, Leo Wanner, and Zeerak Talat. 2022. Directions for nlp practices applied to online hate speech detection. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, page 11794–11805, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Paula Fortuna, Juan Soler, and Leo Wanner. 2020. Toxic, hateful, offensive or abusive? what are we really classifying? an empirical analysis of hate speech datasets. In Proceedings of the Twelfth Language Resources and Evaluation Conference, pages 6786–6794, Marseille, France. European Language Resources Association.

Paula Fortuna, Juan Soler-Company, and Leo Wanner. 2021. How well do hate speech, toxicity, abusive and offensive language classification models generalize across datasets? Information Processing & Management, 58(3):102524.

Antigoni Founta, Constantinos Djouvas, Despoina Chatzakou, Ilias Leontiadis, Jeremy Blackburn, Gianluca Stringhini, Athena Vakali, Michael Sirivianos, and Nicolas Kourtellis. 2018. Large scale crowdsourcing and characterization of twitter abusive behavior. In Proceedings of the international AAAI conference on web and social media, volume 12.

Kathleen C. Fraser, Isar Nejadgholi, and Svetlana Kiritchenko. 2021. Understanding and countering stereotypes: A computational approach to the stereotype content model. In Proceedings ofthe 59th Annual Meeting ofthe Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 600–616, Online. Association for Computational Linguistics.

Isabel O. Gallegos, Ryan A. Rossi, Joe Barrow, Md Mehrab Tanjim, Sungchul Kim, Franck Dernoncourt, Tong Yu, Ruiyi Zhang, and Nesreen K. Ahmed. 2024. Bias and fairness in large language models: A survey. Computational Linguistics, 50(3):1097– 1179.

Pengcheng He, Xiaodong Liu, Jianfeng Gao, and Weizhu Chen. 2021. Deberta: Decoding-enhanced bert with disentangled attention. In International Conference on Learning Representations.

Hakan Inan, Kartikeya Upasani, Jianfeng Chi, Rashi Rungta, Krithika Iyer, Yuning Mao, Michael Tontchev, Qing Hu, Brian Fuller, Davide Testuggine, et al. 2023. Llama guard: Llm-based input-output safeguard for human-ai conversations. arXiv preprint arXiv:2312.06674.

Yiping Jin, Leo Wanner, Vishakha Kadam, and Alexander Shvets. 2023. Towards weakly-supervised hate speech classification across datasets. In The 7th Workshop on Online Abuse and Harms (WOAH), pages 42–59, Toronto, Canada. Association for Computational Linguistics.

Yiping Jin, Leo Wanner, and Alexander Shvets. 2024. GPT-HateCheck: Can LLMs write better functional tests for hate speech detection? In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 7867–7885, Torino, Italia. ELRA and ICCL.

Sachin Kumar, Vidhisha Balachandran, Lucille Njoo, Antonios Anastasopoulos, and Yulia Tsvetkov. 2023. Language generation models can cause harm: So what can we do about it? an actionable survey. In Proceedings ofthe 17th Conference ofthe European Chapter of the Association for Computational Linguistics, pages 3299–3321, Dubrovnik, Croatia. Association for Computational Linguistics.

Ronan Le Bras, Swabha Swayamdipta, Chandra Bhagavatula, Rowan Zellers, Matthew Peters, Ashish Sabharwal, and Yejin Choi. 2020. Adversarial filters of dataset biases. In International conference on machine learning, pages 1078–1088. Pmlr.

Florian Ludwig, Klara Dolos, Torsten Zesch, and Eleanor Hobley. 2022. Improving generalization of hate speech detection systems to novel target groups via domain adaptation. In Proceedings of the Sixth Workshop on Online Abuse and Harms (WOAH), pages 29–39, Seattle, Washington (Hybrid). Association for Computational Linguistics.

Antonis Maronikolakis, Philip Baader, and Hinrich Schütze. 2022. Analyzing hate speech data along racial, gender and intersectional axes. In Proceedings ofthe 4th Workshop on Gender Bias in Natural Language Processing (GeBNLP), pages 1–7, Seattle, Washington. Association for Computational Linguistics.

Binny Mathew, Punyajoy Saha, Seid Muhie Yimam, Chris Biemann, Pawan Goyal, and Animesh Mukherjee. 2021. Hatexplain: A benchmark dataset for explainable hate speech detection. In Proceedings of the AAAI conference on artificial intelligence, volume 35, pages 14867–14875.

Binny Mathew, Sandipan Sikdar, Florian Lemmerich, and Markus Strohmaier. 2020. The polar framework: Polar opposites enable interpretability of pre-trained word embeddings. In Proceedings of The Web Conference 2020, pages 1548–1558.

Nicholas Meade, Elinor Poole-Dayan, and Siva Reddy. 2022. An empirical survey of the effectiveness of debiasing techniques for pre-trained language models. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1878–1898, Dublin, Ireland. Association for Computational Linguistics.

Marzieh Mozafari, Reza Farahbakhsh, and Noël Crespi. 2020. Hate speech detection and racial bias mitigation in social media based on bert model. PloS one, 15(8):e0237861.

Nedjma Ousidhoum, Zizheng Lin, Hongming Zhang, Yangqiu Song, and Dit-Yan Yeung. 2019. Multilingual and multi-aspect hate speech analysis. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 4675– 4684, Hong Kong, China. Association for Computational Linguistics.

Ji Ho Park, Jamin Shin, and Pascale Fung. 2018. Reducing gender bias in abusive language detection. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2799–2804, Brussels, Belgium. Association for Computational Linguistics.

Fabio Poletto, Valerio Basile, Manuela Sanguinetti, Cristina Bosco, and Viviana Patti. 2021. Resources and benchmark corpora for hate speech detection: a systematic review. Language Resources and Evaluation, 55(2):477–523.

Md Mustafizur Rahman, Dinesh Balakrishnan, Dhiraj Murthy, Mucahid Kutlu, and Matthew Lease. 2021. An information retrieval approach to building datasets for hate speech detection. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2).

Alan Ramponi and Sara Tonelli. 2022. Features or spurious artifacts? data-centric baselines for fair and robust hate speech detection. In Proceedings ofthe 2022 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies, pages 3027–3040, Seattle, United States. Association for Computational Linguistics.

Marco Tulio Ribeiro, Tongshuang Wu, Carlos Guestrin, and Sameer Singh. 2020. Beyond accuracy: Behavioral testing of NLP models with CheckList. In Proceedings ofthe 58th Annual Meeting ofthe Associationfor Computational Linguistics, pages 4902– 4912, Online. Association for Computational Linguistics.

Paul Röttger, Haitham Seelawi, Debora Nozza, Zeerak Talat, and Bertie Vidgen. 2022. Multilingual Hate-Check: Functional tests for multilingual hate speech detection models. In Proceedings ofthe Sixth Workshop on Online Abuse and Harms (WOAH), pages 154–169, Seattle, Washington (Hybrid). Association for Computational Linguistics.

Paul Röttger, Bertie Vidgen, Dong Nguyen, Zeerak Waseem, Helen Margetts, and Janet Pierrehumbert. 2021. HateCheck: Functional tests for hate speech detection models. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 41–58, Online. Association for Computational Linguistics.

Sebastin Santy, Jenny Liang, Ronan Le Bras, Katharina Reinecke, and Maarten Sap. 2023. NLPositionality: Characterizing design biases of datasets and models. In Proceedings of the 61st Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 9080–9102, Toronto, Canada. Association for Computational Linguistics.

Maarten Sap, Dallas Card, Saadia Gabriel, Yejin Choi, and Noah A. Smith. 2019. The risk of racial bias in hate speech detection. In Proceedings of the 57th Annual Meeting ofthe Associationfor Computational Linguistics, pages 1668–1678, Florence, Italy. Association for Computational Linguistics.

Maarten Sap, Saadia Gabriel, Lianhui Qin, Dan Jurafsky, Noah A. Smith, and Yejin Choi. 2020. Social bias frames: Reasoning about social and power implications of language. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 5477–5490, Online. Association for Computational Linguistics.

Swabha Swayamdipta, Roy Schwartz, Nicholas Lourie, Yizhong Wang, Hannaneh Hajishirzi, Noah A. Smith, and Yejin Choi. 2020. Dataset cartography: Mapping and diagnosing datasets with training dynamics. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 9275–9293, Online. Association for Computational Linguistics.

Bertie Vidgen and Leon Derczynski. 2020. Directions in abusive language training data, a systematic review: Garbage in, garbage out. Plos one, 15(12):e0243300.

Alex Warstadt, Alicia Parrish, Haokun Liu, Anhad Mohananey, Wei Peng, Sheng-Fu Wang, and Samuel R. Bowman. 2020. BLiMP: The benchmark of linguistic minimal pairs for English. Transactions of the Association for Computational Linguistics, 8:377– 392.

Zeerak Waseem and Dirk Hovy. 2016. Hateful symbols or hateful people? predictive features for hate speech detection on Twitter. In Proceedings ofthe NAACL Student Research Workshop, pages 88–93, San Diego, California. Association for Computational Linguistics.

Michael L. Waskom. 2021. seaborn: statistical data visualization. Journal of Open Source Software, 6(60):3021.

Michael Wiegand, Josef Ruppenhofer, and Thomas Kleinbauer. 2019. Detection of Abusive Language: the Problem of Biased Datasets. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 602–608, Minneapolis, Minnesota. Association for Computational Linguistics.

Wenjie Yin and Arkaitz Zubiaga. 2021. Towards generalisable hate speech detection: a review on obstacles and solutions. PeerJ Computer Science, 7:e598.

Marcos Zampieri, Shervin Malmasi, Preslav Nakov, Sara Rosenthal, Noura Farra, and Ritesh Kumar. 2019. SemEval-2019 task 6: Identifying and categorizing offensive language in social media (OffensEval). In Proceedings ofthe 13th International Workshop on Semantic Evaluation, pages 75–86, Minneapolis, Minnesota, USA. Association for Computational Linguistics.

Xuhui Zhou, Maarten Sap, Swabha Swayamdipta, Yejin Choi, and Noah Smith. 2021. Challenges in automated debiasing for toxic language detection. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pages 3143–3155, Online. Association for Computational Linguistics.

## A Examples from Datasets

Table 7 presents examples from HATECHECK and GPT-HATECHECK.

## B Experiments on SBIC Dataset

We conduct a preliminary experiment on the Social Bias Inference Corpus (SBIC) (Sap et al., 2020) to demonstrate the utility of our framework on realworld social media data. The dataset consists of examples from Twitter, Reddit, and various hate sites. We randomly sampled 200 examples from the following each of the six target identities in the dataset: “Asians”, “Black people”, “Gays”, “Jewish”, “Muslims”, “Women”.

Out of the 200 6=1,200 messages, 705 have detected emotions. Table 8 shows the most frequent emotions broken down by target identities. The emotions most associated with black people are “disgust” and “disapproval”, while “anger” stands out towards Muslims. This result is consistent with the GPT-HATECHECK dataset (cf. Figure 4).

Figure 7 depicts each target identity’s mean “warmth” and “competence” scores. Notably, messages targeting Jewish people have the lowest “warmth” and highest “competence” scores, revealing a strong antisemitism sentiment in the samples. Meanwhile, black people and women receive the lowest “competence” scores, consistent with the result on the GPT-HATECHECK dataset (cf. Table 5). Asians have the highest “warmth” score, showing that stereotypes against this group are less toxic compared to other target minorities.

## C Reliability Analysis of HS Classifiers

Debiasing contributes to different extents to different models, as shown in Table 2. We try to uncover the cause by analyzing the raw model predicted scores.

![](images/5811febccf9edcc2b48a8f6d69379ffae8a372fcb2028ba4cb8eac85b23c90b3.jpg)  
Figure 7: Each target identity’s mean “warmth” and “competence” scores for sample messages in SBIC dataset.

Figure 8a shows the distribution of predicted hateful probabilities from all models. The predicted scores of Perspective API and HateBERT are more evenly distributed, while ToxDect-roberta and Llama Guard 3 models predict almost exclusively near 0 or 1. Furthermore, Figure 8b shows that their predicted scores are much worse calibrated than the other two models. It explains why subtracting the bias (equivalent to adjusting the classification threshold) from the three models’ predictions would contribute much less.

## D Correlation Between Functionality and Emotion

Figure 9 presents the heat map of detected emotions across functionalities. Positive statements about protected identities (F19) predominantly express admiration. Direct thread (F5) is often expressed through anger, while implicit derogation (F4) often demonstrates disapproval.

## E Details for Stereotype Analysis

Figure 10 presents the scatter plot of “warmth” and “competence” scores assigned by the NLI model. The data points are distributed in a grid-like pattern because most “entail” and “contradict” scores are close to 0 or 1 after the softmax operation.

Table 9 shows sample messages with different “warmth” and “competence” scores assigned by the NLI model.

![](images/8a8f2334c576094e8f3b9c387010cd20325aca962472c7c0cf9321759527a9a8.jpg)

![](images/ca1c36dcdc49131662ca1676dd2a8ad3956854229ffc81d5c46a23fa8ee612ea.jpg)

![](images/93efe146db5cf685f93e92513b93a0be59a6dcfde298df1faf02c7b4deaa7b4a.jpg)

![](images/d5bb5731cf60f3164b66c2614b5b758e40d28396adafef920bd9cf3a2bb641fd.jpg)

![](images/aeb7905ebbbed0fd63dbcf7dd810a2b79a047ea9f37660389a20214c270fe313.jpg)  
(a) Distribution of models’ predicted scores, separated by the ground-truth label.

![](images/2398cc805fcdbd5fc1ba9341481debe2bcd457a14e506223106b67509e3a78c9.jpg)

![](images/a1659aeb47e10abecb7810272f4cc9164076b6a5e1d90160fc21d06dea358968.jpg)

![](images/06a908b51b1b79d55651516429ea8c15492cd38488c9e984cc0cf4ed07b6c718.jpg)

![](images/2dd256ea84c873c16ea0e5ec0409d4b283d6d3ea54c6639a22f73877bb10726b.jpg)

![](images/511ab498bac46e21220ae2235cff83cd18af76199432f0fce6bfe67389af8a51.jpg)  
(b) Models’ reliability diagrams plotting the true frequency of the positive label against its predicted probability for binned predictions (n = 20). The closer the dots are to the diagonal line, the more well-calibrated/reliable the predicted scores are.  
Figure 8: Analysis of model prediction scores.

![](images/7a1c6c1a959f79e62e0b5b1f0be49785d1cf2d7d27b1db8cfde60baaf0b7ecd9.jpg)  
Figure 9: Heat map of detected emotions for each functionality in GPT-HATECHECK dataset. Red color denotes hateful functionalities, and green color denotes non-hateful functionalities.

![](images/ce1676c5bc150657a5434c15ae223d0619dbecb31a45aeb278c217beadc2f657.jpg)  
Figure 10: Scatter plot in the warmth-competence semantic space of various target identities for examples in GPT-HATECHECK dataset.

We use the Seaborn library (Waskom, 2021)<sup>14</sup> to produce the kernel density estimate (KDE) plot in Figure 5. We set the parameters levels=4 (reduces the number of contour levels) and bw\_adjust=1.5 (increases smoothing) to unclutter the figures and make them more legible.

Figure 11 shows the correlation between the cluster centroid’s distance to the origin and the average prediction accuracy of HateBERT, ToxDectroberta, Llama Guard 3-1B and -8B. The result is consistent with the experiment on Perspective API (Figure 6).

Table 10 overviews the most frequent stereotype/counter-stereotype phrases extracted from GPT-HATECHECK dataset.

## F Prompts for GPT-4o

Table 11 shows the full prompt used for emotion detection (Section 3.2), and Table 12 shows the full prompt used for stereotype detection (Section 3.3).

## G Comparison of Perspective API Models

Perspective API can return six related attributes: “toxicity”, “severe toxicity”, “identity attack”, “insult”, “profanity”, and “threat”<sup>15</sup>. Röttger et al. (2022) remarked that the default “toxicity” model covers a much broader concept than hate speech, which also includes abuse and profanity. On the other hand, the “identity attack” model aligns with the definition of hate speech in HATECHECK (Röttger et al., 2021) and GPT-HATECHECK (Jin et al., 2024). The official definitions of the two attributes are as follows:

Toxicity: A rude, disrespectful, or unreasonable comment that is likely to make people leave a discussion.

Identity attack: Negative or hateful comments targeting someone because of their identity.

We validate Röttger et al. (2022)’s observation by comparing the accuracy of the two attribute models on GPT-HATECHECK (Table 13). Indeed, the “identity attack” model achieved better accuracy on all target identities, demonstrating that it aligns well with the definition of hate speech.

Additionally, we plot the predicted “toxicity” and “identity attack” scores in Figure 12. We can observe that the functionalities with which the “toxicity” and “identity attack” models disagree the most (the lower right corner) are contrastive nonhateful content such as “non-hateful use of profanity”, “abuse targeted at individuals”, and “reclaimed slurs”. These examples receive a high “toxicity” score and a low “identity attack” score.

![](images/29220a854568375250757fc0f501158deb88692e0c1a4bf77cdd3e10a44cdd75.jpg)  
(a) HateBERT

![](images/6e55a8636488d157f0e4eb4a9738bb27c21007243d0565fa34d2d9009e770eac.jpg)  
(b) ToxDect-roberta

![](images/2de2dd35620301c1ac1366e91cd4265ad310b9a39a15dd2e307cc387bd535f53.jpg)  
(c) Llama Guard 3 1B

![](images/4a7cbc4bf5dba919ffdb6315d29e7eb40436f27662070e21b32682f7367bb08f.jpg)  
(d) Llama Guard 3 8B  
Figure 11: Correlation between distance to the origin of the warmth-competence space and models’ accuracy.

![](images/fb2502e7fb7e9d7b1f3500a153cfc1208f3bc6f89a61cbc6df861bb34fe9e418.jpg)  
Figure 12: Predicted “toxicity” and “identity attack” for examples in GPT-HATECHECK. Different colors denote different functionalities. denotes cases where two attribute models’ predictions disagree using a 0.5 threshold. and vice versa.

<table><tr><td></td><td>Functionality</td><td>HATECHECK Example</td><td>GPT-HATECHECK Example</td><td>Gold Label</td></tr><tr><td rowspan="4">Dergaton</td><td>F1: Expression of strong negative emotions (explicit)</td><td>&quot;I hate [IDENTITY].&quot;</td><td>&quot;Trans people are sick and need help, not acceptance.&quot; &quot;Trans people are nothing more than con-</td><td>hateful</td></tr><tr><td>F2: Description using very negative attributes (explicit)</td><td>&quot;[IDENTITY] are disgusting.&quot;</td><td>fused individuals who need to accept their biological reality.&quot;</td><td>hateful</td></tr><tr><td>F3: Dehumanisation (explicit)</td><td>&quot;[IDENTITY] are scum.&quot;</td><td>&quot;Trans people are nothing more than delu- sional attention seekers.&#x27;</td><td>hateful</td></tr><tr><td>F4: Implicit derogation</td><td>&quot;[IDENTITY] belong in a zoo.&quot;</td><td>&quot;Transgender people are just confused individuals who are trying to seek atten- tion.&quot;</td><td>hateful</td></tr><tr><td rowspan="2">Thnng</td><td>F5: Direct threat</td><td>&quot;I am going to shoot the next [IDEN- TITY] I see.&quot;</td><td>&quot;Trans people are a disgrace and should be eradicated from society. We will make sure you suffer for your existence.&quot;</td><td>hateful</td></tr><tr><td>F6: Threat as normative statement</td><td>&quot;We should get rid of [IDENTITY] once and for all.&quot;</td><td>&quot;Trans people should be eradicated from society. They are a threat to the natural order.&quot;</td><td>hateful</td></tr><tr><td></td><td>F7: Hate expressed using slur</td><td>&quot;You are just a [SLUR] to me.&quot;</td><td>&quot;You&#x27;re just a freak, a tranny.&quot;</td><td>hateful</td></tr><tr><td>SIla uage</td><td>F8: Non-hateful homonyms of slurs</td><td>&quot;I had this queer feeling we were being watched.&quot;</td><td>&quot;I hope you have a transparent and fulfill- ing day!&quot;</td><td>non-hate</td></tr><tr><td></td><td>F9: Reclaimed slurs</td><td>&quot;I&#x27;m the freshest n*gga in the room.&quot;</td><td>&quot;I love being a proud trans individual!&quot;</td><td>non-hate</td></tr><tr><td>Proannity</td><td>F10: Hate expressed using profanity</td><td>&quot;These [IDENTITY] should shut the f*ck up once and for all.&quot;</td><td>&quot;Transgender people are a f*cking dis- grace to humanity.&#x27;</td><td>hateful</td></tr><tr><td>uage</td><td>F11: Non-hateful use of profanity</td><td>&quot;I wish someone could get me the f*ck out of here.&quot;</td><td>&quot;I f*cking hate Mondays, man.&quot;</td><td>non-hate</td></tr><tr><td>rooun e.</td><td>F12: Hate expressed through refer- ence in subsequent clauses</td><td>&quot;I have met many [IDENTITY] and I hate every single one of them.&quot;</td><td>&quot;Every time I see a trans person, I can&#x27;t help but feel disgusted by their exis- tence.&quot;</td><td>hateful</td></tr><tr><td></td><td>F13: Hate expressed through refer- ence in subsequent sentences</td><td>&quot;Of course they are stupid. They are just [IDENTITY] after all.&quot;</td><td>&quot;You think being trans makes you spe- cial? It just makes you even more pa- thetic.&quot;</td><td>hateful</td></tr><tr><td>Neton</td><td>F14: Hate expressed using negated positive statement</td><td>&quot;[IDENTITY] lives do not matter.&quot;</td><td>&quot;trans will never be truly happy.&quot;</td><td>hateful</td></tr><tr><td></td><td>F15: Non-hate expressed using negated hateful statement</td><td>&quot;No [IDENTITY] deserves to die.&quot;</td><td>&quot;Trans rights are not a threat to society.&quot;</td><td>non-hate</td></tr><tr><td>Phiasing</td><td>F16: Hate phrased as a question</td><td>&quot;Why is it that all [IDENTITY] are repulsive?&quot;</td><td>&quot;Why do trans people think they deserve special rights?&quot;</td><td>hateful</td></tr><tr><td></td><td>F17: Hate phrased as an opinion</td><td>&quot;I think that [IDENTITY] should all be killed.&quot;</td><td>&quot;Transgender people are mentally ill and need to seek help.&quot;</td><td>hateful</td></tr><tr><td>No-Hhae</td><td>F18: Neutral statements using pro- tected group identifiers</td><td>&quot;We are a group of [IDENTITY].&quot;</td><td>&quot;Transgender individuals experience gen- der dysphoria, which is a distressing dis- connect between their gender identity and</td><td>non-hate</td></tr><tr><td>r</td><td>F19: Positive statements using pro- tected group identifiers</td><td>&quot;I love [IDENTITY].&quot;</td><td>assigned sex at birth.&quot; &quot;Trans individuals are courageous and in-</td><td>non-hate</td></tr><tr><td></td><td>F20: Denouncements of hate that quote it</td><td>&quot;Statements like [IDENTITY] are</td><td>spiring.&quot; &quot;Statements like transgender people are freaks&#x27; are completely unacceptable.&quot;</td><td>non-hate</td></tr><tr><td>Couter spech</td><td>F21: Denouncements of hate that make direct reference to it</td><td>scum&#x27; are deeply hurtful.&quot; &quot;You have to stop calling [IDEN- TITY] disgusting.</td><td>&quot;Your ignorant comment about trans peo- ple only shows your lack of understand-</td><td>non-hate</td></tr><tr><td></td><td></td><td></td><td>ing and empathy.&#x27;</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>targets</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>&quot;You make me sick.&quot;</td><td>&quot;Hey you f*cking loser, go kill yourself!&quot;</td><td>non-hate</td></tr><tr><td>Nornn-ppot.</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>F23: Abuse targeted at individuals</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>F22: Abuse targeted at objects</td><td>&quot;I really can&#x27;t stand cauliflower.&quot;</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>&quot;You stupid pencil, you can&#x27;t even write</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>non-hate</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>straight!&quot;</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>F24:</td><td>(not as member of a prot. group)</td><td></td><td></td><td></td></table>

Table 7: Examples from HATECHECK and GPT-HATECHECK. The table is reproduced from Jin et al. (2024).

<table><tr><td>Emotion</td><td>Asians</td><td>Black</td><td>Gays</td><td>Jewish</td><td>Muslims</td><td>Women</td><td>Total</td></tr><tr><td>Disgust</td><td>40</td><td>88</td><td>31</td><td>68</td><td>39</td><td>19</td><td>285</td></tr><tr><td>Disapproval</td><td>49</td><td>44</td><td>34</td><td>22</td><td>34</td><td>33</td><td>216</td></tr><tr><td>Anger</td><td>8</td><td>19</td><td>14</td><td>10</td><td>25</td><td>6</td><td>82</td></tr><tr><td>Approval</td><td>14</td><td>21</td><td>17</td><td>8</td><td>11</td><td>10</td><td>81</td></tr><tr><td>Annoyance</td><td>2</td><td>2</td><td>1</td><td>0</td><td>2</td><td>4</td><td>11</td></tr><tr><td>Disappointment</td><td>5</td><td>3</td><td>0</td><td>0</td><td>0</td><td>0</td><td>8</td></tr></table>

Table 8: Most frequent detected emotions from samples in the SBIC dataset. For each emotion, we highlight the target identity with the most examples in bold.

<table><tr><td colspan="1" rowspan="1">Example</td><td colspan="1" rowspan="1">Warm</td><td colspan="1" rowspan="1">Comp.</td></tr><tr><td colspan="1" rowspan="1">F*ck immigrants and their stupid cultures. This is America, not a melting pot.</td><td colspan="1" rowspan="1">-2.00</td><td colspan="1" rowspan="1">-1.99</td></tr><tr><td colspan="1" rowspan="1">It's time to take action against immigrants and ensure they never set foot in ourcountry again.</td><td colspan="1" rowspan="1">-1.93</td><td colspan="1" rowspan="1">-1.00</td></tr><tr><td colspan="1" rowspan="1">It's a waste of time to educate women, they'll never be as smart as men.</td><td colspan="1" rowspan="1">-1.00</td><td colspan="1" rowspan="1">-2.00</td></tr><tr><td colspan="1" rowspan="1">Are black people naturally more violent and aggressive?</td><td colspan="1" rowspan="1">-1.00</td><td colspan="1" rowspan="1">-0.99</td></tr><tr><td colspan="1" rowspan="1">Black people have a natural inclination towards violence and aggression.</td><td colspan="1" rowspan="1">-1.00</td><td colspan="1" rowspan="1">-0.46</td></tr><tr><td colspan="1" rowspan="1">Women are so emotional and irrational, it's no wonder they can't make gooddecisions.</td><td colspan="1" rowspan="1">0.00</td><td colspan="1" rowspan="1">-1.00</td></tr><tr><td colspan="1" rowspan="1">Being a retard is just one aspect of my incredible and diverse personality.</td><td colspan="1" rowspan="1">1.00</td><td colspan="1" rowspan="1">1.00</td></tr><tr><td colspan="1" rowspan="1">No woman should be denied education or employment.</td><td colspan="1" rowspan="1">1.00</td><td colspan="1" rowspan="1">1.00</td></tr><tr><td colspan="1" rowspan="1">Muslims contribute greatly to society through their diverse talents and skills.</td><td colspan="1" rowspan="1">1.00</td><td colspan="1" rowspan="1">1.99</td></tr><tr><td colspan="1" rowspan="1">Being a trans woman is a source of strength and empowerment for me.</td><td colspan="1" rowspan="1">1.85</td><td colspan="1" rowspan="1">1.00</td></tr><tr><td colspan="1" rowspan="1">Cheers to the muzzie community for their resilience and strength in the face ofadversity.</td><td colspan="1" rowspan="1">2.00</td><td colspan="1" rowspan="1">1.01</td></tr><tr><td colspan="1" rowspan="1">Sending love and empowerment to all the incredible bitches in my life.</td><td colspan="1" rowspan="1">2.00</td><td colspan="1" rowspan="1">2.00</td></tr><tr><td>Target Group</td><td>Hate</td><td>Non-Hate</td></tr><tr><td>Women</td><td>just objects for men's pleasure (8), nothing more than objects (7), terrible drivers (6), attention-seeking drama queens (5), needy and clingy (5), so weak and fragile (4), too emo- tional and irrational (4), so manipulative (4), only good for their looks (3)</td><td>Sending love (3), made significant contributions (3), beautiful inside and out (2), keep shining (2), re- silient (2), are not inferior to men (2), have the ability to excel (2), are not meant to serve men (2)</td></tr><tr><td>Trans ppl.</td><td>just confused (11), are an abomination (8), un- natural (6), mentally unstable (6), mentally ill (6), attention seekers (5), a burden on society (5), attention-seeking snowflakes (5), confused and mentally unstable (4), pretending to be something they're not (4)</td><td>should be celebrated (4), an impor- tant part (3), proud to be a part (3), love and acceptance (2), deserve love and respect (2), should not be denied their basic human rights (2), should not be forced to conform (2)</td></tr><tr><td>Gay ppl.</td><td>unnatural (10), are an abomination (9), should be eradicated (8), a disgrace to humanity (6) a disgrace (6), a blight on humanity (6), would just disappear (5), should be shunned (5), an abomination (5), a danger to children (5), is a sin (5), is a perversion (5)</td><td>Love is love (7), promote under- standing and acceptance (4), de- serves to love and be loved (3), Send- ing love and support (3), promote love and acceptance (2), celebrate and support (2)</td></tr><tr><td>Black ppl.</td><td>always play the victim card (9), are inferior (5), always causing trouble (5), a burden on society (4), nothing but a burden (4), exterminate all (4), a blight on society (3), a cancer that needs to be cut out (3), loud and obnoxious (3), a stain on society (3), a better place without (3)</td><td>made significant contributions (5), Keep shining (3), made advance- ments (3), strong sense of com- munity (3), are not untrustworthy (3), should not be discriminated (2), should not face systemic racism (2)</td></tr><tr><td>Disabled ppl.</td><td>a burden (26), just a burden (14), a burden on society (11), a waste of space (8), should be eliminated (8), a drain on resources (8), a drain on the economy (6), should be locked away (6), nothing more than a burden (5), such a burden (5), should be euthanized (5), would just disappear (5)</td><td>an inspiration (4), deserve to be treated with dignity and respect (3), have the right to access (3), knows no bounds (2), inspire others (2), em- bracing your true potential (2), not objects of pity (2)</td></tr><tr><td>Muslims</td><td>a threat (10), a stain on humanity (7), a cancer that needs to be eradicated (6), a drain on our economy (6), are all terrorists (6), a plague that needs to be eradicated (5), are a threat (5), a cancer that needs to be cut out (5), are terrorists in disguise (4), brainwashed by their religion (4)</td><td>made significant contributions (5), an integral part (2), strong sense of family values (2), keep shining your light (2), have the right to practice their religion freely (2), are not ter- rorists (2)</td></tr><tr><td>Immigrants</td><td>a drain on our economy (10), go back to where you came from (9), a threat to our national security (8), a burden (7), go back to your own country (7), nothing but a burden (6), not welcome (6), a burden on our society (5), a threat (5), taking advantage (5), just here to take advantage (5) deport all (5)</td><td>should not be denied the right (11), should not be denied access (4), bring new ideas and perspectives (6), appreciate the diversity (2), bring unique skills and talents (2), are not a threat (2)</td></tr></table>

Table 9: Samples in GPT-HATECHECK with different “warmth” and “competence” scores.

Table 10: Most frequent extracted stereotypes/counter-stereotypes for each target identity. We display the frequency in the bracket.

<table><tr><td>Type</td><td>Prompt</td></tr><tr><td>System</td><td>You are an expert human rater trained to identify emotions in text. The list of emotions of interest are:</td></tr><tr><td></td><td>• admiration</td></tr><tr><td></td><td>• amusement</td></tr><tr><td></td><td>• approval</td></tr><tr><td></td><td>• caring</td></tr><tr><td></td><td>• desire</td></tr><tr><td></td><td>• excitement</td></tr><tr><td></td><td>• gratitude</td></tr><tr><td></td><td>• joy</td></tr><tr><td></td><td>• love</td></tr><tr><td></td><td>• optimism</td></tr><tr><td>• pride</td><td></td></tr><tr><td>• relief</td><td></td></tr><tr><td>• anger</td><td></td></tr><tr><td>• annoyance</td><td></td></tr><tr><td>• disappointment</td><td></td></tr><tr><td>• disapproval</td><td></td></tr><tr><td>• disgust</td><td></td></tr><tr><td>• embarrassment</td><td></td></tr><tr><td>• fear</td><td></td></tr><tr><td>• grief</td><td></td></tr><tr><td>• nerveousness</td><td></td></tr><tr><td>• remorse</td><td></td></tr><tr><td>• sadness</td><td></td></tr><tr><td></td><td></td></tr><tr><td>• confusion</td><td></td></tr><tr><td>• curiosity</td><td></td></tr><tr><td>• realization</td><td></td></tr><tr><td>• surprise</td><td></td></tr><tr><td>User message: Message:</td><td>Label the main emotion towards {target_identity} expressed in the following “{message},</td></tr><tr><td>System</td><td>You are an expert human rater trained to label hateful messages after analyzing the entire message. Also, you are an expert in the task of identifying and extracting terms in the messages relevant to hate or love.</td></tr><tr><td>User</td><td>Identify the bare minimum length text span (a phrase that has a minimum number of words) in the message that expresses hate or love towards the{target_identity} Remove insignificant words from the text span other than those that are essentially required to express hate or love. Only if the text span does not reflect the sentiment of the entire message, modify the text span accordingly by adding negate words like “not" or “no" to that minimum text span so that it reflects the sentiment of the entire message and then return that modified text span without mentioning {target_identity}. . Answer “None" if you can't find any. For example, Message: “No immigrants should be denied access to healthcare". Here, the sentiment is love towards immigrants. Therefore, the original text span “should be denied access" should be negated so that it reflects the message sentiment. i.e., the modified text span has to be “should not be denied access". If the original</td></tr><tr><td></td><td>text span already aligns with the sentiment expressed in the entire message, no modification is necessary. Now try to find the text span for me that reflects the</td></tr><tr><td>Message: {message},</td><td>message. Just return the final answer.</td></tr></table>

Table 11: Prompts for emotion detection.

Table 12: Prompts for stereotype detection.

<table><tr><td>Model</td><td>Women</td><td>Trans</td><td>Gays</td><td>Black</td><td>Disabled</td><td>Muslims</td><td>Immigr.</td></tr><tr><td>Toxicity</td><td>.646</td><td>.772</td><td>.864</td><td>.748</td><td>.697</td><td>.892</td><td>.559</td></tr><tr><td>Identity Attack</td><td>.731</td><td>.897</td><td>.895</td><td>.879</td><td>.707</td><td>.936</td><td>.689</td></tr></table>

Table 13: Per target identity accuracy scores of Perspective API’s different attribute models on GPT-HATECHECK.