# An Interpretable and Crosslingual Method for Evaluating Second-Language Dialogues

Rena Gao♡, Jingxuan Wu♣, Xuetong Wu♡,

Carsten Roever♡, Jing Wu♠, Long Lv♢, Jey Han Lau♡

♡The University of Melbourne, Parkville, 3052, Australia

♣ The Chinese University of Hong Kong, Shatin, NT, Hong Kong, China

♢Central South University, Changsha, 410017, China

♠Tiangong University, Tianjin, 300387, China

{rena.gao,carsten}@unimelb.edu.au, {wfyitf,jeyhan.lau}@gmail.com wujingxuan@cuhk.edu.cn, wujing@tiangong.edu.cn, lvlong@csu.edu.en

## Abstract

We analyse the cross-lingual transferability of a dialogue evaluation framework that assesses the relationships between micro-level linguistic features (e.g. backchannels) and macro-level interactivity labels (e.g. topic management), originally designed for English-as-a-secondlanguage dialogues. To this end, we develop CNIMA (Chinese Non-Native Interactivity Measurement and Automation), a Chinese-asa-second-language labelled dataset with 10K dialogues. We found the evaluation framework to be robust across distinct languages: English and Chinese, revealing language-specific and language-universal relationships between micro-level and macro-level features. Next, we propose an automated, interpretable approach with low data requirement that scores the overall quality of a second-language dialogue based on the framework. Our approach is interpretable in that it reveals the key linguistic and interactivity features that contributed to the overall quality score. As our approach does not require labelled data, it can also be adapted to other languages for second-language dialogue evaluation.

## 1 Introduction

In the context of second language (SL) assessment, speaking has always been considered an essential ability (Smith et al., 2022; Allwood, 2008; Huang et al., 2020), but prior studies have predominantly focused on written correction (Paiva et al., 2022) or pronunciation from ASR (McGhee et al., 2024). Given the lack of researches that capture the unique linguistic features of SL speakers in dialogues (especially in open-domain interactive conversations), this leaves conversational interaction assessment under-explored and ultimately contributes to a limited understanding of conversational fluency and interactivity for SL speakers from different languages.

An exception is Gao et al. (2025), who propose a two-level framework for assessing the interactivity ability of English-as-a-second-language (ESL) speakers in open-domain conversations. The framework introduces micro-level word/utterance features (e.g. backchannels) and macro-level interactivity labels (e.g. topic management) and they find that the micro-level features are highly predictive of the macro-level labels. However, they conduct the study only in the context of ESL, raising questions about the transferability of the evaluation framework to other languages besides English. Furthermore, they do not introduce an automated approach for this evaluation, as the analysis relies on human annotations.

Our work aims to address these shortcomings by: (1) testing the cross-lingual transferability of the evaluation framework by validating it on an annotated Chinese-as-a-second-language (CSL) dialogue dataset; and (2) introducing an automated, interpretable approach with low data requirement that scores the overall quality of a second-language dialogue. Our automated approach is interpretable as it highlights the key features that lead to the overall score, and it has a low data requirement because it does not require labelled data. As it can be adapted to other languages easily, our method paves the way for a new tool for assessing SL conversations.<sup>1</sup> To summarise our contributions:

• We release CNIMA (Chinese Non-Native Interactivity Measurement and Automation), an annotated CSL dialogue dataset with 10K dialogues, based on the evaluation framework of Gao et al. (2025) originally designed for ESL dialogues.

• We evaluate the cross-lingual transferability of the evaluation framework and find it to be robust across different languages. We further reveal language-specific and language-universal relationships between micro-level and macrolevel features, highlighting the subtle differences between CSL and ESL dialogues.

• We introduce an automated, interpretable approach that predicts the micro- and macrolevel features and the overall quality scores of second-language dialogues. Our best method demonstrates strong performance, creating a new tool for second language assessment. Importantly, the method can be adapted to other languages as it does not require labelled data.

## 2 Related work

## 2.1 Assessment of SL Spoken Conversation

Mainstream second language assessments in current industrial practice have mainly focused on grammatical accuracy, pronunciation standardization and vocabulary richness; for example, in TOEFL iBT, PTE Academic and Cambridge IELTS test (Xu, 2018; Paiva et al., 2022; Xu et al., 2021). However, few speaking assessments emphasised the importance of interaction in dialogues for second language speakers and learners (Khabbazbashi et al., 2021) and aspects such as how speakers manage the topics in communication (Shaxobiddin, 2024), how speakers perform social roles from speaking interactions (Chen et al., 2023), and how speakers start and end a talk in an acceptable manner (Yap and Sahoo, 2024). There are, however, some exceptions. For example, Dai (2022) develop a test rubric for Chinese Second Language speakers. More recently, Gao and Wang (2024) introduce an interactivity scoring framework inspired by the IELTS speaking assessment. However, these studies only provide a theoretical framework for evaluation and lack an automated pipeline for large-scale assessments, human rating is still a major scoring approach in mainstream industrial practice.

## 2.2 Automated Scoring System in SL

Automated scoring systems can improve the efficiency of processing and scoring dialogues compared to manual assessment methods, allowing for real-time feedback (Evanini et al., 2017) and explainable suggestions. Some major second language assessment organisations have employed automated speaking scoring, including PTE Pearson (Jones and Liu, 2023), Duolingo English Test (Burstein et al., 2021), and TOEFL (Gong, 2023), but these automated systems failed to capture the interactive nature of SL conversations. That is, they struggle to offer detailed analyses and insights on the common errors and language usage patterns and where SL speakers struggle the most. This motivates a more explainable framework for SL assessment that can provide better feedback and suggestions from the automated scoring.

The main challenge is to design an effective automated evaluation tool. Capturing the nuances of spoken language, such as interaction, attitudes, and cooperation in conversation, is challenging to achieve (Gao and Wang, 2024). Automated scoring systems must handle diverse patterns across varied languages, which can vary widely among different non-native SL speakers across distinct languages, like Asian languages and European languages with distinct differences. Assessing interactive features like turn-taking, interruptions, and response appropriateness further complicates the process. Ultimately, the dynamic nature of conversations makes it challenging for automated systems to accurately evaluate interactive speaking, necessitating sophisticated algorithms and continuous refinement (Engwall et al., 2022; Cumbal, 2024).

## 3 Evaluation Framework

We extend the evaluation framework of Gao et al. (2025) for assessing second-language dialogues, which considers micro-level features and macrolevel interactivity labels. One addition we have made is to introduce an overall dialogue quality score. Figure 1 shows an example of CSL dialogue annotation with the evaluation framework. The framework has 17 micro-level features, and it can be further broken into token-level (such as ‘Reference Word’: she and ‘Backchannel’: hmm) and utterance-level features (such as ‘Formulaic response’: How’s going). The micro-level features are annotated as spans (i.e. annotators mark text spans corresponding to a feature). For macro-level interactivity labels, there are four: Topic Management, Tone Choice Appropriateness, Conversation Opening and Conversation Closing, and they are annotated as (dialogue) labels. Briefly, Topic Management refers to the speaker’s ability to control and navigate the flow of topics; Tone Choice the suitability of the tone; and Conversation Opening/- Closing the naturalness in the initial exchange and conclusion of the discussion. A more elaborate definition can be found in Appendix Table 13. For each macro-level interactivity label, the score ranges from 1 to 5 (categorical), and higher scores indicate a more natural and active interactivity quality. Note that for tone choice appropriateness, higher scores denote a casual tone, and lower scores indicate a formal tone.<sup>2</sup> In addition to micro- and macrolevel features, we introduce an overall score for each dialogue to measure second language speakers’ interaction abilities in open-domain conversation. Like the macro-level labels, it is scored from 1 to 5 (categorical). The overall score is designed to capture the holistic quality of a conversation, integrating elements like contextualisation, responsiveness, and communicative purposes across the whole conversation, and a high score reflects the speaker’s ability to engage in fluid and meaningful interactions. The full description of each score is detailed in Appendix Table 12.

![](images/c853f588faa448d926c61b73642b7e4be8252bfbfbb6a82ec57c4165d00a1936.jpg)  
Figure 1: An example of a CSL dialogue annotated with the micro-level features, macro-level interactivity labels and overall dialogue quality score.

## 4 CNIMA Development

## 4.1 CSL Dialogue Collection

We first extend the CSL dialogue dataset developed by Wu and Roever (2021), which has approximately 8,000 dialogues. We followed a similar process, where we recruited 20 learners of Chinese Mandarin at four different proficiency levels.<sup>3</sup> Each participant was assigned to a dyadic group with a partner of a similar proficiency level. Each pair of participants did one elicited conversation task, in which they discussed an assigned topic, and two role-play activities (three tasks in total; dialogue collection instructions can be found in Appendix A.4).<sup>4</sup> After collecting the conversations, we recruited in-house workers to segment the conversations based on the discussion topics.<sup>5</sup> Including the original 8,000 dialogues from Wu and Roever (2021), our extended CSL dataset has 10,908 dialogues in total as shown in Table 1.

<table><tr><td>Statistic</td><td>Count</td></tr><tr><td>#dialogues</td><td>10,908</td></tr><tr><td>#turns (average)</td><td>6</td></tr><tr><td>#turns (max)</td><td>13</td></tr><tr><td>#tokens /w token-level features</td><td>170,852</td></tr><tr><td>#tokens /w utterance-level features</td><td>94,516</td></tr></table>

Table 1: CNIMA Statistics.
<table><tr><td>Measure</td><td>Micro-level Features</td><td>Macro-level Labels</td><td>Overall Score</td></tr><tr><td>α</td><td>0.66</td><td>0.67</td><td>0.61</td></tr><tr><td>r</td><td>0.65</td><td>0.68</td><td>0.62</td></tr></table>

Table 2: Inter-annotator agreement for micro-level, macro-level features and overall scores.

## 4.2 Data Annotation

Given the dialogues, we annotate them for microlevel features, macro-level interactivity labels and overall quality scores based on the evaluation framework introduced in Section 3. To this end, we recruited twelve postgraduate students who are native Chinese speakers. We (first author) first trained the annotators and the annotator training manual can be found in Appendix A.3. During the training, annotators will see one dialogue as an example to understand the requirements and learn how to use the annotation platform (Appendix A.1). After training, each annotator was assigned 950 dialogues, and each dialogue was annotated by two annotators.<sup>6</sup> The annotation was conducted over two months, where in the first few weeks we (first author) checked for initial agreement, discussed feedback, and fixed any annotation errors (e.g. missing values in the annotation results) before proceeding with the full annotation. The annotation process was guided by an annotation guide (Appendix A.3), which provided definitions and examples for each micro-level feature, macro-level interactivity label and the overall dialogue quality score. For the overall score, in addition to providing the score, the annotators are also asked to write down their reasons for justifying their label.

As explained in Section 3, the micro-level features are annotated as spans, and the macro-level labels and overall quality scores as document (dialogue) labels. To aggregate the span annotations by the two annotators for the micro-level features, for each dialogue and each micro-level feature, we iterate through each turn and select the shorter (longer) span between the two annotators if it is a tokenlevel (utterance-level) feature.<sup>7</sup> For the macro-level interactivity labels, in dialogues where we have disagreement between the two annotators, we use the majority label from their larger unsegmented conversation as the ground truth label.<sup>8</sup> For the overall dialogue quality scores, for each dialogue with disagreement, we (two authors of this paper) manually assess the justification provided by the annotators to determine the ground truth.<sup>9</sup>

To measure the annotation quality, for the microlevel features, we calculate agreement between the annotators at the token level for each micro-level feature, i.e., we compute agreement statistics based on the presence or absence of the feature as marked by the annotators for each word token.<sup>10</sup> We calculate Pearson correlation coefficient r (Cohen et al., 2009) and Krippendorff’s α (Krippendorff, 2018)

![](images/c87293d5473f858af9e67bd6da2f46c49272031deaa56db7625902edea60158d.jpg)  
Figure 2: Pipeline for automated scoring of the CSL dialogue on three steps

and summarise the results in Table 2. The agreement is above 0.6 over all the features/labels, indicating a good consensus between annotators.

## 5 Automated Pipeline

We now propose automating the prediction of the micro-level features (step 1), macro-level interactivity labels (step 2), and overall score (step 3); see Figure 2 for an illustration. We experiment with classical machine learning models (Logistic Regression (LR), Random Forest (RF), and Naïve Bayes (NB)), fine-tuned Chinese BERT (Cui et al., 2021), and GPT-4o (OpenAI, 2024) with prompts to automate these steps. Note that these predictions are done in sequence, where the output from the previous step is used as input for the next step (e.g. when predicting the macro-level labels, we use the predicted micro-level features as input). We partition CNIMA into train, development and test sets (ratio = 7:1:2) for these experiments.

Step 1 This step predicts the spans of the 17 micro-level features given an input dialogue, and we experiment with BERT and GPT-4o here. For BERT, we fine-tune 17 span classifiers (one for each feature) to classify the presence or absence of a feature for each word in the dialogue. For GPT-4o, we do one-shot prompting (i.e. 1 dialogue with the expected output as a demonstration) with instructions to generate spans for the token-level and utterance-level features separately.<sup>11</sup>

<table><tr><td>Models</td><td>Topic</td><td>Tone</td><td>Opening</td><td>Closing</td></tr><tr><td>LR</td><td>0.829</td><td>0.859</td><td>0.821</td><td>0.810</td></tr><tr><td>RF</td><td>0.831</td><td>0.816</td><td>0.858</td><td>0.852</td></tr><tr><td>NB</td><td>0.846</td><td>0.783</td><td>0.811</td><td>0.846</td></tr></table>

Table 3: F1 performance of predicting the macro-level features using human-annotated micro-level features.

Step 2 This step predicts the macro-level interactivity labels for topic management, tone choice appropriateness, and conversation opening and closing. Here, we experiment with classical models (LR, RF and NB), BERT and GPT-4o. For each of them, we train four (one for each interactivity label) 5-class classifiers (each label has 5 classes). For the classical models, we follow Gao et al. (2025) and convert the micro-level feature spans into normalised counts and use them as input features to train LR/RF/NB to predict the interactivity labels. For BERT and GPT-4o, the input is the dialogue concatenated with the micro-level feature spans. BERT is fine-tuned as a document classifier, and GPT-4o is one-shot prompted with instructions.

Step 3 This step predicts the overall dialogue quality score. As this is a document classification task like step 2, we follow the same process for building LR, RF, NB, BERT and GPT-4o. Note that the classical models (LR, RF, NB) use only the normalised macro-level interactivity score as input features (in other words, they have only 4 input features), while BERT and GPT-4o uses a concatenation of the dialogue, micro-level feature spans, and macro-level interactivity scores as input.

For more details on configurations and prompts, see Appendix A.5 for BERT and A.6 for GPT-4o.

## 6 Experiments

We first replicate the ESL experiments of Gao et al. (2025) for CSL and compare their results in Section 6.1. We then assess the performance of our proposed automated approach for SL assessment in Section 6.2.

## 6.1 Transferability of the Evaluation Framework from ESL to CSL

For the human-annotated micro-level features, Gao et al. (2025) convert them into normalised counts and use them as input features to train an LR, NB and RF classifiers to predict macro-level interactivity labels. They found: (1) strong prediction performance; (2) high-impact micro-level features that are common across all interactivity labels (by interpreting feature importance given by the trained classifiers); and (3) high-impact micro-level features that are specifically predictive for an interactivity label. In the following, we replicate the experiments using our annotated CSL data (CN-IMA), to see whether the results transfer across languages; details can be found in Appendix A.10.

For (1), we present the F1 performance of predicting macro-level interactivity labels for CNIMA in Table 3. We see a strong performance, where F1 is over 0.8 in most models over the 4 interactivity labels. These results echo the ESL results in Gao et al. (2025),<sup>12</sup> providing evidence that the evaluation framework is robust across languages.

For (2), we present the results in Table 4. We found that micro-level features such as ‘Feedback in Next Turn’ and ‘Reference Word’ are highimpact features for both ESL and CSL — this underlines their fundamental role in impacting dialogue interactive dynamics, and it is languageuniversal. Interestingly, however, we also found some differences. For example, ‘Noun & Verb Collocation’ and ‘Routinized Resources’ are strong features only for CSL, and this might be because, in Chinese, these fixed terms of expressions are often used to show social closeness in open-domain conversations (Roever and Dai, 2021). ‘Code Switching’ and ‘Tense Choice’, on the other hand, are two strong features only for ESL. This is intuitive, as Chinese has no tenses while English tenses correlate with the social expression in communications (Lam, 2018). For the full comparison results, see Appendix Table 14.

For (3), Table 5 and 6 present the CSL and ESL results, respectively. We found that for topic management, ‘Negotiation of Meaning’ and ‘Question-Based Responses’ are high-impact micro-level features for both ESL and CSL, demonstrating that these are language-universal features important to drive the flow of topics in conversation. For tone appropriateness, we generally see less commonality (exceptions: ‘Feedback in Next Turn’, and ‘Routinized Resources’ which appear in both languages), suggesting that languages tend to use different features for managing tones (Zilio et al., 2017). For conversation opening, we see some similarities (e.g. ‘Question-Based Responses’) and also divergences (e.g. ‘Subordinate Clauses’ and ‘Adj./Adv. Expressing’ for ESL and ‘Epistemic Copulas’ and ‘Epistemic Modals’ for CSL). One explanation is that English tends to use adjectives and adverbs to extend a topic (e.g. Totally, I think...), while Chinese prefers modals and copulas “应该吧，或许呢” (translation: possibly yeah) to control topic change (Alduais et al., 2022). For conversation closing, we see a more similar trend, where ‘Collaborative Finishes’ and ‘Backchannels are strong features across both ESL and CSL. Despite some language-specific variations, the strategies of ending a conversation are largely similar in human communication (Lam, 2021).

<table><tr><td>Language</td><td>LR</td><td>RF</td><td>NB</td></tr><tr><td>ESL</td><td>Code Switching Reference Word* Feedback in Next Turn* Formulaic Responses</td><td>Code Switching Feedback in Next Turn* Question-based responses Non-factive Verb</td><td>Feedback in Next Turn* Formulaic Responses Reference Word* Negotiation of Meaning</td></tr><tr><td>CSL</td><td>Tense Choice Feedback in the Next Turn Noun &amp; Verb Collocation Tense Choice Reference Word* Subordinate Clauses*</td><td>Reference Word* Subordinate Clauses* Routinized Resources Reference Word* Code-Switching Feedback in Next Turn</td><td>Tense Choice Negotiation of Meaning Noun &amp; Verb Collocation Routinized Resources Subordinate Clauses* Reference Word*</td></tr></table>

Table 4: High impact common micro-level features over the three classifiers for predicting macro-level features with overlapping features in two/three classifiers by Bold/asterisk. ESL results are reproduced from Gao et al. (2025).

## 6.2 Evaluation of Automated Pipeline

We now evaluate our automated 3-step approach to predicting the overall dialogue quality score as results in Table 7. Our pipeline predicts micro- and macro-level features and dialogue overall quality scores. In terms of model names, each component denotes the model used in a particular step, e.g. “BERT+LR+LR” means we use BERT for step 1 and LR for steps 2 and 3. Also, “GPT4” refers to GPT-4o. For brevity, we only include LR results for the classical models as they all have similar performances. In addition to our 3-step approach, we also include 2 baselines that predict the overall dialogue quality score directly based on the input dialogue: (1) fine-tuned BERT (“BERT (One-step)”); and (2)

one-shot GPT-4o with instructions (“GPT4 (Onestep)”). We also include a variation where the first step uses human-annotated micro-level features (e.g. “Human+LR+LR”) to understand how much performance degrades when substituting them with predicted features.

Interestingly, the baselines (“BERT (One-step)” and “GPT4 (One-step)”) perform quite poorly, achieving F1 scores of 0.379 and 0.585, respectively. This indicates using the raw dialogue directly for predicting the overall quality is difficult (most studies in NLP, however, follow this setup for dialogue evaluation (Finch et al., 2023; Zhao et al., 2022; Yang et al., 2024)). For the variation where we use human-annotated micro-level features (“Human+LR+LR”, “Human+BERT+BERT”, and “Human+GPT4+GPT4”), we see that BERT is generally the best model and LR the worst, which is no surprise, given that BERT is pre-trained. GPT4, however, is not far from BERT, even though it is not fine-tuned. Overall, the performance is encouraging, and we see that the best models achieve over 0.80 F1.

When we look at the fully automated pipeline (bottom 4 rows in Table 7), “BERT+BERT+BERT” and “GPT4+GPT4+GPT4” perform very strongly (0.807 and 0.791), demonstrating that we have a fully automated system that can reliably assess the quality score of a dialogue. We also notice an interesting trend: When we use either BERT or GPT4 for predicting the micro-level features, LR (i.e. “BERT+LR+LR” and “GPT4+LR+LR”) performs very poorly for predicting the overall score, even though the span prediction performance of BERT and GPT4 for the micro-level features is not poor (we will revisit this in Section 6.3). This suggests that the classical machine learning models are less tolerant of noise.<sup>13</sup>

<table><tr><td>Topic</td><td>Tone</td><td>Opening</td><td>Closing</td></tr><tr><td colspan="4">Logistic Regression</td></tr><tr><td>Negotiation of Meaning Epistemic Copulas Collaborative Finishes Question-Based Responses</td><td>Feedback in Next Turn Collaborative finishes Routinized Resources Formulaic Responses</td><td>Epistemic Copulas Formulaic Responses Question-Based Responses* Epistemic Modals</td><td>Backchannels Collaborative Finishes* Epistemic Copulas</td></tr><tr><td>Backchannels*</td><td>Question-Based Responses Naïve Bayes</td><td>Collaborative finishes</td><td>Subordinate Clauses Formulaic Responses</td></tr><tr><td colspan="4">Non-factive Verb Phrase Epistemic copulas</td></tr><tr><td>Collaborative Finishes Code-switching</td><td>Collaborative finishes Impersonal subject + non-factive verb</td><td>Code-switching Feedback in Next Turn Epistemic Modals</td><td>Code-switching Epistemic Modals Epistemic Copulas</td></tr><tr><td>Backchannels*</td><td>Backchannels</td><td>Epistemic Copulas</td><td>Collaborative Finishes*</td></tr><tr><td>Epistemic Copulas</td><td>Question-Based Responses</td><td>Question-Based Responses*</td><td>Question-Based Responses</td></tr><tr><td colspan="4"></td></tr><tr><td></td><td>Random Forest Tense choice</td><td>Feedback in Next Turn</td><td></td></tr><tr><td>Noun &amp; verb collocation Negotiation of Meaning</td><td>Backchannels</td><td>Noun &amp; Verb Collocation</td><td>Backchannels Subordinate Clauses</td></tr><tr><td>Backchannels*</td><td>Negotiation of Meaning</td><td>Collaborative Finishes</td><td>Formulaic Responses</td></tr><tr><td>Question-Based Responses Collaborative Finishes</td><td>Feedback in Next Turn Routinized Resources</td><td>Question-Based Responses* Formulaic Responses</td><td>Collaborative Finishes* Question-based Responses</td></tr></table>

Table 5: CSL: High impact interactivity-specific micro-level features. For each interactivity label, bold/asterisk indicates overlapping features in two/three classifiers
<table><tr><td>Topic</td><td>Tone</td><td>Opening</td><td>Closing</td></tr><tr><td colspan="4">Logistic Regression</td></tr><tr><td>Negotiation of Meaning* Subordinate Clauses* Noun&amp;Verb Collocation Question-Based Responses</td><td>Routinized Resources* Adj./Adv. Expressing Feedback in Next Turn* Formulaic Responses*</td><td>Epistemic Modals Formulaic Responses Question-Based Responses* Subordinate Clauses*</td><td>Backchannels* Adj./Adv. Expressing Formulaic Responses Collaborative Finishes* Epistemic Copulas</td></tr><tr><td colspan="4">Subordinate clauses* Reference Word Adj./Adv. Expressing*</td></tr><tr><td>Non-factive Verb Phrase Question-Based Responses Adj./Adv. Expressing</td><td>Routinized Resources* Feedback in Next Turn* Epistemic Copulas</td><td>Naïve Bayes Adj./Adv. Expressing* Routinized Resources Subordinate Clauses*</td><td>Adj./Adv. Expressing Epistemic Modals Backchannels*</td></tr><tr><td>Negotiation of Meaning* Subordinate clauses*</td><td>Question-Based Responses Subordinate Clauses*</td><td>Epistemic Copulas Question-Based Responses*</td><td>Collaborative Finishes* Question-Based Responses</td></tr><tr><td colspan="4">Random Forest</td></tr><tr><td>Negotiation of Meaning* Formulaic Responses Subordinate Clauses* Epistemic Copulas Question-Based Responses</td><td>Epistemic Copulas Backchannels Feedback in Next Turn* Negotiation of Meaning Routinized Resources*</td><td>Feedback in Next Turn Subordinate Clauses* Adj./Adv. Expressing* Question-Based Responses* Formulaic Responses</td><td>Feedback in Next Turn Subordinate clauses Collaborative Finishes* Formulaic Responses Backchannels*</td></tr></table>

Table 6: ESL: High impact interactivity-specific micro-level features (reproduced from Gao et al. (2024)). For each interactivity label, bold/asterisk indicates overlapping features in two/three classifiers.

Taking all these results together, we show that when it comes to assessing SL dialogue quality, it is important to take a pipeline approach to predict important intermediate features (i.e. micro- and macro-level) before predicting the overall quality score. This design also has another advantage: it is more interpretable as given an overall score, we can also look at the intermediate results to understand what and where went wrong, providing more meaningful feedback that can benefit both teachers and second language students. Lastly, the strong performance of GPT4 also has another important implication: we can adapt our SL assessment system to another language without requiring large-scale manually-annotated data.

<table><tr><td>Models</td><td>F1</td></tr><tr><td>BERT (One-step) GPT4 (One-step)</td><td>0.379 0.585</td></tr><tr><td>Human+LR+LR Human+BERT+BERT Human+GPT4+GPT4</td><td>0.772 0.860 0.842</td></tr><tr><td>BERT+LR+LR BERT+BERT+BERT</td><td>0.329 0.807</td></tr><tr><td>GPT4+LR+LR GPT4+GPT4+GPT4</td><td>0.667 0.791</td></tr></table>

Table 7: F1 performance of the dialogue overall score for the CSL dialogue in three steps in different models
<table><tr><td>Models</td><td>Topic</td><td>Tone</td><td>Opening</td><td>Closing</td></tr><tr><td>LR</td><td>0.643</td><td>0.357</td><td>0.081</td><td>0.079</td></tr><tr><td>RF</td><td>0.463</td><td>0.235</td><td>0.066</td><td>0.090</td></tr><tr><td>NB</td><td>0.560</td><td>0.290</td><td>0.106</td><td>0.117</td></tr></table>

Table 8: Feature Importance of the four interactivity labels across three machine learning models

## 6.3 Additional Analyses

To understand how macro-level interactivity labels contribute to the overall quality score, we present their feature importance as learned by classical machine learning models in the “Human+LR+LR”, “Human+RF+RF” and “Human+NB+NB” models in Table 8. Topic management appears to be the most important predictor, followed by tone appropriateness. Conversation opening and closing, on the other hand, has relatively small weights. These observations are supported by Roever and Ikeda (2023), who similarly found that topic flow and tone are dominant in deciding interaction quality.

Next, we assess how well BERT and GPT4 perform for the first (micro-level feature span prediction) and the second-step predictions (macro-level label classification); full results in AppendixA.5 Table14 and 15 for BERT and GPT4 respectively. To summarise, for most of the 17 micro-level features (step 1), both BERT and GPT4 perform very well (average F1 for BERT and GPT4 is 0.89 and 0.81, respectively). That said, BERT and GPT4 struggle on two features related to non-factive verbs, and this can be attributed to the rarity of these features in the Chinese (Roever and Ikeda, 2023). Overall, these results show that the proposed micro-features transfer very well from English to Chinese.

For the second step, we look at the performance of “BERT+BERT” and “GPT4+GPT4” for predicting the macro-level labels (Appendix Table 16).

Across all labels, both BERT and GPT4 perform well, averaging about 0.83 F1 for BERT and 0.78 F1 for GPT4, respectively. Together, these encouraging intermediate evaluation results explain why our pipeline approach can consistently score the overall dialogue quality.

## 7 Conclusion

We introduce CNIMA, a large-scale annotated dataset for Chinese-as-a-Second-Language (CSL) dialogues based on the evaluation framework introduced by Gao et al. (2025). This allows us to test the cross-lingual transferability of the framework, as it is originally designed for English. Our experiments demonstrate that the framework works well for CSL dialogues, suggesting that it is robust across languages and that it is a viable framework for evaluating any second-language conversations. Our analyses also reveal language-specific and language-universal relationships between microlevel linguistic features (such as backchannels) and macro-level interactivity features (such as topic management). Next, we propose an automated, interpretable approach that predicts the micro- and macro-level features and the overall quality of second-language dialogues. To this end, we explore a range of models, from classical machine learning models to LLMs. Our approach is interpretable because it does not only predict the overall dialogue quality score, but also reveals the key microlevel and macro-level features that contribute to the score. Our LLM-based approach performs very well, and as it does not require any labelled data for training, it can be adapted to other languages easily. Ultimately, our method paves the way for an interpretable and practical system for automatic second-language dialogue assessment.

## 8 Limitations

The original speech data is transcribed manually into text. Technically we can use speech-to-text software to automate this, but since we are working with second language conversations, this can result in much noise in the translated text. The scope of the current paper is limited to CSL, but it would be equally interesting to see how the evaluation framework would work for other SL dialogues, which calls on a joint contribution of the community effort on more SL language dialogue datasets, like Japanese second language datasets, Spanish second language datasets and etc.

## Ethics Statement

This study is conducted under the guidance of the ACL Code of Ethics. We manually filtered out potentially offensive content and removed all information related to the identification of annotators.

## References

Ahmed Alduais, Issa Al-Qaderi, and Hind Alfadda. 2022. Pragmatic language development: Analysis of mapping knowledge domains on how infants and children become pragmatically competent. Children, 9(9):1407.

Jens Allwood. 2008. Dimensions of embodied communication-towards a typology of embodied communication. Embodied communication in humans and machines, pages 257–284.

Jill Burstein, Geoffrey T LaFlair, Antony John Kunnan, and Alina A von Davier. 2021. A theoretical assessment ecosystem for a digital-first assessment—the duolingo english test. DRR-21-04.

Ming-Bin Chen, Jey Han Lau, and Lea Frermann. 2023. The uncivil empathy: Investigating the relation between empathy and toxicity in online mental health support forums. In Proceedings ofthe 21st Annual Workshop of the Australasian Language Technology Association, pages 136–147.

Israel Cohen, Yiteng Huang, Jingdong Chen, Jacob Benesty, Jacob Benesty, Jingdong Chen, Yiteng Huang, and Israel Cohen. 2009. Pearson correlation coefficient. Noise reduction in speech processing, pages 1–4.

Yiming Cui, Wanxiang Che, Ting Liu, Bing Qin, and Ziqing Yang. 2021. Pre-training with whole word masking for chinese bert. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 29:3504–3514.

Ronald Cumbal. 2024. Robots Beyond Borders: The Role of Social Robots in Spoken Second Language Practice. Ph.D. thesis, KTH Royal Institute of Technology.

David Wei Dai. 2022. Design and validation ofan L2- Chinese interactional competence test. Ph.D. thesis, University of Melbourne (Australia).

Olov Engwall, Ronald Cumbal, José Lopes, Mikael Ljung, and Linnea Månsson. 2022. Identification of low-engaged learners in robot-led second language conversations with adults. ACM Transactions on Human-Robot Interaction (THRI), 11(2):1–33.

Keelan Evanini, Maurice Cogan Hauck, and Kenji Hakuta. 2017. Approaches to automated scoring of speaking for k–12 english language proficiency assessments. ETS Research Report Series, 2017(1):1– 11.

Sarah E. Finch, James D. Finch, and Jinho D. Choi. 2023. Don’t forget your ABC’s: Evaluating the state-of-the-art in chat-oriented dialogue systems. In Proceedings of the 61st Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 15044–15071, Toronto, Canada. Association for Computational Linguistics.

Rena Gao, Carsten Roever, and Jey Han Lau. 2024. Interaction matters: An evaluation framework for interactive dialogue assessment on english second language conversations.

Rena Gao, Carsten Roever, and Jey Han Lau. 2025. Interaction matters: An evaluation framework for interactive dialogue assessment on English second language conversations. In Proceedings ofthe 31st International Conference on Computational Linguistics, pages 10977–11012, Abu Dhabi, UAE. Association for Computational Linguistics.

Wei Gao and Menghan Wang. 2024. Listenership always matters: active listening ability in l2 business english paired speaking tasks. International Review of Applied Linguistics in Language Teaching.

Kaixuan Gong. 2023. Challenges and opportunities for spoken english learning and instruction brought by automated speech scoring in large-scale speaking tests: a mixed-method investigation into the washback of speechrater in toefl ibt. Asian-Pacific Journal ofSecond and Foreign Language Education, 8(1):25.

Tim Greer. 2023. Grammar-in-interaction and its place in assessing interactional competence. Applied Pragmatics, 5(2).

Lishan Huang, Zheng Ye, Jinghui Qin, Liang Lin, and Xiaodan Liang. 2020. GRADE: Automatic graphenhanced coherence metric for evaluating opendomain dialogue systems. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 9230–9240, Online. Association for Computational Linguistics.

Jeremy F Jones and Quanling Liu. 2023. Analyzing test-takers’ experiences of high-stakes automated language testing. English as a Foreign Language International Journal, 3(1):1–41.

Nahal Khabbazbashi, Jing Xu, and Evelina D Galaczi. 2021. Opening the black box: Exploring automated speaking evaluation. Challenges in Language Testing Around the World: Insights for language test users, pages 333–343.

Klaus Krippendorff. 2018. Content analysis: An introduction to its methodology. Sage publications.

Daniel MK Lam. 2018. What counts as “responding”? contingency on previous speaker contribution as a feature of interactional competence. Language Testing, 35(3):377–401.

Daniel MK Lam. 2021. Don’t turn a deaf ear: A case for assessing interactive listening. Applied Linguistics, 42(4):740–764.

Charles McGhee, Katherine Knill, and Mark Gales. 2024. Highly intelligible speaker-independent articulatory synthesis.

OpenAI. 2024. Hello gpt-4o.

José Carlos Paiva, José Paulo Leal, and Álvaro Figueira. 2022. Automated assessment in computer science education: A state-of-the-art review. ACM Transactions on Computing Education (TOCE), 22(3):1–40.

Carsten Roever and David W Dai. 2021. Reconceptualizing interactional competence for language testing. Assessing speaking in context: Expanding the construct and its applications, pages 23–49.

Carsten Roever and Naoki Ikeda. 2023. The relationship between l2 interactional competence and proficiency. Applied Linguistics, page amad053.

Abdullayev Shaxobiddin. 2024. A discourse analysis of modal verbs in modern english: Patterns and functions. Journal of new century innovations, 50(2):145– 147.

Eric Smith, Orion Hsu, Rebecca Qian, Stephen Roller, Y-Lan Boureau, and Jason Weston. 2022. Human evaluation of conversations is an open problem: comparing the sensitivity of various methods for evaluating dialogue agents. In Proceedings of the 4th Workshop on NLP for Conversational AI, pages 77– 97, Dublin, Ireland. Association for Computational Linguistics.

Jingxuan Wu and Carsten Roever. 2021. Proficiency and preference organization in second language mandarin chinese refusals. The Modern Language Journal, 105(4):897–918.

Jing Xu. 2018. Measuring “spoken collocational competence” in communicative speaking assessment. Language Assessment Quarterly, 15(3):255–272.

Jing Xu, Edmund Jones, Victoria Laxton, and Evelina Galaczi. 2021. Assessing l2 english speaking using automated scoring technology: examining automarker reliability. Assessment in Education: Principles, Policy & Practice, 28(4):411–436.

Bohao Yang, Kun Zhao, Chen Tang, Liang Zhan, and Chenghua Lin. 2024. Structured information matters: Incorporating abstract meaning representation into llms for improved open-domain dialogue evaluation. arXiv preprint arXiv:2404.01129.

Foong Ha Yap and Anindita Sahoo. 2024. Versatile copulas and their stance-marking uses in conversational odia, an indo-aryan language. Lingua, 297:103641.

Jianqiao Zhao, Yanyang Li, Wanyu Du, Yangfeng Ji, Dong Yu, Michael Lyu, and Liwei Wang. 2022. Flow-Eval: A consensus-based dialogue evaluation framework using segment act flows. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 10469–10483, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Leonardo Zilio, Rodrigo Wilkens, and Cédrick Fairon. 2017. Using nlp for enhancing second language acquisition. In RANLP, pages 839–846.

## A Appendix

## A.1 Software Availability

To contribute to the research community and facilitate further development and collaboration, we have made the source codes of our innovative annotation tool publicly available. The tool, designed with a focus on enhancing the efficiency and accuracy of data annotation processes, has been developed through meticulous research and development efforts. It incorporates a range of features tailored to meet the needs of researchers and practitioners working in fields that require precise and reliable annotation of datasets.

## Accessing the Source Code

The source codes are hosted on GitHub, a platform widely recognized for its robust version control and collaborative features. Interested parties can access the repository at the following link: https: //anonymous.4open.science/r/AnnotationTool2023-CFE1/README.md. This repository is intended for research usage, underlining our commitment to supporting academic and scientific endeavours.

## Key Features and Capabilities

Our annotation tool stands out for its user-friendly interface, which simplifies the annotation process and allows users to work more efficiently. Among its key features are:

• Customizable Annotation Labels: Users can add their own set of labels to cater to the specific requirements of their projects.

• Collaborative Annotation Support: Facilitating teamwork, the tool allows multiple annotators to work on the same dataset simultaneously, ensuring consistency and reducing the time required for project completion.

• Annotation History Tracking: All the annotation history, such as changes made, can be tracked, and any further modifications can be done at any time for the user’s convenience. Export Functionality: Annotated data can be exported in several formats, accommodating further analysis or use in machine learning models.

## A.2 Pages View For Span Annotation Website Interface

![](images/7591bc41fc5a394abb419133f4f4fd1c20583bddfecaa573fc2e290ffe023efe.jpg)  
Figure 3: Annotation tool Demo

![](images/c53be16653c512a8fde1e28da2011466729718027f2d8790fdbdbe609acdb0f2.jpg)  
Figure 4: Hierarchical Label Assignment Demo

## Annotation Munnal for the CNIMA Dataset

## 1. Introduction to the annotation task

The research aims to investigate the interactive ability of second-language speakers of Chinese through automated dialogue evaluation. This study has been approved by Human Ethics.

In your annotation, two types of dialogue tasks would be included in this study conducted by a pair-wise discussion by second-language speakers of Chinese participants. The first task is an elicited conversation task, in this part, two speakers will share some experience or what they want to deliver based on the instructions (e.g., share some ideas on how you think of education in your life). In the second task, two speakers need to role-play through a joint discussion.

The dialogue of the two tasks was both transcribed into text, and you are ready to annotate based on the text. Please notify the researcher if you find any misaligned information in the transcriptions compared with the original recordings during your annotation.

## 2. Hierarchy sequence of the label

<table><tr><td colspan="1" rowspan="1">Label name</td><td colspan="1" rowspan="1">Label level</td><td colspan="1" rowspan="1">Label tag</td><td colspan="1" rowspan="1">example</td></tr><tr><td colspan="1" rowspan="1">reference word</td><td colspan="1" rowspan="3">Tokenlevellabels</td><td colspan="1" rowspan="1">[RA]</td><td colspan="1" rowspan="1">你到哪里啦？像咱们这样的不是都过了吗？</td></tr><tr><td colspan="1" rowspan="1">noun &amp; verbcollocation in properform</td><td colspan="1" rowspan="1">[NVC]</td><td colspan="1" rowspan="1">能帮我个小忙吗？帮：动词小忙：名词</td></tr><tr><td colspan="1" rowspan="1">code-switching forcommunicativepurposes</td><td colspan="1" rowspan="1">[CS]</td><td colspan="1" rowspan="1">我觉得，well,这个是吧well:英文</td></tr><tr><td colspan="1" rowspan="1">negotiation ofmeaning (appropriatetense to showmeaning)</td><td colspan="1" rowspan="4"></td><td colspan="1" rowspan="1">[NM]</td><td colspan="1" rowspan="1">SPK 1：苏州,苏州SPK_2:哦,苏州是南方的城市就比较热一点SPK 1:那么看你还是没有做SPK_2:的确是的毕竟我一点也没学.</td></tr><tr><td colspan="1" rowspan="1">tense choice toindicate interactiveaims (politeness intalking/ socialdistance/ contextvariance)</td><td colspan="1" rowspan="1">[TT]</td><td colspan="1" rowspan="1">SPK1:一两门，啊，那我们现在还能学啥呀?</td></tr><tr><td colspan="1" rowspan="1">routinized resources(projectorconstruction)</td><td colspan="1" rowspan="1">[RR]</td><td colspan="1" rowspan="1">哎，你说你</td></tr><tr><td colspan="1" rowspan="1">subordinate clauses</td><td colspan="1" rowspan="1">[RC]</td><td colspan="1" rowspan="1">你是数学好但是我还是觉得这种东西需要自己用很长的时间弄了，才是真的会了.</td></tr><tr><td colspan="1" rowspan="1">backchannels</td><td colspan="1" rowspan="4">Utterancelevel labels</td><td colspan="1" rowspan="1">[BC]</td><td colspan="1" rowspan="1">会的，会有点累</td></tr><tr><td colspan="1" rowspan="1">question-basedresponses</td><td colspan="1" rowspan="1">[QR]</td><td colspan="1" rowspan="1">SPK_1:我不是把她教的那个知识给反驳了吗？SPK_2:你既然有反驳的能力，那你还是自学吧.</td></tr><tr><td colspan="1" rowspan="1">formulaic responses(固定用词)</td><td colspan="1" rowspan="1">[FR]</td><td colspan="1" rowspan="1">但是不管怎么样我也学会了。差不了多少。</td></tr><tr><td colspan="1" rowspan="1">collaborative finishes</td><td colspan="1" rowspan="1">[CF]</td><td colspan="1" rowspan="1">SPK1:好的再见</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="7"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">SPK 2:再见</td></tr><tr><td colspan="1" rowspan="1">epistemic copulas</td><td colspan="1" rowspan="1">[H1]</td><td colspan="1" rowspan="1">一个人去还是觉得有点变扭.</td></tr><tr><td colspan="1" rowspan="1">epistemic modals</td><td colspan="1" rowspan="1">[H2]</td><td colspan="1" rowspan="1">我觉得像苏州好像有个这种辅导</td></tr><tr><td colspan="1" rowspan="1">adjectives/ adverbsexpressing possibility</td><td colspan="1" rowspan="1">[H3]</td><td colspan="1" rowspan="1">我估计可能有些这样的情况</td></tr><tr><td colspan="1" rowspan="1">non-factive verbphrase structure</td><td colspan="1" rowspan="1">[H4]</td><td colspan="1" rowspan="1">你可以认识他们我姑且能跟上吧</td></tr><tr><td colspan="1" rowspan="1">impersonal subject +non-factive verb + NP</td><td colspan="1" rowspan="1">[H5]</td><td colspan="1" rowspan="1">我认为可能他会迟到Impersonal Subject (可能): This subject isimpersonal because itdoes not refer to aspecific individual butrather expresses apossibility.Non-Factive Verb (认为): This verb isnon-factive because itintroduces a belief oropinion rather than afact.Noun Phrase (他会迟到): This is the nounphrase that completesthe sentence, statingwhat is believed orthought.</td></tr><tr><td colspan="1" rowspan="1">feedback in the nextturn</td><td colspan="1" rowspan="1">[FB]</td><td colspan="1" rowspan="1">-你感觉如何？-会的，会有点累</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">topic extension withclear new context(change to utterancelevel, but moreinformation contextdepends</td><td colspan="1" rowspan="6">Dialoguelevel labels</td><td colspan="1" rowspan="1">[T1]</td><td colspan="1" rowspan="1">我只有你这个朋友，你又不肯帮我。别人跟我关系都很一般的啊。你知道我什么意思吧？</td></tr><tr><td colspan="1" rowspan="1">topic extension underthe previous direction</td><td colspan="1" rowspan="1">[T2]</td><td colspan="1" rowspan="1">说到朋友，我只有你这个朋友。</td></tr><tr><td colspan="1" rowspan="1">topic extension withthe same content</td><td colspan="1" rowspan="1">[T3]</td><td colspan="1" rowspan="1">你说朋友啊，我觉得吧，很难说。</td></tr><tr><td colspan="1" rowspan="1">repeat and no topicextension</td><td colspan="1" rowspan="1">[T4]</td><td colspan="1" rowspan="1">关于朋友的事吗？</td></tr><tr><td colspan="1" rowspan="1">no topic extensionand stop the topic atthis point</td><td colspan="1" rowspan="1">[T5]</td><td colspan="1" rowspan="1">朋友?</td></tr><tr><td colspan="1" rowspan="1">conversation opening</td><td colspan="1" rowspan="1">[CO1][CO2][CO3][CO4][CO5]</td><td colspan="1" rowspan="1">CO1: nice greeting andshow a goodunderstanding ofconversation opening insocial interactions.CO2: sounded greetingand show a basicunderstanding of thesocial role.CO3: general greetingand didn't demonstrate agood understanding of thesocial role.CO4: basic greeting.CO5: no opening juststart the discussionimmediately.</td></tr><tr><td colspan="1" rowspan="1">conversation closing</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">[CC1][CC2][CC3][CC4][CC5]</td><td colspan="1" rowspan="1">CC1: detailedsummarization andsmooth transition to theclosing of theconversation.</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="6"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">CC2: transit to the closingnaturally, but without anysummarization of thediscussion.CC3: demonstrate atranslation to the end ofthe conversation.CC4: transit to the end ofthe discussion.CC5: no closing, just stopthe conversation.</td></tr><tr><td colspan="1" rowspan="1">overall tone choice:very formal</td><td colspan="1" rowspan="1">[OT1]</td><td colspan="1" rowspan="1">很荣幸与您见面，幸会。</td></tr><tr><td colspan="1" rowspan="1">overall tone choice:quite formal and someexpressions are notthat formal</td><td colspan="1" rowspan="1">[OT2]</td><td colspan="1" rowspan="1">见到你真好啊，最近如何？</td></tr><tr><td colspan="1" rowspan="1">overall tone choice:relatively not formal,most expressions arequite informal</td><td colspan="1" rowspan="1">[OT3]</td><td colspan="1" rowspan="1">好久不见哎，真是有段日子了啊。</td></tr><tr><td colspan="1" rowspan="1">overall tone choice:quite informal, butsome expressions arestill formal</td><td colspan="1" rowspan="1">[OT4]</td><td colspan="1" rowspan="1">真是有阵子不见了，别来无恙啊哥们。</td></tr><tr><td colspan="1" rowspan="1">overall tone choice:very informal</td><td colspan="1" rowspan="1">[OT5]</td><td colspan="1" rowspan="1">我天，真是老久没见了铁子，抱一个！</td></tr></table>

## 3. Label detailed definitions

<table><tr><td>Label Category</td><td>Aspect</td><td>Definition</td></tr><tr><td>Reference word</td><td>Word choice</td><td>A reference word, also known as a referential word or referent, is a linguistic term used to describe a word or expression in a sentence that refers to or stands in place of something</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="3"></td><td colspan="1" rowspan="1">words are used to avoidrepetition and to link differentparts of a text together byindicating what a subsequentword or phrase relates to.Reference words can takevarious forms, includingpronouns, demonstratives, andother words that replace orpoint to nouns or noun phrases.</td></tr><tr><td colspan="1" rowspan="1">Noun &amp; verb collocationin proper form</td><td colspan="1" rowspan="1">Collocations are words orphrases that habitually occurtogether, forming a strong andnatural linguistic association. Inthe case of noun-verbcollocations, a particular noun isoften paired with a particularverb due to convention,tradition, or linguistic patterns.These collocations contribute tothe fluency, idiomaticity, andnaturalness of language.Examples of noun-verbcollocations:Make a decision: "I need tomake a decision."Take a shower: "I usually take ashower in the morning."Catch a cold: "I hope I don'tcatch a cold."Give a speech: "She gave aninspiring speech."</td></tr><tr><td colspan="1" rowspan="1">Code-switching forcommunicative purposes</td><td colspan="1" rowspan="1">Code-switching forcommunicative purposes refersto the deliberate orsubconscious alternationbetween two or more languagesor dialects within a singleconversation or utterance bybilingual or multilingualspeakers. This linguisticphenomenon is employed tofulfill specific communicativeneeds or functions, such asclarifying a point, expressingidentity, signaling solidarity or</td></tr><tr><td>Negotiation of meaning (appropriate tense to</td><td></td><td>distinction, accommodating to the listener's language preference, or conveying concepts and emotions more effectively in one language over another. Code-switching is not merely a random mixing of languages but a sophisticated communicative strategy that reflects the speaker's linguistic competence and cultural awareness, often used to navigate and negotiate the social and contextual dynamics of interaction.</td></tr><tr><td>show meaning)</td><td>Contextual tense usage</td><td>Negotiation of meaning refers to the interactive process through which speakers of different linguistic backgrounds or competencies collaboratively work to understand each other's intentions, messages, and linguistic expressions when communication breakdowns occur. This involves the use of clarification requests, confirmation checks, comprehension checks, and paraphrasing, among other communicative strategies, to ensure mutual understanding is achieved. The negotiation of meaning is a fundamental aspect of second language acquisition and communicative language teaching, highlighting the dynamic nature of language use and the active role learners play in constructing meaning</td></tr><tr><td>Tense choice to indicate interactive aims (politeness / social distance/ context)</td><td>through interaction.</td><td>Tense choice to indicate interactive aims involves the strategic use of verb tenses by speakers to fulfill specific communicative goals or intentions within an interaction. This linguistic strategy encompasses the selection of</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">present, past, future, or perfecttenses to convey nuances oftime, mood, or aspect, directlyinfluencing the interpretationand direction of the dialogue.Through careful tense selection,speakers can clarify the timingof events, express certainty orspeculation about futureoccurrences, reflect on pastexperiences, or emphasize thecontinuity or completion ofactions, all of which serve toenhance the clarity,persuasiveness, or relationaldynamics of thecommunication. Tense choice,therefore, is not merely agrammatical decision but adeliberate tool employed byadept language users tonavigate conversations andachieve specific interactiveaims.</td></tr><tr><td colspan="1" rowspan="1">routinized resources(projector construction)</td><td colspan="1" rowspan="2">Interactionalgrammaticaldevice</td><td colspan="1" rowspan="1">Routinized resources refer topatterns, practices, or tools thathave become standardized andregularly employed withinspecific contexts or activities.These resources are oftendeveloped through repeateduse over time, leading to a levelof automation or routine in theirapplication. In organizational orsocial settings, routinizedresources help in streamliningprocesses, reducing the needfor decision-making aboutroutine tasks, and ensuringconsistency in actions andoutcomes. They can includedocumented procedures,established workflows, habitualpractices, or even commonlanguage and scripts used ininterpersonal interactions.</td></tr><tr><td colspan="1" rowspan="1">subordinate clauses</td><td colspan="1" rowspan="1">Subordinate clauses, alsoknown as dependent clauses,</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">are groups of words thatcontain a subject and a verb butdo not express a completethought and therefore cannotstand alone as a sentence.They function within a sentenceby providing additionalinformation to the main clause,to which they are connected bysubordinating conjunctions(such as "because," "although,""when," "if") or relativepronouns (such as "who,""which," "that"). Subordinateclauses serve various roles insentences, including acting asadjectives, adverbs, or nouns,and are essential for addingcomplexity, detail, and nuanceto communication. Their useenables speakers and writers toarticulate relationships of causeand effect, contrast, condition,time, and more, enriching theexpressiveness and depth oflanguage.</td></tr></table>

## A.4 Speaking Task Collection Instruction

## Task One: Elicited Conversation

Elicited Conversation Task (ECVA)

Instruction: 在这部分中 你将看到一个话题 请就此话题与你的同伴展开讨论

In this section, you will be given a topic. Based on the topic, I would like you to talk together with your partner.

话题：目前，有很多人喜欢独自旅行，也有一些人喜欢和一组人共同去旅行。请讨论：

Nowadays, some people like traveling independently, while others like travels with a group of persons. Please discuss:

你喜欢自己独自出行旅游还是和一组人一起去旅行？为什么？

Do you like traveling individually or traveling with a group of persons? Why?

(温馨提示<sub>：</sub> 以下提供一些观点供你参考<sub>，</sub> 在讨论中<sub>，</sub> 你可以根据需要加入自己的新观点<sub>。</sub> 你不必将图中所有的观点全部提及。请你们至少讨论3分钟，我会在适当的时候停止你们的讨论。)

(Note: Some ideas are provided below. You can use these ideas in your discussions if you want. If you have some new ideas, you can use your own ideas in the discussion. You do not need to cover all the ideas provided below. Please discuss at least 3 minutes. I will stop you if necessary.)

![](images/a49675195410bae5ae0db4f55053c81748ee33a3fb7a66ee5fd60d614399f523.jpg)  
448

## Task Two: Role Play

## Role Play Task One (VARP1)

## Role A

提示：请你阅读以下场景，想象你自己身处此场景中。如果你有任何问题，请提出。

Note: Please read the following situation carefully and imagine yourself in the following scenario. If you have any questions, please feel free to ask.

你这个学期选择了中国文化课，这门课程要求你使用中文写一篇论文，但是你对自己的中文写作并不是那么的自信。你的好朋友中文水平非常高。你和你的朋友住在同一栋公寓。

现在，你想要去你朋友的房间请求他/她帮你改论文，你会怎么做？

You are enrolled in the Chinese culture course for this semester. One assessment task of this course is to write an essay in Chinese. However, you are not confident in your Chinese writing. Your friend’s Chinese proficiency is very high. You and your friend live in the same apartment.

Now, you decide to go to your friend’s room. How would you ask your friend to help you revise your essay?

## Role B

## 提示：

请你阅读下述场景，你需要想象自己身处这个场景中。你需要以灵活自然的方式引导整个对话。  
在你的同伴提出请求之前，请不要接受或拒绝你同伴的请求。如果你有任何问题，请提出。

## Note:

Please read the following scenario and imagine yourself in the following situation. It is your responsibility to respond to the task in a natural and flexible way. Before your interlocutor is on the record, please do not accept/ refuse your friends request. If you have any questions, please feel free to ask.

你这周需要写很多篇论文，下周就是提交这些论文的截止日期了。因此，你非常忙，并且没有足够的睡觉休息时间。你的同学们都知道你的中文水平非常好。

这时，和你同住在一栋公寓的朋友来敲你的房门。

You need to finish writing several papers this week, as the deadline for submitting the papers is next week. Therefore, you are very busy and do not have enough sleep hours. Your classmates know your Chinese proficiency is quite good.

Now, your friend, who lives in the same apartment, rings your doorbell.

## Role Play Task Two (VARP2)

Role A

提示：

请你阅读以下场景，想象你自己身处此场景中。如果你有任何问题，请提出。

Note: Please read the following situation carefully and imagine yourself in the following scenario. If you have any questions, please feel free to ask.

你想要在网上报名参加汉语水平考试（HSK）。报名 HSK 考试需要使用中国银行卡在网上支付考试费用。但是，你没有中国的银行卡。你想要借你同朋友的银行卡来支付报名费。你的这位朋友和你住在同一栋公寓。

现在，你想要去你朋友的房间，请求他借给你银行卡，你会怎么做？

You want to register for HSK examination online. You need to make an online payment by using a Chinese credit card. However, you do not have a Chinese credit card. You want to borrow a credit card from your friend who lives in the same apartment.

Now, you decide to go to your friend’s room. How would ask your friend in real life to borrow the credit card?

## Role B

提示：

请你阅读下述场景，你需要想象自己身处这个场景中。你需要以灵活自然的方式引导整个对话。  
在你的同伴提出请求之前，请不要接受或拒绝你同伴的请求。如果你有任何问题，请提出。

## Note:

Please read the following scenario and imagine yourself in the following situation. It is your responsibility to respond to the task in a natural and flexible way. Before your interlocutor is on the record, please do not accept/ refuse your friends request. If you have any questions, please feel free to ask.

你有一张信用卡，你可以使用这张信用卡在网上付款。你不想把你的信用卡借给别人，因为你觉得告诉别人你银行卡密码十分不安全。

这时，和你同住在一栋公寓的朋友来敲你的房门。

You have a credit card, so you can finish the online payment by using this card. You do not want to lend others your credit card, as you think telling the passwords to other persons is unsafe.

Now, your friend, who lives in the same apartment, rings your doorbell.

## A.5 BERT Configuration

We give detailed experimental settings in the following table. All the experiments can run via a single NVIDIA GeForce RTX 4070 GPU within a reasonable time.

Table 9: Experimental Details
<table><tr><td>Parameters</td><td>Configurations</td></tr><tr><td>Max Length</td><td>128</td></tr><tr><td>Tokenizer</td><td>Bert Pretrained Tokenizer</td></tr><tr><td>Encoder</td><td>BERT-base-uncase</td></tr><tr><td>Activation Function</td><td>GELU</td></tr><tr><td>Batch Size</td><td>32</td></tr><tr><td>Learning Rate</td><td>5e-5</td></tr><tr><td>Loss Function</td><td>CrossEntropyLoss</td></tr><tr><td>Optimizer</td><td>Adam</td></tr><tr><td>Epoch</td><td>15</td></tr></table>

## A.6 Prompts for GPT-4o

The following Table 10 shows the prompts for annotating micro-level features with GPT-4o.

## A.7 Prompts for GPT-4o Dialogue Overall Evaluation

The following Table 11 shows the prompts for dialogue overall quality score with GPT-4o.

## A.8 Overall Score Description and Definitions

The following Table 12 shows the descriptions and definitions for dialogue’s overall quality score.

## A.9 Four Interactivity Aspects Definitions and Descriptions

Table 13 shows the descriptions and definitions for dialogue’s overall quality score.

## A.10 Top-k feature computation method

This is reproduced from (Gao et al., 2024). Given that a trained LR, NB and RF classifier all provide weights to indicate the importance of each feature, for each classifier, we first compute common micro-level features $f _ { \mathrm { c } }$ across the four interactivity labels:

$$
\begin{array} { r l } & { f _ { \mathrm { c } } = \mathrm { { t o p } } 5 \big ( \mathrm { t o p } 1 0 ( f _ { \mathrm { t o p i c } } ) \cap \mathrm { t o p } 1 0 ( f _ { \mathrm { t o n e } } ) } \\ & { \qquad \cap \mathrm { t o p } 1 0 ( f _ { \mathrm { o p e n i n g } } ) \cap \mathrm { t o p } 1 0 ( f _ { \mathrm { c l o s i n g } } ) \big ) } \end{array}
$$

where topk is a function that returns the best k items given by their weights, $f _ { \mathrm { t o p i c } }$ denote the set of micro-level features with their weights for predicting the topic management interactivity label.

For micro-level features that are specific to each of the marco-level interactivity aspects. To that end, for each classifier we compute interactivity-specific features, e.g., for topic management, as follows:

$$
\mathrm { t o p 1 0 } ( f _ { \mathrm { t o p i c } } ) - f _ { c }\tag{1}
$$

## B Experimental Result

<table><tr><td colspan="1" rowspan="1">Span</td><td colspan="1" rowspan="1">Description and Example</td></tr><tr><td colspan="1" rowspan="1">Reference Word</td><td colspan="1" rowspan="1">A linguistic term used to avoid repetition and link different parts.Example: [你]到哪里去 (Where are [you] going?)</td></tr><tr><td colspan="1" rowspan="1">Noun &amp; Verb Collocation</td><td colspan="1" rowspan="1">Words or phrases that habitually occur together, forming a strong andnatural linguistic association.Example: 能[帮我一个忙]吗？ (Can you [do me a favor]?)</td></tr><tr><td colspan="1" rowspan="1">Code-switching</td><td colspan="1" rowspan="1">The alternation between languages within a single conversation.Example: 我觉得， [well], 这个是对的 (I think, well, this is right.)</td></tr><tr><td colspan="1" rowspan="1">Negotiation of Meaning</td><td colspan="1" rowspan="1">Interactive process where speakers clarify and confirm understanding.Example: SPK1：那你是没有做吗？SPK2: [的确]，我一点也没学(SPK1: So you didn't do it? SPK2: [Indeed], I haven't learned at all.)</td></tr><tr><td colspan="1" rowspan="1">Tense Choice to Show In-teractive Aims</td><td colspan="1" rowspan="1">Using verb tenses to fulfil specific communicative goals.Example: 我[现在]还不能，因为我还有很多工作 (I can't [right now]because I still have a lot of work.)</td></tr><tr><td colspan="1" rowspan="1">Routinized Resources</td><td colspan="1" rowspan="1">Prefabricated linguistic elements are used to manage dialogue interac-tions efficiently.Example: [你说的你] ([As you said])</td></tr><tr><td colspan="1" rowspan="1">Subordinate Clauses</td><td colspan="1" rowspan="1">Clauses that provide additional information to the main clauseExample: 你是数学好, [但是]我是因为需要自己努力 (You are goodat math, [but] because I need to work hard myself.)</td></tr><tr><td colspan="1" rowspan="1">Backchannels</td><td colspan="1" rowspan="1">Brief responses from a listener, such as "uh-huh".Example: [哎哎哎。]([Yep, yep, yep.])</td></tr><tr><td colspan="1" rowspan="1">Question Responses</td><td colspan="1" rowspan="1">Replies in a dialogue that directly or answer a preceding question.Example: [我还是 自学吧] (SPK2: [I shall teach myself.])</td></tr><tr><td colspan="1" rowspan="1">Formulaic Responses</td><td colspan="1" rowspan="1">Conventional phrases in dialogue to respond in familiar situations.Example: [差不了多少]([Roughly the same])</td></tr><tr><td colspan="1" rowspan="1">Collaborative Finishes</td><td colspan="1" rowspan="1">Instances in a dialogue where one speaker completes another speaker'ssentence or thought.Example: SPK1:[好嘞再见啊] SPK2:[再见了您](SPK1: [Alright, See you.] SPK2:[Goodbye.])</td></tr><tr><td colspan="1" rowspan="1">Epistemic Copulas</td><td colspan="1" rowspan="1">Phrases that express a speaker's degree of certainty about a statement,often using verbs like "is" or "seems".Example: [一个人去还是觉得有点变扭] ([I felt wired to go therealone])</td></tr><tr><td colspan="1" rowspan="1">Epistemic Modals</td><td colspan="1" rowspan="1">Modal verbs or phrases that express a speaker's judgment about thepossibility, such as "might," "must,".Example:[你应该自己学会这些中文知识的] ([You should learnthese Chinese by yourself.])</td></tr><tr><td colspan="1" rowspan="1">Adjectives &amp; adverbs ofpossibility</td><td colspan="1" rowspan="1">Adjectives or adverbs to show possibility, like "Possibly".Example: [我也许回去故乡] ([I maybe go back to hometown.])</td></tr><tr><td colspan="1" rowspan="1">Non-factive Verb Phrase</td><td colspan="1" rowspan="1">Expressions that use verbs to convey statements without asserting themas true; verbs "think," "believe," or "seem."Example: [我姑且能跟上吧] ([I can barely follow the progress.])</td></tr><tr><td colspan="1" rowspan="1">Impersonal Subject</td><td colspan="1" rowspan="1">An impersonal subject (such as "it" or "there") is followed by a non-factive verb and a noun phrase, often express opinions.Example: [这不好说吧] ([It's hard to tell.])</td></tr><tr><td colspan="1" rowspan="1">Feedback in Next Turn</td><td colspan="1" rowspan="1">Using next turn to respond other speaker.Example: [我认为你有道理] ([You words make sense.])</td></tr><tr><td>Field</td><td>Description</td></tr><tr><td>Conversation</td><td>A dialogue of second language Chinese conversation.</td></tr><tr><td>Output Fields</td><td>score: The score of the interactivity of the Chinese second lan- guage dialogue (1 to 5). rationale: The reason why and how the score is made.</td></tr><tr><td>Evaluation Criteria</td><td>5: Smooth and fluent daily communication, easy and pleasant. 4: Somewhat less fluent communication, but the communication purpose is achieved. 3: Slightly awkward communication, such as not being able to im- mediately understand the other person's question with hesitation. 2: Overall communication is not fluent and awkward, but some parts can be mutually understood. 1: Unable to accurately achieve the communication purpose, awk-</td></tr></table>

Table 10: LLM Annotation Prompts for CSL Dialogue Span annotation for Mirco-level features

Table 11: LLM Dialogue Overall Dialogue Quality Evaluation Prompts

<table><tr><td>Scores</td><td>Descriptions</td></tr><tr><td>5</td><td>Smooth and fluent daily communication,</td></tr><tr><td rowspan="2">4</td><td>easy and pleasant through the whole chat Somewhat less fluent communication,</td></tr><tr><td>but the communication purpose is achieved</td></tr><tr><td rowspan="2">3</td><td>Slightly awkward communication in some places,</td></tr><tr><td>such as not being able to understand the other person&#x27;s question</td></tr><tr><td>2</td><td>Overall communication is not fluent and mostly awkward, but some parts can be mutually understood</td></tr><tr><td>1</td><td>Unable to accurately achieve the communication purpose, awkward conversation, and failed to talk throughout the conversation.</td></tr></table>

Table 12: Score description for overall dialogue quality

<table><tr><td>Interactivity Macro-level Features</td><td>Definition</td></tr><tr><td>Topic Management</td><td>the strategies and techniques used to control and navigate the flow of topics</td></tr><tr><td>Tone Choice Appropriateness</td><td>the suitability of the tone used in communication, ensuring it aligns with the context, audience, and purpose to convey the intended message</td></tr><tr><td>Conversation Opening</td><td>the initial interaction or exchange that begins a dialogue, often setting the tone and context for the dialogue</td></tr><tr><td>Conversation Closing</td><td>the process of ending a dialogue or interaction, which involves signaling the conclusion of the discussion, summarizing key points, and often expressing a farewell</td></tr></table>

Table 13: Definitions of macro-level interactivity features, with higher score emphasising on natural, authentic interaction and active engagement in the dialogue

<table><tr><td>Response Type</td><td>Accuracy</td><td>Precision</td><td>Recall</td><td>F1 Score</td></tr><tr><td>Epistemic Copulas</td><td>0.997</td><td>0.929</td><td>0.843</td><td>0.884</td></tr><tr><td>Formulaic Responses</td><td>0.976</td><td>0.875</td><td>0.781</td><td>0.825</td></tr><tr><td>Question-based Responses</td><td>0.986</td><td>0.834</td><td>0.591</td><td>0.892</td></tr><tr><td>Non-factive Verb Phrase Structure</td><td>0.999</td><td>0.000</td><td>0.000</td><td>0.000</td></tr><tr><td>Impersonal Subject + Non-factive Verb + NP</td><td>0.997</td><td>0.909</td><td>0.243</td><td>0.384</td></tr><tr><td>Reference Word</td><td>0.990</td><td>0.985</td><td>0.989</td><td>0.987</td></tr><tr><td>Routinized Resources</td><td>0.983</td><td>0.783</td><td>0.706</td><td>0.742</td></tr><tr><td>Noun &amp; Verb Collocation in Proper Form</td><td>0.985</td><td>0.970</td><td>0.958</td><td>0.964</td></tr><tr><td>Collaborative Finishes</td><td>0.995</td><td>0.802</td><td>0.631</td><td>0.706</td></tr><tr><td>Tense Choice to Indicate Interactive Aims</td><td>0.994</td><td>0.957</td><td>0.930</td><td>0.943</td></tr><tr><td>Negotiation of Meaning</td><td>0.991</td><td>0.849</td><td>0.713</td><td>0.775</td></tr><tr><td>Code-switching for Communicative Purposes</td><td>0.999</td><td>0.976</td><td>0.954</td><td>0.965</td></tr><tr><td>Feedback in the Next Turn</td><td>0.972</td><td>0.842</td><td>0.833</td><td>0.838</td></tr><tr><td>Epistemic Modals</td><td>0.997</td><td>0.909</td><td>0.945</td><td>0.926</td></tr><tr><td>Backchannels</td><td>0.982</td><td>0.824</td><td>0.731</td><td>0.875</td></tr><tr><td>Subordinate Clauses</td><td>0.990</td><td>0.960</td><td>0.937</td><td>0.948</td></tr><tr><td>Adv &amp; Adj Expressing</td><td>0.964</td><td>0.863</td><td>0.831</td><td>0.847</td></tr></table>

Table 14: Predicted performance of micro-level features on fine-tune BERT
<table><tr><td>Response Type</td><td>Accuracy</td><td>Precision</td><td>Recall</td><td>F1 Score</td></tr><tr><td>Epistemic Copulas</td><td>0.995</td><td>0.700</td><td>0.736</td><td>0.717</td></tr><tr><td>Formulaic Responses</td><td>0.951</td><td>0.559</td><td>0.579</td><td>0.669</td></tr><tr><td>Question-based Responses</td><td>0.972</td><td>0.500</td><td>0.426</td><td>0.460</td></tr><tr><td>Non-factive Verb Phrase Structure</td><td>0.999</td><td>0.000</td><td>0.000</td><td>0.000</td></tr><tr><td>Impersonal Subject + Non-factive Verb + NP</td><td>0.998</td><td>0.000</td><td>0.000</td><td>0.000</td></tr><tr><td>Reference Word</td><td>0.987</td><td>0.982</td><td>0.983</td><td>0.982</td></tr><tr><td>Routinized Resources</td><td>0.971</td><td>0.564</td><td>0.448</td><td>0.500</td></tr><tr><td>Noun &amp; Verb Collocation in Proper Form</td><td>0.972</td><td>0.953</td><td>0.900</td><td>0.826</td></tr><tr><td>Collaborative Finishes</td><td>0.988</td><td>0.565</td><td>0.406</td><td>0.472</td></tr><tr><td>Tense Choice to Indicate Interactive Aims</td><td>0.990</td><td>0.915</td><td>0.826</td><td>0.768</td></tr><tr><td>Negotiation of Meaning</td><td>0.971</td><td>0.560</td><td>0.462</td><td>0.506</td></tr><tr><td>Code-switching for Communicative Purposes</td><td>0.999</td><td>1.000</td><td>0.941</td><td>0.869</td></tr><tr><td>Feedback in the Next Turn</td><td>0.951</td><td>0.679</td><td>0.741</td><td>0.708</td></tr><tr><td>Epistemic Modals</td><td>0.995</td><td>0.973</td><td>0.770</td><td>0.860</td></tr><tr><td>Backchannels</td><td>0.975</td><td>0.697</td><td>0.588</td><td>0.638</td></tr><tr><td>Subordinate Clauses</td><td>0.964</td><td>0.863</td><td>0.831</td><td>0.847</td></tr><tr><td>Adv &amp; Adj Expressing</td><td>0.901</td><td>0.731</td><td>0.706</td><td>0.847</td></tr></table>

Table 15: F1 performance of micro-level span annotation by GPT-4o
<table><tr><td>Models</td><td>Topic</td><td>Tone</td><td>Opening</td><td>Closing</td></tr><tr><td>BERT (raw dialogue)</td><td>0.414</td><td>0.401</td><td>0.414</td><td>0.379</td></tr><tr><td>GPT-4 (raw dialogue)</td><td>0.553</td><td>0.533</td><td>0.585</td><td>0.557</td></tr><tr><td>BERT+BERT (on annotated data)</td><td>0.987</td><td>0.990</td><td>0.993</td><td>0.978</td></tr><tr><td>BERT+BERT (based on BERT predicted micro-level features)</td><td>0.836</td><td>0.855</td><td>0.836</td><td>0.830</td></tr><tr><td>GPT-4+GPT-4 (on annotated data)</td><td>0.761</td><td>0.749</td><td>0.812</td><td>0.809</td></tr></table>

Table 16: F1 performance of Marco-level four interactivity aspects’ score prediction across different model versions