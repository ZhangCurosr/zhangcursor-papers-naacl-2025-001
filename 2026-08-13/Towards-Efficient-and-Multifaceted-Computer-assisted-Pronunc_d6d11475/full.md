# Towards Efficient and Multifaceted Computer-assisted Pronunciation Training Leveraging Hierarchical Selective State Space Model and Decoupled Cross-entropy Loss

Fu-An Chao, Berlin Chen Department of Computer Science and Information Engineering National Taiwan Normal University {fuann, berlin}@ntnu.edu.tw

## Abstract

Prior efforts in building computer-assisted pronunciation training (CAPT) systems often treat automatic pronunciation assessment (APA) and mispronunciation detection and diagnosis (MDD) as separate fronts: the former aims to provide multiple pronunciation aspect scores across diverse linguistic levels, while the latter focuses instead on pinpointing the precise phonetic pronunciation errors made by non-native language learners. However, it is generally expected that a full-fledged CAPT system should perform both functionalities simultaneously and efficiently. In response to this surging demand, we in this work first propose HMamba, a novel CAPT approach that seamlessly integrates APA and MDD tasks in parallel. In addition, we introduce a novel loss function, decoupled cross-entropy loss (deXent), specifically tailored for MDD to facilitate bettersupervised learning for detecting mispronounced phones, thereby enhancing overall performance. A comprehensive set of empirical results on the speechocean762 benchmark dataset demonstrates the effectiveness of our approach on APA. Notably, our proposed approach also yields a considerable improvement in MDD performance over a strong baseline, achieving an F1-score of 63.85%. Our codes are made available at https://github.com/Fuann/hma mba

## 1 Introduction

In this era of globalization and technologization, computer-assisted pronunciation training (CAPT) systems have emerged as an appealing alternative to meet the pressing need for second language (L2)

![](images/42f24d16198b5bd2563d6392c72b6a6097d226650c4b7c7ce05bcd2c3a14bc40.jpg)  
Figure 1: A running example depicts the evaluation differences between APA and MDD systems in the reading-aloud scenario.

learning. In comparison with traditional curriculum learning, CAPT offers advantages in both time efficiency and cost-effectiveness. More critically, it shifts the conventional pedagogical paradigm from teacher-directed to self-directed learning, thereby providing a stress-free environment for L2 learners (Eskenazi et al., 2009). In addition, CAPT applications have achieved marked success in various commercial sectors and testing services, such as the APPs of Duolingo (McCarthy et al., 2021) and the SpeechRater (Zechner et al., 2009) developed by Educational Testing Service (ETS). Typically, a de-facto archetype system for CAPT encompasses a “reading-aloud” scenario, where a non-native speaker is given a text prompt and instructed to pronounce it correctly. In this context, previous literature broadly divides applications of CAPT into two categories: automatic pronunciation assessment (APA) and mispronunciation detection and diagnosis (MDD), with each category dedicated to specific facets of pronunciation training. APA aims to evaluate the spoken proficiency of L2 learners by providing fine-grained feedback on various aspect assessments (e.g., accuracy and fluency) across multiple linguistic levels (e.g., word and utterance level) (Kheir et al., 2023). To assess L2 learners’ spoken proficiency, APA systems typically employ scoring models that are either jointly trained (Gong et al., 2022; Chao et al., 2022) or leverage multiple regressors in an ensemble paradigm (Bannò et al., 2022a; Bannò and Matassoni, 2022b) to generate scores across various aspects. As such, users can receive multi-aspect assessment scores predicted by an APA system, as illustrated in the example shown in Figure 1. In contrast to APA, MDD focuses more on non-native speakers’ phonetic pronunciation errors (Chen and Li, 2016). These errors usually have clear-cut distinctions between correct and incorrect ones, and can be easily quantified through deletions, substitutions, and insertions. For instance, a number of MDD models are designed to capitalize on classifier-based modeling (Truong et al., 2004; Strik et al., 2009; Harrison et al. 2009), enabling precise identification of the exact positions where pronunciation errors occur within an utterance. This capability provides L2 learners with specific feedback on discrepancies between intended pronunciation and actual pronunciation.

Albeit the phonetic (segmental) errors are crucial in the beginning stages of non-native language learning, prosodic (suprasegmental) errors may often cause a detrimental impact on the perception offluency and lead to poor intelligibility (Chen and Li, 2016). This effect may be more pronounced in learning stress-timed languages like English, especially for a learner whose mother tongue is a syllable-timed language, such as Chinese (Ding and Xu, 2016). To tackle this problem, APA can play a pivotal role by offering prosodic or intonation assessment for L2 learners. For example, Lin et al. (2021a) introduced rhythm rubrics to predict the traits of sentence-level stress in L2 English utterances, demonstrating a strong correlation with the prosody scores assessed by the human experts. In addition, Arias et al. (2010) proposed text-independent systems for assessing intonation and stress, focusing on measuring the similarity between a test-taker’s intonation or stress curve and that of a reference response.

On these grounds, it is evident that both APA and MDD are indispensable ingredients of CAPT, playing complementary roles in its success. However, previous studies on APA and MDD appear to have developed independently, with limited research exploring their synergetic use. Ryu et al. (2023) proposed a joint model for APA and MDD, leveraging knowledge transfer and multitask learning. Their findings indicate high negative correlations between several assessment scores and mispronunciations. This also suggests that the human assessors may be influenced by phonetic errors when evaluating overall proficiency scores for various aspects, which to some extent gives away the halo effect present in the human annotations. While the corresponding results show that jointly modeling both tasks can achieve better performance than modeling each task in isolation, only utterance-level holistic assessments are considered in their experiments. In order to provide more comprehensive and fine-grained feedback for L2 learners, other granularities, such as the phone level or the word level, should also be aptly modeled. Recognizing this importance, we propose HMamba, a more effective approach, for multifaceted CAPT. Being aware of the linguistic hierarchy, HMamba can capture the intrinsic multilayered speech structure, delivering both coarse and fine-grained pronunciation assessments while offering more accurate diagnostic feedback of mispronunciations. In addition, to address the extra computational costs introduced by multi-task learning, HMamba leverages a selective state space model (SSM) that can efficiently tackle both APA and MDD tasks in parallel. The main contributions of this paper can be summarized as follows:

• We introduce HMamba, a unified and linguistically hierarchy-aware model that jointly tackles APA and MDD tasks, achieving superior overall performance compared to prior arts that employ either single-task or multi-task models.

• We propose a novel loss function, decoupled cross-entropy loss (termed deXent), which effectively addresses the inherent issue of text prompt-aware MDD methods. Notably, deXent is feasible and well-suited for optimizing the MDD performance, particularly in striking the balance between precision and recall.

• To the best of our knowledge, this is the first work to adopt and extend Mamba in the APA and MDD tasks for a more efficient and comprehensive CAPT application.

![](images/bb08e45428526eb81f99fb15ccecb50e83505737fef6bc78ba99809ac8f41dc1.jpg)  
Figure 2: An overall architectural overview of HMamba, which consists of a bottom-up hierarchical modeling structure with several Mamba blocks across three levels (viz. phone, word, and utterance levels) that can perform multi-granular APA and MDD in parallel.

## 2 Methodology

## 2.1 Problem Definition

Considering an input time sequence of speech signal � uttered by an L2 learner and a reference text prompt � that contains � -length canonical phone sequence $\mathbf { p } = \{ p _ { 0 } , p _ { 1 } , \dots , p _ { N - 1 } \}$ , we adopt a set of feature extractors along with an aligner to extract an acoustic feature sequence $\mathbf { X } =$ $\left\{ \mathbf { x } _ { 0 } , \mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { N - 1 } \right\}$ that aligned with � from � . Our model aims to address APA and MDD tasks simultaneously but with separate processing flows: First, we define � as a set of linguistic granularities, and for each granularity $g \in G$ the model manages to predict a set of aspect scores $\mathbf { s } ^ { g } = \{ s _ { 0 } ^ { g } , s _ { 1 } ^ { g } , \ldots , s _ { M _ { q } - 1 } ^ { g } \}$ , where $M _ { g }$ refers to the number of aspect scores of target granularity $g .$ In this work, $G = \{ g ^ { p h n } , g ^ { w r d } , g ^ { u t t } \}$ , where we have granularities of $g ^ { p h n }$ (phone level), $g ^ { w r d }$ (word level), and $g ^ { u t t }$ (utterance level) for the APA task. Meanwhile, the model also requires to detect error states $\mathbf { e } = \{ e _ { 0 } , e _ { 1 } , \ldots , e _ { N - 1 } \}$ with respect to � and in turn generate the correct diagnostic output $\mathbf { y } = \{ y _ { 0 } , y _ { 1 } , \dots , y _ { N - 1 } \}$ , where $y _ { n }$ denotes the uttered phone of the learner corresponds to $p _ { n }$

## 2.2 HMamba

In this subsection, we shed light on the details of the proposed model, HMamba, which is devised as a hierarchical structure built upon the paradigm of selective SSM. A schematic illustration of the complete architecture is depicted in Figure 2. Specifically, HMamba synthesizes the APA and MDD modules, each of which contains multiple regressors and a classifier, respectively. These modules collectively generate the corresponding aspect score sequence $\mathbf { s } ^ { g }$ for each linguistic granularity $^ { g , }$ as well as the phonetic error states � and diagnosis �. Furthermore, each classifier and regressor is implemented with a simple feedforward network (FFN) and jointly optimized through the training.

Acoustic Feature Extraction: In order to portray the non-native speaker’s pronunciation quality, previous studies on either APA or MDD generally adopt a pre-trained acoustic model to extract goodness of pronunciation (GOP)-based features (Witt and Young, 2000; Hu et al., 2015; Shi et al., 2020). However, these features merely offer the segmental-level information that may not capture prosodic errors of an L2 learner. Given this limitation, we first utilize a pre-trained acoustic model as an aligner to identify phone boundaries (including silence), facilitating the extraction of other prosodic features such as the phone duration and statistics of root mean squared energy (Dong et al., 2024). To mitigate the low-resourced data problem (Chao et al., 2022), we also consider other self-supervised learning (SSL) features including wav2vec 2.0<sup>1</sup> (Baevski et al., 2020), HuBERT<sup>2</sup> (Hsu et al., 2021), and WavLM<sup>3</sup> (Chen et al., 2022). All these features are concatenated and subsequently projected through a linear layer to form a sequence of acoustic features � . The transformation of each time step � is given by

$$
\mathbf { a } _ { t } = [ \mathbf { a } _ { t } ^ { g o p } ; \mathbf { a } _ { t } ^ { d u r } ; \mathbf { a } _ { t } ^ { e n g } ; \mathbf { a } _ { t } ^ { w 2 v } ; \mathbf { a } _ { t } ^ { h b t } ; \mathbf { a } _ { t } ^ { w l m } ]\tag{1}
$$

$$
{ \mathbf { x } } _ { t } = { \mathbf { W } } { \mathbf { a } } _ { t } + { \mathbf { b } }\tag{2}
$$

where � and � are trainable parameters. Notably, a dropout rate of 10% is applied to all SSL features prior to the concatenation due to the discrepancy in dimensionality between these and other features.

Phonological Feature Extraction: In addition to acoustic cues, a common practice in CAPT is to inject the phonological information by introducing the reference text prompt features such as canonical phoneme embeddings (Gong et al., 2022), context-aware sup-phoneme embeddings (Chao et al., 2023), and vowel/consonant embeddings (Fu et al., 2021). In contrast to previous studies (Gong et al., 2022; Chao et al., 2022; Do et al., 2023a), we extract the canonical phoneme embeddings �<sup>�ℎ�</sup> from � using a phone embedding layer that includes the silence (SIL) information which has been shown to be crucial when evaluating ones’ spoken proficiency. Additionally, an absolute positional embedding �<sup>���</sup> and a relative position embedding ${ \bf { E } } ^ { r e l }$ are extracted. Distinct from ${ \bf E } ^ { a b s }$ ${ \bf { E } } ^ { r e l }$ denotes relative positions of phones in a word using tokens such as begin [B], internal [I], end [E], and single-phone word [S] tokens. For special cases of silence positions, we explicitly categorize them as either long silence [LS] or short silence [SS]. Following the guideline suggested by ETS (Evanini et al., 2015), positions with a silence duration exceeding 0.495 seconds are assigned to [LS]; otherwise, they are assigned to [SS]. Finally, all these embedding features are point-wise added to � to obtain phone-level input features for subsequent modeling:

$$
{ \bf H } ^ { g _ { 0 } ^ { p h n } } = { \bf X } + { \bf E } ^ { p h n } + { \bf E } ^ { a b s } + { \bf E } ^ { r e l }\tag{3}
$$

The details of the complete feature ablations (both acoustic and phonological features) are shown in Appendix B.

Mamba Blocks: To foster highly efficient multitask learning, we introduce selective SSMs instead of attention-based models such as the Transformer (Vaswani et al., 2017). Specifically, we adopt Mamba (Gu and Dao, 2023) as our backbone model structure in this work. Different from previous SSM instantiations, Mamba features an input-dependent selection mechanism and a hardware-aware algorithm, allowing for efficient input information filtering by dynamically adjusting the SSM parameters based on the input data. This also facilitate faster recurrent computation of the model using scan. Nevertheless, the vanilla Mamba conducts causal computations in a unidirectional manner, which prevents it from capturing global information as effectively as the multi-head self-attention (MHSA) module involved in Transformer. To address this problem, we explore a bidirectional variant of Mamba as the basic modeling block. In this approach, we replace the MHSA module in the Transformer encoder with a bidirectional Mamba layer, as depicted in Figure 2. Specifically, for input �<sup>�</sup>� to the Mamba block at granularity level $^ { g , }$ the output �<sup>�</sup>�+1 of the block is:

$$
\mathbf { H } ^ { \prime g _ { i } } = \mathrm { B i M a m b a } ( \mathrm { L a y e r N o r m } ( \mathbf { H } ^ { g _ { i } } ) ) + \mathbf { H } ^ { g _ { i } } ( 4 )
$$

$$
\mathbf H ^ { g _ { i + 1 } } = \mathrm { F F N } ( \mathrm { L a y e r N o r m } ( \mathbf H ^ { \prime g _ { i } } ) ) + \mathbf H ^ { \prime g _ { i } }\tag{5}
$$

where BiMamba denotes the bidirectional Mamba layer and FFN refers to the feed-forward module, respectively. Notably, there are several studies investigating the bidirectional processing of Mamba (Liang et al., 2024; Zhang et al., 2024; Jiang et al., 2024). In this work, we use a similar structure as Jiang et al. (2024) to implement the bidirectional Mamba layer. For input �<sup>�</sup>� from the output of layer normalization of $\mathbf { H } ^ { g _ { i } }$ to a bidirectional Mamba layer, the corresponding output �<sup>�</sup>� is computed as follows:

$$
\mathbf { Z } ^ { g _ { i } } = \mathrm { L i n e a r } ( \mathbf { N } ^ { g _ { i } } )\tag{6}
$$

$$
\mathbf { S } ^ { g _ { i }  } = \mathrm { L i n e a r } ( \mathbf { N } ^ { g _ { i } } ) , ~ \mathbf { S } ^ { g _ { i }  } = \mathrm { F l i p } ( \mathbf { S } ^ { g _ { i }  } )\tag{7}
$$

$$
\{ \begin{array} { l l } { \mathbf { C } ^ { g _ { i } \to } = \mathrm { C o n v 1 D } ^ { \to } ( \mathbf { S } ^ { g _ { i } \to } ) } \\ { \mathbf { C } ^ { g _ { i }  } = \mathrm { C o n v 1 D } ^ {  } ( \mathbf { S } ^ { g _ { i }  } ) } \end{array}\tag{8}
$$

$$
\{ \begin{array} { l l } { \mathbf { O } ^ { g _ { i } \to } = \sigma ( \mathbf { Z } ^ { g _ { i } } ) \otimes \mathrm { S S M } ^ { \to } ( \mathbf { C } ^ { g _ { i } \to } ) } \\ { \mathbf { O } ^ { g _ { i }  } = \sigma ( \mathbf { Z } ^ { g _ { i } } ) \otimes \mathrm { S S M } ^ {  } ( \mathbf { C } ^ { g _ { i }  } ) } \end{array}\tag{9}
$$

$$
\mathbf { M } ^ { g _ { i } } = \mathrm { L i n e a r } ( \frac { 1 } { 2 } \mathbf { O } ^ { g _ { i } \right. } + \frac { 1 } { 2 } \mathrm { F l i p } ( \mathbf { O } ^ { g _ { i } \left. } ) )\tag{10}
$$

where ${ \bf S } ^ { g _ { i } \to }$ and $\mathbf { S } ^ { g _ { i }  }$ denote the forward and backward sequence features, respectively. Specifically, $\mathbf { S } ^ { g _ { i }  }$ is derived from ${ \bf S } ^ { g _ { i } \to }$ by a flipping operation Flip(∙). Conv1D(∙), �(∙), and SSM(∙) represents the 1-D convolution, activation function, and selective SSM algorithm described in Mamba (Gu and Dao, 2023), respectively.

Hierarchical Mamba: Since the speech signals are typically distinguished by the complex hierarchical composition, prior studies (Do et al., 2023a; Chao et al., 2023) have suggested that hierarchical modeling structures is more amenable than parallel modeling structures (Gong et al., 2022). To capture the linguistic hierarchy while retaining the cross-aspect relations within the same linguistic unit, we design and instantiate our model with a hierarchical structure and introduce Mamba blocks to model the dependencies at each granularity level. More concretely, our approach generates finer granularity scores at the lower layers and coarser granularity scores at the higher layers, as exhibited in Figure 2. In phone-level modeling, we first use $\mathbf { H } ^ { g _ { 0 } ^ { p h n } }$ as the input into $L _ { p } -$ layer Mamba blocks to obtain the phone-level contextualized representations $\mathbf { H } ^ { g _ { L _ { p } } ^ { p h n } }$

$$
\mathbf { H } ^ { g _ { L _ { p } } ^ { p h n } } = \mathrm { M a m b a B l o c k } _ { p h n } ( \mathbf { H } ^ { g _ { 0 } ^ { p h n } } )\tag{11}
$$

Subsequently, $\mathbf { H } ^ { g _ { L _ { p } } ^ { p h n } }$ are then propagated forward into the APA module and the MDD module for solving a regression and a sequence classification problem, respectively. The APA module contains one regressor that aims to predict the phone-level aspect score $s _ { 0 } ^ { g ^ { p h n } }$ (accuracy). On the other hand, the MDD module comprises a classifier and a softmax function that cooperatively learn a distribution $\hat { y } _ { t }$ over the phoneme classes $C$ for each time step � . The diagnosis $y _ { t }$ can then be identified by applying the argmax function to $\hat { y } _ { t }$ . In this work, we streamline the MDD task by treating it as a process of free phone recognition (Li et al., 2015). As such, we can directly detect the corresponding error state $e _ { t }$ by comparing $y _ { t }$ with $p _ { t }$ , eliminating the need for a separate detection module. Meanwhile, the resulting $\mathbf { H } ^ { g _ { L _ { p } } ^ { p h n } }$ is served as $\mathbf { H } ^ { g _ { 0 } ^ { w r d } }$ for subsequent modeling.

In word-level modeling, $L _ { w }$ -layer Mamba blocks are first adopted and followed by a 1-D convolution layer to capture the local dependencies (Lee, 2016). The reason for utilizing the convolution layer is that the convolution operation can accommodate different realizations of the same underlying phone from various L2 speakers, thereby mitigating the temporal variability. The word-level representations $\mathbf { H } ^ { g _ { L _ { w } } ^ { w r d } }$ can be derived as follows:

$$
\mathbf { H } ^ { \prime g _ { L _ { w } } ^ { w r d } } = \mathrm { M a m b a B l o c k } _ { w r d } ( \mathbf { H } ^ { g _ { 0 } ^ { w r d } } )\tag{12}
$$

$$
\mathbf { H } ^ { g _ { L _ { w } } ^ { w r d } } = \mathrm { C o n v 1 D } _ { w r d } ( \mathbf { H } ^ { \prime g _ { L _ { w } } ^ { w r d } } )\tag{13}
$$

To obtain word-level aspect scores, we put $\mathbf { H } ^ { g _ { L _ { w } } ^ { w r d } }$ into the word-level APA module which contains three regressors to predict the word-level aspect scores $s _ { 0 } ^ { g ^ { \bar { w } r d } } , s _ { 1 } ^ { g ^ { w r d } } , \bar { s } _ { 2 } ^ { g ^ { w r d } }$ (accuracy, stress, and total scores), respectively. To facilitate training efficiency, we propagate the word score to each of its phones during the training stage. In the inference phase, we ensure consistency by averaging the outputs corresponding to each word. In addition, $\mathbf { H } ^ { g _ { L _ { w } } ^ { w r d } }$ is viewed as $\mathbf { H } ^ { g _ { 0 } ^ { u t t } }$ for further modeling.

As for the utterance-level assessments, instead of prepending the [CLS] tokens to learn the utterance-level representation (Gong et al., 2022), we explore pooling-based approaches to aggregate the hidden information. To this end, we utilize an attention pooling layer similar to Peng et al. (2022). Specifically, assuming that a �-dimensional input sequence to the attention pooling layer is $ { \mathbf { h } } _ { 0 } ,  { \mathbf { h } } _ { 1 } \dots ,  { \mathbf { h } } _ { T - 1 }$ , the pooling output is $\mathbf { h } =$ $\textstyle \sum _ { i = 0 } ^ { T - 1 } \alpha _ { i } \mathbf { h } _ { i } ,$ where $\alpha _ { i }$ is calculated by

$$
\alpha _ { i } = \frac { \exp { ( \mathbf { w } ^ { T } \mathbf { q } _ { i } / \tau ) } } { \sum _ { j = 0 } ^ { T - 1 } \exp { ( \mathbf { w } ^ { T } \mathbf { q } _ { j } / \tau ) } }\tag{14}
$$

where � is a learnable vector, � is the concatenated scores of $\bigl [ s _ { 0 } ^ { g ^ { p h n } } , s _ { 0 } ^ { g ^ { w r d } } , s _ { 1 } ^ { g ^ { w r d } } , s _ { 2 } ^ { g ^ { w r d } } \bigr ] .$ and $\tau$ is a controllable temperature hyperparameter. The whole process of utterance-level modeling can then be formulated by

$$
\mathbf { H } ^ { g _ { L _ { u } } ^ { u t t } } = \mathrm { M a m b a B l o c k } _ { u t t } ( \mathbf { H } ^ { g _ { 0 } ^ { u t t } } )\tag{15}
$$

$$
\mathbf { h } ^ { g ^ { u t t } } = \mathrm { A t t e n t i o n P o o l i n g } _ { u t t } ( \mathbf { H } ^ { g _ { L _ { u } } ^ { u t t } } )\tag{16}
$$

After obtaining $\mathbf { H } ^ { g _ { L _ { u } } ^ { u t t } }$ from $L _ { u }$ -layer Mamba blocks, $\mathbf { h } ^ { g ^ { u t t } }$ is derived through the attention pooling layer to predict the utterance-level aspect scores $s _ { 0 } ^ { g ^ { u t t } } , s _ { 1 } ^ { g ^ { u t t } } , s _ { 2 } ^ { g ^ { u t t } } , s _ { 3 } ^ { g ^ { u t t } } , s _ { 4 } ^ { g ^ { u t t } }$ (accuracy, completeness, fluency, prosody, and total scores) via an utterance-level APA module which contains five regressors corresponding to each score.

## 2.3 Optimization

Automatic Pronunciation Assessment Loss: In the proposed model, each APA module is optimized using Mean Square Error (MSE). The loss for multi-aspect multi-granular assessment, $\mathcal { L } _ { A P A }$ , is calculated by assigning weights to each granularity level $g \colon$

$$
\mathcal { L } _ { A P A } = \sum _ { g \in G } \boldsymbol { \omega } _ { g } \cdot \frac { 1 } { N _ { g } } \sum _ { k = 0 } ^ { N _ { g } - 1 } \cdot \mathcal { L } _ { g _ { k } }\tag{17}
$$

where $\omega _ { g }$ and $N _ { g }$ are the tunable hyperparameter and number of aspect scores at granularity level $^ { g , }$ respectively. $\mathcal { L } _ { g _ { k } }$ refers to the MSE loss computed for �-th aspect score at granularity level $g .$

Mispronunciation Detection and Diagnosis Loss: To be in line with previous MDD studies, our model incorporates canonical phoneme embeddings to enhance text prompt-awareness. Despite some performance improvements, the mismatch between the L2 learner’s realized phones and canonical phones can still cause some deteriorating effects. This discrepancy can introduce inaccurate predictions that may potentially affect the overall quality of phonetic analysis. To mitigate this negative impact, we devise a new loss function tailored for the MDD task, as illustrated in Figure 3. Specifically, we first decouple the original cross-entropy loss into two separate losses, one for mispronunciations and the other for correct pronunciations:

![](images/8a92bde96d801e75ef02fb1bb429b63cbf805c64757d5c93fc8229ade0896952.jpg)  
Figure 3: Difference between (a) the original crossentropy loss and (b) the decoupled cross-entropy loss, given the text prompt “crime.”

$$
\mathcal { L } _ { X e n t } ^ { m i s } = - \sum _ { t \in \mathcal { M } } \log ( \hat { y } _ { t } [ y _ { t } ] )\tag{18}
$$

$$
\mathcal { L } _ { X e n t } ^ { h i t } = - \sum _ { t \in \mathcal { H } } \log ( \hat { y } _ { t } [ y _ { t } ] )\tag{19}
$$

where ℳ and ℋ are mispronunciation and correct pronunciation positions, respectively, and $\hat { y } _ { t } [ y _ { t } ]$ is the predicted probability of the true label $y _ { t }$ at time step �. After obtaining two decoupled losses, the proposed decoupled cross-entropy loss (deXent) is obtained by the following formulation:

$$
\mathcal { L } _ { \_ M D D } = \mathcal { L } _ { X e n t } ^ { h i t } + ( \frac { \mu ^ { h } } { \mu ^ { m } } ) ^ { \alpha } \ \mathcal { L } _ { X e n t } ^ { m i s }\tag{20}
$$

where $\mu ^ { m }$ and $\mu ^ { h }$ denote the frequency of the mispronunciations and correct pronunciations in the training set, respectively, and $\alpha$ controls the weight magnitude. After that, we use $\mathcal { L } _ { M D D }$ to optimize the MDD module, and the overall loss thus can be expressed by

$$
\mathcal { L } = \mathcal { L } _ { A P A } + \beta \cdot \mathcal { L } _ { M D D }\tag{21}
$$

where $\beta$ is a tunable hyperparameter.

According to Equation 20, the proposed loss function can be viewed as one of the loss-balancing methods, such as focal loss (Lin et al., 2017) and class-balanced loss (Cui et al., 2019), to tackle the imbalance issue in MDD. However, in most endto-end MDD methods, where the labels are phones instead of mispronunciations (0 or 1s), directly applying the existing loss-balancing methods on phones is implicit and can be sub-optimal when we aim to detect potential mispronunciations. Hence, we believe the proposed deXent provides a better alternative to end-to-end MDD.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Year</td><td colspan="2">Phone Score</td><td colspan="2">Word Score (PCC)</td><td colspan="4">Utterance Score (PCC)</td></tr><tr><td>MSE↓</td><td>PCC↑</td><td>Accuracy↑ Stress↑</td><td>Total↑ Accuracy↑</td><td>Completeness↑ Fluency↑ Prosody↑ Total↑</td><td></td><td></td><td></td></tr><tr><td>Deep Feature</td><td>2021</td><td></td><td></td><td></td><td>1</td><td></td><td></td><td></td><td>0.720</td></tr><tr><td>HuBERT Large</td><td>2022</td><td></td><td></td><td></td><td></td><td></td><td>0.780</td><td>0.770</td><td></td></tr><tr><td>Joint-CAPT-L1</td><td>2023</td><td>一</td><td>1</td><td></td><td>=</td><td>0.719</td><td>0.775</td><td>0.773</td><td>0.743</td></tr><tr><td>LSTM</td><td>2022</td><td>0.089</td><td>0.591</td><td>0.514</td><td>0.294 0.531</td><td>0.720 0.076</td><td>0.745</td><td>0.747</td><td>0.741</td></tr><tr><td>GOPT</td><td>2022</td><td>0.085</td><td>0.612</td><td>0.533</td><td>0.291 0.549</td><td>0.714 0.155</td><td>0.753</td><td>0.760</td><td>0.742</td></tr><tr><td>3M</td><td>2022</td><td>0.078</td><td>0.656</td><td>0.598</td><td>0.289 0.617</td><td>0.760 0.325</td><td>0.828</td><td>0.827</td><td>0.796</td></tr><tr><td>HiPAMA</td><td>2023</td><td>0.084</td><td>0.616</td><td>0.575</td><td>0.320 0.591</td><td>0.730 0.276</td><td>0.749</td><td>0.751</td><td>0.754</td></tr><tr><td>3MH</td><td>2023</td><td>0.071</td><td>0.693</td><td>0.682</td><td>0.361 0.694</td><td>0.782 0.374</td><td>0.843</td><td>0.836</td><td>0.811</td></tr><tr><td>HMamba</td><td>2024</td><td>0.062</td><td>0.739</td><td>0.708</td><td>0.366 0.718</td><td>0.807 0.278</td><td>0.848</td><td>0.843</td><td>0.829</td></tr></table>

Table 1: APA performance evaluations of our model and all strong baselines on the speechocean762 test set.

## 3 Experimental Setup

## 3.1 Dataset

We conducted experiments on speechocean762, a widely-used open-source dataset curated for APA and MDD research (Zhang et al., 2021). The dataset consists of 5,000 English-speaking recordings from 250 Mandarin L2 learners, divided evenly into training and test sets. For the APA task, pronunciation proficiency scores were assessed at various linguistic granularities and across different pronunciation aspects. Each score is evaluated by five experts using standardized rubrics. For the MDD task, the dataset provides an extra mispronunciation transcription annotated using a set of 46 phones. This set comprises 39 phones from the CMU dictionary<sup>4</sup>, 6 L2-specific phones, and a [unk] token for unknown phones. Notably, there are no insertion errors in the utterances, and a [DEL] token is introduced to mark deletion errors of L2 learners. Therefore, the realized phones can be aligned with canonical phones in this dataset.

## 3.2 Training and Evaluation Details

Training: We optimized the model with Adam and a tri-phase rate scheduler (Baevski et al., 2020), where the learning rate was gradually increased during the first 40% of steps, held constant for the following 40%, and then linearly decayed for the remaining steps. The initial learning rate was set to 2e-3 except for the utterance-level APA module, which was set to 9e-5. Other implementation details are presented in Appendix A.

Evaluation: The evaluation metrics employed include the Pearson Correlation Coefficient (PCC) and Mean Square Error (MSE) for the APA task. On the other hand, we used precision, recall, F1- score, and phone error rate (PER) to evaluate the MDD performance, so as to be in accordance with prior studies. To ensure the validity of our experimental results, we conducted 5 independent trials for each experiment, running 20 epochs with different seeds. The metrics for each task are reported as the average of these trials.

## 3.3 Compared Baselines

For APA, we compare our proposed approach, HMamba, with various cutting-edge baselines which can be categorized into two families: singleaspect pronunciation assessment models or multigranular multi-aspect pronunciation assessment models. The first group includes the Deep Feature (Lin et al., 2021b), HuBERT Large (Kim et al., 2022), and Joint-CAPT-L1 (Ryu et al., 2023). The second group encompasses LSTM, GOPT (Gong et al., 2022), 3M (Chao et al., 2022), HiPAMA (Do et al., 2023a), and 3MH (Chao et al., 2023). As for MDD, we compare HMamba with Joint-CAPT-L1, as to our knowledge it is the only attempt that jointly addresses the APA and MDD tasks with the speechocean762 dataset.

<table><tr><td rowspan="2">Model</td><td colspan="2">Phone Score</td><td colspan="3">Word Score (PCC)</td><td colspan="5">Utterance Score (PCC)</td></tr><tr><td>MSE↓</td><td>PCC↑</td><td>Accuracy↑</td><td>Stress↑</td><td>Total↑</td><td>Accuracy↑</td><td>Completeness↑</td><td>Fluency↑</td><td>Prosody↑</td><td>Total↑</td></tr><tr><td>LMamba</td><td>0.071</td><td>0.694</td><td>0.678</td><td>0.299</td><td>0.689</td><td>0.790</td><td>0.234</td><td>0.844</td><td>0.838</td><td>0.816</td></tr><tr><td>PMamba</td><td>0.068</td><td>0.707</td><td>0.689</td><td>0.320</td><td>0.700</td><td>0.784</td><td>0.142</td><td>0.843</td><td>0.832</td><td>0.817</td></tr><tr><td>HMamba</td><td>0.062</td><td>0.739</td><td>0.708</td><td>0.366</td><td>0.718</td><td>0.807</td><td>0.278</td><td>0.848</td><td>0.843</td><td>0.829</td></tr></table>

Table 2: Performance comparison between different modeling structures.

<table><tr><td rowspan="2">Model</td><td colspan="3">Mispronunciations</td><td rowspan="2">PER ↓</td></tr><tr><td>Precision ↑ Recall ↑</td><td></td><td>F1 ↑</td></tr><tr><td>Joint-CAPT-L1</td><td>26.70%</td><td>91.40%41.50%</td><td></td><td>9.93%</td></tr><tr><td>HMamba</td><td>64.35%</td><td>63.41%63.85% 2.72%</td><td></td><td></td></tr></table>

Table 3: MDD performance evaluations of our model, compared with a representative multi-task approach (Ryu et al., 2023) on the speechocean762 test set.

## 4 Experimental Results and Discussion

## 4.1 APA Performance

Overall Results: In Table 1, we compare the APA performance of HMamba with other competitive baselines, leading to several key observations. Firstly, it is notable that our approach, HMamba, consistently outperforms all other methods on nearly all assessment tasks, particularly in terms of accuracy scores at phone, word, and utterance levels. This improvement stems from the joint modeling paradigm of APA and MDD, highlighting that pronunciation assessments can also benefit from phonetic error discovery, consistent with prior research findings (Ryu et al., 2023). In addition, by adopting SSL features, HMamba as well as other approaches like HuBERT Large, 3M, and 3MH, achieves significant improvements over the other APA methods in terms of utterance-level assessments. In comparison to other hierarchical models such as HiPAMA and 3MH, HMamba leverages an SSM structure instead of the Transformer structure, demonstrating superior performance on a variety of assessment tasks (further analysis between Mamba and Transformer are shown in Appendix C). In assessing utterance completeness, while HMamba falls behind 3M and 3MH, it is on par with HiPAMA. According to Zhang et al. (2021), the completeness refers to the percentage of the words that are actually pronounced. This may imply that our approach has a weaker ability to detect incompletely articulated words. However, this limitation could stem from the fact that HMamba focuses more on phoneme accuracy and mispronunciation detection, rather than purely evaluating word-level completeness.

<table><tr><td rowspan="2">Loss</td><td rowspan="2">α</td><td colspan="3">Mispronunciations</td><td rowspan="2">PER ↓</td></tr><tr><td>Precision ↑ Recall ↑</td><td></td><td>F1 ↑</td></tr><tr><td>Xent</td><td></td><td>77.07%</td><td>38.60%</td><td>51.40%</td><td>2.53%</td></tr><tr><td rowspan="4">deXent</td><td>0.3</td><td>70.06%</td><td>54.10%</td><td>61.04%</td><td>2.61%</td></tr><tr><td>0.5</td><td>67.12%</td><td>58.71%</td><td>62.62%</td><td>2.70%</td></tr><tr><td>0.7</td><td>64.35%</td><td>63.41%</td><td>63.85%</td><td>2.72%</td></tr><tr><td>0.9</td><td>57.74%</td><td>71.12%</td><td>63.73%</td><td>3.14%</td></tr></table>

Table 4: Comparison of MDD performance between the original cross-entropy loss (Xent) and proposed decoupled cross-entropy loss (deXent).

Effects of Hierarchical Structure: To inspect the hierarchical structure that influences on the APA performance of the proposed approach, we conducted an experiment to replace the hierarchical structure with two other variants, resulting in two different models: LMamba and PMamba, respectively. LMamba has a similar structure to HMamba but outputs all assessment scores in the last layers regardless of their granular differences. On the other hand, PMamba adopts the parallel structure suggested by previous studies (Gong et al., 2022; Chao et al., 2022) that use prepended [CLS] tokens to predict utterance-level scores. According to the results shown in Table 2, HMamba outperforms PMamba and LMamba across all assessment aspects, highlighting the advantages of its hierarchical structure for the APA task. This finding aligns with previous research (Chao et al., 2023). In addition, the significant performance gaps between HMamba and LMamba also suggest that phone-level and word-level scores should be predicted in lower layers.

## 4.2 MDD Performance

Overall Results: We evaluate the MDD performance of HMamba by comparing it with another celebrated multi-task learning approach, Joint-CAPT-L1. As shown in Table 3, despite of lower recall rate, HMamba achieves a significant improvement in terms of F1-score over Joint-CAPT-L1, with an increase of 22.35%. Additionally, there is a marked reduction in PER by 7.21%. These substantial enhancements demonstrate that HMamba not only delivers accurate pronunciation assessments but also produces more robust and reliable mispronunciation detection and diagnosis results.

Effects of Decoupled Cross-entropy Loss: On the grounds of the distinct improvements in the MDD performance, we further analyze the underlying effects of our proposed decoupled cross-entropy loss (deXent). As illustrated in Table 4, training a text prompt-aware MDD model using the original cross-entropy loss (Xent) often yields high precision but low recall. This is because the model primarily relies on input canonical phones, leading it to predict prior phones and overlook the actual mispronunciations of a learner. Such a model may not be suitable for educational settings where accurately detecting potential mispronunciations is critical. To remedy this, the proposed deXent method sufficiently provides a feasible solution. By adjusting the weighting factor �, we can better strike the balance between precision and recall, thus optimizing the F1-score. This flexibility is particularly vital in CAPT, where both overdetection and under-detection of mispronunciations can severely disrupt the learning process—a challenge often neglected by most existing end-to-end methods. While adopting deXent may result in a minor increase in PER, this slight performance tradeoff is justifiable for the significant gains in overall MDD effectiveness.

Limitations of Decoupled Cross-entropy Loss: In Table 3, the MDD performance of HMamba is reported based on maximizing the F1 score using the deXent, as we believe both the precision and recall metrics are critical for MDD. However, a potential limitation of using deXent as the loss function is that, while it may help balance precision and recall for MDD, it may not simultaneously improve both metrics. This limitation likely stems from the close relationship between the mispronunciation distribution and the lossbalancing mechanism of deXent.

## 5 Conclusion

In this paper, we have presented a novel hierarchical selective state space model (dubbed HMamba) for multifaceted CAPT application. Extensive experimental results substantiate the viability and efficacy of the proposed method compared to several top-of-the-line approaches in terms of both the APA and MDD performance. In future work, we envisage mitigating the issue of data imbalance from an optimization perspective. In addition, another key area for future research involves tackling the assessment of open-response scenarios in CAPT.

## Limitations

Lack of Accent Diversity. The dataset used in this study comprises only Mandarin L2 learners, which would probably limit the generalizability of the proposed model. As a result, it might be inapplicable when assessing L2 learners with diverse accents. This lack of accent diversity could lead to biases and inaccuracies in pronunciation assessment for learners from different linguistic backgrounds.

Limited Interpretability. The proposed model is designed to replicate expert annotations without relying on manual assessment rubrics or external knowledge databases, which would make it challenging to provide clear and reasonable explanations for the assessment results. This insufficiency of interpretability might hinder its acceptance and trustworthiness among educators and learners who require transparent and justifiable assessments.

Limited Generalizability This research is centered on the “reading-aloud” pronunciation training scenario, where it is assumed that the L2 learner accurately pronounces a predetermined text prompt. This would narrow the applicability of our models to other learning contexts, such as spontaneous speech or open-ended conversations.

## Ethics Statement

We acknowledge that all co-authors of this work comply with the ACL Code of Ethics and adhere to the code of conduct. Our experimental corpus, speechocean762, is widely used and publicly available, and we believe there are no potential risks associated with this work.

## Acknowledgments

This work was supported by the Language Training and Testing Center (LTTC), Taiwan. Any findings and implications in the paper do not necessarily reflect those of the sponsor.

## References

Juan Pablo Arias, Nestor Becerra Yoma, and Hiram Vivanco. 2010. Automatic intonation assessment for computer aided language learning. Speech Communication, volume 52, pages 254–267.

Stefano Bannò, Bhanu Balusu, Mark Gales, Kate Knill, and Konstantinos Kyriakopoulos. 2022a. Viewspecific assessment of L2 spoken English. In Proceedings of the Annual Conference of the International Speech Communication Association (INTERSPEECH), pages 4471–4475.

Stefano Bannò and Marco Matassoni. 2022b. Proficiency assessment of L2 spoken English using wav2vec 2.0. In Proceedings of IEEE Spoken Language Technology Workshop (SLT), pages 1088- 1095.

Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli. 2020. wav2vec 2.0: A framework for self-supervised learning of speech representations. In Proceedings of the Conference on Neural Information Processing Systems (NeurIPS), pages 12449–12460.

Fu An Chao, Tien Hong Lo, Tzu I. Wu, Yao Ting Sung, Berlin Chen. 2022. 3M: An effective multi-view, multigranularity, and multi-aspect modeling approach to English pronunciation assessment. In Proceedings of the Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC), pages 575–582.

Fu-An Chao, Tien-Hong Lo, Tzu-I Wu, Yao-Ting Sung, Berlin Chen. 2023. A hierarchical context-aware modeling approach for multi-aspect and multigranular pronunciation assessment. In Proceedings of the Annual Conference of the International Speech Communication Association (INTERSPEECH), pages 974–978.

Nancy F. Chen, and Haizhou Li. 2016. Computerassisted pronunciation training: From pronunciation scoring towards spoken language learning. In Proceedings of the Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC), pages 1–7.

Sanyuan Chen, Chengyi Wang, Zhengyang Chen, Yu Wu, Shujie Liu, Zhuo Chen, Jinyu Li, Naoyuki Kanda, Takuya Yoshioka, Xiong Xiao, Jian Wu, Long Zhou, Shuo Ren, Yanmin Qian, Yao Qian, Jian Wu, Michael Zeng, Xiangzhan Yu, and Furu Wei. 2022. Wavlm: Large-scale self-supervised pretraining for full stack speech processing. IEEE Journal of Selected Topics in Signal Processing, volume 16, number 6, pages 1505-1518.

Yin Cui, Menglin Jia, Tsung-Yi Lin, Yang Song, Serge Belongie. 2019. Class-balanced loss based on effective number of samples. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9268-9277.

Heejin Do, Yunsu Kim, and Gary Geunbae Lee. 2023a. Hierarchical pronunciation assessment with multiaspect attention. In Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5.

Heejin Do, Yunsu Kim, and Gary Geunbae Lee. 2023b. Score-balanced loss for multi-aspect pronunciation assessment. In Proceedings of the Annual Conference of the International Speech Communication Association (INTERSPEECH), pages 4998–5002.

Hongwei Ding, Xinping Xu. 2016. L2 English rhythm in read speech by Chinese students. In Proceedings of the Annual Conference of the International Speech Communication Association (INTERSPEECH), pages 2696-2700.

Bin Dong, Qingwei Zhao, Jianping Zhang, and Yonghong Yan. 2004. Automatic assessment of pronunciation quality. In Proceedings of IEEE International Symposium on Chinese Spoken Language Processing (ISCSLP), pages 137-140.

Maxine Eskenazi. 2009. An overview of spoken language technology for education. Speech Communication, volume 51, pages 832–844.

Keelan Evanini, Michael Heilman, Xinhao Wang, and Daniel Blanchard. 2015. Automated scoring for the TOEFL Junior® comprehensive writing and speaking test. ETS Research Report Series 2015(1):1–11.

Kaiqi Fu, Jones Lin, Dengfeng Ke, Yanlu Xie, Jinsong Zhang, and Binghuai Lin. 2021. A full textdependent end to end mispronunciation detection and diagnosis with easy data augmentation techniques. arXiv preprint arXiv:2104.08428.

Yuan Gong, Ziyi Chen, Iek-Heng Chu, Peng Chang, and James Glass. 2022. Transformer-based multiaspect multigranularity non-native English speaker pronunciation assessment. In Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 7262–7266.

Albert Gu and Tri Dao. 2023. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752.

Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, and Abdelrahman Mohamed. 2021. Hubert: Selfsupervised speech representation learning by masked prediction of hidden units. IEEE/ACM Transactions on Audio, Speech, and Language Processing, volume 29, pages 3451–3460.

Alissa M. Harrison, Wai-Kit Lo, Xiao-Jun Qian, and Helen Meng. 2009. Implementation of an extended recognition network for mispronunciation detection and diagnosis in computer-assisted pronunciation training. In Proceedings of the Workshop on Speech and Language Technology in Education (SLaTE), pages 45-48.

Wenping Hu, Yao Qian, Frank K. Soong, and Yong Wang. 2015. Improved mispronunciation detection with deep neural network trained acoustic models and transfer learning based logistic regression classifiers. Speech Communication, volume 67, pages 154–166.

Xilin Jiang, Cong Han, and Nima Mesgarani. Dualpath mamba: Short and long-term bidirectional selective structured state space models for speech separation. 2024. arXiv preprint arXiv:2403.18257.

Yassine Kheir, Ahmed Ali, and Shammur Chowdhury. 2023. Automatic pronunciation assessment - a review. In Findings of the Association for Computational Linguistics: EMNLP, pages 8304– 8324.

Eesung Kim, Jae-Jin Jeon, Hyeji Seo, Hoon Kim. 2022. Automatic pronunciation assessment using selfsupervised speech representation learning. In Proceedings of the Annual Conference of the International Speech Communication Association (INTERSPEECH), pages 1411–1415.

Ann Lee. 2016. Language-independent methods for computer-assisted pronunciation training, Ph.D. thesis, Massachusetts Institute of Technology.

Aobo Liang, Xingguo Jiang, Yan Sun, Xiaohou Shi, and Ke Li. 2024. Bi-Mamba4TS: Bidirectional mamba for time series forecasting. arXiv preprint arXiv:2404.15772.

Binghuai Lin, Liyuan Wang, Hongwei Ding, Xiaoli Feng. 2021a. Improving L2 English rhythm evaluation with automatic sentence stress detection. In Proceedings of IEEE Spoken Language Technology Workshop (SLT), pages 713-719.

Binghuai Lin and Liyuan Wang. 2021b. Deep feature transfer learning for automatic pronunciation assessment. In Proceedings of the Annual Conference of the International Speech Communication Association (INTERSPEECH), pages 4438–4442.

Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár. 2017. Focal Loss for Dense Object Detection. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pages 2980-2988.

Kun Li, Xiaojun Qian, Shiying Kang, Pengfei Liu, and Helen Meng. 2015. Integrating acoustic and statetransition models for free phone recognition in L2 English speech using multi-distribution deep neural networks. In Proceedings of the Workshop on Speech and Language Technology in Education (SLaTE), pages. 119-124.

Arya D. McCarthy, Kevin P. Yancey, Geoffrey T. LaFlair, Jesse Egbert, Manqian Liao, and Burr Settles. 2021. Jump-starting item parameters for adaptive language tests. In Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 883–899.

Yifan Peng, Siddharth Dalmia, Ian Lane, and Shinji Watanabe. 2022. Branchformer: Parallel mlpattention architectures to capture local and global context for speech recognition and understanding. In Proceedings of the International Conference on Machine Learning (PMLR), pages 17627–17643.

Hyungshin Ryu, Sunhee Kim, and Minhwa Chung. 2023. A joint model for pronunciation assessment and mispronunciation detection and diagnosis with multi-task learning. In Proceedings of the Annual Conference of the International Speech Communication Association (INTERSPEECH), pages 959-963.

Jiatong Shi, Nan Huo, and Qin Jin. 2020. Contextaware goodness of pronunciation for computerassisted pronunciation training. In Proceedings of the Annual Conference of the International Speech

Communication Association (INTERSPEECH), pages 3057-3061.

Helmer Strik, Khiet Truong, Febe De Wet, and Catia Cucchiarini. 2009. Comparing different approaches for automatic pronunciation error detection. Speech Communication, volume 51, number 10, pages 845- 852.

Khiet Truong, Ambra Neri, Catia Cucchiarini, and Helmer Strik. 2004. Automatic pronunciation error detection: an acoustic-phonetic approach. InSTIL/ICALL Symposium.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Proceedings of the Conference on Neural Information Processing Systems (NeurIPS), pages 5998–6008.

Anjana S. Vakil and Jürgen Trouvain. 2015. “Automatic classification of lexical stress errors for German CAPT,” in Proceedings of the Workshop on Speech and Language Technology in Education (SLaTE), pages 47– 52.

Silke M. Witt and Steve J. Young. 2000. Phone-level pronunciation scoring and assessment for interactive language learning. Speech Communication, volume 30, pages 95–108.

Klaus Zechner, Derrick Higgins, Xiaoming Xi, and David M. Williamson. 2009. Automatic scoring of non-native spontaneous speech in tests of spoken English. Speech Communication, volume 51, number 10, pages 883-895.

Lianghui Zhu, Bencheng Liao, Qian Zhang, Xinlong Wang, Wenyu Liu, and Xinggang Wang. 2024. Vision mamba: Efficient visual representation learning with bidirectional state space model. arXiv preprint arXiv:2401.09417.

Xiangyu Zhang, Qiquan Zhang, Hexin Liu, Tianyi Xiao, Xinyuan Qian, Beena Ahmed, Eliathamby Ambikairajah, Haizhou Li, and Julien Epps. 2024. Mamba in Speech: Towards an alternative to selfattention. arXiv preprint arXiv:2405.12609.

Junbo Zhang, Zhiwen Zhang, Yongqing Wang, Zhiyong Yan, Qiong Song, Yukai Huang, Ke Li, Daniel Povey, and Yujun Wang. 2021. Speechocean762: An open-source non-native English speech corpus for pronunciation assessment. In Proceedings of the Annual Conference of the International Speech Communication Association (INTERSPEECH), pages 3710 –3714.

<table><tr><td rowspan="2"></td><td colspan="9">APA</td><td colspan="2">MDD</td><td rowspan="2"></td></tr><tr><td colspan="2">Phone</td><td colspan="2">Word</td><td></td><td colspan="5">Utterance</td><td>F1↑</td></tr><tr><td></td><td>MSE↓ 0.062</td><td>Acc.↑ 0.739</td><td>Acc.↑</td><td>Stress↑ 0.366</td><td>Total↑ 0.718</td><td>Acc.↑ 0.807</td><td>Comp.↑ 0.278</td><td>0.848</td><td>Fluency↑ Prosody↑ 0.843</td><td>Total↑ 0.829</td><td>63.85%</td><td>PER↓ 2.72%</td></tr><tr><td>HMamba 0.708</td><td colspan="14"></td></tr><tr><td>-wav2vec2</td><td>0.062</td><td>0.736</td><td>0.708</td><td>0.326</td><td>0.718</td><td>Acoustic Features 0.801</td><td>0.185</td><td>0.840</td><td>0.833</td><td>0.823</td><td>63.63%</td><td>2.79%</td></tr><tr><td>-HuBERT</td><td>0.063</td><td>0.735</td><td>0.706</td><td>0.344</td><td>0.715</td><td>0.804</td><td>0.216</td><td>0.843</td><td>0.838</td><td>0.825</td><td>63.49%</td><td>2.90%</td></tr><tr><td>-wavLM</td><td>0.063</td><td>0.731</td><td>0.705</td><td>0.355</td><td>0.715</td><td>0.806</td><td>0.247</td><td>0.844</td><td>0.838</td><td>0.827</td><td>63.39%</td><td>2.97%</td></tr><tr><td>-duration</td><td>0.063</td><td>0.734</td><td>0.705</td><td>0.341</td><td>0.715</td><td>0.804</td><td>0.299</td><td>0.844</td><td>0.838</td><td>0.826</td><td>63.70%</td><td>2.80%</td></tr><tr><td>-energy</td><td>0.063</td><td>0.735</td><td>0.706</td><td>0.358</td><td>0.716</td><td>0.802</td><td>0.257</td><td>0.840</td><td>0.834</td><td>0.823</td><td>63.28%</td><td>2.78%</td></tr><tr><td>-GOP</td><td>0.066</td><td>0.719</td><td>0.699</td><td>0.293</td><td>0.706</td><td>0.795</td><td>0.228</td><td>0.837</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>Phonological Features</td><td></td><td></td><td></td><td>0.829</td><td>0.817</td><td>61.72%</td><td>2.79%</td></tr><tr><td colspan="14"></td></tr><tr><td>-absolute-pos</td><td>0.063</td><td>0.735</td><td>0.706</td><td>0.332</td><td>0.715</td><td>0.802</td><td>0.261</td><td>0.843</td><td>0.838</td><td>0.825</td><td>63.48%</td><td>2.79%</td></tr><tr><td>-relative-pos</td><td>0.063</td><td>0.733</td><td>0.704</td><td>0.352</td><td>0.714</td><td>0.804</td><td>0.220</td><td>0.847</td><td>0.841</td><td>0.825</td><td>62.74%</td><td>2.81%</td></tr><tr><td>-canonical-phn</td><td>0.083</td><td>0.624</td><td>0.604</td><td>0.310</td><td>0.617</td><td>0.775</td><td>0.147</td><td>0.842</td><td>0.836</td><td>0.801</td><td>28.06%</td><td>14.55%</td></tr></table>

Table 5: Feature ablations of HMamba (MDD performance is reported with F1 and PER as representative metrics).

## Appendix

## A Implementation Details

Feature Extraction: We adopt an open-source acoustic model<sup>5</sup> to extract GOP features, which also serves as an aligner for force alignment. Subsequently, the phone-level duration, energy statistics, and SSL features (average over time frames) are computed according to the alignment. The resulting acoustic features � and all embeddings are 128 dimensions. For all Mamba blocks, we set the number of hidden units to 128 and use a kernel size of 4 for the 1-D convolution. The SSM modules follow the original configuration used in Mamba. $L _ { p } , L _ { w } , L _ { u }$ are set to 3, 1, 1, respectively. In addition, the word-level 1-D convolution has 256 kernels, each with a size of 3.

Hyperparameters setting: � in attention pooling layer is set to 1.0. The combining weights $\omega _ { g }$ for APA loss are uniformly set to 1.0 for each granularity level �. Parameters � and � are tuned to be 0.7 and 0.003, respectively.

## B Feature Ablations

In Table 5, we conduct an ablation study on the feature extraction to inspect the factors that influence on APA. Specifically, we removed one factor at a time to investigate the performance variations.

Acoustic Features: According to the ablation experiment, each acoustic feature used in this work contributes to specific aspect assessments. While the model may perform better in assessing utterance completeness without phone duration features, the other aspect assessments and MDD performance decreases synchronously. Furthermore, among all acoustic features, GOP is the most crucial factor in relation to both of the APA and MDD performance.

Phonological Features: As for the phonological features, the canonical phoneme embeddings are the most critical features overall, particularly in MDD. Without canonical phoneme embeddings, the performance dramatically degrades in F1 score and PER. Since the number of mispronunciations is typically far less than the correct pronunciations, canonical phoneme embeddings can provide ample text-prompt information to complement the acoustic cues. The impact of canonical phoneme embeddings on APA is consistent with Gong et al., 2022, and we take it a step further in this work by demonstrating that they are also pivotal for MDD.

<table><tr><td rowspan="3">Block Type</td><td colspan="10">APA</td><td rowspan="2">MDD</td></tr><tr><td colspan="2">Phone</td><td></td><td>Word</td><td></td><td></td><td></td><td>Utterance</td><td></td><td>F1↑</td></tr><tr><td>MSE↓</td><td>Acc.↑</td><td></td><td>Acc.↑</td><td>Stress↑</td><td>Total↑</td><td>Acc.↑ Comp.↑</td><td></td><td>Fluency↑ Prosody↑ Total↑</td><td></td><td></td><td>PER↓</td></tr><tr><td>Transformer</td><td>0.071</td><td>0.692</td><td>0.689</td><td>0.294</td><td>0.700</td><td>0.797</td><td>0.165</td><td>0.844</td><td>0.839</td><td>0.819</td><td>60.14%</td><td>3.50%</td></tr><tr><td>Mamba</td><td>0.062</td><td>0.739</td><td>0.708</td><td>0.366</td><td>0.718</td><td>0.807</td><td>0.278</td><td>0.848</td><td>0.843</td><td>0.829</td><td>63.85%</td><td>2.72%</td></tr></table>

Table 6: Performance comparison between Mamba block and Transformer block.

<table><tr><td>Block Type</td><td>Params(M)↓</td><td>MACs(G)↓</td></tr><tr><td>Transformer</td><td>1.469</td><td>3.806</td></tr><tr><td>Mamba</td><td>1.141</td><td>2.954</td></tr></table>

Table 7: Computational efficiency between Mamba block and Transformer block.

![](images/c43442e4e1d54bb4ca29a8cf941b747f16d31568b6a6d54774f841f9d22bcc34.jpg)

![](images/29d1b0cee7ed02eb280a158bba1358506cb2f310ef7729dcfb16dd493330097b.jpg)

![](images/8b296f50fe1f18c87ad806b6efce08ea6a3556b66d2f3ebd9ac8c99300a89100.jpg)

![](images/77a8336b3c4113fcd786934fd8c2b2d024dce3b67f9a02c28e4d0259cbb80b79.jpg)

![](images/afd927ca27ae17e2d902941cac96aba62c3067603e4f47a1deb9a79ddcb5a5fc.jpg)  
Figure 4: Comparison of the training curves for models equipped with Mamba and Transformer.

## C Mamba v.s. Transformer

To validate the effectiveness of Mamba over Transformer in different facets, we replace each Mamba block in HMamba with a vanilla Transformer block (encoder only). In the following, we perform a set of qualitative analyses for comparisons of these two variant structures.

Performance comparison: In the first experiment, we compare the APA and MDD performance by utilizing Mamba or Transformer as a basic block, respectively. According to the results shown in Table 6, Mamba consistently outperforms Transformer across all assessment tasks and MDD, especially in phone- and wordlevel assessments. The key difference between Mamba and Transformer is that Mamba leverages 1-D convolution to model the local context dependency which has been shown crucial in either APA (Do et al., 2023a) or MDD (Lee, 2016) task. These findings also align with other research in the speech community, such as speech separation (Jiang et al., 2024) and speech enhancement (Zhang et al., 2024).

Computational efficiency: We further investigate the computational efficiency of two variants of architectures with the number of their parameters and multiply-accumulate operations (MACs). In Table 7, we observe that the model equipped with Mamba has fewer parameters and requires fewer MACs compared to the model with Transformer. This reduced complexity suggests that Mamba is not only more resource-efficient but also potentially more scalable for practical applications where both performance and resource constraints are critical.

Training efficiency: To track the full training process of the two architectures, we plot various training curves in Figure 4, including the training loss (log scale) and the changes in key metrics, such as PER and the total scores of the phones, words, and utterances. Mamba (red) consistently exhibits lower training loss than Transformer (blue) throughout the steps, suggesting that Mamba enables faster and better convergence, ultimately achieving a smaller loss overall. As training progresses, Mamba rapidly outperforms Transformer in terms of PER and PCC for both phone and word total scores. Although this advantage is less pronounced in the PCC curve for the utterance total score, Mamba still surpasses Transformer to a moderate extent.