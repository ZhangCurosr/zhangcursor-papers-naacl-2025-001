# Token-Level Density-Based Uncertainty Quantification Methods for Eliciting Truthfulness of Large Language Models

Artem Vazhentsev<sup>2,3</sup> Lyudmila Rvanova<sup>4,6</sup> Ivan Lazichny<sup>3</sup>

Alexander Panchenko<sup>2,3</sup> Maxim Panov<sup>1</sup> Timothy Baldwin<sup>1,5</sup> Artem Shelmanov<sup>1</sup>

<sup>1</sup>MBZUAI <sup>2</sup>Center for Artificial Intelligence <sup>3</sup>Computational Semantics Group

<sup>4</sup>Laboratory for Analysis and Controllable Text Generation Technologies RAS

<sup>5</sup>The University of Melbourne <sup>6</sup>Weakly-Supervised NLP Group vazhentsev@airi.net artem.shelmanov@mbzuai.ac.ae

## Abstract

Uncertainty quantification (UQ) is a prominent approach for eliciting truthful answers from large language models (LLMs). To date, information-based and consistency-based UQ have been the dominant UQ methods for text generation via LLMs. Density-based methods, despite being very effective for UQ in text classification with encoder-based models, have not been very successful with generative LLMs. In this work, we adapt Mahalanobis Distance (MD) – a well-established UQ technique in classification tasks – for text generation and introduce a new supervised UQ method. Our method extracts token embeddings from multiple layers of LLMs, computes MD scores for each token, and uses linear regression trained on these features to provide robust uncertainty scores. Through extensive experiments on eleven datasets, we demonstrate that our approach substantially improves over existing UQ methods, providing accurate and computationally efficient uncertainty scores for both sequence-level selective generation and claimlevel fact-checking tasks. Our method also exhibits strong generalization to out-of-domain data, making it suitable for a wide range of LLM-based applications.

## 1 Introduction

Large language models (LLMs) have achieved impressive results over various tasks and applications (OpenAI et al., 2024; Dubey et al., 2024; Rivière et al., 2024). Nevertheless, even the most advanced LLMs are inevitably prone to making mistakes during text generation. Their responses often contain hallucinations or non-factual claims (Xiao and Wang, 2021; Dziri et al., 2022), posing significant challenges for LLM deployment in safetycritical domains.

Many studies have investigated methods for assessing the truthfulness of LLM responses (Manakul et al., 2023; Min et al., 2023; Chen et al., 2023;

Feng et al., 2024). However, many of the proposed techniques have limited practical applicability, as they often rely on external knowledge sources or require ensembling multiple large LLMs, leading to high computational costs that make them economically unfeasible for many use cases.

One of the most promising approaches to addressing this challenge is uncertainty quantification (UQ) (Gal and Ghahramani, 2016; Shelmanov et al., 2021; Baan et al., 2023; Geng et al., 2024; Fadeeva et al., 2023). This research direction recognizes that we will never have complete information about model predictions due to the limited amount of training data and ambiguity of the tasks, and suggests general ways to estimate the reliability of predictions under different conditions. Recently, a suite of UQ methods specifically designed for text generation with LLMs has been developed (Fomicheva et al., 2020; Lin et al., 2023; Kuhn et al., 2023; Farquhar et al., 2024; Duan et al., 2024). However, many of these methods are either ineffective or come with a substantial computational overhead, limiting their practicality for largescale or real-time applications.

For text classification and regression tasks, researchers have identified several groups of techniques that maintain a balance between effectiveness and computational efficiency (Zhang et al., 2019; He et al., 2020; Xin et al., 2021; Wang et al., 2022; Vazhentsev et al., 2023a; He et al., 2024a). One such class of approach is so-called densitybased uncertainty scores (Lee et al., 2018; van Amersfoort et al., 2020; Kotelevskii et al., 2022; Yoo et al., 2022). These methods use embeddings of instances obtained from the top layers of a classification model to fit the density of the training distribution in the latent space. The likelihood of the input data under this estimated distribution is then used for confidence estimation. This has been demonstrated to achieve excellent results in out-ofdistribution detection tasks (Podolskiy et al., 2021), and proven to be useful for selective text classification (Vazhentsev et al., 2022, 2023a). Despite being computationally lightweight, these techniques often outperform more resource-intensive methods, such as deep ensembles (Lakshminarayanan et al., 2017) and Monte Carlo dropout (Gal and Ghahramani, 2016; Tsymbalov et al., 2018). Unfortunately, the reported performance of density-based scores for text generation so far has been notably low (Vashurin et al., 2024).

![](images/08b4bfb6a411a531bb7f7fb732db391d530f534365367310692ed7a8d76bf416.jpg)  
Figure 1: An illustration of the proposed method. After each decoder layer, the embeddings of each generated token are extracted. Subsequently, we compute the Mahalanobis distance for each token and layer and then average over all tokens in the generated sequence. Finally, we train a linear regression on the PCA decomposition of the calculated features with the addition of sequence probability to predict the uncertainty of the generation.

Recent work has demonstrated that the internal states of LLMs carry a lot of information about their uncertainty (Azaria and Mitchell, 2023; Chen et al., 2024; He et al., 2024b; CH-Wang et al., 2024; Vazhentsev et al., 2024). These techniques train a supplementary model on top of the activations of LLM internal layers. However, they often rely on simplistic features and fail to incorporate more advanced, well-established density-based UQ methods, limiting their ability to capture uncertainty.

In this work, we address this gap by adapting density-based techniques for the UQ of LLMs and propose a new supervised method based on densitybased features. Specifically, we adapt one of the most robust methods for UQ in the classification tasks, namely Mahalanobis Distance (MD; Lee et al. (2018)), and train a linear regression on top of the MD scores from various layers of the LLM. These features are supplemented with a probability of the generated sequence. Figure 1 illustrates the scheme of the proposed supervised UQ method. Our extensive experimental evaluation demonstrates that the proposed method provides substantial improvement over the state of the art.

Our key contributions are as follows.

• We conduct a comprehensive investigation of density-based UQ methods for LLMs. While previous research (Vashurin et al., 2024) has indicated that sequence-level density-based methods are ineffective, we propose a tokenlevel adaptation of MD that is on par with or better than state-of-the-art UQ techniques.

• We propose a new computationally efficient supervised method for UQ in LLMs using layer-wise density-based scores as features to improve uncertainty estimation without sacrificing the performance.

• We conduct a vast empirical investigation that demonstrates the effectiveness of the proposed methods for sequence-level selective classification across eleven datasets and claim-level fact-checking.

## 2 Related Work

Many effective UQ methods, such as deep ensembles (Lakshminarayanan et al., 2017) and Monte Carlo (MC) dropout (Gal and Ghahramani, 2016), require sampling multiple predictions from a model, which leads to substantial computational and memory overheads. A key challenge in UQ is developing techniques that balance effectiveness with computational efficiency. Among the most promising approaches in this regard are densitybased methods (Lee et al., 2018; Liu et al., 2020; van Amersfoort et al., 2020; Kotelevskii et al.,

2022; Yoo et al., 2022). These methods leverage latent representations of instances to model the training data distribution, then estimate how likely a new instance belongs to that distribution. Lee et al. (2018) propose to use Mahalanobis Distance (MD) as a measure of uncertainty for out-of-distribution detection in computer vision tasks. Podolskiy et al. (2021) adapt MD to out-of-distribution in text classification tasks. Vazhentsev et al. (2022, 2023b) show that it also provides high performance in selective text classification.

For LLMs, Fomicheva et al. (2020) and Kuhn et al. (2023) proposed UQ methods that sample multiple predictions and leverage their diversity. In the context of black-box LLMs, where we have no access to the logits or embeddings of a model, Fomicheva et al. (2020) propose the use of lexical dissimilarity of sampled texts as a measure of uncertainty. Lin et al. (2023) leverage a similarity matrix between responses for deeper analysis of the diversity of the sampled generations. Some methods also combine sampling diversity measures with the probability of each generation (Kuhn et al., 2023; Duan et al., 2024; Nikitin et al., 2024; Cheng and Vlachos, 2024; Chen et al., 2024; Vashurin et al., 2025).

Recently, it was demonstrated that MD is an efficient approach for out-of-distribution detection in sequence-to-sequence models (Vazhentsev et al., 2023b; Ren et al., 2023; Darrin et al., 2023). However, for selective generation tasks, densitybased methods so far have substantially underperformed compared to trivial baselines (Vashurin et al., 2024).

Supervised methods are another research direction for UQ of LLMs. Azaria and Mitchell (2023) demonstrate that the internal states of the model contain information about uncertainty, and propose to train a multi-layer perceptron over the hidden LLM representation to predict the truthfulness of the model responses. He et al. (2024b) enhance this idea by training a deep neural network with recurrent and convolutional layers. Furthermore, this method uses embeddings from all layers and incorporates features based on the probability and the dynamics of the generated tokens through layers. In contrast to these methods, we employ a simple linear model, but focus on more accurate feature extraction from internal layers, using well-established density-based UQ methods.

## 3 Background on Density-Based Methods

Recently, Mahalanobis distance (MD) and Robust Density Estimation (RDE) were adapted (Vazhentsev et al., 2023b; Ren et al., 2023) to the text generation task by considering the marginal distribution of the training dataset.

Following the assumption of a Gaussian distribution of training instance representations, the MD method (Lee et al., 2018) calculates a centroid of the training data $\mu$ and the empirical covariance matrix Σ. For a given instance x, the uncertainty score is defined as the Mahalanobis distance:

$$
U ^ { \mathrm { M D } } ( \mathbf { x } , l ) = { { ( { h _ { l } } ( \mathbf { x } ) - \mu ) } ^ { T } } { \Sigma ^ { - 1 } } { { ( { h _ { l } } ( \mathbf { x } ) - \mu ) } } ,
$$

where $h _ { l } ( \mathbf { x } )$ is a hidden representation extracted from the layer l.

RDE (Yoo et al., 2022) operates within the reduced dimensionality of $h _ { l } ( \mathbf { x } )$ via the kernel PCA decomposition. To ensure the robustness of the covariance matrix, it uses the Minimum Covariance Determinant estimate (Rousseeuw, 1984). Finally, the uncertainty score is computed as the Mahalanobis distance in the reduced dimensionality.

Ren et al. (2023) proposed a modification of MD – Relative Mahalanobis Distance (RMD). It takes into account a background contrastive MD score. The score aims to assess how close the test instance is to the in-domain training data compared to the background data. The uncertainty score based on RMD is given by the following equation:

$$
\begin{array} { r } { U ^ { \mathrm { R M D } } ( \mathbf { x } , l ) = U ^ { \mathrm { M D } } ( \mathbf { x } , l ) - U _ { 0 } ^ { \mathrm { M D } } ( \mathbf { x } , l ) , } \end{array}
$$

where $U _ { 0 } ^ { \mathrm { M D } } ( \mathbf { x } , l )$ is a Mahalanobis distance computed with the centroid $\mu ^ { 0 }$ and the empirical covariance matrix $\Sigma ^ { 0 }$ calculated using the background dataset, such as C4 (Raffel et al., 2020).

For the sequence-to-sequence tasks, it was proposed to use the last encoder and decoder layer for extracting hidden representation of the model (Vazhentsev et al., 2023b; Ren et al., 2023). In contrast, recent research (Azaria and Mitchell, 2023; Chen et al., 2024) indicates that the middle layers of the model may be more suitable for decoder-only models.

## 4 Proposed Method: Token-Level Mahalanobis Distance

To define the method, we assume access to training data consisting of a set of prompts paired with

LLM responses, each accompanied by an assessment of its correctness. The assessment can be based on ground truth answers (as in tasks like question-answering, machine translation, or summarization) or through alternative means, such as human annotation or another big LLM.

## 4.1 Layer-Wise Uncertainty Score

Embedding extraction. First of all, we need to extract embeddings of instances in the training dataset. We note that previous works use sequencelevel embeddings, which are essentially an average of token-level embeddings. Recent studies (Azaria and Mitchell, 2023; Chen et al., 2024) note that sequence embeddings might be useless for UQ with LLMs and propose to use embeddings of the last or the first generated token, as they encode useful information for estimating the truthfulness of the entire generation. We acknowledge that this property may not always hold, as the informative tokens are likely to vary depending on the specific task. In our method, we first compute individual token-level uncertainty scores and then aggregate them into a sequence-level score.

Embedding selection. To construct a covariance matrix and centroid for MD, a model training set is required. However, unlike standard text classification tasks, where training sets are typically limited and accessible during the development of an ML-based application, the pre-training data for general-purpose LLMs is extremely large and usually not publicly available. Moreover, even if this data were available, LLM performance on it would likely be not homogeneous and could be low for certain tasks. Therefore, to construct the parameters for MD, we propose selecting a subset of token embeddings from high-quality LLM responses.

From the responses generated in the training set, we select a subset of token embeddings that correspond to responses that meet a defined correctness criterion. Let $\tau$ be a training set of input prompts and $| \mathcal { T } | = N _ { | \mathcal { T } | }$ . For each prompt $\mathbf { x } ^ { j } \in \tau$ , the model generates a response as a sequence $\tilde { \mathbf { y } } ^ { j } = \mathbf { t } _ { 1 } ^ { j } , \ldots , \mathbf { t } _ { N _ { j } } ^ { j }$ , where $N _ { j }$ is a length of the j-th generation and $\mathbf { t } _ { i } ^ { j } , i \in [ 1 , \dots , N _ { j } ]$ is an i-th token in the response. We define a set of selected tokens as $\mathcal { D } = \{ \mathbf { t } _ { i } ^ { j } \colon \mathcal { Q } ( \tilde { \mathbf { y } } ^ { j } ) > \tau , i \in [ 1 , \ldots , N _ { j } ] , j \in$ $[ 1 , . . . , N _ { | T | } ] \}$ , where $\mathcal { Q } ( \cdot )$ is a quality metric and $\tau$ is a given threshold. Then $\mathcal { E } _ { l } = \{ h _ { l } ( \mathbf { t } ) \colon \mathbf { t } \in \mathcal { D } \}$ is the set of selected token embeddings. The correctness criterion helps filter out low-quality responses.

Depending on the dataset in the experiment, exact match and AlignScore are employed as quality metrics. The correctness criterion used for token selection is described in Section 5.1.

Layer-wise scores. For each layer $l = 1 , \ldots , L$ of the model, we compute the covariance matrix $\Sigma _ { \mathcal { E } _ { l } }$ and the centroid $\mu \varepsilon _ { l }$ using the set of selected token embeddings $\mathcal { E } _ { l }$ . For each token from the generated sequence $\tilde { \mathbf { y } } ^ { k } = \mathbf { t } _ { 1 } ^ { k } , \ldots , \mathbf { t } _ { N _ { k } } ^ { k }$ , we compute the layer-wise MD as follows:

$$
U ^ { \mathrm { M D } } ( \mathbf { t } _ { i } ^ { k } , l ) = \big ( h _ { l } ( \mathbf { t } _ { i } ^ { k } ) - \mu \varepsilon _ { l } \big ) ^ { T } \Sigma _ { \mathcal { E } _ { l } } ^ { - 1 } \big ( h _ { l } ( \mathbf { t } _ { i } ^ { k } ) - \mu \varepsilon _ { l } \big ) .
$$

For the token-level RMD, we additionally compute the background covariance matrix $\Sigma _ { l } ^ { 0 }$ and the background centroid $\mu _ { l } ^ { 0 }$ using the embeddings of all generated tokens for the input prompts from some background dataset.

Finally, the uncertainty score of the entire generated sequence $\tilde { \mathbf { y } } ^ { k }$ is the Average Token-level Mahalanobis Distances $( A T M D )$ over $\mathbf { t } _ { i } ^ { k } , i = 1 , \ldots , N _ { k }$ (for RMD, we designate it as ATRMD).

## 4.2 Linear Regression on Layer-Wise Scores

The ATMD and ATRMD scores can be computed on various layers. Azaria and Mitchell (2023) indicate that the best-performing layer may vary depending on the generation task. To effectively integrate information from multiple layers, we propose training a regression model on top of the layer-wise scores.

For a generation $\tilde { \mathbf { y } } ^ { k }$ , we construct a vector of features based on ATMD or ATRMD: $f ^ { * } ( \tilde { \mathbf { y } } ^ { k } ) =$ $[ U ^ { * } ( \tilde { \mathbf { y } } ^ { k } , 1 ) , \dots , U ^ { * } ( \tilde { \mathbf { y } } ^ { k } , L ) ]$ (we use instead of ATMD or ATRMD). To learn the uncertainty of the generation, we define target variables as negative values of a quality metric for generations $\tilde { \mathbf { y } } ^ { k } \mathrm { ; }$ $\mathbf { q } ^ { k } = - \mathcal { Q } ( \tilde { \mathbf { y } } ^ { k } )$ . We note that the features $f ^ { * } ( \tilde { \mathbf { y } } ^ { k } )$ might be highly correlated with each other (a multicollinearity problem; Shrestha (2020)), which makes linear models to overfit (Chan et al., 2022). To make our features more robust, we use top $N = 1 0$ components from the PCA decomposition of feature vectors: $\tilde { f } ^ { * } ( \tilde { \mathbf { y } } ^ { k } ) = \mathrm { P C A } _ { N } \big ( f ^ { * } ( \tilde { \mathbf { y } } ^ { k } ) \big )$ .

We train the machine learning model $G ( \cdot )$ to predict an uncertainty score as follows:

1. Split the entire training dataset $\tau$ into two parts $\mathcal { T } _ { 1 }$ and $\mathcal { T } _ { 2 }$ .

2. Using $\mathcal { T } _ { 1 }$ , construct $\mathcal { E } _ { l } , l \in [ 1 , \dots , L ]$ and fit layer-wise covariance matrices and centroids. ATRMD also fits layer-wise background covariance matrices and background centroids.

3. For each generation $\tilde { \mathbf { y } } ^ { k } , k = 1 , \dots , | \mathcal { T } _ { 2 } |$ for the prompts from $\mathcal { T } _ { 2 } .$ , compute features $\tilde { f } ^ { * } ( \tilde { \mathbf { y } } ^ { k } )$ and targets $\mathbf { q } ^ { k }$

4. Train the machine learning model $G ^ { * } ( \cdot )$ to predict the targets $\mathbf { q } ^ { k }$ using the features $\tilde { f } ^ { * } ( \tilde { \mathbf { y } } ^ { k } ) , k = 1 , \dots , | T _ { 2 } |$ . In our work, we use linear regression models as $G ^ { * } ( \cdot )$

5. Re-estimate layer-wise parameters of the distribution using the entire training dataset $\tau$

Finally, the supervised uncertainty score for a test generation $\tilde { \mathbf { y } } ^ { k }$ based on token-level MD or RMD, namely SATMD or SATRMD is:

$$
U ^ { \mathrm { S } ^ { \ast } } ( \tilde { \mathbf { y } } ^ { k } ) = G ^ { \ast } \big ( \tilde { f } ^ { \ast } ( \tilde { \mathbf { y } } ^ { k } ) \big ) .
$$

Following He et al. (2024b), we also experiment with adding the sequence probability $P ( \mathbf { \bar { y } } ^ { k } \mid \mathbf { x } ^ { k } )$ as an additional feature to the features vector: $\tilde { f ^ { * } } _ { p r o b } ( \tilde { \mathbf { y } } ^ { k } ) = [ \tilde { f ^ { * } } ( \tilde { \mathbf { y } } ^ { k } ) ; P ( \tilde { \mathbf { y } } ^ { k } \mid \mathbf { x } ^ { k } ) ]$ , and get

$$
U ^ { \mathrm { S } ^ { \ast } + \mathrm { M S P } } ( \tilde { \mathbf { y } } ^ { k } ) = G ^ { \ast } \big ( \tilde { f } ^ { \ast } { } _ { p r o b } ( \tilde { \mathbf { y } } ^ { k } ) \big ) .
$$

## 4.3 Hybrid Score

In addition, we explore Hybrid Uncertainty Quantification (HUQ; Vazhentsev et al. (2023a)), which empirically combines multiple uncertainty scores. Using HUQ, we combine sequence probability $U _ { 1 } ( \tilde { \mathbf { y } } ^ { k } ) ~ = ~ 1 - { P } ( \tilde { \mathbf { y } } ^ { k } ~ | ~ \mathbf { x } ^ { k } )$ and the proposed SATMD or SATRMD scores: $U _ { 2 } ( \tilde { \mathbf { y } } ^ { k } ) = \mathbf { \hat { \psi } } ^ { \mathbf { \hat { \mathbf { y } } } ^ { \mathbf { \hat { \mathbf { x } } } } } ( \tilde { \mathbf { y } } ^ { k } )$ The hyperparameters of HUQ are tuned on the $\mathcal { T } _ { 2 }$ dataset. A detailed description of the HUQ method is given in Appendix B.

## 5 Experiments

## 5.1 Experimental Setup

For the experimental evaluation, we employ the LM-Polygraph framework (Fadeeva et al., 2023; Vashurin et al., 2024). We consider two tasks: (1) sequence-level selective generation (Ren et al., 2023), in which we can “reject” untruthful generations based on provided uncertainty; (2) claim-level fact-checking (Fadeeva et al., 2024), where we aim to identify nonfactual claims in long generations, consisting of several claims.

Metrics. To evaluate the quality of UQ methods on the selective generation task, we use the standard Prediction Rejection Ratio (PPR) metric (Malinin and Gales, 2021; Vashurin et al., 2024). This metric measures the correctness of the ranking of generations based on uncertainty relative to a specified quality metric. PRR computes the area under the Prediction Rejection (PR) curve, which is constructed by sequentially rejecting the most uncertain generation and calculating the average quality for all stored generations at each possible threshold. Subsequently, this area is normalized by scaling between the PR curve for the random selection and oracle selection. A higher value of the PRR corresponds to a better quality of selective generation. Following previous work (Vashurin et al., 2024), we use ROUGE-L, Accuracy, and AlignScore (Zha et al., 2023) as text generation quality metrics.

For claim-level fact-checking, we follow previous work (Fadeeva et al., 2024) and consider this task as a binary classification problem. We utilize the ROC-AUC and PR-AUC metrics, where nonfactual claims represent a positive class.

Models. For selective generation, we experiment with two state-of-the-art LLMs in their size: Llama@8b v3.1 (Dubey et al., 2024) and Gemma 9b v2 (Rivière et al., 2024). For factchecking, we utilize Mistral 7b v0.1 Instruct (Jiang et al., 2023). The inference hyperparameters are presented in Table 9 in Appendix F.

Datasets. We consider several text generation tasks, including text summarization (TS), questionanswering (QA) with long free-form answers, QA based on reading comprehension, QA with short free-form answers, and multiple-choice QA. Dataset statistics are presented in Table 8 in Appendix E. For TS, we utilize XSum (Narayan et al., 2018), SamSum (Gliwa et al., 2019), and the CNN/- DailyMail (See et al., 2017) dataset. For QA with long free-form answers, we use PubMedQA (Jin et al., 2019), MedQUAD (Abacha and Demner-Fushman, 2019), TruthfulQA (Lin et al., 2022), and GSM8k (Cobbe et al., 2021). For reading comprehension, we use CoQA (Reddy et al., 2019) and SciQ (Welbl et al., 2017). For QA with short free-form answers, we use TriviaQA (Joshi et al., 2017). The last three datasets represent the common benchmarks for evaluating UQ methods in previous work (Kuhn et al., 2023; Duan et al., 2024; Lin et al., 2023). For multiple-choice QA, we utilize MMLU (Hendrycks et al., 2021), which is a common dataset for evaluating LLMs.

UQ baselines. In an experimental evaluation, we compare the proposed methods against several UQ baselines, including trivial yet robust informationbased methods such as Maximum Sequence Probability (MSP) and Perplexity (Fomicheva et al.,

![](images/79d846daf8d986083cdde762464cc7e8d08ab3da9e1698247167709ec3bfd2bd.jpg)

Figure 2: Performance of embeddings from various layers in density-based scores. PRR for density-based methods computed using embeddings from various layers of Llama 8b v3.1 (upper row) and Gemma 9b v2 (lower row) models. Raw ATMD/ATRMD denotes a corresponding method without selecting embeddings using the correctness criterion. Higher values indicate better results.
<table><tr><td rowspan="2">UQ Method</td><td rowspan="2">XSUM</td><td colspan="6">SamSum</td><td rowspan="2">PubMedQA</td><td colspan="2">MedQUAD</td><td rowspan="2">TruthfulQA</td><td colspan="2">CoQA</td><td rowspan="2">TriviaQA AlignScore</td><td colspan="2">GSM8k</td><td rowspan="2">MMLU Mean Rank</td></tr><tr><td>ROUGE-L AlignScore</td><td></td><td>ROUGE-L AlignScore</td><td>CNN ROUGE-L AlignScore</td><td></td><td>ROUGE-L AlignScore</td><td>ROUGE-L AlignScore</td><td>AlignScore</td><td>AlignScore</td><td>SciQ AlignScore</td><td>Accuracy</td><td>Accuracy</td></tr><tr><td>SATMD</td><td>.189</td><td>.062</td><td>.369</td><td>.232</td><td>.059</td><td>.030</td><td>.217</td><td>.183</td><td>.471</td><td>.552</td><td>.230</td><td>.310</td><td>.385</td><td>.293</td><td>.587</td><td>.623</td><td>4.75</td></tr><tr><td>SATRMD</td><td>.389</td><td>.181</td><td>.338</td><td>.282</td><td>.138</td><td>-.004</td><td>.351</td><td>.213</td><td>.459</td><td>.559</td><td>.251</td><td>.324</td><td>.542</td><td>.394</td><td>.606</td><td>.504</td><td>3.69</td></tr><tr><td>HUQ-SATMD</td><td>-.326</td><td>-.123</td><td>441</td><td>.249</td><td>.055</td><td>.030</td><td>.088</td><td>.128</td><td>.417</td><td>.465</td><td>.289</td><td>.450</td><td>577</td><td>.683</td><td>.540</td><td>.552</td><td>4.56</td></tr><tr><td>HUQ-SATRMD</td><td>.395</td><td>.187</td><td>.486</td><td>.297</td><td>.126</td><td>.048</td><td>.351</td><td>211</td><td>.386</td><td>.506</td><td>.308</td><td>.450</td><td>.653</td><td>.646</td><td>.592</td><td>.609</td><td>2.94</td></tr><tr><td>SATMD+MSP</td><td>.234</td><td>.086</td><td>.377</td><td>.420</td><td>.094</td><td>.074</td><td>.371</td><td>.203</td><td>.493</td><td>.527</td><td>.361</td><td>.466</td><td>.178</td><td>.708</td><td>.618</td><td>.836</td><td>2.50</td></tr><tr><td>SATRMD+MSP</td><td>.372</td><td>.179</td><td>.383</td><td>.408</td><td>.135</td><td>.016</td><td>.372</td><td>.202</td><td>.466</td><td>.575</td><td>353</td><td>.419</td><td>.542</td><td>.702</td><td>.642</td><td>.816</td><td>2.56</td></tr></table>

Table 1: Performance of various versions of the proposed supervised methods. PRR for Llama 8b v3.1 model for various tasks for the considered sequence-level aggregation methods. Warmer color indicates better results.

2020), and consistency-based methods considered state-of-the-art for LLMs (Vashurin et al., 2024): Lexical Similarity based on ROUGE-L (Fomicheva et al., 2020), black-box methods (DegMat, Eccentricity, EigValLaplacian; Lin et al. (2023)), Semantic Entropy (Kuhn et al., 2023), and Shifting Attention to Relevance (SAR; Duan et al. (2024)). Furthermore, to ensure the robustness of the proposed methods, the suite of baselines in our experiments also includes methods that utilize model internal states: Factoscope (He et al., 2024b), SAPLMA (Azaria and Mitchell, 2023), and Eigen-Score (Chen et al., 2024). The first two are supervised methods, while EigenScore is unsupervised. Following the previous works (Azaria and Mitchell, 2023; Chen et al., 2024), we use embeddings from the middle layer of the model for the latter two methods. For the methods that require sampling, we sample five generations for each input text.

Configuration of ATMD/ATRMD. In indomain experimental evaluation on the SciQ, CoQA, TriviaQA, MMLU, and GSM8k datasets, we select token embeddings used to construct the covariance matrix and centroids for ATMD and ATRMD from generations that are fully accurate according to the exact match criterion. On other datasets, we utilize generations with

AlignScore greater than 0.3. Raw ATMD/ATRMD denotes a corresponding method without selecting embeddings using the correctness criterion.

## 5.2 Results

Layer-wise comparison of density-based methods. Figure 2 presents the layer-wise comparison of various sequence-level density-based approaches for selective generation for the Llama 8b v3.1 and Gemma 9b v2 models. These results demonstrate the presence of robust patterns across the majority of datasets and models.

Consistent with the findings of Vashurin et al. (2024), we observe that in most cases, densitybased methods that use sequence-level embeddings (MD, RMD, and RDE) yield PRR scores that are close to or below zero, indicating performance comparable to random selection. Only for GSM8k, these methods provide meaningful uncertainty scores, but they still do not outperform the basic MSP baseline. Furthermore, we see that using sequence-level embeddings derived from internal layers does not improve the performance of density-based methods; they usually perform better when using embeddings from the last layer.

MD that uses token-level embeddings performs consistently better than the MD based on sequencelevel embeddings for all datasets except SamSum, where all methods perform similarly to each other. Moreover, density-based methods that compute MD using token-level embeddings from internal layers outperform those that rely on embeddings from the top layers. While for SamSum and MMLU with the Gemma 9b v2 model, ATMD achieves the best performance using the last layer embeddings, for all other cases the best performance is achieved by using embeddings from the middle layers. This finding is consistent with previous research (Azaria and Mitchell, 2023; Chen et al., 2024).

<table><tr><td>UQ Method</td><td colspan="2">XSUM ROUGE-L AlignScore</td><td colspan="2">SamSum ROUGE-L AlignScore</td><td colspan="2">CNN ROUGE-L</td><td colspan="2">PubMedQA ROUGE-L AlignScore</td><td colspan="2">MedQUAD ROUGE-L</td><td colspan="2">TruthfulQA CoQA AlignScore AlignScore</td><td colspan="2">SeiQ TriviaQA AlignScore AlignScore</td><td colspan="2">GSM8k MMLU Accuracy Accuracy</td><td>Mean Rank</td></tr><tr><td>Maximum Sequence Probability</td><td>-.343</td><td>-.128</td><td>.452</td><td>.218</td><td>.021</td><td>.096</td><td>-.155</td><td>-.011</td><td>.297</td><td>.356</td><td>.277</td><td>.450</td><td>.582</td><td></td><td>.687</td><td>.380</td><td>.405 7.81</td></tr><tr><td>Perplexity</td><td>-.384</td><td>-.108</td><td>.080</td><td>.308</td><td>.150</td><td>.242</td><td>.215</td><td>-.044</td><td>.425</td><td>.438</td><td>.178</td><td>.450</td><td>.197</td><td>.689</td><td>.259</td><td>.308</td><td>7.38</td></tr><tr><td>DegMat NLI Score Entail.</td><td>.017</td><td>.093</td><td>.250</td><td>.226</td><td>.084</td><td>.144</td><td>.064</td><td>.058</td><td>.066</td><td>.162</td><td>.156</td><td>.420</td><td>.446</td><td>.714</td><td>.357</td><td>.224</td><td>7.19</td></tr><tr><td>Eccentricity NLI Score Entail.</td><td>-.036</td><td>.013</td><td>.160</td><td>.117</td><td>.028</td><td>.050</td><td>.033</td><td>.021</td><td>.070</td><td>.060</td><td>.122</td><td>.409</td><td>.444</td><td>.654</td><td>.403</td><td>.312</td><td>10.00</td></tr><tr><td>Eig ValLaplacian NLI Score Entail.</td><td>.016</td><td>.099</td><td>.251</td><td>.224</td><td>.087</td><td>.143</td><td>.054</td><td>.054</td><td>.056</td><td>.160</td><td>.152</td><td>.444</td><td>.398</td><td>.669</td><td>.335</td><td>.344</td><td>7.69</td></tr><tr><td>Lexical Similarity ROUGE-L</td><td>.071</td><td>.066</td><td>.320</td><td>.209</td><td>.123</td><td>.122</td><td>.144</td><td>.041</td><td>.252</td><td>.132</td><td>.008</td><td>.403</td><td>.360</td><td>.621</td><td>.467</td><td>.311</td><td>7.81</td></tr><tr><td>SAR</td><td>.040</td><td>.044</td><td>.300</td><td>.217</td><td>.120</td><td>.154</td><td>.122</td><td>.032</td><td>.286</td><td>.192</td><td>.105</td><td>.465</td><td>.440</td><td>.710</td><td>.455</td><td>.284</td><td>6.50</td></tr><tr><td>Semantic Entropy</td><td>.041</td><td>.012</td><td>.311</td><td>.206</td><td>.077</td><td>.096</td><td>.064</td><td>.034</td><td>.075</td><td>.007</td><td>.171</td><td>.416</td><td>.466</td><td>.669</td><td>.424</td><td>.220</td><td>8.38</td></tr><tr><td>SentenceSAR</td><td>-.085</td><td>-.032</td><td>.264</td><td>.215</td><td>.055</td><td>.091</td><td>-.000</td><td>.006</td><td>.015</td><td>.033</td><td>.185</td><td>.472</td><td>.543</td><td>.703</td><td>.151</td><td>.343</td><td>8.75</td></tr><tr><td>Factoscope</td><td>.032</td><td>-.029</td><td>.034</td><td>-.024</td><td>.007</td><td>.004</td><td>-.035</td><td>.001</td><td>.358</td><td>.428</td><td>.017</td><td>.242</td><td>.316</td><td></td><td></td><td></td><td></td></tr><tr><td>EigenScore</td><td>.041</td><td>.029</td><td>.196</td><td>.150</td><td>.040</td><td>.045</td><td>.074</td><td>.027</td><td>.050</td><td>.043</td><td>.023</td><td>.402</td><td>.373</td><td>-.046 .619</td><td>.048 .430</td><td>.727</td><td>11.19</td></tr><tr><td>SAPLMA</td><td>.144</td><td>.129</td><td>.243</td><td>313</td><td>.126</td><td>.179</td><td>.240</td><td>.155</td><td>.407</td><td>.490</td><td>.112</td><td>.082</td><td>.388</td><td>.522</td><td>.598</td><td>.196</td><td>10.44 5.50</td></tr><tr><td>HUQ-SATRMD</td><td></td><td>.187</td><td>.486</td><td></td><td>.126</td><td></td><td></td><td>.211</td><td>.386</td><td></td><td></td><td></td><td></td><td></td><td></td><td>.481</td><td></td></tr><tr><td>SATRMD+MSP</td><td>.395 .372</td><td>.179</td><td>.383</td><td>.297 .408</td><td>.135</td><td>.048 .016</td><td>.351 .372</td><td>.202</td><td>.466</td><td>.506 .575</td><td>.308 .353</td><td>.450 .419</td><td>.653 .542</td><td>.646 .702</td><td>.592 .642</td><td>.609 .816</td><td>3.44 2.94</td></tr></table>

Table 2: Main results on selective generation tasks. PRR for Llama 8b v3.1 model for various tasks for the considered sequence-level methods. Warmer color indicates better results.

Using the selection of correct generations from the training dataset for fitting the covariance matrix and centroid is key to achieving good performance of the methods based on token-level MDs. ATMD/ATRMD consistently outperform raw AT-MD/ATRMD. The highest difference was observed on the MedQUAD and TruthfulQA datasets, where using selection improved PRR by 0.2-0.3.

Comparison of sequence-level aggregations. Tables 1 and 5 in Appendix A.1 present the comparison of various sequence-level supervised approaches for selective generation for the Llama 8b v3.1 and Gemma 9b v2 models. The results demonstrate that SATMD and SATRMD provide stable and robust performance, which is often superior or equal to the performance of the MD/RMD using the embeddings from the best layer. As anticipated, the incorporation of MSP as an additional feature or combining it using HUQ significantly improved the performance of SATMD/SATRMD. While on average by mean rank, the best performance across various datasets was achieved by SATMD+MSP, for XSum and SciQ, HUQ-SATRMD exhibited a slight improvement. It is also noteworthy that using RMD led to a consistent improvement in performance compared to the original MD for all variants of the supervised method.

Main results on the selective generation tasks. The main results on the selective generation tasks for the Llama 8b v3.1 and Gemma 9b v2 models are presented in Tables 2 and 6 in Appendix A.2. In the summarization task, our supervised methods outperform all the baselines on XSum and SamSum. HUQ-SATRMD achieves the best performance on the XSum and SamSum datasets in terms of PRR-ROUGE-L. For SamSum, the SATRMD+MSP method significantly outperforms other methods in terms of PRR-AlignScore. For the CNN dataset, the proposed methods demonstrate the second-best results in terms of PRR-ROUGE-L, but they substantially fall behind unsupervised UQ techniques in terms of PRR-AlignScore.

In the QA tasks with long answers (Pub-MedQA, MedQUAD, TruthfulQA, and GSM8k), SATRMD+MSP consistently demonstrates the best performance, with a notable margin over best supervised and unsupervised techniques, while HUQ-SATRMD ranks second. In the reading comprehension task, HUQ-SATRMD is the best-performing method. Meanwhile, on the MMLU dataset, SATRMD+MSP is the most effective method, significantly outperforming HUQ-SATRMD and other state-of-the-art baselines.

Considering QA with short answers on CoQA, we observe that HUQ-SATRMD performs on par with the MSP baseline, while on the SciQ dataset performs the best with a large margin. On TriviaQA, SATRMD+MSP outperforms the MSP baseline, underperforming only sampling-based methods that require much more computation time.

Overall, we can conclude that HUQ-SATRMD is the most effective method for summarization and reading comprehension tasks, where it significantly outperforms state-of-the-art unsupervised UQ methods. For all other QA datasets, including those with long answers and tasks requiring internal knowledge, the best performance is demonstrated by SATRMD+MSP.

<table><tr><td rowspan="2">UQ Method</td><td colspan="2">SamSum</td><td colspan="2">MedQUAD</td><td>TruthfulQA</td><td>SciQ</td><td>MMLU</td><td rowspan="2">Mean Rank</td></tr><tr><td>ROUGE-L AlignScore</td><td></td><td>ROUGE-L AlignScore</td><td></td><td>AlignScore</td><td>AlignScore</td><td>Accuracy</td></tr><tr><td>Maximum Sequence Probability</td><td>.452</td><td>.218</td><td>297</td><td>.356</td><td>.277</td><td>.582</td><td>.405</td><td>2.29</td></tr><tr><td>DegMat NLI Score Entail.</td><td>.250</td><td>.226</td><td>.066</td><td>.162</td><td>.156</td><td>.446</td><td>.224</td><td>5.00</td></tr><tr><td>SAR</td><td>.300</td><td>.217</td><td>.286</td><td>.192</td><td>.105</td><td>.440</td><td>.284</td><td>4.71</td></tr><tr><td>Semantic Entropy</td><td>.311</td><td>.206</td><td>.075</td><td>.007</td><td>.171</td><td>.466</td><td>.220</td><td>5.29</td></tr><tr><td>Factoscope</td><td>.061</td><td>.037</td><td>.084</td><td>.101</td><td>.045</td><td>.420</td><td>.071</td><td>7.00</td></tr><tr><td>SAPLMA</td><td>.241</td><td>.075</td><td>.079</td><td>.129</td><td>.009</td><td>.091</td><td>-.097</td><td>7.00</td></tr><tr><td>HUQ-SATRMD</td><td>.413</td><td>.229</td><td>.293</td><td>.355</td><td>.283</td><td>.644</td><td>.770</td><td>1.71</td></tr><tr><td>SATRMD+MSP</td><td>.207</td><td>.314</td><td>.304</td><td>.304</td><td>.122</td><td>.598</td><td>.681</td><td>3.00</td></tr></table>

Table 3: Out-of-domain generalization. PRR for Llama 8b v3.1 for selective generation tasks for the considered sequence-level methods in the out-of-domain setting. Warmer color indicates better results.

Out-of-domain generalization. Table 3 presents a comparison of various sequence-level methods for the selective generation task for the Llama 8b v3.1 model in the out-of-domain settings. We train and evaluate supervised methods using a leaveone-out approach: train on all datasets except one, which is left for testing. For each evaluation dataset, the training set is composed of 400 instances sampled from each of the remaining datasets. We use the negative AlignScore generation quality metric as a target for all considered datasets in this setting.

The performance of supervised methods in the out-of-domain setting shows a significant decline compared to the in-domain setting. Despite this, HUQ-SATRMD achieves the best results according to the mean rank, outperforming unsupervised state-of-the-art methods across the majority of datasets and metrics, except MSP on SamSum and MedQUAD in terms of PRR-ROUGE-L. Notably, when testing on the MMLU dataset, the training data consists of texts from summarization tasks and free-form QA, which differ significantly from the multiple-choice QA format used in MMLU. Nevertheless, the strong performance on MMLU demonstrates the potential of our supervised method HUQ-SATRMD for broad generalization.

Other supervised methods, including SATRMD+MSP and the baselines, show significantly poorer results in the out-of-domain setting. SATRMD+MSP underperforms MSP on several datasets, including SamSum, MedQUAD, and TruthfulQA. SAPLMA and Factoscope are not able to provide meaningful uncertainty scores, lagging significantly behind unsupervised UQ methods.

Fact-checking results. Table 4 presents a comparison of various claim-level methods for the factchecking task using the Mistral 7b v0.1 Instruct model. The baseline supervised method SAPLMA performs similarly to a random choice. Our method SATRMD provides meaningful uncertainty scores slightly outperforming Maximum Claim Probability (MCP). We note that CCP, like MCP, is also based on the probabilities derived from the model output but demonstrates better performance than MCP. Consequently, we combine CCP with SATRMD to provide more effective claim-level fact-checking. The results demonstrate that HUQ-SATRMD achieves the best results in terms of ROC-AUC, outperforming CCP by 3.4%, while in terms of PR-AUC, SATRMD+CPP is the best, outperforming CCP by 2.6%. These results demonstrate that the proposed SATRMD methods are effective not only for sequence-level uncertainty quantification but also for estimating uncertainty on the claim level.

<table><tr><td rowspan="2">UQ Method</td><td colspan="2">Mistral 7b</td></tr><tr><td>ROC-AUC</td><td>PR-AUC</td></tr><tr><td>Maximum Claim Probability</td><td>.620</td><td>.271</td></tr><tr><td>P(True)</td><td>.638</td><td>.276</td></tr><tr><td>CCP</td><td>.716</td><td>.388</td></tr><tr><td>SAPLMA</td><td>.489</td><td>.166</td></tr><tr><td>SATRMD</td><td>.647</td><td>.275</td></tr><tr><td>HUQ-SATRMD</td><td>.750</td><td>410</td></tr><tr><td>SATRMD+CCP</td><td>.739</td><td>.414</td></tr></table>

Table 4: Results in fact-checking. ROC-AUC and PR-AUC for the Mistral 7b v0.1 Instruct model for fact-checking for the considered claim-level methods. Warmer color indicates better results.

Impact of training data size. Figure 5 in Appendix A.3 illustrates the dependency of the performance of supervised UQ methods on the size of the training data. As expected, the optimal results on all datasets are achieved when the maximum number of training instances is used. Nevertheless, for all datasets, except SamSum and MedQUAD, the results obtained with 200-500 training instances are only slightly lower than with 2,000-5,000 instances. Furthermore, even with fewer than 200 training instances for MedQUAD, GSM8k, and MMLU, HUQ and SATRMD+MSP are able to substantially outperform the MSP method. These results demonstrate the robustness of the proposed methods with respect to the size of the training dataset.

Impact of the correctness threshold. Figure 3 presents the dependence of the performance of the SATRMD+MSP and HUQ-SATRMD methods on the correctness threshold used for the embedding selection for computing the centroid and covariance matrix for MD.

The results demonstrate that the proposed methods are generally not sensitive to the correctness threshold and consistently show high performance. However, for the MedQUAD dataset, we can see the results with a threshold of 0.3 are significantly better than those with other thresholds. Specifically, lower thresholds (e.g., 0.1) result in selecting the embeddings corresponding to incorrect instances, while higher thresholds (e.g., 0.8) exclude some embeddings associated with correct instances. Both scenarios result in suboptimal estimation of the centroid and covariance matrix, leading to a slight degradation in overall performance.

![](images/cb04cbe5bdaffb6ac13c7f3a170d6f62b13305969d0d7c5466e5a9fc79e4e9b7.jpg)

![](images/3c819a44e30a841aaaa698b2c6cbe4a1a5fba9c500fd09622c10f739799816ce.jpg)

![](images/599665ab7d04fbe548c44e98cb2460df63527489c26a411d80362ee5aa1f3d1f.jpg)

![](images/2951613b79d09d8c657244c9b1979c8929c2f7cb79f849c07aea1060d6565b25.jpg)

![](images/7d1699c4fc7087e3e742247d0777eddfd6c1ee06df7000b210558ccbe018c463.jpg)

![](images/baae8cdc5cd9570457182960e23b146c14a98a3fc634f9515b4e2e7313268fc6.jpg)

Figure 3: Dependency of PRR of the SATRMD+MSP and HUQ-SATRMD methods on the correctness threshold for the embedding selection for the centroid and covariance matrix for MD for the Llama 8b v3.1 model. Higher values indicate better results.  
![](images/d157cf46e033b76942f5c6636863200e70061c79496878943a2cf0291ecb752b.jpg)

![](images/b70e9a370898d4675b24c62325e8daa08caa374b7306ff29dd4c6238413f2426.jpg)

![](images/da053178f93245b5fdd52b811536de6fc7680609cecd754b1da060f2486bd1e1.jpg)

![](images/117e25223e5f1a0d77d650e2894f3df05a7b02af393005f8c1102a2ac9cc3b52.jpg)

![](images/e864eb36010e50f526f688e023a23111cca4c5ccdf06f55deccb3ff69255ff42.jpg)

![](images/f5a3b1202a74b5c2522ef137670a48a3eae6a42dfe4e85e0aaa25eaf7ea8b75e.jpg)  
Figure 4: Dependency of PRR of the SATRMD+MSP and HUQ-SATRMD methods on the number of the PCA components for the features of linear regression for the Llama 8b v3.1 model. Higher values indicate better results.

Impact of the number of PCA components. Figure 4 illustrates the impact of the number of PCA components used for the features of linear regression on the performance of SATRMD+MSP and HUQ-SATRMD methods. The best performance is achieved with 10 or 20 components for most datasets. Only for CNN and TruthfulQA, using just 2 components yields slightly better results than using more. Overall, these results indicate that our choice of 10 components is well-balanced on average across multiple datasets. We also observe that results with more than five PCA components remain stable across all datasets, showing minimal variation. Therefore, methods based on RMD are not sensitive to the precise choice of the number of PCA components.

Computational efficiency. Table 7 in Appendix C summarizes the average runtime per instance for each UQ method, along with the percentage overhead compared to standard LLM inference. State-of-the-art UQ methods that require sampling from the LLM multiple times (Semantic Entropy, SAR, Lexical Similarity) introduce a huge computational overhead (315-700%). In contrast, the proposed methods HUQ-SATRMD and SATRMD+MSP introduce minimal overhead (5.3- 7.6%), which makes them much more practical.

## 6 Conclusion

We have introduced a series of new supervised UQ methods based on layer-wise features derived from the Mahalanobis distance. We show that calculating MD over token-level embeddings yields much better results than previous attempts that leverage sequence-level embeddings. Training a linear regression on top of the layer-wise scores allows us to produce even better uncertainty scores and outperform the state-of-the-art supervised and unsupervised UQ methods in selective classification across eleven datasets and in claim-level fact-checking. We also show that the proposed methods are computationally efficient and have the potential for generalization, which makes them useful in real-world LLM-based applications.

In future work, we aim to improve the generalization capabilities of the supervised UQ methods on out-of-domain data by investigating new features and a more robust training pipeline.

## Limitations

Our approach is supervised, which means that its performance depends on the quality and size of the data available for supervision. We evaluated the robustness of the approach to dataset variation, which demonstrates that the method does not significantly degrade its quality compared to the target dataset. Nevertheless, we observe certain performance drops; thus, the resulting UQ method should be used with care beyond the supervision domain.

We did not test our method on very large LLMs, such as LLaMA 3 70b, as we were limited to using 7-9b models due to constraints in our computational resources.

## Ethical Considerations

In our study, we focused on open-source LLMs and datasets that are not designed to produce harmful content. However, LLMs can still generate potentially harmful texts that may impact various groups. Uncertainty quantification techniques offer a way to enhance the reliability of neural networks and can even be used to detect harmful outputs, though this is not our focus.

Although our proposed method shows substantial performance improvements, it may sometimes incorrectly flag safe and accurate generated text as having high uncertainty. While we explicitly benchmarked the method on robustness to the task change, its applicability across various tasks remains limited.

## Acknowledgments

We thank the anonymous reviewers for their valuable suggestions, which significantly contributed to improving this paper.

## References

Asma Ben Abacha and Dina Demner-Fushman. 2019. A question-entailment approach to question answering. BMC Bioinform., 20(1):511:1–511:23.

Amos Azaria and Tom Mitchell. 2023. The internal state of an LLM knows when it’s lying. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2023, pages 967–976, Singapore. Association for Computational Linguistics.

Joris Baan, Nico Daheim, Evgenia Ilia, Dennis Ulmer, Haau-Sing Li, Raquel Fernández, Barbara Plank, Rico Sennrich, Chrysoula Zerva, and Wilker

Aziz. 2023. Uncertainty in natural language generation: From theory to applications. arXiv preprint arXiv:2307.15703.

Sky CH-Wang, Benjamin Van Durme, Jason Eisner, and Chris Kedzie. 2024. Do androids know they’re only dreaming of electric sheep? In Findings ofthe Associationfor Computational Linguistics ACL 2024, pages 4401–4420, Bangkok, Thailand and virtual meeting. Association for Computational Linguistics.

Jireh Chan, Steven Leow, Khean Bea, Wai Khuen Cheng, Seuk Wai Phoong, Zeng-Wei Hong, and Yen-Lin Chen. 2022. Mitigating the multicollinearity problem and its machine learning approach: A review. Mathematics, 10:1283.

Chao Chen, Kai Liu, Ze Chen, Yi Gu, Yue Wu, Mingyuan Tao, Zhihang Fu, and Jieping Ye. 2024. INSIDE: llms’ internal states retain the power of hallucination detection. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net.

Yuyan Chen, Qiang Fu, Yichen Yuan, Zhihao Wen, Ge Fan, Dayiheng Liu, Dongmei Zhang, Zhixu Li, and Yanghua Xiao. 2023. Hallucination detection: Robustly discerning reliable answers in large language models. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management, pages 245–255.

Julius Cheng and Andreas Vlachos. 2024. Measuring uncertainty in neural machine translation with similarity-sensitive entropy. In Proceedings of the 18th Conference ofthe European Chapter ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 2115–2128, St. Julian’s, Malta. Association for Computational Linguistics.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Maxime Darrin, Pablo Piantanida, and Pierre Colombo. 2023. RainProof: An umbrella to shield text generator from out-of-distribution data. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 5831–5857, Singapore. Association for Computational Linguistics.

Jinhao Duan, Hao Cheng, Shiqi Wang, Alex Zavalny, Chenan Wang, Renjing Xu, Bhavya Kailkhura, and Kaidi Xu. 2024. Shifting attention to relevance: Towards the predictive uncertainty quantification of free-form large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5050–5063, Bangkok, Thailand. Association for Computational Linguistics.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman,

Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurélien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Rozière, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Graeme Nail, Grégoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel M. Kloumann, Ishan Misra, Ivan Evtimov, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, and et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Nouha Dziri, Sivan Milton, Mo Yu, Osmar Zaiane, and Siva Reddy. 2022. On the origin of hallucinations in conversational models: Is it the datasets or the models? In Proceedings of the 2022 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 5271–5285, Seattle, United States. Association for Computational Linguistics.

Ekaterina Fadeeva, Aleksandr Rubashevskii, Artem Shelmanov, Sergey Petrakov, Haonan Li, Hamdy Mubarak, Evgenii Tsymbalov, Gleb Kuzmin, Alexander Panchenko, Timothy Baldwin, Preslav Nakov, and Maxim Panov. 2024. Fact-checking the output of large language models via token-level uncertainty quantification. In Findings of the Association for Computational Linguistics: ACL 2024. Association for Computational Linguistics.

Ekaterina Fadeeva, Roman Vashurin, Akim Tsvigun, Artem Vazhentsev, Sergey Petrakov, Kirill Fedyanin, Daniil Vasilev, Elizaveta Goncharova, Alexander Panchenko, Maxim Panov, Timothy Baldwin, and Artem Shelmanov. 2023. LM-Polygraph: Uncertainty estimation for language models. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 446–461.

Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, and Yarin Gal. 2024. Detecting hallucinations in large language models using semantic entropy. Nature, 630(8017):625–630.

Shangbin Feng, Weijia Shi, Yike Wang, Wenxuan Ding, Vidhisha Balachandran, and Yulia Tsvetkov. 2024. Don’t hallucinate, abstain: Identifying LLM knowledge gaps via multi-LLM collaboration. In Proceedings ofthe 62nd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 14664–14690, Bangkok, Thailand. Association for Computational Linguistics.

Marina Fomicheva, Shuo Sun, Lisa Yankovskaya, Frédéric Blain, Francisco Guzmán, Mark Fishel, Nikolaos Aletras, Vishrav Chaudhary, and Lucia Specia. 2020. Unsupervised quality estimation for neural machine translation. Transactions ofthe Association for Computational Linguistics, 8:539–555.

Yarin Gal and Zoubin Ghahramani. 2016. Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. In Proceedings of The 33rd International Conference on Machine Learning, volume 48 of Proceedings of Machine Learning Research, pages 1050–1059, New York, New York, USA. PMLR.

Jiahui Geng, Fengyu Cai, Yuxia Wang, Heinz Koeppl, Preslav Nakov, and Iryna Gurevych. 2024. A survey of confidence estimation and calibration in large language models. In Proceedings ofthe 2024 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 6577–6595, Mexico City, Mexico. Association for Computational Linguistics.

Bogdan Gliwa, Iwona Mochol, Maciej Biesek, and Aleksander Wawer. 2019. SAMSum corpus: A humanannotated dialogue dataset for abstractive summarization. In Proceedings of the 2nd Workshop on New Frontiers in Summarization, pages 70–79, Hong Kong, China. Association for Computational Linguistics.

Jianfeng He, Linlin Yu, Shuo Lei, Chang-Tien Lu, and Feng Chen. 2024a. Uncertainty estimation on sequential labeling via uncertainty transmission. In Findings ofthe Associationfor Computational Linguistics: NAACL 2024, pages 2823–2835, Mexico City, Mexico. Association for Computational Linguistics.

Jianfeng He, Xuchao Zhang, Shuo Lei, Zhiqian Chen, Fanglan Chen, Abdulaziz Alhamadani, Bei Xiao, and Chang-Tien Lu. 2020. Towards more accurate uncertainty estimation in text classification. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020, pages 8362–8372. Association for Computational Linguistics.

Jinwen He, Yujia Gong, Zijin Lin, Cheng’an Wei, Yue Zhao, and Kai Chen. 2024b. LLM factoscope: Uncovering LLMs’ factual discernment through measuring inner states. In Findings of the Association for Computational Linguistics ACL 2024, pages 10218– 10230, Bangkok, Thailand and virtual meeting. Association for Computational Linguistics.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021. Measuring massive multitask language understanding. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de Las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023. Mistral 7b. arXiv preprint arXiv:2310.06825.

Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William Cohen, and Xinghua Lu. 2019. PubMedQA: A dataset for biomedical research question answering. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 2567– 2577, Hong Kong, China. Association for Computational Linguistics.

Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. 2017. TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. In Proceedings of the 55th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 1601–1611, Vancouver, Canada. Association for Computational Linguistics.

Nikita Kotelevskii, Aleksandr Artemenkov, Kirill Fedyanin, Fedor Noskov, Alexander Fishkov, Artem Shelmanov, Artem Vazhentsev, Aleksandr Petiushko, and Maxim Panov. 2022. Nonparametric uncertainty quantification for single deterministic neural network. In Advances in Neural Information Processing Systems.

Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. 2023. Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023.

Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. 2017. Simple and scalable predictive uncertainty estimation using deep ensembles. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc.

Kimin Lee, Kibok Lee, Honglak Lee, and Jinwoo Shin. 2018. A simple unified framework for detecting outof-distribution samples and adversarial attacks. In Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montréal, Canada, volume 31, pages 7167– 7177.

Stephanie Lin, Jacob Hilton, and Owain Evans. 2022. TruthfulQA: Measuring how models mimic human

falsehoods. In Proceedings ofthe 60th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 3214–3252, Dublin, Ireland. Association for Computational Linguistics.

Zhen Lin, Shubhendu Trivedi, and Jimeng Sun. 2023. Generating with confidence: Uncertainty quantification for black-box large language models. Transactions on Machine Learning Research.

Jeremiah Liu, Zi Lin, Shreyas Padhy, Dustin Tran, Tania Bedrax Weiss, and Balaji Lakshminarayanan. 2020. Simple and principled uncertainty estimation with deterministic deep learning via distance awareness. In Advances in Neural Information Processing Systems, volume 33, pages 7498–7512. Curran Associates, Inc.

Andrey Malinin and Mark J. F. Gales. 2021. Uncertainty estimation in autoregressive structured prediction. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net.

Potsawee Manakul, Adian Liusie, and Mark Gales. 2023. SelfCheckGPT: Zero-resource black-box hallucination detection for generative large language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 9004–9017.

Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Koh, Mohit Iyyer, Luke Zettlemoyer, and Hannaneh Hajishirzi. 2023. FActScore: Fine-grained atomic evaluation of factual precision in long form text generation. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 12076–12100.

Shashi Narayan, Shay B. Cohen, and Mirella Lapata. 2018. Don’t give me the details, just the summary! topic-aware convolutional neural networks for extreme summarization. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, Brussels, Belgium, October 31 - November 4, 2018, pages 1797–1807. Association for Computational Linguistics.

Alexander Nikitin, Jannik Kossen, Yarin Gal, and Pekka Marttinen. 2024. Kernel language entropy: Fine-grained uncertainty quantification for llms from semantic similarities. arXiv preprint arXiv:2405.20003.

OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, Red Avila, Igor Babuschkin, Suchir Balaji, Valerie Balcom, Paul Baltescu, Haiming Bao, Mohammad Bavarian, Jeff Belgum, Irwan Bello, Jake Berdine, Gabriel Bernadett-Shapiro, Christopher Berner, Lenny Bogdonoff, Oleg Boiko, Madelaine Boyd, Anna-Luisa Brakman, Greg Brockman, Tim Brooks, Miles Brundage, Kevin Button, Trevor Cai, Rosie Campbell, Andrew Cann, Brittany

Carey, Chelsea Carlson, Rory Carmichael, Brooke Chan, Che Chang, Fotis Chantzis, Derek Chen, Sully Chen, Ruby Chen, Jason Chen, Mark Chen, Ben Chess, Chester Cho, Casey Chu, Hyung Won Chung, Dave Cummings, Jeremiah Currier, Yunxing Dai, Cory Decareaux, Thomas Degry, Noah Deutsch, Damien Deville, Arka Dhar, David Dohan, Steve Dowling, Sheila Dunning, Adrien Ecoffet, Atty Eleti, Tyna Eloundou, David Farhi, Liam Fedus, Niko Felix, Simón Posada Fishman, Juston Forte, Isabella Fulford, Leo Gao, Elie Georges, Christian Gibson, Vik Goel, Tarun Gogineni, Gabriel Goh, Rapha Gontijo-Lopes, Jonathan Gordon, Morgan Grafstein, Scott Gray, Ryan Greene, Joshua Gross, Shixiang Shane Gu, Yufei Guo, Chris Hallacy, Jesse Han, Jeff Harris, Yuchen He, Mike Heaton, Johannes Heidecke, Chris Hesse, Alan Hickey, Wade Hickey, Peter Hoeschele, Brandon Houghton, Kenny Hsu, Shengli Hu, Xin Hu, Joost Huizinga, Shantanu Jain, Shawn Jain, Joanne Jang, Angela Jiang, Roger Jiang, Haozhun Jin, Denny Jin, Shino Jomoto, Billie Jonn, Heewoo Jun, Tomer Kaftan, Łukasz Kaiser, Ali Kamali, Ingmar Kanitscheider, Nitish Shirish Keskar, Tabarak Khan, Logan Kilpatrick, Jong Wook Kim, Christina Kim, Yongjik Kim, Jan Hendrik Kirchner, Jamie Kiros, Matt Knight, Daniel Kokotajlo, Łukasz Kondraciuk, Andrew Kondrich, Aris Konstantinidis, Kyle Kosic, Gretchen Krueger, Vishal Kuo, Michael Lampe, Ikai Lan, Teddy Lee, Jan Leike, Jade Leung, Daniel Levy, Chak Ming Li, Rachel Lim, Molly Lin, Stephanie Lin, Mateusz Litwin, Theresa Lopez, Ryan Lowe, Patricia Lue, Anna Makanju, Kim Malfacini, Sam Manning, Todor Markov, Yaniv Markovski, Bianca Martin, Katie Mayer, Andrew Mayne, Bob Mc-Grew, Scott Mayer McKinney, Christine McLeavey, Paul McMillan, Jake McNeil, David Medina, Aalok Mehta, Jacob Menick, and Luke Metz et al. 2024. Gpt-4 technical report. Preprint, arXiv:2303.08774.

Alexander Podolskiy, Dmitry Lipin, Andrey Bout, Ekaterina Artemova, and Irina Piontkovskaya. 2021. Revisiting mahalanobis distance for transformer-based out-of-domain detection. In Proceedings ofthe AAAI Conference on Artificial Intelligence, pages 13675– 13682.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. J. Mach. Learn. Res., 21:140:1–140:67.

Siva Reddy, Danqi Chen, and Christopher D. Manning. 2019. CoQA: A conversational question answering challenge. Transactions of the Association for Computational Linguistics, 7:249–266.

Jie Ren, Jiaming Luo, Yao Zhao, Kundan Krishna, Mohammad Saleh, Balaji Lakshminarayanan, and Peter J Liu. 2023. Out-of-distribution detection and selective generation for conditional language models. In The Eleventh International Conference on Learning Representations.

Morgane Rivière, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ramé, Johan Ferret, Peter Liu, Pouya Tafti, Abe Friesen, Michelle Casbon, Sabela Ramos, Ravin Kumar, Charline Le Lan, Sammy Jerome, Anton Tsitsulin, Nino Vieillard, Piotr Stanczyk, Sertan Girgin, Nikola Momchev, Matt Hoffman, Shantanu Thakoor, Jean-Bastien Grill, Behnam Neyshabur, Olivier Bachem, Alanna Walton, Aliaksei Severyn, Alicia Parrish, Aliya Ahmad, Allen Hutchison, Alvin Abdagic, Amanda Carl, Amy Shen, Andy Brock, Andy Coenen, Anthony Laforge, Antonia Paterson, Ben Bastian, Bilal Piot, Bo Wu, Brandon Royal, Charlie Chen, Chintu Kumar, Chris Perry, Chris Welty, Christopher A. Choquette-Choo, Danila Sinopalnikov, David Weinberger, Dimple Vijaykumar, Dominika Rogozinska, Dustin Herbison, Elisa Bandy, Emma Wang, Eric Noland, Erica Moreira, Evan Senter, Evgenii Eltyshev, Francesco Visin, Gabriel Rasskin, Gary Wei, Glenn Cameron, Gus Martins, Hadi Hashemi, Hanna Klimczak-Plucinska, Harleen Batra, Harsh Dhand, Ivan Nardini, Jacinda Mein, Jack Zhou, James Svensson, Jeff Stanway, Jetha Chan, Jin Peng Zhou, Joana Carrasqueira, Joana Iljazi, Jocelyn Becker, Joe Fernandez, Joost van Amersfoort, Josh Gordon, Josh Lipschultz, Josh Newlan, Ju-yeong Ji, Kareem Mohamed, Kartikeya Badola, Kat Black, Katie Millican, Keelin McDonell, Kelvin Nguyen, Kiranbir Sodhia, Kish Greene, Lars Lowe Sjösund, Lauren Usui, Laurent Sifre, Lena Heuermann, Leticia Lago, and Lilly McNealus. 2024. Gemma 2: Improving open language models at a practical size. arXiv preprint arXiv:2408.00118.

Peter J Rousseeuw. 1984. Least median of squares regression. Journal ofthe American statistical association, 79(388):871–880.

Abigail See, Peter J. Liu, and Christopher D. Manning. 2017. Get to the point: Summarization with pointergenerator networks. In Proceedings ofthe 55th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 1073– 1083, Vancouver, Canada. Association for Computational Linguistics.

Artem Shelmanov, Evgenii Tsymbalov, Dmitri Puzyrev, Kirill Fedyanin, Alexander Panchenko, and Maxim Panov. 2021. How certain is your transformer? In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pages 1833–1840, Online. Association for Computational Linguistics.

Noora Shrestha. 2020. Detecting multicollinearity in regression analysis. American Journal of Applied Mathematics and Statistics, 8:39–42.

Evgenii Tsymbalov, Maxim Panov, and Alexander Shapeev. 2018. Dropout-based active learning for regression. In Analysis ofImages, Social Networks and Texts: 7th International Conference, AIST 2018, Moscow, Russia, July 5–7, 2018, pages 247–258.

Joost van Amersfoort, Lewis Smith, Yee Whye Teh, and Yarin Gal. 2020. Uncertainty estimation using a single deep deterministic neural network. In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, volume 119 of Proceedings ofMachine Learning Research, pages 9690–9700. PMLR.

Roman Vashurin, Ekaterina Fadeeva, Artem Vazhentsev, Lyudmila Rvanova, Akim Tsvigun, Daniil Vasilev, Rui Xing, Abdelrahman Boda Sadallah, Kirill Grishchenkov, Sergey Petrakov, Alexander Panchenko, Timothy Baldwin, Preslav Nakov, Maxim Panov, and Artem Shelmanov. 2024. Benchmarking uncertainty quantification methods for large language models with lm-polygraph. arXiv preprint arXiv:2406.15627.

Roman Vashurin, Maiya Goloburda, Preslav Nakov, Artem Shelmanov, and Maxim Panov. 2025. Cocoa: A generalized approach to uncertainty quantification by integrating confidence and consistency of llm outputs. arXiv preprint arXiv:2502.04964.

Artem Vazhentsev, Ekaterina Fadeeva, Rui Xing, Alexander Panchenko, Preslav Nakov, Timothy Baldwin, Maxim Panov, and Artem Shelmanov. 2024. Unconditional truthfulness: Learning conditional dependency for uncertainty quantification of large language models. arXiv preprint arXiv:2408.10692.

Artem Vazhentsev, Gleb Kuzmin, Artem Shelmanov, Akim Tsvigun, Evgenii Tsymbalov, Kirill Fedyanin, Maxim Panov, Alexander Panchenko, Gleb Gusev, Mikhail Burtsev, Manvel Avetisian, and Leonid Zhukov. 2022. Uncertainty estimation of transformer predictions for misclassification detection. In Proceedings ofthe 60th Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 8237–8252, Dublin, Ireland. Association for Computational Linguistics.

Artem Vazhentsev, Gleb Kuzmin, Akim Tsvigun, Alexander Panchenko, Maxim Panov, Mikhail Burtsev, and Artem Shelmanov. 2023a. Hybrid uncertainty quantification for selective text classification in ambiguous tasks. In Proceedings ofthe 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 11659– 11681, Toronto, Canada. Association for Computational Linguistics.

Artem Vazhentsev, Akim Tsvigun, Roman Vashurin, Sergey Petrakov, Daniil Vasilev, Maxim Panov, Alexander Panchenko, and Artem Shelmanov. 2023b. Efficient out-of-domain detection for sequence to sequence models. In Findings of the Association for Computational Linguistics: ACL 2023, pages 1430– 1454, Toronto, Canada. Association for Computational Linguistics.

Yuxia Wang, Daniel Beck, Timothy Baldwin, and Karin Verspoor. 2022. Uncertainty estimation and reduction of pre-trained models for text regression. Transactions ofthe Associationfor Computational Linguistics, 10:680–696.

Johannes Welbl, Nelson F. Liu, and Matt Gardner. 2017. Crowdsourcing multiple choice science questions. In Proceedings ofthe 3rd Workshop on Noisy Usergenerated Text, pages 94–106, Copenhagen, Denmark. Association for Computational Linguistics.

Yijun Xiao and William Yang Wang. 2021. On hallucination and predictive uncertainty in conditional language generation. In Proceedings ofthe 16th Conference ofthe European Chapter ofthe Association for Computational Linguistics: Main Volume, pages 2734–2744, Online. Association for Computational Linguistics.

Ji Xin, Raphael Tang, Yaoliang Yu, and Jimmy Lin. 2021. The art of abstention: Selective prediction and error regularization for natural language processing. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 1040–1051, Online. Association for Computational Linguistics.

KiYoon Yoo, Jangho Kim, Jiho Jang, and Nojun Kwak. 2022. Detection of adversarial examples in text classification: Benchmark and baseline via robust density estimation. In Findings ofthe Associationfor Computational Linguistics: ACL 2022, pages 3656–3672, Dublin, Ireland. Association for Computational Linguistics.

Yuheng Zha, Yichi Yang, Ruichen Li, and Zhiting Hu. 2023. AlignScore: Evaluating factual consistency with a unified alignment function. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 11328–11348, Toronto, Canada. Association for Computational Linguistics.

Xuchao Zhang, Fanglan Chen, Chang-Tien Lu, and Naren Ramakrishnan. 2019. Mitigating uncertainty in document classification. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 3126–3136, Minneapolis, Minnesota. Association for Computational Linguistics.

## A Additional Experimental Results

## A.1 Comparison of Sequence-Level Aggregations

<table><tr><td rowspan="2">UQ Method</td><td colspan="2">XSUM</td><td colspan="2">SamSum</td><td colspan="2">CNN</td><td colspan="2">PubMedQA</td><td colspan="2">MedQUAD</td><td colspan="2">TruthfulQA CoQA</td><td colspan="2">SciQ TriviaQA</td><td colspan="3">GSM8k MMLU Accuracy</td></tr><tr><td>ROUGE-L -.071</td><td>AlignScore .048</td><td>ROUGE-L .298</td><td>AlignScore .304</td><td>ROUGE-L</td><td>AlignScore</td><td>ROUGE-L</td><td>AlignScore</td><td>ROUGE-L -.357</td><td>AlignScore</td><td>AlignScore</td><td>AlignScore .371</td><td>AlignScore .572</td><td>AlignScore .571</td><td>Accuracy</td><td>.657</td><td>Mean Rank 4.25</td></tr><tr><td>SATMD SATRMD</td><td>.444</td><td>.239</td><td>.193</td><td>.282</td><td>.032 .034</td><td>.093 .063</td><td>-.018 .541</td><td>.076 .299</td><td>.610</td><td>-.027 .479</td><td>299</td><td>.326</td><td>.588</td><td>.450</td><td>.682 .550</td><td>.537</td><td>3.94</td></tr><tr><td>HUQ-SATMD</td><td>-.251</td><td>-.055</td><td>391</td><td>332</td><td>.060</td><td>.085</td><td>-.527</td><td>-.225</td><td>-.388</td><td>-.071</td><td>.245</td><td>.525</td><td>.626</td><td>.750</td><td>.677</td><td>.784</td><td>3.75</td></tr><tr><td>HUQ-SATRMD</td><td>441</td><td>.253</td><td>.397</td><td>.317</td><td>.034</td><td>.063</td><td>.515</td><td>.296</td><td>.558</td><td>.496</td><td>.258</td><td>515</td><td>.635</td><td>.750</td><td>.526</td><td>.759</td><td>3.06</td></tr><tr><td>SATMD+MSP SATRMD+MSP</td><td>-.157 .407</td><td>.048 .273</td><td>.316 .328</td><td>.331 .362</td><td>.014 -.048</td><td>.101 .058</td><td>.581 .576</td><td>.297 .304</td><td>.672 .711</td><td>.353 .528</td><td>.171 368</td><td>.495 .475</td><td>.547 .607</td><td>.794 .790</td><td>.690 .669</td><td>.320 .769</td><td>3.44 2.56</td></tr></table>

Table 5: Performance of various versions of the proposed supervised methods. PRR for Gemma 9b v2 model for various tasks for the considered sequence-level aggregation methods. Warmer color indicates better results.

## A.2 Selective Generation Results

<table><tr><td>UQ Method</td><td colspan="2">XSUM ROUGE-L AlignScore</td><td colspan="2">SamSum AlignScore</td><td colspan="2">CNN AlignScore</td><td colspan="2">PubMedQA AlignScore</td><td colspan="2">MedQUAD AlignScore</td><td>| TruthfulQA AlignScore</td><td>CoQA AlignScore</td><td>SciQ AlignScore</td><td>TriviaQA AlignScore</td><td>GSM8k</td><td>MMLU Accuracy</td><td>Mean Rank</td></tr><tr><td></td><td colspan="2">-.345</td><td colspan="2">ROUGE-L .381</td><td colspan="2">ROUGE-L -.025</td><td colspan="2">ROUGE-L -.526</td><td colspan="2">ROUGE-L</td><td colspan="2"></td><td colspan="2"></td><td colspan="2">Accuracy</td><td>8.81</td></tr><tr><td>Maximum Sequence Probability Perplexity</td><td>-.361</td><td>-.139 -.191</td><td>.206</td><td>.246 .298</td><td></td><td>.053 .010</td><td>.557</td><td>-.214 .200</td><td>-.389 .757</td><td>-.071 .321</td><td>.187 .171</td><td>.527 517</td><td>.614 .178</td><td>.772 .779</td><td>.425 .225</td><td>.784 .771</td><td>6.81</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>.078</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DegMat NLI Score Entail.</td><td>.058</td><td>.090</td><td>.169</td><td>.268</td><td>.025</td><td>.104</td><td>.019</td><td>.011</td><td>-.061</td><td>.111</td><td>.201</td><td>.421</td><td>.516</td><td>.759</td><td>.520</td><td>.443</td><td>7.69</td></tr><tr><td>Eccentricity NLI Score Entail.</td><td>-.026</td><td>.065</td><td>.082</td><td>.157</td><td>-.033</td><td>.000</td><td>-.012</td><td>-.021</td><td>-.216</td><td>-.005</td><td>.147</td><td>.459</td><td>.526</td><td>.713</td><td>.504</td><td>.587</td><td>10.25</td></tr><tr><td>Eig ValLaplacian NLI Score Entail.</td><td>.052</td><td>.085</td><td>.174</td><td>.264</td><td>.026</td><td>.106</td><td>.021</td><td>.012</td><td>-.200</td><td>.040</td><td>.198</td><td>.448</td><td>510</td><td>.746</td><td>.493</td><td>.572</td><td>7.88</td></tr><tr><td>Lexical Similarity ROUGE-L</td><td>.122</td><td>.053</td><td>.284</td><td>.267</td><td>.085</td><td>.107</td><td>.047</td><td>.009</td><td>.310</td><td>.066</td><td>.038</td><td>.448</td><td>.495</td><td>.731</td><td>.537</td><td>.581</td><td>7.00</td></tr><tr><td>SAR</td><td>.099</td><td>.054</td><td>.243</td><td>.282</td><td>.055</td><td>.117</td><td>.080</td><td>.003</td><td>.078</td><td>.063</td><td>.139</td><td>.472</td><td>.492</td><td>.776</td><td>.558</td><td>.702</td><td>6.38</td></tr><tr><td>Semantic Entropy</td><td>.099</td><td>.070</td><td>.272</td><td>.261</td><td>.078</td><td>.128</td><td>-.001</td><td>.011</td><td>-.136</td><td>.000</td><td>.015</td><td>.460</td><td>.507</td><td>.727</td><td>.544</td><td>.675</td><td>7.19</td></tr><tr><td>SentenceSAR</td><td>-.043</td><td>-.017</td><td>.186</td><td>.172</td><td>.021</td><td>.076</td><td>-.079</td><td>-.034</td><td>-.263</td><td>-.020</td><td>.151</td><td>512</td><td>.624</td><td>.768</td><td>.324</td><td>.712</td><td>9.25</td></tr><tr><td>Factoscope</td><td>-.023</td><td>-.027</td><td>.105</td><td>.097</td><td>-.062</td><td>.042</td><td>-.044</td><td>-.000</td><td>.334</td><td>.345</td><td>-.069</td><td>.308</td><td>.548</td><td>-.040</td><td>.089</td><td></td><td>11.12</td></tr><tr><td>EigenScore</td><td>.096</td><td>-.006</td><td>.147</td><td>.138</td><td>.040</td><td>.111</td><td>-.050</td><td>-.033</td><td>-.222</td><td>-.017</td><td>.080</td><td>.444</td><td>.494</td><td>.693</td><td>.382</td><td>.425 .444</td><td>10.50</td></tr><tr><td>SAPLMA</td><td>.221</td><td>.194</td><td>.257</td><td>.375</td><td>.079</td><td>.075</td><td>.357</td><td>.221</td><td>.765</td><td>.249</td><td>.460</td><td>.069</td><td>.531</td><td>.667</td><td>.667</td><td>.541</td><td>5.19</td></tr><tr><td>HUQ-SATRMD</td><td>.441</td><td>.253</td><td>.397</td><td>.317</td><td>.034</td><td>.063</td><td>.515</td><td>.296</td><td>.558</td><td>.496</td><td>.258</td><td>.515</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SATRMD+MSP</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>.635</td><td>.750</td><td>.526</td><td>.759</td><td>3.62</td></tr><tr><td></td><td></td><td>.273</td><td></td><td>.362</td><td>-.048</td><td>.058</td><td>.576</td><td>.304</td><td>.711</td><td>.528</td><td>.368</td><td>.475</td><td>.607</td><td></td><td>.669</td><td></td><td>3.31</td></tr><tr><td></td><td></td><td></td><td>.328</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>.790</td><td></td><td></td><td></td></tr><tr><td></td><td>.407</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>.769</td><td></td></tr></table>

Table 6: Main results on selective generation tasks. PRR for Gemma 9b v2 model for various tasks for the considered sequence-level methods. Warmer color indicates better results.

## A.3 Dependency on the Size of the Training Dataset

Figure 5 presents the results when varying the size of the training dataset for the supervised methods. We train the linear regression model on the training datasets of size: 100, 200, 500, 1000, 2000, and additionally on a training dataset of 5000 instances for SciQ and MMLU. Since the TruthfulQA dataset consists of only 817 instances, of which we use 409 instances as the test subset, we train linear regression on the training datasets of sizes: 100, 200, and 408.

![](images/6a55ae4b825317d79c236094d3efd45503999888f138737e682d5e164817e84d.jpg)  
Figure 5: Dependency of PRR of the supervised methods on the size of the training dataset for the Llama 8b v3.1 model. Higher values indicate better results.

## B Hybrid Uncertainty Quantification

We combine the sequence probability $U _ { 1 } ( \tilde { \mathbf { y } } ^ { k } ) = 1 - P ( \tilde { \mathbf { y } } ^ { k } \mid \mathbf { x } ^ { k } )$ with the SATMD and SATRMD methods $U _ { 2 } ( \tilde { \mathbf { y } } ^ { k } ) = U ^ { \mathrm { S } ^ { * } } ( \tilde { \mathbf { y } } ^ { k } )$ . For a given $\mathcal { T } _ { 1 }$ and $\mathcal { T } _ { 2 }$ from Section 4.2, and trained SATRMD method, we fit HUQ hyperparameters on the $\mathcal { T } _ { 2 }$ set.

Following Vazhentsev et al. (2023a), we define the set of in-distribution instances from $\mathcal { T } _ { 2 }$ as follows: $\mathcal { T } _ { \mathrm { I D } } = \{ \mathbf { x } \in \mathcal { T } _ { 2 } \colon U _ { 2 } ( \mathbf { x } ) \leq \delta _ { \operatorname* { m i n } } \}$ . We define the set of arbitrary in-distribution instances $\mathcal { X } _ { \mathrm { I D } } ~ =$ $\{ \mathbf { x } \colon U _ { 2 } ( \mathbf { x } ) \leq \delta _ { \operatorname* { m i n } } \}$ and ambiguous in-distribution instances $\mathcal { X } _ { \mathrm { I D A } } = \{ { \bf x } \in \mathcal { X } _ { \mathrm { I D } } \colon U _ { 1 } ( { \bf x } ) > \delta _ { \mathrm { m a x } } \}$ using $\delta _ { \mathrm { m i n } } , \delta _ { \mathrm { m a x } }$ are thresholds selected on the $\mathcal { T } _ { 2 }$ dataset.

To make different uncertainty scores comparable, we define a ranking function $R ( \mathbf { u } , { \mathfrak { D } } )$ as a rank of u over a sorted dataset D, where $\mathbf { u } _ { 1 } > \mathbf { u } _ { 2 }$ implies $R ( \mathbf { u } _ { 1 } , \mathfrak { D } ) > R ( \mathbf { u } _ { 2 } , \mathfrak { D } )$ . We compute the total uncertainty $U _ { \mathrm { T } } ( \mathbf { x } )$ as a linear combination $U _ { \mathrm { T } } ( \mathbf { x } ) = ( 1 - \alpha ) R ( U _ { 2 } ( \mathbf { x } ) , \mathcal { T } _ { 2 } ) + \alpha R ( U _ { 1 } ( \mathbf { x } ) , \mathcal { T } _ { 2 } )$ , where α is a hyperparameter selected on the $\mathcal { T } _ { 2 }$ dataset. As a result, we define HUQ as follows:

$$
U _ { \mathrm { H U Q } } ( \mathbf { x } ) = \left\{ \begin{array} { l l } { R ( U _ { 1 } ( \mathbf { x } ) , \mathcal { T } _ { \mathrm { I D } } ) , \forall \mathbf { x } \in \mathcal { X } _ { \mathrm { I D } } \setminus \mathcal { X } _ { \mathrm { A I D } } , } \\ { R ( U _ { 1 } ( \mathbf { x } ) , \mathcal { T } _ { 2 } ) , \forall \mathbf { x } \in \mathcal { X } _ { \mathrm { A I D } } , } \\ { U _ { \mathrm { T } } ( \mathbf { x } ) , \forall \mathbf { x } \notin \mathcal { X } _ { \mathrm { I D } } . } \end{array} \right.
$$

## C Computational Efficiency

<table><tr><td>UQ Method</td><td>Runtime per batch</td><td>Overhead</td></tr><tr><td>MSP</td><td>2.10±1.31</td><td></td></tr><tr><td>DegMat NLI Score Entail. Lexical Similarity ROUGE-L</td><td>9.47±3.41</td><td>350%</td></tr><tr><td></td><td>8.69±3.31</td><td>315%</td></tr><tr><td>Semantic Entropy</td><td>9.47±3.41</td><td>350%</td></tr><tr><td>SAR</td><td>16.89±6.85</td><td>700%</td></tr><tr><td>SAPLMA</td><td>2.10±1.31</td><td>0.04%</td></tr><tr><td>Factoscope</td><td>8.45±5.92</td><td>300%</td></tr><tr><td>HUQ-SATRMD</td><td>2.21±1.36</td><td>5.30%</td></tr><tr><td>SATRMD+MSP</td><td>2.26±1.38</td><td>7.61 %</td></tr></table>

Table 7: The evaluation of the inference runtime of UQ methods measured on all test instances from all datasets with predictions from Llama 8b v3.1. The best results are in bold. The second best results are underlined.

## D Computational Resources

All experiments were conducted on a cluster with 6 NVIDIA H100 GPUs. The total time for all conducted experiments for all models across all datasets is approximately 400 GPU hours.

## E Dataset Statistics

<table><tr><td>Task</td><td>Dataset</td><td>N-shot</td><td>Train texts for STMD</td><td>Evaluation texts</td></tr><tr><td rowspan="3">Text Summarization</td><td>CNN/DailyMail</td><td>0</td><td>2,000</td><td>2,000</td></tr><tr><td>XSum</td><td>0</td><td>2,000</td><td>2,000</td></tr><tr><td>SamSum</td><td>0</td><td>2,000</td><td>819</td></tr><tr><td rowspan="4">QA Long answer</td><td>PubMedQA</td><td>0</td><td>2,000</td><td>2,000</td></tr><tr><td>MedQUAD</td><td>5</td><td>2,000</td><td>2,000</td></tr><tr><td>TruthfulQA</td><td>5</td><td>408</td><td>409</td></tr><tr><td>GSM8k</td><td>5</td><td>2,000</td><td>1,319</td></tr><tr><td rowspan="3">QA Short answer</td><td>SciQ</td><td>0</td><td>5,000</td><td>1,000</td></tr><tr><td>CoQA</td><td>all preceding questions</td><td>5,000</td><td>2,000</td></tr><tr><td>TriviaQA</td><td>5</td><td>5,000</td><td>2,000</td></tr><tr><td>MCQA</td><td>MMLU</td><td>5</td><td>5,000</td><td>2,000</td></tr></table>

Table 8: The statistics of the datasets used for evaluation.

## F Inference Hyperparameters

<table><tr><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=2>Task</td><td rowspan=1 colspan=1>Max Input Length</td><td rowspan=1 colspan=1>Generation Length</td><td rowspan=1 colspan=1>Temperature</td><td rowspan=1 colspan=1>Top-p |</td><td rowspan=1 colspan=1>Do Sample</td><td rowspan=1 colspan=1>Beams</td><td rowspan=1 colspan=1>Repetition Penalty</td></tr><tr><td rowspan=2 colspan=1>XSumSamSumCNN</td><td rowspan=1 colspan=2>TS</td><td rowspan=1 colspan=1></td><td rowspan=2 colspan=1>56128128</td><td rowspan=2 colspan=1></td><td rowspan=7 colspan=1>1.0</td><td rowspan=7 colspan=1>False</td><td rowspan=7 colspan=1>1</td><td rowspan=7 colspan=1>1</td></tr><tr><td rowspan=3 colspan=1>PubMedQAMedQUADTruthfulQAGSM8k</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=2 colspan=2>QALong answer</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>128128128</td><td rowspan=1 colspan=1>1.0</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=4 colspan=1></td><td rowspan=3 colspan=1>2562020</td><td rowspan=3 colspan=1></td></tr><tr><td rowspan=3 colspan=1>CoQASciQTriviQAMMLU</td><td rowspan=3 colspan=2>QAShort answerMCQA</td></tr><tr><td rowspan=2 colspan=1></td></tr><tr><td rowspan=1 colspan=1>203</td></tr></table>

Table 9: Text generation hyperparameters for all LLMs used in the experiments.