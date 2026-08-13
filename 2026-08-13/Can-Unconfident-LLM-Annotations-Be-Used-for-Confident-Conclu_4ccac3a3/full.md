# Can Unconfident LLM Annotations Be Used for Confident Conclusions?

Kristina Gligoric´<sup>⋆</sup> Tijana Zrnic<sup>⋆</sup> Cinoo Lee<sup>⋆</sup> Emmanuel J. Candès Dan Jurafsky

Stanford University

{gligoric, tijana.zrnic, cinoolee, candes, jurafsky}@stanford.edu

## Abstract

Large language models (LLMs) have shown high agreement with human raters across a variety of tasks, demonstrating potential to ease the challenges of human data collection. In com putational social science (CSS), researchers are increasingly leveraging LLM annotations to complement slow and expensive human annotations. Still, guidelines for collecting and using LLM annotations, without compromising the validity of downstream conclusions, remain limited. We introduce CONFIDENCE-DRIVEN INFERENCE: a method that combines LLM annotations and LLM confidence indicators to strategically select which human annotations should be collected, with the goal of producing accurate statistical estimates and provably valid confidence intervals while reducing the number of human annotations needed. Our approach comes with safeguards against LLM annotations of poor quality, guaranteeing that the conclusions will be both valid and no less accurate than if we only relied on human annotations. We demonstrate the ef fectiveness of CONFIDENCE-DRIVEN INFER-ENCE over baselines in statistical estimation tasks across three CSS settings—text politeness, stance, and bias—reducing the needed number of human annotations by over 25% in each. Although we use CSS settings for demonstration, CONFIDENCE-DRIVEN INFERENCE can be used to estimate most standard quantities across a broad range of NLP problems.

## 1 Introduction

Large language models (LLMs) have shown strong zero-shot performance across tasks (Kojima et al., 2022), making them a promising tool for generating annotations, particularly when they align closely with human judgments (Ziems et al., 2024). Given this potential, LLM annotations of textual data may be effectively leveraged for statistical estimation, hypothesis testing, and theory development (Park et al., 2023), as well as informing policy decisions (Wei et al., 2024).

Computational Social Science (CSS) research typically focuses not on the annotations themselves but on the social-science insights and conclusions they enable. Thus, understanding how LLM annotations could be used for downstream inferences is crucial in CSS. For example, stance annotations facilitate the study of linguistic differences between media affirming or denying global warming (Luo et al., 2020), while politeness annotations can help examine racial disparities in verbal interactions with law enforcement (Voigt et al., 2017), the relationship between politeness and social power (Danescu-Niculescu-Mizil et al., 2013), and politeness and gender (Newman et al., 2008). Similarly, annotating political leanings in text allows studying the bias of search engines (Robertson et al., 2018), social media (Ribeiro et al., 2018), and political discourse (Sim et al., 2013). Precise statistical estimation, such as prevalence or regression coefficient estimation, is essential for drawing valid conclusions in such studies.

However, whether LLM annotations can be effectively leveraged without compromising the validity of statistical estimation remains uncertain. LLMs exhibit demographic biases (Weidinger et al., 2022; Cheng et al., 2023) and may lack factual accuracy (Gunjal et al., 2024; Li et al., 2023b) and consistency (Sclar et al., 2023; Atreja et al., 2024). Given these limitations, using LLMs without caution may lead to inaccurate conclusions and potential societal harms, especially when such conclusions influence policy or have tangible impacts on peoples’ outcomes (Landers and Behrend, 2023). A potential solution is to rely solely on human annotations; however, human annotations are costly.

Here, we present CONFIDENCE-DRIVEN INFER-ENCE, a method for valid statistical inference using LLM annotations. Given a text corpus and a quantity of interest, our approach builds on active inference (Zrnic and Candès, 2024) to: (1) strategically choose a small number of human annotations, guided by LLM annotations and the LLM’s verbalized confidence scores, and (2) combine the human and LLM annotations into an accurate estimate of the quantity of interest (Fig. 1). The resulting estimate is statistically valid, while reducing reliance on expensive human annotations.

![](images/879d16c2dc258a42164b62a87309fddb4b9e7886ab323ad70a841e0d63681571.jpg)  
Figure 1: Illustration of CONFIDENCE-DRIVEN INFERENCE. Given a text corpus and a quantity of interest θ∗, (1) we collect LLM annotations and indicators of LLM confidence, based on which we strategically choose a small number of human annotations; (2) we then produce an unbiased estimate <sup>ˆ</sup>θ<sup>conf</sup> and a valid confidence interval, allowing valid downstream conclusions.

Our task is statistical estimation of a quantity of interest. We evaluate our approach on five estimation tasks in three CSS settings (politeness, stance, and media bias) in terms of confidence interval coverage and effective sample size, which measures the increase in accuracy due to augmenting human with LLM annotations (Sec. 3.4). We find that naively treating LLM annotations as human data can lead to highly inaccurate estimates and poor coverage. By contrast, our method maintains the target coverage, while outperforming the baselines (defined in Sec. 3.3) in terms of the effective sample size. The latter is enabled partially by the fact that, in all tested settings, the verbalized confidence scores reflect LLM accuracy. Higher confidence scores correspond to higher accuracy with respect to human annotations, allowing for a strategic selection of a smaller number of human annotations.

CONFIDENCE-DRIVEN INFERENCE can be used to estimate a wide range of standard targets (such as regression coefficients, means, and prevalences) across various NLP problems. Our code and data are available at https://github.com/kristinagligoric/confidencedriven-inference.

## 2 Background

## 2.1 LLMs for Data Annotation Tasks

LLMs have shown great potential in handling textannotation tasks without prior task-specific training, sometimes even outperforming crowd workers (Gilardi et al., 2023; Zheng et al., 2023; Liu et al., 2023; Chiang and Lee, 2023; Kim et al., 2023). Beyond NLP, LLMs offer transformative opportunities for any discipline that relies on text as data. Fields such as psychology, political science, sociology, communications, and economics recognize this emerging technology’s potential to enhance simulation-based research (Bail, 2024), and facilitate tasks such as text analysis, concept induction (Lam et al., 2024), and topic modeling (Pham et al., 2024).

However, despite their promise, limited research has explored how to harness the potential of LLMs in ways that are both cost-effective and statistically reliable. Our work addresses this gap.

## 2.2 Collaborative Annotation Paradigms

Much of past work frames human and LLM annotations as competing alternatives, with a focus on determining which is superior (Thapa et al., 2023). More recent work increasingly calls for a collaborative approach that leverages the complementary strengths of both (Allen et al., 1999). These collaborative paradigms aim to balance annotation quality and cost by combining human expertise and LLM efficiency (Li et al., 2023c; Kim et al., 2024).

In the spirit of these collaborative paradigms, our work uses LLM confidence to efficiently and costeffectively allocate annotation tasks, while also ensuring that the statistical inferences derived from the annotated data are valid.

## 2.3 Valid Statistical Inferences in NLP

Statistical inference is vital in NLP research. For example, model evaluation requires determining whether a model performs better than a baseline (Card et al., 2020), which in turn relies on making valid conclusions about whether one is observing meaningful model improvements or noise (Dodge et al., 2019). Chatzi et al. (2024) and Boyeau et al. (2024) leverage prediction-powered inference (Angelopoulos et al., 2023a,b) for valid ranking of LLMs. A similar approach is adopted by Saad-Falcon et al. (2024) to evaluate Retrieval-Augmented Generation (RAG) systems.

Beyond model evaluation, NLP applications involve producing measurements, descriptive statistics, and causal effect estimates (Feder et al., 2022; Card and Smith, 2018). Notably, Keith and O’Connor (2018) introduced the problem of scientifically valid prevalence estimation. They construct Bayesian confidence intervals by proposing a generative model for text documents. We contribute to the existing literature by proposing an entirely model-free approach that is applicable to a broad range of target quantities.

Lastly, Egami et al. (2024) consider the problem of valid statistical inference when combining human and LLM annotations. However, they collect the human annotations for uniformly sampled instances, without adapting to the difficulty of annotation. Given the promise of active learning (Zhang et al., 2023; Margatina et al., 2021), we develop an adaptive approach that samples a limited number of human annotations strategically. At a technical level, our approach builds on active inference (Zrnic and Candès, 2024), which can be seen as a refinement of prediction-powered inference (Angelopoulos et al., $2 0 2 3 \mathrm { a } , \mathrm { b } )$ that uses active data collection for improved efficiency. Furthermore, we make use of power tuning (Angelopoulos et al., 2023b), a technique that ensures that incorporating LLM annotations into the estimation can never be worse than ignoring them completely.

## 3 Methods

## 3.1 Problem Setup

We have a text corpus consisting of n independent and identically distributed (i.i.d.) instances $T _ { 1 } , \ldots , T _ { n }$ . We wish to estimate a quantity of interest $\theta ^ { * }$ , such as the prevalence of political bias in the corpus or the causal effect of using certain linguistic markers on the perceived sentiment. To perform the estimation, we require human annotations $H _ { 1 } , \ldots , H _ { n }$ corresponding to $T _ { 1 } , \ldots , T _ { n }$ For example, $H _ { i }$ might indicate whether $T _ { i }$ contains political bias, or assess the perceived politeness of $T _ { i }$ . In addition to human annotations, we may also have other readily-available information about $T _ { i }$ —covariates $X _ { i }$ —such as the source of $T _ { i }$ or indicators of whether $T _ { i }$ contains certain linguistic markers, computed via a lexicon. Note that $X _ { i }$ is available automatically, without needing human annotation. We use the short-hand notation $T = ( T _ { 1 } , \dots , T _ { n } )$ and define X and H similarly.

The quantity $\theta ^ { * }$ can be estimated via an estimator ${ \hat { \theta } } ( X , H )$ , which we will denote by $\hat { \theta }$ for short. The accuracy of $\hat { \theta }$ improves as the number of samples n increases $( \hat { \theta }$ recovers $\theta ^ { * }$ as n approaches infinity). We assume that $\hat { \theta }$ is an M-estimator (Van der Vaart, 2000), meaning it can be written as

$$
{ \hat { \theta } } = \mathop { \arg \operatorname* { m i n } } _ { \theta } { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \ell _ { \theta } ( X _ { i } , H _ { i } ) ,\tag{1}
$$

for a loss function $\ell _ { \theta }$ that is convex in $\theta .$ Important special cases include the mean label, $\begin{array} { r } { \bar { \theta } = \frac { 1 } { n } \sum _ { i = 1 } ^ { \bar { n } } H _ { i } . } \end{array}$ , and linear regression coefficients, which are pervasive in CSS. Other examples include quantiles, logistic, and other regression coefficients. Notice that in some cases, like calculating the mean, the loss function only depends on $H _ { i }$

Our goal is to produce an estimate of $\theta ^ { * }$ with uncertainty—by providing a confidence interval at a pre-specified level (1  α)—with limited access to human annotations. Specifically, we can only collect $n _ { \mathrm { h u m a n } } \ll n$ annotations (on average). This means that the “ideal estimate” (1) is out of reach.

To supplement the costly human annotations, we assume access to LLM annotations $\hat { H } _ { i }$ for all $n$ instances. However, we make no assumption that the LLM annotations are good: we want to produce a valid confidence interval no matter the quality of the LLM, though we anticipate better gains when their quality is high (i.e., lower mean squared error and a smaller confidence interval).

## 3.2 CONFIDENCE-DRIVEN INFERENCE

We combine LLM annotations with strategically chosen human annotations to produce an unbiased estimate $\hat { \theta } ^ { \mathrm { c o n f } }$ that lends itself to a confidence interval that is both valid and tight around $\theta ^ { * }$ . In particular, in the large-sample limit, the mean of the estimate is exactly $\theta ^ { * }$ , no matter how biased the LLM annotations are.

We first explain how to choose the set of instances to be human-annotated, which is crucial for producing an accurate estimate. We collect a human annotation $H _ { i }$ for instance $T _ { i }$ with probability $\pi _ { i }$ . We let $\xi _ { i } = \mathbf { 1 } \{ H _ { i }$ collected denote the indicator of whether $T _ { i }$ has been human-annotated. Zrnic and Candès (2024) show that the optimal choice of $\pi _ { i }$ is to sample according to the uncertainty of the predicted annotation; roughly speaking, for most estimation problems the optimal rule is

$$
\pi _ { i } ^ { * } \propto \sqrt { \mathbb { E } [ ( \hat { H } _ { i } - H _ { i } ) ^ { 2 } | T _ { i } ] } ,
$$

where $\propto$ hides the normalization required to meet the budget, $\begin{array} { r } { \mathbb { E } [ \sum _ { i = 1 } ^ { n } \xi _ { i } ] = \sum _ { i = 1 } ^ { n } \pi _ { i } ^ { * } = n _ { \mathrm { h u m a n } } . } \end{array}$ Of course, since $H _ { i }$ is unknown, $\pi _ { i } ^ { * }$ is unattainable.

A key idea behind our method is to approximate $\pi _ { i } ^ { * }$ by querying the LLM for verbalized confidence. Since RLHF may cause overconfidence (Geng et al., 2024; Zhou et al., 2024) and miscalibration (Band et al., 2024; Achiam et al., 2023) of the LLM’s conditional token probabilities, verbalized probabilities, i.e., expressions of confidence in token-space, are better-calibrated (Tian et al., 2023). Therefore, to collect confidence scores, we adopt the verbalized two-stage prompting approach introduced by Tian et al. (2023), where the model is first asked to provide an answer via zero-shooting and afterward asked to assign a probability to the correctness of the answer. This gives us a confidence score $C _ { i } \in [ 0 , 1 ]$ for each instance $T _ { i }$ . In our applications, we find that the verbalized confidence scores are calibrated (Fig. 3 (right)), meaning that higher confidence scores correspond to higher accuracy with respect to human annotations.

As we collect human annotations, we use $\{ ( C _ { j } , ( \hat { H } _ { j } - H _ { j } ) ^ { 2 } ) \} _ { j < i , \xi _ { j } = 1 }$ as feature–label pairs to train a black-box predictor $\widehat { \mathsf { e r r } _ { i } }$ . In other words, dwe train a model to predict the LLM error from its confidence. Finally, we set

$$
\pi _ { i } \propto \sqrt { \widehat { \mathsf { e r r } } _ { i } ( C _ { i } ) } ,
$$

normalized so that $\begin{array} { r c l } { \mathbb { E } [ \sum _ { i = 1 } ^ { n } \xi _ { i } ] } & { = } & { \sum _ { i = 1 } ^ { n } \pi _ { i } \ = } \end{array}$ $n _ { \mathrm { h u m a n } } .$ . In practice we do not fine-tune $\widehat { \mathsf { e r r } } _ { i }$ at devery step i, but we do so periodically, after reasonably large batches of data (say, every 50 or 100 data points). See App. A.3 for further details behind the sampling and Table 2 for prompt texts.

After we have collected the human annotations according to π<sub>i</sub>, building on active inference (Zrnic and Candès, 2024) we compute a confidence-driven

estimate of $\theta ^ { * }$ :

$$
\hat { \theta } ^ { \mathrm { c o n f } } = \underset { \theta } { \arg \operatorname* { m i n } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left( \lambda \hat { \ell } _ { \theta , i } + ( \ell _ { \theta , i } - \lambda \hat { \ell } _ { \theta , i } ) \frac { \xi _ { i } } { \pi _ { i } } \right) ,\tag{2}
$$

where we denote $\ell _ { \theta , i } ~ = ~ \ell _ { \theta } ( X _ { i } , H _ { i } )$ and $\hat { \ell } _ { \boldsymbol { \theta } , i } \ =$ $\ell _ { \theta } ( X _ { i } , \hat { H } _ { i } )$ , and $\lambda \in [ 0 , 1 ]$ is a carefully chosen tuning parameter. Notice that every summand in (2) is in expectation over $\xi _ { i }$ equal to $\ell _ { \theta } ( X _ { i } , H _ { i } )$ , and thus the loss (2) is on average equal to “ideal” loss (1). This allows showing that, in the limit, $\hat { \theta } ^ { \mathrm { c o n f } }$ is on average exactly equal to $\theta ^ { * }$ , no matter the bias in the LLM annotations. To give one example, if we want to estimate the mean of $H _ { i } , \hat { \theta } ^ { \mathrm { c o n f } }$ reduces to

$$
\hat { \theta } ^ { \mathrm { c o n f } } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left( \lambda \hat { H } _ { i } + ( H _ { i } - \lambda \hat { H } _ { i } ) \frac { \xi _ { i } } { \pi _ { i } } \right) .
$$

Notice that $\mathbb { E } [ \hat { \theta } ^ { \mathrm { c o n f } } ] = \mathbb { E } [ H _ { i } ] = \theta ^ { * }$ . The parameter λ is called a power-tuning parameter (Angelopoulos et al., 2023b), and it interpolates between ignoring the LLM annotations $( \lambda = 0 )$ and utilizing them fully $( \lambda = 1 )$ . We set λ optimally, so that the mean squared error (MSE) of $\hat { \theta } ^ { \mathrm { c o n f } }$ is minimized over λ. This means that, given any sampling rule $\pi _ { i } .$ , the confidence-driven estimator can never be hurt by leveraging erroneous LLM annotations or miscalibrated confidence scores. The estimator is at least as good as when $\lambda = 0$ . Details behind the optimization of λ are in App. A.2.

Finally, applying the theoretical guarantees of $Z \mathbf { r } -$ nic and Candès (2024), we form a valid confidence interval at level $1 - \alpha$ as

$$
C _ { 1 - \alpha } = ( \hat { \theta } ^ { \mathrm { c o n f } } \pm z _ { 1 - \alpha / 2 } \hat { \sigma } _ { \mathrm { s e } } ) ,
$$

where $z _ { 1 - \alpha / 2 }$ is the $1 - \alpha / 2$ quantile of the standard normal distribution and $\hat { \sigma } _ { \mathrm { s e } }$ is a standard error estimate that has a closed form, stated in App. A.1.

## 3.3 Baselines

Human + LLM (non-adaptive). The first baseline incorporates LLM annotations but does not adapt to the per-instance confidence or accuracy of the LLM—it equally trusts all LLM annotations. In particular, this baseline is a special case of $\hat { \theta } ^ { \mathrm { c o n f } }$ with $\lambda = 1$ and uniform sampling probabilities $\begin{array} { r } { \pi _ { i } ~ = ~ \frac { n _ { \mathrm { h u m a n } } } { n } } \end{array}$ . This is the method evaluated and studied by Egami et al. (2024).

Human only. The second baseline ignores LLM annotations and simply applies the standard estimator to human annotations. It collects each human annotation with equal probability, $\frac { n _ { \mathrm { h u m a n } } } { n }$ , so that $n _ { \mathrm { h u m a n } }$ annotations are collected on average. This is the “classical” approach, and it can be thought of as erring on the side of caution and ignoring potentially biased LLM outputs. Since the baseline only collects human annotations, it allows forming a valid confidence interval via classical statistics. This approach is equivalent to $\hat { \theta } ^ { \mathrm { c o n f } }$ with $\lambda = 0$

LLM only. Finally, we consider the naive baseline which treats LLM annotations as human annotations, applying the standard estimator to those annotations and naively forming a confidence interval. This baseline does not suffer from a budget constraint, since LLM annotations are assumed to be cheap and available for all n instances, but it may be biased.

## 3.4 Evaluation Metrics

We evaluate our approach and the baselines in terms of effective sample size and coverage. The effective sample size measures the increase in accuracy achieved by incorporating LLM annotations alongside human annotations. This is akin to getting more value out of each human annotation. For instance, if one has only 100 human annotations but combines them effectively with a larger pool of LLM annotations, the resulting accuracy could be comparable to having 150 human annotations. The latter metric, coverage, evaluates the statistical validity of the approaches by capturing how often the true value $\theta ^ { * }$ falls within the produced confidence interval. In the following we elaborate on the two metrics, deferring further details behind their computation to $\mathrm { { A p p . A . 4 } }$

Effective sample size. Given an estimate <sup>ˆ</sup>θ<sup>method</sup> produced by a method, we define the effective sample size as the hypothetical value $n _ { \mathrm { e f f e c t i v e } }$ such that $\mathrm { M S E } ( \bar { \theta } ^ { \mathrm { m e t h o d } } ) = \mathrm { M S E } ( \hat { \theta } _ { n _ { \mathrm { e f f e c t i v e } } } ^ { \mathrm { h u m a n } } )$ , where $\hat { \theta } _ { n _ { \mathrm { e f f e c t i v e } } } ^ { \mathrm { h u m a n } }$ is obtained via the human-only approach with $n _ { \mathrm { e f f e c t i v e } }$ annotations. In other words, <sup>ˆ</sup>θ<sup>method</sup> is as accurate as the “classical” estimate with $n _ { \mathrm { e f f e c t i v e } }$ human annotations. An equivalent definition says that $n _ { \mathrm { e f f e c t i v e } }$ is the sample size for which the confidence interval around $\hat { \theta } ^ { \mathrm { { m e t h o d } } }$ is of equal width as the classical confidence interval around $\hat { \theta } _ { n _ { \mathrm { e f f e c t i v e } } } ^ { \mathrm { h u m a n } }$ . We thus have that $n _ { \mathrm { e f f e c t i v e } } \mathrm { ~ - ~ } n$ human is the benefit (if positive) or harm (if negative) of using LLM annotations. We also report the gain in effective sample size, defined as (n<sub>efective</sub> $n _ { \mathrm { h u m a n } } ) / n _ { \mathrm { h u m a n } } \cdot 1 0 0 \%$ . The effective sample size of the human-only approach is always $n _ { \mathrm { h u m a n } }$ . We only report the effective sample size for approaches that use human annotations, i.e. all but LLM only, because the effective sample size measures the increase in value of the human annotations.

Coverage. Coverage is defined as the rate at which the confidence intervals produced by each method cover $\theta ^ { * }$ . Since $\theta ^ { * }$ is an ideal estimate that would require infinite data, we cannot know $\theta ^ { * }$ exactly in our applications. Instead, as a proxy, we compute coverage with respect to the estimate (1) on the full dataset. We compute the intervals with a target coverage rate of 90%. Note that, following the theory of Zrnic and Candès (2024), the coverage of our method is provably equal to 90%, and the same is true of the other two statistically valid baselines (our numbers will be slightly upward biased due to the fact that we use a proxy for $\theta ^ { * } )$ . With this in mind, the main purpose of reporting coverage is to evaluate the performance of the LLM only approach; for all other methods, we show coverage as a proof of concept.

## 4 Results

We evaluate our approach on a set of CSS problems that rely on statistical estimation. We aim to include settings that (1) allow addressing important downstream social-science questions, (2) rely on a human-labeled corpus of text instances (possibly with relevant additional covariates), and (3) have a publicly available dataset. We selected three settings that meet these criteria—politeness, stance, and political bias. For stance and politeness, we leverage publicly available datasets and the corresponding human annotations in their entirety. Given the large size, for political leaning, we randomly sample a smaller subset of texts.

## 4.1 Estimation tasks

Politeness. Texts from online requests posted on Stack Exchange and Wikipedia $( n = 5 , 4 8 0 )$ can be seen as polite or impolite. Politeness annotations help understand how linguistic devices impact perceived politeness (Danescu-Niculescu-Mizil et al., 2013). In this estimation task, $\theta ^ { * }$ corresponds to the logistic regression coefficient $\beta _ { \mathrm { h e d g e } }$ measuring the impact of a linguistic feature such as hedging on the perceived politeness, logit $\begin{array} { r l } { ( P ( H _ { \mathrm { p o l i t e } } ~ = ~ } & { { } 1 | X _ { \mathrm { h e d g e } } ) ) } \end{array}$ = β<sub>0</sub> + $\beta _ { \mathrm { h e d g e } } X _ { \mathrm { h e d g e } }$ , where $X _ { \mathrm { h e d g e } } ~ = ~ 1$ indicates the presence of the hedge marker and $H _ { \mathrm { p o l i t e } } = 1$ indicates annotation as polite. We similarly estimate $\beta _ { \mathrm { 1 p p } }$ , the impact of the use of the first person plural pronouns on the perceived politeness.

![](images/97208646ae75686274d6631c451c646468d609495bd8784c110da4a29e757f68.jpg)  
Figure 2: Confidence intervals, effective sample size, and coverage. Rows correspond to different estimation tasks. The first column shows the confidence intervals in five random trials. The vertical dashed line corresponds to the estimate produced on the full dataset. A method is valid if its confidence interval includes this estimate (in about 90% of the trials), and tighter intervals around $\theta ^ { * }$ indicates better performance. The second and third columns display the effective sample size $n _ { \mathrm { e f f e c t i v e } }$ and coverage, respectively, for different values of the human annotation budget $n _ { \mathrm { h u m a n } }$ . Results are estimated over 100 trials.

<table><tr><td rowspan="2">Estimation task</td><td rowspan="2">Metric</td><td colspan="3">Method</td></tr><tr><td>confidence-driven</td><td>human + LLM (non-adaptive)</td><td>LLM only</td></tr><tr><td>Politeness devices (hedge)</td><td>Gain in eff. sample size Coverage</td><td> $( 3 0 . 0 2 \pm 7 . 8 2 ) \%$  95%</td><td> $( - 1 6 . 7 6 \pm 8 . 0 8 ) \%$  89%</td><td>52%</td></tr><tr><td>Politeness devices (1st person pl.)</td><td>Gain in eff. sample size Coverage</td><td>(319.44 ± 22.09)% 94%</td><td>(-8.05 ± 30.09)% 94%</td><td>69%</td></tr><tr><td>Stance on global warming</td><td>Gain in eff. sample size Coverage</td><td>(102.51 ± 13.01)% 96%</td><td>(13.72 ± 22.79)% 96%</td><td>0%</td></tr><tr><td>Political bias (left-leaning)</td><td>Gain in eff. sample size Coverage</td><td>(29.94 ± 8.19)% 97%</td><td> $( 2 0 . 5 6 \pm 6 . 3 3 ) \%$  94%</td><td>2%</td></tr><tr><td>Political bias (right-leaning)</td><td>Gain in eff. sample size Coverage</td><td> $( 6 3 . 7 3 \pm 1 1 . 4 8 ) \%$  91%</td><td> $( 6 1 . 1 5 \pm 8 . 7 5 ) \%$  95%</td><td>90%</td></tr></table>

Table 1: Results summary. Gain in effective sample size and coverage across the five estimation tasks for $n _ { \mathrm { h u m a n } } = 5 0 0$ , estimated over 100 trials. In each task, the confidence-driven approach achieves a higher gain in effective sample size (bolded) than the non-adaptive approach. Confidence-driven approach always achieves a positive gain, while the non-adaptive approach sometimes achieves a negative gain. Confidence-driven and non-adaptive approaches achieve near 90% coverage, or higher. In contrast, LLM-only coverage is often poor. Gain in effective sample size is not estimated for the LLM-only approach as it does not leverage human annotations. Errors show a standard deviation over 100 trials.

Stance. News headlines $( n = 2 , 3 0 0 )$ are agreeing, neutral, or disagreeing with the stance that global warming is a serious concern (Luo et al., 2020). Stance annotations facilitate the study of linguistic differences between media supporting or rejecting global warming, which have implications for communication and policy (Hmielowski et al., 2014). In this task, we estimate $\theta ^ { * }$ corresponding to $O _ { \mathrm { a g r e e m e n t } } .$ , the odds ratio of agreement given the presence of affirming devices such as “expert,” “proven,” “renowned,” and so on. Formally, denoting by $X _ { \mathrm { a f f i r m } } \in \{ 0 , 1 \}$ the presence of an affirming device and $H _ { \mathrm { a g r e e } } \in \{ 0 , 1 \}$ the annotation of agreement, we have

$$
O _ { \mathrm { a g r e e m e n t } } = \frac { \mu _ { \mathrm { a g r e e } | \mathrm { a f f r m } } / ( 1 - \mu _ { \mathrm { a g r e e } | \mathrm { a f f r m } } ) } { \mu _ { \mathrm { a g r e e } | \lnot \mathrm { a f f r m } } / ( 1 - \mu _ { \mathrm { a g r e e } | \lnot \mathrm { a f f r m } } ) } ,
$$

where $\mu _ { \mathrm { a g r e e | a f f i r m } } = P ( H _ { \mathrm { a g r e e } } = 1 | X _ { \mathrm { a f f i r m } } = 1 )$ and $\mu _ { \mathrm { a g r e e | \neg a f f r m } } = P ( H _ { \mathrm { a g r e e } } = 1 | X _ { \mathrm { a f f r m } } = 0 ) .$ Indicators for affirming devices were extracted using a lexicon derived by Luo et al. (2020).

Political bias. News texts (randomly sampled $n ~ = ~ 2 , 0 0 0 )$ are either leaning left, center, or right (Baly et al., 2020). Annotating political leanings in text allows studying the bias in media outlets, socio-technical systems, or historical and contemporary public discourse. Such biases are often reported in terms of prevalence statistics. Thus, in this setting $\theta ^ { * }$ corresponds to the prevalence of a leaning, i.e., $p _ { \mathrm { l e a n } } = P ( H _ { \mathrm { l e a n } } = 1 )$ , where $H _ { \mathrm { l e a n } } \in \{ 0 , 1 \}$ denotes the presence of a leaning. We estimate $p _ { \mathrm { l e f t } }$ and $p _ { \mathrm { r i g h t } }$ , the prevalences of leftand right-leaning articles in the corpus.

## 4.2 Evaluation

Our main evaluation is based on LLM annotations collected with GPT-4o; analogous results with GPT-3.5 can be found in App. B.1. Table 2 in App. A.3 lists prompt texts and parameters. To test LLM performance out of the box, all annotations are collected using zero-shot prompting. Our evaluation is designed to reflect a typical CSS use-case by using standard classification CSS tasks and testing popular API-based models, without any task-specific tuning or training. Analogous results with different formulations of the annotation task, prompting mechanisms, and models are outlined in App. B.

Overall, the confidence scores are calibrated with accuracy, but the annotations are only in moderate agreement with human annotations in all three settings (see App. A.3). This is aligned with our lack of assumption that the LLM annotations are good: we want to produce a valid confidence interval no matter the quality of the LLM annotations.

We report the two key metrics (effective sample size and coverage), for the three selected settings (the study of politeness, stance, and bias), where the task is to estimate the five target quantities $\beta _ { \mathrm { h e d g e } } ,$ $\beta _ { \mathrm { 1 p p } } , O _ { \mathrm { a g r e e m e n t } } , p _ { \mathrm { l e f t } }$ , and $p _ { \mathrm { r i g h t } }$ . Both metrics are estimated over 100 trials for varying $n _ { \mathrm { h u m a n } }$ , the budget for human annotations. Our main findings are reported in Figure 2 and summarized in Table 1.

Effective sample size. First, across the five target quantities, we find that CONFIDENCE-DRIVEN IN-FERENCE increases the effective sample size compared to the human-only baseline. For a given budget of $n _ { \mathrm { h u m a n } }$ annotations, e.g., $n _ { \mathrm { h u m a n } } = 1 0 0 0$ the confidence-driven approach achieves the effective sample size at minimum 1250 (when estimating $p _ { \mathrm { l e f t } } )$ . This means that the confidence interval around the estimated statistic is of equal width as the confidence interval produced with a larger number of human-only annotations.

Similarly, it is informative to consider the necessary budget of human annotations $n _ { \mathrm { h u m a n } }$ given a desired effective sample size $n _ { \mathrm { e f f e c t i v e } }$ . For instance, to achieve $n _ { \mathrm { e f f e c t i v e } } = 1 0 0 0$ , only between around 250 $( \beta _ { \mathrm { 1 p p } } )$ and $7 5 0 \ : ( p _ { \mathrm { l e f t } } )$ human annotations are needed. We thus reduce the number of human annotations needed to achieve equally accurate estimates by at least 25% for all tasks.

Moreover, we also find that the confidencedriven approach increases the effective sample size compared to the human + LLM (non-adaptive) baseline. For example, to achieve $n _ { \mathrm { e f f e c t i v e } } = 1 0 0 0$ the confidence-driven approach requires 200 (respectively, 750) fewer human annotations than the non-adaptive baseline for $O _ { \mathrm { a g r e e m e n t } }$ (respectively, $\beta _ { 1 \mathrm { p p } } )$ . The confidence-driven approach therefore leads to a further reduction in the required number of human annotations compared to an approach that leverages LLMs, but does so non-adaptively. Moreover, notice that the non-adaptive approach can sometimes even hurt compared to the humanonly baseline: in the two politeness tasks, using LLMs actually reduces the effective sample size.

Table 1 summarizes the gain in effective sample size for $n _ { \mathrm { h u m a n } } = 5 0 0$ . Across the five tasks, the confidence-driven approach achieves a substantial gain in the effective sample size, providing at minimum around +30% gain (when estimating $p _ { \mathrm { l e f t } } )$ going even over +300% (when estimating $\beta _ { 1 \mathrm { p p } } )$ Again, the confidence-driven approach achieves a higher gain than the non-adaptive approach for each task, which can even be negative.

Coverage. The save in human annotations does not come at the cost of diminished validity. As expected, across the five target quantities, the confidence-driven approach has coverage around or over 90%, as do the non-adaptive and human-only baselines (Fig. 2). However, LLM-only intervals have a much lower coverage, only being around 90% for $p _ { \mathrm { r i g h t } }$ , and otherwise ranging between 0% $( O _ { \mathrm { a g r e e m e n t } } )$ and $7 0 \% ( \beta _ { \mathrm { 1 p p } } )$ . This emphasizes how estimates only relying on LLM annotations can be misleading. Notably, when estimating $O _ { \mathrm { a g r e e m e n t } }$ using LLM annotations only, the odds-ratio estimate points in the wrong direction $( O _ { \mathrm { a g r e e m e n t } } > 1$ while $O _ { \mathrm { a g r e e m e n t } } \ < \ 1$ is true), as illustrated in Fig. 2. Interestingly, the overall inter-annotator agreement between human and LLM annotations is the highest in this setting (Cohen inter-rater agreement $\kappa _ { \mathrm { s t a n c e } } ~ = ~ 0 . 5 7 )$ This suggests that even when LLM annotations overall agree with human annotations, downstream statistical estimates relying on LLM annotations only can be biased.

Table 1 summarizes the achieved coverage for $n _ { \mathrm { h u m a n } } \ = \ 5 0 0 .$ Across the five tasks, the confidence-driven and non-adaptive approaches achieve around or over 90% coverage (note that small deviations are possible due to only 100 simulation trials). In contrast, the LLM-only approach only meets the requirement for $p _ { \mathrm { r i g h t } }$ and otherwise severely undercovers.

In summary, our method increases the effective sample size given a fixed budget of human annotations, leading to a substantial save in budget, while maintaining the target coverage.

## 5 Discussion

In this work, we introduce CONFIDENCE-DRIVEN INFERENCE, a method that integrates verbalized confidence of LLMs with active inference to optimally combine human and LLM annotations. Across three distinct CSS settings, results demonstrate that the proposed method consistently outperforms baseline methods (human-only and nonadaptive approaches) in effective sample size. Moreover, the increase in the effective sample size is achieved without a decrease in coverage. In contrast, the LLM-only approach yields invalid estimates and considerably lower coverage.

Wer note that the external validity of our findings is contingent upon two key assumptions: that the text instances are i.i.d. from a relevant distribution, and that the researcher has full control of the annotation process. The first may be violated if the distribution of texts shifts over time, and the collected instances are no longer representative of the current quantity of interest. For example, it is possible that relationships between linguistic devices and perceived politeness evolve over time. The second assumption may be violated in situations where certain annotations are difficult to obtain (e.g., for low-resource languages). Our approach may lead to inaccurate or misleading conclusions under either violation. We thus caution against generalizing to settings where text instances exhibit time-varying shifts or the researcher is not in control over the data collection process.

If the adaptive sampling probabilities π<sub>i</sub> are poorly chosen—potentially due to inaccurate verbalized confidence scores—the resulting estimates could have a higher mean squared error (MSE) than if uniform, non-adaptive sampling were used. This could even result in an estimate with a larger MSE than the human-only baseline (for sensitivity to miscalibration, see App B.2). However, by using power tuning, as detailed in Section 3.2, we ensure that incorporating LLM annotation into the estimation process does not hurt the estimate (i.e., does not increase the MSE) regardless of the sampling method used for human annotations (whether uniform or adaptive).

Thus, CONFIDENCE-DRIVEN INFERENCE allows for researchers to allocate human and LLM annotations in a cost-effective manner while maintaining confidence in the statistical validity of their results. Furthermore, CONFIDENCE-DRIVEN IN-FERENCE also addresses the challenges posed by the variable quality of LLM annotation, by providing validity guarantees when leveraging imperfect LLM annotations.

Although overall LLM annotations moderately agree with human annotations in the tested settings, relying on LLM annotations only can lead to wrong conclusions, as shown in the example of estimating the odds ratio in the stance setting. In contrast, despite the fact that LLM annotations are imperfect, our approach allows carefully combining them with a limited set of human annotations in order to reduce the human annotation budget, without sacrificing the validity.

Finally, the accessibility of our method is an important consideration. Across disciplines, researchers can simply prompt the LLM for its confidence via API access and leverage CONFIDENCE-DRIVEN INFERENCE to combine LLM confidence with LLM and human annotations to produce a valid statistical estimate. This approach can be applied to a wide range of tasks, across fields.

## Limitations

We tested only a limited number of LLMs. We note that establishing a comprehensive benchmark is beyond the scope of this work (see App. B.1 for performance details using a different model).

Additionally, while we treat human annotations as the gold standard in our study, we acknowledge that human annotations are biased, and that reasonable annotators can disagree, making it necessary to account for annotator-specific parameters (Hashemi et al., 2024). Future work could explore ways to account for variability and bias in human annotations.

Human annotations are often obtained through crowdsourcing, which may itself be influenced by LLMs, as crowd workers might use LLMs to increase productivity (Veselovsky et al., 2023). Although we use datasets collected before the widespread availability of LLMs, detecting AIgenerated text remains a challenge (Verma et al., 2024).

This work only conducted experiments on estimation tasks within CSS datasets and only in English. However, CONFIDENCE-DRIVEN INFER-ENCE is generalizable to other types of text-based datasets, and it would be valuable to see more diverse applications in future research.

Lastly, the presented experiments do not address causal effects. For instance, in the context of politeness, to identify the causal effect of hedging on perceived politeness, it would be necessary to compare texts that are otherwise identical but differ only in their use of hedging. Nevertheless, while these evaluations are not causal, our method is still applicable for use in causal estimation.

## Ethical Implications

Our work assumes that the existing human annotations within the leveraged datasets serve as the gold standard. However, we caution against interpreting human annotations as definitive judgments, given the subjective nature of many tasks (Fleisig et al., 2023), the potential for annotator disagreement (Weerasooriya et al., 2023), and the influence of annotator positionality (Santy et al., 2023), beliefs, biases (Sap et al., 2022), as well as variance in cultural (Huang and Yang, 2023) and social norms (Ziems et al., 2023).

In addition to their use in text analysis, LLMs may hold potential for simulating human behavior in social science research, including applications such as pretesting surveys and imputing missing data (Bail, 2024). Our work contributes to establishing reliable principles for doing so. At the same time, we do not advocate for using LLMs as substitutes for human data beyond the constraints of our assumptions, especially seeing that prior studies have shown that LLMs tend to reflect the perspectives of some demographic groups more accurately than others (Santurkar et al., 2023) and may propagate stereotypical portrayals (Cheng et al., 2023).

We also caution against fully replacing human annotators with LLM surrogates, which can not only be harmful for the economy (Cazzaniga et al., 2024), but also exacerbate the exploitation of human labor (Li et al., 2023a). Instead, our work highlights the benefits of human-AI collaboration, showing that a combined approach can yield more accurate and valid outcomes.

## Acknowledgments

This work is supported by the Swiss National Science Foundation (Grant P500PT-211127), the Stanford Institute for Human-Centered Artificial Intelligence, Stanford Data Science, and Navy Grant N00014-24-1-2305.

## References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. GPT-4 technical report. arXiv preprint arXiv:2303.08774.

James E Allen, Curry I Guinn, and Eric Horvitz. 1999. Mixed-initiative interaction. IEEE Intelligent Systems and their Applications, 14(5):14–23.

Anastasios N Angelopoulos, Stephen Bates, Clara Fannjiang, Michael I Jordan, and Tijana Zrnic. 2023a. Prediction-powered inference. Science, 382(6671):669–674.

Anastasios N Angelopoulos, John C Duchi, and Tijana Zrnic. 2023b. PPI++: Efficient prediction-powered inference. arXiv preprint arXiv:2311.01453.

Shubham Atreja, Joshua Ashkinaze, Lingyao Li, Julia Mendelsohn, and Libby Hemphill. 2024. Prompt design matters for computational social science tasks but in unpredictable ways. arXiv preprint arXiv:2406.11980.

Christopher A Bail. 2024. Can generative AI improve social science? Proceedings of the National Academy of Sciences, 121(21):e2314021121.

Ramy Baly, Giovanni Da San Martino, James Glass, and Preslav Nakov. 2020. We can detect your bias: Predicting the political ideology of news articles. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 4982–4991.

Neil Band, Xuechen Li, Tengyu Ma, and Tatsunori Hashimoto. 2024. Linguistic calibration of longform generations. In Proceedings of the Forty-first International Conference on Machine Learning.

Pierre Boyeau, Anastasios N Angelopoulos, Nir Yosef, Jitendra Malik, and Michael I Jordan. 2024. Autoeval done right: Using synthetic data for model evaluation. arXiv preprint arXiv:2403.07008.

Dallas Card, Peter Henderson, Urvashi Khandelwal, Robin Jia, Kyle Mahowald, and Dan Jurafsky. 2020. With little power comes great responsibility. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 9263–9274.

Dallas Card and Noah A Smith. 2018. The importance of calibration for estimating proportions from annotations. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 1636–1646.

Mauro Cazzaniga, Ms Florence Jaumotte, Longji Li, Mr Giovanni Melina, Augustus J Panton, Carlo Pizzinelli, Emma J Rockall, and Ms Marina Mendes Tavares. 2024. Gen-AI: Artificial intelligence and the future ofwork. International Monetary Fund.

Ivi Chatzi, Eleni Straitouri, Suhas Thejaswi, and Manuel Gomez Rodriguez. 2024. Predictionpowered ranking of large language models. arXiv preprint arXiv:2402.17826.

Tianqi Chen and Carlos Guestrin. 2016. XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 785–794.

Myra Cheng, Esin Durmus, and Dan Jurafsky. 2023. Marked personas: Using natural language prompts to measure stereotypes in language models. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1504–1532.

Cheng-Han Chiang and Hung-Yi Lee. 2023. Can large language models be an alternative to human evaluations? In Proceedings ofthe 61st Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 15607–15631.

Cristian Danescu-Niculescu-Mizil, Moritz Sudhof, Dan Jurafsky, Jure Leskovec, and Christopher Potts. 2013. A computational approach to politeness with application to social factors. In Proceedings of the 51st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 250–259.

Jesse Dodge, Suchin Gururangan, Dallas Card, Roy Schwartz, and Noah A Smith. 2019. Show your work: Improved reporting of experimental results. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 2185–2194.

Naoki Egami, Musashi Hinck, Brandon Stewart, and Hanying Wei. 2024. Using imperfect surrogates for downstream inference: Design-based supervised learning for social science applications of large language models. Advances in Neural Information Processing Systems, 36.

Amir Feder, Katherine A Keith, Emaad Manzoor, Reid Pryzant, Dhanya Sridhar, Zach Wood-Doughty, Jacob Eisenstein, Justin Grimmer, Roi Reichart, Margaret E Roberts, et al. 2022. Causal inference in natural language processing: Estimation, prediction, interpretation and beyond. Transactions ofthe Associationfor Computational Linguistics, 10:1138–1158.

Eve Fleisig, Rediet Abebe, and Dan Klein. 2023. When the majority is wrong: Modeling annotator disagreement for subjective tasks. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 6715–6726.

Jiahui Geng, Fengyu Cai, Yuxia Wang, Heinz Koeppl, Preslav Nakov, and Iryna Gurevych. 2024. A survey of confidence estimation and calibration in large language models. In Proceedings ofthe 2024 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 6577–6595, Mexico City, Mexico. Association for Computational Linguistics.

Fabrizio Gilardi, Meysam Alizadeh, and Maël Kubli. 2023. ChatGPT outperforms crowd workers for text-annotation tasks. Proceedings of the National Academy ofSciences, 120(30):e2305016120.

Anisha Gunjal, Jihan Yin, and Erhan Bas. 2024. Detecting and preventing hallucinations in large vision language models. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 38, pages 18135–18143.

Helia Hashemi, Jason Eisner, Corby Rosset, Benjamin Van Durme, and Chris Kedzie. 2024. Llm-rubric: A multidimensional, calibrated approach to automated evaluation of natural language texts. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13806–13834.

Jay D Hmielowski, Lauren Feldman, Teresa A Myers, Anthony Leiserowitz, and Edward Maibach. 2014. An attack on science? media use, trust in scientists, and perceptions of global warming. Public Understanding ofScience, 23(7):866–883.

Jing Huang and Diyi Yang. 2023. Culturally aware natural language inference. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 7591–7609, Singapore. Association for Computational Linguistics.

Katherine Keith and Brendan O’Connor. 2018. Uncertainty-aware generative models for inferring document class prevalence. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 4575–4585, Brussels, Belgium. Association for Computational Linguistics.

Hannah Kim, Kushan Mitra, Rafael Li Chen, Sajjadur Rahman, and Dan Zhang. 2024. MEGAnno+: A human-LLM collaborative annotation system. In Proceedings ofthe 18th Conference ofthe European Chapter of the Association for Computational Linguistics: System Demonstrations, pages 168–176, St. Julians, Malta. Association for Computational Linguistics.

Seungone Kim, Jamin Shin, Yejin Cho, Joel Jang, Shayne Longpre, Hwaran Lee, Sangdoo Yun, Seongjin Shin, Sungdong Kim, James Thorne, et al. 2023. Prometheus: Inducing fine-grained evaluation capability in language models. In The Twelfth International Conference on Learning Representations.

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners. In Advances in Neural Information Processing Systems.

Michelle S Lam, Janice Teoh, James A Landay, Jeffrey Heer, and Michael S Bernstein. 2024. Concept induction: Analyzing unstructured text with highlevel concepts using lloom. In Proceedings of the CHI Conference on Human Factors in Computing Systems, pages 1–28.

Richard N Landers and Tara S Behrend. 2023. Auditing the AI auditors: A framework for evaluating fairness and bias in high stakes AI predictive models. American Psychologist, 78(1):36.

Hanlin Li, Nicholas Vincent, Stevie Chancellor, and Brent Hecht. 2023a. The dimensions of data labor: A road map for researchers, activists, and policymakers to empower data producers. In Proceedings of the 2023 ACM conference on fairness, accountability, and transparency, pages 1151–1161.

Junyi Li, Xiaoxue Cheng, Xin Zhao, Jian-Yun Nie, and Ji-Rong Wen. 2023b. HaluEval: A large-scale hallucination evaluation benchmark for large language models. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 6449–6464, Singapore. Association for Computational Linguistics.

Minzhi Li, Taiwei Shi, Caleb Ziems, Min-Yen Kan, Nancy Chen, Zhengyuan Liu, and Diyi Yang. 2023c. Coannotating: Uncertainty-guided work allocation between human and large language models for data annotation. In Proceedings of the 2023 Conference

on Empirical Methods in Natural Language Processing, pages 1487–1505.

Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. 2023. G-Eval: NLG evaluation using Gpt-4 with better human alignment. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 2511–2522.

Yiwei Luo, Dallas Card, and Dan Jurafsky. 2020. Detecting stance in media on global warming. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2020, pages 3296–3315.

Katerina Margatina, Giorgos Vernikos, Loïc Barrault, and Nikolaos Aletras. 2021. Active learning by acquiring contrastive examples. In Proceedings ofthe 2021 Conference on Empirical Methods in Natural Language Processing, pages 650–663, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Matthew L Newman, Carla J Groom, Lori D Handelman, and James W Pennebaker. 2008. Gender differences in language use: An analysis of 14,000 text samples. Discourse processes, 45(3):211–236.

Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. 2023. Generative agents: Interactive simulacra of human behavior. In Proceedings of the 36th annual acm symposium on user interface software and technology, pages 1–22.

Chau Pham, Alexander Hoyle, Simeng Sun, Philip Resnik, and Mohit Iyyer. 2024. TopicGPT: A promptbased topic modeling framework. In Proceedings of the 2024 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 2956–2984.

Filipe Ribeiro, Lucas Henrique, Fabricio Benevenuto, Abhijnan Chakraborty, Juhi Kulshrestha, Mahmoudreza Babaei, and Krishna Gummadi. 2018. Media bias monitor: Quantifying biases of social media news outlets at large-scale. In Proceedings of the International AAAI Conference on Web and Social Media, volume 12.

Ronald E Robertson, Shan Jiang, Kenneth Joseph, Lisa Friedland, David Lazer, and Christo Wilson. 2018. Auditing partisan audience bias within Google search. Proceedings of the ACM on human-computer interaction, 2(CSCW):1–22.

Jon Saad-Falcon, Omar Khattab, Christopher Potts, and Matei Zaharia. 2024. ARES: An automated evaluation framework for retrieval-augmented generation systems. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 338–354.

Shibani Santurkar, Esin Durmus, Faisal Ladhak, Cinoo Lee, Percy Liang, and Tatsunori Hashimoto. 2023. Whose opinions do language models reflect? In International Conference on Machine Learning, pages 29971–30004. PMLR.

Sebastin Santy, Jenny Liang, Ronan Le Bras, Katharina Reinecke, and Maarten Sap. 2023. NLPositionality: Characterizing design biases of datasets and models. In Proceedings of the 61st Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 9080–9102, Toronto, Canada. Association for Computational Linguistics.

Maarten Sap, Swabha Swayamdipta, Laura Vianna, Xuhui Zhou, Yejin Choi, and Noah A. Smith. 2022. Annotators with attitudes: How annotator beliefs and identities bias toxic language detection. In Proceedings ofthe 2022 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 5884–5906, Seattle, United States. Association for Computational Linguistics.

Melanie Sclar, Yejin Choi, Yulia Tsvetkov, and Alane Suhr. 2023. Quantifying language models’ sensitivity to spurious features in prompt design or: How I learned to start worrying about prompt formatting. In The Twelfth International Conference on Learning Representations.

Yanchuan Sim, Brice D. L. Acree, Justin H. Gross, and Noah A. Smith. 2013. Measuring ideological proportions in political speeches. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, pages 91–101, Seattle, Washington, USA. Association for Computational Linguistics.

Surendrabikram Thapa, Usman Naseem, and Mehwish Nasim. 2023. From humans to machines: Can ChatGPT-like LLMs effectively replace human annotators in NLP tasks. In Workshop Proceedings of the 17th International AAAI Conference on Web and Social Media.

Katherine Tian, Eric Mitchell, Allan Zhou, Archit Sharma, Rafael Rafailov, Huaxiu Yao, Chelsea Finn, and Christopher D Manning. 2023. Just ask for calibration: Strategies for eliciting calibrated confidence scores from language models fine-tuned with human feedback. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 5433–5442.

Aad W Van der Vaart. 2000. Asymptotic statistics, volume 3. Cambridge university press.

Vivek Verma, Eve Fleisig, Nicholas Tomlin, and Dan Klein. 2024. Ghostbuster: Detecting text ghostwritten by large language models. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 1702–1717.

Veniamin Veselovsky, Manoel Horta Ribeiro, and Robert West. 2023. Artificial artificial artificial intelligence: Crowd workers widely use large language models for text production tasks. arXiv preprint arXiv:2306.07899.

Rob Voigt, Nicholas P Camp, Vinodkumar Prabhakaran, William L Hamilton, Rebecca C Hetey, Camilla M Griffiths, David Jurgens, Dan Jurafsky, and Jennifer L Eberhardt. 2017. Language from police body camera footage shows racial disparities in officer respect. Proceedings of the National Academy of Sciences, 114(25):6521–6526.

Tharindu Cyril Weerasooriya, Alexander Ororbia, Raj Bhensadadia, Ashiqur KhudaBukhsh, and Christopher Homan. 2023. Disagreement matters: Preserving label diversity by jointly modeling item and annotator label distributions with DisCo. In Findings of the Associationfor Computational Linguistics: ACL 2023, pages 4679–4695, Toronto, Canada. Association for Computational Linguistics.

Johnny Tian-Zheng Wei, Frederike Zufall, and Robin Jia. 2024. Operationalizing content moderation “accuracy”’ in the Digital Services Act. In Proceedings ofthe AAAI/ACM Conference on AI, Ethics, and Society, volume 7, pages 1527–1538.

Laura Weidinger, Jonathan Uesato, Maribeth Rauh, Conor Griffin, Po-Sen Huang, John Mellor, Amelia Glaese, Myra Cheng, Borja Balle, Atoosa Kasirzadeh, et al. 2022. Taxonomy of risks posed by language models. In Proceedings of the 2022 ACM Conference on Fairness, Accountability, and Transparency, pages 214–229.

Ruoyu Zhang, Yanzeng Li, Yongliang Ma, Ming Zhou, and Lei Zou. 2023. LLMaAA: Making large language models as active annotators. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2023, pages 13088–13103, Singapore. Association for Computational Linguistics.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. Advances in Neural Information Processing Systems, 36:46595–46623.

Kaitlyn Zhou, Jena D Hwang, Xiang Ren, and Maarten Sap. 2024. Relying on the unreliable: The impact of language models’ reluctance to express uncertainty. arXiv preprint arXiv:2401.06730.

Caleb Ziems, Jane Dwivedi-Yu, Yi-Chia Wang, Alon Halevy, and Diyi Yang. 2023. Normbank: A knowledge bank of situational social norms. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7756–7776.

Caleb Ziems, William Held, Omar Shaikh, Jiaao Chen, Zhehao Zhang, and Diyi Yang. 2024. Can large language models transform computational social science? Computational Linguistics, 50(1):237–291.

Tijana Zrnic and Emmanuel J Candès. 2024. Active statistical inference. arXiv preprint arXiv:2403.03208.

## A Further Details on the Method

## A.1 Confidence Intervals

We compute the confidence intervals following the approach in (Zrnic and Candès, 2024). Suppose that $\hat { \theta } ^ { \mathrm { c o n f } }$ is possibly d-dimensional (such as in, for example, linear or logistic regression), and we are interested in coefficient j. If $d = 1$ , such as in the case of prevalence estimation, then j is always equal to 1. We compute the confidence interval as:

$$
C _ { 1 - \alpha } = \left( \hat { \theta } _ { j } ^ { \mathrm { c o n f } } \pm z _ { 1 - \alpha / 2 } \sqrt { \frac { \widehat { \Sigma } _ { j j } } { n } } \right) ,
$$

where $z _ { 1 - \alpha / 2 }$ is the $1 - \alpha / 2$ quantile of the standard normal distribution. The matrix $\widehat { \Sigma }$ is an estimate of the covariance of $\hat { \theta } ^ { \mathrm { c o n f } }$ b, given by:

$$
\widehat { \boldsymbol { \Sigma } } = \hat { \boldsymbol { H } } ^ { - 1 } \widehat { \mathrm { V a r } } \left( \lambda \boldsymbol { \nabla } \hat { \ell } _ { \hat { \theta } ^ { \mathrm { c o n f } } } + ( \boldsymbol { \nabla } \ell _ { \hat { \theta } ^ { \mathrm { c o n f } } } - \lambda \boldsymbol { \nabla } \hat { \ell } _ { \hat { \theta } ^ { \mathrm { c o n f } } } ) \frac { \xi } { \pi } \right) \hat { \boldsymbol { H } } ^ { - 1 } ,
$$

where $\hat { H } = \hat { \mathbb { E } } [ \nabla ^ { 2 } \ell _ { \hat { \theta } ^ { \mathrm { c o n f } } } ]$ is the empirical estimate of the Hessian at $\hat { \theta } ^ { \mathrm { c o n f } }$ and $\widehat { \mathrm { V a r } }$ denotes the empirdical variance. Recall also the short-hand notation $\ell _ { \theta } = \ell _ { \theta } ( X , H )$ and $\hat { \ell } _ { \theta } = \ell _ { \theta } ( X , \hat { H } )$ . This is a generalization of the usual “sandwich” covariance used in linear regression.

Some estimation targets, such as the odds ratio, are not M-estimators but are functions of Mestimators. In those cases a confidence interval is obtained by additionally applying the delta method.

See (Zrnic and Candès, 2024) for further details.

## A.2 Power Tuning

Power tuning, introduced by Angelopoulos et al. (2023b), refers to choosing λ so that the MSE of $\hat { \theta } ^ { \mathrm { c o n f } }$ , or equivalently its variance, is minimized over λ. Since $\widehat { \Sigma } _ { j j }$ is a quadratic in $\lambda ,$ the optimal λ bhas a closed-form analytical expression. As before, suppose we are interesting in estimating coordinate j of $\hat { \theta } ^ { \mathrm { c o n f } }$ . Let $h _ { j }$ denote the j-th column of $\hat { H } ^ { - 1 }$ Then, we set λ according to:

$$
\lambda = \frac { h ^ { \top } \widehat { \mathrm { C o v } } h } { 2 h ^ { \top } \widehat { \mathrm { V a r } } h } ,
$$

where $\begin{array} { r } { \widehat { \mathrm { C o v } } : = \widehat { \mathrm { C o v } } \big ( \nabla \widehat { \ell } _ { \widehat { \theta } ^ { \mathrm { c o n f } } } \big ( \frac { \xi } { \pi } - 1 \big ) , \nabla \ell _ { \widehat { \theta } ^ { \mathrm { c o n f } } } \frac { \xi } { \pi } \big ) } \end{array}$ + $\widehat { \mathrm { C o v } } ( \nabla \ell _ { \widehat { \theta } ^ { \mathrm { c o n f } } } \frac { \xi } { \pi } , \nabla \hat { \ell } _ { \widehat { \theta } ^ { \mathrm { c o n f } } } ( \frac { \xi } { \pi } - \overset { \cdot } { 1 } ) )$ and $\begin{array} { r l } { \widehat { \mathrm { V a r } } } & { { } : = } \end{array}$ $\widehat { \mathrm { V a r } } ( \nabla \hat { \ell } _ { \hat { \theta } ^ { \mathrm { c o n f } } } ( \frac \xi \pi - 1 ) )$ dare empirical (co)variances. dSee (Angelopoulos et al., 2023b) for further details.

## A.3 LLM and Human Annotation Details

For data annotation, we use GPT-4o (gpt-4o-2024- 05-13 version) and GPT-3.5-turbo (gpt-3.5-turbo-0125 version). Prompt texts in both stages are listed in Table 2. To test LLM performance out-of-thebox, all annotations are collected using zero-shot prompting. We set the max\_tokens parameter to $^ { 5 , }$ use default temperature (1), and the default system prompt and the other prompting parameters.

Stage 1 GPT-4o annotations are in moderate agreement with human annotations in all three settings: $\kappa _ { \mathrm { p o l i t e n e s s } } ~ = ~ 0 . 3 9 , ~ \kappa _ { \mathrm { s t a n c e } } ~ = ~ 0 . 5 7$ and $\kappa _ { \mathrm { b i a s } } ~ = ~ 0 . 4 3$ . For context, human annotators had a median inter-annotator pairwise correlation of 0.68 for the politeness dataset, while average inter-annotator agreement ranged from 0.54 to 0.64 across annotation rounds for the stance dataset(Danescu-Niculescu-Mizil et al., 2013; Luo et al., 2020). No agreement data is available for the political bias dataset.

In Stage 2, we find that the collected verbalized confidence scores are calibrated with the Stage 1 accuracy (Fig. 3 (right)), such that higher confidence scores correspond to higher accuracy with respect to human annotations. This implies that verbalized confidence is indeed an informative signal to leverage in estimation tasks. Histograms of the collected verbalized confidence scores are illustrated in Fig. 3 (left). We also observe a variance in the verbalized confidence within each setting, and a relative lack of overconfident responses (where the model is 100% certain).

We choose the sampling probabilities $\pi _ { i }$ according to the theory of Zrnic and Candès (2024). For estimating the prevalences $p _ { \mathrm { l e f t } }$ and $p _ { \mathrm { r i g h t } }$ as well as the odds ratio $O _ { \mathrm { a g r e e m e n t } }$ , we choose $\pi _ { i } \propto \sqrt { \widehat { \mathsf { e r r } } _ { i } ( C _ { i } ) }$ , as described in Section 3.2. For dthe logistic regression coefficient $\beta _ { \mathrm { h e d g e } }$ (respectively, $\beta _ { 1 \mathrm { p p } } ) .$ , we set $\pi _ { i } \propto \sqrt { \overrightarrow { \mathrm { e r r } } _ { i } ( C _ { i } ) } \cdot | X _ { i } ^ { \top } h | .$ where h is the column of $\widehat { H }$ d(defined in App. A.1) corresponding to $X _ { \mathrm { h e d g e } }$ b(respectively, $X _ { \mathrm { 1 p p } } )$

To fit $\widehat { \mathsf { e r r } } _ { i }$ , we train an XGBoost (Chen and dGuestrin, 2016) model. For all problem settings, we use the same training parameters: number of boosting rounds 2000, step size 0.001, maximum depth 3, and squared-error objective.

## A.4 Computation of Evaluation Metrics

We provide further details behind the computation of our two main metrics, effective sample size and coverage. For all problem settings, we run 100 simulation trials. All experiments were run on a single CPU.

Effective sample size. Recall that we define the effective sample size of a method as the hypothetical value $n _ { \mathrm { e f f e c t i v e } }$ such that $\begin{array} { r l } { \mathrm { M S E } ( \hat { \theta } ^ { \mathrm { m e t h o d } } ) = } \end{array}$ $\mathrm { M S E } ( \hat { \theta } _ { n _ { \mathrm { e f f e c t i v e } } } ^ { \mathrm { h u m a n } } )$ , where $\hat { \theta } _ { n _ { \mathrm { e f f e c t i v e } } } ^ { \mathrm { h u m a n } }$ is obtained via the human-only approach with $n _ { \mathrm { e f f e c t i v e } }$ annotations. Since all approaches but the LLM-only approach are unbiased in the large-sample limit, meaning their estimate has mean exactly equal to $\theta ^ { * }$ , the MSE is simply equal to the estimator variance. Estimator variance is used in the confidence interval construction and is estimated as ${ \widehat { \Sigma } } / n .$ , as bexplained in App. A.1. The different baselines differ in their choice of λ and $\pi$ in the definition of $\widehat { \Sigma }$ . We thus compute the effective sample size as $\widehat { \Sigma } _ { j j } ^ { \mathrm { h u m a n } } / \widehat { \Sigma } _ { j j } \cdot n .$ , where $j$ indexes the coordinate of $\hat { \theta } ^ { \mathrm { c o n f } }$ bwhen the estimate has more than one dimension. The final reported effective sample size is the mean of these values over 100 trials.

Coverage. We estimate coverage over 100 trials. For all methods but LLM only, the trials differ in the random annotation decisions $\xi _ { i }$ that determine which points get human-annotated, and we average 0/1 indicators of coverage over those trials. For LLM only, since we only have one fixed dataset of n LLM annotations, in order to estimate coverage we simulate random draws from a population via the bootstrap. In other words, in each trial we draw $n$ LLM annotations with replacement, form a classical confidence interval using those points, and record a 0/1 indicator of coverage.

## B Supplementary Results

## B.1 LLM Data Collection Robustness

To examine the robustness of our evaluation to choices in the LLM data collection, we collected LLM annotations using varying approaches on the task of analyzing impact of hedging on perceived politeness. The default experiment (cf. Figure 2) leverages LLM annotations collected using GPT-4o model, via zero-shot prompting, where the annotation task is binary classification. In Figure 3, for $n _ { h u m a n } = 1 1 0 0$ , we report the gain in effective sample size and coverage using (1) an alternative smaller model (GPT-4o-mini), (2) alternative prompting mechanism (few shot prompting with ten examples), and (3) alternative annotation task (rating on a 7-point bipolar Likert scale, ranging from “very impolite $( 1 ) ^ { \dag }$ to “very polite $( 7 ) ^ { \mathfrak { r } } )$ .

For each LLM data collection method, the confidence-driven approach consistently achieves a higher gain in effective sample size than the non-adaptive approach. Moreover, LLM-only coverage is poor across the different data collection methods (except for the experiment with a smaller model), while non-adaptive and adaptive approaches achieve 90% coverage or higher. We note that, although inter-annotator agreement varies substantially depending on these choices, between low $( \kappa _ { \mathrm { p o l i t e n e s s } } = 0 . 2 1 )$ and moderate $( \kappa _ { \mathrm { p o l i t e n e s s } } = 0 . 5 6 )$ , confidence-driven approach is not harmed by the varying quality of the annotations, and always achieves a positive gain in the effective sample size. We also note that with ten few-shot examples, LLM-only coverage increases (82%), as predicted, since examples help to guide the annotation task. We also note that LLM annotations are lower in quality when collected using a Likert scale, likely due to eliciting more finegrained classification, which makes the rating task more challenging.

Additionally, Figure 4 and Table 4 summarize the results using GPT-3.5. For $n _ { \mathrm { h u m a n } } = 5 0 0$ in each estimation task the confidence-driven approach again achieves a higher gain in effective sample size than the non-adaptive approach. Moreover, it always achieves a positive gain. In contrast, the non-adaptive approach achieves a negative gain in three out of the five estimation tasks (both politeness estimates and the stance estimate). The confidence-driven and non-adaptive approaches always achieve over 90% coverage. In contrast, LLM-only coverage is always poor using GPT-3.5 (while using GPT-4o it was poor on four out of the five estimation tasks).

In summary, our insights regarding the gains of the confidence-driven approach are robust to the choices made in the LLM data collection.

## B.2 Sensitivity to Confidence Calibration

A calibrated LLM will produce higher confidence scores when annotations are in agreement with human annotations, compared to when annotations are in disagreement with human annotations. However, calibration of confidence scores across tasks is not guaranteed.

To understand how the performance of

CONFIDENCE-DRIVEN INFERENCE is affected by the calibration of confidence scores, we conducted a robustness test, adding noise to confidence scores to simulate miscalibration. In particular, for illustration we consider the task of analyzing stance on global warming. We add a varying amount of normally distributed noise ${ \mathcal { N } } ( 0 , \sigma ^ { 2 } )$ to the collected confidence scores $C _ { i }$ , and truncate the sum to [0, 1] to obtain a probability.

We use a t-test to test the difference in calibration score means when LLM and human annotations agree, vs when LLM and human annotations disagree. If the t-statistic is large (equivalently, the corresponding p-value is small), that suggests that the two means differ significantly. As the random noise added to the confidence scores increases, the scores become less calibrated (Table 5). We expect that our method performs worse in terms of $n _ { \mathrm { e f f e c t i v e } }$ when the confidence scores are miscalibrated, although coverage should be maintained.

As predicted, as the amount of miscalibration in the confidence scores increases, the gain in the effective sample size decreases (Table 5). The confidence-driven approach achieves the highest gain for the smallest amount of noise, though it always achieves a positive gain. This suggests that the approach is robust to poor confidence scores. Furthermore, CONFIDENCE-DRIVEN INFERENCE achieves near 90% coverage or higher in each setting, regardless of the amount of miscalibration. Finally, we observe that CONFIDENCE-DRIVEN INFERENCE achieves a higher gain than the nonadaptive approach regardless of the extent of miscalibration. This can be explained through the power tuning parameter $\lambda ;$ even when the confidence scores provide no signal, power tuning makes sure that LLM annotations are leveraged effectively.

![](images/9cbe3da76ecc81efeeeaebca9cc922d6e2b0ad89fe16d26c3bca721af20c1793.jpg)

![](images/17736c2bc8cce9b37a7e2ffce9e4f4fbb24fb856cbdcc6c5c1e8262b8cfbde07.jpg)

![](images/024848ed670122c661122dca7351710d92cd0719607392609804482a03fab626.jpg)

![](images/7f5946819cecefc6a2f33742ddeca687e2ff018a376e420cad1f5aa23dc07c06.jpg)

![](images/042afd209d16a84760afb136e5eb42e045b664a35ac4dd694458cc985d14a379.jpg)

![](images/acc6ffaf84da4aaf7cf143b7f910defffbdac96bba1f365bfd04b310e2ecf9e3.jpg)  
Figure 3: Histograms and calibration curves of verbalized confidence scores. (Left) Confidence score histograms across the three settings (GPT-4o). (Right) LLM annotation accuracy with respect to human annotations (y-axis), among instances where the confidence score is greater than C (x-axis) across the three settings (GPT-4o).

<table><tr><td rowspan=1 colspan=1>Setting</td><td rowspan=1 colspan=2>Stage         Prompt text</td></tr><tr><td rowspan=1 colspan=1>Politeness</td><td rowspan=1 colspan=1>Stage 1</td><td rowspan=1 colspan=1>Is the following text polite? Output either A or B. Output a letter only.A) PoliteB) ImpoliteText: &lt;text&gt;Answer:</td></tr><tr><td rowspan=1 colspan=1>Politeness</td><td rowspan=1 colspan=1>Stage 2</td><td rowspan=1 colspan=1>How likely is it that the following text is &lt;previously provided answer: polite or impolite&gt;?Output the probability only (a number between 0 and 1).Text: &lt;text&gt;Answer:</td></tr><tr><td rowspan=1 colspan=1>Stance</td><td rowspan=1 colspan=1>Stage 1</td><td rowspan=1 colspan=1>A statement can agree, be neutral, or disagree with the statement: “Climate change/globalwarming is a serious concern&quot;. Classify the following statement into one of the threecategories. Output either A, B, or C. Output a letter only.A) AgreeB) NeutralC) DisagreeStatement: &lt;text&gt;Answer:</td></tr><tr><td rowspan=1 colspan=1>Stance</td><td rowspan=1 colspan=1>Stage 2</td><td rowspan=1 colspan=1>How likely is it that the following text &lt;previously provided answer: agrees, neither agreesnor disagrees, or disagrees&gt; with the statement: “Climate change/global warming is aserious concern&quot;?Output the probability only (a number between 0 and 1).Text: &lt;text&gt;Probability:</td></tr><tr><td rowspan=1 colspan=1>Bias</td><td rowspan=1 colspan=1>Stage 1</td><td rowspan=1 colspan=1>What is the political bias of the following article? Output either A,B, or C. Output a letteronly.A) LeftB) CenterC) RightArticle: &lt;text&gt;Answer:</td></tr><tr><td rowspan=1 colspan=1>Bias</td><td rowspan=1 colspan=1>Stage 2</td><td rowspan=1 colspan=1>How likely is it that the following article has a &lt;previously provided answer: left-leaning,centrist, or right-leaning&gt; political bias? Output the probability only (a number between 0and 1).Text: &lt;text&gt;Probability:</td></tr></table>

Table 2: Complete prompt texts. LLM annotation prompts across the three settings, for Stages 1 and 2.

<table><tr><td>Method Gain in eff. sample size</td><td colspan="2">Metric Coverage</td></tr><tr><td>default experiment, cf. Figure 2 (model: GPT-4o, prompting: zero-shot, annotation task: binary classification)</td><td></td><td></td></tr><tr><td>Cohen inter-rater agreement LLM only human + LLM (non-adaptive)</td><td> $\kappa _ { \mathrm { p o l i t e n e s s } } = 0 . 3 9$  -9.85%</td><td>71% 93%</td></tr><tr><td>confidence-driven alternative model: GPT-4o-mini</td><td>248%</td><td>99%</td></tr><tr><td>Cohen inter-rater agreement LLM only human + LLM (non-adaptive)</td><td> $\kappa _ { \mathrm { p o l i t e n e s s } } = 0 . 5 6$  24.15%</td><td>90% 94%</td></tr><tr><td colspan="3">265.69% 100% confidence-driven alternative prompting: few-shot (10 examples)</td></tr><tr><td>Cohen inter-rater agreement LLM only human + LLM (non-adaptive) confidence-driven</td><td> $\kappa _ { \mathrm { p o l i t e n e s s } } = 0 . 4 5$  3.1% 251.25%</td><td>82% 91% 100%</td></tr><tr><td>alternative annotation task: Likert scale (7-point) Cohen inter-rater agreement</td><td> $\kappa _ { \mathrm { p o l i t e n e s s } } = 0 . 2 1$ </td><td></td></tr><tr><td>LLM only human + LLM (non-adaptive) confidence-driven</td><td>-27.45% 239.92%</td><td>26% 92% 99%</td></tr></table>

Table 3: Sensitivity to the LLM data collection method. Gain in effective sample size and coverage for the LLM only, human + LLM (non-adaptive), and confidence-driven approaches, across varying data collection approaches. Results are presented for the task of analyzing impact of hedging on perceived politeness, $n _ { \mathrm { h u m a n } } = 1 1 0 0 .$ estimated over 100 trials. The confidence-driven approach always achieves a large positive gain. For each LLM data collection method, the confidence-driven approach achieves a higher gain in effective sample size than the non-adaptive approach (marked in bold); it also achieves near 90% coverage or higher in each setting. In contrast, LLM-only coverage is often poor. Gain in effective sample size is not estimated for the LLM-only approach as it does not leverage human annotations.

![](images/c35099ed11613402eb6a03caece08d4eb057d057b18dedbaf0308860663dee00.jpg)  
Figure 4: Confidence intervals, effective sample size, and coverage (GPT-3.5). Rows correspond to different estimation tasks. The first column shows the confidence intervals in five random trials. The vertical dashed line corresponds to the estimate produced on the full dataset. A method is valid if its confidence interval includes this estimate (in about 90% of the trials), and tighter intervals around $\theta ^ { * }$ indicates better performance. The second and third columns display the effective sample size n<sub>efective</sub> and coverage, respectively, for different values of the human annotation budget $n _ { \mathrm { h u m a n } } .$ Results are estimated over 100 trials.

<table><tr><td rowspan="2">Estimation task</td><td rowspan="2">Metric</td><td colspan="3">Method</td></tr><tr><td>confidence-driven</td><td>human + LLM (non-adaptive)</td><td>LLM only</td></tr><tr><td>Politeness devices (hedge)</td><td>Gain in eff. sample size Coverage</td><td> $( 3 1 . 5 1 \pm 7 . 8 1 ) \%$  92%</td><td> $( - 1 2 . 2 3 \pm 9 . 1 8 ) \%$  92%</td><td>39%</td></tr><tr><td rowspan="2">Politeness devices (1st person plural)</td><td rowspan="2">Gain in eff. sample size Coverage</td><td> $( 3 2 1 . 0 0 \pm 1 9 . 0 1 ) \%$ </td><td> $( - 5 . 7 7 \pm 3 0 . 8 3 ) \%$ </td><td></td></tr><tr><td>97%</td><td>92%</td><td>67%</td></tr><tr><td>Stance on global warming</td><td>Gain in eff. sample size</td><td>(82.68 ± 9.17)%</td><td>(-11.23 ± 17.15)%</td><td></td></tr><tr><td rowspan="2">Political bias</td><td rowspan="2">Coverage Gain in eff. sample size</td><td>96%</td><td>94%</td><td>0%</td></tr><tr><td>(17.08 ± 6.30)%</td><td> $( 3 . 8 3 \pm 4 . 8 6 ) \%$ </td><td></td></tr><tr><td>(left-leaning) Political bias</td><td>Coverage</td><td>93%</td><td>98%</td><td>18%</td></tr><tr><td rowspan="2">(right-leaning)</td><td rowspan="2">Gain in eff. sample size Coverage</td><td> $( 2 3 . 1 0 \pm 7 . 5 0 ) \%$ </td><td> $( 1 1 . 0 9 \pm 5 . 7 3 ) \%$ </td><td></td></tr><tr><td>95%</td><td>94%</td><td>11%</td></tr></table>

Table 4: Results summary (GPT-3.5). Gain in effective sample size and coverage across the five estimation tasks for $n _ { \mathrm { h u m a n } } = 5 0 0$ , estimated over 100 trials. In each task, the confidence-driven approach achieves a higher gain in effective sample size (bolded) than the non-adaptive approach. Confidence-driven approach always achieves a positive gain, while the non-adaptive approach sometimes achieves a negative gain. Confidence-driven and non-adaptive approaches achieve near 90% coverage, or higher. In contrast, LLM-only coverage is poor. Gain in effective sample size is not estimated for the LLM-only approach as it does not leverage human annotations. Errors show a standard deviation over 100 trials.

<table><tr><td>Method</td><td>Confidence score calibration t-test</td><td>Metric Gain in eff. sample size</td><td>Coverage</td></tr><tr><td colspan="4"> $n _ { \mathrm { h u m a n } } = 5 0 0$ </td></tr><tr><td>LLM only</td><td></td><td></td><td>0%</td></tr><tr><td>human + LLM (non-adaptive)</td><td></td><td>-3.46%</td><td>100%</td></tr><tr><td>confidence-driven  $( \sigma ^ { 2 } = 0 )$ </td><td> $t = 9 . 0 8 , p = 2 . 1 9 \times 1 0 ^ { - 1 9 }$ </td><td>43.48%</td><td>94%</td></tr><tr><td>confidence-driven  $( \sigma ^ { 2 } = 0 . 2 )$ </td><td> $t = 5 . 5 3 , p = 3 . 6 5 \times 1 0 ^ { - 8 }$ </td><td>40.70%</td><td>95%</td></tr><tr><td>confidence-driven  $( \sigma ^ { 2 } = 0 . 4 )$ </td><td> $t = 3 . 3 0 , p = 0 . 0 0 0 9 9 4$ </td><td>42.82%</td><td>94%</td></tr><tr><td>confidence-driven  $( \sigma ^ { 2 } = 0 . 6 )$ </td><td> $t = 2 . 3 4 , p = 0 . 0 1 9 2$ </td><td>39.57%</td><td>93%</td></tr><tr><td>confidence-driven  $( \sigma ^ { 2 } = 0 . 8 )$ </td><td> $t = 1 . 8 8 , p = 0 . 0 6 0 6$ </td><td>40.87%</td><td>94%</td></tr><tr><td colspan="4"> $n _ { \mathrm { h u m a n } } = 1 1 5 0$ </td></tr><tr><td>LLM only</td><td></td><td></td><td>0%</td></tr><tr><td>human + LLM (non-adaptive)</td><td></td><td>17.02%</td><td>100%</td></tr><tr><td>confidence-driven  $( \sigma ^ { 2 } = 0 )$ </td><td> $t = 9 . 0 8 , p = 2 . 1 9 \times 1 0 ^ { - 1 9 }$ </td><td>28.14%</td><td>99%</td></tr><tr><td>confidence-driven  $( \sigma ^ { 2 } = 0 . 2 )$ </td><td> $t = 5 . 2 9 , p = 1 . 3 6 \times 1 0 ^ { - 7 }$ </td><td>25.30%</td><td>100%</td></tr><tr><td>confidence-driven  $( \sigma ^ { 2 } = 0 . 4 )$ </td><td> $t = 3 . 2 2 , p = 0 . 0 0 1 3 0$ </td><td>25.84%</td><td>97%</td></tr><tr><td>confidence-driven  $( \sigma ^ { 2 } = 0 . 6 )$ </td><td> $t = 2 . 2 3 , p = 0 . 0 2 5 7$ </td><td>26.81%</td><td>100%</td></tr><tr><td>confidence-driven  $( \sigma ^ { 2 } = 0 . 8 )$ </td><td> $t = 1 . 8 3 , p = 0 . 0 8 3 1$ </td><td>25.71%</td><td>99%</td></tr></table>

Table 5: Sensitivity to confidence score calibration. Gain in effective sample size and coverage for the LLM only, human + LLM (non-adaptive), and confidence-driven approaches, given varying amounts of miscalibration in confidence scores $( \sigma ^ { 2 } )$ . Results are presented for the task of analyzing stance on global warming, estimated over 100 trials. The t-test tests for the difference in calibration score means when LLM and human annotations agree, vs when LLM and human annotations disagree (larger t means difference is more significant). The confidence-driven approach achieves the largest gain for the smallest amount of noise (bolded), and it always achieves a positive gain. For each amount of miscalibration, the confidence-driven approach achieves a higher gain in effective sample size than the non-adaptive approach; it also achieves near 90% coverage or higher in each setting. In contrast, LLM-only coverage is poor. Gain in effective sample size is not estimated for the LLM-only approach as it does not leverage human annotations.