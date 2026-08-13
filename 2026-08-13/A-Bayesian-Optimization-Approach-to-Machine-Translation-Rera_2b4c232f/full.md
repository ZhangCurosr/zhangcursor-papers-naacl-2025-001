# A Bayesian Optimization Approach to Machine Translation Reranking

Julius Cheng<sup>1</sup> Maike Züfle<sup>2</sup> Vilém Zouhar<sup>3</sup> Andreas Vlachos<sup>1</sup>

<sup>1</sup>University of Cambridge <sup>2</sup>Karlsruhe Institute of Technology <sup>3</sup>ETH Zürich {jncc3,av308}@cam.ac.uk maike.zuefle@kit.edu vzouhar@ethz.ch

## Abstract

Reranking, or scoring a list of prediction candidates from a machine translation system with an external scoring model and returning the highest-scoring candidate, remains a simple and effective method for improving prediction quality. However, reranking with high quality scoring models can add substantial computational cost to the translation pipeline, which we address in this work by framing list reranking as a Bayesian optimization (BayesOpt) problem over the candidate list, where unknown scores are modeled with a Gaussian process. This algorithm scores candidates iteratively, choosing next candidates by balancing between exploration, choosing to score those that differ from candidates already scored, and exploitation, choosing to score those that resemble high-scoring candidates. This procedure finds high-scoring candidates while scoring only a fraction of the candidates list; given candidate lists of 200 random samples (before deduplication), our method achieves the same CometKiwi score using only 70 scoring evaluations on average compared to scoring a random subset of 180 candidates. We also propose multi-fidelity BayesOpt for list reranking, where scores obtained from a noisier but cheaper proxy scoring model are incorporated into the search process. We show that welltrained distilled proxy scorers can further improve the performance of BayesOpt.

## 1 Introduction

Reranking is a framework for prediction where probabilistic generator model produces a list of candidates, and a separate evaluator or scoring model produces scores for each of the candidates which are use to determine the final prediction. Reranking has a long history in natural language processing for sequential prediction problems such as dependency parsing (Collins and Koo, 2005;

![](images/ad3c6f7ce15825e6d502702a68a6536ed5740e7c4ccc40006692e4905e46304e.jpg)  
Figure 1: A machine translation system generates candidates Aa, Bb, Cc, Dd, and Ee. The goal of BayesOpt is to find the highest scoring candidate with fewer scoring calls. An acquisition function selects the next candidate to score repeatedly until budget is reached, and the candidate with the highest score so far is returned.

Charniak and Johnson, 2005) and language modeling problems such as summarization (Ravaut et al., 2022) and machine translation (MT; Fernandes et al., 2022).

The quality of models for automatic MT evaluation has surged in recent years due to innovations in neural network architecture (Rei et al., 2020; Juraska et al., 2023; Sellam et al., 2020) as well as the abundance of training data (Freitag et al., 2023b; Kocmi et al., 2024a). These evaluation models are often repurposed for reranking to further improve the performance of an MT system. For instance, in the WMT 2024 shared task (Kocmi et al., 2024a), 5 out of 19 systems, including the overall best submission (Rei et al., 2024) use reranking with Comet models (Rei et al., 2020) and/or minimum Bayes risk decoding (MBR; Eikema and Aziz, 2020), which can be interpreted as a form of reranking. Prior to the application of automatic evaluation metrics to reranking, other scoring methods have been proposed, including discriminatively trained classifiers (Lee et al., 2021; Bhattacharyya et al., 2021) and noisy channel decoding (Yee et al., 2019).

So, while LMs for MT generation for greatly improved in recent years, scoring models have seen a commensurate increase in quality (Zerva et al., 2022), and thus reranking remains relevant method for improving translation quality. However, the scoring models have also grown dramatically in size, increasing the computational requirements for reranking.

In this work, we address the computational cost of reranking by framing it as a search problem over the list of candidates. The goal of search in this setting is to find high-scoring candidates in a small number of steps, thereby avoiding the cost of scoring the full list. Our proposed algorithm uses Gaussian processes to model uncertainty about unseen scores and Bayesian optimization (BayesOpt; Shahriari et al., 2016) to choose which candidates to score next.

GPs are flexible priors over functions which are able to model the complex and nonlinear relationship between each candidate and its score. GPs make very few assumptions about the distribution and base their predictions are mostly on observed points, which enables them to easily adapt to different candidate lists across translation instances. BayesOpt is a sequential black-box optimization method that uses the posterior mean and variance of unobserved data points to decide which points to evaluate next.

We apply BayesOpt and GPs (BayesOpt+GP) to MT list reranking in a straightforward manner and show that it obtains close to the maximum achievable score with only a fraction of score evaluations. For example, the maximal obtainable score across 200 randomly sampled candidates on our test set is 0.8216 CometKiwi; our method achieves 0.8210 with 70 score evaluations on average, while scoring 70 random candidates attains 0.8149, a difference of 0.0061 which is likely to be human-detectable according to (Kocmi et al., 2024b). We also propose a number of search-based baselines which outperform random selection, all of which are outperformed by BayesOpt+GP.

Then, building upon previous works that use a faster but noisier proxy scoring function to prune the candidate list (Fernandes et al., 2022; Eikema and Aziz, 2022), we propose a multi-fidelity extension to BayesOpt which incorporates proxy scores to improve estimation. This is related in motivation to coarse-to-fine methods (Petrov, 2011) and model cascading (Chen et al., 2023), where the use of a faster proxy model reduces the use of the main model. In our multi-fidelity experiments, we find that smaller proxy scoring models distilled from the main model can assist BayesOpt+GP in finding high-scoring candidates earlier.

## 2 Background

## 2.1 Translation generation and reranking

In a typical machine translation setting, a conditional language model (LM) is trained to model the probability of the next token y<sub>t</sub> given a source sentence x and previous tokens: $p ( y _ { t } | x , y _ { 1 } , . . . , y _ { t - 1 } )$ These probabilities can be autoregressively combined to model a sequence probability $p ( y | x )$ . Usually, beam search is used to search for a y which maximizes log probability combined with a length normalization objective (Wu et al., 2016).

In a basic list reranking setting, given $x ,$ the LM is used to generate a candidate list $\mathcal { C } _ { x } = [ y _ { 1 } , . . . , y _ { n } ]$ with a decoding algorithm such as beam search or ancestral sampling. A scoring function $s ( x , y _ { i } )$ is then applied to each $y _ { i } \in { \mathcal { C } } _ { x }$ , and the best scoring sequence arg $\mathrm { m a x } _ { y _ { i } \in \mathcal { C } _ { x } } s ( x , y _ { i } )$ is returned. A common choice of scoring function is a quality estimation (QE) model which directly predicts a scalar value representing the quality.

Reranking with high-quality evaluation metrics has been shown to be highly effective at improving translation output (Freitag et al., 2022), though it can skew results when the same metric is also used for evaluation (Kocmi et al., 2024a). Reranking performance improves as the number of candidates increases (Vernikos and Popescu-Belis, 2024) and when multiple scoring metrics are combined to form a stronger prediction (Fernandes et al., 2022).

Reranking adds significant computational costs to prediction and may be prohibitive to use at test time, but it can be used to benefit LM training instead of test time prediction; high-quality predictions obtained from reranking can be used for knowledge distillation (Wang et al., 2024) and selftraining (Finkelstein et al., 2024). Such methods can improve the performance of an MT system without additional costs during test time.

Previous work on efficient reranking for MT is relatively limited. Fernandes et al. (2022) and Eikema and Aziz (2022) perform a two-stage reranking by first pruning with a faster and noisier scoring function to a fixed size before evaluating the target score. There has been recent interest in efficient approximations for MBR (Cheng and Vlachos, 2023; Deguchi et al., 2024; Trabelsi et al., 2024; Vamvas and Sennrich, 2024), but these methods are not applicable to general scoring functions. (Singhal et al., 2023) propose to represent the candidate space compactly in a lattice over which a token-level reranker can efficiently score many candidates. In this work, we attempt to address a more general setting: the reranking of candidate lists with arbitrary black-box scoring functions.

## 2.2 Bayesian optimization with Gaussian process prior

Bayesian optimization is a sequential algorithm for optimizing a black-box function $f . \ f$ is assumed to be drawn from a prior distribution over functions. The main loop of BayesOpt is as follows: given a set of (possibly noisy) observations of $f ( a _ { 1 } ) , . . . , f ( a _ { i } )$ , the prior distribution over $f$ is updated to a posterior distribution with Bayes theorem. An acquisition function determines a query point $a _ { i + 1 }$ at which to evaluate $f$ next. $f ( a _ { i + 1 } )$ is evaluated and added to the set of observations. This repeats until a stopping criteria is reached. The principal design choices in BayesOpt are the prior distribution of f and the acquisition function.

A common choice of prior is the Gaussian process, which assumes that any subset of points $f ( a _ { 1 } ) , . . . , f ( a _ { i } )$ are drawn jointly from a multivariate Gaussian distribution $\mathcal { N } ( \boldsymbol { \mu } , \boldsymbol { \kappa } )$ , where is the covariance matrix defined by a kernel function such as the radial basis function kernel (RBF). RBFs define the covariance of two points a and a′ as:

$$
{ \mathcal { K } } _ { \mathrm { R B F } } ( a , a ^ { \prime } ) = \exp \left( - \frac { | | a - a ^ { \prime } | | ^ { 2 } } { 2 w ^ { 2 } } \right) ,\tag{1}
$$

where w is the bandwidth hyperparameter which determines scaling. The choice of kernel dictates prior assumptions about the shape of $f ;$ with RBF, points that are closer in Euclidean space have larger covariance. RBFs are a popular choice of kernel due their ability to adapt to complex nonlinear functions.

The assumption that $f ( a _ { 1 } ) , . . . , f ( a _ { i } )$ are jointly Gaussian gives rise to a convenient posterior distribution. Given a vector of observed data points a and their observed values $f ( \mathbf { a } )$ , the posterior mean $\mu _ { a }$ and variance $\sigma _ { a }$ of a point a are given by the conditional multivariate Gaussian distribution:

$$
\mu _ { a } = \mu + \mathcal { K } ( a , \mathbf { a } ) ( \mathcal { K } ( \mathbf { a } , \mathbf { a } ) + \sigma ^ { 2 } I ) ^ { - 1 } f ( \mathbf { a } )\tag{2}
$$

$$
\begin{array} { c } { { \sigma _ { a } = \mathcal { K } ( a , a ) + { \sigma } ^ { 2 } - } } \\ { { \mathcal { K } ( a , { \bf a } ) ( \mathcal { K } ( { \bf a } , { \bf a } ) + { \sigma } ^ { 2 } I ) { } ^ { - 1 } \mathcal { K } ( { \bf a } , a ) } } \end{array}\tag{3}
$$

where $\mu$ is the unconditional mean of the distribution, $\sigma ^ { 2 }$ is a constant Gaussian noise on observations, I is the identity matrix, and here returns elementwise kernel values when given vector arguments.

The acquisition function in BayesOpt is the strategy for selecting the next point to evaluate in the optimization process. Acquisition functions can seek the highest expected improvement (EI; Mockus, 1974), an upper confidence bound if the scores are noisy (Srinivas et al., 2009), or information gain (Hennig and Schuler, 2011). We use EI, defined as:

$$
\alpha ( a ) = \mathbb { E } [ \operatorname* { m a x } ( f ( a ) - f ( a ^ { + } ) , 0 ) ] ,\tag{4}
$$

where $a ^ { + }$ is the location of the current best observation. When $f$ is Gaussian and there is no observation noise, this has the following closed-form solution (Jones, 2001):

$$
\alpha ( a ) = \sigma _ { a } ( z \cdot \mathrm { c d f } ( z ) + \mathrm { p d f } ( z ) ) ,\tag{5}
$$

where $\begin{array} { r } { z = \frac { f ( a ^ { + } ) - \mu _ { a } } { \sigma _ { a } } } \end{array}$ , and cdf, pdf are the Gaussian cumulative distribution function and probability density function, respectively. EI encourages both exploration of uncertain points and exploitation of high-scoring points; the quantity in Equation 5 can be increased by increasing $\mu _ { a } \mathrm { o r } \sigma _ { a }$

The generality of BayesOpt and modeling freedom enjoyed by GPs make them suitable for a great variety of tasks, including spatial monitoring (Krause et al., 2008) and hyperparameter optimisation (Bergstra et al., 2011). GPs have been applied to text regression tasks (Beck et al., 2013, 2014; Beck and Cohn, 2017), but they are not as wellstudied in NLP compared to many other domains.

## 3 Methods

## 3.1 MT reranking with Bayesian optimization

Our main algorithm is an adaptation of BayesOpt with GPs as described in Section 2.2 to the reranking setting. Each source sentence x and its associated candidate list is treated as a standalone BayesOpt problem, meaning that no observations are shared across different x. Thus for brevity, we omit $x$ from notation when discussing BayesOpt for a particular instance.

Inputs: main metric $s ,$ proxy metric $s ^ { \prime } ,$ budget n for evaluating s, hypotheses ${ \mathcal { C } } ,$ number of initial main   
scores $\alpha ,$ number of initial proxy scores $\beta ,$ scoring budget $n ,$ batch size $k ,$ precomputed multi-fidelity   
kernel ${ { \cal { K } } _ { m u l t } } .$   
Output: hypothesis with the highest observed score arg $\operatorname* { m a x } _ { y \in \mathcal { C } _ { \mathrm { o b s } } } s ( y )$   
1: $\mathcal { C } _ { \mathrm { o b s } } ^ { \prime }  \big ( \textstyle { \operatorname * { m i n } } ( \beta , | \mathcal { C } | ) \big ) , \mathcal { C } _ { \mathrm { o b s } }  \big ( \textstyle { \operatorname * { m i n } } ( \alpha , | \mathcal { C } | ) \big )$ ▷ Sample initial subsets   
2: $S _ { \mathrm { o b s } } \gets \{ s ( y ) | y \in \mathcal { C } _ { \mathrm { o b s } } \}$ ▷ Compute scores for main scoring function   
3: $S _ { \mathrm { o b s } } ^ { \prime }  \{ s ^ { \prime } ( y ) | y \in \mathcal { C } _ { \mathrm { o b s } } ^ { \prime } \}$ ▷ Compute proxy scores   
4: while $| { \mathcal { C } } _ { \mathrm { o b s } } | < n$ and $| \mathcal { C } _ { \mathrm { o b s } } | < | \mathcal { C } |$ do   
5: $\bar { \mathcal { C } } _ { \mathrm { o b s } } \gets \mathcal { C } \setminus \mathcal { C } _ { \mathrm { o b s } }$ ▷ Get complement of $\mathcal { C } _ { \mathrm { o b s } }$   
6: $\hat { S } \gets \mathrm { N o r m } ( S _ { \mathrm { o b s } } ) , \hat { S } ^ { \prime } \gets \mathrm { N o r m } ( S _ { \mathrm { o b s } } ^ { \prime } )$ ▷ Normalize observed scores to 0 mean, 1 variance   
7: $y _ { \mathrm { b e s t } }  \arg \operatorname* { m a x } _ { y \in \mathcal { C } _ { \mathrm { o b s } } } \hat { S } ( y )$ ▷ Get best observed point   
8: $\forall y \in \bar { \mathcal { C } } _ { \mathrm { o b s } } : \mu _ { y } , \sigma _ { y } \gets$ calculate posterior using $y , \kappa _ { \mathrm { m u l t } } , \hat { S } , \hat { S } ^ { \prime }$   
▷ GP posterior as in Equations (2) and (3)   
9: $\forall y \in \bar { \mathcal { C } } _ { \mathrm { o b s } } : \gamma _ { y } \gets \mathrm { E I } ( y _ { \mathrm { b e s t } } , \mu _ { y } , \sigma _ { y } )$ ▷ Expected improvement as in Equation (5)   
10: $\mathcal { C } _ { \mathrm { t o p } - k }  \mathrm { a r g } \mathrm { t o p k } _ { y \in \bar { \mathcal { C } } _ { \mathrm { o b s } } } \gamma _ { y }$ ▷ Select k best hypotheses based on EI   
11: $S _ { \mathrm { o b s } }  S _ { \mathrm { o b s } } \cup \{ s ( y ) | y \in \mathcal { C } _ { \mathrm { t o p } - k } \}$ ▷ Compute scores for selected hypotheses   
12: $\mathcal { C } _ { \mathrm { o b s } }  \mathcal { C } _ { \mathrm { o b s } } \cup \mathcal { C } _ { \mathrm { t o p } - k }$ ▷ Update observed hypotheses   
13: end while   
14: return arg $\operatorname* { m a x } _ { y \in \mathcal { C } _ { \mathrm { o b s } } } s ( y )$  
Algorithm 1: The BayesOpt+GP+P algorithm. BayesOpt+GP is a special case of this where $\beta = 0 .$

Let s be the scoring function, an MT quality estimator. Let  be a set of candidates, ${ \mathcal { C } } _ { \mathrm { o b s } } \subseteq { \mathcal { C } }$ the subset of candidates for which we have observed $s ( y )$ , and $\bar { \mathcal { C } } _ { \mathrm { o b s } }$ be all other y $( \bar { \mathcal { C } } _ { \mathrm { o b s } } = \mathcal { C } \setminus \mathcal { C } _ { \mathrm { o b s } } )$ . To perform reranking for an instance, we first generate candidates and initialize the algorithm by scoring a random α-sized subset of the list with s. In one iteration in the algorithm loop, we normalize the observed scores to mean 0 and 1 variance at every step and assume a 0 unconditional mean. Then we compute the GP posterior of all $y \in \bar { \mathcal { C } } _ { \mathrm { o b s } }$ with Equation 2 and 3 given the scores of $\mathcal { C } _ { \mathrm { o b s } }$ , which is then used to compute EI with Equation 5, assuming no observation noise. We score the k candidates in $\bar { \mathcal { C } } _ { \mathrm { o b s } }$ with the highest EI, adding them to $\mathcal { C } _ { \mathrm { o b s } }$ (as well as removing them from $\bar { \mathcal { C } } _ { \mathrm { o b s } } )$ , and repeat the loop, terminating when a predefined budget of n calls to s is reached (or when all candidates have been evaluated, in the case that $| { \mathcal { C } } | \leq n . )$ . Finally, we choose arg $\operatorname* { m a x } _ { y \in \mathcal { C } _ { \mathrm { o b s } } } s ( y )$ as the prediction.

We now describe our choice of GP kernel. $y \in { \mathcal { C } }$ are strings, and we seek a representation that is fast to compute and to compare, since representations are generated, and the computing the GP covariance matrix requires $| { \mathcal { C } } | ^ { 2 }$ comparisons. Our kernel is ${ \mathcal { K } } _ { \mathrm { M T } } ( y _ { i } , y _ { j } ) = { \mathcal { K } } _ { \mathrm { R B F } } ( \operatorname { e m b } ( y _ { i } ) , \operatorname { e m b } ( y _ { j } ) )$

where emb returns the mean-pooled token-level outputs of the final decoder layer when generating $y ,$ normalized to the unit norm after pooling. emb uses meaning representations produced automatically during candidate list generation, so the additional cost to compute it is negligible. Also, the covariance matrix is fast to compute given the candidate list sizes and embedding dimensionality used in our experiments.

## 3.2 Multi-fidelity BayesOpt

We also propose an extension to BayesOpt+GP for the setting where observations are available from a different but related proxy score function $s ^ { \prime } .$ . We refer to this as BayesOpt+GP+P. s′ is assumed to have non-trivial covariance with the scoring model s and to be cheaper to evaluate. This is known as multi-fidelity BayesOpt in the literature, but while the multi-fidelity settings of Kandasamy et al. (2016); Wu et al. (2020) use acquisition functions that may choose to evaluate lower-fidelity scores, we study a simpler setting: $\beta$ observations of $s ^ { \prime }$ are obtained at the start where $\beta > \alpha$ , and only s may be evaluated during the BayesOpt loop. In the multi-fidelity setting, observations are made on $\langle y _ { i } , s _ { i } \rangle$ , a combination of a data point and scoring function, instead of the data point alone.

Our kernel for BayesOpt+GP+P is the product of the RBF kernel from Section 3.1 and a kernel over score functions $f \colon$

$$
\begin{array} { r } { \mathcal { K } _ { \mathrm { m u l t } } \big ( \langle y _ { i } , s _ { k } \rangle , \langle y _ { j } , s _ { l } \rangle \big ) = \qquad } \\ { \mathcal { K } _ { \mathrm { M T } } ( y _ { i } , y _ { j } ) \mathcal { K } _ { \mathrm { s c o r e } } ( s _ { k } , s _ { l } ) . } \end{array}\tag{6}
$$

${ \boldsymbol { \kappa } } _ { \mathrm { m u l t } }$ is a valid kernel because a product of two kernels defined on different spaces is also a kernel (Rasmussen and Williams, 2005). With ${ \boldsymbol { \kappa } } _ { \mathrm { m u l t } }$ , the covariance between two observations depends on both the difference between scoring functions and the distance between data points. This way, an observation influences the posterior for all other data points at all choices of scoring function, as long as the scoring functions are correlated. This formulation enables the use of any number of scoring functions, but in this work, we consider at most two: the main scorer s and a proxy scorer $s ^ { \prime } .$

We set $\mathcal { K } _ { \mathrm { s c o r e } } ( s _ { k } , s _ { l } )$ to be the empirical covariance between $s _ { k }$ and $s _ { l }$ measured over a validation set, where all scores are normalized per-instance so that in each instance, the scores of all candidates for a particular scorer have 0 mean and 1 variance. Then for each scoring function, concatenate all candidate scores across instances, and compare the resulting lists to obtain the covariance. Covariance is a valid kernel because the covariance calculation can be expressed as a dot product, and dot products are valid kernels.

Proxy scores are incorporated into posterior estimation given by Equations 2 and 3 by redefining a to be a tuple of data point, scoring function and a to be a vector of such tuples. The kernel  is set to ${ \boldsymbol { \kappa } } _ { \mathrm { m u l t } }$ which takes as input two tuples of data point and scoring function. The full BayesOpt+GP+P algorithm is in Algorithm 1.

## 3.3 Proxy scores

We train smaller scoring models to have high covariance with s for use in BayesOpt+GP+P. In this work, our scoring functions are based on the Comet referenceless quality estimation architecture (Rei et al., 2020), also known as CometKiwi. These models encode the source and hypothesis jointly with a bidirectional transformer. Activations from all transformer layers are pooled to form a fixed-size representation, which is passed to a feed-forward regression head. The vast majority of computation in this models is spent in the encoder. Thus, faster Comet models can be obtained by reducing the size of the encoder.

We train Comet models using two differently sized pretrained multilingual encoder models in two ways: (1) training on the same training set as CometKiwi and (2) distillation. Among distillation methods, we attempt in preliminary experiments (1) training on the same training set as CometKiwi with ground truth scores replaced with CometKiwi scores and (2) training on a synthetic dataset comprising of LM samples along with their associated CometKiwi scores. The latter achieves higher correlation with CometKiwi on sampled candidates, which is to be expected since the training distribution is more suitable for the reranking use case. We therefore use this latter distillation method for all subsequent experiments. A similar procedure has been described in Rei et al. (2022a).

## 3.4 Candidate list generation

In preliminary experiments, we consider generating the candidate list using beam search with 128 outputs versus sampling 200 candidates using ϵ- sampling (Hewitt et al., 2022) with $\epsilon = 0 . 0 2 .$ , a setting which effectively balances quality and diversity for MBR (Freitag et al., 2023a). Under beam search, the candidates exhibit high lexical overlap, and while the mean score of candidates is higher, the average maximum score is lower. The effectiveness of truncated sampling over beam search in larger conditional language model has also been observed by Fernandes et al. (2022).

Furthermore, beam search suffers from out-ofmemory errors on long translations, whereas with sampling, we simply reduce the batch size when out of memory. While it is possible to implement beam search in a batched manner, this does not exist in any popular conditional language model libraries, to the best of our knowledge.

For these reasons, we generate 200 candidates per instance with ϵ-sampling, $\epsilon = 0 . 0 2$ in all experiments. The sampled candidate list is then deduplicated, resulting in 178 candidates on average per instance.

## 4 Experiments

We now discuss the details and findings of our Bayesian optimization experiments, followed by analysis of our trained proxy scoring models, concluding with runtime measurements. All run time values are measured on a A100-SXM4-40GB GPU. For exact values for figures in this section, see Table 3 in the Appendix. Appendix A contains extensive statistical significance tests.

![](images/f22d6ea7d53a463cf0462026acaad734dc9f0f30787c337e0770f199e698f0dc.jpg)

![](images/e163d3384d8ee17b51775983d2e3c60616085398128520b4802290670cab9920.jpg)  
Figure 2: Left: Performance of reranking methods measured as the average CometKiwi score of the selected candidate. Beam search with beam size 5 achieves a score of 0.754 and is too low to be pictured here. Right: percentage of instances where the selected candidate had the highest score (right). The x-axis is the scoring budget. Legends show the normalized area under the curve of CometKiwi score of each method in brackets.

For BayesOpt experiments, we grid search for the optimal value of RBF bandwidth parameter w on the entire validation set, setting scoring budget n = 100 and batch size $k = 1$ . While it is possible to optimize it for every unique combination of language pair, $n , k ,$ proxy scoring function, and $\beta ,$ we find that the results are not statistically significantly different within a range of settings. For simplicity, and to demonstrate the robustness of our methods, we use the same w for all experiments.

In all experiments, we use α = 10 initial randomly scored candidates. We set k = 1 in Sections 4.2 and 4.3 to demonstrate the effectiveness of BayesOpt+GP under ideal conditions, but since k can have a large impact on speed, we experiment with varying it in Section 4.5.

## 4.1 Models and datasets

For candidate generation, we use the 600Mparameter distilled NLLB model (Team et al., 2022) in all experiments. For the main scoring model, we use CometKiwi-22 (Rei et al., 2022b).

As a dataset used for proxy model training, we use data from the WMT Metrics Shared Task up to 2022 (Freitag et al., 2023b), which contains tuples of source, hypothesis, human score . The human scores were largely collected with the DA+SQM annotation protocol (Kocmi et al., 2022).

For BayesOpt experiments, we select the first 1000 and 500 source sentences per language pair from the WMT23 Metrics Shared Task dataset as the validation and test set, respectively, for 7 language pairs: English-Czech, English-German,

English-Japanese, English-Chinese, and the reverse directions of the latter 3 pairs.

CometKiwi is based on the encoder of XLM-$_ { \mathrm { R o b e r t a } _ { \mathrm { l a r g e } } }$ (Conneau et al., 2019) (2.2GB memory). For proxy scorers we train smaller models based on XLM-Roberta<sub>base</sub> (1.1GB), and Multilingual-MiniLM-L12-H384 (Wang et al., 2020) (469MB).

## 4.2 BayesOpt+GP

The goal of reranking BayesOpt+GP is to improve the speed by only evaluating a subset of available candidates. We evaluate this through quality-cost tradeoff curves, where quality is determined by final selected candidate’s CometKiwi score, and cost is determined by the number of calls to the scoring function. As another measure of approximation quality, we also show the percentage of instances in which the actual best scoring candidate is returned. We devise several baselines with which to compare BayesOpt+GP. Each is a strategy for selecting a subset of candidates to score from which the best scoring candidate is returned. The baselines are:

• UniqRandom: Shuffle the candidate list before de-duplication, then de-duplicate while preserving the order of the first appearance of each candidate. Select the first min(n, ) candidates in the resulting list.

• Logprob{Avg,Sum}: Sort in order of negative sequence log probability (either average or sum), and then select the first min(n, ).

• HillClimbing: Let $y ^ { + }$ be the highest scoring observation point at any time step. Iteratively select arg min $\boldsymbol { \cdot } _ { y \in \mathcal { \bar { C } } _ { \mathrm { o b s } } } | | \mathrm { e m b } ( y ) - \mathrm { e m b } ( y ^ { + } ) | |$ as the next observation point until min(n, ) candidates are scored.

![](images/147a2e26f57d9d6591c1b4fe542e047bd8962f18581b6bd48063da946e53900d.jpg)

![](images/aeddfabff17d050d0d143e60522294dfeadc4296290eaaca47ae433b0cda53c2.jpg)

Figure 3: Average CometKiwi score of the selected top candidate (y-axis) for BayesOpt+GP+P with Distilled-S (left) and Distilled-M (right) compared to the ProxyFirst baseline. This figure disregards the additional compute costs for these proxy metrics in order to show the marginal score increase from proxy observations.  
![](images/476673a14f23af4f319c9d52e2eaad07e14256bfc7a8e8473428286d91b83d0c.jpg)  
Figure 4: Average CometKiwi score of the selected candidate (y-axis) for BayesOpt+GP+P with different choices of proxy score.

UniqRandom simulates the effect of iteratively sampling candidates until n unique candidates are obtained. LogprobFirst{Avg,Sum} are included to verify whether more advanced methods indeed outperform simple subset selection using statistics obtained for free. HillClimbing is a heuristic iterative selection strategy which, like BayesOpt, is black-box and derivative-free (Conn et al., 2009).

In Figure 2, BayesOpt+GP outperforms all baselines, and HillClimbing is the best among the baselines, with LogprobAvg following behind. LogprobSum severely underperforms UniqRandom in score, confirming findings on the inadequacy of very high probability translations (Eikema and

Aziz, 2020). Informally speaking, UniqRandom is a simple “exploration” strategy that ignores existing observations, while HillClimbing is a simple “exploitation” strategy, only searching over neighbors nearest the best observation while ignoring the full search space. These results confirm that balancing these respective deciderata helps to find the optimal candidate more efficiently.

## 4.3 BayesOpt+GP+P

## 4.3.1 Proxy score evaluation

We first evaluate trained proxy scorers independently of their use in BayesOpt according to (1) actual runtime, (2) correlation with human ratings in the WMT23 dataset, (3) correlation with CometKiwi on source-hypothesis pairs in WMT23, and (4) correlation with CometKiwi on a synthetic candidates for an instance, averaged over instances. For correlations we use Kendall’s $\tau _ { c } ,$ which is commonly used in MT metric evaluation (Freitag et al., 2023b).

Table 1 shows the results for the proxy models. The model size corresponds closely to inference time. As desired, training proxies using distillation results in much higher correlation with CometKiwi, although it loses some correlation with human judgments. In subsequent experiments, we consider Distilled-{S,M} only. While LogprobAvg has comparatively much lower correlation, we nevertheless consider it as a proxy score since it is obtained for free during candidate generation.

<table><tr><td colspan="3"></td><td colspan="2">CometKiwi</td></tr><tr><td>Model</td><td>Time</td><td>Human Test</td><td>Test</td><td>Cands.</td></tr><tr><td>CometKiwi</td><td>51.38s</td><td>0.245</td><td>1.000</td><td>1.000</td></tr><tr><td>Logprobs  $\mathbf { A v \mathbf { g } }$ </td><td>0.00s</td><td>I</td><td></td><td>0.191</td></tr><tr><td>LogprobsSum</td><td>0.00s</td><td>■</td><td></td><td>-0.090</td></tr><tr><td>Authentic-S</td><td>7.13s</td><td>0.193</td><td>0.314</td><td>0.350</td></tr><tr><td>Authentic-M</td><td>18.71s</td><td>0.199</td><td>0.320</td><td>0.448</td></tr><tr><td>Distilled-S</td><td>7.13s</td><td>0.169</td><td>0.488</td><td>0.620</td></tr><tr><td>Distilled-M</td><td>18.71s</td><td>0.188</td><td>0.572</td><td>0.680</td></tr></table>

Table 1: Benchmarking proxy models $( { \mathrm { S e c - } }$ tion 3.3) on speed and correlation with human judgments/CometKiwi using the WMT23 dataset. Speed is measured by runtime per 10000 samples using maximum batch size. Correlation is measured with Kendall’s $\tau _ { c }$ against human judgments and CometKiwi scores. CometKiwi correlation is taken over the provided targets in WMT23 (Test) and a synthetic dataset comprised of 200 samples per source sentence, deduped (Cands). Logprobs{Avg,Sum} is not evaluated on WMT23 targets because they are generated by other MT systems.

## 4.3.2 Reranking results

When $s ^ { \prime }$ is sufficiently fast and correlated with $s ,$ it can further improve the quality-cost tradeoff in BayesOpt+GP. Recall that BayesOpt+GP+P initializes with $\beta$ evaluations of $s ^ { \prime } .$ Figure 4 shows the quality-cost curve when all proxy scores are known, or $\beta = 2 0 0$ . The relative performance when including proxy scores correspond to their correlation with CometKiwi as shown in Table 1; Distilled-M outperforms Distilled-S, and both outperform LogprobAvg. This demonstrates the importance of ensuring high correlation in the proxy score. The addition of LogprobAvg to BayesOpt+GP has little effect, showing that poorly correlated proxies are too noisy to help and may even hinder performance. Beyond $n = 7 0$ , all methods achieve close to the maximum attainable score.

We also examine the effect of initializing with a fraction of proxy observations rather than all of them. For some choice of $\beta _ { v }$ , an appropriate baseline is to rank the top-n candidates among the $\beta$ observed proxy scores. We call this ProxyFirst. The results when using Distilled-M and Distilled-S as proxies are shown in Figure 3. In both cases, the difference between BayesOpt+GP+P and ProxyFirst is smaller when $\beta = 2 0 0$ than when $\beta = 5 0$ and this gap is smaller for Distilled-M. This is to be expected because as the covariance of s and $s ^ { \prime }$ increases, using ProxyFirst with $\beta = 2 0 0$ approaches standard full-list reranking. The marginal benefit of BayesOpt+GP+P is more clear when $\beta = 5 0 .$ where proxy scores help to find promising candidates earlier.

Overall, proxy observations can indeed improve quality for a particular $n .$ However, for sufficiently large $n ,$ BayesOpt+GP converges, so proxy observations are unnecessary. Proxy evaluations add to the runtime cost which we discuss in Section 4.4. Therefore, while we show that the multi-fidelity kernel is capable of leveraging proxy scores to improve search, in practice, the overall computational budget should be considered along with the quality and cost of the proxy scoring function to ensure that using the method is worthwhile.

## 4.4 Runtime

Our reranking algorithm significantly reduces actual runtime compared to scoring all candidates for a source sentence. We profile the full pipeline, from generating candidates to making a final selection, on three settings: (1) BayesOpt+GP with $n = 9 0$ , and (2) multi-fidelity BayesOpt+GP with 50 Distilled-S scores and $n = 7 0$ , and 3) the baseline of evaluating CometKiwi on all candidates. $n , \beta$ are selected to balance the final scores of the two algorithms (0.8213 and 0.8211 respectively, as shown in Table 3).

For the runtime calculations, we select 50 source sentences from each language pair and generate 200 candidates for each. For the baseline, we compute scores for all candidates with a batch size of 200. For BayesOpt+GP methods, we profile the additional steps required: computing the kernel, computing the posteriors at each step, and evaluating proxy scores. BayesOpt+GP(+S) uses batch size $k = 1 0 .$ , which does not affect scores compared to using $k = 1$ (see Section 4.5). Memory bandwidth can be a major overhead in large neural networks, making it inefficient to run small batches. Since BayesOpt+GP obtains k candidates per step, in order to use large batches, we process candidates for multiple instances in parallel.

Results are shown in Table 2. In all cases, candidate generation and CometKiwi calculations dominate the overall runtime. The extra cost from BayesOpt-related computations is compensated by the savings from reducing CometKiwi evaluations, despite similarity matrix computation being $\mathcal { O } ( | \mathcal { C } | ^ { 2 } )$ , and matrix inversion for posterior calculation at each iteration being $\mathcal { O } ( | \mathcal { C } | ^ { 3 } )$

BayesOpt+GP+P with Distilled-S reduces the runtime by further reducing the number of CometKiwi calculations to 70, with the cost of loading and running the Distilled-S proxy metric introducing minimal overhead.

<table><tr><td rowspan="2">Operation</td><td rowspan="2">AllComet</td><td rowspan="2">BayesOpt</td><td rowspan="2">BayesOpt +GP+P n = 70, β = 50</td></tr><tr><td>+GP n = 90</td></tr><tr><td>Candidates</td><td>701.38</td><td>701.38</td><td>701.38</td></tr><tr><td>Similarities</td><td></td><td>1.24</td><td>1.24</td></tr><tr><td>BayesOpt+GP</td><td></td><td>1.92</td><td>2.25</td></tr><tr><td>Comet Loading</td><td>8.43</td><td>8.43</td><td>11.27</td></tr><tr><td>Distilled-S</td><td></td><td></td><td>11.11</td></tr><tr><td>CometKiwi</td><td>274.87</td><td>188.39</td><td>146.33</td></tr><tr><td>Total</td><td>984.68</td><td>901.36</td><td>873.58</td></tr></table>

Table 2: Runtimes for the full reranking baseline (AllComet), BayesOpt+GP, and BayesOpt+GP+P with Distilled-S as proxy score at settings where CometKiwi scores are roughly equal. Time given in seconds per 350 instances.

## 4.5 Batch size k in BayesOpt+GP

We examine the effect of batch size k in BayesOpt+GP for $k = 1 , 2 , 5 , 1 0$ . Figure 5 shows that as expected, larger k diminishes performance, although the differences nearly vanish at n>70.

k impacts how often the BayesOpt loop is run and thus has a large effect on speed. Fortunately, we observe for sufficiently large n, k can be increased without sacrificing quality.

![](images/0045d208e763fc1acaf588234e27baab7d6930f6811058f4527069857acae90d.jpg)  
Figure 5: Difference between BayesOpt+GP with batch size of 1 (top line in red in Figure 2) and BayesOpt+GP with higher batch sizes. Negative values mean that higher batch size performed worse than BayesOpt+GP with batch size of 1.

## 5 Conclusion

In this work, we formalize MT reranking as a Bayesian optimization problem, leveraging the basic observation that similar translations are more likely to have similar quality scores. We also extend the framework to accept observations from proxy scoring functions, which is applicable when the target score is very costly: large QE models, MBR, or human evaluation. In realistic experiments, we show that our methods improve reranking efficiency over strong baselines. We also propose several design choices that make the methods useful in practice; a GP kernel that requires minimal overhead, and effective proxy model training via distillation.

We consider our work a first step in applying BayesOpt to MT reranking. Future directions include integrating BayesOpt with candidate generation, alternative acquisition functions, and further exploration of GP kernels for MT.

## 6 Limitations

The optimization problem considered in this work is to maximize score from a scoring model. We show that BayesOpt is an effective optimizer, but we do not explore to what extent the optimization problem is flawed due to flaws in the scoring model. We refer to Kocmi et al. (2024b) to understand what magnitude of score difference between systems is significant. However, the existence of “metric overfitting” when directly optimizing an evaluation metric is debated and may affect the interpretation of score differences (Fernandes et al., 2022; Wang et al., 2024).

BayesOpt+GP requires matrix inversion, a ( <sup>3</sup>) operation that is performed once per iteration. While it is inexpensive for the we consider, this limits the number of observations that can be used for posterior computation without resorting to approximations (Noack et al., 2023).

As an iterative algorithm, BayesOpt can score no more than k candidates in a batch for a single instance. Small batch sizes introduce a significant bottleneck for large neural networks, so in order to maintain large batch sizes, we propose processing multiple instances in parallel. However, this requires additional engineering.

## Acknowledgements

Julius Cheng is supported by a scholarship from Huawei. Part of this work received support from the European Union’s Horizon research and innovation programme under grant agreement No 101135798, project Meetween (My Personal AI

Mediator for Virtual MEETtings BetWEEN People). We thank the organizers of MT Marathon 2024, where the authors met and this work was conceived. We also thank Béni Egressy for useful discussions and Will Tebbutt for lending expertise on GPs.

## References

Daniel Beck and Trevor Cohn. 2017. Learning kernels over strings using Gaussian processes. In Proceedings ofthe Eighth International Joint Conference on Natural Language Processing (Volume 2: Short Papers), 67–73. Asian Federation of Natural Language Processing.

Daniel Beck, Trevor Cohn, and Lucia Specia. 2014. Joint emotion analysis via multi-task Gaussian processes. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), 1798–1803. Association for Computational Linguistics.

Daniel Beck, Kashif Shah, Trevor Cohn, and Lucia Specia. 2013. SHEF-Lite: When less is more for translation quality estimation. In Proceedings ofthe Eighth Workshop on Statistical Machine Translation, 337–342. Association for Computational Linguistics.

James Bergstra, Rémi Bardenet, Yoshua Bengio, and Balázs Kégl. 2011. Algorithms for hyper-parameter optimization. In Advances in Neural Information Processing Systems, volume 24. Curran Associates, Inc.

Sumanta Bhattacharyya, Amirmohammad Rooshenas, Subhajit Naskar, Simeng Sun, Mohit Iyyer, and Andrew McCallum. 2021. Energy-based reranking: Improving neural machine translation using energybased models. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), 4528–4537. Association for Computational Linguistics.

Eugene Charniak and Mark Johnson. 2005. Coarseto-fine n-best parsing and MaxEnt discriminative reranking. In Proceedings ofthe 43rd Annual Meeting ofthe Associationfor Computational Linguistics (ACL’05), 173–180. Association for Computational Linguistics.

Lingjiao Chen, Matei Zaharia, and James Zou. 2023. Frugalgpt: How to use large language models while reducing cost and improving performance.

Julius Cheng and Andreas Vlachos. 2023. Faster minimum Bayes risk decoding with confidence-based pruning. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 12473–12480. Association for Computational Linguistics.

Michael Collins and Terry Koo. 2005. Discriminative reranking for natural language parsing. Computational Linguistics, 31(1):25–70.

Andrew R. Conn, Katya Scheinberg, and Luis N. Vicente. 2009. Introduction to Derivative-Free Optimization. Society for Industrial and Applied Mathematics.

Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Unsupervised cross-lingual representation learning at scale. CoRR, abs/1911.02116.

Hiroyuki Deguchi, Yusuke Sakai, Hidetaka Kamigaito, Taro Watanabe, Hideki Tanaka, and Masao Utiyama. 2024. Centroid-based efficient minimum Bayes risk decoding. In Findings of the Association for Computational Linguistics ACL 2024, 11009–11018, Bangkok, Thailand and virtual meeting. Association for Computational Linguistics.

Bryan Eikema and Wilker Aziz. 2020. Is MAP decoding all you need? the inadequacy of the mode in neural machine translation. In Proceedings of the 28th International Conference on Computational Linguistics, 4506–4520. International Committee on Computational Linguistics.

Bryan Eikema and Wilker Aziz. 2022. Sampling-based approximations to minimum Bayes risk decoding for neural machine translation. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, 10978–10993. Association for Computational Linguistics.

Patrick Fernandes, António Farinhas, Ricardo Rei, José G. C. de Souza, Perez Ogayo, Graham Neubig, and Andre Martins. 2022. Quality-aware decoding for neural machine translation. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 1396–1412. Association for Computational Linguistics.

Mara Finkelstein, Subhajit Naskar, Mehdi Mirzazadeh, Apurva Shah, and Markus Freitag. 2024. MBR and QE finetuning: Training-time distillation of the best and most expensive decoding methods.

Markus Freitag, Behrooz Ghorbani, and Patrick Fernandes. 2023a. Epsilon sampling rocks: Investigating sampling strategies for minimum Bayes risk decoding for machine translation. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2023, 9198–9209. Association for Computational Linguistics.

Markus Freitag, David Grangier, Qijun Tan, and Bowen Liang. 2022. High quality rather than high model probability: Minimum Bayes risk decoding with neural metrics. Transactions of the Association for Computational Linguistics, 10:811–825.

Markus Freitag, Nitika Mathur, Chi-kiu Lo, Eleftherios Avramidis, Ricardo Rei, Brian Thompson, Tom Kocmi, Frederic Blain, Daniel Deutsch, Craig Stewart, Chrysoula Zerva, Sheila Castilho, Alon Lavie, and George Foster. 2023b. Results of WMT23 metrics shared task: Metrics might be guilty but references are not innocent. In Proceedings ofthe Eighth Conference on Machine Translation, 578–628. Association for Computational Linguistics.

Philipp Hennig and Christian J. Schuler. 2011. Entropy search for information-efficient global optimization. ArXiv, abs/1112.1217.

John Hewitt, Christopher Manning, and Percy Liang. 2022. Truncation sampling as language model desmoothing. In Findings of the Association for Computational Linguistics: EMNLP 2022, 3414–3427. Association for Computational Linguistics.

Donald R. Jones. 2001. A taxonomy of global optimization methods based on response surfaces. Journal of Global Optimization, 21:345–383.

Juraj Juraska, Mara Finkelstein, Daniel Deutsch, Aditya Siddhant, Mehdi Mirzazadeh, and Markus Freitag. 2023. MetricX-23: The Google submission to the WMT 2023 metrics shared task. In Proceedings of the Eighth Conference on Machine Translation, 756– 767. Association for Computational Linguistics.

Kirthevasan Kandasamy, Gautam Dasarathy, Junier B Oliva, Jeff Schneider, and Barnabas Poczos. 2016. Gaussian process bandit optimisation with multifidelity evaluations. In Advances in Neural Information Processing Systems, volume 29. Curran Associates, Inc.

Tom Kocmi, Eleftherios Avramidis, Rachel Bawden, Ondˇrej Bojar, Anton Dvorkovich, Christian Federmann, Mark Fishel, Markus Freitag, Thamme Gowda, Roman Grundkiewicz, Barry Haddow, Marzena Karpinska, Philipp Koehn, Benjamin Marie, Christof Monz, Kenton Murray, Masaaki Nagata, Martin Popel, Maja Popovic, Mariya Shmatova, Steinthór´ Steingrímsson, and Vilém Zouhar. 2024a. Findings of the WMT24 general machine translation shared task: The LLM era is here but MT is not solved yet. In Proceedings ofthe Ninth Conference on Machine Translation, 1–46, Miami, Florida, USA. Association for Computational Linguistics.

Tom Kocmi, Rachel Bawden, Ondˇrej Bojar, Anton Dvorkovich, Christian Federmann, Mark Fishel, Thamme Gowda, Yvette Graham, Roman Grundkiewicz, Barry Haddow, Rebecca Knowles, Philipp Koehn, Christof Monz, Makoto Morishita, Masaaki Nagata, Toshiaki Nakazawa, Michal Novák, Martin Popel, and Maja Popovic. 2022.´ Findings of the 2022 conference on machine translation (WMT22). In Proceedings of the Seventh Conference on Machine Translation (WMT), 1–45. Association for Computational Linguistics.

Tom Kocmi, Vilém Zouhar, Christian Federmann, and Matt Post. 2024b. Navigating the metrics maze: Reconciling score magnitudes and accuracies. In Proceedings of the 62nd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), 1999–2014, Bangkok, Thailand. Association for Computational Linguistics.

Andreas Krause, Ajit Singh, and Carlos Guestrin. 2008. Near-optimal sensor placements in gaussian processes: Theory, efficient algorithms and empirical studies. J. Mach. Learn. Res., 9:235–284.

Ann Lee, Michael Auli, and Marc’Aurelio Ranzato. 2021. Discriminative reranking for neural machine translation. In Proceedings ofthe 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), 7250–7264. Association for Computational Linguistics.

Jonas Mockus. 1974. On bayesian methods for seeking the extremum. In Proceedings ofthe IFIP Technical Conference, 400–404.

Marcus M. Noack, Harinarayan Krishnan, Mark D. Risser, and Kristofer G. Reyes. 2023. Exact gaussian processes for massive datasets via non-stationary sparsity-discovering kernels. Scientific Reports, 13(1).

Slav Petrov. 2011. Coarse-to-Fine Natural Language Processing (Theory and Applications ofNatural Language Processing). Springer Publishing Company, Incorporated.

C.E. Rasmussen and C.K.I. Williams. 2005. Gaussian Processesfor Machine Learning. Adaptive Computation and Machine Learning series. MIT Press.

Mathieu Ravaut, Shafiq Joty, and Nancy Chen. 2022. SummaReranker: A multi-task mixture-of-experts re-ranking framework for abstractive summarization. In Proceedings of the 60th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), 4504–4524. Association for Computational Linguistics.

Ricardo Rei, Ana C Farinha, José G.C. de Souza, Pedro G. Ramos, André F.T. Martins, Luisa Coheur, and Alon Lavie. 2022a. Searching for COMETINHO: The little metric that could. In Proceedings of the 23rd Annual Conference ofthe European Association for Machine Translation, 61–70. European Association for Machine Translation.

Ricardo Rei, Jose Pombal, Nuno M. Guerreiro, João Alves, Pedro Henrique Martins, Patrick Fernandes, Helena Wu, Tania Vaz, Duarte Alves, Amin Farajian, Sweta Agrawal, Antonio Farinhas, José G. C. De Souza, and André Martins. 2024. Tower v2: Unbabel-IST 2024 submission for the general MT shared task. In Proceedings ofthe Ninth Conference on Machine Translation, 185–204, Miami, Florida, USA. Association for Computational Linguistics.

Ricardo Rei, Craig Stewart, Ana C Farinha, and Alon Lavie. 2020. COMET: A neural framework for MT evaluation. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2685–2702. Association for Computational Linguistics.

Ricardo Rei, Marcos Treviso, Nuno M. Guerreiro, Chrysoula Zerva, Ana C Farinha, Christine Maroti, José G. C. de Souza, Taisiya Glushkova, Duarte Alves, Luisa Coheur, Alon Lavie, and André F. T. Martins. 2022b. CometKiwi: IST-unbabel 2022 submission for the quality estimation shared task. In Proceedings ofthe Seventh Conference on Machine Translation (WMT), 634–645. Association for Computational Linguistics.

Thibault Sellam, Dipanjan Das, and Ankur Parikh. 2020. BLEURT: Learning robust metrics for text generation. In Proceedings of the 58th Annual Meeting of the Associationfor Computational Linguistics, 7881– 7892. Association for Computational Linguistics.

Bobak Shahriari, Kevin Swersky, Ziyun Wang, Ryan P. Adams, and Nando de Freitas. 2016. Taking the human out of the loop: A review of bayesian optimization. Proceedings ofthe IEEE, 104:148–175.

Prasann Singhal, Jiacheng Xu, Xi Ye, and Greg Durrett. 2023. EEL: Efficiently encoding lattices for reranking. In Proceedings of the 61st Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), 9299–9316. Association for Computational Linguistics.

Niranjan Srinivas, Andreas Krause, Sham M. Kakade, and Matthias W. Seeger. 2009. Information-theoretic regret bounds for gaussian process optimization in the bandit setting. IEEE Transactions on Information Theory, 58:3250–3265.

NLLB Team, Marta R. Costa-jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe Kalbassi, Janice Lam, Daniel Licht, Jean Maillard, Anna Sun, Skyler Wang, Guillaume Wenzek, Al Youngblood, Bapi Akula, Loic Barrault, Gabriel Mejia Gonzalez, Prangthip Hansanti, John Hoffman, Semarley Jarrett, Kaushik Ram Sadagopan, Dirk Rowe, Shannon Spruit, Chau Tran, Pierre Andrews, Necip Fazil Ayan, Shruti Bhosale, Sergey Edunov, Angela Fan, Cynthia Gao, Vedanuj Goswami, Francisco Guzmán, Philipp Koehn, Alexandre Mourachko, Christophe Ropers, Safiyyah Saleem, Holger Schwenk, and Jeff Wang. 2022. No language left behind: Scaling humancentered machine translation.

Firas Trabelsi, David Vilar, Mara Finkelstein, and Markus Freitag. 2024. Efficient minimum bayes risk decoding using low-rank matrix completion algorithms.

Jannis Vamvas and Rico Sennrich. 2024. Linear-time minimum Bayes risk decoding with reference aggregation. In Proceedings of the 62nd Annual Meeting

of the Association for Computational Linguistics (Volume 2: Short Papers), 790–801, Bangkok, Thailand. Association for Computational Linguistics.

Giorgos Vernikos and Andrei Popescu-Belis. 2024. Don’t rank, combine! combining machine translation hypotheses using quality estimation. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 12087–12105, Bangkok, Thailand. Association for Computational Linguistics.

Jun Wang, Eleftheria Briakou, Hamid Dadkhahi, Rishabh Agarwal, Colin Cherry, and Trevor Cohn. 2024. Don’t throw away data: Better sequence knowledge distillation.

Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou. 2020. MiniLM: Deep selfattention distillation for task-agnostic compression of pre-trained transformers.

Jian Wu, Saul Toscano-Palmerin, Peter I. Frazier, and Andrew Gordon Wilson. 2020. Practical multifidelity bayesian optimization for hyperparameter tuning. In Proceedings of The 35th Uncertainty in Artificial Intelligence Conference, volume 115 of Proceedings of Machine Learning Research, 788–798. PMLR.

Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V. Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, Jeff Klingner, Apurva Shah, Melvin Johnson, Xiaobing Liu, Łukasz Kaiser, Stephan Gouws, Yoshikiyo Kato, Taku Kudo, Hideto Kazawa, Keith Stevens, George Kurian, Nishant Patil, Wei Wang, Cliff Young, Jason Smith, Jason Riesa, Alex Rudnick, Oriol Vinyals, Greg Corrado, Macduff Hughes, and Jeffrey Dean. 2016. Google’s neural machine translation system: Bridging the gap between human and machine translation.

Kyra Yee, Yann Dauphin, and Michael Auli. 2019. Simple and effective noisy channel modeling for neural machine translation. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), 5696–5701. Association for Computational Linguistics.

Chrysoula Zerva, Frédéric Blain, Ricardo Rei, Piyawat Lertvittayakumjorn, José G. C. de Souza, Steffen Eger, Diptesh Kanojia, Duarte Alves, Constantin Orasan, Marina Fomicheva, André F. T. Martins, and˘ Lucia Specia. 2022. Findings of the WMT 2022 shared task on quality estimation. In Proceedings of the Seventh Conference on Machine Translation (WMT), 69–99. Association for Computational Linguistics.

<table><tr><td></td><td colspan="10">CometKiwi runs</td></tr><tr><td>Method</td><td>Figure</td><td>10</td><td>20</td><td>30</td><td>40</td><td>50</td><td>60</td><td>70</td><td>80</td><td>90</td><td>100</td></tr><tr><td>UniqRandom</td><td>2</td><td>0.7917</td><td>0.8022</td><td>0.8074</td><td>0.8104</td><td>0.8124</td><td>0.8140</td><td>0.8149</td><td>0.8160</td><td>0.8168</td><td>0.8175</td></tr><tr><td>LogprobAvg</td><td>2</td><td>0.7956</td><td>0.8055</td><td>0.8101</td><td>0.8129</td><td>0.8149</td><td>0.8162</td><td>0.8171</td><td>0.8181</td><td>0.8187</td><td>0.8193</td></tr><tr><td>LogprobSum</td><td>2</td><td>0.7519</td><td>0.7723</td><td>0.7834</td><td>0.7913</td><td>0.7974</td><td>0.8019</td><td>0.8051</td><td>0.8081</td><td>0.8109</td><td>0.8125</td></tr><tr><td>HillClimbing</td><td>2</td><td>0.7917</td><td>0.8080</td><td>0.8124</td><td>0.8148</td><td>0.8165</td><td>0.8176</td><td>0.8184</td><td>0.8191</td><td>0.8196</td><td>0.8200</td></tr><tr><td>ProxyFirst 200 Distilled-S</td><td>3</td><td>0.8081</td><td>0.8141</td><td>0.8167</td><td>0.8181</td><td>0.8190</td><td>0.8197</td><td>0.8202</td><td>0.8206</td><td>0.8208</td><td>0.8210</td></tr><tr><td>ProxyFirst 200 Distilled-M</td><td>3</td><td>0.8119</td><td>0.8165</td><td>0.8184</td><td>0.8194</td><td>0.8201</td><td>0.8206</td><td>0.8209</td><td>0.8211</td><td>0.8212</td><td>0.8213</td></tr><tr><td>ProxyFirst 50 Distilled-S</td><td>3</td><td>0.8054</td><td>0.8100</td><td>0.8114</td><td>0.8121</td><td>0.8124</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ProxyFirst 50 Distilled-M</td><td>3</td><td>0.8073</td><td>0.8107</td><td>0.8119</td><td>0.8122</td><td>0.8124</td><td></td><td>–</td><td>1</td><td>-</td><td></td></tr><tr><td>BayesOpt+GP</td><td>2,4,3</td><td>0.7917</td><td>0.8121</td><td>0.8167</td><td>0.8190</td><td>0.8201</td><td>0.8206</td><td>0.8210</td><td>0.8212</td><td>0.8213</td><td>0.8214</td></tr><tr><td>BayesOpt+GP+P with LogprobAvg</td><td>4</td><td>0.7956</td><td>0.8123</td><td>0.8166</td><td>0.8187</td><td>0.8198</td><td>0.8205</td><td>0.8208</td><td>0.8210</td><td>0.8213</td><td>0.8214</td></tr><tr><td>BayesOpt+GP+P with 200 Distilled-S</td><td>4,3</td><td>0.8081</td><td>0.8165</td><td>0.8190</td><td>0.8200</td><td>0.8207</td><td>0.8210</td><td>0.8212</td><td>0.8213</td><td>0.8214</td><td>0.8215</td></tr><tr><td>BayesOpt+GP+P with 200 Distilled-M</td><td>4,3</td><td>0.8119</td><td>0.8182</td><td>0.8199</td><td>0.8205</td><td>0.8209</td><td>0.8211</td><td>0.8213</td><td>0.8214</td><td>0.8215</td><td>0.8215</td></tr><tr><td>BayesOpt+GP+P with 50 Distilled-S</td><td>4,3</td><td>0.8054</td><td>0.8153</td><td>0.8184</td><td>0.8196</td><td>0.8204</td><td>0.8208</td><td>0.8210</td><td>0.8213</td><td>0.8214</td><td>0.8214</td></tr><tr><td>BayesOpt+GP+P with 50 Distilled-M</td><td>4,3</td><td>0.8073</td><td>0.8164</td><td>0.8187</td><td>0.8200</td><td>0.8207</td><td>0.8209</td><td>0.8211</td><td>0.8213</td><td>0.8214</td><td>0.8215</td></tr><tr><td></td><td></td><td colspan="10"></td></tr><tr><td>Method</td><td>Figure</td><td>110</td><td>120</td><td>130</td><td>140</td><td>CometKiwi runs 150</td><td>160</td><td>170</td><td>180</td><td>190</td><td>200</td></tr><tr><td>UniqRandom</td><td>2</td><td>0.8182</td><td>0.8188</td><td>0.8192</td><td>0.8197</td><td>0.8200</td><td>0.8205</td><td>0.8208</td><td>0.8211</td><td>0.8214</td><td>0.8216</td></tr><tr><td>LogprobAvg</td><td>2</td><td>0.8199</td><td>0.8203</td><td>0.8205</td><td>0.8209</td><td>0.8211</td><td>0.8212</td><td>0.8213</td><td>0.8214</td><td>0.8216</td><td>0.8216</td></tr><tr><td>LogprobSum</td><td>2</td><td>0.8139</td><td>0.8156</td><td>0.8170</td><td>0.8180</td><td>0.8188</td><td>0.8196</td><td>0.8204</td><td>0.8209</td><td>0.8212</td><td>0.8216</td></tr><tr><td>HillClimbing</td><td>2</td><td>0.8203</td><td>0.8206</td><td>0.8208</td><td>0.8209</td><td>0.8211</td><td>0.8213</td><td>0.8214</td><td>0.8215</td><td>0.8216</td><td>0.8216</td></tr><tr><td>BayesOpt+GP</td><td>2,4,3</td><td>0.8215</td><td>0.8215</td><td>0.8215</td><td>0.8216</td><td>0.8216</td><td>0.8216</td><td>0.8216</td><td>0.8216</td><td>0.8216</td><td>0.8216</td></tr><tr><td>BayesOpt+GP+P with LogprobAvg</td><td>4</td><td>0.8214</td><td>0.8215</td><td>0.8215</td><td>0.8216</td><td>0.8216</td><td>0.8216</td><td>0.8216</td><td>0.8216</td><td>0.8216</td><td>0.8216</td></tr></table>

Table 3: Exact values (selected candidate score) for Figures 2 to 4.

## A Statistical Significance

We measure statistical significance between two methods based on the final candidate CometKiwi scores with either budget 30, 60, 90, or across the budget range from 10 to 190 in Table 4. To determine whether one method is better than another one, we use one-sided paired Student’s t-test with p-value threshold 0.01 which is run across the individual samples.

<table><tr><td>Budget 30</td><td>Unidom IooAvg</td><td>opounmm</td><td></td><td>Hiiiing BayeeptGp</td><td></td><td>Budget 60</td><td>Uniom</td><td>LooAvg</td><td>Iopounm</td><td>Hiing</td><td>Basp+tGPp</td></tr><tr><td>LogprobAvg</td><td>←</td><td></td><td>←</td><td>↑</td><td>↑</td><td>LogprobAvg</td><td>←</td><td></td><td>←</td><td>↑</td><td>↑</td></tr><tr><td>LogprobSum</td><td>个</td><td>↑</td><td></td><td>↑</td><td>↑</td><td>LogprobSum</td><td>↑</td><td>↑</td><td></td><td>↑</td><td>↑</td></tr><tr><td>HillClimbing</td><td>←</td><td>←</td><td>←</td><td></td><td>↑</td><td>HillClimbing</td><td>←</td><td>←</td><td>←</td><td></td><td>↑</td></tr><tr><td>ProxyFirst 200 Distilled-S</td><td>←</td><td>←</td><td>←</td><td>←</td><td></td><td>ProxyFirst 200 Distilled-S</td><td>←</td><td>←</td><td>←</td><td>←</td><td>↑</td></tr><tr><td>ProxyFirst 200 Distilled-M</td><td>←</td><td>←</td><td>←</td><td>←</td><td>←</td><td>ProxyFirst 200 Distilled-M</td><td>←</td><td>←</td><td>←</td><td>←</td><td></td></tr><tr><td>ProxyFirst 50 Distilled-S</td><td>←</td><td>←</td><td>←</td><td>↑</td><td>↑</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ProxyFirst 50 Distilled-M</td><td>←</td><td>←</td><td>←</td><td></td><td>↑</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>BayesOpt+GP</td><td>←</td><td>←</td><td>←</td><td>←</td><td></td><td>BayesOpt+GP</td><td>←</td><td>←</td><td>个</td><td>←</td><td></td></tr><tr><td>BayesOpt+GP+P with LogprobAvg</td><td>←</td><td>←</td><td>←</td><td>←</td><td></td><td>BayesOpt+GP+P with LogprobAvg</td><td>←</td><td>←</td><td>←</td><td>←</td><td></td></tr><tr><td>BayesOpt+GP+P with 200 Distilled-S</td><td>←</td><td>←</td><td>←</td><td>←</td><td>←</td><td>BayesOpt+GP+P with 200 Distilled-S</td><td>←</td><td>←</td><td>←</td><td>个</td><td>←</td></tr><tr><td>BayesOpt+GP+P with 200 Distilled-M</td><td>←</td><td>←</td><td>←</td><td>←</td><td>←</td><td>BayesOpt+GP+P with 200 Distilled-M</td><td>←</td><td>←</td><td>←</td><td>←</td><td>←</td></tr><tr><td>BayesOpt+GP+P with 50 Distilled-S BayesOpt+GP+P with 50 Distilled-M</td><td>←</td><td>←</td><td>← ←</td><td>← ←</td><td>← ←</td><td>BayesOpt+GP+P with 50 Distilled-S BayesOpt+GP+P with 50 Distilled-M</td><td>← ←</td><td>← 个</td><td>← ←</td><td>← ←</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Budget 90</td><td>Unidom</td><td>LooAvg</td><td>Iopouunm</td><td>Hiliiing</td><td>Baye+tGPp</td><td>Across budgets 10 to 190</td><td></td><td>LooAvg</td><td>Iopouunm</td><td>Hiiiing</td><td>BaeetGp</td></tr><tr><td>LogprobAvg</td><td>←</td><td></td><td>←</td><td>↑</td><td>↑</td><td>LogprobAvg</td><td></td><td></td><td>←</td><td>↑</td><td>↑</td></tr><tr><td>LogprobSum</td><td>↑</td><td>↑</td><td></td><td>↑</td><td>↑</td><td>LogprobSum</td><td></td><td></td><td></td><td>↑</td><td>↑</td></tr><tr><td>HillClimbing</td><td>←</td><td>←</td><td>←</td><td></td><td>←</td><td>HillClimbing</td><td></td><td>←</td><td>←</td><td></td><td>←</td></tr><tr><td>ProxyFirst 200 Distilled-S</td><td>←</td><td>←</td><td>←</td><td>←</td><td>↑</td><td>ProxyFirst 200 Distilled-S</td><td></td><td>← ←</td><td>← ←</td><td>←</td><td>←</td></tr><tr><td>ProxyFirst 200 Distilled-M</td><td>←</td><td>←</td><td>←</td><td>←</td><td></td><td>ProxyFirst 200 Distilled-M</td><td></td><td></td><td></td><td>←</td><td>←</td></tr><tr><td>BayesOpt+GP+P</td><td>←</td><td>←</td><td>←</td><td>←</td><td></td><td>BayesOpt+GP</td><td></td><td>← ←</td><td>←</td><td>←</td><td></td></tr><tr><td>BayesOpt+GP+P with LogprobAvg</td><td>←</td><td>←</td><td>←</td><td>←</td><td></td><td>BayesOpt+GP+P with LogprobAvg</td><td></td><td>←</td><td>←</td><td>←</td><td></td></tr><tr><td>BayesOpt+GP+P with 200 Distilled-S</td><td>←</td><td>←</td><td>←</td><td>←</td><td></td><td>BayesOpt+GP+P with 200 Distilled-S</td><td></td><td></td><td>← ←</td><td>个 个</td><td>←</td></tr><tr><td>BayesOpt+GP+P with 200 Distilled-M</td><td>←</td><td>←</td><td>←</td><td>←</td><td>←</td><td>BayesOpt+GP+P with 200 Distilled-M BayesOpt+GP+P with 50 Distilled-S</td><td></td><td></td><td>←</td><td>←</td><td>← ←</td></tr><tr><td>BayesOpt+GP+P with 50 Distilled-S</td><td>←</td><td>←</td><td>←</td><td>←</td><td></td><td></td><td></td><td>←</td><td>←</td><td>←</td><td>←</td></tr><tr><td>BayesOpt+GP+P with 50 Distilled-M</td><td>←</td><td>←</td><td>←</td><td>←</td><td></td><td>BayesOpt+GP+P with 50 Distilled-M</td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 4: Statistical significance comparison between proposed methods across various CometKiwi calls budgets. Within a cell,  means that the column method (in header) is statistically significantly better than the row method and means the opposite. If a cell is empty, none of the methods are significantly better than the other one. For example, in Budget 30 (top left) table, in third row and first column,  means that HillClimbing is significantly better than UniqRandom in the setup of budget of 30.