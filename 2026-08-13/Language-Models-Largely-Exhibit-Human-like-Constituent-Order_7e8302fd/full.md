# Language Models Largely Exhibit Human-like Constituent Ordering Preferences

Ada Defne Tur†‡<sup>\*</sup>, Gaurav Kamath†‡<sup>\*</sup>, Siva Reddy† ‡ §

†McGill University, Canada, ‡Mila - Quebec AI Institute, Canada, §Canada CIFAR AI Chair <sup>\*</sup>Equal Contribution

Correspondences to ada.tur@mila.quebec

## Abstract

Though English sentences are typically inflexible vis-à-vis word order, constituents often show far more variability in ordering. One prominent theory presents the notion that constituent ordering is directly correlated with constituent weight: a measure of the constituent’s length or complexity. Such theories are interesting in the context of natural language processing (NLP), because while recent advances in NLP have led to significant gains in the performance of large language models (LLMs), much remains unclear about how these models process language, and how this compares to human language processing. In particular, the question remains whether LLMs display the same patterns with constituent movement, and may provide insights into existing theories on when and how the shift occurs in human language. We compare a variety of LLMs with diverse properties to evaluate broad LLM performance on four types of constituent movement: heavy NP shift, particle movement, dative alternation, and multiple PPs. Despite performing unexpectedly around particle movement, LLMs generally align with human preferences around constituent ordering.<sup>1</sup>

## 1 Introduction

Despite the fact that word order in English is typically strict, constituents in post-verbal positions can be highly flexible in their ordering (Chomsky, 2002; Wasow and Arnold, 2003). A number of specific phenomena are prime examples of this movement; we show these in Figure 1.

The cause of this movement has been attributed to a variety of factors, including lexical bias (Baars et al., 1975; Hartsuiker et al., 2005; Dell and Reich, 1981), semantic connectedness (Kayne, 1983;

Behaghel, 1932), and information structure (Lambrecht, 1994; Chafe, 1976; Behaghel, 1932; Gundel, 1988). However, one of the most prominent factors, as prior research has suggested, is constituent weight and complexity (Quirk et al., 1975; Behaghel, 1909; Wasow, 1997a).

As the following examples show, we tend to prefer sentences where longer and more complex constituents are moved to the end of the sentence:

(5) a. I met [the man]<sub>NP</sub> [at the park]<sub>PP</sub>.

b. \*I met [at the park]<sub>PP</sub> [the man]<sub>NP</sub>.

c. I met [at the park]<sub>PP</sub> [the tall man selling water to marathon runners]<sub>NP</sub>.

d. ?I met [the tall man selling water to marathon runners]<sub>NP</sub> [at the park]<sub>PP</sub>.

The typical constituent order shown in (5-a), for example, is not readily perturbed to the constituent order shown in (5-b). In sentences (5-c) and (5-d), however, where the NP is considerably longer and more complex than the PP, the reverse is true.<sup>2</sup>

Linguistic theory thus suggests that phrases and constituents are specifically ordered to be presented in increasing complexity, or weight; essentially, the larger the constituent, the further to the end of the sentence we expect it to appear (Quirk et al., 1975; Behaghel, 1909; Wasow, 1997a; Futrell et al., 2015). Consider the example in Figure 3:

Other orderings of this sentence, if greatly violating this principle of weight, would likely be considered undesirable. This relationship between the complexity of post-verbal constituents and their ordering raises several questions:

• What are the exact effects of weight on con-

## Heavy NP Shift (HNPS)

(1) a. I met [the tall man selling water to marathon runners]<sub>NP</sub> [at the park]<sub>PP</sub>.

b. I met [at the park]<sub>PP</sub> [the tall man selling water to marathon runners]<sub>NP</sub>.

## Particle Movement (PM)

(2) a. She looked [up]<sub>particle</sub> [her question]<sub>NP</sub> on her computer.

b. She looked [her question]<sub>NP</sub> [up]<sub>particle</sub> on her computer.

## Dative Alternation (DA)

(3) a. He sent [her]<sub>IndirectObj</sub> [a gift]<sub>DirectObj</sub> for her birthday.

b. He sent [a gift]<sub>DirectObj</sub> [to her]<sub>IndirectObj</sub> for her birthday.

## Multiple PP Shift (MPP)

(4) a. I went [to the mall]<sub>PP</sub> [with my sister]<sub>PP</sub> on Sunday.

b. I went [with my sister]<sub>PP</sub> [to the mall]<sub>PP</sub> on Sunday.

Figure 1: Examples of constituent movement types: Heavy NP Shift (HNPS), Particle Movement (PM), Dative Alternation (DA) and Multiple PP Shift (MPP).

stituent ordering, in terms of gradient and ceiling effects, of increasing complexities on sentence acceptability?

• Which measures of ‘weight/complexity’ best explain the effects on constituent ordering?

• How exactly do LLMpreferences around constituent shifting align with human constituent shifting preferences?

Psycholinguistic research has provided some insight into these questions for human language processing, but the same cannot yet be said about increasingly powerful non-human language processors (Medeiros et al., 2021; Wasow, 2002). Prior research, however, supports the abilities of modern language models in assigning relative linguistic plausibility scores aligning similar to human preferences (Linzen et al., 2016; Marvin and

Linzen, 2018). Additionally, we hypothesize that the human-feedback mechanism incorporated in instruction-tuned models will present even more similar judgements.

In this work, we study the behavior of LLMs with constituent movement in English. We model Heavy NP Shift (HNPS), Dative Alternation (DA), Particle Movement (PM), Multiple PP Shift (MPP)—see Figure 1—as a function of its weight. Weight corresponds to a number of selected measures: word length, syllable weight, token length, and modifier weight. We analyze these measures to determine which best explains constituent ordering effects. Figure 2 outlines our contributions, which are as follows:

• We evaluate the preferences of models in regards to constituent movement, using a novel constituent shift dataset containing both synthetic and naturally occurring data.

• We study the motivating factors for constituent ordering preferences in models, over a variety of candidates.

• We compare the behaviors of models with human judgements, analyzing correlative trends between the two.

## 2 Background

At a high level, English follows a subject-verbobject (SVO) ordering; beyond this basic structure, other objects, modifiers, constituents, and clauses can be added to form more complex sentences (Hengeveld, 1992). The organization and format of how and when each constituent in a sentence is delivered, or its ordering, can be highly flexible (Bakker, 1998; Namboodiripad, 2019, 2017).

Constituent shifting is the process of reordering the constituents of a sentence, such that the original meaning of the sentence is maintained, and all semantic truth conditions are unchanged. This work focuses on four specific types of shift, with a prime commonality: each shift involves the movement of constituents from a post-verbal position (i.e. appearing after the verb of a sentence) to another post-verbal position (Wasow and Arnold, 2003; Wasow, 2002).<sup>3</sup> Table 1 demonstrates how we define shifted/unshifted sentences.

![](images/6842004519603dddcb615e12bc13f727b28507b528fbf9e02e6a19a7d452ff70.jpg)

Figure 2: We categorize our work into three main experiments. Our first experiment evaluates model response to constituent movement, our second experiment analyzes what motivates LLM constituent ordering preferences, and our third experiment compares model preferences with human judgements.
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Unshifted Form</td><td rowspan=1 colspan=1>Shifted Form</td></tr><tr><td rowspan=1 colspan=1>HNPS</td><td rowspan=1 colspan=1>S + V + NP + PPI met [the tall man selling water]nP [at the park]pp.</td><td rowspan=1 colspan=1>S + V + PP + NPI met [at the park]pp [the tall man selling water]NP.</td></tr><tr><td rowspan=1 colspan=1>PM</td><td rowspan=1 colspan=1>S + V + NP + PRTShe looked [up]pRT [her question]Np on her computer.</td><td rowspan=1 colspan=1>S + V + PRT + NPShe looked [her question]NP [up]pRT on her computer.</td></tr><tr><td rowspan=1 colspan=1>DA</td><td rowspan=1 colspan=1>S + V + NP1 + NP2He sent [her]NP [a gift]Np for her birthday.</td><td rowspan=1 colspan=1>S + V + NP1 + PPHe sent [a gift]NP [to her]pp for her birthday.</td></tr><tr><td rowspan=1 colspan=1>MPP</td><td rowspan=1 colspan=1>S + V + PP1 + PP2I went [to the mall]pp [with my sister]Pp on Sunday.</td><td rowspan=1 colspan=1>S + V + PP2 + PP1I went [with my sister]pP [to the mall]Pp on Sunday.</td></tr></table>

Table 1: Unshifted versus shifted forms for each shift. Note that the unshifted and shifted form for MPP is ambiguous; the unshifted and shifted forms cannot be derived separately given an example.

"In my laboratory we use it as an   
easily studied instance of mental   
grammar, allowing us to document   
[in great detail] [the psychology of   
linguistic rules] [from infancy to old   
age] [in both normal and neurologically   
impaired people] [in much the same way   
that biologists focus on the fruit fly   
Drosophila to study the machinery of the   
genes]."  
Figure 3: Example from (Pinker, 2007).

In our experiments, we consider the following measures of weight: word length, which corresponds to the number of words in a constituent; syllable weight, which corresponds to the number of syllables in a constituent; token length, which refers to the number of tokens in a constituent, determined by respective model tokenizers; modifier weight, which refers to the number of adjective phrase (AdjP) and prepositional phrase (PP) modifying the constituent itself (plus 1 to include the weight of the base constituent itself as normalization).

Absolute weight itself, however, is not the most effective metric for observing motivations for constituent movement; instead, there is reason to believe that it is the relative weight of constituents that determines their ordering (Wasow, 1997b). For instance, for phrases "with her grandmother" and "around the garden", the ratio of word lengths would be 1 (3:3); the ratio of syllables would be 1 (5:5), etc. For phrases "with her grandmother" and "around the decorated entryway garden with the large fountain", the ratio of word length would be 17 5 (5:17), etc. As the ratio of any metric increases beyond 1, we know that the weight of the first constituent is larger than the second, and thus, we predict that the motivation to shift will be greater.

## 3 Related Work

Constituent movement has been the subject of considerable linguistic literature; we categorize relevant contributions by our three research questions.

## 3.1 What are the exact effects of weight on constituent ordering?

Significant prior work has focused on investigating the effects of weight on constituent ordering (Arnold et al., 2000; Wasow and Arnold, 2003; Wasow, 1997a; Arnold et al., 2004; Hawkins, 1995; Behaghel, 1909; Hawkins, 2004); many contributions find gradient effects by which the shift becomes more frequent in examined language corpora as the relative weight of the relative constituent changes, suggesting that weight is the predominant factor in triggering the shift, even cross-linguistically (Wasow, 1997b; Wasow and Arnold, 2003; Faghiri and Thuilier, 2018; Wang and Liu, 2014; Hawkins, 1999; Quirk et al., 1975; Manetta, 2012). Furthermore, studies with human participants find similar results, where weight presents a primary role in the shift (Medeiros et al., 2021). The study, however, also suggests that this movement is constrained by ceiling effects, by which the efficacy of additional weight and complexity plateaus. More closely relevant to this work, Futrell and Levy (2018) conduct a similar analysis on post-verbal constituent movement using weight as a binary feature (i.e. ‘long’ vs ‘short’), and find similar trends with LSTMs.

## 3.2 What measure of weight best explains effects of constituent ordering?

Weight is a measure of a constituent’s complexity or size, but how best to measure it is less straightforward (Chomsky, 2008; Haegeman, 1991; Wasow, 1997a). Related research categorizes and analyzes the effects of three primary measures of weight on constituent movement, particularly with HNPS: the word length of the NP, the number of nodes in the NP’s syntactic structure, and the number of modifiers applied to the NP (Wasow and Arnold, 2003; Medeiros et al., 2021; Wasow, 1997b). The analyses found that, although the word length was statistically the strongest predictor for HNPS, “no single factor can account for observed constituent order alternation” (Medeiros et al., 2021, pg.6). Similarly, Wasow and Arnold (2003) find in a corpus study that for HNPS and DA, constituent movement was best accounted for when considering both word length and modifier weight together, as opposed to either on its own.

## 3.3 How exactly do LLM preferences around constituent shifting align with human constituent shifting preferences?

Research concerning the behaviors of computational models has also shown that models exhibit human-like preferences (Fujihara et al., 2022; Linzen et al., 2016; Marvin and Linzen, 2018; Kamath et al., 2024). Notably, prior work shows models learn syntactic alternations (Wilcox et al., 2019; Lau et al., 2017); more directly relevant to us, Futrell and Levy (2018) finds that behaviors of LSTMs appear to correlate closely with observed judgements of humans on corresponding data, suggesting that constituent movement is motivated similarly in both humans and models. This work, however, predates preference-aligned models (Ouyang et al., 2022); we hypothesize that the behaviors of such models will align even more closely with human preferences around constituent ordering.

## 4 Models and Data

## 4.1 Data

We both generate synthetic data using a template modified for the various shifts and structures and mine natural data from the Penn Treebank-2 corpus (Marcus et al., 1995). Each shift we consider has a standardized form that we can utilize for both processes, noted in Table 1. We also annotate for the aforementioned weight measures; syllable weight is computed using Syllapy;<sup>4</sup> token weight is retrieved simultaneously with the model scoring process; modifier weight is counted when constructing modifier chunks on constituents.

## 4.1.1 Synthetic Data

We synthetically generate data using the process shown in Figure 4 in order to accumulate large amounts of iteratively more complex data for model evaluations. Using an overall frame for the sentence, we alternate subjects, verbs, moving constituents, and their modifiers; we do this for all variables on a variety of constituents.

## 4.1.2 Mined Data

Our synthetic data, however, contains limited syntactic variation, and may not represent naturally occurring data; thus, we mine from existing data to ground our results. We use the Penn Treebank-2 to retrieve sentences following the structure of each shift (Marcus et al., 1995). We collect over 1,000 examples for HNPS and DA, and approximately 400 for PM and MPP, selecting a random sample of 500 and 400 sentences, respectively. We manually inspect and exclude low-quality datapoints.<sup>5</sup> As weight measures, we include constituent word length, syllable weight, and token length, but exclude modifier weight due to the complexity of accurately extracting this from such data.

![](images/38c3faa0e90e04cadabe537e19fc87f3e1b3fa89ce1023d398271cdd7e947058.jpg)  
Figure 4: The outline for creating synthetic data, using varying modifier weights.

<table><tr><td>Synthetic Data</td><td>Mined Data</td></tr><tr><td>HNPS 3,888</td><td>314</td></tr><tr><td>PM 4,136</td><td>131</td></tr><tr><td>DA 210,304</td><td>123</td></tr><tr><td>MPP 180,224</td><td>130</td></tr><tr><td>Total Size 398,552</td><td>698</td></tr></table>

Table 2: Dataset sizes by sentence count. Synthetic datasets for DA and MPP are larger due to modification of both constituents, rather than one, leading to more constituent weight ratios.

## 4.2 Models

We select a range of open autoregressive models with diverse attributes: the entire GPT-2 model family (Radford et al., 2019) was analyzed to study behaviors over scaled model sizes; Llama-3 8B, Llama-3 8B Instruct (AI@Meta, 2024), Mistral v0.3 7B, Mistral v0.3 7B Instruct (Jiang et al., 2023), OLMo 7B, and OLMo 7B Instruct (Groeneveld et al., 2024) were used to compare standard and instruction-tuned models; BabyFlamingo and BabyOPT (Warstadt et al., 2023) were used to study LLMs trained on BabyLM data, a childdirected dataset to simulate language stimulus during the early human language acquisition period. We do not inspect closed-source models such as

GPT-4 (OpenAI, 2024), due to inaccessibility of their underlying logits.

## 5 Shifting Preference of Models

## 5.1 Approach

To observe the effects of constituent weight on ordering preferences, we compute the difference in log probabilities assigned by models to shifted and unshifted sentences at a range of constituent weight ratios (see Section 2). We begin by extracting log probabilities assigned to each sentence by the aforementioned models, using the minicons library (Misra, 2022). This score corresponds to the model’s judgement of the sentence’s linguistic plausibility, computed with the following equation:

$$
\begin{array} { r } { M _ { s c o r e } ( \mathbf { w } ) = \sum _ { t = 1 } ^ { T } \log P _ { \mathrm { M } } ( w _ { t } | w _ { 1 } , w _ { 2 } , \dots , w _ { t - 1 } ; \pmb \theta ) . } \end{array}
$$

$\mathbf { M } _ { s c o r e }$ is the log probability score of the sequence $\textbf { w } = \mathbf { \Psi } [ w _ { 1 } , w _ { 2 } , \ldots , w _ { t } , \ldots , w _ { T } ] ^ { T }$ , where $w _ { t }$ is the token at position t. The term $P _ { \mathbf { M } }$ is the conditional probability from the model M of token w<sub>t</sub> given the preceding tokens, while θ are the model parameters. The output of this formula is the sum of the probabilities of all tokens in the sequence, given previous tokens and model parameters, which equates to the overall sequence probability. The closer $\mathbf { M } _ { s c o r e }$ is to 0, the more strongly the model judges the sequence to likely occur in human language.

Upon computing this score for each minimal pair of shifted-unshifted sentences (denoted as U and S), we calculate the difference of the $\mathbf { M } _ { s c o r e }$ for each sentence:

$$
\mathbf { M } _ { p r e f e r e n c e } = \mathbf { M } _ { s c o r e } ( U ) - \mathbf { M } _ { s c o r e } ( S ) .
$$

This metric aligns closely with the surprisalbased measure used by Futrell and Levy (2018), and is in line with other surprisal-based metrics commonly used in work in psycholinguistics and computational linguistics (Linzen et al., 2016; Futrell et al., 2018; Wilcox et al., 2018; Schuster and Linzen, 2022; Baroni, 2022).

![](images/3ac7ee8cc8921650b2b59e2806461151075fd8821325969c9170fabe215dffd9.jpg)  
Figure 5: OLMo 7B $\mathbf { M } _ { p r e f e r e n c e }$ scores with respect to different measures of weight.

Intuitively, the value of $\mathbf { M } _ { p r e f e r e n c e }$ captures the model’s relative preference for the unshifted version of the sentence. If this value is >0, the model has a stronger preference for the unshifted sentence, and if it is <0, the model has a stronger preference for the shifted version of the sentence; values approaching 0 suggest no clear preference between the two.

Though the usage of instruction-tuned models attracts the idea of directly prompting such LMs for their linguistic preferences, Hu and Levy (2023) and Kamath et al. (2024) show results that suggests meta-linguistic prompting underestimates linguistic capacities; thus, we only consider raw probability scores in our experimentation.

## 5.2 Results: Are models motivated by weight to shift?

We analyze model $\mathbf { M } _ { p r e f e r e n c e }$ scores by weight ratios (see Section 2) considering various metric types in the analysis. Figure 5 shows OLMo 7B performance on HNPS; see Section A.2.1 for remaining models and variables. We observe similar trends overall across models.

For HNPS, we find that model $\mathbf { M } _ { p r e f e r e n c e }$ scores initially start positive, i.e., models prefer unshifted sentence, but converge above 0 as constituent weight ratios increase, indicate some continued preference for unshifted constituent orderings beyond a given point. In the case of DA and MPP, we see a similar effect of weight $\mathbf { M } _ { p r e f e r e n c e }$ scores, but with a plateau below zero, indicating that models eventually converge upon a relative preference for the shifted version of a sentence. In the case of PM, however, we see scores initially drop sharply from around 0, but later somewhat rise.

## 6 Motivating Factors of Model Preference

Literature regarding human language finds varying levels of importance correlated with different measures of constituent weight, particularly in corpus studies and human judgement tasks (Wasow, 1997a; Wasow and Arnold, 2003; Medeiros et al., 2021). Whether or not this same trend is seen with language models is unknown. We conduct a regression analysis on the data collected for the first experiment, using a generalized additive mixed model (Wood, 2017; Sóskuthy, 2017), with the goal of measuring how significantly each weight measure both impacts the models’ judgements and serves to fit regression lines on the data.

## 6.1 Approach

To compare how well different measures of weight explain shifting preferences, we fit Generalized Additive Mixed Models, or GAMMs (Wood, 2017; Sóskuthy, 2017), on our $\mathbf { M } _ { p r e f e r e n c e }$ scores (see Section 5.1) as a function of various weight measures: word length, token length, syllable weight, and modifier weight. GAMMs allow for the fitting of highly non-linear relationships as the sum of multiple predictor-wise smooth functions: basis functions that allow for an arbitrary degree of smoothness. Crucially, aside from providing interpretable measures of goodness of fit, GAMMs also allow for grouping structures in the data to be captured as random effects (Wood, 2017, ch.6).

Bearing in mind that multiple measures of weight may jointly determine the accessibility of a shift (Wasow, 1997b; Wasow and Arnold, 2003), we analyze the relative importance of each weight measure in the following manner. For each model, first, we fit a GAMM on the model’s $\mathbf { M } _ { p r e f e r e n c e }$ scores as a function of all weight predictors, with verb-wise random intercepts and slopes. We then iteratively ablate each weight predictor while retaining all others, and compare the quality of fit yielded by the full model with that of the ablated model. Intuitively, this provides an indication of how important the dropped predictor is for the LLM: it captures how much less of the LLM’s behaviour is explained when information about that given measure of weight is ignored.

<table><tr><td>Model</td><td>Var</td><td>Full</td><td>Tokenab</td><td>Wordab</td><td>Syllab</td><td>Modsab</td></tr><tr><td>GPT-2 Medium Llama-3</td><td>HNPS</td><td>.654</td><td>.629</td><td>.616</td><td>.540</td><td>.635</td></tr><tr><td>Llama-3 Instruct</td><td></td><td>.542 .452</td><td>.524</td><td>.519</td><td>.485</td><td>.538</td></tr><tr><td>BabyLlama</td><td></td><td></td><td>.438</td><td>.408</td><td>.359</td><td>.451</td></tr><tr><td>GPT-2 Medium</td><td></td><td>.527</td><td>.514</td><td>.466</td><td>.415</td><td>.509</td></tr><tr><td>Llama-3</td><td>PM</td><td>.605</td><td>.602</td><td>.581</td><td>.534</td><td>.580</td></tr><tr><td>Llama-3 Instruct</td><td></td><td>.608</td><td>.603</td><td>.586</td><td>.564</td><td>.590</td></tr><tr><td>BabyLlama</td><td></td><td>.627</td><td>.619</td><td>.599</td><td>.555</td><td>.602</td></tr><tr><td>GPT-2 Medium</td><td></td><td>.719</td><td>.702</td><td>.663</td><td>.651</td><td>.701</td></tr><tr><td>Llama-3</td><td>DA</td><td>.571</td><td>.561</td><td>.552</td><td>.568</td><td>.568</td></tr><tr><td>Llama-3 Instruct</td><td></td><td>.554</td><td>.544</td><td>.532</td><td>.538</td><td>.549</td></tr><tr><td>BabyLlama</td><td></td><td>.503</td><td>.493</td><td>.490</td><td>.486</td><td>.496</td></tr><tr><td>GPT-2 Medium</td><td></td><td>.630</td><td>.616</td><td>.603</td><td>.622</td><td>.623</td></tr><tr><td>Llama-3</td><td></td><td>.368</td><td>.309</td><td>.302</td><td>.300</td><td>.356</td></tr><tr><td></td><td>MPP</td><td>.358</td><td>.316</td><td>.321</td><td>.271</td><td>.351</td></tr><tr><td>Llama-3 Instruct</td><td></td><td>.320</td><td>.284</td><td>.291</td><td>.208</td><td>.313</td></tr><tr><td>BabyLlama</td><td></td><td>.310</td><td>.297</td><td>.298</td><td>.281</td><td>.306</td></tr></table>

Table 3: R-squared scores for a subset of models; see full table in Appendix A.1. Full denotes Rsquared values from GAMMs with all weight measure predictors included, while $[ M e t r i c ] _ { a b }$ denotes the Rsquared score when excluding the metric; a larger difference compared to the original R-squared score denotes higher significance in contributing to the overall R-squared. Numbers bolded denote high R-sq.; numbers underlined denote low R-sq.

## 6.2 Results: Which measure of weight best explains shifting?

Table 3 presents a subset of the results of our regression analysis (full results in Appendix Table 5). Crucially, we find that syllable weight is often the most important predictor of LLM behavior around constituent ordering preference, since the drop in R-squared scores is highest when syllable weight is not used. For DA, word length seems to be the best predictor.

We find that GPT-2 Medium achieves the highest R-squared overall on both HNPS and MPP. Further, contrary to our hypotheses, across almost all shift types, instruction-tuned models consistently achieve lower R-squared scores than their base model counterparts. Table 3, for example, shows these results for Llama-3 and Llama-3 Instruct; see Appendix A.2.1 for remaining plots on synthetic data. The BabyLM models also present high R-squared scores on both PM and DA, with BabyLlama yielding the highest R-squared values (see Table 3 for BabyLlama results and Appendix A.2.1 for BabyOPT results). Finally, despite the high performance of GPT-2 Medium, we do not observe consistent improvements in Rsquared values as model sizes scale.

![](images/a67b6a884b3106d018cd85b0131be85a47dfc75b0fa49bc901c0d2a15a6bbdef.jpg)  
Figure 6: Scatterplot of human judgment scores and OLMo 7B $\mathbf { M } _ { p r e f e r e n c e }$ scores on Heavy NP Shift data.

## 7 Human-Model Preference Correlation

## 7.1 Approach

To adequately compare the behaviors of LLMs with those of humans, a direct study of preferences on identical data points is necessary. We collect human judgements on a subset of data presented to models; though human judgements and model scores are not identical metrics, they can act as proxies when comparing relative trends.

We conduct a crowdsourced study through Prolific, collecting judgments from 126 native English speakers residing in Anglophone countries, on 500 sentence pairs. Each participant is presented 25 sentence pairs and asked to judge how natural they sound in relation to each other, assigning a score between 1 and 7; 1 corresponds to the first sentence presented appearing far more natural than the second, and 7 the reverse. We exclude datapoints with low inter-annotator agreement to minimize noise.<sup>6</sup>

## 7.2 Results: Do LLM and human preferences correlate?

To analyze how this human judgment data compares with model results, we compute the Spearman correlation between the average human score for each data point and the model’s $\mathbf { M } _ { p r e f e r e n c e }$ score. We present these scores in Table 4. We also plot model scores against human judgment data to observe the correlation visually; these plots are presented in Figure 6 and Section A.3.

Comparing preferences of humans and models on a statistical level introduces interesting findings. Notably, the GPT-2 and OLMo 7B classes of models appear to align most closely with human behaviors, achieving the highest correlation scores across all shift types.

<table><tr><td></td><td>GPT-2</td><td>GPT-2 Med</td><td>GPT-2 Large</td><td>GPT-2 XL</td><td>BabyOPT</td><td>BabyLlama</td><td>Llama-3</td><td>Llama-3 I</td><td>Mistral v0.3</td><td>Mistral v0.3 I</td><td>OLMo</td><td>OLMo I</td></tr><tr><td>HNPS</td><td>.410</td><td>.440</td><td>.342</td><td>.390</td><td>.261</td><td>.420</td><td>.428</td><td>.386</td><td>.414</td><td>.344</td><td>.509</td><td>.431</td></tr><tr><td>PM</td><td>.212</td><td>.125</td><td>.213</td><td>.295</td><td>.196</td><td>.261</td><td>.305</td><td>.293</td><td>.315</td><td>.347</td><td>.430</td><td>.431</td></tr><tr><td>DA</td><td>.651</td><td>.565</td><td>.505</td><td>.541</td><td>.256</td><td>.391</td><td>.600</td><td>.511</td><td>.524</td><td>.449</td><td>.494</td><td>.478</td></tr><tr><td>MPP</td><td>.371</td><td>.395</td><td>.361</td><td>.579</td><td>.233</td><td>.357</td><td>.402</td><td>.222</td><td>.487</td><td>.412</td><td>.513</td><td>.263</td></tr></table>

Table 4: Spearman correlation values (absolute) between LLMs and humans on sentence judgement tasks.

Remarkably, most of the more highly correlated models are the base versions of their respective model category—the instruction-tuned models quite often performed in manners less correlated with human preferences than their basemodel counterparts. This contrast is most stark in the case of MPP. Interestingly, no clear trend in alignment can be seen in the GPT-2 family as model sizes scale on any of the judgement tasks. We present remaining correlation figures in Section A.3. Crucially, correlations between model scores and human judgments are particularly low on the PM data.

## 8 Discussion

## 8.1 What are the exact effects of weight on constituent ordering?

In Figure 5 and Appendix A.2.1, we observe converging effects as weight increases, suggesting that prior theories defining weight as a prime factor in motivating the shift hold with computational models as well. Further, we observe, specifically with PM, that weight, beyond a certain threshold, begins to detriment motivation for shifting.

We observe similar effects on the scraped corpus data, in Appendix A.2.2, though with more noise. Given the specificity of the data itself, being rooted in financial reports and news articles, some noise and outliers were expected, and in some cases, observable trends remain. Similar to the synthetic data plots, we see convergence on HNPS, and some on MPP, as well as an initial drop followed by a rise in $\mathbf { M } _ { p r e f e r e n c e }$ scores for PM.

## 8.2 What measure of weight best explains effects of constituent ordering?

The syllable weight was the best measure of weight for explaining motivations to shift in LLMs. This raises an obvious question–why is syllable weight a more effective predictor of model behavior than token weight, which would intuitively be most aligned with a model’s processing of weight and complexity? This finding acts as initial evidence that models may induce linguistic information not just at the token level, but also implicitly at the level of syllables.<sup>7</sup>

## 8.3 How exactly do LLM preferences around constituent shifting align with human constituent shifting preferences?

Broadly speaking, a clear trend is maintained across humans and models, following what was presented by Futrell and Levy (2018) in their analysis. Where human language sees motivation for movement with increasing weight, model behavior follows closely. Our experiment, which includes graded data beyond binary weight categorizations, and a wider range of models, yields relatively high correlation effects between preferences of models and humans, as presented in Table 4—suggesting noticeably similar behaviors between the two on particular linguistic tendencies, with the notable exception of particle movement.

Interestingly, we observe an unexpected trend where instruction-tuned models, which consistently correlate less with human data than their corresponding base model, as well as, quite often, yield lower R-squared scores. This runs against our initial hypothesis around instructiontuned models, and suggests inadequacy in providing consistent and explainable trends compared to base models.

## 8.4 Future Work

Our findings suggest that even though newer models are equipped with more parameters, training data, and the human-feedback mechanism, they fail to align better with human linguistic preferences than their earlier counterparts, raising questions for future study. Equally, it invites further research into how models generate such sentences in standard conversational usage. We also wish to study further the motivations of LLMs in constituent movement, primarily regarding analysis of theories suggesting disparities between ordering preference motivations of listeners and speakers (Wasow, 2002).

## 9 Conclusions

In this work, we present a thorough analysis of the behaviors of LLMs in response to constituent movement, using both a novel set of nearly 400K minimal pairs of variably ordered sentences, as well as naturally occurring data. We collect human judgements and model preference scores and observe comparable trends between the behaviors of humans and models. Such comparisons indicate that humans and LLMs largely hold similar linguistic preferences around constituent ordering, with the exception of particle movement. Our findings—and in particular, the surprising gap we find between instruction-tuned models and their vanilla counterparts—invite further research into when and how linguistic preferences of models and humans align.

## 10 Limitations

We only focus on constituent movement in English, even though this phenomena is known to manifest cross-linguistically (Faghiri and Thuilier, 2018; Wang and Liu, 2014; Hawkins, 1999; Quirk et al., 1975; Manetta, 2012; Fujihara et al., 2022).

## 11 Ethics Statement

Our experimentation poses no risks or harms for any participants involved. Human participants from our data study through Prolific were compensated on average US\$12 per hour. Anonymous participants were informed of the purpose of the study and how their responses would be used, as well as their rights regarding submitted data.

## 12 Acknowledgements

This work was partly funded by IVADO R3AI NLP Regroupement and a Doctoral Training Award from the Fonds de Recherche du Québec—Société et Culture. We also thank Morgan Sonderegger for his invaluable feedback and guidance.

## References

AI@Meta. 2024. Llama 3 model card.

Jennifer E. Arnold, Anthony Losongco, Thomas Wasow, and Ryan Ginstrom. 2000. Heaviness vs. newness: The effects of structural complexity and discourse status on constituent ordering. Language, 76(1):28–55.

Jennifer E. Arnold, Thomas Wasow, Ash Asudeh, and Peter Alrenga. 2004. Avoiding attachment ambiguities: The role of constituent ordering. Journal of Memory and Language, 51(1):55–70.

Bernard J Baars, Michael T Motley, and Donald G MacKay. 1975. Output editing for lexical status in artificially elicited slips of the tongue. Journal of verbal learning and verbal behavior, 14(4):382– 391.

Dik Bakker. 1998. Flexibility and consistency in word order patterns in the languages of europe. Empirical Approaches to Language Typology, 20:383–420.

Marco Baroni. 2022. On the proper role of linguistically-oriented deep net analysis in linguistic theorizing. Preprint, arXiv:2106.08694.

Otto Behaghel. 1909. Beziehungen zwischen umfang und reihenfolge von satzgliedern. Indogermanische Forschungen (1909), 25(1909):110–142.

Otto Behaghel. 1932. Deutsche Syntax: Eine geschichtliche Darstellung. Band IV: Wortstellung, Periodenbau. Carl Winters Universitätsbuchhandlung, Heidelberg.

Wallace L. Chafe. 1976. Givenness, contrastiveness, definiteness, subjects, topics, and point of view. In Charles N. Li, editor, Subject and topic: A new typology of language, pages 25–55. Academic Press, New York, NY.

Noam Chomsky. 2002. Syntactic Structures. A Mouton classic. Mouton de Gruyter.

Noam Chomsky. 2008. The logical structure of linguistic theory. Language, 84:795–814.

Gary S Dell and Peter A Reich. 1981. Stages in sentence production: An analysis of speech error data. Journal of verbal learning and verbal behavior, 20(6):611–629.

Pegah Faghiri and Juliette Thuilier. 2018. Relative weight and givenness in constituent ordering of typologically different languages: Evidence from French and Persian. The 31st Annual CUNY Conference on Human Sentence Processing. Poster.

Riki Fujihara, Tatsuki Kuribayashi, Kaori Abe, Ryoko Tokuhisa, and Kentaro Inui. 2022. Topicalization in language models: A case study on japanese. In International Conference on Computational Linguistics.

Richard Futrell and Roger P. Levy. 2018. Do rnns learn human-like abstract word order preferences? Preprint, arXiv:1811.01866.

Richard Futrell, Kyle Mahowald, and Edward Gibson. 2015. Large-scale evidence of dependency length minimization in 37 languages. Proceedings of the National Academy of Sciences, 112(33):10336– 10341.

Richard Futrell, Ethan Wilcox, Takashi Morita, and Roger Levy. 2018. Rnns as psycholinguistic subjects: Syntactic state and grammatical dependency. Preprint, arXiv:1809.01329.

Dirk Groeneveld, Iz Beltagy, Pete Walsh, Akshita Bhagia, Rodney Kinney, Oyvind Tafjord, Ananya Harsh Jha, Hamish Ivison, Ian Magnusson, Yizhong Wang, Shane Arora, David Atkinson, Russell Authur, Khyathi Raghavi Chandu, Arman Cohan, Jennifer Dumas, Yanai Elazar, Yuling Gu, Jack Hessel, Tushar Khot, William Merrill, Jacob Morrison, Niklas Muennighoff, Aakanksha Naik, Crystal Nam, Matthew E. Peters, Valentina Pyatkin, Abhilasha Ravichander, Dustin Schwenk, Saurabh Shah, Will Smith, Emma Strubell, Nishant Subramani, Mitchell Wortsman, Pradeep Dasigi, Nathan Lambert, Kyle Richardson, Luke Zettlemoyer, Jesse Dodge, Kyle Lo, Luca Soldaini, Noah A. Smith, and Hannaneh Hajishirzi. 2024. Olmo: Accelerating the science of language models. Preprint, arXiv:2402.00838.

Jeannette K. Gundel. 1988. Universals of topiccomment structure. In Michael Hammond, Edith A. Moravcsik, and Jessica R. Wirth, editors, Studies in Syntactic Typology, pages 209–239. Benjamins, Amsterdam.

Liliane Haegeman. 1991. Introduction to Government and Binding Theory. Blackwell, Oxford.

Robert J Hartsuiker, Martin Corley, and Heike Martensen. 2005. The lexical bias effect is modulated by context, but the standard monitoring account doesn’t fly: Related beply to baars et al.(1975). Journal of Memory and Language, 52(1):58–70.

John A. Hawkins. 1995. A Performance Theory ofOrder and Constituency. Cambridge Studies in Linguistics. Cambridge University Press.

John A. Hawkins. 1999. The relative order of prepositional phrases in english: Going beyond manner–place–time. Language Variation and Change, 11(3):231–266.

John A. Hawkins. 2004. Efficiency and Complexity in Grammars. Oxford University Press.

Kees Hengeveld. 1992. Non-verbal predication: Theory, typology, diachrony.

Jennifer Hu and Roger Levy. 2023. Prompting is not a substitute for probability measurements in large language models. Preprint, arXiv:2305.13264.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023. Mistral 7b. Preprint, arXiv:2310.06825.

Gaurav Kamath, Sebastian Schuster, Sowmya Vajjala, and Siva Reddy. 2024. Scope Ambiguities in Large Language Models. Transactions of the Association for Computational Linguistics, 12:738–754.

Richard S. Kayne. 1983. Connectedness. Linguistic Inquiry, 14(2):223–249.

Knud Lambrecht. 1994. Information Structure and Sentence Form: Topic, Focus, and the Mental Representations ofDiscourse Referents. Cambridge Studies in Linguistics. Cambridge University Press.

J. H. Lau, A. Clark, and S. Lappin. 2017. Grammaticality, acceptability, and probability: A probabilistic view of linguistic knowledge. Cognitive Science, 41(5):1202–1241.

Tal Linzen, Emmanuel Dupoux, and Yoav Goldberg. 2016. Assessing the ability of lstms to learn syntax-sensitive dependencies. Preprint, arXiv:1611.01368.

Emily Walker Manetta. 2012. Reconsidering rightward scrambling: Postverbal constituents in hindi-urdu. Linguistic Inquiry, 43:43–74.

Mitchell P. Marcus, Beatrice Santorini, and Mary Ann Marcinkiewicz. 1995. Penn treebank ii 2 - ldc95t7.

Rebecca Marvin and Tal Linzen. 2018. Targeted syntactic evaluation of language models. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 1192–1202, Brussels, Belgium. Association for Computational Linguistics.

David J. Medeiros, Paul Mains, and Kevin B. Mc-Gowan. 2021. Ceiling effects on weight in heavy np shift. Linguistic Inquiry, 52(2):426–440.

Kanishka Misra. 2022. minicons: Enabling flexible behavioral and representational analyses of transformer language models. arXiv preprint arXiv:2203.13112.

Savithry Namboodiripad. 2017. An experimental approach to variation and variability in constituent order. University of California, San Diego.

Savithry Namboodiripad. 2019. A gradient approach to flexible constituent order.

OpenAI. 2024. Gpt-4 technical report. Preprint, arXiv:2303.08774.

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback. Preprint, arXiv:2203.02155.

Steven Pinker. 2007. The Language Instinct. Harper-Collins.

Randolph Quirk, Sidney Greenbaum, Geoffrey Leech, and Jan Svartvik. 1975. Review of: A grammar of contemporary english, by randolph quirk, sidney greenbaum, geoffrey leech and jan svartvik.

Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 2019. Language models are unsupervised multitask learners.

Sebastian Schuster and Tal Linzen. 2022. When a sentence does not introduce a discourse entity, transformer-based models still sometimes refer to it. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 969–982.

Márton Sóskuthy. 2017. Generalised additive mixed models for dynamic analysis in linguistics: a practical introduction. Preprint, arXiv:1703.05339.

Hua Wang and Haitao Liu. 2014. The effects of length and complexity on constituent ordering in written english. Poznan Studies in Contemporary Linguistics, 50(4):477–494.

Alex Warstadt, Aaron Mueller, Leshem Choshen, Ethan Wilcox, Chengxu Zhuang, Juan Ciro, Rafael Mosquera, Bhargavi Paranjabe, Adina Williams, Tal Linzen, and Ryan Cotterell. 2023. Findings of the BabyLM challenge: Sample-efficient pretraining on developmentally plausible corpora. In Proceedings ofthe BabyLM Challenge at the 27th Conference on Computational Natural Language Learning, pages 1–34, Singapore. Association for Computational Linguistics.

Thomas Wasow. 1997a. End-weight from the speaker’s perspective. Journal of Psycholinguistic Research, 26(3):347–361.

Thomas Wasow. 1997b. Remarks on grammatical weight. Language Variation and Change, 9(1):81–105.

Thomas Wasow. 2002. Postverbal behavior. Language, 80(2):327–331.

Thomas Wasow and Jennifer Arnold. 2003. Postverbal constituent ordering in English, page 119–154. De Gruyter Mouton, Berlin, New York.

Ethan Wilcox, Roger Levy, and Richard Futrell. 2019. What syntactic structures block dependencies in rnn language models? Preprint, arXiv:1905.10431.

Ethan Wilcox, Roger Levy, Takashi Morita, and Richard Futrell. 2018. What do RNN language models learn about filler–gap dependencies? In Proceedings of the 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP, pages 211–221, Brussels, Belgium. Association for Computational Linguistics.

Simon N. Wood. 2017. Generalized Additive Models: An Introduction with R, Second Edition, 2 edition. Chapman and Hall/CRC.

## A Appendix

## A.1 Regression Table

<table><tr><td>Model</td><td>Var</td><td>Full</td><td> $\mathbf { T o k e n } _ { a b }$ </td><td> $\mathbf { W o r d } _ { a b }$ </td><td> $\mathbf { S y l l } _ { a b }$ </td><td> $\mathbf { M o d } s _ { a b }$ </td></tr><tr><td>GPT-2</td><td>PM</td><td>0.543</td><td>0.533</td><td>0.515</td><td>0.457</td><td>0.517</td></tr><tr><td>GPT-2 Med</td><td>PM</td><td>0.605</td><td>0.602</td><td>0.581</td><td>0.534</td><td>0.580</td></tr><tr><td>GPT-2 Large</td><td>PM</td><td>0.611</td><td>0.605</td><td>0.579</td><td>0.556</td><td>0.585</td></tr><tr><td>GPT-2 XL</td><td>PM</td><td>0.613</td><td>0.610</td><td>0.583</td><td>0.561</td><td>0.590</td></tr><tr><td>Llama-3</td><td>PM</td><td>0.608</td><td>0.603</td><td>0.586</td><td>0.564</td><td>0.590</td></tr><tr><td>Llama-3 I</td><td>PM</td><td>0.627</td><td>0.619</td><td>0.599</td><td>0.555</td><td>0.602</td></tr><tr><td>BabyOPT</td><td>PM</td><td>0.562</td><td>0.550</td><td>0.534</td><td>0.507</td><td>0.542</td></tr><tr><td>BabyLlama</td><td>PM</td><td>0.719</td><td>0.702</td><td>0.663</td><td>0.651</td><td>0.701</td></tr><tr><td>Mistral v0.3</td><td>PM</td><td>0.606</td><td>0.588</td><td>0.563</td><td>0.565</td><td>0.588</td></tr><tr><td>Mistral v0.3 I</td><td>PM</td><td>0.630</td><td>0.618</td><td>0.585</td><td>0.601</td><td>0.618</td></tr><tr><td>OLMo</td><td>PM</td><td>0.655</td><td>0.639</td><td>0.623</td><td>0.612</td><td>0.637</td></tr><tr><td>OLMo I</td><td>PM</td><td>0.511</td><td>0.505</td><td>0.494</td><td>0.469</td><td>0.481</td></tr><tr><td>GPT-2</td><td>MPP</td><td>0.379</td><td>0.325</td><td>0.337</td><td>0.264</td><td>0.370</td></tr><tr><td>GPT-2 Med</td><td>MPP</td><td>0.368</td><td>0.309</td><td>0.302</td><td>0.300</td><td>0.356</td></tr><tr><td>GPT-2 Large</td><td>MPP</td><td>0.311</td><td>0.265</td><td>0.273</td><td>0.213</td><td>0.302</td></tr><tr><td>GPT-2 XL</td><td>MPP</td><td>0.323</td><td>0.249</td><td>0.260</td><td>0.244</td><td>0.310</td></tr><tr><td>Llama-3</td><td>MPP</td><td>0.358</td><td>0.316</td><td>0.321</td><td>0.271</td><td>0.351</td></tr><tr><td>Llama-3 I</td><td>MPP</td><td>0.320</td><td>0.284</td><td>0.291</td><td>0.208</td><td>0.313</td></tr><tr><td>BabyOPT</td><td>MPP</td><td>0.441</td><td>0.433</td><td>0.435</td><td>0.368</td><td>0.432</td></tr><tr><td>BabyLlama</td><td>MPP</td><td>0.310</td><td>0.297</td><td>0.298</td><td>0.281</td><td>0.306</td></tr><tr><td>Mistral v0.3</td><td>MPP</td><td>0.325</td><td>0.302</td><td>0.315</td><td>0.239</td><td>0.314</td></tr><tr><td>Mistral v0.3 I</td><td>MPP</td><td>0.269</td><td>0.238</td><td>0.258</td><td>0.186</td><td>0.256</td></tr><tr><td>OLMo</td><td>MPP</td><td>0.306</td><td>0.303</td><td>0.296</td><td>0.222</td><td>0.294</td></tr><tr><td>OLMo I</td><td>MPP</td><td>0.263</td><td>0.257</td><td>0.253</td><td>0.237</td><td>0.250</td></tr><tr><td>GPT-2</td><td>HNPS</td><td>0.356</td><td>0.343</td><td>0.332</td><td>0.278</td><td>0.349</td></tr><tr><td>GPT-2 Med</td><td>HNPS</td><td>0.654</td><td>0.629</td><td>0.616</td><td>0.540</td><td>0.635</td></tr><tr><td>GPT-2 Large</td><td>HNPS</td><td>0.589</td><td>0.570</td><td>0.554</td><td>0.430</td><td>0.585</td></tr><tr><td>GPT-2 XL</td><td>HNPS</td><td>0.592</td><td>0.587</td><td>0.557</td><td>0.527</td><td>0.589</td></tr><tr><td>Llama-3</td><td>HNPS</td><td>0.542</td><td>0.524</td><td>0.519</td><td>0.485</td><td>0.538</td></tr><tr><td>Llama-3 I</td><td>HNPS</td><td>0.452</td><td>0.438</td><td>0.408</td><td>0.359</td><td>0.451</td></tr><tr><td>BabyOPT</td><td>HNPS</td><td>0.491</td><td>0.457</td><td>0.472</td><td>0.421</td><td>0.450</td></tr><tr><td>BabyLlama</td><td>HNPS</td><td>0.527</td><td>0.514</td><td>0.466</td><td>0.415</td><td>0.509</td></tr><tr><td>Mistral v0.3</td><td>HNPS</td><td>0.488</td><td>0.476</td><td>0.461</td><td>0.451</td><td>0.458</td></tr><tr><td>Mistral v0.3 I</td><td>HNPS</td><td>0.467</td><td>0.441</td><td>0.436</td><td>0.438</td><td>0.434</td></tr><tr><td>OLMo</td><td>HNPS</td><td>0.518</td><td>0.500</td><td>0.468</td><td>0.443</td><td>0.513</td></tr><tr><td>OLMo I</td><td>HNPS</td><td>0.411</td><td>0.391</td><td>0.377</td><td>0.290</td><td>0.395</td></tr><tr><td>GPT-2</td><td>DA</td><td>0.562</td><td>0.554</td><td>0.542</td><td>0.545</td><td>0.549</td></tr><tr><td>GPT-2 Med</td><td>DA</td><td>0.571</td><td>0.561</td><td>0.552</td><td>0.568</td><td>0.568</td></tr><tr><td>GPT-2 Large</td><td>DA</td><td>0.541</td><td>0.528</td><td>0.506</td><td>0.533</td><td>0.533</td></tr><tr><td>GPT-2 XL</td><td>DA</td><td>0.561</td><td>0.553</td><td>0.540</td><td>0.552</td><td>0.555</td></tr><tr><td>Llama-3</td><td>DA</td><td>0.554</td><td>0.544</td><td>0.532</td><td>0.538</td><td>0.549</td></tr><tr><td>Llama-3 I</td><td>DA</td><td>0.503</td><td>0.493</td><td>0.490</td><td>0.486</td><td>0.496</td></tr><tr><td>BabyOPT</td><td>DA</td><td>0.252</td><td>0.242</td><td>0.238</td><td>0.241</td><td>0.237</td></tr><tr><td>BabyLlama</td><td>DA</td><td>0.630</td><td>0.616</td><td>0.603</td><td>0.622</td><td>0.623</td></tr><tr><td>Mistral v0.3</td><td>DA</td><td>0.540</td><td>0.529</td><td>0.526</td><td>0.528</td><td>0.534</td></tr><tr><td>Mistral v0.3 I</td><td>DA</td><td>0.478</td><td>0.469</td><td>0.461</td><td>0.467</td><td>0.467</td></tr><tr><td>OLMo</td><td>DA</td><td>0.467</td><td>0.460</td><td>0.458</td><td>0.449</td><td>0.459</td></tr><tr><td>OLMo I</td><td>DA</td><td>0.392</td><td>0.384</td><td>0.388</td><td>0.374</td><td>0.385</td></tr></table>

Table 5: R-sq. scores for each model on each variable. $[ M e t r i c ] _ { a b }$ denotes the R-sq. score when excluding the metric; a larger difference from the original score denotes higher significance in contributing to the overall score.

## A.2 Model Behavior Plots

## A.2.1 Synthetic Data

Model Preference Scores on Heavy NP Shift  
![](images/1e188f14f817d26fc416b4ef48e60447638bb3fbe8377f86029f8c686d32fa8c.jpg)

Model Preference Scores on Particle Movement

![](images/a78b081ce7c98d7d2ddc36c53ef2043ee0f88f086d2162d50b400cad8641d420.jpg)

## A.2.2 Mined Data

Model Preference Scores on Dative Alternation  
![](images/498c96d7e1e9d85a0b9deb8013d832de3877eb9ffc557e9f10538828ebdfb0cb.jpg)

Model Preference Scores on Multiple PPs  
![](images/5f556f8a1a9e5eebcbc807b541fa266a419cc4e9e2be597a0cfc4dd4b6cb99d2.jpg)

Model Preference Scores on Mined Heavy NP Shift  
![](images/2a3960907c898cba415899473c2dca9c01a684563474435802c9a5b1e302c276.jpg)

Model Preference Scores on Mined Particle Movement  
![](images/600aa9518306a176ba54ffa8d623110e591cd74cb4d3d7f107fc303b92be55c2.jpg)

Model Preference Scores on Mined Dative Alternation  
![](images/bb6bbf1a15a5753c0c3ccf9e9c02ea21f5b24cce336347edb69cb44d2c84bdd9.jpg)

Model Preference Scores on Mined Multiple PPs  
![](images/8aad0ef83cbe141691c9c5bfcabb3403718fdadc44107b51cdb70b3fc99a06e2.jpg)

## A.3 Human-Model Agreement Plots

Correlation Between GPT-2 and Human Responses on HNPS  
![](images/b813c4c2d2e4cf6ad4077ca564206a97a6277dfe7675a5b5649c12e262318e8c.jpg)

Correlation Between GPT-2 XL and Human Responses on HNPS  
![](images/a47edf24afc89cda8030aa71c702c6dc64da9f34e866114030cc782582d5febc.jpg)

Correlation Between GPT-2 Medium and Human Responses on HNF  
![](images/044ce81d9ae9f1715c8e79bc4d455d8fb2086cf031390d32b8cb8075a4448ee7.jpg)

Correlation Between BabyOPT and Human Responses on HNPS  
![](images/0546e6f6cf40be22f3a18aab0c7df3ab21d3863053b5eab5ea598646f95d1bda.jpg)

Correlation Between GPT-2 Large and Human Responses on HNPS  
![](images/aebc39c54c14907cd61406dd6cfa2db7106fa6d6b0a777f69375436a9bf6bcbf.jpg)

Correlation Between BabyLlama and Human Responses on HNPS  
![](images/3bc604eede0ea51b66527e2522c1af22bd9b2fa16b3fa99f7f01622aac05aa97.jpg)

Correlation Between Llama-3 and Human Responses on HNPS  
![](images/742817fc9aa8d4df69c66d6d8aa718993e7d2f511dbd659e46685c7845925642.jpg)

Correlation Between Llama-3 Instruct and Human Responses on HN  
![](images/5ecb0af18de1e50574f27223a320468db9c40c319a3f5a72eee8b4025e8c9a40.jpg)

Correlation Between OLMo Instruct and Human Responses on HNP  
![](images/178888a92b47e75d36683c5e7a39c1c6c9ec017f399e73cefa95513ce226d484.jpg)

Correlation Between Mistral v0.3 and Human Responses on HNPS  
![](images/76ce8777a6751c2199ff49fe59251e2a761934ddc8675e9c4f982bb3199f5d2c.jpg)

Correlation Between GPT-2 and Human Responses on PM  
![](images/233f866cf24818b3ab43f9b4c36a7914df2e4d0d55d573911350d465fe0afc1e.jpg)

Correlation Between Mistral v0.3 Instruct and Human Responses on  
![](images/a7ef960d1d82771464249d627204bd85554e7bb3ac68decfb39d1a23ef7789f1.jpg)

Correlation Between GPT-2 Medium and Human Responses on PM  
![](images/18c812078c3fb1bc745de96cc53791ddeae60a0fe8261b1111e71889a99c1585.jpg)

Correlation Between OLMo 7B and Human Responses on HNPS  
![](images/d1da66994376206ab6922c138cfed742249ab800cec88074b1ae1c443c822bf3.jpg)

Correlation Between GPT-2 Large and Human Responses on PM  
![](images/66341756afe0de354e444c9bc32f3cb40c06bec7a9069549016c63e9b876ceee.jpg)

Correlation Between GPT-2 XL and Human Responses on PM  
![](images/292e3cccde38b41f4cf42133a8432af7cce0a01bf1626ac9bc1c22522f46b590.jpg)

Correlation Between Llama-3 Instruct and Human Responses on PM  
![](images/f8cb7c4d5b8645d5c3d6532f771b9b1cd5c2a751a668fdd53ed92a03ec2de8ae.jpg)

Correlation Between BabyOPT and Human Responses on PM  
![](images/4f37c2f82dcbd972da2147ec99868cd71f7b3baa0dd8c11ec63dedc2259e92d3.jpg)

Correlation Between Mistral v0.3 and Human Responses on PM  
![](images/79113efc502f03a8a660588ee721e8f5d4a179f0a5270205be01f3754871a65e.jpg)

Correlation Between BabyLlama and Human Responses on PM  
![](images/fc399fb75dd9c4ece8d7dc533824340b999851c8d967397803cfb887ae16638b.jpg)

Correlation Between Mistral v0.3 Instruct and Human Responses on  
![](images/c8b642b79ef2cf4ba87f0b465cec914f1673c28d7faf2b6588f849aed22f5a04.jpg)

Correlation Between Llama-3 and Human Responses on PM  
![](images/ebdcdc98831c6fc6fdd2a490c31b0e85e668987b159365d15e101f2eb4d3162b.jpg)

Correlation Between OLMo and Human Responses on PM  
![](images/f468a4eca43d297111df2cace78872893781ee6d47e81f71ca92aa0c9c30d7cf.jpg)

Correlation Between OLMo Instruct and Human Responses on PM  
![](images/89e33172d2440b824ed2b071797edf03baa5865e329bcca764c70ebcfbec38d1.jpg)

Correlation Between GPT-2 XL and Human Responses on DA  
![](images/fa05e0e936f019a6363710829669094873b9336b8b1c3a82dde9f932a101f9af.jpg)

Correlation Between GPT-2 and Human Responses on DA  
![](images/afc10ea26825d430c8f935778c72b3cef437e6092332bbfdf7008c1c8b261871.jpg)

Correlation Between BabyOPT and Human Responses on DA  
![](images/7a1a7b01a0a5a9ffe998cd3a3d374fc637a7e81a95f5fdccc57a88c5b53efc0e.jpg)

Correlation Between GPT-2 Medium and Human Responses on DA  
![](images/c44fa6033f696fc9c9889633bb1d22362a4a477b4a586a570b9288891565ca09.jpg)

Correlation Between BabyLlama and Human Responses on DA  
![](images/218a17df544d153f707fc21b0699777f519b7aeb770fa1fcecc37cc2889893a8.jpg)

Correlation Between GPT-2 Large and Human Responses on DA  
![](images/4e195f0ce8cda2d6a420781f713ee93f8489790831a2d65ef298262da4d7395e.jpg)

Correlation Between Llama-3 and Human Responses on DA  
![](images/73a37c4484021b0bce4736537691f309bbb6a23d1c9aa62e4ec6ad636de3dcf4.jpg)

Correlation Between Llama-3 Instruct and Human Responses on DA  
![](images/9d9d637b84e1dbab9121ddab5343e63b95e99f9231006b9a0f475578ff7e4ad8.jpg)

Correlation Between OLMo Instruct and Human Responses on DA  
![](images/5e5db4951e325c3801a3a376e0fd69b5b7c9698161b274245125c1526ce8148b.jpg)

Correlation Between Mistral v0.3 and Human Responses on DA  
![](images/27316a761be8a23d6090837a946902c1a77ad9426b7f555044201dd451db6232.jpg)

Correlation Between GPT-2 and Human Responses on MPP  
![](images/70fa833d694fb23344c1c87c2dbea97c183acae85e30b88e7209786e496f932a.jpg)

Correlation Between Mistral v0.3 Instruct and Human Responses or  
![](images/8454e5206df531fe8f7edf7d1761e28a9ceb035e62084203597468b3270b9d0d.jpg)

Correlation Between GPT-2 Medium and Human Responses on MPR  
![](images/453a45a71dd7733153d096aebd6689dc6c92c7905ff10fd51402f115f62b7ee1.jpg)

Correlation Between OLMo and Human Responses on DA  
![](images/60b14f50e688cb2845ff813f5b8d9f1475c1fe54ca3ac3581fe7beae70db5bcc.jpg)

Correlation Between GPT-2 Large and Human Responses on MPP  
![](images/fa3fe3091c46c0ec6a4fabd1776e1d35f4ea7ca8c7af4bf07b6e8a1bdc169cdd.jpg)

Correlation Between GPT-2 XL and Human Responses on MPP  
![](images/980ec1d69b2c80cb0d9c0d1b550426b6f5b7c4907c4ee4fbae984b906f1f1cb1.jpg)

Correlation Between Llama-3 Instruct and Human Responses on MF  
![](images/3aa4bde8f2d5c00e975a5404b92bac7d5b369c31c50df496064e83370f48626d.jpg)

Correlation Between BabyOPT and Human Responses on MPP  
![](images/ac1dfc29c078976695d3d03ba8cecc6f2953fa79775748dece94d995292fa0ce.jpg)

Correlation Between Mistral v0.3 and Human Responses on MPP  
![](images/5b338b87d617d06940734b94380ecd17b710a5f50bcff157dbe1802919a868b5.jpg)

Correlation Between BabyLlama and Human Responses on MPP  
![](images/2bdbd32bc8240cab12197d6b1152c0e518a518104b13dede29c34ec578e3c2cd.jpg)

Correlation Between Mistral v0.3 Instruct and Human Responses on  
![](images/d12d5f9ef00584b596365078b33e73d07501ab2c8b182cc42acc3556f7d537c7.jpg)

Correlation Between Llama-3 and Human Responses on MPP  
![](images/8cc5639e32fb968c348f750b9fb58b55fb8a6c2632001a5628b51e2700b432a4.jpg)

Correlation Between OLMo and Human Responses on MPP  
![](images/6672521b63e9685f4844c966a2409fc29f474a94595caa171fd9c77f7bfcdaf3.jpg)

Correlation Between OLMo Instruct and Human Responses on MPF  
![](images/0dd1184ce8cc4f88546107153668c99ad90cb943b9ee9ec84c20ccfc267f452c.jpg)