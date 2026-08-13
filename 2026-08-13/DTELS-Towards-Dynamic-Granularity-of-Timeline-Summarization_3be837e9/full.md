# DTELS: Towards Dynamic Granularity of Timeline Summarization

Chenlong Zhang<sup>1,2,\*</sup>, Tong Zhou<sup>1,2,\*</sup>, Pengfei Cao<sup>1,2</sup>, Zhuoran Jin<sup>1,2</sup>, Yubo Chen<sup>1,2,†</sup>, Kang Liu<sup>1,2</sup>, Jun Zhao <sup>1,2</sup>

<sup>1</sup>The Key Laboratory of Cognition and Decision Intelligence for Complex Systems, Institute of Automation, Chinese Academy of Sciences, Beijing, China, <sup>2</sup>School of Artificial Intelligence, University of Chinese Academy of Sciences, Beijing, China, {zhangchenlong2023, tong.zhou}@ia.ac.cn {pengfei.cao, zhuoran.jin, yubo.chen, kliu, jzhao}@nlpr.ia.ac.cn

## Abstract

The rapid proliferation of online news has posed significant challenges in tracking the continuous development of news topics. Traditional timeline summarization constructs a chronological summary of the events but often lacks the flexibility to meet the diverse granularity needs. To overcome this limitation, we introduce a new paradigm, Dynamic-granularity TimELine Summarization, (DTELS), which aims to construct adaptive timelines based on user instructions or requirements. This paper establishes a comprehensive benchmark for DTLES that includes: (1) an evaluation framework grounded in journalistic standards to assess the timeline quality across four dimensions: Informativeness, Granular Consistency, Factuality, and Coherence; (2) a large-scale, multi-source dataset with multiple granularity timeline annotations based on a consensus process to facilitate authority; (3) extensive experiments and analysis with two proposed solutions based on Large Language Models (LLMs) and existing state-of-the-art TLS methods. The experimental results demonstrate the effectiveness of the proposed solutions. However, even the most advanced LLMs struggle to consistently generate timelines that are both informative and granularly consistent, highlighting the challenges of the DTELS task.<sup>1</sup>

## 1 Introduction

With the surge in news production, the volume of news articles published on the internet is expanding rapidly, making it increasingly challenging to track the developments of news topics.

TimeLine Summarization (TLS) (Wang et al., 2016; Li et al., 2021; Chen et al., 2023b; Cao et al., 2023; Zhang et al., 2024) aims to construct a sequence of chronologically ordered summaries.

![](images/3134337b025f7b5ffb8bf8a1422f4d973a494aea81d75955b2e41034e21216c9.jpg)  
Figure 1: (a) In traditional TLS, a timeline with a predefined number of node summaries is constructed. (b) DTELS provides timelines at different granular levels: network engineers require the technical causes and solutions to data breaches, therefore, a fine-grained granularity is preferred to track the technical details. For investors, a coarse-grained timeline showing the full picture of the breach’s influence on investment may suffice.

These timelines provide traceable skeletons to support various applications including event modeling (He et al., 2024), policymaking (Chen et al., 2024b), crisis management, and temporal analysis (Ambe, 2023; Hu et al., 2022).

Traditional TLS typically constructs static timelines at a fixed granularity: in Figure 1a, for a specific news topic, the granularity is heuristically predefined by the number of “salient events”. However, in practice, the granularity of the timeline should change dynamically, depending on user needs and the nature of news topics: For readers: Different readers have very different requirements on granularity for the same topic (see example in Figure 1b). For news topics: A reader’s need for granularity varies across topics. One may require fine-grained timelines for trending news such as local disasters to follow the progression and immediate impacts. In contrast, for long-standing topics like the Russian-Ukrainian war, people may warrant coarse-grained timelines with wider intervals to capture broader developments.

Unfortunately, existing TLS ignores the importance of providing timelines at dynamic granularities. Existing evaluations also lack appropriate reference annotations and metrics to comprehensively evaluate timelines at dynamic granularities.

In this paper, we propose a new paradigm: Dynamic granularity TimELine Summarization (DTELS). We define the granularity of a timeline by the degree of omission between the node summaries. Given a collection of news articles on the specific news topic and granularity requirements, our task aims to construct dynamic-granularity timelines tailored to various requirements.

Meanwhile, to take the study a step further, grounded in the criteria from journalism (Kunelius, 2006), an ideal timeline should: (1) convey information effectively, avoiding redundant events, and ensuring that no important events are missed. (2) maintain consistency with the granular requirements. (3) ensure the mentioned events in each summary are factually correct. (4) be selfcontained, allowing the reader to clearly understand the context. By adhering to these criteria, we set the standard that not only meets the dynamic granularity needs but also upholds high quality.

We construct a benchmark including:

Evaluation Framework. To comprehensively measure a timeline, We propose metrics that address the aforementioned criteria:

Informativeness: This metric evaluates the effective volume of information in the node summaries. We propose a “mount-then-measure” paradigm to align predicted node summaries to those in the reference timeline based on the entailment score of the “event atoms”, which represents the smallest unit of event information within a sentence.

Granular Consistency: The granularity is reflected by the amount of event information omitted between adjacent nodes. The more events omitted, the coarser the granularity is. We regard adjacent nodes as edges and calculate the ratio of mounts on the correct reference granularity edge.

Factuality: Considering the hallucinated contents and misinformation in the era of Large Language Models (LLMs) (Ji et al., 2023; Li et al., 2023; Zhang et al., 2023; Sun et al., 2024,?), it is crucial to ensure the information accuracy. We introduce a factuality metric that incorporates atomslevel entailment verification from reference news articles to measure the non-fabricated information in each summary.

Coherence: Coherence is pivotal in summarization tasks (Goyal et al., 2022; Steen and Markert, 2022; Jing et al., 2023). We adopt this metric for our task, ensuring that summaries are generated in a structurally, linguistically, and stylistically coherent manner. To facilitate this, we design a review form to guide the most advanced LLMs for coherence evaluation.

We verify the effectiveness of the metrics, showcasing high alignment with humans.

Dataset Construction. To ensure evaluation across varying granularities, we meticulously construct a dataset called DTELS-Bench. We initially collect diverse news topics and journalists’ annotations on timelines from news events websites<sup>2</sup>. We then gather corresponding large-scale news articles from diverse sources, resulting in a large-scale, multi-source Chinese dataset. Subsequently, the reference timelines are annotated at three predefined granularities through a consensus-based automated annotation. Finally, the timelines are refined by specialists to ensure the authority.x

Comprehensive Evaluation. In the experiments, we present two LLM-based solutions for long-context and context-limited LLMs. We systematically evaluate our proposed solutions with multiple LLMs. In addition, we compare existing state-of-the-art extractive TLS approaches. Experiments show that our LLM-based solutions dominate in all dimensions, however, they fall short of providing high-quality information and aligning the required granularity. We then analyze the performance of these methods across various settings of DTELS. The results indicate that there is still substantial room for improvement in DTELS.

To sum up, our contributions are as follows:

• We propose a new task: Dynamic granularity TimELine Summarization (DTELS). It aims to summarize timelines tailored to the unique needs of dynamic granularities.

• We build an event-centric evaluation framework. Extending from journalism, we propose metrics to evaluate timelines in four dimensions: informativeness, granular consistency, factuality, and coherence. Experiments with human annotators demonstrate the effectiveness of our metrics.

• We collect a large-scale, multi-source Chinese dataset, DTELS-Bench<sup>3</sup>, which contains 543 news topics with 55,432 articles from 2,858 sources. It covers three predefined granularities annotated via a consensus-based mechanism. The expert’s refinement enhances the annotation authority.

• We evaluate existing state-of-the-art TLS methods as well as LLMs with two proposed DTELS methods. Through extensive experiments, we find the proposed solutions outperform existing TLS methods, however, they are far from being an ideal solution to DTELS.

## 2 Related Works

## 2.1 Timeline Summarization Task

Timeline Summarization (TLS) has been a longstanding task in Natural Language Processing. The challenge of this task is to chronologically condense information from hundreds of articles. Existing work mainly focuses on generating and evaluating timelines at fixed granularity with sole evaluation metrics. The task is first proposed by Swan and Allan (2000). Kessler et al. (2012) presents an approach for detecting important dates to automatically construct timelines. Tran et al. (2013) provides a clear definition of TLS with fixed numbers of nodes for each news topic. Nguyen et al. (2014) introduces a system by selecting and ranking events from multiple documents. Martschat and Markert (2017) proposes an alignment-based ROUGE score and they proposed a submodularity Framework (Martschat and Markert, 2018) to construct timelines. La Quatra et al. (2021) propose a novel date selection method. However, these works disregard the varying granularity requirements. Besides, the ROUGE-based evaluation (Gholipour Ghalandari and Ifrim, 2020) results can be significantly affected by the narrative styles.

## 2.2 Timeline Summarization Dataset

Tran et al. (2013) proposes the “T17” dataset discussing famous topics. Tran et al. (2013) constructs the “Crisis” dataset focusing on long-span armed conflict topics. Wang et al. (2015). Gholipour Ghalandari and Ifrim (2020) builds “Entities” with longer time-ranges topics typed around ‘people’ and ‘disasters’. Rajaby Faghihi et al. (2022) constructs a dataset called “CrisisTLS” focusing on the local crisis. Li et al. (2021) build a larger dataset $\mathrm { T L S } _ { 1 0 0 }$ covering various topics. Some recent works also propose LLM-based methods (Song et al., 2024; Hu et al., 2024; Chen et al., 2024a). However, they lack annotations across multiple levels. Besides, topics on existing datasets are likely to have been leaked in the pretraining corpus of LLMs, leading to potential unfair evaluations.

## 3 Task Definition

## 3.1 Timeline Summarization

In traditional text summarization, previous works have explored controlling granularity levels but lack a clear definition of "granularity" and focus on varying annotations. For the first time in TLS, we define and measure timeline granularity from an event-centric perspective. Consider a news topic q spans over a time range $\mathcal { T } = \{ t _ { 1 } , \ldots , t _ { n } \}$ and a corresponding set of news articles $\mathcal { A } = \{ A _ { t _ { 1 } } , A _ { t _ { 2 } } , . . . , A _ { t _ { n } } \}$ as inputs. Each date $t _ { i } \in \tau$ is accompanied by multiple articles $A _ { t _ { i } } = \{ a _ { t _ { i } , 1 } , \dots , a _ { t _ { i } , m } \}$ . The task is to generate a temporal sequence of summaries by model Θ:

$$
S = \Theta ( \mathbf { q } , \mathcal { A } ) ,\tag{1}
$$

where $\boldsymbol { S } = \{ S _ { t _ { 1 } } , \ldots , S _ { t _ { k } } \}$ and k corresponds to the node numbers. $S _ { t _ { i } }$ includes a timestamp $t _ { i } \in T$ and summary $s _ { i } , \mathrm { i } . \mathrm { e } . , s _ { i }$ is a concise summary of the news event at time $t _ { i }$ . Typically, for a specific news topic, the amount of node summaries k is fixed based on the number of salient events.

## 3.2 Dynamic-granularity Timeline Summarization

In traditional text summarization, previous works (Zhong et al., 2022; Shen et al., 2022) have explored controlling granularity levels but lack a clear definition of "granularity" and focus on varying annotations. For the first time in TLS, we define and measure timeline granularity from an event-centric perspective. In traditional text summarization, previous works (Zhong et al., 2022; Shen et al., 2022) have explored controlling granularity levels but lack a clear definition of "granularity" and focus on varying annotations. For the first time in TLS, we define and measure timeline granularity from an event-centric perspective. Considering that event information passed through nodes is certain, we define granularity as the degree to which neighboring nodes are omitted: coarse-grained timelines with fewer nodes should omit less important events, while fine-grained timelines capture detailed chronological chains.

![](images/3f33a4193ce50e7a17035a55b6c2e15c08497bb6156e69065e0889d6434c0160.jpg)  
Figure 2: Examples of metrics. Green nodes indicate positive examples and red nodes indicate negative examples.

We introduce a granularity indicator “Granularity: $[ \mathcal { G } _ { o } ] ^ { \dag }$ as an additional input to indicate the desired granularity. It can be either a specific number of nodes or a natural language instruction. Here, m denotes the chosen granularity level of the timeline. Based on this, the model Θ generates a timeline summarization at the specific granularity :

$$
\begin{array} { r } { S ^ { \mathcal { G } _ { o } } = \Theta ( \mathcal { G } _ { o } , \mathbf { q } , \mathcal { A } ) , } \end{array}\tag{2}
$$

where $\mathcal { S } ^ { \mathcal { G } _ { o } } = \{ ( t _ { 1 } ^ { \mathcal { G } _ { o } } , s _ { 1 } ^ { \mathcal { G } _ { o } } ) , \dots , ( t _ { k } ^ { \mathcal { G } _ { o } } , s _ { k } ^ { \mathcal { G } _ { o } } ) \}$ . the granularity of a timeline for a topic can vary: ${ \mathcal { G } } = \{ { \mathcal { G } } _ { 1 } , { \mathcal { G } } _ { 2 } , \ldots , { \mathcal { G } } _ { n } \}$ , where n ranges from coarse to fine granularity. This approach ensures that the summarization output matches the specified granularity requirements. The reference timelines are annotated at multiple granularity levels<sup>4</sup>.

## 4 Evaluation Framework

## 4.1 Event Atoms

The references in narrative summarization are influenced by the annotator’s preference and narrative style. Existing ROUGE-based evaluation (Lin, 2004) approaches evaluate the n-gram similarity, which can be inadequate in fairly reflecting the quality (Ng and Abrecht, 2015) of the outputs. For example, given reference and predicted summaries “ Barcelona announced the departure of Lionel Messi ” and “King of Football, Messi , has left the club he served, Barcelona . the $R o u g e _ { 1 } = 3 / 7$ varies with different narrative styles in predicted nodes. We hope to find a consistent measurement that is not affected by narrative style and granularity.

![](images/3661f851124bff60c12b7c57827d1e9eea5651c7b4495c8d8c55ec9cb19a198d.jpg)  
Figure 3: The predicted timeline is mounted to the reference according to “Optimal Matching”. The colored nodes denote mounted nodes.

Inspired by recent advances in atom-based evaluations (Min et al., 2023; Setty, 2024; Xu et al., 2024), we introduce the concept of “event atoms” as the fundamental units for evaluation, which remain consistent despite changes in narrative style and granularity. We define the “event atoms” as the smallest distinguishable unit of events within a sentence. Each node summary $s _ { i }$ can be decomposed into a certain number of atoms: $\mathcal { E } _ { i }$ = $\{ e _ { i , 1 } , . . . , e _ { i , m } \} = D e c o m p o s e ( s _ { i } )$ , where m indicates the number of atoms. This function can be achieved by LLMs (detailed in Appendix A.2).

To evaluate a predicted node summary, we measure the amount of valuable event information it provides compared to the reference node summary using an entailment score. For a predicted node summary $\hat { s } _ { i }$ and a reference node summary $s _ { j }$ , their event atoms are $\hat { \mathcal { E } } _ { i }$ and $\mathcal { E } _ { j }$ , respectively. The entailment precision $e n t _ { p }$ can be measured by:

$$
e n t _ { p } ( \hat { s } _ { i } , s _ { j } ) = \frac { 1 } { | \hat { \mathcal { E } } _ { i } | } \sum _ { \hat { \varepsilon } _ { i , s } \in \hat { \mathcal { E } } _ { i } } E n t a i l ( \mathcal { E } _ { j } , \hat { \varepsilon } _ { i , s } ) ,\tag{3}
$$

where event atoms $\hat { \varepsilon } _ { i , s }$ derives from $\hat { \mathcal { E } } _ { i }$ Entail(Evidence, Claim) quantifies the entailment of event atoms: it returns 1 if the evidence entails the claim, and 0 if it contradicts or is unrelated to the claim. The function can be implemented by the widely used Natural Language Inference models. (Camburu et al., 2018; Klemen et al., 2024).

Similarly, we can get the entailment recall $e n t _ { r } ( { \hat { s } } _ { i } , s _ { j } )$ . The entailment F1 can be calculated:

$$
e n t _ { f 1 } ( \hat { s } _ { i } , s _ { j } ) = \frac { 2 * e n t _ { p } ( \hat { s } _ { i } , s _ { j } ) * e n t _ { r } ( \hat { s } _ { i } , s _ { j } ) } { e n t _ { p } ( \hat { s } _ { i } , s _ { j } ) + e n t _ { r } ( \hat { s } _ { i } , s _ { j } ) } .\tag{4}
$$

By adopting the score, we can evaluate the coverage of the node summaries over the references.

We propose a “mount-then-measure” paradigm, illustrated in Figure 3, to find the optimal mapping from the predicted timeline to the reference timeline. For a predicted node $\hat { \cal S } _ { i } = ( \hat { t } _ { i } , \hat { s } _ { i } )$ , we mount it to a specific reference node $\boldsymbol { S } _ { j } = \left( t _ { j } , s _ { j } \right)$ by computing the information score $I n f o S c o r e ( \hat { S } _ { i } , S _ { j } )$ Considering the matching of event information on the temporal dimension for timelines, we introduce a temporal interval penalty term δ:

$$
\delta _ { \hat { t } _ { i } , t _ { j } } = \frac { 1 } { | \hat { t } _ { i } - t _ { j } | ^ { 2 } + 1 } .\tag{5}
$$

Then, we can define the information score:

$$
I n f o S c o r e ( \hat { S } _ { i } , { S } _ { j } ) = \delta _ { \hat { t _ { i } } , { t _ { j } } } * e n t _ { f 1 } ( \hat { s } _ { i } , { s } _ { j } ) .\tag{6}
$$

## 4.2 Mount-then-measure Paradigm

The InfoScore() provides an objective measurement of the predicted nodes’ coverage from an eventcentric perspective. We can get the mapping cost between predicted and reference nodes via:

$$
m a p ( \hat { S } _ { i } \to S _ { j } ) = - I n f o S c o r e ( \hat { S } _ { i } , { S } _ { j } ) .\tag{7}
$$

The mount process for the entire timeline can be automatically completed by Hungarian algorithm (Kuhn, 1955) for a global optimal matching:

$$
{ \mathcal M } _ { \hat { \mathcal S } , S } = \arg \operatorname* { m i n } _ { { \mathcal M } } \sum _ { ( \hat { S } _ { i } , S _ { j } ) \in { \mathcal M } } m a p ( \hat { S } _ { i }  S _ { j } ) .\tag{8}
$$

This process determines a maximum coverage of the predicted timeline to the reference, enabling fine-grained evaluation that requires references.

## 4.3 Evaluation Metrics

To evaluate timelines from multiple perspectives, we adopt criteria in journalism (Kunelius, 2006) and categorize the quality of a timeline into four dimensions: Informativeness, Granular Consistency, Factuality, and Coherence. The subsequent section details the definition of these metrics.

Informativeness. Informativeness measures the extent to which the node summary captures the essential information of events (Cao et al., 2023). As illustrated in Figure 2a, it is important to ensure the timeline contains all key atoms at correct timestamps and is not overly verbose. We match references for each node summary by “mountthen-measure”. We calculate the informativeness $I n f o ( )$ after mounting the predicted timeline $\hat { S }$ to the reference timeline :

$$
I n f o ( \hat { \cal S } ) = \frac { 1 } { | \hat { \cal S } | } \sum _ { ( \hat { \cal S } _ { i } , { \cal S } _ { j } ) \in { \cal M } } I n f o { \cal S } c o r e ( \hat { \cal S } _ { i } , { \cal S } _ { j } ) .\tag{9}
$$

$\hat { S } _ { i }$ and $S _ { j }$ are predicted and reference nodes in Equation 8.

Granular Consistency. Granular consistency measures how well the timeline aligns with its reference in terms of granularity. As illustrated in Figure 2b, differences in granularity emerge not from individual node content but from relationships between adjacent nodes.

we extend “mount-then-measure” to edge views: For a predicted timeline at $\mathcal { G } _ { o } .$ , its edges are E<sup>ˆ</sup> = $\{ \hat { e } _ { 1 } , \hat { e } _ { 2 } , \dots , \hat { e } _ { k - 1 } \}$ , where $\hat { e } _ { m } = ( \hat { S } _ { m } , \hat { S } _ { m + 1 } )$ . The reference edges across all granularities are ${ \bf E } ^ { \mathcal { G } } =$ $\{ \mathbf { E } ^ { \mathcal { G } _ { 1 } } , \mathbf { E } ^ { \mathcal { G } _ { 2 } } , \ldots , \mathbf { E } ^ { \mathcal { G } _ { n } } \}$ . We calculate the mapping cost of aligning a predicted edge $\boldsymbol { \hat { e } } _ { m }$ to a reference edge $e _ { n } = ( S n , S n + 1 ) \in \mathbf { E } ^ { \mathcal { G } }$ using the formula:

$$
\begin{array} { c } { { m a p ( \hat { e } _ { m } \to e _ { n } ) = - I n f o S c o r e ( \hat { S } _ { m } , S _ { n } ) } } \\ { { - I n f o S c o r e ( \hat { S } _ { m + 1 } , S _ { n + 1 } ) . } } \end{array}\tag{10}
$$

We then mount $\boldsymbol { \hat { e } } _ { m }$ to a minimum cost $e _ { n }$

$$
\mathcal { M } _ { \hat { \mathbf { E } } , \mathbf { E } } = \arg \operatorname* { m i n } _ { \mathcal { M } } \sum _ { ( \hat { e } _ { m } , e _ { n } ) \in \mathcal { M } } m a p ( \hat { e } _ { m } \to e _ { n } ) .\tag{11}
$$

Finally, granular consistency is measured by the number of edges that are aligned with the correct granularity level $\mathcal { G } _ { o }$ :

$$
G r a n u _ { i } ( \hat { \mathcal { S } } ) = \frac { 1 } { | \mathbf { E } | } \sum _ { ( \hat { e } _ { m } , e _ { n } ) \in \mathcal { M } } [ e _ { n } \in \mathbf { E } ^ { \mathcal { G } _ { o } } ] ,\tag{12}
$$

where [] is a binary function.

Factuality. Factuality measures the faithfulness of summaries, which is crucial given the potential for hallucinated and fabricated content in LLMs (Chen et al., 2023a; Gekhman et al., 2023). In DTELS, factuality assesses whether the information in the timeline can be traced back to support articles. We use a selection mechanism to choose reference articles as support for each predicted node: For a given timestamp $\hat { t } _ { i }$ in the predicted node $\hat { S } _ { i }$ we select reference articles $\boldsymbol { \mathcal { A } } _ { \hat { t } _ { i } }$ that are closest to the timestamp. The factuality score is then computed using entailment precision:

$$
F a c t ( \hat { S } ) = \frac { 1 } { | \hat { S } | } \sum _ { ( \hat { s } _ { i } , \hat { t } _ { i } ) \in \hat { S } } e n t _ { p } ( \hat { s } _ { i } , \mathcal { A } _ { \hat { t } _ { i } } ) .\tag{13}
$$

The articles are decomposed into a set of event atoms $\mathcal { E } _ { A }$ as reference event atoms in equation 3. If the node summary contains hallucinated or fabricated content, it won’t be fully entailed by the reference articles (see Figure 2c).

Coherence. While coherence is crucial in document summarization tasks (Wu and Hu, 2018; Chang et al., 2024), directly applying it to timeline summarization is insufficient. Unlike standard summaries that emphasize narrative coherence, timeline summaries demand structural coherence, including linguistic and stylistic consistency.

<table><tr><td>Dataset</td><td>#Topics</td><td>#Topic types</td><td></td><td>#Articles #Sources #Granu</td><td></td></tr><tr><td>T17</td><td>9</td><td>1</td><td>4,650</td><td>2</td><td>1</td></tr><tr><td>Crisis</td><td>4</td><td>1</td><td>9,240</td><td>3</td><td>1</td></tr><tr><td>Entities</td><td>47</td><td>2</td><td>45,075</td><td>1</td><td>1</td></tr><tr><td>CrisisTL</td><td>1,000</td><td>1</td><td>10,610</td><td>1</td><td>1</td></tr><tr><td>TI  $\boldsymbol { \mathcal { S } } _ { 1 0 0 }$ </td><td>100</td><td>4</td><td>10,379</td><td>2</td><td>1</td></tr><tr><td>Ours</td><td>543</td><td>7</td><td>55,432</td><td>2,858</td><td>3</td></tr></table>

Table 1: Comparison with existing datasets.

Figure 2d shows common coherence issues. We introduce an evaluation process similar to the ACL Review Form<sup>5</sup>, assessing Structural, Linguistic, and Style Coherence, with details in Appendix B.

The process involves: (1) Paraphrasing content to improve understanding and reduce bias; (2) Rating each aspect from 1 to 3 and explaining the rationale for fine-grained evaluation; (3) Giving an overall score from 1 to 5 for a holistic assessment.

To reduce reviewers’ workload, we use GPT-4o $\mathrm { \ A P I ^ { 6 } }$ for automatic coherence assessment. Domain experts provide annotated examples to guide the model in understanding the criteria.

## 5 Dataset Construction

To ensure comprehensive evaluation across timelines at different granularity levels, dataset construction must meet two key premises: (1) The dataset should include news topics of varying complexity, types, and scales, with articles from diverse sources to simulate different granularity needs, enabling robust evaluation. (2) During annotations, annotators should minimize personal biases and annotate nodes at multiple granularity levels to facilitate the evaluation of both fine-grained and coarse-grained timelines. Our solutions to these challenges will be discussed in the following sections.

## 5.1 Data Collection

For data collection, we aim to assemble a diverse and representative set of news topics. We begin by leveraging Baidu’s event news websites, known for expert fine-grained timeline annotation, to obtain news topics and their corresponding reference timelines. We then manually filter these to ensure quality and diversity based on a standard.

![](images/d96d3f90bbaa98e4902e4693a4be200158efe29e64d240345efcf2ebafd8f58b.jpg)  
Figure 4: Dataset statistics.

The final dataset includes 543 news topics after October 2023, categorized into seven major types (Politics, Economy, Society, Science, Technology, Sports, and Entertainment), with reference timelines ranging from 9 to 200 nodes.

To gather reference articles, we use Baidu, Google, and Bing, employing multiple keywords to ensure each node is supported by at least 5 articles on average. The final dataset includes 55,432 articles, averaging 102 articles per topic. Table 1 compares our dataset with existing datasets.<sup>7</sup>

## 5.2 Consesus-based Annotation

DTELS requires multiple granularity levels, but annotating all is impractical. To aid evaluation, we define three levels: fine-grained $( G _ { N } )$ , mediumgrained $( G _ { 1 0 } )$ , and coarse-grained $( G _ { 5 } )$ , where N, 10, and 5 denote the number of nodes in the reference timelinee<sup>8</sup>. $G _ { N }$ reflects the original timeline with an unspecified node count, while medium and coarse timelines are annotated through consensus.

Even experienced journalists may differ in selecting events for coarse-grained timelines from fine-grained ones. To ensure uniformity, maximizing consensus among annotators on salient events and granularity is essential. However, this process can be costly and time-consuming in DTELS, especially with numerous articles per topic. We utilize GPT-4o to facilitate consensus through roleplaying (He et al., 2023; Tao et al., 2024). The consensus-based annotation process involves three stages: (1) Salient Events Decomposition: For medium-grained timelines $S ^ { \mathcal { G } _ { 1 0 } }$ , we decompose the fine-grained timeline $\mathcal { S } ^ { \mathcal { G } _ { N } }$ into event atoms and group them by timestamp. (2) Consensus-based Selection: For each news topic, we prompt GPT-4o in different roles to select the 10 most important event groups from the atom groups, based on consensus among three roles. (3) Expert Refinement: Domain experts refine the selected groups to ensure quality, summarizing them into a 10-node timeline. The fine-grained timelines are annotated similarly. We show details and agreement in Appendix C.2. The results with high inter-annotator agreement show the effectiveness of our annotation.

We list the statistics of the dataset in Figure 4. A more detailed description of dataset construction and annotation can be found in Appendix C.

## 6 Experiments

## 6.1 Experimental Settings

For extractive methods, We implement two stateof-the-art methods (Gholipour Ghalandari and Ifrim, 2020) as baselines:

Datewise: This method selects key dates in a regression-based manner and then applies centroidopt (Gholipour Ghalandari, 2017) to extract summaries for each date.

Clustering: This method clusters articles using TF-IDF vectors and then converts the clusters into a temporal graph. Dates are assigned to each cluster through a regression model. For DTELS, we constrain the number of nodes in the timeline according to the specified granularity level.

For generative methods, we select LLMs with Chinese ability. We propose two solutions:

Long-context Prompting (LP): For the longcontext model, we directly prompt the model by providing the news topic, the entire articles with timestamps, and the granularity instruction.

Hierarchical Merging (HM): For models with limited context length, they generate summaries for each date according to the input articles’s timestamps. Subsequently, these summaries are hierarchically merged following the merging prompts.

We also establish two distinct experimental settings to evaluate the task’s characteristics:

Gold Timestamps (GT): We instruct models with correct timestamps to guide content generation, ensuring the focus is on content quality rather than timestamp accuracy. This setting can be used for both LP $( \mathrm { L P _ { G T } } )$ and HM $( \mathrm { H M } _ { \mathrm { G T } } )$ .

<table><tr><td rowspan="2">Methods</td><td rowspan="2">Models</td><td colspan="4">Granularity  $\mathcal { G } _ { N }$ </td><td colspan="4">Granularity  $\mathcal { G } _ { 1 0 }$ </td><td colspan="4">Granularity G5</td></tr><tr><td>Info</td><td>GranuN</td><td>Fact</td><td>Coherence</td><td>Info</td><td>Granu10</td><td>Fact</td><td>Coherence</td><td>Info</td><td>Granu5</td><td>Fact</td><td>Coherence</td></tr><tr><td>Datewise</td><td></td><td>17.35</td><td>79.15</td><td>76.7</td><td>56.27</td><td>5.2</td><td>16.13</td><td>74.37</td><td>55.21</td><td>4.46</td><td>8.06</td><td>72.24</td><td>57.99</td></tr><tr><td>Cluster</td><td></td><td>4.14</td><td>72.01</td><td>69.6</td><td>55.33</td><td>2.65</td><td>16.90</td><td>66.27</td><td>52.56</td><td>2.32</td><td>10.73</td><td>64.79</td><td>54.10</td></tr><tr><td>TO</td><td>GPT-3.5-Turbo</td><td>1.45</td><td>60.21</td><td>41.19</td><td>91.20</td><td>0.83</td><td>20.54</td><td>41.22</td><td>94.76</td><td>0.65</td><td>11.58</td><td>38.99</td><td>97.46</td></tr><tr><td rowspan="3">LP</td><td>GPT-40</td><td>6.55</td><td>61.94</td><td>65.78</td><td>69.21</td><td>0.92</td><td>8.80</td><td>86.82</td><td>77.15</td><td>0.74</td><td>3.99</td><td>88.11</td><td>87.55</td></tr><tr><td>GLM-3-Turbo</td><td>1.51</td><td>56.16</td><td>45.04</td><td>60.20</td><td>4.45</td><td>20.64</td><td>71.84</td><td>62.99</td><td>4.69</td><td>11.51</td><td>70.58</td><td>70.95</td></tr><tr><td>Yi-medium</td><td>9.87</td><td>66.45</td><td>65.39</td><td>63.10</td><td>4.91</td><td>17.91</td><td>77.48</td><td>65.49</td><td>8.69</td><td>23.36</td><td>51.32</td><td>71.88</td></tr><tr><td>LPGT</td><td>GPT-40</td><td>1.91</td><td>59.69</td><td>48.24</td><td>55.89</td><td>2.17</td><td>26.74</td><td>46.56</td><td>56.28</td><td>1.76</td><td>14.78</td><td>47.94</td><td>56.30</td></tr><tr><td rowspan="7">HM</td><td>GPT-3.5-Turbo</td><td>24.24</td><td>81.72</td><td>91.95</td><td>65.87</td><td>0.82</td><td>7.96</td><td>91.96</td><td>68.38</td><td>0.72</td><td>4.74</td><td>91.96</td><td>76.92</td></tr><tr><td>GLM-3-Turbo</td><td>21.07</td><td>72.43</td><td>87.39</td><td>67.40</td><td>1.37</td><td>15.65</td><td>87.61</td><td>68.01</td><td>0.91</td><td>8.74</td><td>88.33</td><td>71.34</td></tr><tr><td>Yi-medium</td><td>17.46</td><td>75.32</td><td>82.26</td><td>64.28</td><td>2.36</td><td>14.34</td><td>86.02</td><td>65.56</td><td>1.75</td><td>6.91</td><td>85.60</td><td>73.41</td></tr><tr><td>Qwen1.5-110b</td><td>28.00</td><td>76.51</td><td>83.99</td><td>78.36</td><td>2.24</td><td>10.75</td><td>83.27</td><td>79.77</td><td>1.78</td><td>6.81</td><td>81.09</td><td>86.69</td></tr><tr><td>Qwen1.5-72b</td><td>24.69</td><td>80.25</td><td>85.14</td><td>74.82</td><td>0.92</td><td>10.31</td><td>85.4</td><td>80.86</td><td>0.74</td><td>5.48</td><td>84.56</td><td>85.57</td></tr><tr><td>Qwen1.5-32b</td><td>23.37</td><td>73.97</td><td>86.29</td><td>68.64</td><td>0.61</td><td>10.14</td><td>86.32</td><td>75.47</td><td>0.55</td><td>5.54</td><td>88.04</td><td>82.08</td></tr><tr><td>Qwen1.5-14b</td><td>25.26</td><td>67.78</td><td>85.76</td><td>69.69</td><td>0.71</td><td>13.06</td><td>86.31</td><td>69.98</td><td>0.56</td><td>6.98</td><td>85.58</td><td>78.64</td></tr><tr><td>HMGT</td><td>GPT-3.5-Turbo 36.82</td><td></td><td>78.59</td><td>94.63</td><td>64.20</td><td>1.21</td><td>9.41</td><td>93.59</td><td>70.00</td><td>1.02</td><td>6.07</td><td>93.48</td><td>68.60</td></tr></table>

Table 2: Main results of different methods on DTELS task. The best results for different methods are in bold. The best results across all methods are underlined.

Topic Only (TO): Only providing the news topics and granularity requirements to generate a fabricated timeline.

The full implementations of the methods and model are detailed in Appendix D.

## 6.2 Main Results

We conduct experiments with the proposed metrics. The main results are shown in Table 2. From the results, we can observe the following conclusions:

LLMs dominate in DTELS. LLM-based methods outperform state-of-the-art models across all metrics. The HM excels at $\mathcal { G } _ { N }$ in Info and Granu. while LP performs robustly at coarse and medium granularities, indicating the hierarchical method’s strength in capturing details and LP’s capability in managing timelines with long-context windows.

Context window mattersfor long-contextprompting. With 200k context windows, Yi-medium-200k outperforms models with 128k windows, particularly at coarse granularities, demonstrating the effectiveness in broad event overviews.

Model capacity influencesfine-grained metrics. Results from Qwen at different scales show that as model size decreases, performance in informativeness and granular consistency declines, suggesting that larger models are better at capturing and conveying detailed information.

Observationfrom the variants. With gold timestamps in $\mathbf { \tilde { \mu } } _ { \mathrm { H M } } \mathbf { \vec { \mu } } _ { \mathrm { \Sigma } }$ , the factuality is enhanced with temporal guidance. However, timestamps provide minimal benefits to LP and may reduce factuality and coherence. The “Topic Only” approach achieves the highest coherence scores but significantly lags in factuality, indicating that it maintains narrative continuity at the cost of factual accuracy.

![](images/55b400da6b7412a85e017bd5d73bb8e6006d154c7d82e370da684ac562dcf956.jpg)  
Figure 5: Extended evaluation on granularity levels.

## 7 Analysis

To further assess performance across different granularities, we conduct an extended evaluation using hierarchical merging with GPT-3.5-Turbo.

## 7.1 Extended Evaluation on Granularities

More Detailed Granularities. We define five distinct granularity levels: $\mathcal { G } _ { 4 0 } , \mathcal { G } _ { 2 0 } , \mathcal { G } _ { 1 0 } , \mathcal { G } _ { 5 }$ , and $\mathcal { G } _ { 3 }$ We collect a subset of the dataset with each topic containing over 50 nodes at $\mathcal { G } _ { N }$ . Reference timelines are annotated at these levels, and performance is assessed using Hierarchical Merging with GPT-3.5-Turbo. Results, shown in Figure 5, reveal that while coarser summaries generally offer better informativeness, factuality, and coherence, they may struggle with granular consistency, highlighting a trade-off between detail and summary quality.

Natural Language Granularity Instructions We evaluate natural language granularity instructions, defining fine- and coarse-grained instructions (see Table 3). Reference timelines include the both fine- $( S ^ { \mathcal { G } _ { N } } )$ and coarse-grained $( S ^ { G _ { 5 } } )$ ). Results in Table 3 show that the one-shot method performs competitively with the #Node method, indicating that models learn to generate accurate timelines with natural language granularity instructions.

<table><tr><td rowspan="2">Granularity</td><td>Granular Instruction</td><td>Info</td><td>Granu</td><td>Fact</td><td>Coherence</td></tr><tr><td>Prompt</td><td>10.10</td><td>75.61</td><td>91.67</td><td>68.42</td></tr><tr><td rowspan="2"> $\mathcal { G } _ { N }$ </td><td>One-shot</td><td>22.65</td><td>81.10</td><td>93.88</td><td>70.86</td></tr><tr><td>#Node</td><td>24.24</td><td>81.72</td><td>91.95</td><td>65.87</td></tr><tr><td rowspan="3"> $\mathcal { G } _ { 5 }$ </td><td>Prompt</td><td>0.74</td><td>7.26</td><td>91.44</td><td>69.2</td></tr><tr><td>One-shot</td><td>0.75</td><td>7.41</td><td>92.37</td><td>71.79</td></tr><tr><td>#Node</td><td>0.72</td><td>4.74</td><td>91.96</td><td>76.92</td></tr></table>

Table 3: Results of natural language granularity instructions, where #Node represents the HM method with the number of nodes as the granularity instruction.

## 7.2 Influential Factors on Metrics

We analyze the influence of topics and the article numbers. We conclude that the two aspects greatly influence the performance. Results in Appendix E suggest improvement in stability is necessary.

## 7.3 Metrics Alignment with Human

Agreement Score. Annotators are asked to rate timelines on a scale from 1 to 5 based on the metrics. Pearson correlations are: informativeness (78.74%), granular consistency (76.66%), factuality (95.87%), and coherence (99.14%).

Consistency Score. Given pairs of timelines for a topic generated by two models, annotators rate the better one for each metric. We calculate the consistent score between annotators and metrics. Each metric’s consistency exceeds 90%, showing high consistency between humans and the metrics.

We also conduct empirical comparisons on correlation alignment with existing metrics (e.g., Alignment ROUGE-L (Martschat and Markert, 2017), BERTScore (Zhang et al., 2020), and QAEval (Deutsch et al., 2021)). Details can be found in the Appendix F. (the annotation process is to let evaluators independently rate the timelines on a scale from 1 to 5 for the four aspects. Then after a group discussion, they revise their scores. Details are described in Appendix F):

## 7.4 Robustness of the Automatic Coherence Scoring

Since coherence can be subjective due to model biases. We conduct a robustness test by comparing the automatic coherence scores produced by different models: we use three evaluators: Qwen1.5 110B (Team, 2023), GPT-3.5-turbo, and GPT-4o-mini to score the timelines generated by the datewise method. We then assess the mean absolute error (MAE): $M A E ( m _ { 1 } , m _ { 2 } ) =$ $\begin{array} { r } { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } | s c o r e _ { m _ { 1 } } - s c o r e _ { m _ { 2 } } | } \end{array}$ in Table 4. The results show that the automatic coherence scores are robust across different models, indicating the reliability of the automatic scoring.

<table><tr><td> $m _ { 1 }$ </td><td> $m _ { 2 }$ </td><td> $m a e ( m _ { 1 } , m _ { 2 } )$ </td></tr><tr><td>Qwen1.5</td><td>gpt-3.5-turbo</td><td>0.2596</td></tr><tr><td>Qwen1.5</td><td>gpt-4o-mini</td><td>0.2615</td></tr><tr><td>gpt-3.5-turbo</td><td>gpt-4o-mini</td><td>0.1114</td></tr></table>

Table 4: Robustness of automatic coherence scoring across different models.

## 8 Conclusions

In this paper, we introduce a Dynamic-granularity TimELine Summarization (DTELS) task, which aims to construct timeline summaries at dynamic granularity levels following the granularity requirements. We build a comprehensive benchmark including: (1) Evaluation Framework: We propose an event-centric evaluation along with metrics: informativeness, granular consistency, factuality, and coherence. Evaluation of alignment with the human annotator proves the rationality of the proposed metrics. (2) Dataset Construction: We construct a large-scale Chinese dataset for DTELS with consensus-based annotation for multi-granularity references. We apply expert refinement to ensure the authority of the annotation. (3) Comprehensive Evaluation: We present two solutions for large language models. Through experiments on existing state-of-the-art timeline summarization methods as well as LLM-based solutions on multiple models, we find that the DTELS task remains challenging. Further research is required to improve the informativeness granularity consistency. In the future, we plan to diversify the language sources and improve LLM-based methods to better capture information and enhance granular consistency.

## Limitations

Though our DTELS approach has shown promising results, there are several limitations that need to be addressed in future work: Our approach relies heavily on the availability of a large-scale, annotated dataset. The creation of such datasets is timeconsuming, which may limit the scalability and applicability of our approach to other domains or languages where such resources are not available. To evaluate the generated timelines, we rely on large language models’ APIs, which are costly and may not be accessible to all researchers. Besides, The language of our dataset is Chinese, which may limit the generalizability of our approach to other languages. Further research is needed to develop more efficient data collection and evaluation methods that can be applied to a wider range of languages and domains.

## Acknowledgements

This work is supported by the National Natural Science Foundation of China ( No.U24A20335, No. 62176257, No, 62406321). This work is also supported by the Youth Innovation Promotion Association CAS and the China Postdoctoral Science Foundation under Grant Number 2024M753500.

## References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Rex Ambe. 2023. Classification and quantification of timestamp data quality issues and its impact on data quality outcome. Data Intelligence, pages 1–39.

Oana-Maria Camburu, Tim Rocktäschel, Thomas Lukasiewicz, and Phil Blunsom. 2018. e-snli: Natural language inference with natural language explanations. Advances in Neural Information Processing Systems, 31.

Pengfei Cao, Yupu Hao, Yubo Chen, Kang Liu, Jiexin Xu, Huaijun Li, Xiaojian Jiang, and Jun Zhao. 2023. Event ontology completion with hierarchical structure evolution networks. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 306–320, Singapore. Association for Computational Linguistics.

Yapei Chang, Kyle Lo, Tanya Goyal, and Mohit Iyyer. 2024. Booookscore: A systematic exploration of book-length summarization in the era of LLMs. In The Twelfth International Conference on Learning Representations.

Jianhao Chen, Haoyuan Ouyang, Junyang Ren, Wentao Ding, Wei Hu, and Yuzhong Qu. 2024a. Timelinebased sentence decomposition with in context learning for temporal fact extraction. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3415–3432, Bangkok, Thailand. Association for Computational Linguistics.

Liang Chen, Yang Deng, Yatao Bian, Zeyu Qin, Bingzhe Wu, Tat-Seng Chua, and Kam-Fai Wong. 2023a. Beyond factuality: A comprehensive evaluation of large

language models as knowledge generators. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 6325– 6341, Singapore. Association for Computational Linguistics.

Tianyu Chen, Yiming Zhang, Guoxin Yu, Dapeng Zhang, Li Zeng, Qing He, and Xiang Ao. 2024b. EFSA: Towards event-level financial sentiment analysis. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7455–7467, Bangkok, Thailand. Association for Computational Linguistics.

Xiuying Chen, Mingzhe Li, Shen Gao, Zhangming Chan, Dongyan Zhao, Xin Gao, Xiangliang Zhang, and Rui Yan. 2023b. Follow the timeline! generating an abstractive and extractive timeline summary in chronological order. ACM Transactions on Information Systems, 41(1):1–30.

Daniel Deutsch, Tania Bedrax-Weiss, and Dan Roth. 2021. Towards question-answering as an automatic metric for evaluating the content quality of a summary. Preprint, arXiv:2010.00490.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.

Zorik Gekhman, Jonathan Herzig, Roee Aharoni, Chen Elkind, and Idan Szpektor. 2023. TrueTeacher: Learning factual consistency evaluation with large language models. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 2053–2070, Singapore. Association for Computational Linguistics.

Demian Gholipour Ghalandari. 2017. Revisiting the centroid-based method: A strong baseline for multidocument summarization. In Proceedings of the Workshop on New Frontiers in Summarization, pages 85–90, Copenhagen, Denmark. Association for Computational Linguistics.

Demian Gholipour Ghalandari and Georgiana Ifrim. 2020. Examining the state-of-the-art in news timeline summarization. In Proceedings ofthe 58th Annual Meeting of the Association for Computational Linguistics, pages 1322–1334, Online. Association for Computational Linguistics.

Tanya Goyal, Junyi Jessy Li, and Greg Durrett. 2022. SNaC: Coherence error detection for narrative summarization. In Proceedings ofthe 2022 Conference on Empirical Methods in Natural Language Processing, pages 444–463, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Zhitao He, Pengfei Cao, Yubo Chen, Kang Liu, Ruopeng Li, Mengshu Sun, and Jun Zhao. 2023. LEGO: A multi-agent collaborative framework with role-playing and iterative feedback for causality explanation generation. In Findings of the Association for Computational Linguistics: EMNLP 2023,

pages 9142–9163, Singapore. Association for Computational Linguistics.

Zhitao He, Pengfei Cao, Zhuoran Jin, Yubo Chen, Kang Liu, Zhiqiang Zhang, Mengshu Sun, and Jun Zhao. 2024. Zero-shot cross-lingual document-level event causality identification with heterogeneous graph contrastive transfer learning. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 17833–17850, Torino, Italia. ELRA and ICCL.

Danyang Hu, Meng Wang, Feng Gao, Fangfang Xu, and Jinguang Gu. 2022. Knowledge representation and reasoning for complex time expression in clinical text. Data Intelligence, 4(3):573–598.

Qisheng Hu, Geonsik Moon, and Hwee Tou Ng. 2024. From moments to milestones: Incremental timeline summarization leveraging large language models. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 7232–7246, Bangkok, Thailand. Association for Computational Linguistics.

Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. 2023. Survey of hallucination in natural language generation. ACM Computing Surveys, 55(12):1–38.

Liqiang Jing, Yiren Li, Junhao Xu, Yongcan Yu, Pei Shen, and Xuemeng Song. 2023. Vision enhanced generative pre-trained language model for multimodal sentence summarization. Machine Intelligence Research, 20(2):289–298.

Rémy Kessler, Xavier Tannier, Caroline Hagège, Véronique Moriceau, and André Bittar. 2012. Finding salient dates for building thematic timelines. In Proceedings of the 50th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 730–739, Jeju Island, Korea. Association for Computational Linguistics.

Matej Klemen, Aleš Žagar, Jaka Cibej, and Marko<sup>ˇ</sup> Robnik-Šikonja. 2024. SI-NLI: A Slovene natural language inference dataset and its evaluation. In Proceedings ofthe 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 14859– 14870, Torino, Italia. ELRA and ICCL.

Harold W. Kuhn. 1955. The Hungarian Method for the Assignment Problem. Naval Research Logistics Quarterly, 2(1–2):83–97.

Risto Kunelius. 2006. Good journalism: On the evaluation criteria of some interested and experienced actors. Journalism studies, 7(5):671–690.

Moreno La Quatra, Luca Cagliero, Elena Baralis, Alberto Messina, and Maurizio Montagnuolo. 2021. Summarize dates first: A paradigm shift in timeline

summarization. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’21, page 418–427, New York, NY, USA. Association for Computing Machinery.

Junyi Li, Xiaoxue Cheng, Wayne Xin Zhao, Jian-Yun Nie, and Ji-Rong Wen. 2023. Halueval: A largescale hallucination evaluation benchmark for large language models. arXiv preprint arXiv:2305.11747.

Manling Li, Tengfei Ma, Mo Yu, Lingfei Wu, Tian Gao, Heng Ji, and Kathleen McKeown. 2021. Timeline summarization based on event graph compression via time-aware optimal transport. In Proceedings ofthe 2021 Conference on Empirical Methods in Natural Language Processing, pages 6443–6456, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Chin-Yew Lin. 2004. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pages 74–81.

Sebastian Martschat and Katja Markert. 2017. Improving ROUGE for timeline summarization. In Proceedings ofthe 15th Conference ofthe European Chapter of the Association for Computational Linguistics: Volume 2, Short Papers, pages 285–290, Valencia, Spain. Association for Computational Linguistics.

Sebastian Martschat and Katja Markert. 2018. A temporally sensitive submodularity framework for timeline summarization. In Proceedings ofthe 22nd Conference on Computational Natural Language Learning, pages 230–240, Brussels, Belgium. Association for Computational Linguistics.

Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Koh, Mohit Iyyer, Luke Zettlemoyer, and Hannaneh Hajishirzi. 2023. FActScore: Fine-grained atomic evaluation of factual precision in long form text generation. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 12076–12100, Singapore. Association for Computational Linguistics.

Jun-Ping Ng and Viktoria Abrecht. 2015. Better summarization evaluation with word embeddings for ROUGE. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, pages 1925–1930, Lisbon, Portugal. Association for Computational Linguistics.

Kiem-Hieu Nguyen, Xavier Tannier, and Veronique Moriceau. 2014. Ranking multidocument event descriptions for building thematic timelines. In Proceedings of COLING 2014, the 25th International Conference on Computational Linguistics: Technical Papers, pages 1208–1217, Dublin, Ireland. Dublin City University and Association for Computational Linguistics.

Hossein Rajaby Faghihi, Bashar Alhafni, Ke Zhang, Shihao Ran, Joel Tetreault, and Alejandro Jaimes. 2022. CrisisLTLSum: A benchmark for local crisis

event timeline extraction and summarization. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2022, pages 5455–5477, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Vinay Setty. 2024. Factcheck editor: Multilingual text editor with end-to-end fact-checking. arXiv preprint arXiv:2404.19482.

Zejiang Shen, Kyle Lo, Lauren Yu, Nathan Dahlberg, Margo Schlanger, and Doug Downey. 2022. Multilexsum: Real-world summaries of civil rights lawsuits at multiple granularities. Advances in Neural Information Processing Systems, 35:13158–13173.

Jiayu Song, Jenny Chim, Adam Tsakalidis, Julia Ive, Dana Atzil-Slonim, and Maria Liakata. 2024. Combining hierachical VAEs with LLMs for clinically meaningful timeline summarisation in social media. In Findings of the Association for Computational Linguistics ACL 2024, pages 14651–14672, Bangkok, Thailand and virtual meeting. Association for Computational Linguistics.

Julius Steen and Katja Markert. 2022. How to find strong summary coherence measures? a toolbox and a comparative study for summary coherence measure evaluation. In Proceedings of the 29th International Conference on Computational Linguistics, pages 6035–6049, Gyeongju, Republic of Korea. International Committee on Computational Linguistics.

Tianxiang Sun, Xiaotian Zhang, Zhengfu He, Peng Li, Qinyuan Cheng, Xiangyang Liu, Hang Yan, Yunfan Shao, Qiong Tang, Shiduo Zhang, Xingjian Zhao, Ke Chen, Yining Zheng, Zhejian Zhou, Ruixiao Li, Jun Zhan, Yunhua Zhou, Linyang Li, Xiaogui Yang, Lingling Wu, Zhangyue Yin, Xuanjing Huang, Yu-Gang Jiang, and Xipeng Qiu. 2024. Moss: An open conversational large language model. Machine Intelligence Research, 21(5):888–905.

Russell Swan and James Allan. 2000. Automatic generation of overview timelines. In Proceedings of the 23rd annual international ACM SIGIR conference on Research and development in information retrieval, pages 49–56.

Yufei Tao, Ameeta Agrawal, Judit Dombi, Tetyana Sydorenko, and Jung In Lee. 2024. ChatGPT roleplay dataset: Analysis of user motives and model naturalness. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 3133–3145, Torino, Italia. ELRA and ICCL.

Qwen Team. 2023. Qwen technical report. arXiv preprint arXiv:2309.16609.

Giang Binh Tran, Tuan A Tran, Nam-Khanh Tran, Mohammad Alrifai, and Nattiya Kanhabua. 2013. Leveraging learning to rank in an optimization framework for timeline summarization. In SIGIR 2013 Workshop on Time-aware Information Access (TAIA.

Junjie Wang, Yuxiang Zhang, Lin Zhang, Ping Yang, Xinyu Gao, Ziwei Wu, Xiaoqun Dong, Junqing He, Jianheng Zhuo, Qi Yang, Yongfeng Huang, Xiayu Li, Yanghan Wu, Junyu Lu, Xinyu Zhu, Weifeng Chen, Ting Han, Kunhao Pan, Rui Wang, Hao Wang, Xiaojun Wu, Zhongshen Zeng, Chongpei Chen, Ruyi Gan, and Jiaxing Zhang. 2022. Fengshenbang 1.0: Being the foundation of chinese cognitive intelligence. CoRR, abs/2209.02970.

Lu Wang, Claire Cardie, and Galen Marchetti. 2015. Socially-informed timeline generation for complex events. In Proceedings of the 2015 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 1055–1065, Denver, Colorado. Association for Computational Linguistics.

William Yang Wang, Yashar Mehdad, Dragomir R. Radev, and Amanda Stent. 2016. A low-rank approximation approach to learning joint embeddings of news stories and images for timeline summarization. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 58–68, San Diego, California. Association for Computational Linguistics.

Yuxiang Wu and Baotian Hu. 2018. Learning to extract coherent summary via deep reinforcement learning. In Proceedings ofthe AAAI conference on artificial intelligence, volume 32.

Jundong Xu, Hao Fei, Liangming Pan, Qian Liu, Mong-Li Lee, and Wynne Hsu. 2024. Faithful logical reasoning via symbolic chain-of-thought. arXiv preprint arXiv:2405.18357.

Alex Young, Bei Chen, Chao Li, Chengen Huang, Ge Zhang, Guanwei Zhang, Heng Li, Jiangcheng Zhu, Jianqun Chen, Jing Chang, et al. 2024. Yi: Open foundation models by 01. ai. arXiv preprint arXiv:2403.04652.

Aohan Zeng, Xiao Liu, Zhengxiao Du, Zihan Wang, Hanyu Lai, Ming Ding, Zhuoyi Yang, Yifan Xu, Wendi Zheng, Xiao Xia, et al. 2022. Glm-130b: An open bilingual pre-trained model. arXiv preprint arXiv:2210.02414.

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. 2020. Bertscore: Evaluating text generation with bert. Preprint, arXiv:1904.09675.

Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu, Tingchen Fu, Xinting Huang, Enbo Zhao, Yu Zhang, Yulong Chen, et al. 2023. Siren’s song in the ai ocean: a survey on hallucination in large language models. arXiv preprint arXiv:2309.01219.

Zhihan Zhang, Yixin Cao, Chenchen Ye, Yunshan Ma, Lizi Liao, and Tat-Seng Chua. 2024. Analyzing temporal complex events with large language models? a benchmark towards temporal, long context understanding. Preprint, arXiv:2406.02472.

Ming Zhong, Yang Liu, Suyu Ge, Yuning Mao, Yizhu Jiao, Xingxing Zhang, Yichong Xu, Chenguang Zhu, Michael Zeng, and Jiawei Han. 2022. Unsupervised multi-granularity summarization. In Findings of the Associationfor Computational Linguistics: EMNLP 2022, pages 4980–4995, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

## A Event Atoms Decomposition

Evaluation metrics based on event atoms require decomposition on both the reference and generated timelines. The event atoms in reference timelines are annotated in advance by human annotators. For the generated timelines, the decomposition process should be completed in real-time. Similar to the decomposition for atomic facts (Min et al., 2023), we adopt GPT-3.5 to complete the automatic annotation.

## A.1 Manual Decomposition Protocol

To maintain consistency and accuracy in event atom decomposition, human annotators should follow this protocol:

(1) Understanding the Context: Read the entire node summary carefully to understand the overarching event or narrative described and then identify the primary subject(s) and action(s) within the sentence.

(2) Segmentation of Events: Down each sentence into smaller units by identifying distinct actions or states that involve a subject and an object. Then, consider each clause within a complex sentence as a potential event atom if it represents a unique action or state. For instance, for the sentence “John arrived at the station and met his friend.”, two event atoms can be identified:

• Event Atom 1: “John arrived at the station.”

• Event Atom 2: “John met his friend.”

## A.2 Automatic Decomposition

The automatic decomposition process using GPT-3.5 is implemented by prompts in Table 5.

## B Coherence Review Form

In this section, we introduce the details of the Coherence Review Form and the sub-metrics.

Coherence is assessed through a review process similar to the ACL Review Form. As illustrated in Figure 6, we decompose the review form into three steps. We ask experts to provide a comprehensive categorization of the main coherence errors that occur in timeline summarization. Based on these errors, we propose three sub-metrics along with their evaluation criteria for the scores (detailed in Figure 7). We provide annotators with the task definition, detailed descriptions of the sub-metrics, and examples, both positive and negative, annotated by domain experts. For automatic evaluation, we apply GPT-4o API and apply the Task Descriptions as system prompts.

## C Details of Dataset Construction

In this section, we provide a detailed description of the dataset construction.

## C.1 Data Collection.

The data collection encompasses news topics of varying complexity, types, and scales. The original dataset contains 1,012 news topics. We apply a filtering standard for the news topics detailed as follows:

Diversity of Sources. To ensure a broad representation of perspectives and avoid bias in data collection, we include timelines and articles sourced from a diverse set of reputable news sources. For the news topics: the dataset encompasses diverse news topics from various geographical regions across domestic and international events (a total of 32 countries/regions are included). For the timeline, the timeline nodes should come from multiple sources. Only timelines with events sourced from at least three news sources are included. For the news articles, we select articles from 2,858 sources including global press, forums, and social media,

Timeliness of News Topics.The pretraining data for the models used in our evaluation are all prior to October 2023. To minimize the risk of using contaminated data from LLMs pretraining corpus, we exclude older or stale news topics that no longer reflect current events or have lost relevance over time. We select news topics after Oct. 2023 for dataset creation.

Event-centric and Complete News Topics. To ensure that each news topic revolves around a specific, well-defined event or series of related events. We retain topics that provide comprehensive coverage of the development and conclusion of events, capturing key milestones and outcomes. We let annotators evaluate each selected topic to confirm it narrates a coherent storyline from beginning to end, avoiding fragmented or ambiguous narratives, and verify that the reference timelines associated with each topic adequately depict the chronology and significance of events.

<table><tr><td>Atoms Decomposition Prompts</td></tr><tr><td>System Prompt You are a Fact Decomposer. ## Your task is:</td></tr><tr><td>As a specialized journalist, you will be provided with a sentence that may describe multiple events. Your task is to decompose the sentence into atomic propositions. An atomic proposition consists of, and only of, a subject, a predicate, and an object.</td></tr><tr><td>## Output format:</td></tr><tr><td>Please use the following format for your output:</td></tr><tr><td>[“Atom_1”, “Atom_2”, ... ] ## Example:</td></tr><tr><td>Here is an example for you to better understand the task: Input: “Myanmar military: one-year state of emergency imposed&quot;</td></tr></table>

Table 5: Prompts used for Atoms Decomposition Process. We show examples for the model to better comprehend the task.

By applying the standards, our dataset filter to 543 high-quality, multiple-sources news topics. Then, news articles for each topic are collected with the following steps: (1) Retrieve on search engines (Baidu, Google, and Bing) for articles with news topics as keywords and time limits to the beginning and end of the corresponding timeline to get the most relevant 5 articles. (2) For each reference node summaries annotated in Baidu event websites, we directly get one source article from the website. Then we apply the summary as keywords on previously mentioned search engines to get 4 articles. (3) Filter low-quality articles. By thresholding the article titles against the news topics, we filter out low-quality articles. Articles with a BERT embedding similarity score of less than 0.3 between the title and the news topic are filtered out.

We list the statistics of the dataset in Figure 4.

## C.2 Consensus-based Annotation

To facilitate consensus, we prompt GPT-4o to play different roles as annotators, including news editor, journalist, and NLP researcher. These annotators focus on different aspects. The prompts are listed in Table 6.

GPT-4o annotators are instructed with the decomposed “event atoms” from the fine-grained timeline. These event atoms are grouped based on their timestamps, which could later be used to construct medium-grained and coarse-grained timelines. To determine the consensus among GPT-4o agents, we employed the following approach: (1) Each of the three GPT-4o agents independently selected their top 10 groups. (2) Groups selected by all three agents are automatically included in the final selection. (3) Groups selected by two agents are reviewed for inclusion based on their relevance and importance. After the independent selections are made, the selected event atom groups from all three annotators are compared. The primary focus here is to identify the level of consensus among the annotators. We list the agreement degree of the annotated nodes as shown in Table 7. Once GPT-4 has selected the initial set of events, domain experts review and refine these selections to ensure accuracy and completeness. The refinement process includes: (1) Fact-Checking: Ensuring that each selected event was factually accurate and well-supported by credible sources. (2) Composing Atoms: Composing Atomic Facts into node summaries. (3) Coherence Refinement: Refine the summary in total to ensure that the timeline as a whole presents a coherent narrative. (4) Detail Adjustment: Adding or removing details as necessary to meet the target granularity.

## D Methods Implementation

The extractive baselines are implemented based on the original code provided by the authors. For LLMs, we use the official API. For Qwen (Team,

![](images/986f4a8a3590d8d2766f0b4863ad7872004f23a31cd6a46e7f7367795da8636a.jpg)  
Table 6: Prompts for event consensus-based annotation, where N denotes the number of reference timeline nodes, in our case 10 and 5 for medium- and coarse-grained annotations.

<table><tr><td>Agreement Type</td><td>Count</td><td>Percentage</td></tr><tr><td>Full Agreement</td><td>3118</td><td>45.09%</td></tr><tr><td>Partial (1, 2)</td><td>2316</td><td>33.49%</td></tr><tr><td>Partial (1, 3)</td><td>573</td><td>8.28%</td></tr><tr><td>Partial (2, 3)</td><td>380</td><td>5.50%</td></tr><tr><td>No Agreement</td><td>525</td><td>7.59%</td></tr></table>

Table 7: Agreement among annotators, where 1, 2, and 3 correspond to the news editor, journalist, and NLP researcher, respectively. The agreement can be categorized as: (1) Full Agreement: The annotators selected the same event atom group. (2) Partial Agreement: Two out of three annotators selected the same event atom group. (3) No Agreement: No common event atom groups were selected by the annotators.

2023), we build upon their open-sourced weights<sup>9</sup>. The NLI model is implemented by BERT-based models (Devlin et al., 2018) fine-tuned on Chinese NLI datasets (Wang et al., 2022). The evaluation metrics include informativeness (Info), granular consistency (Granu), factuality (Fact), and coherence.

## D.1 Details of LLM-based Methods

We choose LLMs with advanced Chinese capability, including both open-source and closed-source models for our analysis. For closed-source LLMs, we select the most representative GPT series and widely known Chinese model Yi-medium. For open-source LLMs, we select GLM-3 and Qwen series with multiple model sizes. Particularly, for LP, we evaluate models including GPT-4o (128k) (Achiam et al., 2023), Yi-medium (200k) (Young et al., 2024) and GLM-3-Turbo (128k) (Zeng et al., 2022). For hierarchical merging, we evaluate GPT-3.5-Turbo<sup>10</sup>, GLM-3-Turbo (Zeng et al., 2022), Yimedium (Young et al., 2024), and Qwen1.5 (Team, 2023). The temperature is set to 0 for greedy sampling.

## D.2 Long-congtext Prompting

The prompts used for long-context prompting are illustrated in Table 8. To handle topics with hundreds of articles that may exceed the maximum token length, we truncate the last paragraph of each article recursively until the total content falls within the token limit. Figure 8 shows the distribution of token consumption for long-context prompting.

## D.3 Hierarchical Merging

The hierarchical merging method first generates a day summary for the news topic based on prompts in Table 9. Then, all nodes are hierarchically merged to form a complete timeline. Similarly, if the input exceeds the token length, we do the same operation as in long-context prompting.

## D.4 Natural Language Granular Instruction

We list the natural language granularity instruction in Table 10.

## E Influential Factors on Metrics

We analyze the influence of topic types on the performance of hierarchical merging with GPT-3.5- Turbo. The results are shown in Figure 9. We observe that the performance on different topic types varies significantly. “Military” topics consistently achieve the highest scores in all metrics, suggesting that the model handles structured and well-defined content more effectively. Conversely, “Politics” and “Technology” topics present the greatest challenges, particularly in informativeness and coherence, likely due to the complexity and variability of information required in these domains. This suggests that the model’s performance is closely tied to the nature of the topic.

We also assess the influence of the number of news articles. The results are shown in Figure 10. We find that the model performs better with fewer news articles, as the model can better capture the key information and generate more coherent summaries. However, the factuality of the summaries decreases with fewer news articles, as the model may lack sufficient information to generate faithful summaries.

## F Details of Alignment Evaluation

To measure how well human evaluators’ assessments of timelines align with the proposed metrics.

## F.1 Evaluation Process

We choose three evaluators with a background in journalism and experience in summarization or timeline construction. Then, we prepare a set of 50 timelines generated by our DTELS system. Include a mix of high and low scores across different dimensions. We have each evaluator independently assess the timelines using the scoring sheets in Figure 11. After the initial round, we facilitate a group discussion where evaluators can compare their scores and discuss discrepancies. This can help in understanding different perspectives and potentially refining the evaluation criteria. Then, we allow evaluators to revise their scores based on insights gained from the discussion.

![](images/c07fe542e379747660bfe9d7038e5031f51dddebda5cce0b4c3931169ef7eb73.jpg)  
Table 8: Prompts used in Long-context Prompting (LP) for long-context large language models. Here, N denotes the required node amounts for the timeline.

## F.2 Correlation Evaluation

we calculate the correlation coefficient between existing metrics and the human assessment results. The results in Table 11 indicate that existing candidates-references alignment-based metrics (Alignment ROUGE-L (Martschat and Markert, 2017) and BERTScore (Zhang et al., 2020)) focus more on the amount of effective information transferred from the reference. QAEval (Deutsch et al., 2021), which is similar to recalling factual information from references, achieves a competitive correlation with factuality. However, when it comes to other aspects like granularity, factuality, and coherence, existing metrics failed to align with all aspects. In contrast, we can observe that our proposed metrics align closely with the specific need of the DTELS task, demonstrating the effectiveness of our metrics.

## F.3 Metrics Definitions

## F.3.1 Informativeness

Informativeness measures how much useful information is provided by each node in the timeline. A high score indicates that the node adds significant and relevant detail to the overall understanding of the event.

## F.3.2 Granular Consistency

Granular Consistency assesses how well the timeline maintains a coherent level of detail across different nodes. A high score reflects that the granularity of events is consistent and appropriate throughout the timeline.

## F.3.3 Factuality

Factuality evaluates the accuracy and truthfulness of the information presented in each node. A high score indicates that the node contains verified and accurate facts.

## F.3.4 Coherence

Coherence measures how logically and smoothly the nodes are connected to form a comprehensible narrative. A high score suggests that the timeline is well-organized and the events are presented in a logical order.

```markdown
Day Summary Prompts
System Prompt
You are a News Event Timeline Generator.
## Your task is:
As a specialized journalist, you will be provided with a news [topic] and related news [articles]. Based on this information,
construct a chronologically ordered timeline summarizing the key events of the [topic]. Each event summary should be
accompanied by an accurate timestamp.
## Output format:
Please use the following format for your output:
1. yyyy-mm-dd: Event summary 1
2. yyyy-mm-dd: Event summary 2
{N}. yyyy-mm-dd: Event summary {N}
## Note:
- There can only be ONE event summary per day.
- It’s important to select key events to build the timeline, as not all [articles] are worth summarizing.
Input
[Topic]
{Topic of the Timeline}
[Article 0]
Title: {Title of Article 0}
Release-time: {Release-time of Article 0}
Content: {Content of Article 0}
Timeline Merging Prompts
System Prompt
You are a News Event Timeline Generator.
## Your task is:
As a specialized journalist, you will be provided with a news [topic], multiple partially completed timelines. Based on
this information, merge the timelines to create a chronologically ordered timeline summarizing the key events of the [topic].
## Output format:
Please use the following format for your output:
1. yyyy-mm-dd: Event summary 1
2. yyyy-mm-dd: Event summary 2
N. yyyy-mm-dd: Event summary N
## Note:
- There can only be ONE event summary per day.
- It’s important to select key events to build the timeline, as not all events are worth summarizing.
Input
[Timeline 0]
Timeline 0
```  
Table 9: Prompts used in Hierarchical Merging (HM) for context length-limited large language models, where N denotes the required node amounts for the timeline.

<table><tr><td rowspan=1 colspan=1>Type</td><td rowspan=1 colspan=1>|#Node|</td><td rowspan=1 colspan=1> $\mathcal { G } _ { o }$ </td></tr><tr><td rowspan=2 colspan=1>Prompt</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>Please generate a coarse-grained timeline.[Task Prompt*]</td></tr><tr><td rowspan=1 colspan=1>N</td><td rowspan=1 colspan=1>|Please generate a fine-grained timeline. [Task Prompt*]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>Please generate a timeline like:{Timeline with 5 nodes}</td></tr><tr><td rowspan=1 colspan=1>One-shot</td><td rowspan=1 colspan=1>N</td><td rowspan=1 colspan=1>Please generate a timeline like:{Timeline with {N} nodes}</td></tr></table>

Table 10: Natural language granularity instructions used in the experiments. [Task Prompt\*] denotes the prompts used for timeline summarization (LP and HM prompts in our paper). The “\*” indicates that the node amounts {N} in the [Task Prompt] is replaced with “N” to represent an arbitrary number.

![](images/3782e189c0720e59a3d0a820bce2c49af1265bd2ffbc98e7aa64cee7c574b435.jpg)  
Figure 6: Example of the Coherence Review Form.

![](images/1925663e959810b8e1a9790ade9f9e690918cb22fb4ab7278ccf9bfbaf0e5ac0.jpg)  
Figure 7: Sub-metrics and overall assessment definition with their corresponding score criteria.

<table><tr><td>Metrics</td><td>Info</td><td>Granu</td><td>Fact</td><td>Coherence</td></tr><tr><td>Informativeness</td><td>0.7874</td><td>0.6128</td><td>0.6823</td><td>0.7004</td></tr><tr><td>Granular Consistency</td><td>0.6128</td><td>0.7666</td><td>0.6389</td><td>0.6795</td></tr><tr><td>Factuality</td><td>0.6823</td><td>0.6389</td><td>0.9587</td><td>0.7512</td></tr><tr><td>Coherence</td><td>0.7004</td><td>0.6795</td><td>0.7512</td><td>0.9914</td></tr><tr><td>AR-1</td><td>0.6213</td><td>0.5432</td><td>0.6589</td><td>0.7031</td></tr><tr><td>BERTScore</td><td>0.7032</td><td>0.5214</td><td>0.6317</td><td>0.6894</td></tr><tr><td>QAEval</td><td>0.7125</td><td>0.4987</td><td>0.9274</td><td>0.6342</td></tr></table>

Table 11: Correlation coefficients between existing metrics and human assessment results.

![](images/e0ef8ef34c21696c9f3b692c34e3f88e3ac1b218a50eb24f98aa46ba5c246e43.jpg)  
Figure 8: Token consumption histograms distribution for Long-context Prompting.

![](images/6e699457b5a153daec23858c36327c2f3f898e3047f1a740124172585b01200c.jpg)  
Figure 9: Topic types’ influence on hierarchical merging GPT-3.5-Turbo.

![](images/9ce8c1f52e01bb07f9c964e1deeb063deff27575fc130eb1e0b3edf9876e67ed.jpg)  
Figure 10: The influence of the number of news articles on evaluation metrics.

![](images/08d1d656d0ea6029f4c77bcb2c459c959dfc893618882a729dbe7dabbabe87ae.jpg)  
Figure 11: Human annotation scoring sheets of the proposed metrics.