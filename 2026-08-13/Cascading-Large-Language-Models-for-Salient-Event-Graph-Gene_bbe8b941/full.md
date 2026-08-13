# Cascading Large Language Models for Salient Event Graph Generation

Xingwei Tan<sup>1,2</sup>, Yuxiang Zhou<sup>2</sup>, Gabriele Pergola<sup>1</sup>, Yulan He<sup>1,2,3</sup> <sup>1</sup>Department of Computer Science, University of Warwick, UK <sup>2</sup>Department of Informatics, King’s College London, UK <sup>3</sup>The Alan Turing Institute, UK {Xingwei.Tan, Gabriele.Pergola.1}@warwick.ac.uk {Yuxiang.Zhou, Yulan.He}@kcl.ac.uk

## Abstract

Generating event graphs from long documents is challenging due to the inherent complexity of multiple tasks involved such as detecting events, identifying their relationships, and reconciling unstructured input with structured graphs. Recent studies typically consider all events with equal importance, failing to distinguish salient events crucial for understanding narratives. This paper presents CALLM-SAE, a CAscading Large Language Model framework for SAlient Event graph generation, which leverages the capabilities of LLMs and eliminates the need for costly human annotations. We first identify salient events by prompting LLMs to generate summaries, from which salient events are identified. Next, we develop an iterative code refinement prompting strategy to generate event relation graphs, removing hallucinated relations and recover-<sub>Fine-tuned</sub> <sub>PLMs</sub> ing missing edges. Powered by CALLMSAE, we present NYT-SEG, a large-scale automatically annotated event graph dataset which can serve as distant supervision signals. Finetuning contextualised graph generation models on NYT-SEG outperforms the models trained on CAEVO data. Results on a human-annotated test set show that the proposed method generates salient and more accurate graphs, outperforming competitive baselines<sup>1</sup>.

## 1 Introduction

Events are fundamental discourse units which form the backbone of human communication. They are interconnected through various event relations such as hierarchical, temporal, or causal relations. Event relation graphs are vital for representing and understanding complex event narratives, with nodes representing events and edges denoting relationships between them. High-quality event relation graphs can enhance numerous downstream tasks, such as question answering (Lu et al., 2022) and reasoning (Melnyk et al., 2022).

![](images/09f03a3c12354d1982f110a03c4418e6775cf074583e0abcc4583580325899dc.jpg)  
Figure 1: An example of salient event relation graph (top) generated from the NYT article (bottom).

Existing studies on contextualized event graph generation primarily focus on fine-tuning language models for end-to-end generation of linearized graphs from documents (Madaan et al., 2021; Tan et al., 2024). These methods rely on distant supervision, such as events and event temporal relations detected using CAEVO (McDowell et al., 2017), due to the data-intensive nature of language models and the significant manual effort required for annotating event graphs. However, CAEVO’s bottom-up, predicate-centric extraction often results in sparse, low-quality event graphs populated with non-salient events. For example, CAEVO frequently identifies all verbs as events, including trivial ones like "say" and "think," which have minimal connections to other events and contribute little to narrative understanding, thus introducing noise. In recent research involving human annotation (Dunietz et al., 2014; Liu et al., 2018), a summarisation test is leveraged to identify salient events or entities, where an event or entity is considered salient if a human-written summary is likely to include it.

Motivated by these insights, we aim to improve the quality of distant supervision and mitigate noise in event graph generation by incorporating event saliency. Our hypothesis is that effective graph generation benefits from a top-down strategy, where the main events are identified first, rather than extracted solely in a bottom-up manner based on the predicate. Capitalizing on the powerful summarization capabilities of large language models (LLMs), we propose a method that first instructs LLMs to summarize documents before identifying salient events. Specifically, we introduce a codeprompting strategy for constructing event relation graphs encompassing hierarchical, temporal, and causal relations<sup>2</sup> (see Figure 1). Unlike vanilla prompting, which queries each potential event pair individually, our code prompt format generates each relation type in a single pass. Furthermore, we incorporate an iterative refinement process using a hallucination grader to filter spurious edges and iterative generation to recover missing ones. Finally, the abstractive nature of salient events presents an evaluation challenge, as they rarely match gold standard events exactly, even when semantically equivalent. To address this, we propose evaluation metrics based on semantic text embeddings for assessing the generated event relation graphs.

Our experimental results on the New York Times corpus (Sandhaus and Sandhaus, 2008) show that our proposed CALLMSAE, a novel CAscading Large Language Model framework for SAlient Event graph generation, outperforms the baselines in terms of event saliency and edge quality. The fine-tuned model surpasses previous models trained with CAEVO-generated graphs. Our contributions are summarised as follows:

• We propose CALLMSAE, a CAscading Large Language Model framework for SAlient Event graph generation, serving as a distant signal generator for contextualised graph generation models.

• We propose a novel contextualised evaluation metric for comparing salient event graphs. Our extensive experimental evaluation on LLM-generated event relation graphs in terms of event saliency and event relation on the NYT corpus, demonstrating how higher quality salient event graphs can improve contextualised graph generation.

• We provide a large-scale LLM-generated salient event graph dataset NYT-SEG with three major event relation types for distant supervision (10, 231 documents), along with a human-annotated test set (100 documents).

## 2 Related Work

Event Relation Graph Construction The early idea of event relation graph construction comes from UzZaman et al. (2013), which introduces a dataset for evaluating an end-to-end system which takes raw text as input and output TimeML annotations (i.e., temporal relations). CAEVO (McDowell et al., 2017) and Cogcomptime (Ning et al., 2018) both utilise a wide range of manually designed features to train MaxEnt and averaged perception for extracting events and relations. Han et al. (2019b) proposed a joint event and relation extraction model based on BERT (Devlin et al., 2019) and BiLSTM (Panchendrarajan et al., 2018). Other researchers focus on developing specialised sub-systems to classify extracted event pairs for relations (Ning et al., 2019; Han et al., 2019a; Wang et al., 2020; Pergola et al., 2021a,b; Tan et al., 2021, 2023). ATOMIC (Sap et al., 2019) is a large-scale commonsense knowledge graph containing the causes and effects of events. MAVEN-ERE (Wang et al., 2022) is built with event coreference, temporal, causal and subevent relations. However, ATOMIC and MAVEN-ERE completely rely on crowdsourcing and thus are difficult to extend. MAVEN-ERE is less than half the size of our dataset and does not consider the saliency of events.

Madaan et al. (2021) fine-tune GPT-2 to generate linearised graphs from documents in an end-to-end manner. Their temporal relation graphs used for training are produced by CAEVO. Following this direction, Tan et al. (2024) instead view the task as set generation and propose a framework based on set property regularisation and data augmentation. In this paper, we focus on generating multi-relation graphs via in-context learning, prompt interaction, and iterative refinement.

Salient Event Identification Several existing papers investigate the problem of identifying salient events. Choubey et al. (2018) build a rule-based classifier to identify central events by exploiting human-annotated event coreference relations. They find the central events either have large numbers of coreferential event mentions or have large stretch sizes. Jindal et al. (2020) propose a contextual model to identify salient events based on BERT and BiLSTM. They also mention several features, such as event trigger frequency, which are essential features to identify the salient events. Liu et al. (2018) propose a feature-based method using LeToR (Liu et al., 2009) and a neural-based method called Kernel-based Centrality Estimation. To train and evaluate their methods, they build a dataset based on the summarisation test: an event is considered salient if a summary written by a human is likely to include it. Zhang et al. (2021) combine the Kernel-based Centrality Estimation with the event and temporal relation extraction model of Han et al. (2019b) to build a salience-aware event chain modelling system. However, they only focus on single-dimensional chains and only model temporal relations.

## 3 Cascading LLMs to Generate Salient Event Graphs

CALLMSAE combines various prompts in a pipelined manner to generate salient event graphs. In this section, we introduce the prompts for generating salient events, and then the method for generating relation graphs based on the salient events. Lastly, we define an evaluation metric for comparing event graphs: Hungarian Graph Similarity.

## 3.1 Generate Salient Events

The summarisation test (Section 1) is often used to guide the annotation of salient events or entities (Dunietz et al., 2014; Liu et al., 2018; Lu et al., 2023; Lyu et al., 2024b). These studies identify events or entities included in human-written summaries as salient. Similarly, we instruct LLMs to generate a summary first and then extract events from it. Examples of the summarisation and salient event generation prompts are shown in the Appendix (Table 11 and 12).

## 3.2 Generate Graphs as Code Completion

While LLMs can extract salient events, they often struggle with identifying event relations (Chan et al., 2023; Sun et al., 2022, 2024; Tan et al., 2024). Prompt engineering for extracting event relations is complex due to the need to incorporate various terminologies and graph constraints. Moreover, prompt efficiency is crucial as generating a largescale dataset with LLMs can still incur significant computational costs, albeit less than crowdsourcing.

In our method, the main prompt for generating the event relation graph is formulated as a Python code completion task. The graph is defined using the NetworkX<sup>3</sup> package in Python, with nodes representing the salient events generated in Section 3.1. LLMs are instructed to complete the code by adding relation edges using NetworkX’s APIs.

Recent research suggests that formulating prompts as code can enhance LLMs’ reasoning abilities (Wang et al., 2023; Zhang et al., 2023). In our task, the Python code format effectively incorporates all necessary terminologies, enabling LLMs to understand them without confusion. The Python code format also allows for the inclusion of constraints (e.g., ensuring the graph is a directed acyclic graph) and additional instructions (e.g., ask for explanations) as comments. LLMs can generate explanations as comments without disrupting the main content of the graph, which is difficult to achieve in other formats (e.g., JSON). Moreover, the code template simplifies parsing the response, as LLMs are directed to use the “.add\_edge()” function to add the relations.

Since hierarchical, temporal, and causal relations are asymmetric, each can be represented by a Directed Acyclic Graph (DAG). We formulate three distinct prompts to guide LLMs in generating three DAGs, each representing one of these relation types. This approach avoids the complexity of a multi-label graph, and LLMs can focus on a single relation type and carefully consider the topological structure of the graph. We can also use the “.find\_cycle()” function from NetworkX to detect constraint violations reliably. In addition, if relation types are interconnected, the initially generated graphs can help the generation of subsequent graphs (as will be explained in Section 3.4). We provide an example of the code prompt in Appendix (Table 13).

## 3.3 Iterative Refinement

Hallucination Grader The code prompt efficiently guides LLMs to generate graphs, but it often generates hallucinated relations. Based on our preliminary experiments, these hallucinations stem from the models’ overconfidence in their relation predictions. Specifically, LLMs tend to infer event relations without explicit linguistic cues or strong evidence for logical inference. Consequently, LLMs predict far more relations than the gold standards, leading to low precision.

![](images/23f658a9d6d98444cdb8cf659ef28ce54caf396b6b886b214b75ec8d52ac613b.jpg)  
Figure 2: The proposed CALLMSAE framework.

Recent studies show that LLMs can evaluate and correct their own outputs (Madaan et al., 2023; Asai et al., 2024). Thus, we introduce a hallucination grader to address hallucination. For each relation edge generated, we pose a question to the LLMs to determine whether the relation is grounded in the given document. If the LLMs respond with a “yes”, the edge is retained; otherwise, it is discarded. An example of the hallucination grader prompt is shown in Appendix (Table 14).

Recover Missing Edges The main benefit of the hallucination grader is that it increases precision by removing low-confident edges. However, this process inevitably reduces recall. To mitigate this side effect, we introduce an iterative refinement process. After discarding hallucinated edges, we reinsert the code block containing the relation edges into the graph generation prompt and ask the LLMs to complete the code again. In this way, the LLMs can reconsider whether there are any missing relations in the document, thereby improving recall.

Once the LLMs generate a new graph, the hallucination grader checks the relation edges again. This process is repeated for a fixed number of times. We set the maximum number of iterations to 5 in our experiments, as the LLMs stop discovering new edges after 2 or 3 iterations in most documents.

## 3.4 Complement Relation Types

Hierarchical, temporal, and causal relations are not independent of each other. In this case, we found that providing the graph for the first relation can benefit the generation of the dependent relation’s graph. Specifically, we predict the hierarchical relation graph first. Then, we provide this graph to the LLMs and ask them to generate the temporal relation graph. Lastly, with both the hierarchical and temporal relation graphs available, the LLMs predict the causal relation graph.

The hierarchical relation describes two closely related events at different granularity levels. It focuses on the inherent semantics of the events and does not depend on other relation types. For example, “writing a dissertation” is a subevent of “doing a PhD”. Therefore, we choose to predict the hierarchical relations first. Temporal relations can depend on hierarchical relations. For example, knowing “doing a PhD” happened before “being prompted to Professor” allows us to deduce that “writing a thesis” also happened before “being prompted to Professor”. Thus, we predict temporal relations after hierarchical relations. Lastly, causal relations depend on both hierarchical and temporal relations, as the antecedent event in a causal relation must occur before the consequence. Therefore, the causal relation is predicted in the last step. For more details about the entire prompting process, please refer to the pseudocode in Appendix D.

## 3.5 Hungarian Graph Similarity

It is challenging to compare event relation graphs generated by LLMs due to the abstractive nature of generation, making it difficult to align the generated events with the gold standard events (Li et al., 2023). Moreover, salient events are often high-level and abstract rather than fine-grained and concrete, which means some variations in wording are not only acceptable but also expected. Instead of using exact matching (Zhao et al., 2024) or rule-based token matching (Tan et al., 2024) on events and relations to calculate $F _ { 1 }$ , adopting semantic-based evaluation metrics is more reasonable and fair. As more tasks adopt text generation frameworks, many researchers are also turning to metrics based on language models rather than traditional token matching metrics like ROUGE and BLUE (Goyal et al., 2022; Pratapa et al., 2023; Lyu et al., 2024a).

In this study, we propose a novel metric for evaluating LLM-generated event graphs, called Hungarian Graph Similarity (HGS). The metric is based on the Hungarian assignment algorithm (Kuhn and Kuhn, 1955), which is widely used in object detection to match generated objects and target objects (Carion et al., 2020). It can find the optimal assignment given a cost matrix containing the distance between elements in two lists of objects. We adapt this algorithm to match predicted edges with edges in the gold standard graphs as follows:

1. Encode the events using SFR-Embedding-Mistral (Meng et al., 2024).

2. Given two edges of the same relation type, let $\bar { e } _ { 1 } ^ { h } , \bar { e } _ { 1 } ^ { t }$ be the embeddings of the head event and the tail event in the first edge. Let $\bar { e } _ { 2 } ^ { h } , \bar { e } _ { 2 } ^ { t }$ be the embeddings of the head and tail events in the second edges. We define the distance between the edges as max $\left( D _ { c o s } ( \bar { e } _ { 1 } ^ { h } , \bar { e } _ { 2 } ^ { h } ) , D _ { c o s } ( \bar { e } _ { 1 } ^ { t } , \bar { e } _ { 2 } ^ { t } ) \right)$ , where $D _ { c o s } ( \cdot , \cdot )$ is the cosine distance.

3. Build a cost matrix by computing the distance between every edge pair in the gold and predicted edge sets. Pad the matrix to a square matrix with the maximum cost value of 1.

4. Apply the Hungarian algorithm to the cost matrix to get the minimal cost value. The final score is 1 cost value, making the value more intuitive (higher is better). To compute the HGS over all the documents, we weight the scores by the number of gold edges to obtain an average value.

In step 2, we take the maximum value of the distances between head and tail events because relation edges are considered matched only if both the head and tail events match.

For more detailed analysis, we define precisionoriented HGS and recall-oriented HGS. We match edges without padding the cost matrix in step 3 to obtain the total cost values of all matched edge pairs. Then, the total matched similarity is the number of matched edges minus the total cost. Precision-oriented HGS is computed by dividing the total matched similarity by the total number of predicted edges. Recall-oriented HGS is computed by dividing the total matched similarity by the total number of edges in the target graph.

## 4 Dataset

In this section, we describe how we construct the NYT-SEG which includes LLM-generated distant training pairs and a human-annotated dataset based on the New York Times (NYT) corpus.

## 4.1 Document Selection

We follow the same procedures as in (Tan et al., 2024) to select documents from the NYT corpus, one of the largest news datasets, with additional filtering based on document length. We select 10, 347 documents based on their descriptors indicating they are related to event narratives instead of opinions and discussions, such as sports and international politics. Among them, 100 documents are randomly sampled as the test set to be annotated by humans. More details about data selection are shown in Appendix A.1.

## 4.2 Annotation Settings

We recruited annotators from Prolific<sup>4</sup>. There are two subtasks: salient event identification and event relation identification. In the first subtask, the participants are asked to identify the salient event triplets: actor, trigger, and object (optional). We provide the definition of an event and several examples in the guidelines. They are instructed to do the summarisation test: the salient events should be the events they would include in the summary of the given article. Moreover, we provide some prominent features for helping annotators to identify salient events (Choubey et al., 2018; Jindal et al., 2020):

• Frequency: salient events are frequently mentioned in the articles.

• First appearance: salient events are often mentioned at the beginning of the article.

• Stretch size: salient events are often mentioned throughout the articles. Stretch size is the distance between the location where the event is first mentioned and last mentioned. A salient event usually has a large stretch size.

In the second stage, we ask participants to identify relation triplets: a source event, a relation type, and a target event. Both the source and target events should be among the salient events identified in the first stage. In the guideline, we define three relation types: happened\_before, caused\_by, and is\_subevent\_of. happened\_before indicates that the source event happened earlier than the target event. caused\_by means the source event would not have happened if the target event did not happen. is\_subevent\_of signifies that the source event is a subevent of the target event. Annotators were informed that relations would be either explicitly mentioned in the article or inferred based on evidence within the article. Further details about the guidelines and user interface can be found in Appendix A.4.

## 4.3 Inter Annotator Agreement

Identifying salient events and event relations is complicated and time-consuming. We found it challenging to educate participants about these concepts because, in daily life, the meanings of events and relations differ from their definitions in the field of information extraction. Moreover, the technical definitions are much less intuitive to those outside the academic field. As a result, thorough training of participants is important to obtain high-quality annotations.

In total, we recruited 3 annotators to annotate   
100 documents. Due to their varying availability,   
annotator 1 and 2 each annotated 45 documents,   
while annotator 3 annotated 20 documents. Among   
these, 5 documents were annotated by all three an  
notators. Following prior research in information   
extraction (Gurulingappa et al., 2012; Zhao et al.,   
2024), we used $F _ { 1 }$ to measure the inter-annotator   
agreement on these 5 documents. To compute inter  
annotator agreement, events or relations identified   
by one annotator are represented as set $S _ { 1 }$ . An  
other annotator’s annotation $S _ { 2 }$ serves as a pseudo  
reference to compute $\begin{array} { r } { \mathrm { p r e c i s i o n } = \frac { | S _ { 1 } \cap S _ { 2 } | } { | S _ { 1 } | } } \end{array}$ , recall |<sup>S1</sup>∩<sup>S2</sup>|, and the $F _ { 1 }$ score $\begin{array} { r } { = \frac { 2 | S _ { 1 } \cap S _ { 2 } | } { | S _ { 1 } | + | S _ { 2 } | } } \end{array}$ |<sup>S</sup>2|

Table 1 shows the agreement scores for stages 1 and 2. Identifying salient events is subjective, which makes it difficult to reach a complete agreement. Moreover, event relation identification is even more subjective and dependent on the previous stage, leading to less unanimous agreement.

<table><tr><td>Annotator</td><td>Stage 1</td><td>Stage 2</td></tr><tr><td>1 &amp; 2</td><td>0.838</td><td>0.676</td></tr><tr><td>1&amp;3</td><td>0.771</td><td>0.645</td></tr><tr><td>2&amp;3</td><td>0.847</td><td>0.710</td></tr><tr><td>Average</td><td>0.819</td><td>0.677</td></tr></table>

Table 1: Inter-annotator agreement measured by $F _ { 1 }$ .

## 4.4 Dataset Statistics

Table 2 shows the distributions of the relation types after applying the transitive closure to the annotated graphs. happened\_before emerges as the most frequent relation type, reflecting the predominant focus on temporal sequences in news articles, and they are relatively straightforward to identify. Conversely, caused\_by is the least frequent as it is the most challenging to identify.

<table><tr><td>Relation Type</td><td>Number</td></tr><tr><td>happened_before</td><td>310</td></tr><tr><td>caused_by</td><td>202</td></tr><tr><td>is_subevent_of</td><td>245</td></tr><tr><td>Total</td><td>757</td></tr></table>

Table 2: The distributions of the relation types.

## 5 Experiments

## 5.1 Model Settings

We compare against the following baselines:

• CAEVO (McDowell et al., 2017) is a pipeline system based on a MaxEnt classifier and manual features for extracting events and relations.

• Madaan et al. (2021) trained language models on CAEVO-generated linearised graphs with LM objective. We implemented their method to train a Flan-T5 model.

• Tan et al. (2024) also trained language models on CAEVO-generated graphs, but applied data augmentations and regularisations to mitigate the set element misalignment issue. We applied their method to train a Flan-T5.

• Han et al. (2019b) proposed a joint event and temporal relation extraction model. We adapted the model to hierarchical and causal relations by training on MAVEN-ERE. We also replaced BERT with Longformer (Beltagy et al., 2020) to make it suitable for long documents.

• GPT-4 (OpenAI, 2024) and GPT-3.5 are based on the generative pre-train framework.

<table><tr><td></td><td>Mean event number</td><td>Event Frequency ↑</td><td>First Appearance ↓</td><td>Stretch Size ↑</td></tr><tr><td>CAEVO</td><td>34.71</td><td>0.05</td><td>0.46</td><td>0.07</td></tr><tr><td>Human</td><td>8.26</td><td>0.11</td><td>0.31</td><td>0.20</td></tr><tr><td>GPT-4</td><td>6.49</td><td>0.09</td><td>0.37</td><td>0.18</td></tr><tr><td>Llama3</td><td>5.17</td><td>0.09</td><td>0.30</td><td>0.19</td></tr><tr><td>Mixtral</td><td>10.60</td><td>0.10</td><td>0.33</td><td>0.20</td></tr></table>

Table 3: The average number of extracted events and the saliency features (in percentage values).

We used “gpt-4-1106-preview” and “gpt-3.5- turbo” respectively.

• MIXTRAL is an LLM based on the Mistral model and the mixture of expert framework. We used the 8x7B instruct (Jiang et al., 2024).

• LLAMA3 is an LLM based on the Llama framework. We used the 70B-instruct 8bit version for a balance between speed and performance.

We fine-tuned a Flan-T5-base (250M) with the relation graphs generated by CALLMSAE, following the same method as in Tan et al. (2024). The baseline prompt evaluates whether each event pair is supported by the document, akin to the hallucination grader described in Section 3.3. Thus, it serves as an ablation of our method without incorporating the code prompt. Another baseline, which asks for the relation type given an event pair, is also tested and included in our experiment.

CALLMSAE is designed to be model-agnostic. Due to budget constraints and the preliminary test results, we chose Llama3 as the backbone of all the prompt-based methods detailed in Table 5.

## 5.2 Event Saliency Evaluation

Table 3 shows the salient features (defined in Section 4.2, computation formulas in Appendix B) extracted from various backbone LLMs using summarisation prompts, alongside comparison with CAEVO and human annotations. The LLM-generated events are much more salient than CAEVO-generated events and exhibit similarity to human annotations.

We also use human annotations to evaluate the saliency. In the salient event identification annotation, we provide the events generated by CAEVO and Mixtral as candidate salient events. Note that only the top CAEVO events ranked in saliency features are shown. Half of the candidates are from CAEVO and the other half are from Mixtral. They are randomly shuffled and then shown to the annotators. We compute the precision, recall, and $F _ { 1 }$ based on how the annotators select them. We also compute HGS using human-annotated salient events as references (Table 4). It is clear that although CAEVO extracted more events than Mixtral, many of them are not salient. Mixtral outperforms CAEVO significantly across all evaluation metrics.

<table><tr><td></td><td> $P$ </td><td> $R$ </td><td> $F _ { 1 }$ </td><td>HGS</td></tr><tr><td>CAEVO</td><td>3.29</td><td>3.72</td><td>3.49</td><td>18.18</td></tr><tr><td>Mixtral</td><td>48.97</td><td>56.77</td><td>52.59</td><td>67.15</td></tr></table>

Table 4: Precision, recall, and $F _ { 1 }$ based on the choices of the annotators. Hungarian graph similarity (HGS) is defined in Section 3.5. The values are in percentage.

## 5.3 Salient Event Relation Graph Evaluation

The salient event relation graph evaluation results are shown in Table 5. The compared methods (row 1 - 4) are outperformed by Baseline Prompt (row 5) on all relation types. Baseline Prompt (rel type) (row 6), which asks the model to generate all possible relation types given the event pair, performs slightly better than Baseline Prompt. However, Baseline Prompt and Baseline Prompt (relation type) are slow and costly because the number of prompts they need for building one graph is $O ( n ^ { 2 } )$ where n is the number of events in the document. On the other hand, the time complexity of Code Prompt is O(1). Moreover, Code Prompt’s overall HGS is significantly higher than Baseline Prompt and Baseline Prompt (relation type) on all relation types. Baseline Prompt check the event pairs more thoroughly and thus have higher recall but its precision is much lower. The complete CALLMSAE combines the code prompt and hallucination grader for iterative refinement, checking missing relations and verifying them to prevent hallucination. Code Prompt (dependent rels) (the 8th row) is an ablation of CALLMSAE (the 9 row), differing only in the absence of iterative refinement. These results highlight the effectiveness of the hallucination grading approach, which effectively increases the precision and strikes a balance with recall.

<table><tr><td></td><td colspan="3">Hierarchical</td><td colspan="3">Temporal</td><td colspan="3">Causal</td></tr><tr><td></td><td>PHGS</td><td>RHGS</td><td>HGS</td><td>PHGS</td><td>RHGS</td><td>HGS</td><td>PHGS</td><td>RHGS</td><td>HGS</td></tr><tr><td>Han et al. (2019b)</td><td>0.158</td><td>0.247</td><td>0.098</td><td>0.092</td><td>0.352</td><td>0.148</td><td>0.084</td><td>0.316</td><td>0.116</td></tr><tr><td>CAEVO</td><td></td><td></td><td></td><td>0.030</td><td>0.558</td><td>0.092</td><td></td><td></td><td></td></tr><tr><td>Madaan et al. (2021)</td><td></td><td></td><td></td><td>0.061</td><td>0.439</td><td>0.116</td><td></td><td></td><td></td></tr><tr><td>Tan et al. (2024)</td><td></td><td></td><td></td><td>0.126</td><td>0.335</td><td>0.187</td><td></td><td></td><td></td></tr><tr><td>Baseline Prompt</td><td>0.076</td><td>0.651</td><td>0.248</td><td>0.085</td><td>0.627</td><td>0.195</td><td>0.062</td><td>0.657</td><td>0.207</td></tr><tr><td>Baseline Prompt (rel type)</td><td>0.288</td><td>0.375</td><td>0.268</td><td>0.135</td><td>0.604</td><td>0.261</td><td>0.185</td><td>0.513</td><td>0.256</td></tr><tr><td>Code Prompt</td><td>0.174</td><td>0.559</td><td>0.315</td><td>0.153</td><td>0.678</td><td>0.283</td><td>0.121</td><td>0.632</td><td>0.272</td></tr><tr><td>Code Prompt (dependent rels)</td><td>N.A.</td><td>N.A.</td><td>N.A.</td><td>0.211</td><td>0.601</td><td>0.341</td><td>0.135</td><td>0.599</td><td>0.272</td></tr><tr><td>CALLMSÁE (ours)</td><td>0.196</td><td>0.544</td><td>0.334</td><td>0.294</td><td>0.509</td><td>0.327</td><td>0.198</td><td>0.529</td><td>0.295</td></tr><tr><td>Fine-tuned T5 (CALLMSAE)</td><td>0.314</td><td>0.434</td><td>0.339</td><td>0.244</td><td>0.544</td><td>0.362</td><td>0.366</td><td>0.397</td><td>0.343</td></tr></table>

Table 5: The Hungarian Graph Similarity (HGS) of the LLM-generated graphs on the human-annotated NYT dataset. PHGS is precision-oriented HGS. RHGS is recall-oriented HGS. Code Prompt (dependent rels) means adding hierarchical graphs in the prompts for temporal graphs; and adding hierarchical and temporal for causal graphs. Fine-tuned T5 (CALLMSAE) means fine-tuning a flan-T5 using the graphs generated by CALLMSAE. All prompt-based methods (row 5 - 9) are based on Llama3-70B-instruct.

In the temporal category, the results of Code Prompt (dependent rels) are obtained when provided with hierarchical graphs generated by CALLMSAE to LLMs. It has much higher overall HGS and precision than Code Prompt without hierarchical information, showing that hierarchical information can mitigate hallucinations during the temporal graph generation. In the casual category, the results of Code Prompt (dependent rels) are obtained when given both hierarchical and temporal graphs generated by CALLMSAE. The additional information also increases precision.

Fine-tuned T5 outperform all the methods based on CAEVO (McDowell et al., 2017; Madaan et al., 2021; Tan et al., 2024), showing that the highquality graphs generated by CALLMSAE can boost the contextualised graph generation. Interestingly, the performance of the Fine-tuned T5, finetuned on CALLMSAE-generated data, exceeds that of CALLMSAE itself, implying that the fine-tuned model can effectively adapt the reasoning patterns provided by Llama3 and generalise them.

<table><tr><td></td><td>Hier</td><td>Temp</td><td>Causal</td><td>Overall</td></tr><tr><td>Baseline (rel type)</td><td>0.58</td><td>0.66</td><td>0.48</td><td>0.60</td></tr><tr><td>CALLMSAE</td><td>0.71</td><td>0.71</td><td>0.65</td><td>0.69</td></tr><tr><td>Fine-tuned T5</td><td>0.72</td><td>0.74</td><td>0.62</td><td>0.70</td></tr></table>

Table 6: The human evaluation scores.

## 5.4 Human Evaluation of Event Graph

We recruited additional annotators to evaluate the generated graph on 50 of the test documents. We asked them whether the relation edges are correct. These scores $( \frac { \mathrm { c o r r e c t e d g e s } } { \mathrm { g e n e r a t e d e d g e s } } )$ can be viewed as precision (Table 6). The human evaluation correlates with the HGS scores, verifying our conclusion.

<table><tr><td colspan="2">Format Error ↓</td><td rowspan="2">Cycle ↓</td></tr><tr><td>GPT-3.5</td><td>0%</td></tr><tr><td>GPT-4</td><td>3.67%</td><td>10.67% 1.67%</td></tr><tr><td>Mixtral</td><td>3.33%</td><td>2.33%</td></tr><tr><td>Llama3</td><td>0%</td><td>0%</td></tr></table>

Table 7: The average number of CALLMSAE-generated graphs out of 100 with format errors or cycles.

## 5.5 Format Error and Cycles in the Graphs

We specified the relation graphs as directed acyclic graphs in the prompt. If there is a cycle in the generated graph, it means that the LLM failed to follow the instructions. A cycle also indicates constraint violations because all the relations in the graphs are asymmetric. We use the APIs in NetworkX package to detect cycles in the transitive closure of the graphs. If the Python interpreter returns an error, it is classified as a format error. We prompt each LLM three times on the test set. Table 7 shows the average number of documents encountering format errors or cycles. All LLMs have low rates of format errors which shows that state-of-the-art LLMs can understand the instruction well and generate executable Python code. Among them, GPT-3.5 and Llama3 have no errors. About 10% of graphs generated by GPT-3.5 have cycles, suggesting that

GPT-3.5 may have inferior reasoning ability compared to other LLMs. GPT-4 and Mixtral both have low rates of cycle occurrence, but they are beaten by Llama3 which has no cycle in all generations, showing its remarkable understanding of the transitive and asymmetric constraints in the complex event relation graphs.

## 6 Conclusion

This study explored utilising LLMs to generate salient event relation graphs from news documents without relying on human annotations. We studied how the events generated by LLMs are compared to the traditional methods in terms of event saliency. We further demonstrated that CALLMSAE-generated graphs can serve as distant signals to fine-tune smaller models and outperform those based on CAEVO. The CALLMSAEgenerated event graphs, together with the humanannotated test set are collected as NYT-SEG.

## Limitations

CALLMSAE is more demanding than CAEVO in terms of computational power and time. To generate the NYT-SEG, we spent 2, 200 wall clock hours in inference. More details about resource cost are disclosed in Appendix D.

Although we have tested many prompting methods and included several of the most effective ones in this paper, we have not explored all possible combinations due to the extensive volume of recent literature on prompt engineering. There might still exist combinations of prompts that could further improve performance (Alzaid et al., 2024; Zhou et al., 2024). However, we are almost certain that any potential combinations, if they exist, are likely to be more complex and thus less efficient for building large-scale datasets. For example, we did not add demonstrations in graph generation because the code template is already quite lengthy. Adding more documents could potentially exceed the context windows of some LLMs, making it challenging for them to interpret the instructions effectively. The main goal of this work is to demonstrate the potential of LLM-based generation can help the data-demanding event graph generation task.

## Ethics Statement

Event relation graph generation is a powerful tool for understanding text. A potential misuse of the proposed method is mining user behaviours on their private data. For example, salient event relation graphs can be extracted from users’ tweets to analyse their potential reactions to advertisements and scams. That could be a huge risk to social media users.

Another potential risk is that the saliency may introduce bias. LLMs may have their preferences in selecting a specific group of events as important events due to the data they were trained on. This is a question which requires further large-scale investigation. However, we think this risk is negligible in this study because we work on document-level information. There is little room for selection given that the news articles are already the products of choice and distillation. If the system is used to extract information from a border information source, such as social media, the risk must be carefully assessed.

## Acknowledgements

This work was supported in part by the UK Engineering and Physical Sciences Research Council (EPSRC) through a Turing AI Fellowship (grant no. EP/V020579/1, EP/V020579/2). Xingwei Tan was supported by the Warwick Chancellor’s International Scholarship. This work was conducted on the Sulis Tier-2 HPC platform hosted by the Scientific Computing Research Technology Platform at the University of Warwick. Sulis is funded by EP-SRC Grant EP/T022108/1 and the HPC Midlands+ consortium.

## References

Ethar Alzaid, Gabriele Pergola, Harriet Evans, David Snead, and Fayyaz Minhas. 2024. Large multimodal model-based standardisation of pathology reports with confidence and its prognostic significance. The Journal ofPathology: Clinical Research, 10(6):e70010.

Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2024. Self-RAG: Learning to retrieve, generate, and critique through self-reflection. In The Twelfth International Conference on Learning Representations.

Iz Beltagy, Matthew E. Peters, and Arman Cohan. 2020. Longformer: The long-document transformer. CoRR, abs/2004.05150.

Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. 2020. End-to-end object detection with transformers. In European conference on computer vision, pages 213–229. Springer.

Chunkit Chan, Jiayang Cheng, Weiqi Wang, Yuxin Jiang, Tianqing Fang, Xin Liu, and Yangqiu Song. 2023. Chatgpt evaluation on sentence level relations: A focus on temporal, causal, and discourse relations.

Prafulla Kumar Choubey, Kaushik Raju, and Ruihong Huang. 2018. Identifying the most dominant event in a news article by mining event coreference relations. In Proceedings ofthe 2018 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers), pages 340–345, New Orleans, Louisiana. Association for Computational Linguistics.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings ofthe 2019 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.

Jesse Dunietz and Daniel Gillick. 2014. A new entity salience task with millions of training examples. In Proceedings of the 14th Conference of the European Chapter of the Association for Computational Linguistics, volume 2: Short Papers, pages 205–209, Gothenburg, Sweden. Association for Computational Linguistics.

Tanya Goyal, Junyi Jessy Li, and Greg Durrett. 2022. News summarization and evaluation in the era of gpt-3. arXiv preprint arXiv:2209.12356.

Harsha Gurulingappa, Abdul Mateen Rajput, Angus Roberts, Juliane Fluck, Martin Hofmann-Apitius, and Luca Toldo. 2012. Development of a benchmark corpus to support the automatic extraction of drugrelated adverse effects from medical case reports. Journal ofbiomedical informatics, 45(5):885–892.

Rujun Han, I-Hung Hsu, Mu Yang, Aram Galstyan, Ralph Weischedel, and Nanyun Peng. 2019a. Deep structured neural network for event temporal relation extraction. In Proceedings of the 23rd Conference on Computational Natural Language Learning (CoNLL), pages 666–106, Hong Kong, China. Association for Computational Linguistics.

Rujun Han, Qiang Ning, and Nanyun Peng. 2019b. Joint event and temporal relation extraction with shared representations and structured prediction. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 434–444, Hong Kong, China. Association for Computational Linguistics.

Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las

Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2024. Mixtral of experts.

Disha Jindal, Daniel Deutsch, and Dan Roth. 2020. Is killed more significant than fled? a contextual model for salient event detection. In Proceedings of the 28th International Conference on Computational Linguistics, pages 114–124, Barcelona, Spain (Online). International Committee on Computational Linguistics.

Harold W Kuhn. 1955. The hungarian method for the assignment problem. Naval research logistics quarterly, 2(1-2):83–97.

Bo Li, Gexiang Fang, Yang Yang, Quansen Wang, Wei Ye, Wen Zhao, and Shikun Zhang. 2023. Evaluating chatgpt’s information extraction capabilities: An assessment of performance, explainability, calibration, and faithfulness.

Tie-Yan Liu et al. 2009. Learning to rank for information retrieval. Foundations and Trends® in Information Retrieval, 3(3):225–331.

Zhengzhong Liu, Chenyan Xiong, Teruko Mitamura, and Eduard Hovy. 2018. Automatic event salience identification. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 1226–1236, Brussels, Belgium. Association for Computational Linguistics.

Junru Lu, Jiazheng Li, Byron Wallace, Yulan He, and Gabriele Pergola. 2023. NapSS: Paragraph-level medical text simplification via narrative prompting and sentence-matching summarization. In Findings of the Association for Computational Linguistics: EACL 2023, pages 1079–1091, Dubrovnik, Croatia. Association for Computational Linguistics.

Junru Lu, Xingwei Tan, Gabriele Pergola, Lin Gui, and Yulan He. 2022. Event-centric question answering via contrastive learning and invertible event transformation. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2022, pages 2377–2389, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Chen Lyu and Gabriele Pergola. 2024a. SciGisPy: a novel metric for biomedical text simplification via gist inference score. In Proceedings of the Third Workshop on Text Simplification, Accessibility and Readability (TSAR 2024), pages 95–106, Miami, Florida, USA. Association for Computational Linguistics.

Chen Lyu and Gabriele Pergola. 2024b. Society of medical simplifiers. In Proceedings of the Third Workshop on Text Simplification, Accessibility and Readability (TSAR 2024), pages 61–68, Miami, Florida, USA. Association for Computational Linguistics.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. 2023. Self-refine: Iterative refinement with self-feedback. In Thirty-seventh Conference on Neural Information Processing Systems.

Aman Madaan and Yiming Yang. 2021. Neural language modeling for contextualized temporal graph generation. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 864–881, Online. Association for Computational Linguistics.

Bill McDowell, Nathanael Chambers, Alexander Ororbia II, and David Reitter. 2017. Event ordering with a generalized model for sieve prediction ranking. In Proceedings of the Eighth International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 843–853, Taipei, Taiwan. Asian Federation of Natural Language Processing.

Igor Melnyk, Pierre Dognin, and Payel Das. 2022. Knowledge graph generation from text. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2022, pages 1610–1622, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Rui Meng, Ye Liu, Shafiq Rayhan Joty, Caiming Xiong, Yingbo Zhou, and Semih Yavuz. 2024. Sfrembedding-mistral:enhance text retrieval with transfer learning. Salesforce AI Research Blog.

Qiang Ning, Sanjay Subramanian, and Dan Roth. 2019. An improved neural baseline for temporal relation extraction. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 6203–6209, Hong Kong, China. Association for Computational Linguistics.

Qiang Ning, Ben Zhou, Zhili Feng, Haoruo Peng, and Dan Roth. 2018. CogCompTime: A tool for understanding time in natural language. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 72–77, Brussels, Belgium. Association for Computational Linguistics.

OpenAI, 2024. 2024. Gpt-4 technical report. OpenAI.

Rrubaa Panchendrarajan and Aravindh Amaresan. 2018. Bidirectional LSTM-CRF for named entity recognition. In Proceedings of the 32nd Pacific Asia Conference on Language, Information and Computation, Hong Kong. Association for Computational Linguistics.

Gabriele Pergola, Lin Gui, and Yulan He. 2021a. A disentangled adversarial neural topic model for separating opinions from plots in user reviews. In Proceedings of the 2021 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 2870–2883, Online. Association for Computational Linguistics.

Gabriele Pergola, Elena Kochkina, Lin Gui, Maria Liakata, and Yulan He. 2021b. Boosting low-resource biomedical QA via entity-aware masking strategies. In Proceedings ofthe 16th Conference ofthe European Chapter ofthe Associationfor Computational Linguistics: Main Volume, pages 1977–1985, Online. Association for Computational Linguistics.

Adithya Pratapa, Kevin Small, and Markus Dreyer. 2023. Background summarization of event timelines. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 8111–8136, Singapore. Association for Computational Linguistics.

Evan Sandhaus. 2008. The New York Times Annotated Corpus.

Maarten Sap, Ronan Le Bras, Emily Allaway, Chandra Bhagavatula, Nicholas Lourie, Hannah Rashkin, Brendan Roof, Noah A. Smith, and Yejin Choi. 2019. Atomic: An atlas of machine commonsense for if-then reasoning. In Proceedings of the Thirty-Third AAAI Conference on Artificial Intelligence and Thirty-First Innovative Applications ofArtificial Intelligence Conference and Ninth AAAI Symposium on Educational Advances in Artificial Intelligence, AAAI’19/IAAI’19/EAAI’19. AAAI Press.

Zhaoyue Sun, Jiazheng Li, Gabriele Pergola, Byron Wallace, Bino John, Nigel Greene, Joseph Kim, and Yulan He. 2022. PHEE: A dataset for pharmacovigilance event extraction from text. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 5571–5587, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Zhaoyue Sun, Gabriele Pergola, Byron Wallace, and Yulan He. 2024. Leveraging ChatGPT in pharmacovigilance event extraction: An empirical study. In Proceedings ofthe 18th Conference ofthe European Chapter of the Association for Computational Linguistics (Volume 2: Short Papers), pages 344–357, St. Julian’s, Malta. Association for Computational Linguistics.

Xingwei Tan, Gabriele Pergola, and Yulan He. 2021. Extracting event temporal relations via hyperbolic geometry. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 8065–8077, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Xingwei Tan, Gabriele Pergola, and Yulan He. 2023. Event temporal relation extraction with Bayesian

translational model. In Proceedings ofthe 17th Conference of the European Chapter of the Association for Computational Linguistics, pages 1125–1138, Dubrovnik, Croatia. Association for Computational Linguistics.

Xingwei Tan, Yuxiang Zhou, Gabriele Pergola, and Yulan He. 2024. Set-aligning framework for autoregressive event temporal graph generation. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 3872–3892, Mexico City, Mexico. Association for Computational Linguistics.

Naushad UzZaman, Hector Llorens, Leon Derczynski, James Allen, Marc Verhagen, and James Pustejovsky. 2013. SemEval-2013 task 1: TempEval-3: Evaluating time expressions, events, and temporal relations. In Second Joint Conference on Lexical and Computational Semantics (\*SEM), Volume 2: Proceedings of the Seventh International Workshop on Semantic Evaluation (SemEval 2013), pages 1–9, Atlanta, Georgia, USA. Association for Computational Linguistics.

Haoyu Wang, Muhao Chen, Hongming Zhang, and Dan Roth. 2020. Joint constrained learning for eventevent relation extraction. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 696–706, Online. Association for Computational Linguistics.

Xiaozhi Wang, Yulin Chen, Ning Ding, Hao Peng, Zimu Wang, Yankai Lin, Xu Han, Lei Hou, Juanzi Li, Zhiyuan Liu, Peng Li, and Jie Zhou. 2022. MAVEN-ERE: A unified large-scale dataset for event coreference, temporal, causal, and subevent relation extraction. In Proceedings ofthe 2022 Conference on Empirical Methods in Natural Language Processing, pages 926–941, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Xingyao Wang, Sha Li, and Heng Ji. 2023. Code4Struct: Code generation for few-shot event structure prediction. In Proceedings ofthe 61st Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 3640–3663, Toronto, Canada. Association for Computational Linguistics.

Li Zhang, Hainiu Xu, Yue Yang, Shuyan Zhou, Weiqiu You, Manni Arora, and Chris Callison-Burch. 2023. Causal reasoning of entities and events in procedural texts. In Findings of the Association for Computational Linguistics: EACL 2023, pages 415–431, Dubrovnik, Croatia. Association for Computational Linguistics.

Xiyang Zhang, Muhao Chen, and Jonathan May. 2021. Salience-aware event chain modeling for narrative understanding. In Proceedings ofthe 2021 Conference on Empirical Methods in Natural Language Processing, pages 1418–1428, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Runcong Zhao, Qinglin Zhu, Hainiu Xu, Jiazheng Li, Yuxiang Zhou, Yulan He, and Lin Gui. 2024. Large language models fall short: Understanding complex relationships in detective narratives. arXiv preprint arXiv:2402.11051.

Yuxiang Zhou, Jiazheng Li, Yanzheng Xiang, Hanqi Yan, Lin Gui, and Yulan He. 2024. The mystery of in-context learning: A comprehensive survey on interpretation and analysis. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 14365–14378, Miami, Florida, USA. Association for Computational Linguistics.

## A Additional Details of Dataset Construction

## A.1 Document Selection

We select news documents from the NYT corpus based on the descriptors available. With regards to the generation of salient even graphs, the most relevant documents tend to be centered around event narratives, so that they could be rich in event relations. Tan et al. (2024) investigated which descriptors are rich in event narrative using event frequency inverse-descriptor frequency. We chose the documents using the same descriptors as them (e.g., “airlines and airplanes”, “united states international relations”, “civil war and guerrilla warfare”, “track andfield”, “soccer”, etc.).

We applied additional filtering based on the number of words in the documents. Documents with more than 8500 words or less than 100 words are excluded. Based on our preliminary observations, the extremely long documents are not typically news articles (only takes 0.02% in the entire NYT). They tend to be collections of articles over longer time spans, making them not suitable as focus of this study. Additionally, very long articles may affect the performance of open-source LLMs only due to limitations in the context length rather than their reasoning abilities. On the other hand, articles that are too short are less likely to contain complex event relation graphs, so we also exclude them. The final average word count of the selected 10347 documents is 780.

## A.2 Frequent words and descriptors in the annotated dataset

<table><tr><td rowspan="2">Rank</td><td colspan="2">Test</td><td colspan="2">Train</td></tr><tr><td>Word</td><td>Count</td><td>Word</td><td>Count</td></tr><tr><td>1</td><td>win</td><td>41</td><td>win</td><td>2,964</td></tr><tr><td>2</td><td>express</td><td>15</td><td>make</td><td>1,591</td></tr><tr><td>3</td><td>play</td><td>14</td><td>face</td><td>1,564</td></tr><tr><td>4</td><td>make</td><td>13</td><td>express</td><td>1,411</td></tr><tr><td>5</td><td>defeat</td><td>12</td><td>include</td><td>1,307</td></tr></table>

Table 8: The top 5 most frequent trigger words in the human-annotated test set and the distant train set.

Table 8 reports the most frequent trigger words among the human-identified salient events and LLM-generated salient events after filtering out the light words (words that have no semantic meaning). We could see that “win”, “play”, and “defeat” are prominent triggers due to the sports topics within the dataset. These articles usually mention multiple events with these triggers. Triggers like “express”, “include”, and “make” are instead common across different scenarios.

Table 9 shows the most frequent descriptors in the human-annotated test set and the distant train set. These are the typical event-rich topics and are full of narratives.

## A.3 Disclaimers of Risks

Consider that a large portion of the new articles in the New York Times corpus are about violent incidences, such as terrorist attacks and war. To prevent inflicting harm to traumatised victims, we show the information clearly in the recruitment description on the Prolific platform (Figure 3).

![](images/fae74d11a58a9914ff36328e6fc2a7d799580058ee2f8547ba6d396d2e3abe4d.jpg)  
Figure 3: The recruitment descriptions.

## A.4 Guidelines and User Interface

A well-designed user interface is essential for collecting high-quality data efficiently. We fully cooperate with participants to improve the user interface iteratively based on their feedback.

In the salient event identification stage (Figure 4), we show the title, abstract, and content of the article on the right side. We show candidate events, which are extracted through CAEVO and Mixtral, on the left sidebar. The shown CAEVO events are the top events ranked based on the saliency feature score. The participants can choose the candidate events which they think are accurate and salient. The guideline also informs them that if multiple options refer to the same event, they can only choose the most accurate and informative one. If a salient event is not present among the candidates, they could write it in the text input box and add it.

In the event relation identification stage (Figure 5), they could choose a source event, a relation type, and a target event to add a relation triplet. The source event and the target event need to be chosen from the salient event list from the first stage. We automatically detect and prevent any new event that will lead to duplication and contradiction.

![](images/d358e219a198aa3ce886dbece71fea91b870b0386892ebfbe0769586360fdcee.jpg)  
Figure 4: The user interface of salient event identification.

![](images/705bf423f7a15d07d3302940c3c231870d0ea6bb9dc5453295aa1980220371e0.jpg)  
Figure 5: The user interface of event relation identification.

<table><tr><td rowspan="2">Rank</td><td colspan="2">Test</td><td colspan="2">Train</td></tr><tr><td>Descriptor</td><td>Count</td><td>Descriptor</td><td>Count</td></tr><tr><td>1</td><td>U.S. International Relations</td><td>27</td><td>Terrorism</td><td>2,885</td></tr><tr><td>2</td><td>Terrorism</td><td>21</td><td>U.S. International Relations</td><td>2,574</td></tr><tr><td>3</td><td>Bombs and Explosives</td><td>17</td><td>Bombs and Explosives</td><td>1,727</td></tr><tr><td>4</td><td>U.S. Armament and Defense</td><td>15</td><td>U.S. Armament and Defense</td><td>1,717</td></tr><tr><td>5</td><td>Politics and Government</td><td>15</td><td>Politics and Government</td><td>1,649</td></tr></table>

Table 9: The top 5 most frequent descriptors in the human-annotated test set and the distant train set.

The participants can also deselect the added event if they change their minds. The participants were asked to finish the first stage first, and then annotate the second stage based on their own annotations in the first stage.

In the following are reported the screenshots of the guideline pages (Figure 6).

## A.5 More details about the annotation

We started the annotation process by releasing several trial rounds, during which we chose participants based on their dedication and understanding of the terminologies. It required considerable communication efforts to ensure they had an accurate understanding of the task definition.

During training, we found a common mistake among the annotators was that they tended to overestimate the is\_subevent\_of relation. They often confused it with the caused\_by relation or temporal inclusion.

We advised them that is\_subevent\_of pertains to two events on different granularity levels but referring to the same subject. To distinguish is\_subevent\_of from temporal overlap, they could check whether the actor in the subevent is the same as or a part of the actor or object in the parent event. For example, if a parent event is “a team did something” the subevent can be “a member ofthe team did something”.

## A.6 Information about the Annotators

The annotators were paid at the rate of 8£/h. We screened native English speakers from all over the world to ensure they could read English articles fluently. We also selected participants based on their previous submissions and approval rates to ensure they were familiar with the platform and were high-quality annotators.

Two of the final annotators are identified as male, and they both come from the UK. One of the final annotators is identified as female, and she comes from Canada. They all identified as white.

## A.7 Dataset Licensing

The original NYT corpus is available for noncommercial research license. One of our authors has obtained the license. Based on the license, we could not include the original text in our dataset. Thus, we will only release the generated/annotated graphs. Our dataset will also be in noncommercial research license.

## B Saliency Features

Inspired by (Choubey et al., 2018), we calculate the saliency features to show how our proposed method differs from previous methods in terms of event saliency. Unlike conventional computation methods, these saliency features are calculated on the sentence level to be comparable across documents of various lengths. These saliency features are:

Event frequency: A salient event tends to appear frequently in the document. Let $\begin{array} { r l } { D } & { { } = } \end{array}$ $\left\{ s _ { 0 } , s _ { 1 } , . . . , s _ { n - 1 } , s _ { n } \right\}$ be the document and the list of sentences in the document. Let e be the event. Let $M ( e ) = \{ s _ { i } , s _ { j } , . . . , s _ { k } \} , 0 \leqslant i < j < k \leqslant n$ be the list of sentences which mention the event e. The event frequency is calculated as:

$$
f r e q u e n c y ( D , e ) = \frac { | M ( e ) | } { n + 1 } .\tag{1}
$$

First appearance: News writers usually mention the salient event as early as possible to attract readers’ attention. The first appearance of the event e is computed as:

$$
f i r s t \_ a p p e a r a n c e ( D , e ) = \frac { i } { n } .\tag{2}
$$

Stretch size: Salient events tend to be mentioned all across the document. The stretch size of event e is calculated as:

$$
s t r e t c h \_ s i z e ( D , e ) = \frac { k - i } { n } .\tag{3}
$$

To detect which sentences mention the event e, we first lemmatise the words in the document and the given event. Then, detect whether there is a matched substring the same as the given event in each sentence. However, the abstractive nature of LLM-based salient event generation makes exact matching not viable. To detect the event mention of LLM-generated events, we formulate a series of prompts. We first ask: “Which sentence in the document below mentions the event "{event}"? Please enclose that sentence in () and show it. Document: """{doc\_content}"""”. Then, we employ iterative refinement in case the LLM misses any other sentences: “Is there any other sentence in the document directly mentioning the event "{event}"? Please enclose that sentence in () and show it.” Lastly, we collect the sentences from the responses.

We run the methods on the human-annotated dataset (100 documents). We compute the saliency features of the events in each document and take the average across the events. Lastly, all the values are averaged across all the documents.

<table><tr><td></td><td>Hier</td><td>Temp</td><td>Causal</td></tr><tr><td>JSON</td><td>0.310</td><td>0.277</td><td>0.251</td></tr><tr><td>Python</td><td>0.315</td><td>0.283</td><td>0.272</td></tr></table>

Table 10: The comparison between Python format prompt and JSON format prompt in Hungarian Graph Similarity.

## C Comparison of Formats

In our preliminary experiments, we tested asking LLMs to generate event relations in JSON format (Table 10). When generating JSON response, we found LLMs often make format errors (e.g., using double quotation marks in the wrong place) which makes the responses difficult to parse, resulting in lower HGS. We also found the JSON format is not as flexible as the Python format which can integrate more information (e.g., explanations and definitions) as comments.

## D Prompting Details

Due to limited resources, the sampling of the distant supervision dataset was conducted on multiple machines with different specifications, including one with 6 RTX 3090, one with an A100, and one with 2 A40. The total time cost for prompting Llama3 to construct the training data (10, 231 documents) is about 2, 200 hours (the wall-clock time of all the machines combined).

Table 13 shows an example of the code prompt for hierarchical graph generation and the response from Llama3. Table 14 shows an example of the hallucination prompt and the response.

Algorithm 1 is the pseudo-code of the entire salient event graph generation process.

In the summarization prompt, we use a temperature of 0.8 and a top\_p of 0.9. For the salient event generation prompt, we use a temperature of 0.5 and a top\_p of 0.9. The relation graph generation prompt also uses a temperature of 0.5 and a top\_p of 0.9. The hallucination grader prompt uses a temperature of 0.

<table><tr><td>Input to the LLMs Write a summary of the document below using one paragraph.</td><td>Response from the LLMs The Liberal Party government in British</td></tr><tr><td>Document: """Residents of British Columbia will receive a big cut in their income taxes on New Year's Day, their second in six months. But far from winning applause, the province's govern- ing Liberal Party is experiencing a drop in public support. Tax cuts were a major promise in the campaign that brought the Liberals to office in the province in a landslide election win last May. The Liber- als, led by Gordon Campbell, a former secondary school teacher, real-estate executive and three- term mayor of Vancouver, won all but two of the 79 seats in the western province's legislative assembly. They defeated the left-leaning New Democrats, whose 10 years in office were marked by a growing public role in the economy and nu- merous tales of economic mismanagement. With the Jan. 1 reductions, personal income tax rates for provincial taxes – which constitute a much larger share of overall taxes for Canadi- ans than state taxes do for Americans – will have been chopped by an average of 25 percent, turn- ing British Columbia from one of Canada's most heavily taxed provinces to one with some of the lowest tax rates. "…" Summary:</td><td>Columbia, led by Gordon Campbell, has imple- mented a significant cut in income taxes on New Year's Day, fulfilling a campaign promise, but it has coincided with an economic downturn. De- spite the tax cuts, the province is facing a large deficit due to falling tax revenues and increased de- mand for social services. To address this, the gov- ernment plans to reduce the civil service by 28%, impose a three-year spending freeze on health- care and education, and tighten benefits under the government-financed drug plan. However, many British Columbians are unhappy with these mea- sures, and the Liberal Party's approval rating has dropped significantly, amidst concerns about the return of political polarization in the province.</td></tr><tr><td>Input to the LLMs A structured event is something that happened as</td><td>Response from the LLMs</td></tr><tr><td>described in the text. A structured event is rep- resented as a tuple, which consists of actors, a trigger, and objects. Could you list all the struc- tured events in the following article? Example: 1. (John; married; Alice). 2. (Alice; was hired; by Google). Article: """summary"""</td><td>Here are the structured events extracted from the article: 1. (Gordon Campbell; implemented; a significant cut in income taxes) 2. (The Liberal Party government; fulfilled; a campaign promise) 3. (The government; plans to reduce; the civil service by 28%) 4. (The government; will impose; a three-year spending freeze on healthcare and education) 5. (The government; will tighten; benefits under the government-financed drug plan) 6. (Many British Columbians; are unhappy with; these measures) 7. (The Liberal Party's approval rating; has dropped significantly) Note that some of these events may be implicit or inferred from the text, rather than explicitly stated.</td></tr></table>

Table 11: An example of the hallucination prompt input and the response from Llama3.

Table 12: An example of the salient event generation prompt input and the response from Llama3.
<table><tr><td>Input to the LLMs Could you please finish the following code?</td><td>Response from the LLMs Here is the completed code:</td></tr><tr><td>import networkx as nx document = {document} event_list = {event_list_str} # This is a graph representing the hierarchical relation between the events in the document # Each edge in the graph represents a subevent relation between the head and tail nodes which are events # An edge means the head event is a subevent of the tail event. They are closely related but on different granularity levels. hierarchical_graph = nx.DiGraph() # This is a directed acyclic graph. There should not be any cycle in the graph. # Add events as nodes for event in event_list: hierarchical_graph.add_node(event) # Add hierarchical relations as edges to the graph using .add_edge() function. The function takes two strings as inputs. Each string represents an event. # There should not be any cycle in the graph. Ex- plain the reason for each added edge as a comment after each function call</td><td>import networkx as nx hierarchical_graph = nx.DiGraph() # Add events as nodes for event in event_list: hierarchical_graph.add_node(event) # Add hierarchical relations as edges to the graph hierarchical_graph.add_edge(“the government re- sponded by reducing the provincial civil service by 28%", “liberals cut spending”) # The event “the government responded by re- ducing the provincial civil service by 28%" is a subevent of “liberals cut spending" because it is one way the Liberals cut spending.</td></tr><tr><td>Input to the LLMs</td><td>Response from the LLMs</td></tr><tr><td>You are a grader assessing whether an answer is grounded in / supported by a set of facts. Give a bi- nary score yes’ or ‘no’ score to indicate whether the answer is grounded in or supported by a set of facts. Then, provide a short explanation. Here are the facts: {document} Here is the answer: Event “the government re- sponded by reducing the provincial civil service by 28%" is a subevent of event “liberals cut spend- ing".</td><td>Score: Yes Explanation: The answer is grounded in the facts because it accurately identifies a specific action taken by the government (reducing the provincial civil service by 28%) as a subevent of the broader event of cutting spending, which is mentioned in the text.</td></tr></table>

Table 13: An example of the relation graph generation prompt input and the response from Llama3.

Table 14: An example of the hallucination prompt input and the response from Llama3.

## Guideline

![](images/1917f912fcb79a85dd5f244b3b1e185a86774927c2b51c67d5c890ec989a48bd.jpg)

Algorithm 1 CALLMSAE: CAscading Large Language Models for SAlient Event graph generation   
Input: Document d, Max Refinement Round k   
Output: An Event Relation Graph g   
1: summary Summary\_Generation(d)   
2: salient\_events Event\_Generation(summary)   
3: hierarchical\_graph null   
4: current\_round 0   
5: while current\_round < n do   
6: hierarchical\_graph Hierarchical\_Graph\_Generation(d, salient\_events,   
hierarchical\_graph)   
7: hierarchical\_edges Get\_Edges(hierarchical\_graph)   
8: for edge in hierarchical\_edges do   
9: remove\_edge Hallucination\_Grader(d, edge )   
10: if remove\_edge then   
11: hierarchical\_graph Remove\_edge(hierarchical\_graph, edge )   
12: end if   
13: end for   
14: current\_round current\_round + 1   
15: end while   
16: temporal\_graph null   
17: current\_round 0   
18: while current\_round < n do   
19: temporal\_graph Temporal\_Graph\_Generation(d, salient\_events, temporal\_graph,   
hierarchical\_graph)   
20: temporal\_edges Get\_Edges(temporal\_graph)   
21: for edge in temporal\_edges do   
22: remove\_edge Hallucination\_Grader(d, edge )   
23: if remove\_edge then   
24: temporal\_graph Remove\_edge(temporal\_graph, edge )   
25: end if   
26: end for   
27: current\_round current\_round + 1   
28: end while   
29: causal\_graph null   
30: current\_round 0   
31: while current\_round < n do   
32: causal\_graph Causal\_Graph\_Generation(d, salient\_events, causal\_graph,   
temporal\_graph, hierarchical\_graph)   
33: causal\_edges Get\_Edges(causal\_graph)   
34: for edge in causal\_edges do   
35: remove\_edge Hallucination\_Grader(d, edge )   
36: if remove\_edge then   
37: causal\_graph Remove\_edge(causal\_graph, edge )   
38: end if   
39: end for   
40: current\_round current\_round + 1   
41: end while   
42: g  hierarchical\_graph, temporal\_graph, causal\_graph

2. If you think there is an event that isn't listed, you can add it by entering the event in the text box.The event should at least contain a subject and a trigger.

3. If you think a ticked event makes no sense, untick it. When two options are referring to the same event, untick the one you think is less informative.

Every modification will be saved automatically

## Definition of Event

An event is anything that happens as described in the article. We represent the events in a structured format: actor; trigger; target. The actor of the event is usually the subject of a sentence. The trigger can be seen as the predicate of a sentence. The target is usually the object in the sentence which is optional.

![](images/fd72c455bad1b860b879bd85cdeeac2a69d76ced66fd3ab0b87a24d31b000e7d.jpg)

In example 1, New York: is one of the four candidate cities: competina to be presented to the lOC should not be chosen because the predicate isn't something that can be considered as an event. An event is essentially a change of state. Predicates like "is" is only describing one state. On the other hand, New York; is competing; with three cities to be presented to the lOC should be chosen because the predicate can indicate an event.

![](images/1507101a46282a341e050a1c27d9e5094ef795a3c2024f5bc666fc0ecc4b64ea.jpg)  
In example 2, Daniel L. Doctoroff; said; we don't want any sympathy for that is an event but not a salient event because simply describing someone said something isn't important enough in this article.

![](images/4c5bb84c046bd5e4e092eaf62500051307ac0171c071c8e67d83aa62b07a046a.jpg)  
In example 3, MetroStars; won; Major League isn't selected because it is less accurate and imformative than the second option. They are referring to the same event and we don't want duplication. The fourth and fifth options are not selected due to they are more about a description of state than a change of state. Game; was marked; by physical play isn't selected because there is no direct reference in the article that the game is marked by physical play

![](images/4cb4dfbb6ab035675e7ec792a818a8d2b84e66dbe0a70ada0d1f8bc75ee88233.jpg)  
Figure 6: Annotation guidelines of salient event identification shown to the annotators.

![](images/35a6cd3e59cc61ddb6dff25717864a820cd80ce64903607f613d112457eb478d.jpg)  
In example 2, (Gareth H. Edmondson-Jones; will be flying back; on a new jet from the Airbus factory in Toulouse, France) is part of the vacation in the event (Gareth H. Edmondson-Jones; took a European vacation; to avoid the disruption of the Republican National Convention).  
Figure 7: Annotation guidelines of relation identification shown to the annotators.