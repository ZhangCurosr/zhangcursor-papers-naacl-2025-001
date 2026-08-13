# LLaSA: Large Language and Structured Data Assistant

Yao Xu<sup>1,2</sup>, Shizhu He<sup>1,2</sup>, Jiabei Chen<sup>1,2</sup>, Xiangrong Zeng<sup>3</sup>,

Bingning Wang<sup>3</sup>, Jun Zhao<sup>1,2</sup>, Kang Liu<sup>1,2,4\*</sup>

<sup>1</sup> The Laboratory of Cognition and Decision Intelligence for Complex Systems,

Institute of Automation, Chinese Academy of Sciences, Beijing, China

<sup>2</sup> School of Artificial Intelligence, University of Chinese Academy of Sciences, Beijing, China <sup>3</sup> Baichuan Inc, Beijing, China

<sup>4</sup> Shanghai Artificial Intelligence Laboratory, Shanghai, China {yao.xu, shizhu.he, jzhao, kliu}@nlpr.ia.ac.cn, chenjiabei2024@ia.ac.cn

## Abstract

Structured data, such as tables, graphs, and databases, play a critical role in plentiful NLP tasks such as question answering and dialogue system. Recently, inspired by Vision-Language Models, Graph Neutral Networks (GNNs) have been introduced as an additional modality into the input of Large Language Models (LLMs) to improve their performance on Structured Knowledge Grounding (SKG) tasks. However, those GNN-enhanced LLMs have the following limitations: (1) They employ diverse GNNs to model varying types of structured data, rendering them unable to uniformly process vari ous forms of structured data. (2) The pretraining of GNNs is coupled with specific LLMs, which prevents GNNs from fully aligning with the textual space and limits their adaptability to other LLMs. To address these issues, we propose Large Language and Structured Data Assistant (LLaSA), a general framework for enhancing LLMs’ ability to handle structured data. Specifically, we represent various types of structured data in a unified hypergraph format, and use self-supervised learning to pretrain a hypergraph encoder, and a G-Former compressing encoded hypergraph representations with cross-attention. The compressed hypergraph representations are appended to the serialized inputs during training and inference stages of LLMs. Experimental results on multiple SKG tasks show that our pretrained hypergraph encoder can adapt to various LLMs and enhance their ability to process different types of structured data. Besides, LLaSA, with LoRA finetuning, outperforms previous SOTA method using full parameters tuning.

## 1 Introduction

Structured data, such as tables, knowledge graphs, and databases, is prevalent in real-world applications and plays a crucial role in fields like finance, healthcare, and data analytics. Therefore, Structured Knowledge Grounding (SKG) (Xie et al., 2022) has attracted significant research interest and has been widely studied. SKG tasks, such as question answering (Pasupat and Liang, 2015; Nan et al., 2022; Talmor and Berant, 2018), summarization (Nan et al., 2020; Parikh et al., 2020), fact verification (Chen et al., 2019), utilizing corresponding structured data as input and produce different outputs depending on the task types.

![](images/1621ea05246aa1c0f825ff9867de92674a05f4ce0c2e05f7abf299c8b8ed33ee.jpg)  
Figure 1: Overview of LLaSA, which can handle various types of structured data by transforming them into a unified format and encoding them with a universal encoder. The serialized structured data and the graph representations are then used as input to the LLM.

In recent years, with the rapid development of Large Language Models (LLMs) (Bang et al., 2023; Zhao et al., 2023), researchers have shifted their focus from building task-specific models for different tasks (Xie et al., 2022) to developing a generalist model capable of handling a variety of SKG tasks (Zhuang et al., 2024; Zhang et al., 2024b). These approaches that leverage LLMs for SKG tasks commonly serialize structured data (e.g., representing tables in markdown format) as pure textual input to the LLMs. However, this method can lead to the partial loss of structured information, as all these LLMs are decoder-only Transformer models (Vaswani et al., 2017) (e.g., in the table data, cells from the same column or rows in the original table may become distant from each other after linear serialization).

![](images/4932ce57dac2300dce83e0db052cf5224c263dfe3a26764c25c85e0512255826.jpg)

(a) LLM-based GNN Pretraining  
![](images/31e0f0863c8ceb23ef7db5433d3c332789e9264e01d8771a0fb958170190b1b5.jpg)  
(b) G-Former-based GNN Pretraining  
Figure 2: Comparison between LLM-based and the proposed G-Former-based GNN pretraining strategies.

Recently, to enhance the utilization of large language models in the visual domain, researchers have crafted Vision-Language Models (VLM) (Zhang et al., 2024a) that transform image data into discrete language tokens via a learnable interface. Inspired by the success of VLM, another line of research introduces GNNs as an additional modality into the input of LLMs. For example, G-Retriever (He et al., 2024) combines GNNs encoding knowledge graphs with LLMs, enhancing the graph-based question-answering abilities of the LLMs. HGT (Jin et al., 2024) propose a heterogeneous graph enhanced large language model for table-based question answering. However, they employ diverse networks to model varying types of structured data, rendering them unable to uniformly process various forms of structured data, for instance, G-Retriever and HGT can only handle graphs and tables, respectively. Besides, the pretraining of GNNs is coupled with specific LLMs in these methods, for instance, HGT pretrains a GNN based on a frozen LLM by self-supervised learning, as shown in Figure 2 (a). This prevents the GNN from fully aligning with the textual embedding space because the serialized table is already included as input during the pretraining process, making the GNN unnecessary in this situation. As a result, it is unclear whether the GNN effectively encodes the table data as expected during pretraining, and the adaptability of this GNN to other LLMs also remains a question.

Aiming to address these drawbacks, we introduce Large Language and Structured Data Assistant (LLaSA) for SKG tasks. Specifically, we first model various forms of structured data, such as tables and knowledge graphs, uniformly as hypergraphs (Chen et al., 2023a), enabling the use of a unified GNN for encoding. Specifically, We treat the cells in a table as nodes, with rows and columns as hyperedges, and for graphs, we treat entities as nodes and relationships as hyperedges. We then pretrain a GNN and a G-Former (a cross attention model similar to Q-Former (Li et al., 2023), but it extracts features from GNN) with self-supervised learning which includes question answering and Graph-Text Matching, as illustrated in Figure 2 (b). This pretraining approach not only aligns the GNN with the text more effectively but also avoids coupling with a specific LLM, making it adaptable to various LLMs. During fine-tuning for downstream tasks, we use the G-Former to bridge the modality gap, transforming the encoded structured data into a fixed number of soft tokens that can be understood by LLMs, as shown in Figure 1.

Results on multiple SKG datasets, including table, knowledge graph and database, demonstrate that the proposed LLaSA significantly enhances LLM’s ability to handle these structured data. With the frozen LLM, LLaSA Llama-7B achieves an average improvement of 12% across ten datasets. With LoRA the tuned LLM, it still yields an average improvement of 0.4%. Besides, LLaSA, with LoRA fine-tuning, outperforms previous SOTA method using full parameters tuning. The codes and data are available at https://github.com/ YaooXu/LLaSA.

The main contributions of this paper can be summarized as follows:

1. We propose LLaSA, a framework that integrates the encoded representations of structured data as an additional modality into the input of LLMs.

2. We represent various forms of structured data as hypergraphs, enabling unified encoding through a single GNN, and pretrain the GNN and G-Former with self-supervised learning.

3. Experimental results demonstrate that our pretrained GNN can be adapted to various LLMs, enhancing their ability to handle structured data. Furthermore, ablation studies confirm the importance of both the GNN and the pretraining process.

## 2 Related Work

## 2.1 Models for SKG tasks

SKG data, such as graphs and tables, exhibit heterogeneous data formats, leading to a line of research focuses on modeling these heterogeneous representations during encoding structured data. For example, TaBERT (Yin et al., 2020) introduces vertical self-attention, a self-attention mechanism that processes vertically aligned vectors across different rows. TAPAS (Herzig et al., 2020) captures tabular structure with additional embeddings, such as Column / Row ID, based on BERT’s architecture (Devlin et al., 2019). HyTrel (Chen et al., 2023a) converts a table into a hypergraph to allow the GNN to incorporate row/column permutation invariances. All these methods can also be used in LLaSA, and we use HyTrel as our default hypergraph encoder in this work.

USKG (Xie et al., 2022) is the first work that unifies multiple SKG tasks into a text-to-text format. However, their results show that multi-task finetuning is worse than single-task finetuning on many tasks. Following USKG, StructLM (Zhuang et al., 2024) finetunes LLMs on multiple SKG tasks and show strong zero-shot generalization capability on unseen SKG tasks. TableLlama (Zhang et al., 2024b) finetunes LLMs with LongLoRA (Chen et al., 2023b) on multiple table-based datasets to build a generalist model. However, these methods all serialize structured data and could lead to the partial loss of structured information.

## 2.2 Combine LLMs and GNN

There are many works that combine LMs and GNNs (Malaviya et al., 2020; Yasunaga et al., 2022; Zhang et al., 2022; Zhao et al., 2022). In the era of LLMs, researchers are increasingly focused on how to convert GNN representations into tokens that LLMs can understand, thereby avoiding modifications to the model architecture and minimizing the impact on other capabilities. LLaGA (Chen et al., 2024) reorganizes graph nodes to structureaware sequences and then mapping these into the token embedding space through a projector. G-Retriever (He et al., 2024) uses a standard Graph Attention Network (GAT) (Velickoviˇ c et al.´ , 2017) to encode the retrieved graphs and treats graph embeddings as soft prompting, but it doesn’t involve pretraining stage. GraphGPT (Tang et al., 2023) not only appends the representations of graph nodes to the textual input, but also employs self-supervised training to align the encoding of graph structures with the natural language space. Another similar work is HGT (Jin et al., 2024), which introduces an GNN to encode the heterogeneous graph converted by the corresponding table. Both GraphGPT and HGT need to pretrain a GNN or a adapter based on a frozen LLM by self-supervised learning before task-specific instruction tuning. In contrast, our LLaSA pretrain a general GNN and G-Former that are decoupled from the specific LLM, allowing them to be used with any LLM without the need for re-pretraining, which is time-consuming.

![](images/d1ff992008f4f64f1070b10eef5f2125a070de72d95d6b795c6d22db856f0558.jpg)  
(a) Table to Hypergraph

![](images/6360a858d0c25b35c0b1b27ddfff60230b866e8ecc0e090fa4e225a0ffd3bcef.jpg)  
(b) Knowledge graph to Hypergraph  
Figure 3: Examples of converting structured data to a unified hypergraph format, where yellow nodes represent hyperedges. In Figure (a), the arrows are omitted as the edges in the hypergraph are bidirectional.

## 3 Method

## 3.1 Hypergraph Construction

We represent a hypergraph as $\mathcal { G } = \{ \nu , \varepsilon \}$ , where and  denote the set of nodes and hyperedges. A hypergraph can be regarded as a type of bipartite graph, that is, every edge connects a node in  to one in .

Table to hypergraph. We represent a table as $\mathcal { T } =$ $\{ \mathcal { H } , \mathcal { R } \}$ , where $\mathcal { H } = [ h _ { 1 } , h _ { 2 } , . . . , h _ { n } ]$ represents n column headers, $\mathcal { R } = [ r _ { 1 } , r _ { 2 } , . . . , r _ { m } ]$ represents m rows, and each row $r _ { i } = [ c _ { i 1 } , c _ { i 2 } , . . . , c _ { i n } ]$ has n cells. We treat each cell $c _ { i j }$ as node $v _ { i j } \in \mathcal V$ , and each row $r _ { i } ,$ each column header $h _ { j }$ as hyperedges $e _ { i } , e _ { j } \in \mathcal { E }$ . Each node $v _ { i j }$ is only connected to its corresponding hyperedges $e _ { i }$ and $e _ { j }$ , as shown in Figure 3 (a).

![](images/4a3fd9e27be2ca602f89716e6e381d620990d49c15818fc07ca44be9e520f05f.jpg)  
(a) Pretraining G-Former and Hypergraph encoder on template questions

![](images/b614dc84f8e70cc07ce37450a5a023daa1aa7a70d0dea9a202a86e17bfaa4769.jpg)  
(b) Finetuning LLM on SKG tasks  
Figure 4: (a) We employ two pretraining objectives to train the hypergraph encoder and G-Former, with the upper left corner showing the attention masks used for different pretraining tasks. (b) In the LLM finetuning stage, we only use the graph transformer to extract a fixed number of representations of hypergraph, and treat them as soft prompts in LLM’s input.

Graph to hypergraph. In this work, our research focuses on Text-Attributed Graphs (TAG), which are graphs enriched with textual information associated with their nodes or edges. We represent a TAG as a set of factual triples, i.e., $G = \{ ( h , r , t ) \}$ where $h , r , t$ are texts, and $h ,$ t denote the head and tail entity, r denotes the relation of them. We treat $h$ and t as nodes $v _ { h } , v _ { t } \in \mathcal { V }$ , r as hyperedge $e _ { r } \in \mathcal { E } _ { \mathfrak { r } }$ and $v _ { h } , v _ { t }$ are connected to $e _ { r }$ . Besides, to preserve the directional information in the original graph, we create a reverse relation node for each relation node, as shown in Figure 3 (b) (e.g., the texts of normal/reverse relation node are Relation: team and Reverse Relation: team, respectively).

## 3.2 Model Architecture

## 3.2.1 Hypergraph Encoder

Following HyTrel (Chen et al., 2023a), we utilize a HyperTrans, which is a structure-aware transformer module, to encode the hypergraphs. Each layer of HyperTrans contains two attention module: Node2Hyperedge and Hyperedge2Node, and a Hyperedge Fusion module. The initial representation of nodes are obtained by sentence bert (Reimers, 2019).

The Node2Hyperedge attention module aggregates information to hyperedge e from its neighbor nodes $v \in \mathcal { N } _ { e }$ . This process is defined as follows:

$$
\tilde { \mathbf { h } } _ { e } ^ { l + 1 } = f _ { \mathcal { V }  \mathcal { E } } ( K _ { e } ^ { l } )\tag{1}
$$

where $f _ { \mathcal { V } \to \mathcal { E } }$ is a attention function, $K _ { e } ^ { l } = \{ \mathbf { h } _ { v } ^ { l } | v \in$ $\mathcal { N } _ { e } \}$ represents the set of representations at layer l of all nodes connected to the hypernode $e$ .

The Hyperedge Fusion module is a Multilayer Perceptron (MLP) that integrates the information collected from both the neighbors of hypernode e and itself. This process is defined as follows:

$$
{ \mathbf { h } } _ { e } ^ { l + 1 } = \mathrm { M L P } ( { \mathbf { h } } _ { e } ^ { l } ; \tilde { { \mathbf { h } } } _ { e } ^ { l + 1 } )\tag{2}
$$

The Hyperedge2Node attention module then aggregates information to node v from its neighbor hypernodes $e \in \mathcal { N } _ { v }$

$$
\tilde { \mathbf { h } } _ { v } ^ { l + 1 } = f \varepsilon \to \nu ( K _ { v } ^ { l } )\tag{3}
$$

where $f _ { \mathcal { E }  \mathcal { V } }$ is another attention function, $K _ { v } ^ { l } =$ $\{ \mathbf { h } _ { e } ^ { l } | e \in \mathcal { N } _ { v } \}$ represents the set of representations at layer l of all hyernodes connected to the node e.

The attention function $f$ used in the equation (1, 3) is similar to transformer (Vaswani et al., 2017), the function f is defined as follows:

$$
f _ { \mathcal { V }  \mathcal { E } o r \mathcal { E }  \mathcal { V } } ( \mathbf { X } ) = \mathrm { L N } ( \mathbf { Y } + \mathrm { F F N } ( \mathbf { Y } ) )\tag{4}
$$

$$
\mathbf { Y } = \operatorname { L N } ( \omega + \operatorname { S e t M H A } ( \omega , \mathbf { X } , \mathbf { X } ) )\tag{5}
$$

where X is the representations of input nodes or hyperedges. Y is the intermediate representations. SetMHA is the multi-head set attention mechanism defined as follows (for simplicity, we only consider single-head self-attention here):

$$
\operatorname { S e t M H A } ( \omega , \mathbf { X } , \mathbf { X } ) = \operatorname { S o f t m a x } ( \omega ( \mathbf { X } \mathbf { W } ^ { K } ) ^ { T } ( \mathbf { X } \mathbf { W } ^ { V } ) )\tag{6}
$$

where ω is a learnable query vector, $\mathbf { W } ^ { K }$ and $\mathbf { W } ^ { V }$ are the key and value matrices.

In summary, HyperTrans first updates the representation of a hypernode based on the neighboring nodes, and then updates the neighboring nodes using the updated representation of the hypernode.

## 3.2.2 G-Former

To bridge the gap between a hypergraph encoder and text, and to compress the hypergraph node representations into fixed-length tokens, we propose G-Former based on Q-Former (Li et al., 2023). As demonstrated in Figure 4 (a), our G-Former consists of two transformer sub-modules: (1) A graph transformer which interacts with hypergraph representations. (2) A text transformer that encodes and generates text. The graph transformer uses a fixed number of learnable query tokens, which first interact with each other through self-attention, then interact with hypergraph nodes representation through cross-attention. During the pretraining stage, we use different attention mechanisms based on the specific pretraining tasks to control the interaction between query and text tokens.

## 3.3 Training

## 3.3.1 Pretraining

Even though HyTrel (Chen et al., 2023a) also trains an encoder for table encoding, its pretraining tasks, such as column type classification and table similarity prediction, does not truly align the hypergraph encoder space with the textual space. Similar to Q-Former (Li et al., 2023), we also introduce two tasks to effectively align these two spaces, and their attention mechanisms are shown in the upper left corner of Figure 4. The details of constructing pretraining dataset can be founded in Appendix A. Graph-Depended Answer Generation. This tasks trains the G-Former to generate answers, given input tables as the condition. The information required for generating the answer is first extracted by the query tokens with cross-attention, and then passed to the text transformer through self-attention. The graph transformer learns to compress all the graph node representations into a fixed number of query tokens. The multimodal causal attention allow query tokens to interact with each others but not the text tokens while each text token can interact with all query tokens and its previous text tokens.

Graph-Text Matching. Since some answers can be easily deduced even without any structural information, for example, the column name for "Miami Heat" is likely to be "Team". Therefore, we also introduce Graph-Text Matching. We employ a bidirectional self-attention mask, allowing all queries and text tokens to attend to each other. As a result, the output query embeddings, denoted as $Q ,$ integrate multimodal information effectively. Each query embedding is then passed through a MLP to generate a corresponding logit. Finally, we compute the overall matching score by averaging the logits across all queries.

## 3.3.2 Task-specific Instruction Tuning

In the instruction tuning stage, we use multiple SKG tasks to finetune LLMs through Parameter-Efficient Fine-Tuning (Ding et al., 2023). In this stage, we only use the graph transformer module pretrained in the pretraining stage, that is, we extract fixed-length query embeddings $\hat { \mathbf { q } }$ from the node representations of hypergraph ${ \mathcal { G } } .$ . Then the extracted query embeddings qˆ are projected into the same dimension as the text embedding of the LLM through a fully connected layer. This process

<table><tr><td colspan="10"></td><td colspan="5">Held Out</td></tr><tr><td>Dataset</td><td>WikiTQ</td><td></td><td>HybridQA FeTaQA</td><td>TabMWP</td><td>Held In WikiSQL</td><td>TabFact</td><td>ToTTo</td><td>KVRet</td><td>CWQ</td><td>DART Avg</td><td>SQA</td><td>WTT</td><td>FinQA</td><td>Avg</td></tr><tr><td>Metric</td><td>Ex</td><td>Acc</td><td>BLEU</td><td>Acc</td><td>Ex</td><td>Acc BLEU</td><td>Micro</td><td>Acc</td><td>BLEU</td><td></td><td>Acc</td><td>BLEU</td><td>Acc</td><td></td></tr><tr><td></td><td colspan="10">1-shot Learning</td><td colspan="3"></td></tr><tr><td>Mistral 7B Instruct</td><td>20.0</td><td>22.9</td><td>8.9</td><td>30.5</td><td>24.6 54.8</td><td>16.8</td><td>54.0</td><td>34.5</td><td>43.5</td><td>31.1</td><td>3.2</td><td>3.8</td><td>6.7</td><td>4.6</td></tr><tr><td>ChatGPT 3.5</td><td>42.6</td><td>38.4</td><td>15.1</td><td>52.9</td><td>50.4 53.0</td><td>22.4</td><td>53.0</td><td>50.1</td><td>57.0</td><td>43.5</td><td>9.7</td><td>4.0</td><td>12.2</td><td>8.6</td></tr><tr><td>ChatGPT 4</td><td>60.8</td><td>50.8</td><td>8.4</td><td>72.6</td><td>35.8 79.0</td><td>21.4</td><td>60.3</td><td>66.6</td><td>53.7</td><td>50.9</td><td>5.4</td><td>3.1</td><td>18.0</td><td>8.8</td></tr><tr><td colspan="14">Full Parameters Tuning</td></tr><tr><td>USKG 3B × N</td><td>49.3</td><td>59.2</td><td>36.0</td><td></td><td></td><td>49.0</td><td></td><td>73.3</td><td>46.7</td><td></td><td>0</td><td>0</td><td></td><td>0</td></tr><tr><td>FLAN-UL2 20B</td><td>54.6</td><td>61.0</td><td>35.8</td><td></td><td>86.0</td><td>80.8</td><td></td><td>67.9</td><td></td><td></td><td></td><td></td><td>0</td><td></td></tr><tr><td>StructLM 7B-M</td><td>56.8</td><td>62.6</td><td>37.5</td><td>73.5</td><td>87.3 87.0</td><td>87.1 84.6</td><td>49.8</td><td>72.2</td><td>75.9 79.9</td><td>50.4 63.2 66.7</td><td>70.1† 41.9</td><td>19.4† 16.7</td><td>5.9† 24.6</td><td>31.8† 27.7</td></tr><tr><td colspan="9"></td><td colspan="3"></td></tr><tr><td></td><td>35.0*</td><td>39.4*</td><td>39.0</td><td></td><td>50.5*</td><td>Lora Tuning 82.5</td><td>20.8*</td><td>48.7*</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TableLlama 7B Mistral 7B Instruct</td><td>56.9</td><td>62.4</td><td>36.7</td><td></td><td>86.8</td><td>84.1</td><td>49.0</td><td>71.2</td><td></td><td>- 66.7</td><td>2.6 47.1</td><td>3.0</td><td>1.4</td><td>2.3</td></tr><tr><td>HGT 7B-M</td><td>57.2</td><td>62.4</td><td></td><td>76.9</td><td></td><td>83.8 49.2</td><td></td><td>78.1 78.3</td><td>65.0 65.4</td><td>66.7</td><td>51.0</td><td>10.7 8.2</td><td>14.2</td><td>24.0 25.7</td></tr><tr><td>G-Retrieve 7B-M</td><td>57.4</td><td>62.6</td><td>36.6</td><td>75.8</td><td>87.0</td><td>84.2 48.9</td><td></td><td>70.8</td><td>78.4 65.2</td><td>66.7</td><td>50.9</td><td>12.5</td><td>18.0 16.6</td><td>26.7</td></tr><tr><td>LlaSA 7B-M (Ours)</td><td>56.2</td><td>62.9</td><td>36.5</td><td>76.0</td><td>86.6</td><td>84.3</td><td>49.0</td><td>71.6 72.3</td><td>78.4 64.7</td><td>66.9</td><td>51.3</td><td>16.3</td><td>13.9</td><td></td></tr><tr><td></td><td></td><td></td><td>37.0</td><td>76.7</td><td>87.1</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>27.2</td></tr></table>

Table 1: The evaluation results of our model against other baselines. Cells with "\*" represent that the model did not train on this dataset. Cells in the held-out section with $" \dagger "$ are held-in results. 7B-M represents using Mistral-7B-Instruct-v0.2 as the base model. The results of HGT 7B-M and G-Retrieve 7B-M were re-implemented by us. The results of StructLM 7B-M are from their paper. USKG 3B  N indicates training a 3B model for each task. The boldface indicates the best result.

is defined as follows:

$$
\hat { \mathbf { q } } = \mathrm { F C } ( f _ { g } ( \mathbf { q } , \mathbf { X } ) )\tag{7}
$$

where $\mathbf { X } \in \mathcal { R } ^ { n \times d _ { 1 } }$ is the hypergraph node embeddings, n is the number of nodes in the graph, $d _ { 1 }$ is the dimension of node embeddings, $\mathbf { q } \in \mathcal { R } ^ { b \times d _ { 1 } }$ is the original query embeddings, $\hat { \mathbf { q } } \in \mathcal { R } ^ { m \times d _ { l } }$ is the extracted query embeddings, m is the number of query tokens, $d _ { l }$ is the dimension of the LLM’s text embeddings, $f _ { g }$ and $F C$ represents G-Former and fully connected layer.

These projected query embeddings are treated as soft prompts and appended to the text embeddings. The LLM learns to predict answers based on these text embeddings and soft prompts. This process is defined as follows:

$$
{ \mathbf { h _ { t } } } = \mathrm { T e x t E m b e d d e r } ( [ \mathrm { s e r i a l i z e } ( \mathcal { G } ) ; x _ { q } ] )\tag{8}
$$

$$
p _ { \theta } ( Y | \mathcal { G } , x _ { q } ) = \prod _ { i = 1 } ^ { r } p _ { \theta } ( y _ { i } | y _ { \le i } , [ \mathbf { h _ { t } } ; \hat { \mathbf { q } } ] )\tag{9}
$$

where θ is the LLM’s parameters, serialize denotes function that serializes structured data to text sequence, [;] represents concatenation operation, $x _ { q }$ and $Y$ represents the question and answer, respectively.

## 4 Experiment

## 4.1 Datasets

To validate the effectiveness of our approach, we collected 10 SKG tasks as our training data, which can be categorized into the following four types: (1) Structured Data Question Answering: This task requires the LLM to answer questions based on the given tables, knowledge graphs, and textual information. The datasets for this category include WikiTQ (Pasupat and Liang, 2015), CompWebQ (Talmor and Berant, 2018), and TabMWP (Lu et al., 2022). (2) Fact Verification: This task requires the LLM to determine whether a given statement is entailed or refuted based on the information in the table. The corresponding dataset is TabFact (Chen et al., 2019). (3) Structured Data to Text: This task requires the LLM to summarize or describe the content of a given table or knowledge graph in one or two sentences. The relevant datasets for this category include ToTTo (Parikh et al., 2020) and DART (Nan et al., 2020).

To evaluate the generalization ability of our method, we use SQA(Iyyer et al., 2017), WikiTableText (Bao et al., 2018) and FinQA (Chen et al., 2021) as held-out datasets, where SQA belongs to table-based question, WikiTableText belongs to structured data to Text and FinQA requires generating python-executable math expression based on the given questions and tables.

Statistics of these datasets can be found in Appendix B.

## 4.2 Baselines

In this work, we compare LLaSA with other LLMs based methods. We primarily select StructLM (Zhuang et al., 2024), which performs full parameters fine-tuning on various SKG datasets, as the main baseline. It is important to note that StructLM utilizes a broader range of datasets such as SQL2Text (Shu et al., 2021). These datasets are excluded from LLaSA’s training set because their inputs could not be transformed into hypergraphs. We also compare with TableLLama (Zhang et al., 2024b), which not only leverages a broader range of foundational table tasks, such as Column Type Annotation and Entity Linking, but also uses a longer 8K context length to finetune the LLMs. As HGT(Jin et al., 2024) and G-Retrieve(He et al., 2024) use different models and training datasets, we re-implement and train them under our framework for a fairer comparison, where HGT concatenates the representations of all GNN nodes to the LLM input, G-Retrieve takes the average of all GNN node representations and concatenate it to the LLM input. Additionally, we also evaluate the performance of GPT-3.5, GPT-4, and Mistral-7B-Instruct-v0.2 (Jiang et al., 2023) under a 1-shot setting.

<table><tr><td colspan="10"></td><td colspan="5">Held Out</td></tr><tr><td>dataset</td><td>WikiTQ</td><td>HybridQA FeTaQA</td><td></td><td>TabMWP</td><td>Held In WikiSQL</td><td>TabFact</td><td>ToTTo</td><td>KVRet</td><td>CWQ</td><td>DART</td><td>Avg</td><td>SQA WTT</td><td>FinQA</td><td></td><td>Avg</td></tr><tr><td>metric</td><td>Ex</td><td>Acc</td><td>BLEU</td><td>Acc</td><td>Ex</td><td>Acc</td><td>BLEU</td><td>Micro</td><td>Acc</td><td>BLEU</td><td></td><td>Acc</td><td>BLEU</td><td>Acc</td><td></td></tr><tr><td></td><td colspan="15">Freeze LLM</td></tr><tr><td>Phi 3B</td><td>31.2</td><td>39.0</td><td>13.6</td><td>41.9</td><td>47.1 59.9</td><td>32.4</td><td>46.3</td><td>45.1</td><td>58.0</td><td>41.5</td><td>14.8</td><td>8.1</td><td>9.2</td><td>10.7</td></tr><tr><td>LlaSA-Phi 3B</td><td>35.1</td><td>49.1</td><td>26.6</td><td>62.5</td><td>65.4</td><td>70.3</td><td>39.9</td><td>62.1</td><td>60.2</td><td>59.7</td><td>53.1</td><td>21.8 11.7</td><td>13.7</td><td>15.7</td></tr><tr><td>Llama2 7B</td><td>27.2</td><td>42.3</td><td>8.9</td><td>27.3</td><td>45.0</td><td>50.8</td><td>32.7</td><td>46.8 52.2</td><td>51.6</td><td>38.5</td><td>11.0</td><td>7.7</td><td>2.1</td><td>6.9</td></tr><tr><td>LlaSA-Llama2 7B</td><td>32.5</td><td>50.2</td><td>26.6</td><td>48.5</td><td>62.9</td><td>66.2 40.6</td><td>62.0</td><td>62.1</td><td>58.3</td><td>51.0</td><td>19.9</td><td>17.5</td><td>3.5</td><td>13.6</td></tr><tr><td>Mistral 7B</td><td>34.1</td><td>44.2</td><td>5.3</td><td>37.9</td><td>55.3</td><td>59.4 35.2</td><td></td><td>43.7 55.4</td><td>58.1</td><td>42.9</td><td>26.7</td><td>14.8</td><td>12.1</td><td>17.9</td></tr><tr><td>LlaSA-Mistral 7B</td><td>38.4</td><td>50.6</td><td>27.3</td><td>59.9</td><td>70.1</td><td>73.6</td><td>42.4</td><td>65.7 67.1</td><td>59.2</td><td>55.4</td><td>25.9</td><td>6.0</td><td>7.4</td><td>13.1</td></tr><tr><td>Llama3 8B</td><td>41.2</td><td>50.1</td><td>20.3</td><td>52.7</td><td>67.0</td><td>66.1</td><td>38.2</td><td>53.2</td><td>62.4 59.3</td><td>51.1</td><td>31.2</td><td>13.0</td><td>17.0</td><td>20.4</td></tr><tr><td>LlaSA-Llama3 8B</td><td>45.9</td><td>53.8</td><td>29.9</td><td>70.5</td><td>74.8</td><td>78.2</td><td>43.1</td><td>64.3</td><td>69.6 60.4</td><td>59.1</td><td>29.0</td><td>12.2</td><td>23.1</td><td>21.4</td></tr><tr><td colspan="14">Lora Tuning LLM</td></tr><tr><td>Phi 3B</td><td>45.8</td><td>53.6</td><td>30.7</td><td>70.0</td><td>80.2</td><td>75.5</td><td>42.6</td><td>62.5 68.3</td><td>62.9</td><td>59.2</td><td>34.3</td><td>9.5</td><td>11.4</td><td>18.4</td></tr><tr><td>LlaSA-Phi 3B</td><td>47.4</td><td>55.4</td><td>31.6</td><td>72.4</td><td>81.5</td><td>77.5</td><td>44.5</td><td>67.8</td><td>70.8</td><td>62.0</td><td>61.1</td><td>45.8 13.0</td><td>7.1</td><td>22.0</td></tr><tr><td>Llama2 7B</td><td>45.0</td><td>59.5</td><td>32.5</td><td>62.8</td><td>82.9</td><td>78.1</td><td>46.3</td><td>67.1</td><td>75.0 63.8</td><td>61.3</td><td>35.3</td><td>8.6</td><td>6.5</td><td>16.8</td></tr><tr><td>LlaSA-Llama2 7B</td><td>45.9</td><td>60.0</td><td>33.0</td><td>64.2</td><td>83.0</td><td>78.6</td><td>47.0</td><td>66.6 76.0</td><td>63.1</td><td>61.7</td><td>35.5</td><td>8.6</td><td>9.7</td><td>17.9</td></tr><tr><td>Mistral 7B</td><td>56.9</td><td>62.4</td><td>36.7</td><td>76.9</td><td>86.8</td><td>84.1</td><td>49.0</td><td>71.2</td><td>78.1 65.0</td><td>66.7</td><td>47.1</td><td>10.7</td><td>14.2</td><td>24.0</td></tr><tr><td>LlaSA-Mistral 7B</td><td>56.2</td><td>62.9</td><td>37.0</td><td>76.7</td><td>87.1</td><td>84.3</td><td>49.0</td><td>72.3</td><td>78.4 64.7</td><td>66.9</td><td>51.3</td><td>16.3</td><td>13.9</td><td>27.2</td></tr><tr><td>Llama3 8B</td><td>59.4</td><td>62.8</td><td>34.1</td><td>77.0</td><td>86.2</td><td>85.8</td><td>47.7</td><td>69.2</td><td>78.9 64.0</td><td>66.5</td><td>47.7</td><td>11.7</td><td>21.5</td><td>27.0</td></tr><tr><td>LlaSA-Llama3 8B</td><td>60.4</td><td>63.0</td><td>34.7</td><td>77.8</td><td>86.3</td><td>86.1</td><td>48.0</td><td>69.2</td><td>79.0 63.1</td><td>66.8</td><td>52.6</td><td>10.8</td><td>22.8</td><td>28.7</td></tr></table>

Table 2: The evaluation results of LLaSA with different base models under different finetuning strategies. The soft tokens in prompt tuning (Freeze LLM) is set 10, which is the same as the number of query tokens in G-Former. The lora rank is set to 32 in lora tuning.

## 4.3 Implement Details

We choose Phi-3B (Marah Abdin, 2024), LLama2- 7B (Hugo Touvron, 2023), Mistral-7B (Jiang et al., 2023) and LLama3-8B (Abhimanyu Dubey, 2024) as our base models. We use a learning rate of 2e-5 with a 3% warm-up cosine scheduler, set the batch size as 3, epoch as 3. The default lora rank is set to 32. All this models are trained on 8 H800 80G using DeepSpeed ZeRO-2 (Aminabadi et al., 2022). A training on a 7B model takes about 12 hours. The maximum sequence length is set to 2048 during training, the maximum generation length to set as 1024 during inference. We set the dimension of the hypergraph encoder to 768, with 12 layers, and use RoBERTa-base as the initial parameters for the G-Former. The total number of parameters for both components is 400M. The G-Former and GNN are pretrained on 25M tables for one epoch, and these data are taken from TaBERT (Yin et al., 2020).

## 4.4 Main results

Table 1 presents the results of our LLaSA compared to previous baselines across 10 datasets. From the table, we can see that GPT-3.5 and GPT-4 still fall short in handling SKG tasks, trailing behind LLaSA 7B-M by 23.3% and 15.9% points, respectively, across the ten tasks. Moreover, our LLaSA 7B-M achieves state-of-the-art (SOTA) performance in 4 out of 10 tasks within the LLM-based method.

It can be found that HGT 7B-M and G-Retrieve 7B-M did not achieve much improvement compared to the naive LLM, which may be due to the following reasons: 1) The projector-based strategy introduces noise by feeding all node representations into the LLM. In real-world structured data question-answering scenarios, many cells are irrelevant to the current question and may distract the LLM. In contrast, our G-Former compresses all graph node representations into a fixed-length token sequence, retaining only the most relevant information. 2) The projector-based strategy fails to fully leverage the alignment objectives in pretraining. During pretraining, we aligned the Q-Former, which acts as a bridge between GNN representations and the textual space, rather than directly aligning the GNN itself. Consequently, even pretrained GNNs cannot directly enhance the final performance of the LLM.

From the perspective of model parameters, we find that the performance of LoRA-tuned Mistral 7B closely approaches that of the fully fine-tuned StructLM 7B-M. When using LLaSA framework, the LoRA-tuned Mistral 7B can surpass StructLM 7B-M, despite the former only requiring 400M trainable parameters compared to the latter’s 7B parameters, and outperforms StructLM 7B-M on 6 tasks. Additionally, we observe that fully finetuning LLMs on SKG tasks may lead to a decline in their performance on other tasks, whereas LoRAtuned LLMs experience a smaller drop. One piece of evidence is the TabMWP dataset, which requires mathematical reasoning, where LLaSA 7B-M significantly outperforms StructLM 7B-M by 4.2%.

On the held-out data, although StructLM 7B achieves a higher average performance, our LLaSA significantly outperforms StructLM 7B on the SQA dataset.

## 4.5 LLaSA with Different Base Models

To verify the generality of our pretrained hypergraph encoder and G-Former, we evaluate LLaSA under different base models with different finetuning strategies (Prompt Tuning and Lora Tuning), the results are demonstrated in Table 2.

As shown in the table, the pretrained hypergraph encoder and G-Former enhance the models’ ability to handle SKG tasks and improve their generalization to unseen datasets across most base models (except for the prompt-tuned LLaSA-Mistral 7B, which shows a performance drop on the held-out data). Especially under the Freeze LLM setting, LLaSA achieves significant improvements compared to basic prompt tuning. Specifically, it delivers an approximate 10% performance boost across Phi-3B, Llama2-7B, Mistral-7B, and Llama3-8B models. This indicates that our pre-trained hypergraph encoder and G-Former can be effectively adapted to various LLMs, enhancing their ability to handle structured data.

![](images/3eccd4747839de17c429839ff4b2b60dcbabf5b7ae5a30d84e01c75bcc170e1e.jpg)  
Figure 5: The average performance of models using different pretrained Hypergraph Encoder (GNN).

Under the LoRA tuning LLM setting, although LLaSA achieves smaller improvements on the heldin datasets, it consistently enhances model performance on held-out datasets. This suggests that our approach genuinely improves the model’s ability of handling structured data rather than merely overfitting to the training data. Additionally, we observed that LLaSA’s 0.9% performance improvement on Phi-3B is notably greater than the 0.3% improvement on Llama3-8B. This difference may be attributed to Phi-3B’s inherently weaker ability to process structured data, making the introduction of the hypergraph encoder more impactful in enhancing its performance.

## 4.6 Comparison Between Different Pretraining Strategies

We compared two strategies for pre-training the GNN: (1) Llama based pretraining, which use question answering task to pretrain a GNN based on a frozen LLM, as shown in Figure 2 (a). (2) G-Former-based pretraining, using both answer prediction and Graph-Text Matching tasks, as shown in Figure 2 (b). We select checkpoints with the same training time for both strategies and conduct experiments under the frozen LLM setting, the results are shown in Figure 5. The experimental results indicate that the GNN pretrained with G-Former exhibits superior adaptability, showing greater improvements across various models compared to the GNN pretrained with Llama. The main reasons are as follows: (1) In the Llama based pretraining, the soft tokens are not essential since serialized text is already included in the input, which prevents the GNN from fully aligning with the textual embedding space. As a result, it is unclear whether the GNN effectively encodes the table data as expected or just helps the LLMs fit the training data better. (2) The GNN pretrained with Llama primarily aligns with the Llama text space, limiting its adaptability to other models.

<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Avg-I</td><td rowspan=1 colspan=1>Avg-O</td><td rowspan=1 colspan=1>Avg</td></tr><tr><td rowspan=1 colspan=1>LLaSA Llama-7B</td><td rowspan=1 colspan=1>51.0</td><td rowspan=1 colspan=1>13.6</td><td rowspan=1 colspan=1>32.3</td></tr><tr><td rowspan=1 colspan=1>w/o pretraining</td><td rowspan=1 colspan=1>47.2</td><td rowspan=1 colspan=1>8.6</td><td rowspan=1 colspan=1>27.9</td></tr><tr><td rowspan=1 colspan=1>w/o GNN</td><td rowspan=1 colspan=1>42.2</td><td rowspan=1 colspan=1>7.3</td><td rowspan=1 colspan=1>24.8</td></tr><tr><td rowspan=2 colspan=1>w/o G-Formerprompt tuning</td><td rowspan=1 colspan=1>42.7</td><td rowspan=1 colspan=1>7.7</td><td rowspan=1 colspan=1>25.2</td></tr><tr><td rowspan=1 colspan=1>38.5</td><td rowspan=1 colspan=1>6.9</td><td rowspan=1 colspan=1>22.7</td></tr></table>

Table 3: Ablation results on LLaSA Llama-7B. Avg-I and Avg-O represent the average score of Held-In and Held-Out datasets. The "w/o pretraining": randomly initializing GNN and G-Former without pre-training. The "w/o GNN": ignoring cross attention in G-Former. The "w/o G-Former": ignoring the whole G-Former.

## 4.7 Ablation Study

Table 3 presents the results of our ablation study on LLaSA Llama2-7B under the frozen LLM setting. From the table, we can observe that compared to the randomly initialized GNN, the pretrained GNN helps the LLM achieve improvements of 3.8% on Held-In datasets and 5.0% on Held-Out datasets. This clearly demonstrates the effectiveness of the pretraining process.

In the "w/o GNN" and "w/o G-Former" settings, hypergraph information is ignored. The former directly passes the query token through multiple layers of self-attention, while the latter only applies a linear transformation via a fully connected layer. They can be viewed as more complex forms of prompt tuning. Although these two settings achieved a small 4% improvement on Held-In datasets compared to basic prompt tuning, they do not show significant gains on Held-Out datasets. This suggests that simple prompt tuning mainly helps the model fit the training data better, without truly enhancing its generalization capability.

## 5 Conclusion

In this work, we propose LLaSA, a framework that converts structured data into hypergraphs and integrates the hypergraphs representations as an additional modality into the input of LLMs. We pretrain the hypergraph encoder on 25M tables with selfsupervised learning. The experimental results on different LLMs over multiple datasets demonstrate the effectiveness and generalization of our method.

## Limitation

The limitations of our proposed LLaSA are as follows: (1) We used a fixed number of query tokens, but the number of nodes in the hypergraph varies significantly, with some graphs having as few as a dozen nodes and others having over a hundred. As a result, when faced with graphs that have a large number of nodes, the G-Former may struggle to capture information effectively. (2) Due to resource constraints, we conduct our experiments using a context length of 2K instead of the 8K used in TableLlama. The performance of LLaSA in longer contexts remains to be evaluated further.

## Ethics Statement

This paper proposes a method for SKG, and the experiments are conducted on public available datasets. As a result, there is no data privacy concern. Meanwhile, this paper does not involve human annotations, and there are no related ethical concerns.

## Acknowledgment

This work was supported by the National Key RD Program of China (No. 2022ZD0160503) and Beijing Natural Science Foundation (L243006) and the National Natural Science Foundation of China (No.62376270).

## References

Abhinav Pandey Abhishek Kadian Ahmad Al-Dahle Aiesha Letman et al. Abhimanyu Dubey, Abhinav Jauhri. 2024. The llama 3 herd of models. Preprint, arXiv:2407.21783.

Reza Yazdani Aminabadi, Samyam Rajbhandari, Minjia Zhang, Ammar Ahmad Awan, Cheng Li, Du Li, Elton Zheng, Jeff Rasley, Shaden Smith, Olatunji Ruwase, and Yuxiong He. 2022. Deepspeed inference: Enabling efficient inference of transformer models at unprecedented scale. Preprint, arXiv:2207.00032.

Yejin Bang, Samuel Cahyawijaya, Nayeon Lee, Wenliang Dai, Dan Su, Bryan Wilie, Holy Lovenia, Ziwei Ji, Tiezheng Yu, Willy Chung, Quyet V. Do, Yan Xu, and Pascale Fung. 2023. A Multitask, Multilingual, Multimodal Evaluation of ChatGPT on Reasoning, Hallucination, and Interactivity. arXiv preprint. ArXiv:2302.04023 [cs].

Junwei Bao, Duyu Tang, Nan Duan, Zhao Yan, Yuanhua Lv, Ming Zhou, and Tiejun Zhao. 2018. Table-totext: Describing table region with natural language. In Proceedings of the AAAI conference on artificial intelligence, volume 32.

Pei Chen, Soumajyoti Sarkar, Leonard Lausen, Balasubramaniam Srinivasan, Sheng Zha, Ruihong Huang,

and George Karypis. 2023a. HYTREL: Hypergraphenhanced Tabular Data Representation Learning. arXiv preprint. ArXiv:2307.08623 [cs].

Runjin Chen, Tong Zhao, Ajay Jaiswal, Neil Shah, and Zhangyang Wang. 2024. LLaGA: Large Language and Graph Assistant. In ICML 2024. arXiv.

Wenhu Chen, Hongmin Wang, Jianshu Chen, Yunkai Zhang, Hong Wang, Shiyang Li, Xiyou Zhou, and William Yang Wang. 2019. Tabfact: A largescale dataset for table-based fact verification. arXiv preprint arXiv:1909.02164.

Yukang Chen, Shengju Qian, Haotian Tang, Xin Lai, Zhijian Liu, Song Han, and Jiaya Jia. 2023b. Longlora: Efficient fine-tuning of long-context large language models. arXiv preprint arXiv:2309.12307.

Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Dylan Langdon, Reema Moussa, Matt Beane, Ting-Hao Huang, Bryan Routledge, et al. 2021. Finqa: A dataset of numerical reasoning over financial data. arXiv preprint arXiv:2109.00122.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv:1810.04805 [cs]. ArXiv: 1810.04805.

Ning Ding, Yujia Qin, Guang Yang, Fuchao Wei, Zonghan Yang, Yusheng Su, Shengding Hu, Yulin Chen, Chi-Min Chan, Weize Chen, et al. 2023. Parameter-efficient fine-tuning of large-scale pretrained language models. Nature Machine Intelligence, 5(3):220–235.

Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh V. Chawla, Thomas Laurent, Yann LeCun, Xavier Bresson, and Bryan Hooi. 2024. G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering. arXiv preprint. ArXiv:2402.07630 [cs].

Jonathan Herzig, Pawel Krzysztof Nowak, Thomas Müller, Francesco Piccinno, and Julian Eisenschlos. 2020. TaPas: Weakly Supervised Table Parsing via Pre-training. In Proceedings ofthe 58th Annual Meeting of the Association for Computational Linguistics, pages 4320–4333, Online. Association for Computational Linguistics.

Kevin Stone Peter Albert Amjad Almahairi-Yasmine Babaei Nikolay Bashlykov et al. Hugo Touvron, Louis Martin. 2023. Llama 2: Open foundation and fine-tuned chat models. Preprint, arXiv:2307.09288.

Mohit Iyyer, Wen-tau Yih, and Ming-Wei Chang. 2017. Search-based neural structured learning for sequential question answering. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1821– 1831.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023. Mistral 7b. Preprint, arXiv:2310.06825.

Rihui Jin, Yu Li, Guilin Qi, Nan Hu, Yuan-Fang Li, Jiaoyan Chen, Jianan Wang, Yongrui Chen, and Dehai Min. 2024. HGT: Leveraging Heterogeneous Graph-enhanced Large Language Models for Fewshot Complex Table Understanding. arXiv preprint. ArXiv:2403.19723 [cs].

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023. BLIP-2: Bootstrapping Language-Image Pretraining with Frozen Image Encoders and Large Language Models. arXiv preprint. ArXiv:2301.12597 [cs].

Pan Lu, Liang Qiu, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, Tanmay Rajpurohit, Peter Clark, and Ashwin Kalyan. 2022. Dynamic prompt learning via policy gradient for semi-structured mathematical reasoning. arXiv preprint arXiv:2209.14610.

Chaitanya Malaviya, Chandra Bhagavatula, Antoine Bosselut, and Yejin Choi. 2020. Commonsense knowledge base completion with structural and semantic context. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pages 2925–2933.

Hany Awadalla Ahmed Awadallah Ammar Ahmad Awan Nguyen Bach et al. Marah Abdin, Jyoti Aneja. 2024. Phi-3 technical report: A highly capable language model locally on your phone. Preprint, arXiv:2404.14219.

Linyong Nan, Chiachun Hsieh, Ziming Mao, Xi Victoria Lin, Neha Verma, Rui Zhang, Wojciech Krysci ´ nski,´ Hailey Schoelkopf, Riley Kong, Xiangru Tang, et al. 2022. Fetaqa: Free-form table question answering. Transactions of the Association for Computational Linguistics, 10:35–49.

Linyong Nan, Dragomir Radev, Rui Zhang, Amrit Rau, Abhinand Sivaprasad, Chiachun Hsieh, Xiangru Tang, Aadit Vyas, Neha Verma, Pranav Krishna, et al. 2020. Dart: Open-domain structured data record to text generation. arXiv preprint arXiv:2007.02871.

Ankur P Parikh, Xuezhi Wang, Sebastian Gehrmann, Manaal Faruqui, Bhuwan Dhingra, Diyi Yang, and Dipanjan Das. 2020. Totto: A controlled table-to-text generation dataset. arXiv preprint arXiv:2004.14373.

Panupong Pasupat and Percy Liang. 2015. Compositional semantic parsing on semi-structured tables. arXiv preprint arXiv:1508.00305.

N Reimers. 2019. Sentence-bert: Sentence embeddings using siamese bert-networks. arXiv preprint arXiv:1908.10084.

Chang Shu, Yusen Zhang, Xiangyu Dong, Peng Shi, Tao Yu, and Rui Zhang. 2021. Logic-consistency text generation from semantic parses. arXiv preprint arXiv:2108.00577.

Alon Talmor and Jonathan Berant. 2018. The web as a knowledge-base for answering complex questions. arXiv preprint arXiv:1803.06643.

Jiabin Tang, Yuhao Yang, Wei Wei, Lei Shi, Lixin Su, Suqi Cheng, Dawei Yin, and Chao Huang. 2023. GraphGPT: Graph Instruction Tuning for Large Language Models. arXiv. ArXiv:2310.13023 [cs].

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention Is All You Need. arXiv preprint. ArXiv:1706.03762 [cs].

Petar Velickoviˇ c, Guillem Cucurull, Arantxa Casanova,´ Adriana Romero, Pietro Lio, and Yoshua Bengio. 2017. Graph attention networks. arXiv preprint arXiv:1710.10903.

Tianbao Xie, Chen Henry Wu, Peng Shi, Ruiqi Zhong, Torsten Scholak, Michihiro Yasunaga, Chien-Sheng Wu, Ming Zhong, Pengcheng Yin, Sida I. Wang, Victor Zhong, Bailin Wang, Chengzu Li, Connor Boyle, Ansong Ni, Ziyu Yao, Dragomir Radev, Caiming Xiong, Lingpeng Kong, Rui Zhang, Noah A. Smith, Luke Zettlemoyer, and Tao Yu. 2022. Unified-SKG: Unifying and Multi-Tasking Structured Knowledge Grounding with Text-to-Text Language Models. arXiv preprint. ArXiv:2201.05966 [cs].

Michihiro Yasunaga, Antoine Bosselut, Hongyu Ren, Xikun Zhang, Christopher D Manning, Percy S Liang, and Jure Leskovec. 2022. Deep bidirectional language-knowledge graph pretraining. Advances in Neural Information Processing Systems, 35:37309– 37323.

Pengcheng Yin, Graham Neubig, Wen-tau Yih, and Sebastian Riedel. 2020. TaBERT: Pretraining for Joint Understanding of Textual and Tabular Data. ACL 2020. ArXiv: 2005.08314.

Jingyi Zhang, Jiaxing Huang, Sheng Jin, and Shijian Lu. 2024a. Vision-language models for vision tasks: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence.

Tianshu Zhang, Xiang Yue, Yifei Li, and Huan Sun. 2024b. TableLlama: Towards Open Large Generalist Models for Tables. arXiv preprint. ArXiv:2311.09206 [cs].

Xikun Zhang, Antoine Bosselut, Michihiro Yasunaga, Hongyu Ren, Percy Liang, Christopher D Manning, and Jure Leskovec. 2022. Greaselm: Graph reasoning enhanced language models for question answering. arXiv preprint arXiv:2201.08860.

Jianan Zhao, Meng Qu, Chaozhuo Li, Hao Yan, Qian Liu, Rui Li, Xing Xie, and Jian Tang. 2022. Learning on large-scale text-attributed graphs via variational inference. arXiv preprint arXiv:2210.14709.

Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al. 2023. A survey of large language models. arXiv preprint arXiv:2303.18223.

Alex Zhuang, Ge Zhang, Tianyu Zheng, Xinrun Du, Junjie Wang, Weiming Ren, Stephen W. Huang, Jie Fu, Xiang Yue, and Wenhu Chen. 2024. StructLM: Towards Building Generalist Models for Structured Knowledge Grounding. arXiv preprint. ArXiv:2402.16671 [cs].

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Overall Length</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>Train</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>Test</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=2 colspan=1>Dataset</td><td rowspan=2 colspan=2>Input    Output(avg)    (avg)</td><td rowspan=2 colspan=1>Count</td><td rowspan=1 colspan=3>Input   Output  #Nodes</td><td rowspan=1 colspan=1># Trunc</td><td rowspan=1 colspan=3>Input   OutputCount</td><td rowspan=2 colspan=1># Nodes(avg)</td><td rowspan=2 colspan=1># Trunc</td></tr><tr><td rowspan=1 colspan=2>(max)   (max)</td><td rowspan=1 colspan=1>(avg)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>(max)   (max)</td></tr><tr><td rowspan=1 colspan=1>TabMWP</td><td rowspan=1 colspan=2>208      5</td><td rowspan=1 colspan=1>23059</td><td rowspan=1 colspan=1>709</td><td rowspan=1 colspan=1>33</td><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>7686</td><td rowspan=1 colspan=2>703     31</td><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>ToTTo</td><td rowspan=1 colspan=1>252</td><td rowspan=1 colspan=1>31</td><td rowspan=1 colspan=1>120761</td><td rowspan=1 colspan=1>2040</td><td rowspan=1 colspan=1>155</td><td rowspan=1 colspan=1>110</td><td rowspan=1 colspan=1>467</td><td rowspan=1 colspan=1>7700</td><td rowspan=1 colspan=1>2048</td><td rowspan=1 colspan=1>119</td><td rowspan=1 colspan=1>111</td><td rowspan=1 colspan=1>31</td></tr><tr><td rowspan=1 colspan=1>KVRet</td><td rowspan=1 colspan=1>573</td><td rowspan=1 colspan=1>17</td><td rowspan=1 colspan=1>6288</td><td rowspan=1 colspan=1>1217</td><td rowspan=1 colspan=1>161</td><td rowspan=1 colspan=1>57</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>807</td><td rowspan=1 colspan=1>1147</td><td rowspan=1 colspan=1>82</td><td rowspan=1 colspan=1>56</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>HybridQA</td><td rowspan=1 colspan=1>700</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>62682</td><td rowspan=1 colspan=1>2047</td><td rowspan=1 colspan=1>91</td><td rowspan=1 colspan=1>92</td><td rowspan=1 colspan=1>200</td><td rowspan=1 colspan=1>3466</td><td rowspan=1 colspan=2>2048    79</td><td rowspan=1 colspan=1>93</td><td rowspan=1 colspan=1>6</td></tr><tr><td rowspan=1 colspan=1>CompWebQ</td><td rowspan=1 colspan=1>1350</td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>27639</td><td rowspan=1 colspan=1>2047</td><td rowspan=1 colspan=1>321</td><td rowspan=1 colspan=1>265</td><td rowspan=1 colspan=1>321</td><td rowspan=1 colspan=1>2816</td><td rowspan=1 colspan=2>2048    256</td><td rowspan=1 colspan=1>264</td><td rowspan=1 colspan=1>8</td></tr><tr><td rowspan=1 colspan=1>TabFact</td><td rowspan=1 colspan=1>660</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>92283</td><td rowspan=1 colspan=1>2045</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>94</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>12779</td><td rowspan=1 colspan=2>1687     4</td><td rowspan=1 colspan=1>93</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>WikiTQ</td><td rowspan=1 colspan=1>832</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>11321</td><td rowspan=1 colspan=1>2028</td><td rowspan=1 colspan=1>273</td><td rowspan=1 colspan=1>114</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>4344</td><td rowspan=1 colspan=2>2048    148</td><td rowspan=1 colspan=1>115</td><td rowspan=1 colspan=1>10</td></tr><tr><td rowspan=1 colspan=1>WikiSQL</td><td rowspan=1 colspan=1>689</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>56355</td><td rowspan=1 colspan=1>2047</td><td rowspan=1 colspan=1>518</td><td rowspan=1 colspan=1>96</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>15878</td><td rowspan=1 colspan=2>2048    244</td><td rowspan=1 colspan=1>98</td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>FeTaQA</td><td rowspan=1 colspan=1>653</td><td rowspan=1 colspan=1>39</td><td rowspan=1 colspan=1>7326</td><td rowspan=1 colspan=1>1853</td><td rowspan=1 colspan=1>158</td><td rowspan=1 colspan=1>97</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>2003</td><td rowspan=1 colspan=2>1548    114</td><td rowspan=1 colspan=1>95</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>DART</td><td rowspan=1 colspan=1>134</td><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>62659</td><td rowspan=1 colspan=1>406</td><td rowspan=1 colspan=1>258</td><td rowspan=1 colspan=1>17</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>5097</td><td rowspan=1 colspan=2>261    109</td><td rowspan=1 colspan=1>17</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>SQA</td><td rowspan=1 colspan=2>657      35</td><td rowspan=1 colspan=1>12275</td><td rowspan=1 colspan=2>1812    1012</td><td rowspan=1 colspan=1>98</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>3011</td><td rowspan=1 colspan=2>1725    769</td><td rowspan=1 colspan=1>102</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>WikiTableText</td><td rowspan=1 colspan=2>150      27</td><td rowspan=1 colspan=1>10000</td><td rowspan=1 colspan=2>313     97</td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>2000</td><td rowspan=1 colspan=2>226     89</td><td rowspan=1 colspan=1>14</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>Finqa</td><td rowspan=1 colspan=2>1230      21</td><td rowspan=1 colspan=1>6251</td><td rowspan=1 colspan=2>2040     72</td><td rowspan=1 colspan=1>29</td><td rowspan=1 colspan=1>186</td><td rowspan=1 colspan=1>1147</td><td rowspan=1 colspan=2>2048     61</td><td rowspan=1 colspan=1>31</td><td rowspan=1 colspan=1>25</td></tr></table>

Table 4: The statistics of numbers of input and output tokens in the training and test sets for each task. "# Trunc" indicates the number of samples where the input length exceeds 2048 tokens and has been truncated. "# Nodes" indicates the average number of nodes in hypergraphs.

## A Pretraining Dataset

We use the 25 million tabled collected by TaBERT (Yin et al., 2020). We designed three types of question templates and used them to generate 10 questions for each table. The specific templates are as follows:

1. What’s the column name of"{node\_name}" ?

2. In the row where the value of {first\_col\_name} is "{row\_value}", what is the corresponding value of{col\_name}?

3. Are "{node\_name1}" and "{node\_name2}" in the same row?.

## B SKG Datasets

Some datasets are not used in our study, such as FEVEROUS and Infotabs, because the tables in these datasets are not well-structured, with some rows having a different number of cells than the table headers. The statistics of the SKG datasets we used are shown in Table 4.