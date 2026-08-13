# Generating Long-form Story Using Dynamic Hierarchical Outlining with Memory-Enhancement

Qianyue Wang <sup>1</sup> <sup>2</sup>\* Jinwu Hu <sup>1</sup> <sup>2</sup>\* Zhengping Li <sup>1</sup> Yufeng Wang <sup>1</sup> <sup>3</sup> Daiyuan Li <sup>1</sup> Yu Hu <sup>4</sup> Mingkui Tan <sup>1</sup> <sup>†</sup>

<sup>1</sup>South China University of Technology, <sup>2</sup>Pazhou Laboratory, <sup>3</sup>Peng Cheng Laboratory, <sup>4</sup>Hong Kong Polytechnic University

## Abstract

Long-form story generation task aims to produce coherent and sufficiently lengthy text, essential for applications such as novel writing and interactive storytelling. However, existing methods, including LLMs, rely on rigid outlines or lack macro-level planning, making it difficult to achieve both contextual consistency and coherent plot development in long-form story generation. To address this issues, we propose Dynamic Hierarchical Outlining with Memory-Enhancement long-form story generation method, named DOME, to generate the long-form story with coherent content and plot. Specifically, the Dynamic Hierarchical Outline (DHO) mechanism incorporates the novel writing theory into outline planning and fuses the plan and writing stages together, improving the coherence of the plot by ensuring the plot completeness and adapting to the uncertainty during story generation. A Memory-Enhancement Module (MEM) based on temporal knowledge graphs is introduced to store and access the generated content, reducing contextual conflicts and improving story coherence. Finally, we propose a Temporal Conflict Analyzer leveraging temporal knowledge graphs to automatically evaluate the contextual consistency of long-form story. Experiments demonstrate that DOME significantly improves the fluency, coherence, and overall quality of generated long stories compared to state-of-the-art methods.

## 1 Introduction

Among natural language processing (NLP) tasks, the automatic generation of a long-form story is a representative task that requires creativity and long-term planning skills. By learning from humanwritten stories, an automated storyteller mimics humans and becomes competent in producing stories useful for various application scenarios, such as novel writing and interactive storytelling (Riedl and Young, 2010). Large Language Models (LLMs) are rapidly developing, generating a long-form story that further increases dramatically in length, complexity, and fluency (Yang et al., 2022).

![](images/05ad15e303d9104c28764c259a14afaa9f40520370be10c61ded7bcbda8a4ce1.jpg)  
(c) Dynamic Hierarchical outlining with Memory-Enhancement (Ours)  
Figure 1: Illustration and comparison of three strategies of long-form story generation. (a) Applying a fixed hierarchical outline to guide the story generation, is challenging to adapt to the uncertainty in story creation. (b) Adapting to uncertainty through interaction with humans allows for flexibility but lacks a high-level storyline to guide story development. (c) Our method aims to enhance story coherence from both plot and expression and enjoy the advantages of the former two strategies to improve the coherence of the plot and reduce contextual conflicts.

Unfortunately, it is difficult for the LLMs to generate a long-form story maintaining contextual consistency in semantic and coherent plot development for the following reasons. 1) Memory limitations: The black-box self-attention mechanism forms the core of LLMs for contextual connections and still suffers from the long-range dependency issue (Vaswani et al., 2017), making it hard for them to precisely and unambiguously recall and thereby leading to contextual incoherence. 2) Planning difficulties: LLMs cannot effectively apply knowledge of coherent plot planning partly because their training involves learning from vast datasets that include a variety of texts, but they do not inherently understand or apply the principles of storytelling and thus hinders the generation of engaging stories with complete and fluent plots (Xie et al., 2023).

Existing long-form story generation methods often leverage higher-level attributes such as plots or commonsense knowledge (Xie et al., 2023), primarily aiming to enhance story development fluency, and can be divided into two kinds based on whether humans are involved. The first category emulates the human writing process by adopting a plan-and-write framework (Yang et al., 2022) in generating the long-form story (see Fig. 1(a)). This approach separates the generation process into the planning and writing stages, utilizing detailed and plot-fluent outlines to guide the writing phase. While these methods can extend the length of the story and improve plot development fluency, the inflexibility of a fixed outline can impede its adaptability to uncertainty in the writing stage, often leading to plot incoherence, such as contextual repetition or conflicts. Other works (Zhou et al., 2023; Brahman et al., 2020) explore generating content progressively without outlining based on human interaction and relevant preceding content (see Fig. 1 (b)). They can address the influence of local plot development caused by uncertainty in generation. However, the generated story lacks a macro-level of rational planning, affecting the plot completeness.

To address the above limitations, we propose Dynamic Hierarchical Outlining with Memory-Enhancement long-form story generation method, called DOME, to generate the long-form story with coherent content and plot (see Fig. 1(c)). DOME is based on the collaboration of a Dynamic Hierarchical Outline (DHO) mechanism and a Memory-Enhancement Module (MEM) to generate long-form story. Specifically, DHO mechanism to guide long-form story generation based on the plan-write writing framework (Yang et al., 2022) and the novel-writing theory (Campbell, 1949). The DHO mechanism requires the generation of a rough outline conforming to the novel writing theory to ensure plot completeness, and the dynamic planning of a detailed outline based on the rough outline during the writing process so that it can adapt to the uncertainty of the generation and improve the plot fluency. The MEM stores and accesses generated stories through temporal knowledge graphs and provides contextual content for outline planning and story writing to reduce content conflicts in long story texts. To automatically evaluate the contextual consistency, we propose a conflict detection matrix, Temporal Conflict Analyzer which is based on the information representation rules of the temporal knowledge graph and LLM.

Our main contributions are as follows:

• A new paradigm for long-form story generation. Since we find that existing methods for long-form story generation either cause plot incoherence due to fixed outlines or lack of fluency and readability due to poor macro-level planning, we propose a Dynamic Hierarchical Outline (DHO) mechanism to fuse the planning and writing stages, making it adaptable to the generation uncertainty and improving plot coherence. Experiments show that DHO improves 6.87% on Ent-2 metric.

• A new approach to contextual conflict resolution. We propose a Memory-Enhancement Module (MEM) based on the temporal knowledge graph to store and access the generated story. Applying LLM to perform semantic filtering on KG retrieval results keeps the conciseness of historical information and thus improves the contextual consistency of generated stories. Experiments show that MEM reduces conflicts by 87.61%.

• A new evaluation for conflict detection. To evaluate the degree of contextual consistency automatically, we propose a potential conflict detection method based on the information representation rules of the temporal knowledge graph and apply LLM to further determine whether a conflict exists. Experiments show that the judgment results of this method are consistent with human preferences.

## 2 Related Work

## 2.1 Long-form Story Generation

Existing works have attempted to use language models to automatically generate novels (Yang et al., 2022, 2023). With the widespread use of LLMs, people begin to generate longer stories (Zhou et al., 2023). Recent works for generating long-form stories can be classified into two categories based on whether human participation is engaged. The first category imitates the human writing process through a plan-and-write framework (Yao et al., 2019). Fan et al. (2018) proposes a hierarchical story generation method, which first generates a story premise and then generates the story based on it. Yao et al. (2019) introduces a "plan-and-write" framework for open-domain story generation, which divides the writing process into planning and generating. Yang et al. (2022) further subdivides writing into plan, draft, rewrite, and edit module. Yang et al. (2023) makes efforts on outline control to generate a more detailed and hierarchical story. Although this technique can lengthen the narrative and enhance plot fluidity, the rigid nature of a predetermined outline can hinder flexibility in the writing phase, resulting in plot inconsistencies, such as repeated or contradictory contexts.

![](images/8f2ec6bef4e3405468d6cb811ecd9eb3c78e1016922cce7df33dd44b846bbd0c.jpg)  
Figure 2: General diagram of proposed DOME. The story generation process is divided into several stages based on the amount of rough outline. At stage i, we expand rough outline i into several detailed outlines based on the relevant content provided by MEM, and these outlines generate partial story sequentially based on their relevant content querying from MEM. Every generated partial story is stored in a Temporal Knowledge Graph (TKG) for the following query.

Other works (Zhou et al., 2023; Brahman et al., 2020) explore using human interaction to replace pre-generated outlines. Brahman et al. (2020) introduce a content-inducing approach, which involves human being using cue-phrases. Zhou et al. (2023) propose a language simulacrum of RNN recurrence, enabling human-guided story planning. This reduces plot inconsistencies but results in less fluent and readable narratives.

## 2.2 KG-enhanced LLM Inference

Knowledge Graphs (KGs) can effectively model the key information in text by triples in the form of < subject, action, object > (Ji et al., 2022). After the appearance of LLMs, like GPT-3.5 (Achiam et al., 2023), Llama 3 (AI@Meta, 2024) and Qwen1.5 (Bai et al., 2023), the integration of LLMs and KGs has attracted widespread attention due to the potential improvement KGs can make for LLMs (Ji et al., 2022). KG can be used as a knowledge base to provide references when LLM generates content. There are two kinds of work to inject the knowledge in KG into LLM. The first one fuse knowledge in KG and input by directly concatenating them in language level (Zhang et al., 2024; Wen et al., 2024; Sun et al., 2021) or in token level (Zhang et al., 2019; He et al., 2020; Su et al., 2021). These methods can easily fuse the knowledge between them but with little interaction between knowledge and input. The second one applies the addition knowledge fusion module in which the knowledge is updated and simplified (Sun et al., 2022; Zhang et al., 2022).

## 3 Problem definition and motivation

## 3.1 Problem Definition

For a given story premise I from a writer, we design a framework $F ^ { * }$ to generate a long-form story S. The score of story S on the plot coherence is $C ^ { P l o t }$ and the score of story S on the context coherence is $C ^ { C o n t e x t }$ . We aim to generate a story S with improved $C ^ { P l o t }$ and $C ^ { C o n t e x t }$

## 3.2 Motivaiton

Existing long-form story generation methods separate planning and writing stages based on the planwrite framework, making it impossible to adapt to the uncertainty of the writing stage. Besides, stories generated through this two-stage method tend to stop abruptly at the beginning, resulting in missing plots. Another method includes humaninvolved detail outline preferences, improving the flexibility of outlines but making the overall storyline uncontrollable or incomplete. An intuitive idea arises: What if we could further improve story coherencefrom plotfluency and completion?

![](images/f48d0487e41250b4bd7566c14f75e31a206e6536117e0fe2f7e84848e709893d.jpg)  
Figure 3: Five stages novel writing theory from Joseph Campbell (Campbell, 1949).

The answer is yes. We follow the idea of the hierarchical outline (Yang et al., 2023) which is composed of the rough outline and the detailed outline. To ensure the plot completeness, we consider the rough outline as the macro plot guidance and apply the novel writing theory to guide its generation to ensure it contains all the story stages of the theory. The detailed outlines are expanded gradually based on the corresponding rough outline referred to the relevant content in the leading story to improve the adaptability to the generation uncertainty. In addition, with the increasing length of the generated story, the contextual inconsistency becomes obvious (Yang et al., 2022). Therefore, we design a memory module to store stories and access concise relevant content. Since KG can model information and retrieve the relevant information (Pan et al., 2024), we apply KG to store and access the semantically relevant content with a filter based on LLM. By providing accurate and concise relevant content during the generation stage, this module can improve contextual consistency.

## 4 Proposed Method

We propose a Dynamic Hierarchical Outlining with Memory-Enhancement long-form story generation method, named DOME, aiming to improve the story coherence from plot and context description. Based on the advanced understanding and generation ability of LLM (Liu et al., 2023), DOME is mainly composed of two parts: the Dynamic Hierarchical Outline module (DHO) and the KG-based Memory-Enhancement Module (MEM). As shown in Fig. 2, given the user input I, including story setting, character introduction, and storyline requirements, the rough outlines of a story are planned at a time under the guidance of the writing theory, ensuring the completeness of the plot. Then, the detailed outline and the corresponding part of the story are generated alternately. It means the detailed outlines are not expanded by the rough outline incrementally until its previous stories are generated, enabling it to adjust to the uncertainty of the previously generated stories, and then the detailed outlines guide the generation of sequential stories. MEM stores user input at the very beginning and the generated story during the writing process. The module provides relevant content in natural language for the generation stage, ensuring contextual consistency. Through the collaboration of these two modules, DOME improve the story’s coherence from the plot and contextual consistency.

4.1 Dynamic Hierarchical Outline Mechanism The hierarchical outline H = R, D is composed of a rough outline R and a detailed outline D. Previous works have used higher-level attributes like plots or commonsense knowledge to improve the quality of generated stories. We find that certain novel writing theories (Campbell, 1949; Freytag; Vogler and Montez, 2007) enhance story generation. By incorporating the novel writing theory (Campbell, 1949) (see Fig. 3) into the rough outline planning stage, we improve the overall structure. This is done by stating the theory in the rough outline generation prompt. Thus, the rough outline R was generated based on user input I:

![](images/89dad37d072a9c740e139da9ff819eb7b70185a24200664cedd525df960e09f3.jpg)  
Figure 4: The details of querying for the relevant content. Notably the last arrow points to the historical information semantically related to the input content.

$$
R = L L M ( W T , I , P _ { r o u g h \_ o u t l i n e } ) ,\tag{1}
$$

where $W T$ is the novel writing theory, $L L M ( \cdot )$ is LLM inference operation, and $P _ { r o u g h . }$ <sub>\_outline</sub> is the prompt guiding LLM to generate a rough outline.

The detailed outline $D = \{ d _ { i } \} _ { i = 1 } ^ { 5 }$ is generated step by step according to the corresponding part of the rough outline $r _ { i }$ and its relevant content $R I n f o . i$ , ensuring its adaptation to the uncertainty in the story generation stage.

$$
d _ { i } = L L M ( r _ { i } , R I n f o . _ { i } , P _ { d e t a i l e d \_ o u t l i n e } ) ,\tag{2}
$$

where P<sub>detailed\_outline</sub> is the prompt that indicates LLM to generate a detailed outline. The $d _ { i }$ $\{ d o _ { i } ^ { t } \} _ { t = 1 } ^ { M }$ contains detailed outlines expanded from the rough outline. The M is the number of detailed outlines expanded from the given rough outline $r _ { i }$ which should be set at the beginning.

The story content $s _ { i } ^ { t }$ is generated step-by-step based on the detailed outline $d o _ { i } ^ { t }$ and relevant generated content $D I n f o . _ { i } ^ { t }$ of the sub-detailed outline $d o _ { i } ^ { t } .$ , which is as follows:

$$
s _ { i } ^ { t } = L L M ( d o _ { i } ^ { t } , D I n f o . _ { i } ^ { t } , P _ { g e n _ { - } s t o r y } ) ,\tag{3}
$$

where $P _ { g e n \_ s t o r y }$ is the prompt that indicates LLM to generate a partial story. The detailed process of writing story $S$ in Algorithm 1. The details of prompts in the DHO are shown in Appendix B.1.

## 4.2 Memory-Enhancement Module

Recently, many works (Liu et al., 2024) have shown that LLMs tend to ignore the information in the middle of the long inputs. As the story generation process progresses, the increasing story length affects the LLM’s attention on the relevant content when generating new content, increasing computing overhead (Choromanski et al., 2021) and even leading to contextual conflict (Yang et al., 2022).

Algorithm 1 Inference Pipeline for DOME   
Require: Novel writing theory WT, the Large Language   
Model applied in DOME LLM, user input I, rough   
outline prompt P<sub>rough\_outline</sub>, detailed outline prompt   
$P _ { d e t a i l e d \_ o u t l i n e } ,$ story generation prompt $P _ { g e n . }$ <sub>\_story</sub>.   
1: Initialize MEM KG.   
2: Generate rough outline R via Eq. (1).   
3: for $r _ { i }$ in $R = \{ r _ { i } \} _ { i = 1 } ^ { i = 5 }$ do   
4: Find relevant content $R I n f o _ { i }$ about $r _ { i }$ from $K G .$   
5: Generate detailed outlines $d _ { i }$ via Eq. (2).   
6: Add d into $D .$   
7: for $d o _ { i } ^ { t }$ in $d _ { i } = \{ d o _ { i } ^ { t } \} _ { t = 1 } ^ { t = M }$ do   
8: Find relevant content $D I n f o . _ { i } ^ { t }$ about $d o _ { i } ^ { t }$ from   
KG.   
9: Generate partial story $s _ { i } ^ { t }$ via Eq. (3).   
10: Add $s _ { i } ^ { t }$ into story $S .$   
11: Store $s _ { i } ^ { t }$ into $K { \dot { G } } .$   
12: end for   
13: end for   
14: return R, D and S.

We propose an additional Memory-Enhancement Module, named MEM, to store the generated story and provide concise relevant content.

KGs and their variants are well-established tools for content storage and query (Wen et al., 2024). KGs extract the key information for the content like subject, action, and object and ignore unimportant information like adverbs and modifiers. Besides, the structure of the knowledge graph supports reasoning and information fusion, which is conducive to information integration (Pan et al., 2024). Based on the analysis above, the MEM applies Temporal Knowledge Graph (TKG) (Roddick and Spiliopoulou, 2002) to store the generated story. TKG is stored by quadruples in the form of $<$ subject, action, object, index >. The index means the chapter number of the information. The module stores the input story premise at the very beginning and the generated content and provides query-relevant content based on entity retrieval in TKG each time the DHO generates content.

To store entity, we use LLM to extract triples for every sentence and then add the current chapter number to form quadruples. To access the rele-

Figure 5: The criterion of filtering on semantic relevance.

vant content (see Fig. 4), we first conduct entitybased quadruple retrieval (Reinanda et al., 2020) on TKG and then apply LLM to filter the retrieved results through semantics correlation by evaluating relevance based on rules (see Fig. 5). MEM can provide the top-k most relevant information as query-relevant content, making it concise and semantically relevant with input (more details can be seen in Appendix B.2).

## 4.3 Temporal Conflict Analyzer

The evaluation of long-form stories in previous work mainly relies on humans, which is timeconsuming and laborious (Yang et al., 2023; Wang et al., 2023a). MEM applies TKG to store story and it is easy for TKG to associate context (Wen et al., 2024). Thus we propose an auto metric that measures the contextual consistency of the story by calculating the rate of conflict quadruples to total quadruples. To detect the conflict quadruples, the quadruples in TKG $Q ~ = ~ \{ q _ { i } \} _ { i = 1 } ^ { N }$ of generated stories are grouped by rules (see Fig. 26 in Appendix D) sequentially and without repetition and the potential conflict information is aggregated. Since LLM-as-a-judge has been verified by experiments (Zheng et al., 2023) and it can evaluate some features of a text within a certain length according to rules (Achiam et al., 2023; Liu et al., 2021), we apply LLM to further detect the conflict quadruples based on time order. More details about the temporal conflict analyzer are reported in Appendix E. In this way, we can find the quadruples Q<sup>conf</sup> <sup>lict</sup> = $\{ q _ { i } ^ { c o n f i c t } \} _ { i = 1 } ^ { m }$ containing conflict information. The conflict rate is expressed as fol-

lows:

$$
C R . = m / N \times 1 0 0 \% ,\tag{4}
$$

where N is the number of quadruples in TKG and m is the number of conflict quadruples in TKG.

## 5 Experiment

Implementation details. DOME \* contains DHO and MEM. In DHO, every rough outline is expanded into 3 detailed outlines which means the M mentioned in Section 4.1 is set to 3. The embedding model we applied is bge-large-en-v1.5 (Chen et al., 2024). In MEM, we apply cosine similarity (Lahitani et al., 2016) and set the filter threshold to 0.75 to implement the entity-based query. Besides, we switch off the default historical content input of LLM where MEM is applied. The temperature of LLM is set to 0.5 and the max token of LLM is 1000 while other parameters are kept by default. All the experiments are conducted on 2 A800 GPU with CUDA version 11.3.

Dataset and baselines. Following the settings of DOC (Yang et al., 2023), we use its story premises<sup>†</sup> as the input and generate 20 long stories for evaluation. The details of the dataset are in Appendix A. We use Qwen1.5-72B-Chat (Bai et al., 2023) for long-form story generation as it has considerable performance (Chiang et al., 2024) and is able to deployed locally. We compare DOME with two types of methods. The first is prompting LLM to generate stories as long as possible with input.

<table><tr><td rowspan="2">Methods</td><td colspan="3">Auto Evaluation</td><td colspan="5">Human Evaluation</td></tr><tr><td>Word Num.</td><td>CR. ↓</td><td>Ent-2↑</td><td>PCo.↓</td><td>PCoh.↓</td><td>Rel. ↓</td><td>Int. ↓</td><td>Ecoh. ↓</td></tr><tr><td>Llama3-70B-Instruct (AI@Meta, 2024)</td><td>612.65</td><td>6.78</td><td>9.25</td><td>3.77</td><td>3.81</td><td>3.58</td><td>3.96</td><td>3.72</td></tr><tr><td>Qwen1.5-72B-Chat (Bai et al., 2023)</td><td>495.70</td><td>0.66</td><td>9.06</td><td>4.77</td><td>4.80</td><td>4.58</td><td>4.96</td><td>4.72</td></tr><tr><td>Re3 (Yang et al., 2022)</td><td>3802.15</td><td>0.77</td><td>11.56</td><td>2.73</td><td>2.53</td><td>2.96</td><td>2.56</td><td>2.83</td></tr><tr><td>DOC (Yang et al., 2023)</td><td>3904.90</td><td>1.21</td><td>11.55</td><td>2.58</td><td>2.43</td><td>2.63</td><td>2.34</td><td>2.56</td></tr><tr><td>Ours (Qwen1.5-72B-Chat)</td><td>7113.75</td><td>0.56</td><td>12.29</td><td>1.15</td><td>1.44</td><td>1.26</td><td>1.77</td><td>1.16</td></tr></table>

Table 1: Comparison with LLMs and SOTA baselines. We use the same premises as DOC Yang et al. (2023).

<table><tr><td rowspan="2">Methods</td><td colspan="3">Auto Evaluation</td><td colspan="5">Human Evaluation</td></tr><tr><td>Word Num.</td><td>CR.↓</td><td>Ent-2↑</td><td>PCo. ↓</td><td>PCoh.↓</td><td>Rel. ↓</td><td>Int. ↓</td><td>Ecoh. ↓</td></tr><tr><td>w/o MEM</td><td>6511.10</td><td>4.52</td><td>10.00</td><td>1.88</td><td>2.24</td><td>1.97</td><td>1.96</td><td>2.22</td></tr><tr><td>w/o DHO</td><td>1471.90</td><td>0.65</td><td>11.50</td><td>2.92</td><td>2.36</td><td>2.86</td><td>2.88</td><td>2.46</td></tr><tr><td>Ours (Qwen1.5-72B-Chat)</td><td>7113.75</td><td>0.56</td><td>12.29</td><td>1.20</td><td>1.40</td><td>1.17</td><td>1.62</td><td>1.32</td></tr></table>

Table 2: Ablation study of different modules.

The second are the state-of-the-art methods including DOC (Yang et al., 2023) and $\mathrm { R e ^ { 3 } }$ (Yang et al., 2022). To validate the scalability of DOME, we report the improvement when applying DOME on Llama-3-70b-Instruct (AI@Meta, 2024) and Yi1.5- 34B-Chat (Young et al., 2024).

Metrics. Since the content repetition is an important aspect reflecting the quality of stories in terms of coherence (Yang et al., 2022), We apply the auto evaluation metrics including n-gram entropy (Zhang et al., 2018) and set $n = 2 \left( E n t - 2 \right)$ to evaluate the degree of content repetition by measuring the diversity of vocabulary. Besides, we apply our proposed conflict rate (CR.) to evaluate the contextual consistency of the generated stores.

To evaluate the alignment of the stories with human preferences in terms of contextual consistency and coherence and other basic story quality, we further conduct a human evaluation to evaluate the story quality of plot completeness, coherence, relevance, interest level, and expression coherence. We compare all methods according to each indicator and calculate their average rank. The description of every metric is as follows: 1) Plot Completeness (P Co.): the extent to which the generated story covers all the stages mentioned in the story theory. 2) Plot Coherence (P Coh.): the fluency of the development of the generated story. 3) Relevance (Rel.): the consistency between the generated story and the input premise. It demonstrates the actual usability of the method. 4) Interesting (Int.): the reading interest to the user. Since the reading interest reflects peoples’ preferences on the completeness and coherence of stories (Wang et al., 2023b). 5) Expression Coherence (ECoh.): the contextual consistency of the story. More details of human evaluation are shown in Appendix C.

## 5.1 Comparison Experiments

DOME is consistently better in auto evaluation. From Table 1, the proposed DOME achieves superior performance in both CR. and Ent-2. It also generates the longest content. Specifically, DOME surpasses the LLM baseline, Qwen1.5-72B-Chat, by 35.7% in Ent-2 and reduce the CR. by 15.2%. Additionally, DOME outperforms the $\mathrm { R e ^ { 3 } }$ by 6.3% in Ent-2 and reduces the CR. by 27.3%. The results indicate that our DOME delivers diverse content with fewer conflicts. We attribute the improvement to the following facts. Firstly, our TKG is capable of fine-grained modeling of generated stories. It provides accurate relevant content through semantic filtering during the generation stage, which ensures consistency and reduces contextual conflicts in the generated content. Besides, Our DHO dynamically adjusts the detailed outlines during the story generation process, thereby increasing the space for plot development and enhancing content diversity.

DOME is consistently better in human evaluation. We conduct a human evaluation to assess how well DOME aligns with human preferences. In Table 1, our DOME ranks first across all five metrics. The results demonstrate that MEM provides relevant content during the generation stage, reducing the conflict due to the unclear memory of LLM itself (Zhang et al., 2023) and thus ensuring contextual consistency and enhancing relevance and expression coherence. Furthermore, our DHO dynamically generates detailed outlines based on the development of the storyline, adapting to the uncertainty in generation and achieving improved fluency. Additionally, our novel-writing mechanism encourages Large Language Models to create stories like writing a complete novel. This approach ensures the plot’s completeness and to some extent, introduces fluctuations that capture and maintain reader interest.

![](images/af5eb663a43b1187ace74125814ce038988fc883b19539699a8a0ad8b2b9db72.jpg)  
Figure 6: Scalability results of applying DOME on different LLMs. Left: the increase rates of Ent-2 metric. Right: the decrease rates of CR. metric.

## 5.2 Ablation Studies

Effectiveness of DHO. We remove the DHO so that the outline is fixed and consistent with the input storyline. As shown in Table 2, this variant exhibits higher content conflicts and reduced content variety. Additionally, it ranks lowest across all human evaluation metrics. It supports the notion that dynamic adjustment of the detailed outline helps to control content generation, thereby ensuring plot coherence. Furthermore, the application of novel writing theory enhances plot completeness and thus improves the reading interest, this is because a complete story encourages the reader to read to the end. See the example of DHO in Appendix F.

Effectiveness of MEM. Similarly, we remove the MEM to eliminate the knowledge graph for reference and resort to the multi-round chatting capability of LLM to generate content. Limited by computing cost, we set the maximum chat rounds to 2. As shown in Table 2, the conflict rate significantly increases from 0.56 to 4.52 and Ent-2 reduces from 12.29 to 10.00 without MEM. Additionally, all human evaluation metrics deteriorate to varying degrees. These results indicate that MEM ensures contextual coherence by providing relevant content during generation, which also improves plot coherence and alignment with human preferences. See the example of MEM in Appendix E.

## 5.3 The Scalability of DOME

To demonstrate the applicability of DOME on different LLMs, we apply our method to Yi1.5-34B-

<table><tr><td>LLM (in DOME)</td><td colspan="4">Nodes Relations Quadruples API calls</td></tr><tr><td>Yi1.5-34B-Chat</td><td>691.38</td><td>328.45</td><td>354.55</td><td>16</td></tr><tr><td>Llama3-70B-Instruct 844.62</td><td></td><td>345.65</td><td>434.00</td><td>16</td></tr><tr><td>Qwen1.5-72B-Chat 791.35</td><td></td><td>258.23</td><td>404.85</td><td>16</td></tr></table>

Table 3: The computation and storage cost of DOME. Notable the API calls means the average number of API calls for KG construction. The column in the table named “LLM (in DOME)” means the Large Language Model applied in DOME.
<table><tr><td>Method</td><td>Ent-2↑</td><td>CR.↓</td><td>Word Num.</td></tr><tr><td>Qwen1.5-72B-Chat</td><td>9.06</td><td>0.66</td><td>495.70</td></tr><tr><td>AgentWriter</td><td>11.13</td><td>2.14</td><td>3140.00</td></tr><tr><td>DOME</td><td>12.29</td><td>0.56</td><td>7113.75</td></tr></table>

Table 4: The Comparison of the performance with the agent-based method, using the same Large Language Model (Qwen1.5-72B-Chat).

Chat, Llama3-70B-Instruct and Qwen1.5-72B-Chat. As illustrated in Fig. 6, DOME significantly reduces conflicts and enhances content diversity on all LLMs with different parameter scales. Specifically, DOME reduce the CR. by more than 18% on Llama3-70B-Instruct. It also improves the Ent-2 by more than 29% on Yi1.5-34B-Chat. This scalability is due to the straightforward and easyto-follow prompts in our DOME, making it easy to be adaptable across different LLMs.

## 5.4 Computation and Storage Cost of DOME

We report the computation and storage cost for constructing the knowledge graph dynamically, as shown in Table 3. With acceptable additional computation and storage overhead, the proposed method improves the quality of generating longform stories. Specifically, as shown in Tables 1 and 3, the proposed DOME only adds 791 nodes of storage overhead and 16 LLM API calls to construct the KG dynamically, but it outperforms the SOTA methods in all performance metrics. This is because the structured storage of KG and the access advantages of KG facilitate LLM in achieving accurate and concise relevant memory and reducing context inconsistency.

## 5.5 Compare with the Agent-based Method

To further validate the effectiveness of DOME, we compare DOME with the agent-based long story generation method, AgentWriter (Bai et al., 2024). As shown in Table 4, the proposed DOME outperforms AgentWriter across all metrics. Compared to AgentWriter, DOME generates stories twice as long, with CR. dropping to a quarter and Ent-2 increasing by 1.16. This means that DOME generates longer stories with both plot coherence and contextual consistency.

## 6 Conclusion

In this paper, we propose a Dynamic Hierarchical Outlining with Memory Enhancement, named DOME, to generate a coherent long-form story. Specifically, we propose a Dynamic Hierarchical Outline (DHO) mechanism for long-form story generation, based on the plan-write framework and novel-writing theory. The DHO mechanism creates a rough outline that aligns with novel-writing theory and dynamically generates it during the writing process, enhancing plot fluency. Additionally, the Memory-Enhancement Module (MEM) uses temporal knowledge graphs to store and access generated stories, providing contextual content for both detailed outline planning and story writing, and reducing contextual conflicts. Lastly, the temporal conflict analyzer detects potential conflicts using temporal knowledge graphs and integrates with LLMs to automatically assess the contextual consistency of the generated text. Experiments demonstrate that DOME significantly improves the coherence and overall quality of generated longform stories from plot and expression compared to SOTA methods.

## Limitations

The lack of automatic evaluation matrics constrains our experiments. Specifically, we have to rely on human evaluation to evaluate the quality of the generated stories which is time-consuming and costly. Besides, the amount of experimental data is limited since there are no specific datasets for long-form story generation and we follow the experiment setting of DOC (Yang et al., 2023). In addition, our framework requires massive LLM API calls which are time-consuming. On average, Generating a story requires about 200 API calls. Massive API calls result in a limited story generation speed, taking about 4 hours to complete a 7,000-word story. Thus the long text generation and a large number of API calls limit our usage of paid closed-source LLMs, such as ChatGPT. Although we report the result about the extensibility of our framework, many steps in our framework are realized based on custom-designed prompts and it is better to re-design these prompts to achieve

better performance.

Although experiments demonstrate the effectiveness and scalability of our proposed DOME, the prompts in DOME are designed based on human experience. Therefore, the use of automatic prompts may further exploit the capabilities of DOME. However, automatic prompting methods often struggle with capturing the nuanced context required for a specific task (Si et al., 2023). Applying them to long-form story generation tasks may not ensure the completeness of long-form stories and further affects the effectiveness and scalability of DOME.

## Ethics Statement

This work fully complies with the ACL Ethics Policy. We declare that there are no ethical issues in this paper, to the best of our knowledge.

## References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

AI@Meta. 2024. Llama 3 model card.

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. 2023. Qwen technical report. arXiv preprint arXiv:2309.16609.

Yushi Bai, Jiajie Zhang, Xin Lv, Linzhi Zheng, Siqi Zhu, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. 2024. Longwriter: Unleashing 10,000+ word generation from long context llms. CoRR, abs/2408.07055.

Faeze Brahman, Alexandru Petrusca, and Snigdha Chaturvedi. 2020. Cue me in: Content-inducing approaches to interactive story generation. In Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing, pages 588–597, Suzhou, China. Association for Computational Linguistics.

Joseph Campbell. 1949. The Hero with a Thousand Faces. Pantheon Books, New York, NY, USA.

Jianlyu Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. 2024. M3- embedding: Multi-linguality, multi-functionality, multi-granularity text embeddings through selfknowledge distillation. In Findings of the Association for Computational Linguistics: ACL 2024, pages 2318–2335, Bangkok, Thailand. Association for Computational Linguistics.

Wei-Lin Chiang, Lianmin Zheng, Ying Sheng, Anastasios Nikolas Angelopoulos, Tianle Li, Dacheng Li, Banghua Zhu, Hao Zhang, Michael I. Jordan, Joseph E. Gonzalez, and Ion Stoica. 2024. Chatbot arena: An open platform for evaluating llms by human preference. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net.

Krzysztof Marcin Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamás Sarlós, Peter Hawkins, Jared Quincy Davis, Afroz Mohiuddin, Lukasz Kaiser, David Benjamin Belanger, Lucy J. Colwell, and Adrian Weller. 2021. Rethinking attention with performers. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net.

Angela Fan, Mike Lewis, and Yann N. Dauphin. 2018. Hierarchical neural story generation. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics, ACL 2018, Melbourne, Australia, July 15-20, 2018, Volume 1: Long Papers, pages 889–898. Association for Computational Linguistics.

Gustav Freytag. Die technik des dramas.

Bin He, Di Zhou, Jinghui Xiao, Xin Jiang, Qun Liu, Nicholas Jing Yuan, and Tong Xu. 2020. Integrating graph contextualized knowledge into pre-trained language models. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2020, Online Event, 16-20 November 2020, volume EMNLP 2020 of Findings of ACL, pages 2281–2290. Association for Computational Linguistics.

Shaoxiong Ji, Shirui Pan, Erik Cambria, Pekka Marttinen, and Philip S. Yu. 2022. A survey on knowledge graphs: Representation, acquisition, and applications. IEEE Trans. Neural Networks Learn. Syst., 33(2):494–514.

Alfirna Rizqi Lahitani, A. E. Permanasari, and N. A. Setiawan. 2016. Cosine similarity to determine similarity measure: Study case in online essay assessment. 2016 4th International Conference on Cyber and IT Service Management, pages 1–6.

Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2024. Lost in the middle: How language models use long contexts. Trans. Assoc. Comput. Linguistics, 12:157–173.

Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig. 2023. Pretrain, prompt, and predict: A systematic survey of prompting methods in natural language processing. ACM Comput. Surv., 55(9):195:1–195:35.

Zeming Liu, Haifeng Wang, Zheng-Yu Niu, Hua Wu, and Wanxiang Che. 2021. DuRecDial 2.0: A bilingual parallel corpus for conversational recommendation. In Proceedings of the 2021 Conference on

Empirical Methods in Natural Language Processing, pages 4335–4347, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Shirui Pan, Linhao Luo, Yufei Wang, Chen Chen, Jiapu Wang, and Xindong Wu. 2024. Unifying large language models and knowledge graphs: A roadmap. IEEE Trans. Knowl. Data Eng., 36(7):3580–3599.

Ridho Reinanda, Edgar Meij, and Maarten de Rijke. 2020. Knowledge graphs: An information retrieval perspective. Found. Trends Inf. Retr., 14(4):289–444.

Mark O. Riedl and Robert Michael Young. 2010. Narrative planning: Balancing plot and character. J. Artif. Intell. Res., 39:217–268.

John F. Roddick and Myra Spiliopoulou. 2002. A survey of temporal knowledge discovery paradigms and methods. IEEE Trans. Knowl. Data Eng., 14(4):750– 767.

Chenglei Si, Zhe Gan, Zhengyuan Yang, Shuohang Wang, Jianfeng Wang, Jordan L. Boyd-Graber, and Lijuan Wang. 2023. Prompting GPT-3 to be reliable. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net.

Yusheng Su, Xu Han, Zhengyan Zhang, Yankai Lin, Peng Li, Zhiyuan Liu, Jie Zhou, and Maosong Sun. 2021. Cokebert: Contextual knowledge selection and embedding towards enhanced pre-trained language models. AI Open, 2:127–134.

Yu Sun, Shuohuan Wang, Shikun Feng, Siyu Ding, Chao Pang, Junyuan Shang, Jiaxiang Liu, Xuyi Chen, Yanbin Zhao, Yuxiang Lu, Weixin Liu, Zhihua Wu, Weibao Gong, Jianzhong Liang, Zhizhou Shang, Peng Sun, Wei Liu, Xuan Ouyang, Dianhai Yu, Hao Tian, Hua Wu, and Haifeng Wang. 2021. ERNIE 3.0: Large-scale knowledge enhanced pre-training for language understanding and generation. CoRR, abs/2107.02137.

Yueqing Sun, Qi Shi, Le Qi, and Yu Zhang. 2022. Jointlk: Joint reasoning with language models and knowledge graphs for commonsense question answering. In Proceedings of the 2022 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, NAACL 2022, Seattle, WA, United States, July 10-15, 2022, pages 5049–5060. Association for Computational Linguistics.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems, 30.

Christopher Vogler and Michele Montez. 2007. The writer’s journey: Mythic structure for writers.

Yichen Wang, Kevin Yang, Xiaoming Liu, and Dan Klein. 2023a. Improving pacing in long-form story planning. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 10788– 10845, Singapore. Association for Computational Linguistics.

Yichen Wang, Kevin Yang, Xiaoming Liu, and Dan Klein. 2023b. Improving pacing in long-form story planning. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 10788– 10845, Singapore. Association for Computational Linguistics.

Yilin Wen, Zifeng Wang, and Jimeng Sun. 2024. Mindmap: Knowledge graph prompting sparks graph of thoughts in large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, pages 10370–10388. Association for Computational Linguistics.

Zhuohan Xie, Trevor Cohn, and Jey Han Lau. 2023. The next chapter: A study of large language models in storytelling. In Proceedings of the 16th International Natural Language Generation Conference, pages 323–351, Prague, Czechia. Association for Computational Linguistics.

Kevin Yang, Dan Klein, Nanyun Peng, and Yuandong Tian. 2023. DOC: improving long story coherence with detailed outline control. In Proceedings ofthe 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023, pages 3378– 3465. Association for Computational Linguistics.

Kevin Yang, Yuandong Tian, Nanyun Peng, and Dan Klein. 2022. Re3: Generating longer stories with recursive reprompting and revision. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022, pages 4393–4479. Association for Computational Linguistics.

Lili Yao, Nanyun Peng, Ralph M. Weischedel, Kevin Knight, Dongyan Zhao, and Rui Yan. 2019. Planand-write: Towards better automatic storytelling. In The Thirty-Third AAAI Conference on Artificial Intelligence, AAAI 2019, The Thirty-First Innovative Applications of Artificial Intelligence Conference, IAAI 2019, The Ninth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2019, Honolulu, Hawaii, USA, January 27 - February 1, 2019, pages 7378–7385. AAAI Press.

Alex Young, Bei Chen, Chao Li, Chengen Huang, Ge Zhang, Guanwei Zhang, Heng Li, Jiangcheng Zhu, Jianqun Chen, Jing Chang, Kaidong Yu, Peng Liu, Qiang Liu, Shawn Yue, Senbin Yang, Shiming Yang, Tao Yu, Wen Xie, Wenhao Huang, Xiaohui Hu, Xiaoyi Ren, Xinyao Niu, Pengcheng Nie, Yuchi Xu, Yudong Liu, Yue Wang, Yuxuan Cai, Zhenyu Gu,

Zhiyuan Liu, and Zonghong Dai. 2024. Yi: Open foundation models by 01.ai. CoRR, abs/2403.04652.

Qinggang Zhang, Junnan Dong, Hao Chen, Daochen Zha, Zailiang Yu, and Xiao Huang. 2024. KnowGPT: Knowledge graph based prompting for large language models. In The Thirty-eighth Annual Conference on Neural Information Processing Systems.

Xikun Zhang, Antoine Bosselut, Michihiro Yasunaga, Hongyu Ren, Percy Liang, Christopher D Manning, and Jure Leskovec. 2022. GreaseLM: Graph REA-Soning enhanced language models. In International Conference on Learning Representations.

Yizhe Zhang, Michel Galley, Jianfeng Gao, Zhe Gan, Xiujun Li, Chris Brockett, and Bill Dolan. 2018. Generating informative and diverse conversational responses via adversarial information maximization. In Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montréal, Canada, pages 1815–1825.

Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu, Tingchen Fu, Xinting Huang, Enbo Zhao, Yu Zhang, Yulong Chen, Longyue Wang, Anh Tuan Luu, Wei Bi, Freda Shi, and Shuming Shi. 2023. Siren’s song in the AI ocean: A survey on hallucination in large language models. CoRR, abs/2309.01219.

Zhengyan Zhang, Xu Han, Zhiyuan Liu, Xin Jiang, Maosong Sun, and Qun Liu. 2019. ERNIE: enhanced language representation with informative entities. In Proceedings ofthe 57th Conference ofthe Associationfor Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2, 2019, Volume 1: Long Papers, pages 1441–1451. Association for Computational Linguistics.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging llm-as-a-judge with mt-bench and chatbot arena. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.

Wangchunshu Zhou, Yuchen Eleanor Jiang, Peng Cui, Tiannan Wang, Zhenxin Xiao, Yifan Hou, Ryan Cotterell, and Mrinmaya Sachan. 2023. Recurrentgpt: Interactive generation of (arbitrarily) long text. CoRR, abs/2305.13304.

## A Dataset Detsils

We follow the setting of DOC (Yang et al., 2023) which uses 20 story premises generated by InstructGPT3-175B as input for story generation. Every premise includes a story setting, a character introduction, and a necessary storyline. An example of an input is shown in the Fig. 7.

## B Prompt Details

We design prompts for LLM to finish specific tasks in our framework and evaluate the contextual consistency in CR..

We design our prompt following the suggestions instead of applying the automatic prompt method since the automated prompt methods often struggle with capturing the nuanced context required for specific tasks (Si et al., 2023) and may affect the generation quality.

## B.1 Prompt in DHO

In DHO, we prompt (shown in Fig. 8) LLM to plan a rough outline based on input and referring to the story writing theory. The detailed outline is planned by prompting (shown in Fig. 9) LLM to realize based on the corresponding part of the rough outline referring to the relevant content.

## B.2 Prompt in MEM

In MEM, quadruples are extracted from the generated story based on the few-shot learning ability of LLM, and the prompt is shown in Fig. 10. When querying relevant content from TKG, the entities from subjects and objects of quadruples extracted from a query are used to retrieve quadruples in TKG. LLM evaluates the relevant scores between the query and the retrieved quadruples based on the prompt shown in Fig. 11.

## B.3 Prompt in Temporal Conflict Analyzer

In the temporal conflict analyzer, we first apply rules to group quadruples and then use LLM to further determine the existence of contextual conflict. In detail, quadruples of every group are first described in natural language and then judged to be reasonable or not based on common sense. All these are completed by prompting LLM applied in the framework. Evaluating the contextual consistency by LLM is also based on prompts and is further divided into two steps: 1)describe the information expressed by a group of quadruples in natural language based on the grouping feature (shown in Fig. 12, Fig. 13, Fig. 14, Fig. 15 and Fig. 16.) 2)judge if the description is reasonable based on the grouping feature (shown in Fig. 17, Fig. 18, Fig. 19, Fig. 20, Fig. 21). The mining on the group from filtered information use the same information expression prompts which are shown Fig. 12, Fig. 13, Fig. 14, Fig. 15.

## B.4 Prompt in Experiments

One of our compared methods is based on prompting LLM to realize long-form story generation and the prompt is shown in Table 22.

## C Human Evaluation Details

We describe the experimental details for human evaluation in the experiments mentioned in section 5. For every evaluation, we prepare 20 groups of stories generated by different methods or experiment settings and ask human evaluators to rank stories in a group from best fit to least fit according to the given human metrics. Specifically, we recruited three well-educated graduate students as evaluators and asked them to perform a blind ranking process. We designed a platform for the ranking process, where the interface displays input, generates detailed outlines, and generates stories from different methods or experiment settings. Fig. 23, Fig. 24 and Fig. 25 show the interface used for human evaluation in different experiments. Evaluators are required to read input, detailed outlines, and stories choose to the best stories in the current interface that fits the human metrics. Every chosen story disappears and its corresponding method is recorded. For every human metric, evaluators continue to choose the story best fits the metric in the current interface until there is no story shown in the interface.

## D Grouping Rules

Fig. 26 shows the grouping rules we applied to classify quadruples based on the knowledge expression features.

## E The Example of Output from MEM

## E.1 The Example of Extracted Quadruples

Here is an example of quadruples which are constructed by story content and the result is shown in Fig. 27. Specifically, LLM is restricted to building triples from text, including subject, action and object and then the program adds the current chapter number to each triplet to form a quadruple.

![](images/610e08f4226fffb49b6bb9d4ebd6c96bfd7b4c10fcfa9e3b9980cac15453b347.jpg)  
Figure 7: The example of input.

![](images/4e581085ccc8fafed9e306891b426f62a52b287ce4abed487953a4c623349c80.jpg)  
Figure 8: The prompt for plan a rough outline.

![](images/811f412c58a360a436bbd37dcb0bdddfe2374703cb3200142d1bffb5acd6f4cc.jpg)  
Figure 9: The prompt for planning a detailed outline.

![](images/df8543bebad297b7368fd591d57578f198c9538ad50b2db22bd5f5de4ca9d115.jpg)  
Figure 10: The prompt for extract quadruple from story.

![](images/85cf1589f8dd1c58d7ab47e0ef7182e1bba991a61d5aee3d795fb26761245e20.jpg)  
Figure 11: The prompt for score relevance between the description and the given outline.

![](images/e867f2e726e235e2b5b1209aaf8d9e7b305e4ec19007cf552f28e093a5531ce0.jpg)  
Figure 12: The prompt for describing the information based on quadruples grouped by rule 1.

![](images/0b05635bb49e79943adba8293f1990c975d868192eb7afb3ecf9785e35b5a0cf.jpg)

Figure 13: The prompt for describing the information based on quadruples grouped by rule 2.  
![](images/1922b303bfd49132576a1f2eb86a2fd11ce6da8ee9f938c3abe7a69c96a5127f.jpg)  
Figure 14: The prompt for describing the information based on quadruples grouped by rule 3.

![](images/7a9d1d3d76787c9f740680bb316a6ba0eed64ef958cdcd0068cb48dc038c835d.jpg)  
Figure 15: The prompt for describing the information based on quadruples grouped by rule 4.

![](images/2a0025bcb53040182d2db14928b050c7e4c11b968a56051963e7cc10e8da107a.jpg)  
Figure 16: The prompt for describing the information based on quadruples grouped by rule 5.

![](images/15ab3170f9773c480e624325848c719c4e9bbabdb295284e6e2093b0b1391e16.jpg)  
Figure 17: The prompt for judging whether the description is reasonable based on rule 1.

![](images/a6335617e995ab1a22e507adcb82e23eadf9226341f16a81a170271d6d735126.jpg)  
Figure 18: The prompt for judging whether the description is reasonable based on rule 2.

![](images/c6c927cc004f5bb9908bfbedacb65abdf12808864fb629c873ca5140738d0ab7.jpg)  
Figure 19: The prompt for judging whether the description is reasonable based on rule 3.

![](images/b9e4a803c9d8607849c5bb89ad8c02740c587d75d851a89b9b152d74d25177e4.jpg)  
Figure 20: The prompt for judging whether the description is reasonable based on rule 4.

![](images/d317352800a7c94bb675c73f120af136364a7c59cc0c28e0ea358101c7394bc2.jpg)  
Figure 21: The prompt for judging whether the description is reasonable based on rule 5.

![](images/10c62f9f698a9b129c9e4ee24b655aaea3cdde3d1fc1f2d36a13aea4269aa349.jpg)  
Figure 22: The prompt for generate story by LLMs.

![](images/bf3eb5c560bfa5fcb598022c6abf69082545ab4f407d584ead5448fb2c25eadf.jpg)  
Figure 23: The interface for human interface in comparison experiment.

![](images/0ac58640c3a3b75dce4e400f02499a994f331f6bf3294e0297a479add33d7d0f.jpg)  
Figure 24: The interface for human interface in ablation experiment.

![](images/e16e01b8a51e1d089fe9574b8b0c4b4ac878b7641af5f10fe612d7c007cb5b41.jpg)  
Figure 25: The interface for human interface in scalability experiment.

## E.2 The Example of Provided Relevant Information

Here are some examples of the query and their relevant information provided by MEM. The examples are shown in Fig. 28 and Fig. 29. These results are relevant to the query since they contain the same entity and are semantically similar. In Fig. 28, we show the query result about a rough outline. The information provided by MEM mentioned the same protagonists with the query, like "Brad" and "Shannon". Besides, the information provided by MEM gives more details about the query. For example, for "Brad’s secretive behavior" in the rough outline, there are details in the provided information "late nights at work and hurried phone calls" and there are more summarized details about the changes in the time dimension. In Fig. 29, we show the query result about a detailed outline. Consistent with the information for the rough outline, the information for the detailed outline provides more relevant details about the detailed outline, For example, for "Sergei Snipe’s connection to the murders" in the detailed outline, there are details about connection "Sergei Snipe was observed close to crime scenes".

## F The Example of Generated Hierarchical Outline in DHO

Here we post an example of a hierarchical outline including the rough outline and the detailed outline and made some analysis to validate the effectiveness of DHO module. The rough outline is shown in Fig. 30 and the detailed outline Table 5.

![](images/ae2e109304b265c4dc897246140cdffcbde95edd83d31d4252cf0bfbc51017c9.jpg)  
Figure 26: The rules for grouping quadruples.

## Text

GabrielMartin, a young man in his early twenties, found himself nestled in the cozy confines of his newly acquired apartment. The transition to independence had been both exhilarating and daunting, but he had managed to keep his sanity amidst the flurry of change. One evening, as he sat by the window, watching the city lights twinkle in the distance, his thoughts drifted to Lina Sanchez, his girlfriend who was now miles away.Their long-distance relationship, once filled with excitement and promise, began to show the strain of their new living situations. The once effortless connection seemed to require more effort, their laughter now tinged with a hint of sadness. The silence between them on phone calls grew heavier, and it was during one such call that the tension finally broke."Lina, I can’t ignore the fact that things feel different lately," Gabriel said, his voice filled with concern. "We’re both adjusting to new lives, and it’s hard not to feel the distance."Lina’s voice, though soft, conveyed her own struggles. "I know, Gabriel. I miss you too, and sometimes it’s overwhelming. But we love each other, and we’ll find a way through this, won’t we?"Gabriel’s mind wandered back to his parents’ relationship, their love having weathered storms far greater than the one they now faced. He remembered how they had maintained their bond even when separated by duty and distance. Angelica, his wise older sister, had once shared her insights on long-distance love, emphasizing the importance of communication and trust.Recalling Angelica’s advice, Gabriel took a deep breath and spoke from the heart. "Lina, let’s be honest with each other about our fears and insecurities. We need to make an effort to stay connected, even if it means having these tough conversations. Small gestures, like sharing our daily routines, can mean the world."Lina agreed, her voice gaining strength. "You’re right, Gabriel. I’ll try to be more open. And we’ll plan visits, surprise each other, and make the distance worth it."Gabriel felt a weight lifted off his shoulders as they mapped out a plan to nurture their relationship. Their conversation, though difficult, brought them closer than ever. As they hung up, Gabriel looked at the glowing screen of his phone, a symbol of their determination to overcome the test of distance. In this new chapter of their lives, they would find a way to keep their love alive, no matter the obstacles they faced.

## Quadruples

<GabrielMartin, finds himself, nestled in newly acquired apartment, 2>   
<transition, is, exhilarating and daunting, 2>   
<Gabriel, manages to keep, sanity, 2>   
<Gabriel, sits by the window, watching city lights, 2>   
<Gabriel’s thoughts, drift to, Lina Sanchez, 2>   
<relationship, fills with, excitement and promise, 2>   
<connection, seems to require, more effort, 2>   
<laughter, tinged with, sadness, 2>   
<silence, grows heavier, on phone calls, 2>   
<Gabriel, speaks, with concern, 2>   
<Gabriel, says, things feel different, 2>   
<Gabriel, acknowledges, adjusting to new lives, 2>   
<distance, is felt, by Gabriel, 2>   
<Lina’s voice, conveys, her struggles, 2>   
<Lina, admits, missing Gabriel, 2>   
<Gabriel, thinks of, parents’ relationship, 2>   
<Angelica, shares, insights on long-distance love, 2>   
<Gabriel, remembers, importance of communication and trust, 2>   
<Gabriel, takes a deep breath, speaks from the heart, 2>   
<Gabriel, suggests, honesty about fears and insecurities, 2>   
<Gabriel, proposes, staying connected, 2>   
<Lina, agrees, to be more open, 2>   
<Lina, plans, visits and surprises, 2>   
<Gabriel, feels weight lifted off, shoulders, 2>   
<conversation, brings, them closer, 2>   
<Gabriel, looks at, glowing phone screen, 2>   
<Gabriel and Lina, determined, to overcome distance, 2>   
<they, plan, to keep love alive, 2>   
<obstacles, they, will face, 2>

Figure 27: The example of quadruples.

![](images/139289a157ac842f73ce94b0f6711c4dd84e055b705c857806301c9bf90416d1.jpg)  
Figure 28: The example of rough query of MEM.

![](images/9bb737b599c25fa5c1385b2d5d3a902e1386ab3586c543b16c6ad1ace4949d13.jpg)  
Figure 29: The example of detailed query of MEM.

![](images/039460169529bc5d78629c5cb7e6160d1d7c672f18b80839f4a9a7e720d1b42f.jpg)  
Figure 30: The example of rough outline in our output.

## Example Detailed Outline

## Brushstrokes of Solace

Shannon continues to transform the attic into an art studio, filling it with her mother’s memories.

Shannon and Missy bond over their shared love for art, creating paintings together.

Uncle Jeff introduces Shannon to various art techniques and materials, nurturing her talent.

The paintings start to reflect Shannon’s emotional journey, evoking both sadness and hope.

Shannon discovers a hidden letter from her mother in one of Sara’s old art supplies.

## A New Color in the Canvas

Shannon and Katie Matthews grow closer, discussing their shared interests and dreams.

Katie, inspired by Shannon’s art, starts painting herself, discovering a hidden talent.

The paintings in the attic exhibit an inexplicable energy, hinting at a connection to Sara’s spirit.

Shannon, Missy, and Katie have a joint art exhibition at school, showcasing their emotional growth.

Shannon receives a positive response from her peers, helping her find a sense of belonging.

## Whispers from the Canvas

Shannon starts to sense her mother’s presence through the paintings, sensing guidance and comfort.

Uncle Jeff, noticing the changes in the artwork, shares his belief in the spiritual connection.

Shannon, Missy, and Katie have a late-night painting session, where they all feel their loved ones’ presence.

Shannon finds closure in her grief, understanding that her mother’s love is with her always.

The chapter ends with Shannon painting a vibrant, uplifting piece, symbolizing her newfound hope.

## The Mysterious Brushstrokes

Shannon, Katie, and Missy start investigating the phenomenon, visiting libraries and seeking advice from art experts.

Shannon has a revealing conversation with Rosemary Jacobs about the potential supernatural aspects of her art.

The trio experiments with Shannon’s painting techniques, trying to recreate the living effect.

A late-night encounter with one of Shannon’s paintings leaves them more confused and intrigued.

## A Glimpse Beyond the Canvas

Shannon has a vivid dream where her mother communicates through the paintings.

The girls attend an art seminar where they discuss the concept of energy and its influence on art.

Uncle Jeff shares stories of his own unexplained experiences, providing a broader perspective.

Shannon starts to suspect that her mother’s spirit may be channeling through her art.

A heartfelt conversation with Katie and Missy solidifies their bond and their dedication to understanding the mystery.

## The Living Gallery

Shannon, Katie, and Missy decide to showcase Shannon’s enchanted paintings in a small gallery exhibit.

Visitors react to the paintings in unexpected ways, experiencing emotions and memories triggered by the art.   
Rosemary Jacobs attends the exhibit, offering insights and support.

Shannon receives a mysterious note from someone claiming to know the truth behind the phenomenon.

The chapter ends with Shannon feeling a mix of excitement and apprehension, eager to uncover the secrets behind her artwork.

## Unraveling the Message

Shannon studies the hidden message within her painting, seeking to decipher its meaning.

With Katie and Missy’s help, Shannon investigates her mother’s past for clues. Shannon visits Rosemary Jacobs for guidance, learning more about Sara’s history and her own supernatural gift. A confrontation with skepticism from those around her, testing Shannon’s resolve.

## The Choice Looms

Shannon grapples with the implications of embracing her gift fully or suppressing it for a "normal" life.

The strain on her relationships with friends and family is highlighted, especially with her father’s disapproval.

Shannon has a heart-to-heart conversation with Katie about the importance of truth and acceptance.

A dream sequence reveals more about Sara’s intentions and the importance of Shannon’s role.

## Embracing the Unknown

Shannon decides to embrace her supernatural ability, understanding the impact it can have on others.

She begins to integrate her gift into her art, creating new works that offer healing and understanding.

A public demonstration of her gift brings both support and skepticism, solidifying Shannon’s resolve.

The chapter ends with Shannon feeling a newfound sense of purpose and connection to her mother’s memory.

## The Healing Canvas

Shannon continues to develop her unique art therapy, using her ability to connect with mourners through her paintings.

She holds a workshop with a group of grieving individuals, including Katie and other classmates, demonstrating the healing power of her art.

Shannon’s mother’s essence is felt by all, fostering a sense of unity and understanding among the participants.

Uncle Jeff and friends support Shannon in her newfound mission, helping her organize more events.

## A Community’s Response

The local community becomes aware of Shannon’s talent, and she receives invitations to share her gift at different gatherings.

Shannon learns to navigate both the admiration and skepticism, strengthening her conviction in her purpose.

She faces a challenging session with a skeptic, who eventually experiences a breakthrough, validating Shannon’s approach.

The town begins to heal collectively, with Shannon’s art playing a central role.

## The Letter’s Revelation

Shannon discovers a hidden letter from her mother in one of Sara’s old paintings, providing a personal message of love and encouragement.

The letter deepens Shannon’s understanding of her mother’s love and her own destiny.

Shannon reads the letter publicly at an event, connecting with the audience on a deeper emotional level.

The story concludes with Shannon finding closure and embracing her role as a healer, solidifying her bond with her mother’s memory.

## Preparation for Hope

Shannon finalizes her artwork for the exhibition, incorporating her emotional journey and her mother’s spirit.

Shannon reflects on her growth and the impact of Sara’s love on her art.

Katie and Missy offer support and encouragement, strengthening their friendship.

Shannon’s father starts to show interest in her art, hinting at a shift in their relationship.

## Exhibition of Resilience

The art exhibition opens, with Shannon’s paintings attracting attention and evoking emotions in visitors.

Shannon shares her story, inspiring others to find hope in their own struggles.

Shannon has a heartfelt conversation with her father, who acknowledges her talent and experiences.

Shannon, Katie, and Missy share a bonding moment, solidifying their friendship bond.

## A New Beginning

Shannon receives positive feedback on her work, boosting her confidence as an artist.

She reads the hidden letter from her mother, which brings a sense of closure and understanding.

Shannon decides on her artistic path, embracing her destiny to share hope through her art.

## New Beginnings

Gabriel settles into his new apartment, experiencing the initial excitement and anxiety of independence.

He interacts with his neighbor, Charles, who extends a warm welcome and offers advice about apartment living.

## The Test of Distance

Gabriel’s long-distance relationship with Lina Sanchez starts to show strain due to their new living situations.

They have a heartfelt conversation over the phone, discussing their concerns and how to maintain their connection.

Gabriel reflects on his parents’ relationship and Angelica’s advice on handling long-distance love.

## Navigating Challenges

Gabriel faces unexpected challenges of apartment living, such as noisy neighbors and learning to manage household tasks.

He turns to Charles for guidance, who shares his own experiences and offers practical solutions.

Gabriel’s independence grows as he troubleshoots these problems, strengthening his resolve and self-reliance.

## Budgeting Blues

Gabriel realizes the financial realities of independent living, encountering unexpected expenses and budget constraints.

He attempts to economize, cutting back on non-essentials and searching for ways to increase his income.

The strain of budgeting causes tension between Gabriel and his friends, as they indulge in activities he can no longer afford.

## Friendship Friction

Richie and Jeremy’s carefree lifestyle contrasts with Gabriel’s newfound responsibilities, leading to misunderstandings.

Gabriel feels resentment over their lack of understanding, while Richie and Jeremy struggle to empathize with his situation.

A confrontation between Gabriel and his friends highlights the growing rift in their relationships.

## Reconnecting with Lina

Gabriel acknowledges Lina’s feelings of neglect and makes an effort to prioritize their relationship.

They have a heart-to-heart conversation about the challenges they both face, seeking mutual understanding.

A plan is devised to balance Gabriel’s responsibilities with quality time for the couple, setting the stage for mending their bond.

## In the Wake of Disaster

Gabriel grapples with the aftermath of the apartment fire, sorting through the remnants of his belongings.

He meets with the landlord to discuss the incident, leading to a heated argument over responsibility.

Gabriel’s frustration deepens as he realizes the legal implications and potential financial burden.

## Seeking Support

Gabriel turns to his family for help, sharing his struggles and the recent events with them.

They offer emotional support and advice on how to handle the situation with the landlord.

Gabriel starts exploring temporary housing options, with the assistance of his family and friends.

## Lina’s ultimatum

Lina visits Gabriel amidst the chaos, expressing her concern for both their relationship and his well-being.

The fire serves as a backdrop for their conversation, heightening the urgency for change.

Gabriel commits to addressing not only the fire-related issues but also to making their relationship a priority, as threatened by Lina’s breakup warning.

## Rebuilding and Reconciliation

Gabriel, Richie, and Jeremy continue the repairs on Gabriel’s apartment, turning it into a welcoming home.

Gabriel and Richie mend their friendship, discussing the past misunderstandings and finding common ground.

Jeremy offers emotional support, sharing his own experiences with overcoming challenges.

## A New Routine and Priorities

Gabriel establishes a weekly cleaning and self-care schedule to maintain balance in his life.

He meets with Lina to discuss their relationship, openly addressing his fears and insecurities.

Lina and Gabriel set goals for strengthening their connection, overcoming the distance created by the recent events. New Routine and Priorities

Gabriel establishes a weekly cleaning and self-care schedule to maintain balance in his life.

He meets with Lina to discuss their relationship, openly addressing his fears and insecurities.

Lina and Gabriel set goals for strengthening their connection, overcoming the distance created by the recent events. Compensation and Community Support

The landlord, influenced by the community’s pressure, meets with Gabriel to discuss compensation for his losses.   
Gabriel negotiates a fair deal, showing maturity and resilience in the face of adversity.

The chapter ends on a positive note as Gabriel’s relationships with both Lina and his friends deepen, signaling a new beginning.

## A New Beginning

Gabriel and Lina solidify their commitment to each other, focusing on their deepening love.

Gabriel starts the support group, sharing his experiences and lessons learned with others.

The group dynamics and friendships formed within the support group are described.

Gabriel’s relationship with his family improves, as they see his growth and responsibility.

## Growth and Resolution

Gabriel confronts and resolves any lingering conflicts with his friends, showcasing his maturity.

Lina’s trembling subsides as she finds comfort in Gabriel’s support and open communication.

The couple faces and conquers a specific challenge together, demonstrating their newfound strength as a team.

The community’s acceptance of Gabriel and Lina is highlighted, fostering a sense of belonging.

## Celebrating Unity

A community event is organized, where Gabriel’s support group is recognized and celebrated.

Gabriel’s landlord surprises him with a gesture of appreciation, reflecting the positive change in their relationship. Gabriel and Lina share a special moment, symbolizing their growth and love.   
The volume concludes with a look at the future, as Gabriel and Lina plan their next steps together, ready for whatever challenges lie ahead.

Table 5: The example of detail outline.

## G The Example of Generated Story

There is an example of a generated story which is shown in Table 6. These results were generated based on Qwem1.5-72B-Chat.We make a case study of the story example shown in Table 6. Its hierarchical outline of the story is shown in Fig. 30 and Table 5.

While there are still some issues in this story, it shows coherence in plot and expression. The rough outline is complete and it covers all stages mentioned in the novel writing theory. Besides, the development of the story and the character description are consistent with the input, indicating high relevance to the input story premise. The story mentions Gabriel’s girlfriend is in a different place which is the creation during the generation stage and the subsequent follows this statement, indicating the story development fluency due to the adaption to the uncertainty of the wertting stage. In the example story, the plots from Gabriel encounter challenges to solve them have expanded in detail, explaining the details of reasonable solutions and the status changes of Characters. As the story proceeds, many difficulties are overcome by Gabriel, improving the readability and attraction to readers. His neighbor appeared from the beginning to the end with a consistent expression like "wise oldest man", "the wisdom of his neighbor", "older gentleman with a friendly smile" and "Charles listened intently, nodding empathetically". This expression is consistent, indicating the contextual consistency ensured through the relevant content provided in the writing stage by MEM. Therefore, the example story can demonstrate all the features that DOME attempts to improve and indicate the effectiveness of DOME.

## Example Story

Gabriel Martin, a sprightly young man with a touch of nervousness in his eyes, stepped into his new apartment for the very first time. The space was modest, but it represented a significant milestone in his life - the first step towards independence. The walls, still bare, echoed the silence of the uncharted territory he had just entered. A mix of excitement and anxiety pulsed through him, his heart thumping with anticipation.

As he stood in the middle of the room, taking in the blank canvas before him, a knock on the door startled him out of his thoughts. Peering through the peephole, he saw an older gentleman with a friendly smile - his neighbor, Charles. Charles extended a warm welcome, his weathered hand engulfing Gabriel’s in a reassuring grip. "Welcome to the building, Gabriel. If you ever need anything, just give me a holler." His kind demeanor instantly put Gabriel at ease.

Over a cup of steaming coffee, Charles shared his wisdom about apartment living - from the quirks of the building’s heating system to the best local grocery store. Gabriel listened intently, grateful for the guidance. Charles’s stories of his own experiences navigating independence made Gabriel feel less alone in his new adventure.

A few days later, the doorbell rang again, this time bringing a burst of energy and laughter. Richie and Jeremy, Gabriel’s childhood friends, had arrived to lend a hand with unpacking. The room was soon filled with a jumble of boxes and a flurry of nostalgia. As they worked, they reminisced about their shared memories - from the time they built a fort in Gabriel’s backyard to their wild teenage escapades.

Richie, the charismatic charmer, regaled them with tales of his recent travels, while Jeremy, the introverted thinker, shared his latest philosophical musings. Their banter filled the apartment with a familiar warmth, easing Gabriel’s anxieties about his new life. Together, they laughed, reminisced, and forged ahead, turning the empty apartment into a place that felt like home.

Gabriel’s new beginnings were marked by challenges, but with the support of his friends and the wisdom of his neighbor, he was learning to navigate the complexities of apartment living. Each day was a new adventure, and he was determined to embrace every moment of it, keeping his sanity intact and his heart open to the experiences that lay ahead.

Gabriel Martin, a young man in his early twenties, found himself nestled in the cozy confines of his newly acquired apartment. The transition to independence had been both exhilarating and daunting, but he had managed to keep hi sanity amidst the flurry of change. One evening, as he sat by the window, watching the city lights twinkle in the distance, his thoughts drifted to Lina Sanchez, his girlfriend who was now miles away.

Their long-distance relationship, once filled with excitement and promise, began to show the strain of their new living situations. The once effortless connection seemed to require more effort, their laughter now tinged with a hint of sadness. The silence between them on phone calls grew heavier, and it was during one such call that the tension finally broke.

"Lina, I can’t ignore the fact that things feel different lately," Gabriel said, his voice filled with concern. "We’re both adjusting to new lives, and it’s hard not to feel the distance."

Lina’s voice, though soft, conveyed her own struggles. "I know, Gabriel. I miss you too, and sometimes it’s overwhelming. But we love each other, and we’ll find a way through this, won’t we?"

Gabriel’s mind wandered back to his parents’ relationship, their love having weathered storms far greater than the one they now faced. He remembered how they had maintained their bond even when separated by duty and distance. Angelica, his wise older sister, had once shared her insights on long-distance love, emphasizing the importance of communication and trust.

Recalling Angelica’s advice, Gabriel took a deep breath and spoke from the heart. "Lina, let’s be honest with each other about our fears and insecurities. We need to make an effort to stay connected, even if it means having these tough conversations. Small gestures, like sharing our daily routines, can mean the world."

Lina agreed, her voice gaining strength. "You’re right, Gabriel. I’ll try to be more open. And we’ll plan visits, surprise each other, and make the distance worth it."

Gabriel felt a weight lifted off his shoulders as they mapped out a plan to nurture their relationship. Their conversation, though difficult, brought them closer than ever. As they hung up, Gabriel looked at the glowing screen of his phone, a symbol of their determination to overcome the test of distance. In this new chapter of their lives, they would find a way to keep their love alive, no matter the obstacles they faced.

Gabriel Martin had always imagined apartment living as a glamorous, carefree existence, but reality quickly proved him wrong. The walls of his cozy one-bedroom apartment seemed paper-thin, amplifying the sounds of his neighbors’ late-night parties and early-morning commotion. The once-appealing idea of independence was now tinged with frustration, as he found himself struggling to adapt to his new environment.

One afternoon, Gabriel decided to seek advice from his wise friend, Charles, an elderly gentleman who lived on the same floor. Charles had lived in the building for decades and was known for his wealth of knowledge and life experiences. Over cups of steaming tea in Charles’s well-worn living room, Gabriel shared his woes about the noisy neighbors and his own struggles with managing household chores.

Charles listened intently, nodding empathetically. "When I first moved in," he began, "I, too, faced similar challenges. But with time, I learned a few tricks to maintain my peace of mind." He offered Gabriel practical tips, like using noise-cancelling headphones during the day and investing in thick curtains to block out unwanted sounds at night He also shared his methods for staying organized and tackling household tasks efficiently.

Armed with Charles’s wisdom, Gabriel returned to his apartment with a newfound determination. He implemented Charles’s suggestions, filling his apartment with potted plants to absorb sound and establishing a weekly cleaning schedule. Each time he successfully navigated a new challenge, Gabriel felt a surge of pride and independence. His resolve strengthened, and he began to see these obstacles not as setbacks, but as opportunities for growth.

As the days turned into weeks, Gabriel’s self-reliance blossomed. He formed connections with his neighbors, even the noisy ones, finding common ground and understanding. Their late-night parties became chances to bond over shared music tastes, turning a once-irritating situation into a source of community.

Gabriel’s journey was far from over, but he was learning that navigating life’s challenges was as much about adapting as it was about resilience. With each obstacle he overcame, he became more grounded in his new life, grateful for the support of friends and the wisdom of Charles that guided him along the way.

Gabriel sat at the small, worn kitchen table in his new apartment, staring at a pile of bills and receipts spread out before him. The once exciting thrill of independence had given way to a sobering realization – life on his own came with a price tag he hadn’t fully anticipated. Rent, utilities, groceries, and the seemingly endless list of little expenses had added up quickly. The budget he’d naively drawn up before moving out now seemed hopelessly inadequate.

He picked up a flyer for a local gym, a membership he’d indulged in shortly after moving out. It seemed like a luxury now, a frivolous expense that needed to be cut. Sighing, he made a mental note to cancel it later. Next, he turned to his food expenses. Eating out, once a weekly treat, was now a thing of the past. He began scouring the internet for recipes that were both cheap and nutritious.

As days turned into weeks, Gabriel found himself working longer hours at the bookstore, hoping to boost his income. He started selling some of his old possessions online, items that once held sentimental value but were now deemed unnecessary. The apartment felt emptier, but his wallet felt slightly fuller.

The strain of budgeting began to seep into his social life. His friends, Charles included, still expected the same Gabriel who could afford to join them for drinks after work or spontaneous weekend getaways. Every time they made plans, Gabriel felt a pang of guilt and sadness. He couldn’t afford to be a part of their adventures anymore, and he hated the idea of letting them down.

Charles, sensing Gabriel’s growing distance, confronted him one evening after work. "Man, what’s going on with you lately? You’ve been distant, always finding excuses not to hang out," Charles said, concern etched on his face.

Gabriel hesitated, unsure of how to explain his newfound financial struggles without appearing weak. Then he remembered his own advice – honesty was key. He took a deep breath and shared his budgeting woes with Charles, hoping his friend would understand.

Charles nodded empathetically, "I get it, Gabe. Money’s tight for all of us at some point. We’ll find ways to make it work. Maybe we can have game nights at your place or go for hikes instead of hitting up bars."

Gabriel felt a wave of gratitude, but the shadow of Lina Sanchez still lingered in his thoughts. He wondered how their long-distance relationship would fare under the weight of his budgeting blues. Despite the challenges, he remained determined to find a way to make it work, both for himself and for Lina. After all, adjusting to this new life was just another obstacle to overcome, and he was not one to back down easily.

The air in Gabriel’s small apartment was heavy with the tension of unspoken words, a stark contrast to the usual carefree atmosphere that accompanied Richie and Jeremy’s visits. The two friends, always up for an adventure, seemed oblivious to the changes in Gabriel’s life. They laughed and joked as they helped with the unpacking, their laughter echoing through the empty rooms, amplifying Gabriel’s newfound responsibilities.

Gabriel had been working longer hours at the bookstore, sacrificing his leisure time to make ends meet. The decision to sell his old possessions weighed heavily on his heart, each item a reminder of a life he was leaving behind. He had confided in Richie and Jeremy about his budgeting struggles, hoping for understanding, but their lightheartedness only served to deepen the divide.

Feeling resentment simmering beneath the surface, Gabriel watched them, envious of their freedom. They seemed to have no worries, no responsibilities holding them back. The more he observed, the more he wished they could comprehend the gravity of his situation. But how could he make them understand when they were still living in a world of carefree whims?

One evening, as the three friends sat around the modest dinner table, the silence between them grew too loud to ignore. Gabriel’s frustration erupted in a burst of pent-up emotion. "You have no idea what it’s like," he said, his voice shaking. "I’m trying to keep my head above water here, but it feels like I’m drowning."

Richie and Jeremy exchanged a surprised look, their easy smiles faltering. They hadn’t realized the depth of Gabriel’s struggles or the extent to which their lifestyle was rubbing salt in his wounds. Attempting to bridge the gap, Jeremy spoke softly, "We didn’t mean to hurt you, Gabe. We just... we didn’t know."

Gabriel’s eyes met Jeremy’s, and for a moment, he saw the genuine concern in his friend’s gaze. The weight of their misunderstanding began to lift, but the rift was still palpable. It was a confrontation that marked the beginning of a difficult journey for all three, one where they would have to learn to balance their friendship with the realities of growing up.

In the midst of this turmoil, Gabriel couldn’t help but think of Charles, the kind-hearted man he had recently met at the bookstore. Charles, with his wisdom and empathy, seemed to understand the complexities of life in a way that Richie and Jeremy couldn’t yet grasp. Gabriel wondered if his new acquaintance could provide the guidance he needed to navigate these turbulent waters of friendship and responsibility.

Gabriel sat across from Lina at their cozy kitchen table, the warm glow of the overhead light casting a soft halo around her. He could see the exhaustion etched on her face, the shadows beneath her eyes a testament to the strain their recent distance had caused. He knew he had been neglectful, his longer hours at the bookstore leaving little room for the intimacy and connection they once shared.

"I’ve been aware of how I’ve been neglecting our relationship, Lina," Gabriel began, his voice filled with sincerity. "I’ve been so focused on work that I’ve let our bond slip away. That was a mistake, and I’m sorry."

Lina looked up at him, her own voice laced with hurt. "It’s not just the time, Gabriel. It’s the feeling that I’m not a priority anymore. That our problems and fears don’t matter as much as the books."

Gabriel took a deep breath, remembering the importance of honesty and trust he had often preached. "You’re right. I’ve been so consumed by the bookstore, trying to make it successful, that I’ve forgotten the foundation of our relationship. We need to talk about our struggles and support each other."

They delved into a heart-to-heart conversation, discussing their fears and insecurities, the pressures they both faced in their individual lives. Lina shared her longing for the simple moments they used to cherish, while Gabriel opened up about the mounting responsibilities at work.

Together, they devised a plan. Gabriel would set clear boundaries for his work hours, dedicating specific days and evenings for just the two of them. They would plan regular date nights, surprise each other with small gestures, and make a conscious effort to communicate daily, no matter how busy they were.

As the night wore on, the tension between them began to dissipate, replaced by a renewed sense of understanding and commitment. Gabriel vowed to prioritize their relationship, not just in words but in action, and Lina’s eyes shone with hope, knowing that their bond was on the path to mending.

With a new determination, they held hands, their fingers entwining in a promise of dedication and love. Gabriel felt a wave of relief wash over him, grateful for the chance to reconnect with Lina and rekindle the spark that had brought them together.

In the wake of the devastating apartment fire, Gabriel found himself sifting through the charred remains of his once-tidy life. The acrid smell of smoke still lingered in the air, a constant reminder of the disaster that had befallen him. As he picked up a blackened photo frame, the edges crumbled under his touch, revealing a barely recognizable snapshot of happier times. The weight on his heart grew heavier with each item he salvaged.

The landlord, a stern man with a thick mustache, met Gabriel the following day in what had become a desolate landscape of debris. They stood amidst the ruins, the words between them as cold as the ashes beneath their feet. Gabriel argued that the building’s outdated wiring had been the cause, while the landlord stubbornly insisted it was Gabriel’s negligence. The exchange quickly escalated into a heated argument, with accusations and defenses flying back and forth.

Gabriel’s frustration mounted as the conversation delved into the legal complexities of the situation. He realized that he might be held responsible for damages, a prospect that left him reeling. The potential financial burden loomed over him like a dark cloud, casting doubt on his ability to recover from this disaster.

As the reality of his circumstances set in, Gabriel’s guilt and sadness intertwined with a sense of determination. He had neglected some basic safety precautions, and now he was paying the price. The guilt ate at him, but he knew he had to move forward. Friends and neighbors, their own homes spared from the fire, offered words of comfort and help in the form of advice and assistance with selling his possessions.

Gabriel’s apartment, once a sanctuary, now felt cavernous and empty. But with each item he sold, he found a small sense of relief, as the slightly fuller wallet brought a glimmer of hope amidst the strain of budgeting. He knew the road ahead would be challenging, but the outpouring of support from those around him gave him the strength to face the journey.

In the midst of the chaos, Gabriel found gratitude in the unexpected kindness of others. It was a bittersweet realization, one that taught him the value of community and the fragility of life. The fire had taken so much from him, but it had also illuminated the resilience of his spirit and the enduring bonds he shared with those around him.

Gabriel sat in the warm glow of his parents’ living room, the familiar surroundings providing a small comfort amidst the turmoil in his life. The weight of recent events rested heavily on his heart, but he knew he couldn’t bear it alone anymore. With a deep breath, he began to share the story of the fire, the accusations from the landlord, and the overwhelming sense of guilt that had been haunting him.

His parents listened intently, their faces etched with concern. His mother’s hand found his, offering silent reassurance, while his father’s eyes reflected a mixture of anger and sympathy. They didn’t doubt Gabriel’s account, and their belief in him was a balm to his soul.

"You know, Gabriel," his father said, his voice firm, "nobody can blame you for something that’s out of your control. You’ve always been responsible, and this is just an unfortunate incident. Now, let’s focus on dealing with the situation."

His mother chimed in, "We’ll help you find a way through this. In the meantime, consider staying with us for a while. It’ll give you time to sort things out without the added stress of finding a new place right away."

Gabriel felt a wave of gratitude wash over him, grateful for their unwavering support. Friends from the neighborhood had also reached out, offering their assistance. Their kindness made him realize he wasn’t as alone as he had thought.

With newfound determination, Gabriel started researching temporary housing options, aided by his family and friends. They scoured online listings and local community boards, looking for affordable places that would accept short-term tenants. His best friend, Leo, even suggested they pool their resources to rent a small house together until Gabriel got back on his feet.

As the days passed, Gabriel found himself growing stronger, buoyed by the love and support around him. The guilt and resentment began to dissipate, replaced by a sense of resilience and hope. He knew that the road ahead would be challenging, but with his family and friends by his side, he was ready to face it head-on.

In the midst of the chaos, Lina stepped into the heart of the fire zone, her eyes searching for Gabriel amidst the swirling smoke and the roar of the flames. The once peaceful neighborhood was now an inferno, with firefighters battling to contain the blaze that threatened to consume everything in its path. Lina’s heart pounded with fear for Gabriel and their relationship, which had been hanging by a thread.

Gabriel appeared, his face smudged with soot and his eyes filled with concern. He noticed Lina’s trembling form and rushed to her side, enveloping her in a protective embrace. The flickering orange and red light from the burning buildings cast an eerie glow on their faces, amplifying the gravity of their conversation.

"I can’t do this anymore, Gabriel," Lina said, her voice wavering with emotion. "The fire, the stress, the neglect—it’s all taking a toll on us. I want us to be a priority, not just another item on your to-do list."

Gabriel’s gaze softened as he looked into Lina’s eyes. He knew she was right, that their bond had suffered in the face of his preoccupations. The fire, a symbol of the neglect and pitfall that had been festering in their lives, illuminated the urgency for change.

"I promise, Lina," he vowed, his voice sincere and resolute. "I’ll not only address the fire safety issues in our community but also in our relationship. We’ll make time for each other, and I’ll be more open about my fears and insecurities. Together, we’ll work through this."

Lina’s voice, still tinged with hurt, softened at Gabriel’s words. She remembered the longing she had expressed for simple moments, for the intimacy they once shared. It wasn’t about the quantity of time, but the quality.

"I believe you, Gabriel," she said, her hand finding his. "But actions speak louder than words. Let’s start by being honest and rebuilding what we’ve lost."

With a renewed sense of commitment, Gabriel and Lina stood together, their hands clasped tightly, as the firefighters battled the blaze. The fire raged on, but in their hearts, a spark of hope ignited, promising a future where love and safety would coexist.

Gabriel, Richie, and Jeremy worked diligently to restore the small apartment to its former glory, turning the once-damaged space into a warm and inviting home. The scent of fresh paint and sawdust filled the air as they moved from one task to another, laughter and conversation easing the tension that had built up between them.

Gabriel and Richie, brushes in hand, paused to discuss the misunderstandings that had strained their friendship. Gabriel explained the fear and insecurity that had kept him from being open about his struggles, while Richie admitted to his own obliviousness. They found common ground in their shared experiences of life’s challenges, and a new understanding began to form between them.

Jeremy, sensing the moment, softly shared his own story of overcoming adversity. His words were a balm to Gabriel’s soul, reminding him that they were all in this together. The three friends, now bonded by their vulnerabilities and the effort to rebuild, created an atmosphere of healing and support.

As the evening light filtered through the newly replaced windowpanes, the apartment took on a cozy glow. The once-bare walls were adorned with artwork, and the worn furniture had been restored to its former charm. Gabriel, looking around at the fruits of their labor, felt a profound sense of accomplishment and gratitude.

The weight of recent events still lingered, but with each passing moment, it felt lighter. The guilt and resentment began to fade, replaced by a resolve to face the future with hope and resilience. The emptiness of the apartment was now filled not just with physical objects, but with the warmth of friendship and a renewed sense of purpose.

Gabriel, Richie, and Jeremy sat together on the couch, a comfortable silence settling between them. There was a newfound peace in the air, one that spoke of reconciliation and the promise of a better tomorrow. As they exchanged warm smiles, Gabriel felt a wave of relief wash over him, knowing that he was no longer alone in his journey. The stress of budgeting and the weight of responsibility still existed, but now, it was tempered by the love and support of his friends.

With the sun beaming through his window, Gabriel started his day with a newfound determination. He sat at his kitchen table, penning out a weekly cleaning and self-care schedule, carefully allocating time for exercise, meditation, and quality moments with Lina. He understood that maintaining balance in his life was crucial, not just for himself but also for the wellbeing of their relationship.

Gabriel called Lina, inviting her over for a heart-to-heart conversation. As she arrived, her warm smile filled the room, but Gabriel could sense the underlying tension. They sat on the couch, their hands intertwined, as he began to share his insecurities. He spoke about the fear of losing her due to his recent neglect and the anxiety that sometimes overwhelmed him. Lina listened patiently, her eyes filled with empathy.

In return, Lina shared her own thoughts, acknowledging the strain recent events had put on them. She admitted to feeling distant from Gabriel but expressed her willingness to work through it together. They then set joint goals to strengthen their bond, such as having regular date nights, engaging in deep conversations, and supporting each other’s personal growth.

The air in the room seemed to lighten as they talked, the weight of their concerns slowly lifting. Gabriel felt a renewed sense of hope, remembering the importance of communication and trust, something he had often advised in his professional life. He vowed to prioritize their relationship, no longer letting work or other distractions come between them.

As the afternoon sun cast long shadows across the room, Lina leaned against Gabriel, their hearts beating in unison. They knew it wouldn’t be easy, but with honesty, understanding, and a commitment to rebuilding their connection, they were ready to face whatever challenges lay ahead. Together, they were determined to turn their newfound priorities into a stronger, more resilient love.

Gabriel sat by the window of his small apartment, the city lights flickering outside, a stark contrast to the turmoil brewing within. The incident, a fire sparked by an overlooked candle, had brought him face to face with the reality of apartment living. While his friends and neighbors had rallied around him, providing emotional support and helping with the aftermath, his landlord saw the situation differently. He was convinced that Gabriel’s negligence had caused the blaze, and there was talk of compensation for the damages.

Despite the pressure, Gabriel maintained his composure. He knew he had been careless, but he also believed in fairness. When the landlord finally agreed to meet, Gabriel prepared himself for a difficult negotiation. He wanted to rectify the situation without sacrificing his own stability. As the meeting began, Gabriel’s heart raced, but he remained steadfast, his voice steady as he presented his case.

"I understand the concerns, and I take responsibility for my actions," Gabriel said, his gaze unwavering. "But I hope we can find a middle ground that acknowledges the unforeseen circumstances and the efforts I’ve made to correct the situation."

The landlord, initially rigid, seemed to soften at Gabriel’s mature approach. They eventually reached an agreement that was fair to both parties. Gabriel would cover a portion of the damages through a payment plan, while the landlord would cover the rest as an acknowledgment of the unforeseen nature of the accident.

With the compensation settled, a weight lifted from Gabriel’s shoulders. The support from his friends and the community, along with his own resilience, had not only helped him navigate the crisis but also deepened his relationships. Lina, a neighbor who had been a source of strength, now looked at him with a newfound respect. Their conversations, once surface-level, had taken on a more profound meaning, as they both shared their fears and insecurities.

Gabriel’s friends, too, had shown their loyalty through the ordeal. They had stood by him, offering a listening ear and a helping hand when he needed it most. This experience had not only tested Gabriel’s character but had also forged stronger bonds between them.

As the chapter closed, the apartment that once represented a milestone and independence now seemed like a symbol of growth and resilience. Gabriel, having faced adversity and come out the other side, felt a renewed sense of purpose. The urge to drown in self-blame had been replaced by a determination to learn from his mistakes and embrace the connections he had forged. It was a new beginning, one marked by understanding, forgiveness, and the promise of a brighter future.

Gabriel and Lina sat on the worn couch in their small apartment, their hands entwined as they gazed into each other’s eyes. The room was dimly lit by a single lamp, casting warm shadows on the walls. They had been through so much together, and in that moment, they both knew their commitment to each other was stronger than ever.

"I want us to build a life based on love and understanding," Gabriel said softly, his voice filled with conviction. Lina smiled, her eyes shimmering with tears of joy. They had found solace in each other’s embrace, and their deepening love was a beacon of hope in the midst of life’s challenges.

Gabriel’s next step was to give back, and he started a support group for others who were struggling with the same issues he had. In a small community center, he shared his experiences, the lessons he’d learned, and the importance of vulnerability. The group dynamic was a mix of pain, hope, and resilience, as strangers became friends, united by their shared struggles.

Each week, the group would gather, their stories weaving a tapestry of human experience. There was John, a recovering alcoholic, who found strength in the support of others. There was Maria, who grappled with the loss of her job and the fear of the unknown. Their friendships grew, providing a sense of belonging and purpose that was invaluable to all.

As Gabriel’s involvement in the support group grew, so did his relationship with his family. They saw the changes in him, the newfound responsibility and growth. His parents, who had once been worried, now expressed pride in their son’s transformation. Gabriel’s sister, Sarah, reached out to him more often, their bond deepening as they talked about their shared experiences.

Gabriel’s apartment, once a symbol of his struggles, now felt like a haven of progress. The walls that once echoed with anxiety now housed laughter and meaningful conversations. The fire safety issues he vowed to address were slowly being rectified, and the relationship with Lina, which had been through its own trials, was on the mend.

With each passing day, Gabriel felt a new beginning taking shape. The road ahead was still rocky, but he was no longer walking alone. He had Lina, his friends from the support group, and a family that was slowly healing. Embracing the challenges, he knew that together, they would forge a future filled with love, understanding, and growth.

Gabriel sat across from Lina in their cozy apartment, the afternoon sun casting warm shadows across the room. The tension that had been hanging in the air for days seemed to have reached its peak, and Gabriel knew it was time to confront any remaining issues head-on. He took a deep breath and spoke softly, "Lina, I want you to know that I’ve been thinking a lot about our conversations, and I understand now how my anxiety has affected us both."

Lina’s trembling hands stilled as she looked into Gabriel’s sincere eyes. She felt a wave of relief wash over her, grateful for his newfound maturity and willingness to address their concerns. "Gabriel, I’m glad you’re seeing things clearly now. It’s been hard being apart, and I’ve missed you so much," she admitted, her voice steadier than before.

Together, they decided to tackle a specific challenge that had been causing strain in their relationship – the distance between them due to their demanding jobs. They brainstormed ways to bridge the gap, taking turns to suggest ideas and compromises. Their teamwork was evident, and they both felt a renewed sense of unity as they worked through the issue.

![](images/3d7fb070f729e14efc94316d750b1453728d94c5bf69e70805135afeef8ec75f.jpg)  
Table 6: The example story generated u by DOME.