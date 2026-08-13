# Mamba-Shedder: Post-Transformer Compression for Efficient Selective Structured State Space Models

J. Pablo Muñoz <sup>1</sup>\*, Jinjie Yuan<sup>2</sup>\*, Nilesh Jain<sup>1</sup> <sup>1</sup>Intel Labs, <sup>2</sup>Intel Corporation {pablo.munoz, jinjie.yuan, nilesh.jain}@intel.com

## Abstract

Large pre-trained models have achieved outstanding results in sequence modeling. The Transformer block and its attention mechanism have been the main drivers of the success of these models. Recently, alternative architectures, such as Selective Structured State Space Models (SSMs), have been proposed to address the inefficiencies of Transformers. This paper explores the compression of SSM-based models, particularly Mamba and its hybrids. We study the sensitivity of these models to the removal of selected components at different granularities to reduce the model size and computational overhead, thus improving their efficiency while maintaining accuracy. The proposed solutions, collectively referred to as Mamba-Shedder, achieve a speedup of up to 1.4x during inference, demonstrating that model efficiency can be improved by eliminating several redundancies with minimal impact on the overall model performance.

## 1 Introduction

We have seen an outstanding increase in the number of Transformer-based models (Vaswani et al., 2017) developed to tackle tasks from Natural Language Processing (NLP) and other domains (Parmar et al., 2018; Dosovitskiy et al., 2021; Arnab et al., 2021; Gong et al., 2021) due to their effectiveness at modeling sequences. However, these models also present critical efficiency challenges. For example, the cost of training these models scales quadratically in the sequence length. In the generation stage, Transformers, in their original form, require large caches to store the previously seen tokens. Several variants of Transformers have been proposed to address these efficiency challenges, but researchers have also explored alternative post-Transformer architectures to address these limitations. Structured state space models (SSMs), e.g.,

S4 (Gu et al., 2022), followed by Selective state space models, e.g., Mamba (Gu and Dao, 2023; Dao and Gu, 2024) have been proposed as efficient alternatives that achieve training time with linear scaling in sequence length, and during generation, maintain constant state size.

Model compression methods, e.g., pruning and quantization, have been broadly explored and applied to Transformer-based models. However, more must be done to explore compression in their structured state space counterparts. This paper explores the pruning of these alternative architectures, providing insights into potential opportunities to increase their efficiency without sacrificing accuracy. We discuss the following contributions:

• A pruning solution, Mamba-Shedder, which targets structures in selective structured state space models, improving their computational and memory efficiency.

• Comprehensive experiments to determine the tolerance of SSM-based models to the removal of their structures.

• Insights on how the differences in the SSM building blocks and their interaction with Transformer blocks in hybrid models affect the trade-off between efficiency and accuracy.

The rest of the paper is organized as follows: §2 provides details of the alternative architectures utilized in our study and popular strategies for element removal in large models. §3 describes methods to study network pruning in Mamba and hybrid architectures. §4 presents the results of our experiments and ablation studies, and we offer concluding remarks in §5. A Related Work section is included in the Appendix.

## 2 Preliminaries

## 2.1 State Space Models

State space models (SSMs) have a long history of modeling sequences and dynamic systems. Recently, structured SSMs, e.g., S4 (Gu et al., 2022), have been proposed as an alternative to Transformers because of their efficient capabilities for mapping input to output signals. When dealing with discrete sequences as in Natural Language Processing (NLP), the parameters A, B and C of these models are discretized to transform an input sequence, $x _ { t }$ and hidden state, $h _ { t } .$ , to obtain the output sequence, $y _ { t }$ . It can be formalized as:

$$
h _ { t } = A h _ { t - 1 } + B x _ { t } , y _ { t } = C ^ { \top } h _ { t } .\tag{1}
$$

Mamba: Selective State Space Models S4 and other structured SSMs are linear time-invariant (LTI), i.e., their parameters are fixed, limiting their effectiveness for sequence modeling. For instance, structured state space models fail in many contentand context-based reasoning tasks. These limitations have motivated the development of timevarying alternatives, e.g., Mamba (Gu and Dao, 2023), which incorporate selection mechanisms and are suitable for solving tasks previously SSM generations failed. Specifically, Mamba’s SSM module, S6, allows its parameters to depend on the input, thereby modifying the formulation from time-invariant to time-varying. A second improvement proposed in Mamba compared to previous SSMs is a hardware-aware algorithm that speeds up execution while reducing memory IOs.

Furthermore, Mamba-2 (Dao and Gu, 2024) improves the original Mamba architecture by proposing state space duality (SSD), which improves its efficiency on hardware accelerators compared to S6. This improvement is achieved by changing the state matrix, A, which directly controls the latent state, h. A is modified from being structured as a diagonal matrix to a formulation that utilizes a scalar-times-identity structure. Additionally, Mamba-2 introduces the concept of heads in SSMs inspired by how multi-head attention (MHA) works and implementing a grouped-value attention (GVA) head structure. Overall, the Mamba-2 architecture, with its SSD core component, allows for improved parallelism of the block’s projections.

Mamba block Mamba models comprise several blocks stacked after each other. Figure 1 on the left illustrates a single Mamba block. Each block has the selective SSM mechanism (S6 for Mamba-1 and SSD for Mamba-2) at its core, placed within a larger structure that combines a gated multilayer perceptron (MLP), a convolution, and SILU activation functions (Elfwing et al., 2018).

For more details about selective structured state space models, we refer the reader to Gu and Dao (2023) and Dao and Gu (2024).

## 2.2 Hybrid Models

Lately, new models achieve the best of both worlds (Transformers and Selective SSMs) by proposing architectures with both classes of blocks. Zamba (Glorioso et al., 2024) is one example of such a hybrid model. It combines the strengths of Mamba’s backbone and the efficiency of selective SSMs with a shared Transformer block that incorporates Transformers’ powerful in-context learning capabilities. The shared attention mechanism, in which two attention blocks are reused and interleaved in an ABAB pattern throughout the network, is a characteristic innovation of Zamba. This model also applies LoRA adapters (Hu et al., 2022) to the shared MLP blocks, achieving specialization when interacting with the affected layers, memory efficiency, and faster inference with reduced computational overhead.

Another example of a hybrid model is Hymba (Dong et al., 2024). This model takes a different approach than Zamba, proposing an entirely new hybrid-head module, illustrated in Figure 1 on the right, in which the SSM and Attention mechanisms contribute in parallel to the sequence modeling. Additionally, Hymba benefits from group query attention, cross-layer KV cache sharing, and learnable meta-tokens, resulting in higher throughput, reduced memory requirements, and competitive performance compared to models of similar size.

## 2.3 Model Pruning

A popular model compression technique, pruning (LeCun et al., 1989), has been effectively used to reduce the size of deep learning models and improve their efficiency. Network pruning operates at two levels: (1) Unstructured pruning identifies the importance of individual weights that can be masked to minimize their impact on overall model behavior. At a different level, (2) structured pruning focuses on removing more significant structural components of the model, such as whole Transformer blocks (Men et al., 2024), or reducing the granularity to target subcomponents of these layers (Zhong et al., 2024). Other dimensions for pruning include groups of channels in the Transformer’s MLPs or heads from the MHA layer (Muñoz et al., 2025). In this paper, the focus is solely on structured pruning applied to Mamba-based models. Next, we discuss Mamba-Shedder’s methodology to study redundancies in Mamba and hybrid models.

![](images/672fd8e9bc4cba144f22faba33aaf46c1c899178cfa9aac41f7b8a2bf5ae3498.jpg)  
Figure 1: Overview of Mamba-Shedder. This figure illustrates the pruning strategy for three types of Mamba-based models. The first type includes Mamba models such as Mamba-1 (Gu and Dao, 2023), Mamba-2 (Dao and Gu, 2024), and Falcon-Mamba (Zuo et al., 2024). The second type comprises Mamba + Transformers architectures, including Zamba (Glorioso et al., 2024). The third type is Hymba (Dong et al., 2024), a novel architecture with hybrid heads. Red dashed lines indicate potential removal. In Transformers, channel pruning can also be applied to MLP block (width pruning).

## 3 Methodology

Due to the large sizes of current state-of-the-art sequence models, Mamba-Shedder requires an efficient strategy to identify structures that can be removed without significantly affecting the model’s accuracy. We approach this problem using a training-free approach, in which the least essential elements are considered for removal. Similar strategies have been explored in Transformer-based large language models (Ashkboos et al., 2024; Men et al., 2024; Zhong et al., 2024). However, to our knowledge, no study explores the removal of structures in Selective Structured State Space models. Mamba-Shedder conducts structure removal of Mamba models and their hybrid variants at different granularities. As illustrated in the left of Figure 1, in the case of models with only Mamba blocks, we explore the iterative removal of entire Mamba blocks (§2.1), or their SSM subcomponents, either S6 or SSD modules depending on the version of Mamba (Figure 1).

The proponents of the Mamba architecture do not provide a rationale for the number of Mamba blocks required to build robust models, opening an opportunity for Mamba-Shedder to investigate whether some components might be redundant and hence removed from the model with a minor impact in accuracy.

In addition to these components, in the case of hybrid models that also contain Transformer blocks (middle of Figure 1), we also explore the removal of entire Transformer blocks or their subblocks: multilayer perceptrons (MLP) modules and multihead attention (MHA) modules. In hybrid models, Mamba-Shedder also explores the removal of structures at a finer granularity by targeting groups of channels in the MLP’s linear layers, i.e., based on a channel group size, g, Mamba-Shedder explores the removal of ng channels, where n is the number of groups that could be removed based on their impact of the overall model performance.

Algorithm 1 Block / Module Pruning   
Input: Set of blocks/modules from a model m, Calibration   
dataset , Metric ϕ, Target pruning steps t.   
Output: Pruned model m   
1: for k 1 to t do   
2: for all M do   
3: S<sub>i</sub> Importance(M<sub>i</sub>, m, , ϕ)   
4: end for   
5: Mmin arg min<sub>M</sub> S<sub>i</sub>   
M ← M \ {<sup>M</sup>min} ▷ Block/Module Pruning   
7: end for   
8: return $m ^ { * }$ with the remaining blocks/modules in

Algorithm 1 details the procedure to remove entire structures, e.g., Mamba or Transformer blocks, MLPs, MHA, or SSM modules. Given a set of structures selected for potential removal, a proxy data set C and a metric ϕ are used to measure the importance of an individual structure and the impact of removing it from the model (Zhong et al., 2024). In addition to entire structures, Mamba-Shedder follows the same logic to remove channel groups as detailed in Algorithm 2.

![](images/d6e884bbe9056df9ed8d1fb2b8c49acae5d0a1aa5d0656713fdfb7a8dc76788b.jpg)

![](images/8d5aa9f0115ce293a47e77c6abcd55b502316c8a02a243d20cada2153bf22327.jpg)

![](images/f6342cb021d5578e7b4544f298af9efd261df40096682bc139a7a9d9c953b2a4.jpg)

Figure 2: Pruning Mamba blocks. Avg. Accuracy indicates the average accuracy for seven tasks. The model composed of Mamba 1 blocks (left) can tolerate the removal of entire blocks without significantly increasing its perplexity or decreasing accuracy compared to Mamba-2 and Zamba-2. In all three models, removing each Mamba block reduces 0.04B parameters from the model. These are training-free results, and drops in accuracy can be reduced by a subsequent fine-tuning stage (§4.5).
<table><tr><td>Model</td><td>Method</td><td>Num. of Pruned Mamba Blocks</td><td>Ratio</td><td>Lambada PPL (↓)</td><td></td><td></td><td></td><td></td><td></td><td>Lambada HellaS PIQA ARC-e ARC-c WinoG OBQA Average</td><td></td><td></td></tr><tr><td rowspan="3">Mamba-2.8B</td><td rowspan="3">Dense</td><td>0/64</td><td>0%</td><td>4.23</td><td>69.2</td><td>66.1</td><td>75.2</td><td>69.7</td><td>36.3</td><td>63.5</td><td>39.6</td><td>59.9</td></tr><tr><td>7/64</td><td>10.43%</td><td> $\underline { { \phantom { - } 4 . 9 4 _ { + 0 . 7 1 } } }$ </td><td>65.8</td><td>63.7</td><td>73.8</td><td>68.0</td><td>33.5</td><td>62.5</td><td>36.8</td><td> $5 7 . 7 _ { - 2 . 2 }$ </td></tr><tr><td>14/64</td><td>20.86%</td><td> $\underline { { 7 . 5 1 _ { + 3 . 2 8 } } }$ </td><td>58.9</td><td>57.6</td><td>71.0</td><td>62.7</td><td>32.0</td><td>61.1</td><td>33.2</td><td> $\underline { { 5 3 . 8 . 6 . 1 } }$ </td></tr><tr><td rowspan="3">Mamba2-2.7B</td><td rowspan="2">Dense</td><td>0/64</td><td>0%</td><td>4.10</td><td>69.7</td><td>66.6</td><td>76.4</td><td>69.6</td><td>36.4</td><td>64.0</td><td>38.8</td><td>60.2</td></tr><tr><td>7/64</td><td>10.42%</td><td> $\overline { { { \bf 8 . 4 3 } } } _ { + 4 . 3 3 } ^ { - }$ </td><td>53.0</td><td>63.8</td><td>73.9</td><td>66.6</td><td>36.4</td><td>64.5</td><td>35.0</td><td> $\overline { { 5 6 . 2 . 4 . 0 } }$ </td></tr><tr><td>Mamba Block Pruning</td><td>14/64</td><td>20.83%</td><td> $\mathbf { 1 1 . 5 3 _ { + 7 . 4 3 } }$ </td><td>47.0</td><td>59.4</td><td>71.1</td><td>60.6</td><td>35.6</td><td>60.8</td><td>35.0</td><td> ${ \bf 5 2 . 8 . } _ { 7 . 4 }$ </td></tr><tr><td rowspan="2">Zamba2-2.7B</td><td>Dense</td><td>0/54</td><td>0%</td><td>4.01</td><td>69.7</td><td>77.0</td><td>79.8</td><td>77.5</td><td>48.5</td><td>72.1</td><td>45.8</td><td>67.2</td></tr><tr><td>Mamba Block Pruning</td><td>7/54</td><td>10.38%</td><td> $\overline { { 6 . 8 0 } } _ { + 2 . 7 9 } ^ { - }$ </td><td>58.9</td><td>69.7</td><td>77.0</td><td>69.8</td><td>39.6</td><td>67.0</td><td>41.8</td><td> $\overline { { { \bf 6 0 . 5 } } } _ { \cdot 6 . 7 } ^ { - }$ </td></tr><tr><td></td><td></td><td>14/54</td><td>20.77%</td><td> $\mathbf { 1 5 . 8 _ { + 1 1 . 7 9 } }$ </td><td>44.3</td><td>62.8</td><td>72.7</td><td>54.3</td><td>34.5</td><td>64.3</td><td>37.2</td><td> $\pmb { 5 2 . 9 . 1 4 . 3 }$ </td></tr></table>

Table 1: Detailed results of Mamba-Shedder with training-free Mamba block pruning. Lambada, HellaS, PIQA, ARC-e, ARC-c, WinoG, and OBQA represent their respective accuracies. Underlined numbers indicate the smallest average accuracy gap with the dense model under the same level of pruning

Algorithm 2 MLP Channel Pruning   
Input: Set of MLP blocks $\mathcal { M } _ { \mathrm { M L P } }$ from a model m, Calibration   
dataset , Metric $\phi ,$ Target pruning steps t, MLP channel   
group size g.   
Output: Pruned model $m ^ { * }$   
1: for $k \gets 1$ to t do   
2: for all $M _ { i } \in \mathcal { M } _ { \mathrm { M L P } }$ do   
3: $S _ { i } \gets$ Importance $( M _ { i } [ : , : - g ] , m , \mathcal { C } , \phi )$   
4: end for   
5: $M _ { \mathrm { { m i n } } } = { \mathrm { : } }$ arg min $M _ { i } \in \mathcal { M } _ { \mathrm { M L P } } \ S _ { i }$   
6: $M _ { \mathrm { m i n } } = M _ { \mathrm { m i n } } [ : , : - g ]$ ▷ Channel Pruning   
7: end for   
8: return $m ^ { * }$ with the altered MLP blocks in

Depending on the pruning objective, Mamba-Shedder might treat these pruning targets in isolation, but Section 4 also presents the results of configurations in which Mamba-Shedder sequentially prunes larger structures (e.g., Mamba blocks) and, at a later stage, smaller components, e.g., SSM modules in the remaining Mamba blocks. Future work will explore larger search spaces with more complex configurations of candidate structures for removal. For example, the importance of Mamba blocks and their SSM modules can be assessed in the same pruning iteration.

## 4 Experiments

We evaluate Mamba-Shedder and study the removal of structures from SSM-based models utilizing several open-source models and datasets. We analyze their absolute and relative drop in accuracy and quantify the inference speedup obtained by the pruned models. Next, we discuss the resources utilized for our experiments and details of our setup and results.

## 4.1 Models

Our experiments employed the following pretrained Mamba and hybrid models: Mamba-2.8b (Gu and Dao, 2023), consists of 64 S6 blocks<sup>1</sup>. Mamba2-2.7b (Dao and Gu, 2024), consists of 64 SSD blocks <sup>2</sup>. Both Mamba models were trained on 300B tokens on the Pile dataset (Gao et al., 2020). For our choice of a hybrid model, we explored Zamba2-2.7B (Glorioso et al., 2024)<sup>3</sup>. It has 54 layers, including 45 single Mamba-2 Blocks and 9 hybrid layers composed of both Mamba-2 Blocks and Transformer Blocks. Zamba-2 was trained on 3T tokens from open web datasets, including Zyda (Tokpanov et al., 2024), and subsequently annealed with 100B additional tokens. The aforementioned models are all of the same size and can be compared directly. For Mamba models of different sizes, we also explored Falcon-Mamba-7B (Zuo et al., 2024)<sup>4</sup>, which is based on the Mamba-1 architecture and is the best-performing Mamba model at this scale in the literature, as well as Hymba-1.5B-Base (Dong et al., 2024)<sup>5</sup>, which features a hybrid architecture incorporating both Mamba and Attention heads.

![](images/bf75d875848502b94d5e56fa03d8b8aa451eb20474a396509f1d00bae37517bb.jpg)

![](images/46bad657145ab6345fd4efd4747bdc92f4c4c4df881535bcbe6ff851275dcce0.jpg)

![](images/8832c965f380ee64e071b6c81af2c53b9cc66eebc6405fcd9bcdd6751395a061.jpg)

Figure 3: Pruning SSM (S6 and SSD modules). Mamba-2.8B and Mamba2-2.7B have 64 SSM modules, while Zamba2-2.7B has 54 SSM (SSD) modules. Avg. Accuracy is for the seven tasks evaluated.
<table><tr><td>Model</td><td>Method</td><td>Num. of Pruned SSMs</td><td>Lambada PPL (↓)</td><td>Lambada</td><td>HellaS</td><td>PIQA</td><td>ARC-e</td><td>ARC-c</td><td>WinoG</td><td>OBQA</td><td>Average</td></tr><tr><td rowspan="4">Mamba-2.8B</td><td>Dense</td><td>0/64</td><td>4.23</td><td>69.2</td><td>66.1</td><td>75.2</td><td>69.7</td><td>36.3</td><td>63.5</td><td>39.6</td><td>59.9</td></tr><tr><td rowspan="2">SSM Pruning</td><td>16/64</td><td> $\overline { { \mathbf { 9 . 2 3 } } } _ { + 5 . 0 0 } ^ { - }$ </td><td>55.2</td><td>52.1</td><td>68.1</td><td>57.8</td><td>28.4</td><td>55.6</td><td>31.6</td><td> $\overline { { 4 9 } } . \overline { { 8 } } _ { - 1 0 . 1 } \overline { { } }$ </td></tr><tr><td>20/64</td><td> $\mathbf { 1 0 . 1 0 _ { + 5 . 8 7 } }$ </td><td>57.1</td><td>48.2</td><td>65.5</td><td>50.9</td><td>25.9</td><td>56.0</td><td>29.4</td><td> $\mathbf { 4 7 . 6 . } _ { 1 2 . 3 }$ </td></tr><tr><td></td><td>24 /64</td><td> $2 2 . 5 5 _ { + 1 8 . 3 2 }$ </td><td>44.4</td><td>43.2</td><td>64.4</td><td>47.4</td><td>25.8</td><td>53.6</td><td>29.8</td><td>44.1.15.8</td></tr><tr><td rowspan="4">Mamba2-2.7B</td><td>Dense</td><td>0/64</td><td>4.10</td><td>69.7</td><td>66.6</td><td>76.4</td><td>69.6</td><td>36.4</td><td>64.0</td><td>38.8</td><td>60.2</td></tr><tr><td></td><td>16/64</td><td> $4 . 2 6 _ { + 0 . 1 6 }$ </td><td>66.9</td><td>66.1</td><td>76.4</td><td>68.6</td><td>37.2</td><td>64.0</td><td>39.2</td><td>59.8.0.4</td></tr><tr><td rowspan="2">SSM Pruning</td><td>20/64</td><td> $\pmb { 5 . 8 9 _ { + 1 . 7 9 } }$ </td><td>59.8</td><td>66.0</td><td>76.1</td><td>68.9</td><td>36.7</td><td>63.6</td><td>39.2</td><td>58.6.1.6</td></tr><tr><td>24/64</td><td> $\mathbf { 1 4 . 9 5 _ { + 1 0 . 8 5 } }$ </td><td>43.4</td><td>65.8</td><td>74.8</td><td>67.1</td><td>36.6</td><td>62.9</td><td>38.0</td><td>55.5.4.7</td></tr><tr><td rowspan="4">Zamba2-2.7B</td><td>Dense</td><td>0/54</td><td>4.01</td><td>69.7</td><td>77.0</td><td>79.8</td><td>77.5</td><td>48.5</td><td>72.1</td><td>45.8</td><td>67.2</td></tr><tr><td rowspan="3">SSM Pruning</td><td>16/54</td><td> $\underline { { 4 . 1 4 _ { + 0 . 1 3 } } }$ </td><td>69.2</td><td>75.8</td><td>79.2</td><td>75.8</td><td>46.5</td><td>72.2</td><td>45.8</td><td>66.4.0.8</td></tr><tr><td>20/54</td><td> $\underline { { \pmb { 5 . 0 7 } } } _ { + 1 . 0 6 }$ </td><td>64.2</td><td>75.8</td><td>79.3</td><td>75.5</td><td>46.2</td><td>73.2</td><td>46.0</td><td>65.7-1.5</td></tr><tr><td>24/54</td><td> $\underline { { 5 . 4 6 _ { + 1 . 4 5 } } }$ </td><td>62.3</td><td>74.7</td><td>79.0</td><td>75.4</td><td>44.3</td><td>70.9</td><td>46.4</td><td>64.7.2.5</td></tr></table>

Table 2: Detailed results of Mamba-Shedder with training-free SSM pruning. The remaining tasks represent their respective accuracy. Here, we do not consider the pruning ratio, as the number of SSM’s parameter weights is small. Its benefit is the reduction of computational overhead. Underlined numbers indicate the smallest gap with Dense under the same level of pruning.

## 4.2 Datasets

Following the language modeling evaluation of Mamba (Gu and Dao, 2023; Dao and Gu, 2024), we utilize lm-eval-harness (Gao et al., 2023) to assess the zero-shot performance, which includes measuring perplexity on Lambada (Paperno et al., 2016), and accuracy on the following downstream tasks: HellaSwag (Zellers et al., 2019), Physical Interaction Question Answering (PIQA) (Bisk et al., 2020), AI2 Reasoning Challenges (Arc-e, Arc-c) (Clark et al., 2018), Large-scale Winograd Schema Challenge (WinoGrande) (Sakaguchi et al., 2021), and the Open Book Question Answering (Mihaylov et al., 2018) dataset.

Regarding the calibration dataset, we follow BlockPruner (Zhong et al., 2024) in using the Alpaca dataset <sup>6</sup> as the calibration dataset and employ perplexity as the metric for calculating importance scores. All the hyperparameters used in our experiments are detailed in the Appendix.

## 4.3 Results

## 4.3.1 Pruning Target: Mamba Block

This section explores the impact of pruning Mamba blocks on model performance. Figure 2 and Table 1 present the results of applying Mamba-Shedder to Mamba-2.8B, Mamba2-2.7B, and Zamba2-2.7B models with a focus on removing redundant entire Mamba blocks. The model that utilizes the first version of Mamba blocks (S6) appears to tolerate a higher number of removed blocks without significantly affecting its performance. Specifically, the Mamba-2.8B model demonstrates robustness, with its perplexity (PPL) increasing from 4.23 to 7.51 and average accuracy dropping from 59.9 to 53.8 when the pruning ratio reaches 20.86%. In contrast, the Mamba2-2.7B and Zamba2-2.7B models exhibit more significant performance degradation, although they performed better before pruning (Dense). The poorer pruning performance of Zamba2-2.7B may be attributed to the pruning of Mamba blocks disrupting a certain balance within the hybrid layers. Overall, the effects of Mamba block pruning vary across different models, depending on the model architecture and the characteristics of the pre-training stage.

<table><tr><td>Pruning Target</td><td>Ratio</td><td>Additional Lambada (Block, Width) Pruned SSMs PPL (↓)</td><td></td><td>Lambada HellaS PIQA ARC-e ARC-c WinoG OBQA</td><td></td><td></td><td></td><td></td><td></td><td></td><td>Avg.</td></tr><tr><td>1</td><td>0%</td><td>0/54</td><td>4.01</td><td>69.7</td><td>77.0</td><td>79.8</td><td>77.5</td><td>48.5</td><td>72.1</td><td>45.8</td><td>67.2</td></tr><tr><td>Mamba Block &amp; Transformer Block</td><td>10.40%</td><td>0/54</td><td> $9 . 1 8 _ { + 5 . 1 7 }$ </td><td>53.5</td><td>67.3</td><td>76.3</td><td>63.5</td><td>37.8</td><td>64.3</td><td>40.6</td><td> $5 7 . 6 _ { - 9 . 6 }$ </td></tr><tr><td>Mamba Block &amp; MLP &amp; MHA</td><td>10.33%</td><td>0/54</td><td> ${ \bf 5 . 0 1 _ { + 1 . 0 0 } }$ </td><td>65.6</td><td>73.6</td><td>78.5</td><td>75.3</td><td>43.8</td><td>69.3</td><td>45.2</td><td> $6 4 . 5 _ { - 2 . 7 }$ </td></tr><tr><td>Mamba Block &amp; MLP &amp; MHA + MLP Channel</td><td>10.27%</td><td>0/54</td><td> $5 . 4 5 _ { + 1 . 4 4 }$ </td><td>63.4</td><td>74.9</td><td>80.1</td><td>79.0</td><td>49.7</td><td>70.9</td><td>46.0</td><td> ${ \bf 6 6 . 3 . _ { 0 . 9 } }$ </td></tr><tr><td>Mamba Block &amp; MLP &amp; MHA + MLP Channel + SSM</td><td>10.27%</td><td>18/54</td><td> $5 . 1 8 _ { + . 1 . 1 7 }$ </td><td>63.4</td><td>73.9</td><td>80.0</td><td>79.0</td><td>48.7</td><td>69.5</td><td>46.6</td><td> $6 5 . 9 . 1 . 3 $ </td></tr><tr><td>Mamba Block &amp; Transformer Block</td><td>15.89%</td><td>0/54</td><td> $1 0 . 3 8 _ { + . 6 . 3 7 }$ </td><td>51.4</td><td>65.6</td><td>74.0</td><td>61.7</td><td>37.7</td><td>63.5</td><td>39.6</td><td> $5 6 . 2 _ { - 1 1 . 0 }$ </td></tr><tr><td>Mamba Block &amp; MLP &amp; MHA</td><td>15.54%</td><td>0/54</td><td> $1 0 . 6 4 _ { + . 6 . 6 3 }$ </td><td>49.3</td><td>69.2</td><td>76.9</td><td>66.1</td><td>38.1</td><td>66.0</td><td>41.8</td><td> $5 8 . 2 _ { - 9 . 0 }$ </td></tr><tr><td>Mamba Block &amp; MLP &amp; MHA + MLP Channel</td><td>15.48%</td><td>0/54</td><td> $7 . 3 9 _ { + . 3 . 3 8 }$ </td><td>57.6</td><td>70.0</td><td>78.5</td><td>74.5</td><td>43.9</td><td>67.5</td><td>43.8</td><td> ${ \bf 6 2 . 3 . _ { 4 . 9 } }$ </td></tr><tr><td>Mamba Block &amp; MLP &amp; MHA + MLP Channel + SSM</td><td>15.48%</td><td>18/54</td><td> $7 . 4 3 _ { + . 3 . 4 2 }$ </td><td>56.5</td><td>68.9</td><td>77.9</td><td>73.4</td><td>41.8</td><td>67.7</td><td>42.8</td><td> $6 1 . 3 . 5 . 9$ </td></tr></table>

Table 3: Results of Zamba2-2.7B were achieved by pruning its Mamba-2 and Transformers blocks at multiple granularities, including entire Mamba-2 block, MHA block, MLP block, MLP channel, and SSM module. The remaining tasks represent their respective accuracies. $\because \&$ indicates that the pruning targets are considered together in the same pruning step, while “+” signifies the distinction between pruning stages, with pruning occurring sequentially. Bold numbers indicate the best performance under the same level of pruning (excluding Dense).

## 4.3.2 Pruning Target: SSM Module

In this section, we delve into assessing the impact of pruning only the SSM modules within Mamba blocks on the performance of various models, as illustrated in Table 2 and Figure 3. When using the same target in Mamba-2.8B, we observe that further pruning SSMs results in a noticeable increase in perplexity, soaring to 22.55 and decreasing average accuracy to 44.1. This result indicates a significant sensitivity to SSM pruning for Mamba-1, where performance degradation is pronounced even at moderate pruning levels. Conversely, Mamba2-2.7B and Zamba2-2.7B exhibit remarkable resilience to SSM pruning. Even with 24 SSMs pruned, the model maintains a relatively stable performance. This robustness suggests that Mamba-2 blocks can tolerate higher SSM module pruning, potentially due to Mamba-2’s optimizations or different training strategies with Mamba-1. The Zamba2-2.7B model, with the hybrid architecture, outperforms both Mamba-1 and Mamba-2. Pruning 12 out of its 54 SSMs results in a negligible PPL increase from 4.01 to 4.02, while the average accuracy slightly decreases from 67.2% to 67.0%. The hybrid nature of Zamba2-2.7B may contribute to its ability to maintain performance despite SSM pruning. Overall, these findings underscore the importance of model architecture and training strategies in determining the impact of SSM pruning. They offer valuable insights for optimizing model efficiency without compromising performance.

## 4.3.3 Pruning Target: Finer-grained removal of Mamba and Transformer blocks, and their subcomponents

Table 3 presents the results of pruning various components of the Zamba2-2.7B model, including combinations of Mamba-2 blocks, entire Transformer blocks, and their subcomponents, i.e., MHA blocks, MLP blocks, MLP channels, and SSM modules. We design four search spaces to study the effectiveness of different granularities and their combinations. “&” indicates that the pruning targets are considered together in the same pruning step, while “+” signifies the distinction between pruning stages, with pruning occurring sequentially:

Mamba Block & Transformer Block Pruning This experiment involves pruning the entire Mamba-2 blocks and Transformer blocks.

Mamba Block & MLP & MHA Pruning This experiment decomposes the transformer block into sub-blocks, pruning Mamba-2 blocks as well as MHA and MLP.

Mamba Block & MLP & MHA + MLP Channel Pruning This experiment prunes the Mamba-2 blocks, MHA, and MLP at the first stage and further prunes the MLP channels at the next stage.

Mamba Block & MLP & MHA + MLP Channel Pruning + SSM Add additional SSM pruning following the previous solution.

The results indicate that pruning Mamba blocks and Transformer blocks alone leads to significant performance degradation. However, more granular pruning strategies show a more favorable trade-off between pruning ratio and performance. Specifically, pruning Mamba blocks, MLP, MHA (single stage), and MLP channels subsequently performs the best. Inspired by the SSM pruning of Mamba-2 in Section 4.3.2, we further add SSM pruning to the third strategy, and the results show that removing around 18 SSMs can maintain accuracy performance while reducing computational overhead. An interesting finding is that pruning SSMs can even lower PPL; for instance, at a 10% pruning ratio, PPL decreases from 5.45 to 5.18, suggesting that some SSM modules are redundant after the second pruning stage. Overall, these findings indicate that multi-granularity pruning methods, particularly those including MLP channels and SSM modules, can effectively reduce the complexity of hybrid Mamba models while maintaining a higher level of performance.

## 4.3.4 Pruning Mamba Models of Other Sizes

Hymba Table 4 shows the results of Mamba-Shedder with training-free Hymba Block pruning for Hymba-1.5B-Base. The dense configuration achieves an average accuracy of 63.8, which decreases as more blocks are pruned, dropping to 60.5 when 8 blocks are pruned, indicating a general decline in performance across benchmarks. Further analysis of inference acceleration and recovery tuning experiments for Hymba-1.5B-Base will be discussed in the subsequent sections.

Falcon-Mamba While the previous sections focused on exploring the pruning of Mamba models with sizes around 2.7B or 2.8B, we also investigated the impact of Mamba-Shedder on a largerscale Mamba model, specifically Falcon-Mamba-7B (Table 5). Pruning SSM modules in the Falcon-

Mamba-7B model shows better tolerance in terms of perplexity, suggesting that SSM pruning is more effective in maintaining lower perplexity. Regarding average accuracy, pruning entire Mamba blocks is more beneficial.

Additionally, it is important to note that pruning entire Mamba blocks yields more significant computational benefits than SSM pruning, suggesting that while SSM pruning is advantageous for maintaining perplexity, pruning Mamba blocks offers a better trade-off between computational efficiency and accuracy. The choice of pruning strategy should be guided by the specific performance metric of interest and the desired balance between computational efficiency and model accuracy.

None of the above results have undergone finetuning to improve the performance of the pruned models. As in other works, the drop in the accuracy performance of pruned models can be recovered by fine-tuning, which will be incorporated in §4.5.

## 4.4 Inference Acceleration

Through the above analysis, we have gained a good understanding and insights into the impact of Mamba-Shedder’s structured pruning on model accuracy and perplexity performance. In this section, we discuss the impact of pruning on inference acceleration. All the following tests were conducted on a single Tesla V100 32GB GPU.

Mamba-1 When removing entire Mamba blocks, as shown in Table 6, Mamba-Shedder speeds up the decoding stage up to 1.29x when removing 14 blocks, and 1.13x when removing only 7 blocks, which highlights the potential of Mamba-Shedder to optimize computational efficiency in Mamba models. The user’s decision on how aggressively to prune will impact the average accuracy or the perplexity as observed in Table 1.

Mamba-2 As detailed in Table 7, removing 24 SSM modules (44% of the total number of modules) results in up to a 1.20x speedup in the prefill stage and a 1.18x speedup in the decoding stage during of inference. A more conservative pruning ratio achieves 1.11x speedup when removing 16 SSM modules. Based on previous observations, the impact on performance metrics is minimal (0.4% for accuracy and 0.16 for PPL). These results underscore the effectiveness of SSM pruning in enhancing computational efficiency while barely affecting model performance, making it a viable strategy for optimizing Mamba models.

<table><tr><td>Model</td><td>Method</td><td>Num. of Pruned Hymba Blocks</td><td></td><td></td><td>HellaS PIQA ARC-e ARC-c WinoG Average</td><td></td><td></td><td></td></tr><tr><td rowspan="4">Hymba-1.5B-Base</td><td rowspan="2">Dense</td><td>0/32</td><td>53.5</td><td>77.1</td><td>76.6</td><td>45.4</td><td>66.1</td><td>63.8</td></tr><tr><td>6/32</td><td>50.5</td><td>75.8</td><td>76.0</td><td>44.9</td><td>64.1</td><td>62.3</td></tr><tr><td rowspan="2">Hymba Block Pruning</td><td>7/32</td><td>49.9</td><td>74.9</td><td>74.8</td><td>43.9</td><td>64.9</td><td>61.7</td></tr><tr><td>8/32</td><td>49.2</td><td>74.3</td><td>74.2</td><td>43.2</td><td>61.5</td><td>60.5</td></tr></table>

Table 4: Results of Mamba-Shedder with training-free Hymba block pruning for Hymba-1.5B-Base (Dong et al., 2024). Five commonsense reasoning tasks are used for evaluation.
<table><tr><td>Model</td><td>Method</td><td>Num. of Pruned Mamba Blocks / SSMs</td><td>Lambada PPL (↓)</td><td></td><td></td><td>Lambada HellaS PIQA ARC-e ARC-c WinoG</td><td></td><td></td><td></td><td></td><td>OBQA Average</td></tr><tr><td rowspan="8">Falcon-Mamba-7B</td><td rowspan="4">Dense</td><td>0/64</td><td>3.15</td><td>74.3</td><td>80.3</td><td>82.0</td><td>84.4</td><td>58.9</td><td>75.1</td><td>49.0</td><td>72.0</td></tr><tr><td>5/64</td><td>4.01</td><td>69.2</td><td>78.6</td><td>81.9</td><td>82.2</td><td>54.6</td><td>72.5</td><td>47.6</td><td>69.5</td></tr><tr><td>10/64</td><td>4.97</td><td>65.1</td><td>75.0</td><td>79.5</td><td>79.7</td><td>51.5</td><td>70.2</td><td>43.8</td><td>66.4</td></tr><tr><td>15/64</td><td>5.63</td><td>62.4</td><td>71.2</td><td>77.8</td><td>76.1</td><td>49.1</td><td>70.2</td><td>41.8</td><td>64.1</td></tr><tr><td></td><td>20/64</td><td>39.31</td><td>31.5</td><td>65.9</td><td>74.3</td><td>72.2</td><td>42.3</td><td>65.2</td><td>38.4</td><td>55.7</td></tr><tr><td rowspan="4">SSM Pruning</td><td>5/64</td><td>3.47</td><td>71.6</td><td>77.3</td><td>81.2</td><td>77.8</td><td>49.2</td><td>73.2</td><td>47.2</td><td>68.2</td></tr><tr><td>10/64</td><td>4.24</td><td>67.2</td><td>73.6</td><td>79.8</td><td>75.3</td><td>48.3</td><td>70.2</td><td>43.0</td><td>65.4</td></tr><tr><td>15/64</td><td>5.37</td><td>63.3</td><td>69.6</td><td>78.2</td><td>72.4</td><td>43.4</td><td>68.8</td><td>41.8</td><td>62.5</td></tr><tr><td>20/64</td><td>14.14</td><td>46.3</td><td>63.4</td><td>74.9</td><td>60.7</td><td>36.7</td><td>65.7</td><td>37.8</td><td>55.1</td></tr></table>

Table 5: Results of Mamba-Shedder with training-free Mamba block and SSM pruning for Falcon-Mamba-7B.

<table><tr><td>Model</td><td>Method</td><td>Num. of Pruned Inference Speedup Mamba Blocks</td><td>Prefill</td><td>Decode</td></tr><tr><td rowspan="3">Mamba-2.8B</td><td>Dense</td><td>0/64</td><td>1.00×</td><td>1.00×</td></tr><tr><td>Mamba-Shedder</td><td>7/64</td><td>1.12×</td><td>1.13×</td></tr><tr><td></td><td>14/64</td><td>1.31×</td><td>1.29×</td></tr></table>

Table 6: Inference benchmark results for Mamba-2.8B. The batch size is 1. Number of batches is 10. The prompt length is 512. Number of new tokens is 16.

Table 7: Inference benchmark results for Mamba2-2.7B, with test-related hyperparameters consistent with Table 6.
<table><tr><td>Model</td><td>Method</td><td>Num. of Pruned SSMs</td><td>Inference Speedup Prefill</td><td>Decode</td></tr><tr><td rowspan="3">Mamba2-2.7B</td><td>Dense</td><td>0/64</td><td>1.00×</td><td>1.00×</td></tr><tr><td></td><td>16/64</td><td>1.13×</td><td>1.11×</td></tr><tr><td>Mamba-Shedder</td><td>20 / 64</td><td>1.16×</td><td>1.14×</td></tr><tr><td colspan="2"></td><td>24 /64</td><td>1.20×</td><td>1.18×</td></tr></table>

<table><tr><td>Model</td><td>Method</td><td>Num. of Pruned Inference Speedup Hymba Blocks</td><td> Prefill</td><td>Decode</td></tr><tr><td>Hymba-1.5B-Base</td><td>Dense Mamba-Shedder</td><td>0/64 7/64</td><td>1.00×</td><td>1.00×</td></tr></table>

Table 8: Inference benchmark results for Zamba2-2.7B, with test-related hyperparameters consistent with Table 6. The calculation of Ratio includes block pruning (Mamba Block, MHA, and MLP) and width pruning (MLP Channel). Refer to Table 3 for more information.
<table><tr><td>Model</td><td>Method</td><td>Ratio</td><td>(Block, Width) Pruned SSMs Prefill</td><td>Additional Inference Speedup</td><td>Decode</td></tr><tr><td>Zamba2</td><td>Dense</td><td>0%</td><td>0/54</td><td>1.00×</td><td>1.00×</td></tr><tr><td>2.7B</td><td>Mamba-Shedder</td><td>15.48% 15.48%</td><td>0/54 18 / 64</td><td>1.16× 1.25×</td><td>1.34× 1.39×</td></tr></table>

Zamba-2 We observe significant acceleration on inference after multiple granularities pruning of

Table 9: Inference benchmark results for Hymba-1.5B-Base, where the test-related hyperparameters consistent with Table 6, except that number of new tokens is 256.
<table><tr><td>Model</td><td>Method</td><td>Num. of Pruned SSMs</td><td></td><td>Lambada Average PPL (↓) Accuracy</td></tr><tr><td rowspan="3"></td><td>Dense</td><td>0/64</td><td>4.10</td><td>60.2</td></tr><tr><td>Mamba2-2.7B Mamba-Shedder</td><td>20/64</td><td>5.89</td><td>58.6</td></tr><tr><td>Mamba-Shedder w/ tune</td><td>20/64</td><td>4.44.1.45</td><td>59.6+1.0</td></tr></table>

Table 10: Results of the compressed Mamba2-2.7B model with recovery tuning (post-training).

<table><tr><td>Model</td><td>Method</td><td>Ratio (Block &amp; Width) SSMs</td><td>Additional Lambada Pruned</td><td>PPL (↓)</td><td>Avg. Acc. (↑)</td></tr><tr><td rowspan="5">Zamba2 2.7B</td><td>Dense</td><td></td><td>0/54</td><td>4.01</td><td>67.2</td></tr><tr><td>Mamba-Shedder</td><td>10.27%</td><td>18/54</td><td>5.18</td><td>65.9</td></tr><tr><td>Mamba-Shedder w/ tune 10.27%</td><td></td><td>18/54</td><td>4.58.0.60</td><td>67.0+1.1</td></tr><tr><td>Mamba-Shedder</td><td>15.48%</td><td>18/54</td><td>7.43</td><td>61.3</td></tr><tr><td>Mamba-Shedder w/ tune 15.48%</td><td></td><td>18/54</td><td>5.88.1.55</td><td>64.4+3.1</td></tr></table>

Table 11: Results of the compressed Mamba2-2.7B and Zamba2-2.7B models with recovery tuning.

Zamba-2 (Table 8). Specifically, pruning Mamba blocks, MLP, and MHA blocks along with MLP channels results in a 1.34x speedup in the decoding stage. When SSM pruning is included, the speedup increases to 1.39x, indicating that a comprehensive pruning strategy that includes multiple components can significantly enhance inference speed while maximizing the preservation of model performance.

<table><tr><td>Model</td><td>Method</td><td>Num. of Pruned Average</td><td>Hymba Blocks Accuracy</td></tr><tr><td rowspan="3"></td><td>Dense</td><td> $_ { 0 / 3 2 }$ </td><td></td></tr><tr><td>Hymba-1.5B-Base Mamba-Shedder</td><td> $7 \overline { { 1 } } 3 \overline { { 2 } }$ </td><td> $\stackrel { 6 3 . 8 } { 6 1 . 7 } - \stackrel { . } { . }$ </td></tr><tr><td>Mamba-Shedder w/ tune</td><td>7/32</td><td> ${ \bf 6 3 . 7 _ { + 2 . 0 } }$ </td></tr></table>

Table 12: Results of the compressed Hymba-1.5B-Base model with recovery tuning. Average Accuracy is calculated over HellaSwag, PIQA, ARC-e, ARC-c, and WinoGrande tasks (Table 4).

Hymba Block pruning of Hymba-1.5B-Base demonstrates notable improvements in inference speed (Table 9). By removing 7 out of 64 Hymba blocks, Mamba-Shedder achieves a 1.15x speedup in the prefill stage and a 1.24x speedup in the decoding stage, suggesting that significant computational efficiency gains can be realized even with a relatively modest pruning ratio. The results highlight the potential of Mamba-Shedder to optimize the performance of Hymba models, making them more efficient for real-time applications without substantial sacrifices in model accuracy.

## 4.5 Recovery Tuning of the Pruned Model

Following related work (Ma et al., 2023; Zhong et al., 2024), we performed post-training on the Mamba-Shedder compressed model using the cleaned version of Alpaca. The results summarized in Tables 10, 11, and 12 demonstrate substantial performance gains after just two epochs of recovery tuning. For instance, the Mamba-Shedder model obtained by removing Mamba Blocks & MLPs & MHAs + MLP Channels + SSM in Zamba-2 (Table 3), initially exhibits a perplexity of 5.18 and an average accuracy of 65.9 when 18 out of 54 SSMs are pruned. However, after recovery tuning, it achieves a significantly reduced PPL of 4.58 and an improved average accuracy of 67.0, which is almost on par with the Dense model. Similarly, the recovery tuning of the Hymba-1.5B-Base model also yields significant improvements (Table 12). Initially, the pruned model with 7 out of 32 Hymba blocks removed shows an average accuracy of 61.7. After recovery tuning, the average accuracy increases to 63.7, which is nearly equivalent to the dense model’s accuracy of 63.8.

![](images/56952d4b5d8263d25d47016f9070e133b0b46440479f9019ab714cfce060605c.jpg)  
Number of Pruned Mamba Blocks

![](images/198af6445e7f751cc414bb044ee40013e0d3219c1e3d0afec7b60bf09a7c9ba9.jpg)  
Figure 4: Close examination of the impact of removing Mamba blocks or SSMs from the two versions of Mamba models reveals distinct differences in their tolerance levels. Mamba-1 exhibits a higher tolerance for removing its blocks, while Mamba-2 exhibits greater tolerance for removing the SSM subcomponent.

This phase effectively enhances the performance of the pruned model, bringing it closer to the original dense model’s performance while maintaining computational efficiency. In summary, recovery tuning is crucial to optimize pruned models, making them more viable for practical applications.

## 4.6 Insights on the Compression Sensitivity of the Variants of Mamba

The proponents of Mamba modified the original architecture to restrict the expressivity in Mamba-2 and increase the training efficiency. As illustrated on the left side of Figure 4, our experiments suggest that these changes make Mamba-2 models less robust to removing entire blocks than the previous version of the Mamba block. As soon as we remove blocks with the least importance, Mamba-1 exhibits a more robust behavior. However, Mamba-2 demonstrates a significantly higher tolerance to removing SSMs, maintaining a stable perplexity even as more SSMs are pruned, suggesting that while Mamba-2’s architectural improvements have made it more sensitive to the removal of Mamba blocks, they have also enhanced its robustness to SSM pruning.

## 5 Conclusion

Selective structure state space models have become an efficient alternative to Transformer-based models. In this paper, we propose Mamba-Shedder and investigate structured pruning strategies to remove elements from Mamba and hybrid models and reduce model size, accelerating inference. The results demonstrate that selective structured state space architectures have several redundancies that can be removed without significantly affecting the model’s performance.

## Limitations

Despite their outstanding results, large sequence models are still under investigation to better understand their capabilities and limitations. Mamba-Shedder is, to the best of our knowledge, the first work to investigate the removal of structures in Mamba-based models, including hybrids with Transformer blocks. Our goal is to motivate the research community to better understand this class of models to identify opportunities for future improvements in the model architecture and applicable compression techniques. The results indicate that these models contain redundant elements that might be removed to improve their efficiency. However, future work must explore and attempt to better understand the trade-offs between efficiency and accuracy when removing these models’ components. Even more research questions can be entertained when considering Transformer blocks and hybrid models, as in the case of Zamba. For instance, there is much to understand about the right mix of the SSM- and Transformer-based elements.

## Ethics Statement

Due to the well-known flaws in modern sequence models, e.g., hallucinations, many guard rails must be in place when considering deploying them in production. Our research focuses on improving the efficiency of these models in existing downstream tasks and datasets. However, further experimentation and analysis are needed when considering deploying these compressed models in environments where their output might affect people’s well-being.

## References

A. Arnab, M. Dehghani, G. Heigold, C. Sun, M. Lucic, and C. Schmid. 2021. Vivit: A video vision transformer. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 6816–6826, Los Alamitos, CA, USA. IEEE Computer Society.

Saleh Ashkboos, Maximilian L. Croci, Marcelo Gennari do Nascimento, Torsten Hoefler, and James Hensman. 2024. SliceGPT: Compress large language models by deleting rows and columns. In The Twelfth International Conference on Learning Representations.

Iz Beltagy, Matthew E. Peters, and Arman Cohan. 2020. Longformer: The long-document transformer. arXiv:2004.05150.

Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. 2020. Piqa: Reasoning about

physical commonsense in natural language. In Thirty-Fourth AAAI Conference on Artificial Intelligence.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. In Advances in Neural Information Processing Systems, volume 33, pages 1877–1901. Curran Associates, Inc.

Krzysztof Marcin Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Quincy Davis, Afroz Mohiuddin, Lukasz Kaiser, David Benjamin Belanger, Lucy J Colwell, and Adrian Weller. 2021. Rethinking attention with performers. In International Conference on Learning Representations.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question answering? try arc, the ai2 reasoning challenge. ArXiv, abs/1803.05457.

Gonçalo M. Correia, Vlad Niculae, and André F. T. Martins. 2019. Adaptively sparse transformers. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 2174– 2184, Hong Kong, China. Association for Computational Linguistics.

Zihang Dai, Guokun Lai, Yiming Yang, and Quoc V. Le. 2020. Funnel-transformer: filtering out sequential redundancy for efficient language processing. In Proceedings ofthe 34th International Conference on Neural Information Processing Systems, NIPS ’20, Red Hook, NY, USA. Curran Associates Inc.

Tri Dao and Albert Gu. 2024. Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality. In International Conference on Machine Learning (ICML).

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Associationfor Computational Linguistics (ACL).

Xin Dong, Yonggan Fu, Shizhe Diao, Wonmin Byeon, Zijia Chen, Ameya Sunil Mahabaleshwarkar, Shih-Yang Liu, Matthijs Van Keirsbilck, Min-Hung Chen, Yoshi Suhara, et al. 2024. Hymba: A hybrid-head architecture for small language models. arXiv preprint arXiv:2411.13676.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. 2021. An image is worth 16x16 words: Transformers for image recognition at scale. Preprint, arXiv:2010.11929.

Stefan Elfwing, Eiji Uchibe, and Kenji Doya. 2018. Sigmoid-weighted linear units for neural network function approximation in reinforcement learning. Neural Networks, 107:3–11. Special issue on deep reinforcement learning.

Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. 2022. GPTQ: Accurate post-training compression for generative pretrained transformers. arXiv preprint arXiv:2210.17323.

Daniel Y. Fu, Tri Dao, Khaled K. Saab, Armin W. Thomas, Atri Rudra, and Christopher Ré. 2023. Hungry Hungry Hippos: Towards language modeling with state space models. In International Conference on Learning Representations.

Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, and Connor Leahy. 2020. The pile: An 800gb dataset of diverse text for language modeling. Preprint, arXiv:2101.00027.

Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. 2023. A framework for few-shot language model evaluation.

Paolo Glorioso, Quentin Anthony, Yury Tokpanov, James Whittington, Jonathan Pilault, Adam Ibrahim, and Beren Millidge. 2024. Zamba: A compact 7b ssm hybrid model. Preprint, arXiv:2405.16712.

Yuan Gong, Yu-An Chung, and James Glass. 2021. AST: Audio Spectrogram Transformer. In Proc. Interspeech 2021, pages 571–575.

Albert Gu and Tri Dao. 2023. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752.

Albert Gu, Karan Goel, and Christopher Ré. 2022. Efficiently modeling long sequences with structured state spaces. In The International Conference on Learning Representations (ICLR).

Albert Gu, Isys Johnson, Karan Goel, Khaled Saab, Tri Dao, Atri Rudra, and Christopher Ré. 2024. Combining recurrent, convolutional, and continuous-time models with linear state-space layers. In Proceedings of the 35th International Conference on Neural Information Processing Systems, NIPS ’21, Red Hook, NY, USA. Curran Associates Inc.

Torsten Hoefler, Dan Alistarh, Tal Ben-Nun, Nikoli Dryden, and Alexandra Peste. 2021. Sparsity in deep learning: pruning and growth for efficient inference and training in neural networks. J. Mach. Learn. Res., 22(1).

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations (ICLR).

Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. 2020. Transformers are rnns: fast autoregressive transformers with linear attention. In Proceedings of the 37th International Conference on Machine Learning, ICML’20. JMLR.org.

François Lagunas, Ella Charlaix, Victor Sanh, and Alexander Rush. 2021. Block pruning for faster transformers. In Proceedings ofthe 2021 Conference on Empirical Methods in Natural Language Processing, pages 10619–10629, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Yann LeCun, John Denker, and Sara Solla. 1989. Optimal brain damage. In Advances in Neural Information Processing Systems, volume 2. Morgan-Kaufmann.

Xinyin Ma, Gongfan Fang, and Xinchao Wang. 2023. LLM-pruner: On the structural pruning of large language models. In Thirty-seventh Conference on Neural Information Processing Systems.

Xin Men, Mingyu Xu, Qingyu Zhang, Bingning Wang, Hongyu Lin, Yaojie Lu, Xianpei Han, and Weipeng Chen. 2024. Shortgpt: Layers in large language models are more redundant than you expect. Preprint, arXiv:2403.03853.

Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. 2018. Can a suit of armor conduct electricity? a new dataset for open book question answering. In Conference on Empirical Methods in Natural Language Processing.

J. Pablo Muñoz, Jinjie Yuan, and Nilesh Jain. 2024. Shears: Unstructured sparsity with neural low-rank adapter search. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 6: Industry Track), pages 395–405, Mexico City, Mexico. Association for Computational Linguistics.

J. Pablo Muñoz, Jinjie Yuan, and Nilesh Jain. 2025. Multipruner: Balanced structure removal in foundation models. Preprint, arXiv:2501.09949.

Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Ngoc Quan Pham, Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernández. 2016. The LAMBADA dataset: Word

prediction requiring a broad discourse context. In Proceedings of the 54th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 1525–1534, Berlin, Germany. Association for Computational Linguistics.

Niki J. Parmar, Ashish Vaswani, Jakob Uszkoreit, Lukasz Kaiser, Noam Shazeer, Alexander Ku, and Dustin Tran. 2018. Image transformer. In International Conference on Machine Learning (ICML).

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning (ICML).

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2021. Winogrande: An adversarial winograd schema challenge at scale. Commun. ACM, 64(9):99–106.

Mingjie Sun, Zhuang Liu, Anna Bair, and J. Zico Kolter. 2023. A simple and effective pruning approach for large language models. arXiv preprint arXiv:2306.11695.

Yury Tokpanov, Beren Millidge, Paolo Glorioso, Jonathan Pilault, Adam Ibrahim, James Whittington, and Quentin Anthony. 2024. Zyda: A 1.3 t dataset for open language modeling. arXiv preprint arXiv:2406.01981.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc.

Peng Xu, Wenqi Shao, Mengzhao Chen, Shitao Tang, Kaipeng Zhang, Peng Gao, Fengwei An, Yu Qiao, and Ping Luo. 2024. Besa: Pruning large language models with blockwise parameter-efficient sparsity allocation. Preprint, arXiv:2402.16880.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019. Hellaswag: Can a machine really finish your sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics.

Li Zhang, Jiachen Lu, Sixia Zheng, Xinxuan Zhao, Xiatian Zhu, Yanwei Fu, Xiang Tao, and Jianfeng Feng. 2023. Vision transformers: From semantic segmentation to dense prediction. arXiv.

Lin Zheng, Chong Wang, and Lingpeng Kong. 2022. Linear complexity randomized self-attention mechanism. In International Conference on Machine Learning, pages 27011–27041. PMLR.

Longguang Zhong, Fanqi Wan, Ruijun Chen, Xiaojun Quan, and Liangzhi Li. 2024. Blockpruner: Finegrained pruning for large language models. Preprint, arXiv:2406.10594.

Jingwei Zuo, Maksim Velikanov, Dhia Eddine Rhaiem, Ilyas Chahed, Younes Belkada, Guillaume Kunsch, and Hakim Hacid. 2024. Falcon mamba: The first competitive attention-free 7b language model. arXiv preprint arXiv:2410.05355.

## Supplementary Material

## A Related Work

Transformers (Vaswani et al., 2017) and its variants are the primary building block of successful deep learning architectures, e.g., Llama (Touvron et al., 2023) and GPT (Brown et al., 2020), that have revolutionized Natural Language Processing (NLP) (Devlin et al., 2019; Gao et al., 2023), Computer Vision (CV) (Parmar et al., 2018; Radford et al., 2021; Zhang et al., 2023), and many other domains. Due to the Transformer’s popularity, researchers have proposed variants to improve their computational and memory efficiency further and tackle issues like their quadratic complexity in sequence length during training (Correia et al., 2019; Beltagy et al., 2020; Dai et al., 2020; Choromanski et al., 2021; Katharopoulos et al., 2020; Zheng et al., 2022).

A parallel research effort investigates alternatives to Transformers in the form of structured state space models (SSMs) that can power the next generation of sequence models. The initial proposals of structured SSMs were linear time-invariant, e.g., LSSL (Gu et al., 2024), S4 (Gu et al., 2022), H3 (Fu et al., 2023). Recent improvements to the state space model formulation have resulted in the proposal of time-varying selective SSMs, e.g., Mamba (Gu and Dao, 2023; Dao and Gu, 2024).

To our knowledge, Mamba-Shedder is the first study on pruning selective structured state space models (Mamba) and their hybrids. On the other hand, many works have proposed pruning techniques for Transformer-based models (Hoefler et al., 2021). Several of these works focus on unstructured pruning (Sun et al., 2023; Xu et al., 2024; Frantar et al., 2022), which can achieve higher sparsity levels. However, it requires highly optimized runtimes to realize the benefits of sparsity. Sophisticated solutions have been proposed to fine-tune sparse models and recover any accuracy drop from the pruning stage (Muñoz et al., 2024). Recently, training-free approaches have been proposed for structured pruning of Transformers. These approaches cannot achieve high sparsity levels as the unstructured pruning approaches. However, they are very convenient because their compressed models do not require specialized runtimes and exhibit beneficial inference acceleration. In this line of research, LLMPruner (Ma et al., 2023), ShortGPT (Men et al., 2024), BlockPruner (Lagunas et al., 2021), SliceGPT (Ashkboos et al.,

<table><tr><td>Hyper-parameter</td><td>Value</td></tr><tr><td>Pruning Stage:</td></tr><tr><td>Calibration Dataset tatsu-lab/alpaca</td></tr><tr><td>Importance Metric Perplexity (PPL)</td></tr><tr><td>Number of Calibration Samples 256</td></tr><tr><td>MLP Channel Group Size (Zamba2) 1024</td></tr><tr><td>Steps of MLP Channel Pruning (Zamba2) 20</td></tr></table>

Table 13: Hyper-parameters used in the experiments.

2024), and MultiPruner (Muñoz et al., 2025) have demonstrated efficient methods for Transformer pruning. BlockPruner improved over many previous approaches by proposing a global metric that can be used to determine the importance of a selected network structure. MultiPruner extended this approach to pruning the width dimension, as well. Mamba-Shedder builds on these works and the rest of the extensive literature on structured block pruning to explore opportunities for removing redundancies in models with Mamba blocks.

## B Hyperparameters

Table 13 offers a detailed summary of the hyperparameters employed in our experiments, promoting both reproducibility and clarity.