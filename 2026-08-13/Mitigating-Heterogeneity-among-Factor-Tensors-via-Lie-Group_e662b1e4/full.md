# Mitigating Heterogeneity among Factor Tensors via Lie Group Manifolds for Tensor Decomposition Based Temporal Knowledge Graph Embedding

Jiang Li<sup>1,2,3</sup> , Xiangdong Su<sup>1,2,3</sup> \*, Guanglai Gao<sup>1,2,3</sup>

<sup>1</sup> College of Computer Science, Inner Mongolia University, China

<sup>2</sup> National & Local Joint Engineering Research Center of Intelligent Information Processing Technology for Mongolian, China

<sup>3</sup> Inner Mongolia Key Laboratory of Multilingual Artiffcial Intelligence Technology, China lijiangimu@gmail.com, cssxd@imu.edu.cn

## Abstract

Recent studies have highlighted the effectiveness of tensor decomposition methods in the Temporal Knowledge Graphs Embedding (TKGE) task. However, we found that inherent heterogeneity among factor tensors in tensor decomposition significantly hinders the tensor fusion process and further limits the performance of link prediction. To overcome this limitation, we introduce a novel method that maps factor tensors onto a unified smooth Lie group manifold to make the distribution of factor tensors approximating homogeneous in tensor decomposition. We provide the theoretical proof of our motivation that homogeneous tensors are more effective than heterogeneous tensors in tensor fusion and approximating the target for tensor decomposition based TKGE methods. The proposed method can be directly integrated into existing tensor decomposition based TKGE methods without introducing extra parameters. Extensive experiments demonstrate the effectiveness of our method in mitigating the heterogeneity and in enhancing the tensor decomposition based TKGE models<sup>1</sup>.

## 1 Introduction

Knowledge graphs (KGs) are data structures that encapsulate knowledge triples of real-world entities and their interrelationships, and are widely used to improve information retrieval (Liang et al., 2023), reasoning (Xu et al., 2023), Q&A (Hu et al., 2021), etc. Temporal knowledge graphs (TKGs) extend this paradigm by introducing timestamps into knowledge triplets to reflect the validity of facts over time and provide a deeper understanding and analysis of dynamic changes in the facts. Due to the data incompleteness in both KGs and TKGs, researchers propose many KG embedding (KGE) and TKG embedding (TKGE) methods to predict the missing facts, thereby enhancing the richness and accuracy of the KGs and TKGs. This work mainly focuses on TKGE.

![](images/cbf051b53ba12e505b2cfdddc562100392ce172104e911bbcc9e169c64ab07c1.jpg)  
Figure 1: (a) illustrates the heterogeneity in the distribution of entities, relations and timestamps within TKGs, as evidenced by the differing distribution curves. (b) illustrates the homogeneous distribution curves of entities, relations and timestamps when using our method to mitigate the heterogeneity among these three elements.

As the interest in TKG grows, researchers proposed many TKGE methods and greatly promoted the development of TKG. Concerning the success of tensor decomposition in KGE (Nickel et al., 2016; Trouillon et al., 2016; Lacroix et al., 2018), recent works (Lacroix et al., 2020; Xu et al., 2021; Li et al., 2023) further extended tensor decomposition into TKGE and obtained very excellent performance. These works demonstrated that tensor decomposition can guarantee full expressiveness under specific embedding dimensionality bounds in TKG, thus enhancing the link prediction.

However, existing TKGE methods based on tensor decomposition suffer from inherent heterogeneity among factor tensors. Recent research (Wu et al., 2020; Li et al., 2021) also highlights the intrinsic heterogeneity in TKGs, specifically in terms of entity and temporal heterogeneity. According to our analysis, the heterogeneity of entity, relation and timestamp originates from their semantic roles within the knowledge graphs. That is, the entities represent the static components of the graph, the relations delineate the interactions among the entities, and the timestamp characterizes the temporal aspects of these interactions, specifying when they occur and their duration. This heterogeneity leads to the learned factor tensor expliciting different distributions in TKGE, as shown in Figure 1(a). This further limits the tensor fusion in TKGE models and lowers the link prediction accuracy. More discussion about heterogeneity can be found in $\mathbf { A } \mathbf { p } \mathbf { - }$ pendix A.

Therefore, it is necessary to address the heterogeneity for tensor decomposition-based TKGE methods to enhance link prediction. To this target, we propose to map the factor tensors onto a unified smooth Lie group manifold to make the distribution of factor tensors approximating homogeneous in tensor decomposition, as shown in Figure 1(b). Since the manifold in Lie group looks the same at every point and all tangent spaces at any point are alike (Solà et al., 2018), the factor tensors mapped by the Lie group have a smooth and unified distribution, which mitigates the heterogeneity among the factor tensors. We provide the theoretical proof of our motivation that homogeneous factor tensors are more effective in approximating the target compared to heterogeneous factor tensors in TKGE models in Sec. 4.1. We integrate the proposed method into several existing tensor decomposition based TKGE models and conduct extensive experiments to evaluate its effectiveness. The experimental results present the heterogeneity among factor tensors in TKGE methods and illustrate that the proposed method brings significant performance improvement. This confirms the effectiveness of our method in alleviating the heterogeneity. Our contributions are summarized as follows:

• To the best of our knowledge, we are the first to investigate the negative effect of the heterogeneity among the factor tensors for tensor decomposition based TKGE models and propose to enhance these models by diminishing the heterogeneity via Lie group manifold.

• We provide the theoretical proof of our motivation that homogeneous factor tensors are more effective than heterogeneous factor tensors in approximating the target in TKGE.

• Our proposed method can be directly integrated into the tensor decomposition based TKGE models without introducing any additional parameters, and extensive experiments on several TKGE models demonstrate its effectiveness and generalization.

## 2 Related Work

## 2.1 Static Knowledge Graph Embedding

Drawing inspiration from the concept of translation invariance featured in word2vec (Mikolov et al., 2013), TransE assesses the relations between entities and their links by calculating the distance from $e _ { s } + e _ { \hat { r } }$ to $e _ { o }$ using standard $l _ { 1 }$ or $l _ { 2 }$ norms, where $e _ { s }$ and $e _ { o }$ are the vectors that represent the starting and ending entities, and $e _ { \hat { r } }$ represents the linking relation. Following TransE, TransH (Wang et al., 2014), TransR (Lin et al., 2015), and TransD (Ji et al., 2015) introduce various mapping ways and thus refine these embeddings for better KGE representation. ComplEx (Trouillon et al., 2016) employs 3-th order tensor decomposition to capture the interactions within KGs. TorusE (Ebisu and Ichise, 2018) utilizes a torus (a donut-shaped manifold) for its embeddings. TorusE introduces a torus, which is a compact Abelian Lie group, and defines distance functions on the torus. The torus can be considered as a collection of multiple Lie groups. Instead, we map the factor tensors to the Lie group space, thus mitigating the distributional heterogeneity among them.

## 2.2 Temporal Knowledge Graph Embedding

In TKGE models, the temporal information is added, and the scoring function is calculated for the quadruples to assess their reasonableness. Therefore, most TKGE models use existing KGE models as a foundation. TTransE (Leblay and Chekol, 2018) extends TransE and encodes time stamps τ as translations same as relations. Hence, the score function of TTransE is denoted as $\phi ( s , \hat { r } , o , \tau ) = | | s + \hat { r } + \tau - o | | _ { p }$ . Furthermore, TA-TransE (García-Durán et al., 2018) encode timestamps based on TransE. RotateQVS (Chen et al., 2022) uses quaternion embeddings to represent both entities and relations. Recently, BoxTE (Messner et al., 2022) models the TKGE based on a box embedding model BoxE (Abboud et al., 2020). TCompoundE (Ying et al., 2024) employs relation-specific and time-specific compound geometric operations to enhance the modeling of temporal dynamics and relational patterns.

## 2.3 Tensor Decomposition Based Temporal Knowledge Graph Embedding

TComplEx (Lacroix et al., 2020) and TNTComplEx (Lacroix et al., 2020) expand upon the ComplEx model by executing a fourth-order tensor decomposition in temporal knowledge graphs (TKGs). This method offers a more nuanced understanding of the temporal dimensions in knowledge graphs. TeLM (Xu et al., 2021) utilizes the asymmetric geometric product, a method that allows for a more sophisticated and expressive representation of temporal relationships and entities. TeAST (Li et al., 2023) maps relations onto Archimedean spiral timelines, and ensures that relations occurring simultaneously are placed on the same timeline, with all relations evolving over time. The above works are all based on tensor decomposition to optimize the TKGE representation. In this paper, we focus on exploring the problem of the heterogeneity of factor tensors in tensor decomposition based TKGE models.

## 3 Background and Notation

## 3.1 TKGE Task

Given a TKG, let denote the set of entities, denote the set of relations, and $\tau$ denote the set of timestamps. A TKG can be defined as a collection of quadruples $( s , \hat { r } , o , \tau )$ , where $s \in \mathcal { E } , \hat { r } \in \mathcal { R }$ $o \in { \mathcal { E } }$ and $\tau \in \mathcal { T }$ denote the subject entity, relation, object entity and timestamp, respectively. The TKGE task aims to accurately learn embedded representations of entities, relations and timestamps to facilitate predictions of missing entities in TKGs. Specifically, it involves predicting the object entity o given a tuple $( s , \hat { r } , \ ? , \tau )$ , or conversely, predicting the subject entity s for a tuple $( ? , \hat { r } , o , \tau )$ , thereby capturing the dynamic nature of relationships over time. In this paper, we denote the relation in TKG as $\hat { r }$ and the rank in mathematics as r.

## 3.2 Tensor Decomposition for TKGE

In the existing tensor decomposition based TKGE methods, each relation quadruple is represented by a 0, 1 -valued 4-th order tensor $\mathcal { V } \in$ $\{ 0 , 1 \}$ (Lacroix et al., 2020). This representation allows each element $\mathcal { V } _ { s , \hat { r } , o , \tau } = 1$ to indicate that at a specific time $\tau ,$ , there is a relationship rˆ between entities s and $o .$ In link prediction, tensor decomposition algorithms learn to infer a predicted tensor that approximates the ground truth , as

$$
\mathcal { Y } \sim \mathcal { X } = \sum _ { r = 1 } ^ { R } \mathbf { u } _ { r } \otimes \mathbf { v } _ { r } \otimes \mathbf { w } _ { r } \otimes \mathbf { t } _ { r } ,\tag{1}
$$

where rank $r \in \{ 1 , . . . , R \}$ , denotes the tensor product. $\mathbf { u } _ { r } , \mathbf { v } _ { r } , \mathbf { w } _ { r }$ and $\mathbf { t } _ { r }$ denote the subject entity, relation, object entity and timestamp factor tensors. Tensor decomposition based TKGE methods aim to optimize the factor tensors to make $\mathcal { X }$ as close as possible to the tensor $\mathcal { V }$ and thus achieve more accurate link prediction.

## 4 Methodology

This paper employs Lie group manifold to diminish the heterogeneity of factor tensors in tensor decomposition based TKGE models and thus improves the performance of these models. In Sec. 4.1, we provide the theoretical proof of our motivation that homogeneous tensors are more effective than heterogeneous tensors in approximating the target for tensor decomposition based TKGE methods. In Sec. 4.2, we explain why Lie groups can mitigate the heterogeneity among factor tensors, and describe how to map the factor tensors to the Lie group space. In Sec. 4.3, we introduce a Logarithmic Mapping $\cdot l o g ( f ( \cdot ) ) ^ { , }$ operation and alleviate the heterogeneity among factors by minimizing the difference between the original factor tensors and the mapped factor tensors in the Lie group through the N3 regularization in the loss function.

## 4.1 Theoretical Analysis of Homogeneous vs. Heterogeneous Factor Tensors

Proposition 1. Homogeneous factor tensors $( \mathbf { u } _ { r } , \ \mathbf { v } _ { r } , \ \mathbf { w } _ { r } , \ \mathbf { t } _ { r } )$ with a low rank can effectively approximated  while heterogeneous factor tensors $( \mathbf { u } _ { r } , \mathbf { v } _ { r } , \mathbf { w } _ { r } , \mathbf { t } _ { r } )$ require a higher rank to approximate  in TKGE.

Proof. Given a 4th-order tensor decomposition in TKGE

$$
\mathcal { Y } \approx \sum _ { r = 1 } ^ { R }  { \mathbf { u } } _ { r } \otimes  { \mathbf { v } } _ { r } \otimes  { \mathbf { w } } _ { r } \otimes  { \mathbf { t } } _ { r } ,\tag{2}
$$

where R is the rank of the decomposition, and $( \mathbf { u } _ { r } , \mathbf { v } _ { r } , \mathbf { w } _ { r } , \mathbf { t } _ { r } ) \in \mathbb { R }$ are factor tensors.

If the factor tensors $( \mathbf { u } _ { r } , \mathbf { v } _ { r } , \mathbf { w } _ { r } , \mathbf { t } _ { r } )$ are homogenous, there are a common set of basis vectors $\mathbf { B } = \{ \mathbf { b } _ { 1 } , \mathbf { b } _ { 2 } , \dots , \mathbf { b } _ { m } \}$ between these factor tensors. Based on this homogeneity, the column vectors of each factor matrix can be expressed as linear combinations of the basis vectors in B.

$$
\begin{array} { r } { \mathbf { u } _ { r } = \mathbf { B } \cdot \boldsymbol { \alpha } _ { r } , \quad \mathbf { v } _ { r } = \mathbf { B } \cdot \boldsymbol { \beta } _ { r } , } \\ { \mathbf { w } _ { r } = \mathbf { B } \cdot \boldsymbol { \gamma } _ { r } , \quad \mathbf { t } _ { r } = \mathbf { B } \cdot \boldsymbol { \delta } _ { r } , } \end{array}\tag{3}
$$

where $r = 1 , \ldots , R ,$ and $\alpha _ { r } , \beta _ { r } , \gamma _ { r } ,$ , and $\delta _ { r }$ are the coefficient vectors for the r-th component in their respective factor matrices. Consequently, we can get

$$
\begin{array} { r } { \mathcal { Y } \approx \sum _ { r = 1 } ^ { R } ( \mathbf { B } \cdot { \boldsymbol { \alpha } } _ { r } ) \otimes ( \mathbf { B } \cdot { \boldsymbol { \beta } } _ { r } ) \otimes ( \mathbf { B } \cdot { \boldsymbol { \gamma } } _ { r } ) \otimes ( \mathbf { B } \cdot { \boldsymbol { \delta } } _ { r } ) . } \end{array}\tag{4}
$$

Given the homogeneity among factor tensors, we can further obtain the representation as follows

$$
\begin{array} { r } { \mathcal { V } \approx \sum _ { j = 1 } ^ { m } \mathbf { b } _ { j } \otimes \mathbf { b } _ { j } \otimes \mathbf { b } _ { j } \otimes \mathbf { b } _ { j } \cdot \sum _ { r = 1 } ^ { R } \alpha _ { j r } \beta _ { j r } \gamma _ { j r } \delta _ { j r } , } \end{array}\tag{5}
$$

where $\alpha _ { j r } , \beta _ { j r } , \gamma _ { j r }$ , and $\delta _ { j r }$ are the scalar coefficients corresponding to the projection of the r-th component onto the j-th basis vector in their respective factor matrices. The set of basis vectors $\{ \mathbf { b } _ { 1 } , \mathbf { b } _ { 2 } , \dots , \mathbf { b } _ { m } \}$ are orthogonal to each other, ensuring that each dimension represented by these vectors is independent. $\begin{array} { r } { \lambda _ { j } = \sum _ { r = 1 } ^ { R } \alpha _ { j r } \beta _ { j r } \gamma _ { j r } \delta _ { j r } } \end{array}$ can be considered as scaling constants. Thus, we can get a reduced-rank tensor $\mathcal { V } ^ { \prime }$ as

$$
\mathcal { V } \approx \mathcal { V } ^ { \prime } = \sum _ { j = 1 } ^ { m } \lambda _ { j } \mathbf { b } _ { j } \otimes \mathbf { b } _ { j } \otimes \mathbf { b } _ { j } \otimes \mathbf { b } _ { j } ,\tag{6}
$$

which captures the essence of the homogeneity within the factor matrices. The rank m of $\mathcal { V } ^ { \prime }$ is less than the original rank R $( m < R )$ . Thus, homogeneous factor tensors $( \mathbf { u } _ { r } , \mathbf { v } _ { r } , \mathbf { w } _ { r } , \mathbf { t } _ { r } )$ with a low rank can effectively approximate the target quadruple tensor .

In contrast, if the factor tensors $( \mathbf { u } _ { r } , \ \mathbf { v } _ { r } , \ \mathbf { w } _ { r }$ $\mathbf { t } _ { r } )$ are highly heterogeneous, they exhibit distinct semantics and distributions characteristics. This implies that these factor tensors cannot be effectively approximated by a simple, small number of rank-1 tensors. Each rank-1 tensor can be viewed as a representation of a specific pattern or feature. For these heterogeneous tensors, the patterns they capture, or the semantics they represent within the data necessitate a larger number of rank-1 tensors to capture their diverse characteristics individually. Hence, the heterogeneous factor tensors $( \mathbf { u } _ { r } , \mathbf { v } _ { r } ,$ $\mathbf { w } _ { r } , \mathbf { t } _ { r } )$ require a higher rank or even full rank to approximate  in TKGE. Since higher rank means more parameters to estimate and more computation, homogeneous factor tensors are more effective than heterogeneous factor tensors in approximating the target for tensor decomposition based TKGE methods.

![](images/5c91942785723bdabc460ba5d54ba3bccde70e67294bc0e0a25342d13a655532.jpg)  
Figure 2: An illustration of the relation between the Lie group and the Lie algebra. The Lie algebra so(n) is the tangent space to the Lie group’s manifold $S ( n )$

## 4.2 Mitigating Heterogeneity via Lie Group

In TKGE, the underlying reason why highly heterogeneous factor tensors require a higher rank to approximate the target vector is that the heterogeneity among factor tensors can limit the fusion process of subsequent computations, which can be analogous to the multimodal fusion process (Chen and Zhang, 2020). Therefore, it is crucial to mitigate the heterogeneity among factor tensors to approximate the target tensor more efficiently. Our motivations for choosing Lie groups to mitigate heterogeneity in TKGE are as follows.

Firstly, Lie groups are adept at maintaining structural integrity and handling data’s dynamic nature over time, making them a suitable choice for TKGE. The application of Lie groups in fields like robotics (Solà et al., 2018), machine learning (Sommer et al., 2020; Lee and Civera, 2022), and computer vision (Teed and Deng, 2021) underscores its capability to model complex geometric transformations effectively. Secondly, as shown in Figure 2, Lie group is a mathematical structure that simultaneously satisfies the axioms of a group and the properties of a smooth manifold. It is like a curved, smooth hyper-surface, with no edges or spikes, embedded in a space of higher dimension. The smoothness of the manifold implies the existence of a unique tangent space at each point. In a Lie group, the manifold looks the same at every point, and therefore all tangent spaces at any point are alike. Thus, the factor tensors mapped by the Lie group have a smooth and unified distribution, which further mitigates the heterogeneity among the factor tensors.

In this study, we map the factor tensors to the same Lie group space and make the factor tensors have a unified distribution to mitigate the heterogeneity among them. To facilitate the description of our method, we employ factor tensor of rank 4 in the subsequent discussions. Given a factor tensor e of rank 4 and map it to the Lie group $S O ( 2 )$ space, we get

$$
f ( \cdot ) : \mathbb { R } ^ { n } \to S O ( 2 ) ; \quad e \mapsto \mathbf { R }\tag{7}
$$

where $f ( \cdot )$ denots Lie group mapping operation. ${ \bf R } _ { e }$ is a rotation matrix in $S O ( 2 )$ , denoted as

$$
\mathbf { R } _ { e } = \binom { \cos e } { \sin e } \quad \frac { - \sin e } { \cos e } \Biggr ) .\tag{8}
$$

Accordingly, given a quadruple $( s , \hat { r } , o , \tau )$ in TKG, its corresponding factor tensors are $( \mathbf { u } , \mathbf { v } ,$ w, t) in the tensor decomposition based TKGE models. We map these four factor tensors onto Lie group space, and we get the rotation matrices

$$
\begin{array} { r l } { \mathbf { R } _ { \mathbf { u } } = \left( \cos { \mathbf { u } } } & { - \sin { \mathbf { u } } \right) , \mathbf { R } _ { \mathbf { v } } = \left( \cos { \mathbf { v } } - \sin { \mathbf { v } } \right) , } \\ { \sin { \mathbf { u } } } & { \cos { \mathbf { u } } } \end{array}\tag{9}
$$

When generalized to n rank, $S O ( { \sqrt { n } } )$ can be denoted as a Givens rotation matrix. In an $n \mathrm { - }$ dimensional space, a Givens rotation is performed by fixing $\sqrt { n } - 2$ dimensions and applying a rotation transformation within the plane formed by the remaining two dimensions. This effectively isolates the rotation to a specific two-dimensional subspace. The Givens rotation matrix $G ( i , j , e )$ for rotating the i-th and $j \mathrm { - t h }$ coordinates in $\sqrt { n }$ -dimensional space is given by:

$$
\begin{array} { r } { G ( i , j , \epsilon ) = \left[ \begin{array} { c c c c c c c c } { 1 } & { \cdots } & { 0 } & { \cdots } & { 0 } & { \cdots } & { 0 } \\ { \vdots } & { \ddots } & { \vdots } & & & { \vdots } & & { \vdots } \\ { 0 } & { \cdots } & { \cos ( \epsilon ) } & { \cdots } & { - \sin ( \epsilon ) } & { \cdots } & { 0 } \\ { \vdots } & & { \vdots } & { \ddots } & { \vdots } & & & { \vdots } \\ { 0 } & { \cdots } & { \sin ( \epsilon ) } & { \cdots } & { \cos ( \epsilon ) } & { \cdots } & { 0 } \\ { \vdots } & & { \vdots } & & & { \vdots } & { \ddots } & { \vdots } \\ { 0 } & { \cdots } & { 0 } & { \cdots } & & { 0 } & { \cdots } & { 1 } \end{array} \right] , } \end{array}\tag{10}
$$

where the rotation occurs in the plane spanned by the i-th and j-th basis vectors. The matrix is an identity matrix except for the four elements $G _ { i i }$ $G _ { i j } , G _ { j i }$ , and $G _ { j j }$ , which form the $2 \times 2$ rotation block within the larger matrix. Hence, we focus on the $2 \times 2$ base rotation matrices in the code implementation.

## 4.3 Logarithmic Mapping from $S O ( n )$ to so(n)

Our training goal is to diminish the heterogeneity among the factor tensors in TKGE model training and thus improve the link prediction performance. Due to the Lie group $S O ( n )$ residing on a non-Euclidean manifold, we introduce a variant of Logarithmic Mapping operation (Huang et al., 2017) (denoted as $l o g ( \cdot ) )$ on the Lie group space and converts the rotation matrices into the usual skew-symmetric matrices which are situated in the Euclidean space, as

$$
l o g ( \cdot ) : S O ( n )  \mathfrak { s o } ( n ) , \quad \mathbf { R } \mapsto l o g ( \mathbf { R } ) .\tag{11}
$$

The logarithm mapping log(R) is

$$
\begin{array} { r } { l o g ( { \mathbf { R } } ) = \left\{ \begin{array} { l l } { 0 , } & { \mathrm { i f } \quad \theta ( { \mathbf { R } } ) = 0 , } \\ { \frac { \theta ( { \mathbf { R } } ) } { 2 \sin ( \theta ( { \mathbf { R } } ) ) } ( { \mathbf { R } } - { \mathbf { R } } ^ { T } ) , } & { \mathrm { o t h e r w i s e } , } \end{array} \right. } \end{array}
$$

where θ(R) is the angle of $\mathbf { R } ,$ as

(12)

$$
\theta ( \mathbf { R } ) = \operatorname { a r c c o s } \left( { \frac { t r a c e ( \mathbf { R } ) - 1 } { 2 } } \right) .\tag{13}
$$

Here, trace( ) is a square matrix is the sum of its diagonal elements.

The mathematical derivation of logarithmic mappings references this work (Solà et al., 2018). After logarithmic mapping, we get $l o g ( f ( \mathbf { u } _ { r } ) )$ , $l o g ( f ( \mathbf { v } _ { r } ) )$ , $l o g ( f ( \mathbf { w } _ { r } ) )$ , $l o g ( f ( \mathbf { t } _ { r } ) )$ . Then, we calculate the differences between the original tensors and their corresponding mapped tensors on the Lie group

$$
\begin{array} { r l } & { \mathbf { u } _ { r } ^ { \prime } = \mathbf { u } _ { r } - l o g ( f ( \mathbf { u } _ { r } ) ) , } \\ & { \mathbf { v } _ { r } ^ { \prime } = \mathbf { v } _ { r } - l o g ( f ( \mathbf { v } _ { r } ) ) , } \\ & { \mathbf { w } _ { r } ^ { \prime } = \mathbf { w } _ { r } - l o g ( f ( \mathbf { w } _ { r } ) ) , } \\ & { \mathbf { t } _ { r } ^ { \prime } = \mathbf { t } _ { r } - l o g ( f ( \mathbf { t } _ { r } ) ) . } \end{array}\tag{14}
$$

Finally, we perform the standard tensor decomposition with $\mathbf { u } _ { r } ^ { \prime } , \mathbf { v } _ { r } ^ { \prime } , \mathbf { w } _ { r } ^ { \prime } , \mathbf { t } _ { r } ^ { \prime }$ , as

$$
\mathcal { Y } \sim \mathcal { X } = \sum _ { r = 1 } ^ { R } \lambda _ { r } \mathbf { u } _ { r } ^ { \prime } \otimes \mathbf { v } _ { r } ^ { \prime } \otimes \mathbf { w } _ { r } ^ { \prime } \otimes \mathbf { t } _ { r } ^ { \prime } .\tag{15}
$$

Following previous works (Lacroix et al., 2018, 2020; Xu et al., 2021; Li et al., 2023), we use the full multiclass log-softmax loss function and N3 regularization to optimize the factor tensors, which are defined as follows:

$$
\begin{array} { r } { \mathcal { L } = - \log ( \frac { \exp ( \phi ( s , \hat { r } , o , \tau ) ) } { \sum _ { s ^ { \prime } \in \mathcal { E } } \exp ( \phi ( s ^ { \prime } , \hat { r } , o , \tau ) ) } ) } \\ { - \log ( \frac { \exp ( \phi ( o , \hat { r } ^ { - 1 } , s , \tau ) ) } { \sum _ { o ^ { \prime } \in \mathcal { E } } \exp ( \phi ( o ^ { \prime } , \hat { r } ^ { - 1 } , s , \tau ) ) } ) } \\ { + \lambda _ { \mu } \displaystyle \sum _ { i = 1 } ^ { R } ( \| \mathbf { u } _ { r } ^ { \prime } \| _ { 3 } ^ { 3 } + \| \mathbf { v } _ { r } ^ { \prime } \| _ { 3 } ^ { 3 } + \| \mathbf { w } _ { r } ^ { \prime } \| _ { 3 } ^ { 3 } + \| \mathbf { t } _ { r } ^ { \prime } \| _ { 3 } ^ { 3 } ) , } \end{array}\tag{16}
$$

where $\lambda _ { \mu }$ denotes N3 regularization weight, $\hat { r } ^ { - 1 }$ is the inverse relation. We use $\| \mathbf { t } _ { r } ^ { \prime } \| _ { 3 } ^ { 3 }$ to represent the temporal regularizer for simplicity, which is computed in N3 regularization way.

By using the N3 regularization, we minimize the $\mathbf { u } _ { r } ^ { \prime } , \mathbf { v } _ { r } ^ { \prime } , \mathbf { w } _ { r } ^ { \prime } , \mathbf { t } _ { r } ^ { \prime }$ . That is, we drive the $\mathbf { u } _ { r } , \mathbf { v } _ { r } , \mathbf { w } _ { r } , \mathbf { t } _ { r }$ to be homogeneous in Euclidean space, since the $l o g ( f ( \mathbf { u } _ { r } ) )$ , log(f(v<sub>r</sub>)), log(f(w<sub>r</sub>)), log(f(t<sub>r</sub>)) are tend to be homogeneous in Lie group space. Therefore, the proposed method can mitigate the heterogeneity among factor tensors in tensor decomposition based TKGE methods.

## 5 Experiments

## 5.1 Datasets

To evaluate the effectiveness of the proposed method, we evaluate our method on two popular TKGE benchmark datasets. ICEWS14 and ICEWS05-15 (García-Durán et al., 2018) are both extracted from the Integrated Crisis Early Warning System (ICEWS) dataset (Lautenschlager et al., 2015), which consists of temporal sociopolitical facts starting from 1995. ICEWS14 consists of sociopolitical events in 2014 and ICEWS05- 15 involves events occurring from 2005 to 2015. ICEWS14 is a fine temporal granularity dataset, while ICEWS05-15 has a wider temporal granularity relative to ICEWS14. See Appendix B for summary statistics of the dataset and more discussion of the dataset.

## 5.2 Evaluation Protocol

In this research, we follow the previous works (Lacroix et al., 2020; Xu et al., 2021; Li et al., 2023) to evaluate our method. Specifically, to evaluate the quality of the ranking for each test quadruples, we calculate all possible substitutions for the subject and object entities, denoted as $( s ^ { \prime } , \hat { r } , o , \tau )$ and $( s , \hat { r } , o ^ { \prime } , \tau )$ , where $s ^ { \prime } , o ^ { \prime } \in \mathcal { E }$ . After that, we sort the score of candidate quadruples under the time-wise filtered settings (Lacroix et al., 2020; Xu et al., 2021; Li et al., 2023). The performance is evaluated using standard evaluation metrics, including Mean Reciprocal Rank (MRR) and Hits@n. The Hits@n metric measures the percentage of correct entities in the top n predictions. Higher values of MRR and Hits@n indicate better performance. Hits ratio with cut-off values $n = 1 , 3 , 1 0$ In this paper, we utilize H@n to denote Hits@n for convenience.

## 5.3 Experimental Setup

We implement our method based on the existing training framework<sup>2</sup>. All experiments are trained on a single NVIDIA Tesla A100. The hyperparameters used in the experiment are consistent with the optimal hyperparameters of the original paper report. The best models are selected by early stopping (threshold of 10) on the validation datasets. The max epoch is 200. We report the average results on the test set for five runs. To ensure a fair validation of the effectiveness of our method, we employ the same hyperparameter configuration in both the before and after comparison experiments.

According to Lie group mapping described in Sec. 4.2, it is essential to ensure that the rank r of the matrix can be satisfied by the square root of $( 2 \times r )$ is an integer in our method. This requirement arises from the implementation of the matrix logarithm map for TcomplEx, TNTcomplEx and TeAST. The rank of TeLM needs to be satisfied by the square root of $( 4 \times r )$ is an integer.

## 6 Results and Analysis

## 6.1 Main Results

In our experiments, we validate the effectiveness of our proposed method for dealing with TKGs heterogeneity in tensor decomposition on ICEWS14 and ICEWS05-15 datasets. The improvements are marked in red in Table 1, highlighting the advancements over the baselines. When our method is applied to different tensor decomposition based TKGE models, they all achieve meaningful improvements in different metrics. This significant improvement confirms our Proposition 1, in which homogeneous factor tensors can be effectively approximated  with a low rank. Additionally, the

<table><tr><td rowspan="2"></td><td colspan="5">ICEWS14</td><td rowspan="2"></td><td colspan="6">ICEWS05-15</td></tr><tr><td>rank</td><td>Para.</td><td>MRR</td><td>H@1</td><td>H@3</td><td>H@10</td><td>rank Para.</td><td>MRR</td><td>H@1</td><td>H@3</td><td>H@10</td></tr><tr><td colspan="10">Tensor Decomposition Based TKGE Models</td><td></td><td></td><td></td><td></td></tr><tr><td>TComplEx</td><td>128</td><td>2.04M</td><td>55.3</td><td>46.3</td><td>60.7</td><td>71.5</td><td>128</td><td>3.84M</td><td>58</td><td>49</td><td>64</td><td>76</td></tr><tr><td>TNTComplEx</td><td>128</td><td>2.15M</td><td>55.7</td><td>46.3</td><td>61.5</td><td>73.0</td><td>128</td><td>3.97M</td><td>60</td><td>50</td><td>65</td><td>78</td></tr><tr><td>TeLM</td><td>121</td><td>3.85M</td><td>50.6</td><td>42.1</td><td>55.0</td><td>67.1</td><td>121</td><td>7.26M</td><td>56.8</td><td>48.7</td><td>61.1</td><td>72.0</td></tr><tr><td>TeAST</td><td>128</td><td>2.13M</td><td>53.4</td><td>43.9</td><td>58.8</td><td>70.9</td><td>128</td><td>4.87M</td><td>48.8</td><td>38.4</td><td>54.7</td><td>68.3</td></tr><tr><td colspan="11">Tensor Decomposition Based TKGE Models+log(f(·))</td></tr><tr><td>TComplEx</td><td>128</td><td>2.04M</td><td>56.2 (+0.9)</td><td>46.8 (+0.5)</td><td>61.6 (+ 0.9)</td><td>73.5 (+0.9)</td><td>128</td><td>3.84M</td><td>59.6 (+1.6)</td><td>50.2</td><td>65.3</td><td>77.2</td></tr><tr><td>TNTComplEx</td><td>128</td><td>2.15M</td><td>56.3</td><td>46.7</td><td>61.8</td><td>74.1</td><td>128</td><td>3.97M</td><td>60.2</td><td>(+0.3) 50.8</td><td>(+1.3) 65.9</td><td>(+1.2) 78.1</td></tr><tr><td>TeLM</td><td>121</td><td>3.85M</td><td>(+0.6) 54.5 (+3.9)</td><td>(+0.4) 45.5</td><td>(+0.3) 59.5</td><td>(+1.1) 71.7</td><td>121</td><td>7.26M</td><td>(+0.2) 59.0</td><td>(+0.8) 50.6</td><td>(+0.9) 63.8</td><td>(+0.1) 74.7</td></tr><tr><td>TeAST</td><td>128</td><td>2.13M</td><td>56.1 (+2.7)</td><td>(+3.4) 47.3 (+3.4)</td><td>(+4.5) 61.4 (+2.6)</td><td>(+3.6) 72.3 (+1.4)</td><td>128</td><td>4.87M</td><td>(+2.2) 59.2 (10.4)</td><td>(+1.9) 50.5 (+12.1)</td><td>(+2.7) 64.8 (+10.1)</td><td>(+2.7) 76.9 (+8.6)</td></tr></table>

Table 1: Link prediction results on ICEWS14 and ICEWS05-15. The results of $\heartsuit$ are taken from Lacroix et al. (2020). Other results are obtained from our experiments. log(f( )) indicates that our proposed method.

ICEWS05-15 dataset validates the effectiveness of our method in mitigating data heterogeneity, with TeAST notably exhibiting an average improvement of 10.3 points.

In conclusion, the experimental results provide robust evidence supporting Proposition 1. The experimental outcomes validate the theoretical framework of our study and demonstrate that our novel method effectively alleviates data heterogeneity in tensor decomposition and enhances the link prediction performance of these models. More experiments on the large TKG dataset GDELT can be found in Appendix C.

## 6.2 Quantitative Analysis on Heterogeneity

In this section, we perform a quantitative analysis to prove the effectiveness of our proposed method. For any factor tensors $e _ { x }$ and $e _ { y } .$ , the skew-symmetric matrices in so(n) are given by

$$
A _ { x } = l o g ( f ( e _ { x } ) ) , \quad A _ { y } = l o g ( f ( e _ { y } ) ) .\tag{17}
$$

We define the difference between $e _ { x }$ and $e _ { y }$ in so(n) to be denoted as $d ( A _ { x } , A _ { y } )$ . The relationship between the set of skew-symmetric matrices $\{ A _ { x } \}$ obtained from the mapping of a set of vectors $e _ { x }$ can be described using the operations in the Lie algebra, such as computing their Lie brackets. This process of mapping vectors to $S O ( n )$ and then to ${ \mathfrak { s o } } ( n )$ transforms them into elements with a unified algebraic structure, mitigating the differences in structural distribution between them.

To quantify the structural differences between the skew-symmetric matrices $A _ { x }$ and $A _ { y }$ in so(n), we consider the Frobenius norm of their difference

$$
\| A _ { x } - A _ { y } \| _ { F } = { \sqrt { \operatorname { t r a c e } ( ( A _ { x } - A _ { y } ) ^ { T } ( A _ { x } - A _ { y } ) ) } } .\tag{18}
$$

This norm provides a measure of the difference between the corresponding matrices.

Based on the above quantitative formulas, we evaluate on the representative models TComplEx and TNTComplEx. Specifically, we calculate the difference between entity and relation, entity and timestamp, and relation and timestamp. As shown in Table 2, we calculate their average distance difference on ICEWS14.

<table><tr><td>d(|ε|, |R|)</td><td>d(|ε|, |T|)</td><td>d(R, |T|)</td></tr><tr><td>TComplEx</td><td>15.71 7.61</td><td>15.72</td></tr><tr><td>TComplEx +log(f(·))</td><td>13.74 6.89</td><td>12.43</td></tr><tr><td>TNTComplEx</td><td>22.20</td><td>5.82 22.06</td></tr><tr><td>TNTComplEx +log(f(·))</td><td>17.92 5.61</td><td>17.10</td></tr></table>

Table 2: Quantitative analysis results for TComplEx vs. TComplEx+log(f( )) and TNTComplEx vs. TNTComplEx+log(f( )).

As shown in Table 2, for the standard TComplEx model, the average difference in distance between entities and relations is 15.71, between entities and timestamps is 7.61, and between relations and timestamps is 15.72. These results demonstrate significant differences in quantification between different types of embeddings.

Further, when our method log(f( )) is employed in the TComplEx model, we observe a reduction in all three distance difference. This significant improvement points to the effectiveness of our method in mitigating the heterogeneity among different factor tensors. Similarly, the results of the TNTComplEx model support the above statement.

## 6.3 Visualisation Analysis

To further verify that our method can effectively mitigate the heterogeneity between factor embeddings, we utilize t-SNE (Van der Maaten and Hinton, 2008) to visualize the learned entity, relation, and timestamp embeddings. As shown in Figure 3, we can observe that the learned factor embeddings through our method exhibit a trend towards homogeneity. This further demonstrates the inherent heterogeneity present among different types of embeddings in TKGE based on tensor decomposition. Our method effectively mitigates this issue, demonstrating that meaningful performance improvements can be achieved.

![](images/69d3534303a6be9a713231c6f939dc916d8a46154b688f3dcc3370df2398dd95.jpg)  
(a) TComplEx

![](images/3fa75a528bd876f30e4f5d6b5994e361f4e2427f1f6348dd21c8b794c689923a.jpg)  
(b) TComplEx+log(f( ))

![](images/ff5131782d319356ef5ed8f6a923c19c501c5295c4a7d36a00e3c8916200c9eb.jpg)  
(c) TNTComplEx

![](images/4666b55fd45f9fec601287476c6dc80cc24f1b325893d68bc5f9803cf9d7d6c3.jpg)  
<sup>(d)</sup> <sup>TNTComplEx+log(f(</sup>·<sup>))</sup>  
Figure 3: Visualisations of the learned entity, relation and timestamp embeddings on ICEWS14.

## 6.4 Effect of Rank

In this work, we compare the performance of the standard TeAST (Li et al., 2023) model with our proposed TeAST model enhanced by $+ \log ( f ( \cdot ) )$ across different rank values on ICEWS14. As shown in Figure 4, we observe that the performance of both models improves with the increase of rank values. However, after the rank value reaches 800, the pace of performance improvement slows down. This is because the representation capacity of the model reaches saturation at a certain level, beyond which the marginal benefits of increasing rank diminish. Additionally, higher rank values might lead to overfitting, especially in cases of sparse data, negatively affecting the model’s generalization ability on unseen data.

![](images/8b8ec972bff9d6d3cf4e42eb40fb117b491ab346dabfd672465d3cb16e6dd1f0.jpg)  
(a) MRR

![](images/316bbde2ff915f2bcb1b3bfc6337b451c20e1bb7225eb782197b55b4e3216746.jpg)

![](images/3dfd572872c58f4ad829770a316518dd5fe2b75a4a3df87d64a2c741be70dc75.jpg)  
(c) Hits@3

(b) Hits@1  
![](images/fa4957f548a160b1e05e02c308a3d15685643358f8bcf85e8b0406554d0f39b4.jpg)  
(d) Hits@10  
Figure 4: Results of TeAST and TeAST+log(f( )) with different rank on ICEWS14.

## 7 Conclusion

In this study, we are the first to introduce methods to mitigate heterogeneity in factor tensors within tensor decomposition-based TKGE models. We reveal that the heterogeneity primarily stems from diverse semantic content among elements (entities, relation and timestamps), which impedes the effective fusion of factor tensors and limits link prediction accuracy. We prove that homogeneous tensors are more effective than heterogeneous tensors in tensor fusion and approximating the target for tensor decomposition based TKGE methods. Our method maps factor tensors onto a smooth Lie group manifold to standardize their distribution and mitigate heterogeneity without increasing model complexity. Our experimental results demonstrate the effectiveness of this method in mitigating tensor heterogeneity and enhancing performance. We hope that this work can offer fresh insights for research in the field of TKGE.

## Limitations

In this paper, we investigate the effect of the heterogeneity among factor tensors on link prediction in tensor decomposition based TKGE models. We mainly focus on addressing the issue of heterogeneity among elements within TKGs, which is recognized as a key challenge in this domain. Moreover, similar to the majority of TKGE models, our method is unable to process new entities that are not present in the training data.

## Acknowledgments

This work was funded by National Natural Science Foundation of China (Grant No. 62366036), National Education Science Planning Project (Grant No. BIX230343), Key R&D and Achievement Transformation Program of Inner Mongolia Autonomous Region (Grant No. 2022YFHH0077), The Central Government Fund for Promoting Local Scientific and Technological Development (Grant No. 2022ZY0198), Program for Young Talents of Science and Technology in Universities of Inner Mongolia Autonomous Region (Grant No. NJYT24033), Inner Mongolia Autonomous Region Science and Technology Planning Project (Grant No. 2023YFSH0017), Science and Technology Program of the Joint Fund of Scientific Research for the Public Hospitals of Inner Mongolia Academy of Medical Sciences (Grant No.2023GLLH0035).

## References

Ralph Abboud, <sup>˙</sup>Ismail <sup>˙</sup>Ilkan Ceylan, Thomas Lukasiewicz, and Tommaso Salvatori. 2020. Boxe: A box embedding model for knowledge base completion. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual.

Weishan Cai, Yizhao Wang, Shun Mao, Jieyu Zhan, and Yuncheng Jiang. 2022. Multi-heterogeneous neighborhood-aware for knowledge graphs alignment. Information Processing & Management, 59(1):102790.

Jiayi Chen and Aidong Zhang. 2020. Hgmf: heterogeneous graph-based fusion for multimodal data with incompleteness. In Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining, pages 1295–1305.

Kai Chen, Ye Wang, Yitong Li, and Aiping Li. 2022. Rotateqvs: Representing temporal information as rotations in quaternion vector space for temporal knowledge graph completion. In Proceedings ofthe 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2022, Dublin, Ireland, May 22-27, 2022, pages 5843– 5857. Association for Computational Linguistics.

Takuma Ebisu and Ryutaro Ichise. 2018. Toruse: Knowledge graph embedding on a lie group. In Proceedings ofthe Thirty-Second AAAI Conference on Artificial Intelligence, (AAAI-18), the 30th innovative Applications ofArtificial Intelligence (IAAI-18), and the 8th AAAI Symposium on Educational Advances in Artificial Intelligence (EAAI-18), New Orleans, Louisiana, USA, February 2-7, 2018, pages 1819– 1826. AAAI Press.

Alberto García-Durán, Sebastijan Dumancic, and Mathias Niepert. 2018. Learning sequence encoders for temporal knowledge graph completion. In Proceedings ofthe 2018 Conference on Empirical Methods in Natural Language Processing, Brussels, Belgium, October 31 - November 4, 2018, pages 4816–4821. Association for Computational Linguistics.

Zikun Hu, Yixin Cao, Lifu Huang, and Tat-Seng Chua. 2021. How knowledge graph and attention help? A qualitative analysis into bag-level relation extraction. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, ACL/IJCNLP 2021, (Volume 1: Long Papers), Virtual Event, August 1-6, 2021, pages 4662–4671. Association for Computational Linguistics.

Zhiwu Huang, Chengde Wan, Thomas Probst, and Luc Van Gool. 2017. Deep learning on lie groups for skeleton-based action recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Guoliang Ji, Shizhu He, Liheng Xu, Kang Liu, and Jun Zhao. 2015. Knowledge graph embedding via dynamic mapping matrix. In Proceedings of the 53rd Annual Meeting ofthe Associationfor Computational Linguistics and the 7th International Joint Conference on Natural Language Processing ofthe Asian Federation of Natural Language Processing, ACL 2015, July 26-31, 2015, Beijing, China, Volume 1: Long Papers, pages 687–696. The Association for Computer Linguistics.

Timothée Lacroix, Guillaume Obozinski, and Nicolas Usunier. 2020. Tensor decompositions for temporal knowledge base completion. In 8th International Conference on Learning Representations, ICLR 2020,

Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net.

Timothée Lacroix, Nicolas Usunier, and Guillaume Obozinski. 2018. Canonical tensor decomposition for knowledge base completion. In Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholmsmässan, Stockholm, Sweden, July 10-15, 2018, volume 80 of Proceedings of Machine Learning Research, pages 2869–2878. PMLR.

Jennifer Lautenschlager, Steve Shellman, and Michael Ward. 2015. Icews event aggregations.

Julien Leblay and Melisachew Wudage Chekol. 2018. Deriving validity time in knowledge graph. In Companion ofthe The Web Conference 2018 on The Web Conference 2018, WWW 2018, Lyon , France, April 23-27, 2018, pages 1771–1776. ACM.

Seong Hun Lee and Javier Civera. 2022. Hara: A hierarchical approach for robust rotation averaging. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15777–15786.

Jiang Li, Xiangdong Su, and Guanglai Gao. 2023. TeAST: Temporal knowledge graph embedding via archimedean spiral timeline. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15460–15474, Toronto, Canada. Association for Computational Linguistics.

Zhifei Li, Hai Liu, Zhaoli Zhang, Tingting Liu, and Neal N Xiong. 2021. Learning knowledge graph embedding with heterogeneous relation attention networks. IEEE Transactions on Neural Networks and Learning Systems, 33(8):3961–3973.

Ke Liang, Lingyuan Meng, Meng Liu, Yue Liu, Wenxuan Tu, Siwei Wang, Sihang Zhou, and Xinwang Liu. 2023. Learn from relational correlations and periodic events for temporal knowledge graph reasoning. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 1559–1568.

Yankai Lin, Zhiyuan Liu, Maosong Sun, Yang Liu, and Xuan Zhu. 2015. Learning entity and relation embeddings for knowledge graph completion. In Proceedings of the Twenty-Ninth AAAI Conference on Artificial Intelligence, January 25-30, 2015, Austin, Texas, USA, pages 2181–2187. AAAI Press.

Johannes Messner, Ralph Abboud, and <sup>˙</sup>Ismail <sup>˙</sup>Ilkan Ceylan. 2022. Temporal knowledge graph completion using box embeddings. In Thirty-Sixth AAAI Conference on Artificial Intelligence, AAAI 2022, Thirty-Fourth Conference on Innovative Applications of Artificial Intelligence, IAAI 2022, The Twelveth Symposium on Educational Advances in Artificial Intelligence, EAAI 2022 Virtual Event, February 22 - March 1, 2022, pages 7779–7787. AAAI Press.

Tomás Mikolov, Ilya Sutskever, Kai Chen, Gregory S. Corrado, and Jeffrey Dean. 2013. Distributed representations of words and phrases and their compositionality. In Advances in Neural Information Processing Systems 26: 27th Annual Conference on Neural Information Processing Systems 2013. Proceedings ofa meeting held December 5-8, 2013, Lake Tahoe, Nevada, United States, pages 3111–3119.

Maximilian Nickel, Kevin Murphy, Volker Tresp, and Evgeniy Gabrilovich. 2016. A review of relational machine learning for knowledge graphs. Proceedings of the IEEE, 1(104):11–33.

Namyong Park, Fuchen Liu, Purvanshi Mehta, Dana Cristofor, Christos Faloutsos, and Yuxiao Dong. 2022. Evokg: Jointly modeling event time and network structure for reasoning over temporal knowledge graphs. In Proceedings of the fifteenth ACM international conference on web search and data mining, pages 794–803.

Joan Solà, Jérémie Deray, and Dinesh Atchuthan. 2018. A micro lie theory for state estimation in robotics. CoRR, abs/1812.01537.

Christiane Sommer, Vladyslav Usenko, David Schubert, Nikolaus Demmel, and Daniel Cremers. 2020. Efficient derivative computation for cumulative b-splines on lie groups. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11148–11156.

Zachary Teed and Jia Deng. 2021. Tangent space backpropagation for 3d transformation groups. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10338–10347.

Théo Trouillon, Johannes Welbl, Sebastian Riedel, Éric Gaussier, and Guillaume Bouchard. 2016. Complex embeddings for simple link prediction. In Proceedings of the 33nd International Conference on Machine Learning, ICML 2016, New York City, NY, USA, June 19-24, 2016, volume 48 of JMLR Workshop and Conference Proceedings, pages 2071–2080. JMLR.org.

Laurens Van der Maaten and Geoffrey Hinton. 2008. Visualizing data using t-sne. Journal of machine learning research, 9(11).

Zhen Wang, Jianwen Zhang, Jianlin Feng, and Zheng Chen. 2014. Knowledge graph embedding by translating on hyperplanes. In Proceedings of the Twenty-Eighth AAAI Conference on Artificial Intelligence, July 27 -31, 2014, Québec City, Québec, Canada, pages 1112–1119. AAAI Press.

Jiapeng Wu, Meng Cao, Jackie Chi Kit Cheung, and William L Hamilton. 2020. Temp: Temporal message passing for temporal knowledge graph completion. arXiv preprint arXiv:2010.03526.

Chengjin Xu, Yung-Yu Chen, Mojtaba Nayyeri, and Jens Lehmann. 2021. Temporal knowledge graph completion using a linear temporal regularizer and

multivector embeddings. In Proceedings ofthe 2021 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies, NAACL-HLT 2021, Online, June 6-11, 2021, pages 2569–2578. Association for Computational Linguistics.

Yi Xu, Junjie Ou, Hui Xu, and Luoyi Fu. 2023. Temporal knowledge graph reasoning with historical contrastive learning. In Thirty-Seventh AAAI Conference on Artificial Intelligence, AAAI 2023, Thirty-Fifth Conference on Innovative Applications of Artificial Intelligence, IAAI 2023, Thirteenth Symposium on Educational Advances in Artificial Intelligence, EAAI 2023, Washington, DC, USA, February 7-14, 2023, pages 4765–4773. AAAI Press.

Rui Ying, Mengting Hu, Jianfeng Wu, Yalan Xie, Xiaoyi Liu, Zhunheng Wang, Ming Jiang, Hang Gao, Linlin Zhang, and Renhong Cheng. 2024. Simple but effective compound geometric operations for temporal knowledge graph completion. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 11074–11086, Bangkok, Thailand. Association for Computational Linguistics.

Chuxu Zhang, Dongjin Song, Chao Huang, Ananthram Swami, and Nitesh V Chawla. 2019. Heterogeneous graph neural network. In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining, pages 793–803.

## A Definition and Discussion of Heterogeneity in TKG

In KGs, ‘heterogeneity’ refers to the semantic difference of the entity and relation. Similarly, in TKGs, ‘heterogeneity’ refers to the semantic difference of the entity, relation and timestamp. The heterogeneity among entities, relations, and time is as follows: (1) The heterogeneity between entities and relations is reflected in their structural roles within the graph, with entities existing as nodes and relations represented as edges between nodes. (2) The heterogeneity between entities and time is reflected in the fact that entities represent the static components of the graph, while time affects the changes in entity attributes and their relations. (3) The heterogeneity between relations and time is reflected in that relations delineate the interactions among entities, while time characterizes the temporal aspects of these interactions, specifying when they occur and their duration. Recent work (Zhang et al., 2019; Li et al., 2021; Cai et al., 2022) also indicates that KGs have an intrinsic property of heterogeneity, which contains various types of entities and relations. Since TKG extends the KG paradigm, they inherently exhibit this heterogeneity as well. Additionally, TKGs incorporate temporal information, which further contributes to time heterogeneity.

The heterogeneity in TKG leads to the learned factor tensor expliciting different distributions in tensor decomposition based TKGE methods. Unlike previous works (Wu et al., 2020; Park et al., 2022), we do not propose a model for modeling heterogeneous TKGs, but rather a unified approach for mitigating heterogeneity among entities, relations and timestamps via Lie group.

## B Statistics of Datasets

Statistics of all the datasets used in this work are listed in Table 3. denotes the set of entities, denotes the set of relations, and denotes the set of timestamps.

<table><tr><td></td><td>ICEWS14</td><td>ICEWS05-15</td><td>GDELT</td></tr><tr><td>E</td><td>7,128</td><td>10,488</td><td>500</td></tr><tr><td>R</td><td>230</td><td>251</td><td>20</td></tr><tr><td>T</td><td>365</td><td>4017</td><td>366</td></tr><tr><td>#Train</td><td>72,826</td><td>386,962</td><td>2,735,685</td></tr><tr><td>#Vaild</td><td>8,963</td><td>46,092</td><td>341,961</td></tr><tr><td>#Test</td><td>8,941</td><td>46,275</td><td>341,961</td></tr></table>

Table 3: Statistics of ICEWS14, ICEWS05-15 and GDELT datasets in the experiment.

The ICEWS14 and ICEWS05-15 datasets exhibit heterogeneity across multiple dimensions, encompassing a wide array of entities, relations, and temporal variations. These datasets include diverse entities such as countries, governmental bodies, individuals, and organizations, each with unique attributes and patterns of behavior. The relations captured within these datasets are equally varied, detailing interactions ranging from diplomatic engagements to military conflicts, each bearing distinct characteristics and impacts. Furthermore, the chronological recording of events introduces a dynamic aspect to the data, with entities and their interrelations evolving over different times.

## C Results on Larger Dataset

The above experiments have validated that our proposed method can improve the TKGE performance on the high-heterogeneity KGs by mitigating heterogeneity among factor tensors for tensor decomposition based methods. To further validate the effectiveness, we conduct experiments on a larger and more challenging TKG dataset GDELT. The GDELT covers only 500 most common entities and 20 most frequent relations, while the number of quadruples achieves 2M. This is reflected in the denser relations between entities in KGs. Hence, the GDELT dataset is a challenging large-scale TKG.

We chose TComplEx and TNTComplEx models as the backbone model in the experiment on GDELT. The results are shown in Table 4. From Table 4, we observe that there is significant performance improvement in terms of H@1, H@3, and H@10. This proves that the proposed method can effectively diminish the heterogeneity among the factor tensors in TKGE. It exemplifies the potential of our method in handling large-scale TKGs.

<table><tr><td rowspan="2"></td><td colspan="6">GDELT</td></tr><tr><td>rank</td><td>Para.</td><td>MRR</td><td>H@1</td><td>H@3</td><td>H@10</td></tr><tr><td>TComplEx</td><td>128</td><td>0.23M</td><td>21.3</td><td>13.4</td><td>22.7</td><td>36.5</td></tr><tr><td>TComplEx +log(f(·))</td><td>128</td><td>0.23M</td><td>22.7</td><td>14.7</td><td>24.3</td><td>38.3</td></tr><tr><td>TNTComplEx</td><td>128</td><td>0.24M</td><td>21.9</td><td>13.9</td><td>23.3</td><td>37.4</td></tr><tr><td>TNTComplEx +log(f(·))</td><td>128</td><td>0.24M</td><td>22.1</td><td>14.0</td><td>23.5</td><td>37.6</td></tr></table>

Table 4: Link prediction results on GDELT.

## D Method Efficiency Comparison

Since we implement logarithmic mapping in our implementation of Lie group mapping using the following method, our method is theoretically a linear operation with O(n) time complexity. The following table shows the training time required for our method on the ICEWS05-15 dataset compared to other methods, confirming the linear operation of our method without significantly increasing the computational cost. As shown in Table 5, we can observe that the training time for models using the log(f( )) method shows a slight increase. However, considering the potential performance improvements brought by our method, this additional time cost is acceptable.

<table><tr><td>Method</td><td>Para.</td><td>Train-time</td></tr><tr><td>TComplEx</td><td>3.84M</td><td>30 min</td></tr><tr><td>TComplEx +log(f(·))</td><td>3.84M</td><td>32 min</td></tr><tr><td>TNTComplEx</td><td>3.97M</td><td>32 min</td></tr><tr><td>TNTComplEx +log(f(·))</td><td>3.97M</td><td>33 min</td></tr><tr><td>TeLM</td><td>7.26M</td><td>35 min</td></tr><tr><td>TeLM +log(f(·))</td><td>7.26M</td><td>36 min</td></tr><tr><td>TeAST</td><td>4.87M</td><td>34 min</td></tr><tr><td>TeAST +log(f(·))</td><td>4.87M</td><td>35 min</td></tr></table>

Table 5: Comparison of training times and parameters for different TKGE models on ICEWS05-15.