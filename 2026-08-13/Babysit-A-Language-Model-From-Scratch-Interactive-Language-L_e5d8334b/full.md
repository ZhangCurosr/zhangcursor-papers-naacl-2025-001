# Babysit A Language Model From Scratch: Interactive Language Learning by Trials and Demonstrations

Ziqiao Ma University of Michigan marstin@umich.edu

Zekun Wang\* University of Michigan zekun@umich.edu

Joyce Chai University of Michigan chaijy@umich.edu

## Abstract

Humans are efficient language learners and inherently social creatures. Our language development is largely shaped by our social interactions, for example, the demonstration and feedback from caregivers. Contrary to human language learning, recent advancements in large language models have primarily adopted a noninteractive training paradigm, and refined pretrained models through feedback afterward. In this work, we explore how corrective feedback from interactions influences neural language acquisition from scratch through systematically controlled experiments, assessing whether it contributes to word learning efficiency in language models. We introduce a trial-and-demonstration (TnD) learning framework that incorporates three distinct components: student trials, teacher demonstrations, and a reward conditioned on language competence at various developmental stages. Our experiments reveal that the TnD approach accelerates word acquisition for student models of equal and smaller numbers of parameters, and we highlight the significance of both trials and demonstrations. We further show that the teacher's choices of words influence students'word-specific learning efficiency, and a practice-makes-perfect effect is evident by a strong correlation between the frequency of words in trials and their respective learning curves. Our findings suggest that interactive language learning, with teacher demonstrations and active trials, can facilitate efficient word learning in language models.

## 1 Introduction

Humans are social beings and we learn language from interactions (Vygotsky, 1934; Bruner, 1985; Palincsar, 1986; Kuhl, 2004; Tomasello, 2005). Long before children's linguistic skills are mature, they could engage in early forms of conversational exchange with others (Halliday, 1975;

Clark, 2018). A critical component of social interactions that language grounds to is the feedback provided by the caregivers (Warlaumont et al., 2014). This, for example, includes the communicative feedback (Bates et al., 1975; Nikolaus and Fourtassi, 2023) that highlights the success and failure of communication, and the corrective feedback (Farrar, 1992; Chouinard and Clark, 2003; Saxton et al., 2005; Hiller, 2016) that is more direct and emphasizes the responses from caregivers, which offer corrections to possible errors in children's speech, encompassing variants like negative evidence, reformulations, or recasts. Unlike human learners who acquire language skills through feedback during interactions, most language models differ in terms of their inductive biases and data sources (Warstadt and Bowman, 2022). These models typically learn from massive text corpora using cross-entropy loss for self-supervised learning.

Recently, several lines of cognitively motivated language modeling research have looked into the learnability and learning efficiency of language (Portelance et al., 2020; Chang and Bergen, 2022; Evanson et al., 2023). By incorporating nonlinguistic inputs such as multimodal stimuli (Shi et al., 2019; Ma et al., 2023; Portelance et al., 2024) and/or communicative feedback (Nikolaus and Fourtassi, 2021; Zhu et al., 2022; Liu et al., 2022), recent studies have explored potential mechanisms that contribute to efficient language learning in (vision-)language models. Through controlled ablation studies (Warstadt and Bowman, 2022), these models can serve as proof of concept to verify mechanisms that are practically effective for machines, and generate hypotheses that are possible for cognitive learners (Portelance, 2022; Portelance and Jasbi, 2023). In a similar spirit, we seek to investigate the role of explicit corrective feedback in neural language learning through controlled computational experiments. Rather than making direct comparisons to human learning, our goal is to examine if student trials and teacher demonstrations promote efficient word learning in language model training, and if so, which words benefit the most.

![](images/477899b2c1a288e864367f53ac56da550e29c049c25821be27c10deac5c766d5.jpg)  
(a) Stage 1: Training a (b) Stage 2: Training a neural age | (c) Stage 3: The student interactively learn (d) Alternating between typical language model by predictor from the trajectory of a from trials and demonstrations by a pre-trained interactive learning and causal language modeling. | typical language model. teacher, score by an age-conditioned reward. non-interactive learning.  
Figure 1: The learning by trial-and-demonstration (TnD) framework. In stage 1, we start by training a language model with the causal language modeling objective. In stage 2, we prompt the models along the learning trajectory for (text, step) pairs and train a neural age predictor to predict the training step given a text. In stage 3, we use the final model in stage 1 as the teacher model. In an interactive step, the student model is prompted to complete a trial, and the teacher model is prompted to provide a demonstration. The trials and demonstrations are scored by an age-conditioned reward function (Eq. 1), and the student model updates the policy with reinforcement learning. The student alternates between interactive and non-interactive steps.

We introduce Trial-and-Demonstration (TnD), an interactive learning framework that incorporates corrective feedback with three components: student model trials, teacher model demonstrations, and a reward conditioned on the training trajectory of the model (Figure 1). Our experiments reveal that the TnD approach accelerates word acquisition, highlighting the significance of both trials and demonstrations. From the teacher's perspective, their word choices affect students' word-specific learning efficiency. From the student's side, the word frequency in trials closely aligns with the learning curve, supporting the idea that learning by language production accelerates word proficiency. Our findings highlight that trials and demonstrations can facilitate word learning in language models, and further, suggest an efficient alternative to building language models interactively.

## 2 Interactive Language Learning by Trials-and-Demonstrations (TnD)

Modeling corrective feedback in computational language learning presents significant challenges, as recruiting human subjects to supervise the development of a language model from the ground up over numerous iterations is impractical. Consequently we present a human-free Trial-and-Demonstration (TnD) learning framework that streamlines the process (Figure 1). In a scenario of corrective feedback, the student model engages in productionbased learning: to produce an initial utterance, followed by the teacher model generating its version of the text as a demonstration. For the student model to recognize the teacher's response as preferable and to facilitate learning, these language outputs are evaluated by a reward function, which is based on the competence of the student's language use that is expected for its developmental stage (i.e., training steps). The TnD framework thus includes three components: a student model's trials, a teacher model's demonstrations, and rewards. This framework allows us to incorporate massive corpora to study modern generative language models, offering a general and unrestricted approach to simulate interactive language learning with corrective feedback on massive corpora.

## 2.1 The student model and trials

We employ randomly initialized GPT-2 (Radford et al., 2019) as the student model for our investigation into language acquisition, leveraging its causal language modeling (CLM) objective and inherent generative capabilities for production-based learning. To encourage the student model to attempt text production, it is essential to provide an appropriate context. In each trial, we prompt the student with the first 5 tokens from a natural sentence, asking it to generate the continuation as a trial.

## 2.2 The teacher model and demonstrations

Inspired by recent work (Bai et al., 2022; Lee et al. 2023; Saha et al., 2023), we utilize pre-trained lan-

![](images/e5166d192de5a9322b9bd2ccce6ff6e040cd505a1765b88ec1cf7138f0d0e1ec.jpg)

![](images/459a654211a2b54f78b49d9cd393da5f35e7c5c326c3608b615367dbe73db709.jpg)  
(a) BookCorpus dataset.  
(b) BabyLM dataset.

Figure 2: We sample reward model predictions at different steps and compare them to ground truth logarithm. The reward models are satisfactory as the model predicted age/step highly overlaps with the true age/step. guage models as proxies for human language teachers. Employing language models as “caregivers" for language models offers two advantages. Firstly, it eliminates the need for recruiting human participants to engage with a language model across thousands of iterations. Secondly, we can consistently control the behavior of the teacher model across experiments, and adjust the teacher's language behaviors by modifying its decoding strategies for language generation. The process of developing a teacher model is identical to the typical language model pre-training, as shown in Figure 1(a). We adopted the same GPT-2 architecture and pre-trained the model with the CLM objective for 100k steps, with all hyper-parameters following the default setup. To generate a natural language demonstration for the student's trial, we prompt the pre-trained teacher model with the same 5 tokens used for the student model, thereby obtaining the teacher's completion of the sentence.

## 2.3 The reward and reward model

Defining an effective reward in our context is challenging due to the absence of communication games and the lack of access to large-scale human preference annotations. Heuristic reward metrics do not consider the developmental trajectory of language models, which is critical for simulating language acquisition. It's akin to human development where early words are celebrated as milestones, but prolonged reliance on initial language abilities can become a concern. We treat the number of training steps as the neural model's “age"(Chang and Bergen, 2022). A language model that generates fluent text at 500 steps, which typically emerges around 5,000 steps, should be rewarded for its accelerated learning. Conversely, if the language production quality in the student remains the same at 50,000 steps, it should be penalized. This process is illustrated in Figure 1(c).

```latex
Algorithm 1 TRIAL-AND-DEMONSTRATION
1: Input: student model $\pi _ { \theta } ,$ teacher model ${ \hat { \pi } } _ { \varphi } ,$ reward
model $R _ { \phi } ,$ alternating schedule $( c , r )$ , training corpus ${ \mathcal { C } } .$
2: for $n , S$ in enumerate $( \mathcal { C } )$ do
3: $t _ { 1 } , \cdot \cdot \cdot , t _ { l } = \mathrm { T o K E N I Z E } ( S )$
4: if r $\imath \% ( c + r ) \leq c$ then
5: //Non-interactive learning
6: Gradient descent $\nabla _ { \theta } \ \mathcal { L } _ { \theta } ^ { \mathrm { c l m } } ( [ t _ { 1 } , \cdot \cdot \cdot \ , t _ { l } ] )$ (Eq. 2)
7: else
8: // Interactive learning
9: $S _ { \mathrm { t r i a l } } \gets \mathrm { P R O M P T } \big ( \pi _ { \theta } , [ t _ { 1 } , t _ { 2 } , \cdot \cdot \cdot , t _ { 5 } ] \big )$
10: $r _ { \mathrm { t r i a l } } = R _ { \phi } ( S _ { \mathrm { t r i a l } } ) \stackrel { \cdot } { - } \log n$
11: $S _ { \mathrm { d e m o } } \gets \mathrm { P R O M P T } \left( \hat { \pi } _ { \varphi , } \left[ t _ { 1 } , t _ { 2 } , \cdot \cdot \cdot , t _ { 5 } \right] \right)$
12: $r _ { \mathrm { d e m o } } = R _ { \phi } ( S _ { \mathrm { d e m o } } ) ^ { ` } { \ - } \log n$
13: Train batch $B = ( S _ { \mathrm { t r i a l } } , r _ { \mathrm { t r i a l } } ) \cup ( S _ { \mathrm { d e m o } } , r _ { \mathrm { d e m o } } )$
14: Policy update $\nabla _ { \theta } \ { \mathcal { L } } _ { \theta } ^ { \mathrm { p p o } } ( B )$ (See A.3)
15: end if
16: end for
17: Output: πθ
```

To train a neural age predictor, we utilize the developmental trajectory of the teacher model by saving over 100 checkpoints at various training steps. For each checkpoint, we select 25,000 contexts (each consisting of 5 tokens) from the test set and prompt the model to generate continuations for each context. This process generates a dataset comprising over 2.5 million (text, step) pairs. We then use this dataset to fine-tune a LLaMA-2-7B language model (Touvron et al., 2023), incorporating a 1-layer linear head for regression. At step $n ,$ the student model $\pi _ { \theta }$ parameterized by θ produces a sentence $s$ of l tokens $t _ { 1 } , \cdots , t _ { l }$ . We use the neural age predictor $R _ { \phi }$ to estimate the logarithm of the expected training step î where this sentence typically emerges. Hence, the age-conditioned reward $r ( S , n )$ is given by:

$$
r : = \log ( \hat { n } / n ) = R _ { \phi } ( S ) - \log n\tag{1}
$$

## 2.4 Alternating interactive and non-interactive language learning

As shown in Figure 1(c-d), our TnD framework alternates between two forms of language learning: (1) interactive learning, in which corrective feedback is taken through reinforcement learning, utilizing rewards derived from both the student's trials and the teacher's demonstrations, and (2) noninteractive learning, which emulates the natural language exposure experienced by learners and is facilitated through causal language modeling as adopted by generative language models.

## 2.4.1 Interactive language learning setup.

Reinforcement learning has been applied to language models, such as the success of games (Narasimhan et al., 2015; He et al., 2016), heuristic scores (Ranzato et al., 2016; Nikolaus and Fourtassi, 2021), and models of human preferences (Ziegler et al., 2019; Ouyang et al., 2022). The key is to view the language production in language models as actions within a vocabularydefined action space. Formally, the model $\pi _ { \theta }$ is given a context of 5 tokens (i.e., the initial state $s _ { 0 } = \{ t _ { 1 } , \cdots , t _ { 5 } \} )$ to produce the next token (i.e., the next action $a _ { 1 } = t _ { 6 } )$ . It lands in the next state $s _ { 1 } ~ = ~ \{ t _ { 1 } \cdot \cdot \cdot t _ { 6 } \}$ , and this process repeats until the sentence concludes. Following this, rewards can be computed according to $\operatorname { E q }$ 1 for the student's trials and the teacher's demonstrations. The goal is to maximize the expected return (i.e., the expected cumulative future reward) following πθ along the interaction, with the trials and demonstrations both in the training batch. Inspired and taking the computational infrastructure in recent advances in reinforcement learning from human feedback (RLHF) (Ouyang et al., 2022), we use Proximal Policy Optimization (PPO) (Schulman et al., 2017) algorithm with a clipped surrogate objective $\mathcal { L } _ { \theta } ^ { \mathrm { p p o } } .$ which is the primary variant in modern large language models.1 We applied two modifications in the implementation of the RLHF algorithm, which include the involvement of demonstration in policy update, and the removal of KL-divergence. We refer to Appendix A.3 for mathematical details.

Demonstration in policy update. The original PPO algorithm only learns from the agent's own trials, i.e., the sentences it generated before. We expand the training batch to add teacher's demonstrations: At each step, we sample the sentences generated by the latest student model and also collect those from the teacher model with the same prompts. Subsequently, we regard both of them as the training batch, and apply the policy update to improve the student model. Intuitively, our goal is to encourage the student model to imitate and repeat the teacher's demonstration during training. As there is a reward disparity between the student's trial and the teacher's demonstration, we motivate the student model to learn a better language generation policy.

Removal of KL-divergence objective. The loss function in the conventional RLHF algorithm involves a KL-divergence term between the updated student model and a reference model, which is usually a fine-tuned language model or the initial student model before gradient updates. The goal is to penalize the learned policy that largely deviates from the referenced policy. Different from the conventional approach, this penalty is not preferred in the TnD setting. This change encourages the student model to explore unfamiliar words during the training, which enables relatively significant updates, as well as eliminates biases from overadherence to a reference model.

## 2.4.2 Non-interactive language learning setup.

While our interactive language learning replicates active engagement with language through corrective feedback, it's also crucial to simulate human's passive exposure to language, as emphasized in prior psychological (Smith and Yu, 2008) and computational (Nikolaus and Fourtassi, 2021) studies. We implement this with the causal language modeling objective, which is adopted by most generative language models. Consider a sentence $t _ { 1 } , \cdots , t _ { l }$ in the corpus with l tokens. The causal language modeling objective is to predict the next token $t _ { i + 1 }$ given the previous context $t _ { < i }$ by minimizing:

$$
\mathcal { L } _ { \theta } ^ { \mathrm { c l m } } = - \sum _ { i = 1 } ^ { l - 1 } \log P ( t _ { i + 1 } | t _ { \le i } ; \theta )\tag{2}
$$

## 2.4.3 Alternating interactive and non-interactive learning.

Following the setup of Nikolaus and Fourtassi (2021), we adopt an alternating learning schedule over interactive and non-interactive language learning, i.e., perform c steps of causal language modeling, followed by r steps of reinforcement learning, in a continuous cycle (Algorithm 1). We select $c = 3$ and $r = 1$ , respectively, leading to 1 PPO update following every 3 CLM updates. We justify this design choice and present experiments on other hyper-parameters in Appendix B.6.

## 3 Experiment and Evaluations

## 3.1 Experiment setup

Training corpora. We repeat our study on two training corpora: (1) the BookCorpus (Zhu et al., 2015), which is commonly used for training neural language models; and (2) the BabyLM Corpus, a more developmentally plausible dataset provided by the BabyLM Challenge (Round 1) (Warstadt et al., 2023). Notably, the BabyLM dataset has less than 100M words and contains a higher proportion of transcribed speech, e.g., the CHILDES corpus (MacWhinney, 2000). For each corpus, we keep 80% for training and 20% for evaluation.

![](images/0ecf21fbb5747731cf8b69f033097943b169580a3de42be1b70558e84aa0c758.jpg)  
(a) Learning curve of CMN words on BookCorpus.

![](images/001479d557d14d7275a8615bb475caec406e706ebf53162fece84b924db81b4b.jpg)  
(b) Learning curve of CDI words on BookCorpus.

![](images/091b53dc57f0dadda97b894ba15c62359197fbbdf06170861e8d1eecd006db41.jpg)  
(c) Learning curve of CMN words on BabyLM.

![](images/49ec2bc2b7b2b033f5dc417c21ceaf2febe968a5341808acf66ba26e36d0acc8.jpg)  
(d) Learning curve of CDI words on BabyLM.

Figure 3: On 2 training corpora and 2 test vocabulary, we aggregate 5 random seeds and present the fitted learning curves of mean surprisal over log10training steps, with nAoA@0. 5 of each curve indicated by a vertical dashed line.  
![](images/6bb4cb2317080e5dd6aa27d1f3134d5024271052e66e759658d589f515d87c53.jpg)  
(a) nAoA of CMN words on BookCorpus.

![](images/d6615904b08c7044145f7c1c6a57ae8139d76a28909f4c707fcc67c4c2b363ab.jpg)  
(b) nAoA of CDI words on BookCorpus.

![](images/8f59df75d9289bb4ddc49b1282c48874162c3e7569e3a2a563245b92fc684618.jpg)  
(c) nAoA of CMN words on BabyLM.

![](images/c1721e2f5bb2021d22e020164ab3ca911a453de99a545f425ab2f8a38f4f017a.jpg)  
(d) nAoA of CDI words on BabyLM.

Figure 4: On 2 training corpora and 2 test vocabulary, we aggregate 5 random seeds and present the neural age of acquisition (nAoA) at different surprisal thresholds from 0.5 to 0.95 with a step of 0.05.  
![](images/fbefa6d016540f41227dae6d2bad601c51a4a4ef50c7542e3987139a3007a122.jpg)  
(a) Effective vocabulary size

![](images/fd118d289549b384e7e56f6d9a84031c88dfaf84a8b9ab2e1363c1fd00bf2d58.jpg)  
(b) Effective vocabulary size

![](images/90ddfbe4d9715fb42a5de6d5521651869bb8b2cb8b820d3fed3a3eb949aa315a.jpg)  
(c) Effective vocabulary size in CMN words on BabvIM

![](images/8146f9754125821ddc48d31b1a39e1c40f102c5f87c633dea9f3479befd2b5ee.jpg)  
(d) Effective vocabulary size in CDI words on BabyLM.  
Figure 5: On 2 training corpora and 2 test vocabulary, we aggregate 5 random seeds and evaluate the effective vocabulary size over $\log _ { 1 0 }$ training steps. The dashed lines mark the tested vocabulary size.

Baselines and ablation variants. To reliably assess the importance or impact of a mechanism in a language learning system, computational experiments should be conducted under controlled ablation studies (Warstadt and Bowman, 2022). We describe our baselines below and study other possible setups in Appendix B.5.

• The CLM model, which adheres to the original GPT-2 pre-training with only CLM objective;

• The TnD model, which implements the trial-anddemonstration framework described in Section 2;

• The Trial model, which is the TnD framework with only student trials (no demonstrations);

• The Demo model, which is the TnD framework with only teacher demonstrations (no trials);

These baselines are designed in a controlled manner to ensure fair ablations. For each combination of corpus and baseline, the training process is conducted on 5 random seeds for 10k steps. We discuss more about other possible baseline setups and the hyper-parameters in Appendix B.

## 3.2 Evaluation

Testing vocabulary. We specify two sets of vocabulary for evaluation.

• The CMN set, consisting of common words that appear frequently in both corpora, covering a wide range of words and parts of speech.

• The CDI set, consisting of words from the MacArthur-Bates Communicative Development Inventories (CDIs) (Fenson et al., 2006). Following Portelance et al. (2020), we excluded any items comprising multiple words (such as “choochoo") from our dataset, as the tokenizer would treat these as distinct items.

We select these two vocabulary sets as CMN offers broader coverage while CDI is more frequencyneutral and is used to assess children's early vocabulary development. Following Chang and Bergen (2022), we remove words with less than 100 occurrences in the evaluation set of each corpus to ensure reliable results, and keep at most 512 samples for each word. This leads to 309 common words in

<table><tr><td rowspan="2">Method</td><td rowspan="2">nAoA↓</td><td colspan="2">BabyLM Corpus</td><td colspan="2">BookCorpus</td></tr><tr><td>CMN</td><td>CDI</td><td>CMN</td><td>CDI</td></tr><tr><td rowspan="2">CLM</td><td>@.5</td><td> $2 . 9 4 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $2 . 9 3 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $2 . 9 0 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $3 . 0 0 { \scriptstyle \pm 0 . 0 1 }$ </td></tr><tr><td>@[.5:.95]</td><td> $3 . 1 8 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $3 . 1 9 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $3 . 1 4 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $3 . 2 6 { \scriptstyle \pm 0 . 0 1 }$ </td></tr><tr><td rowspan="2">Trial</td><td>@.5</td><td></td><td> $3 . 1 7 _ { \pm 0 . 0 1 } ^ { 2 . 8 6 _ { \pm 0 . 0 1 } }$ </td><td> $2 . 9 7 _ { \pm 0 . 0 1 }$ </td><td> $3 . 0 5 _ { \pm 0 . 0 1 }$ </td></tr><tr><td>@[.5:.95]</td><td> $\stackrel { 2 . 9 0 \pm 0 . 0 1 } { 3 . 2 0 \pm 0 . 0 1 }$ </td><td></td><td> $3 . 2 1 _ { \pm 0 . 0 1 }$ </td><td> $3 . 3 0 { \scriptstyle \pm 0 . 0 1 }$ </td></tr><tr><td rowspan="2">Demo</td><td>@.5</td><td> $2 . 6 0 { \scriptstyle \pm 0 . 0 2 }$ </td><td> $2 . 4 7 _ { \pm 0 . 0 3 }$ </td><td> $2 . 5 1 _ { \pm 0 . 0 2 }$ </td><td> $2 . 6 6 _ { \pm 0 . 0 2 }$ </td></tr><tr><td>@[.5:.95]</td><td> $3 . 1 5 _ { \pm 0 . 0 1 }$ </td><td> $3 . 0 8 _ { \pm 0 . 0 2 }$ </td><td> $3 . 0 5 _ { \pm 0 . 0 1 }$ </td><td> $\pm . 9 7 _ { \pm 0 . 0 2 }$ </td></tr><tr><td rowspan="2">TnD</td><td>@.5</td><td> $\mathbf { 2 . 1 0 { \scriptstyle \pm 0 . 0 2 } }$ </td><td> $\mathbf { 2 . 1 0 _ { \pm 0 . 0 3 } }$ </td><td> ${ \bf 2 . 1 1 { \scriptstyle \pm 0 . 0 2 } }$ </td><td> $2 . 4 6 _ { \pm 0 . 0 3 }$ </td></tr><tr><td>@[.5:.95]</td><td> $2 . 9 5 _ { \pm 0 . 0 2 }$ </td><td>2.87±0.03</td><td> $\pm . 9 \mathbf { 0 } _ { \pm 0 . 0 2 }$ </td><td> $3 . 1 1 { \scriptstyle \pm 0 . 0 2 }$ </td></tr></table>

Table 1: For each baseline and setup, we report the neural age of acquisition (nAoA) with standard errors at 0.50 cutoff and averaged over surprisal thresholds from 0.50 to 0.95 with a step of 0.05.

CMN, as well as 345 words in CDI for BookCorpus and 243 words in CDI for BabyLM Corpus.

Surprisal and learning curves. In line with previous work, we use the mean surprisal (logperplexity) of a word to quantify the quality of the model's predictions for this word. For each occurrence of a word w, the surprisal is given by $- \log _ { 2 } P ( w )$ , and we average all occurrences. To visualize word acquisition from high surprisal to lower surprisal, we evaluate the student model throughout training and plot the surprisal values over logarithm training steps, leading to a learning curve for individual words and the overall vocabulary. We observe a similar pattern reported by Chang and Bergen (2022) that learning curves tend to level off at a local plateau, which aligns with the unigram surprisal. This phenomenon renders the single-sigmoid model unreliable for capturing the complexity, and we adopt a double-sigmoid function to fit the learning curve. We justify this choice and discuss more on the patterns in Appendix B.2 Neural age of acquisition (nAoA). To evaluate the speed at which the student model acquires a word, we employ the neural age of acquisition (nAoA). Prior research (Chang and Bergen, 2022; Chang et al., 2023) has used a surprisal cutoff of 50% between the minimum and maximum surprisal levels. This is akin to the method used to determine children's age of acquisition, where the cutoff is set at the point when 50% of children are observed to produce a word (Braginsky et al., 2016). To further enhance the robustness of this metric, we average nAoA over different surprisal thresholds from 0.5 to 0.95 with a step of 0.05, denoted as nAoA@[0.50:0.95]. It's important to note that nAoA serves as a metric to assess the speed at which a model acquires a word, rather than the quality of word learning. A model might learn a word quickly, indicated by a low nAoA, yet not master it effectively, indicated by a high surprisal. Combining both metrics offers a thorough evaluation of word learning.

Effective vocabulary size. Finally, we assess the effective vocabulary size relative to a test set of vocabulary. A word is deemed acquired at step n if $\mathsf { n A o A @ } . 5 \theta \leq n$ . This approach yields a monotonically increasing curve that illustrates the growth of the effective vocabulary over time.

## 4 Main Results and Findings

## 4.1 Corrective feedback accelerates neural word acquisition

To evaluate CMN and CDI words on two corpora, we aggregate 5 random seeds and present the learning curves in Figure 3, the neural age of acquisition in Figure 4, and the effective vocabulary size in Figure 5. Figure 3 reveals that the TnD learning framework significantly accelerates word acquisition in training, outperforming other baselines. This acceleration is attributed to the critical roles of both trials and demonstrations in the learning process. With only teacher demonstrations, the student model acquires words faster than with the plain CLM baseline alone, though not as rapidly as when active trials are incorporated in the TnD framework. Conversely, without the teacher's demonstrations, the student's trials in the wild do not yield a marked improvement, resulting in performance comparable to the CLM baseline. We refer to Figure 13 and 14 in the Appendix for the ridgeline and scatter plot of words and their nAoA.

Figure 4 and Table 1 present nAoA at different surprisal thresholds from 0.50 to 0.95 with a step of 0.05. We find that the TnD learning framework is particularly beneficial during the earlier stages of word acquisition, as it significantly outperforms the CLM baseline on nAoA@0.50, but is eventually onpar with the CLM baseline on nAoA@0.95 towards the end of training. As a result, it can be seen from Figure 5 that students under the TnD framework quickly picked up a large volume of effective vocabulary, but eventually their vocabulary capacities have converged to the CLM baseline as expected. We further evaluate the final model on downstream natural language understanding (NLU) tasks. The TnD model performs on par with the CLM model (See Appendix B.4). Overall, our findings show that corrective feedback accelerates the student model's neural word acquisition process, yet the student eventually converges to the teacher model's performance. This contradicts the previ-

![](images/7171e2553bc9a96061d4f3b541b22b660d398c315731da2148574147bad8187e.jpg)  
(a) Learning curve sof smaller models of CMN words on BookCorpus.

![](images/8bca5adff33f21b6dc2d599207307ebb900760ad93bc0d115dc644d8f5265ee7.jpg)  
(b) Learning curve sof smaller models of CDI words on BookCorpus.

![](images/010314876c085d24112d8c4fd352d535379edf508339ce4df4111d1837493de8.jpg)  
(c) Learning curve sof smaller models of CMN words on BabyLM.

![](images/619ea390b377d169690f9f21f312835300f4ea2050602a9ee7cdb353077cbc3c.jpg)  
(d) Learning curve sof smaller models of CDI words on BabyLM.

Figure 6: We further present the fitted learning curves on smaller student models $\scriptstyle ( \mathsf { T n D } _ { d = 5 8 8 / 3 6 0 / 2 5 0 } )$ of mean surprisal over $\log _ { 1 0 }$ training steps, with nAoA@0. 5 of each curve indicated by a vertical dashed line.  
![](images/0bf9c57d5b9802210ce7cfd73dc7c8d2a2025183edd98354f503a580e69d800f.jpg)  
(a) Influence of teacher's word preferences in CMN words on BabyLM.

![](images/4ae20fde7797d09ad72bf6ad15633665ca770664a5103a244d93e416cab6454f.jpg)  
(b) Influence of teacher's word preferences in CDI words on BabyLM.

![](images/314cda2cd89cf633e57a8f10c0efcbad96edc866a04cd7bb1df268f5aa5856f6.jpg)  
(c) Influence of teacher's word preferences in CMN words on BookCorpus.

![](images/2dee43129cbce6cd9244406e5d79ae4eab70575c75e320a13e8b157befbdd5e6.jpg)  
(d) Influence of teacher's word preferences in CDI words on BookCorpus.

Figure 7: For the 40 words to be “masked out"from teacher demonstrations, we repeat the experiment with 5 random seeds and plot the learning curves of these words with those in CLM and TnD baselines.
<table><tr><td rowspan="2">Method</td><td rowspan="2">nAoA↓</td><td colspan="2">BabyLM Corpus</td><td colspan="2">BookCorpus</td></tr><tr><td>CMN</td><td>CDI</td><td>CMN</td><td>CDI</td></tr><tr><td rowspan="2">CLM</td><td>@.5</td><td> $\begin{array} { c } { 3 . 0 1 _ { \pm 0 . 0 3 } } \\ { 3 . 2 9 _ { \pm 0 . 0 2 } } \end{array}$ </td><td> $\stackrel { 3 . 0 2 _ { \pm 0 . 0 7 } } { 3 . 3 7 }$ </td><td> $\stackrel { 3 . 0 2 _ { \pm 0 . 0 2 } } { 3 . 2 8 _ { \pm 0 . 0 2 } }$ </td><td> $2 . 8 6 _ { \pm 0 . 0 1 }$ </td></tr><tr><td>@[.5:.95]</td><td></td><td></td><td></td><td>3.09±0.01</td></tr><tr><td rowspan="2">TnD</td><td>@.5</td><td> $^ { 2 . 3 4 _ { \pm 0 . 1 5 } } _ { 3 . 1 8 _ { \pm 0 . 0 7 } }$ </td><td> $^ { 2 . 5 7 _ { \pm 0 . 1 4 } } _ { 3 . 2 2 }$ </td><td></td><td> $^ { 1 . 9 4 _ { \pm 0 . 0 2 } } _ { 2 . 7 6 _ { \pm 0 . 0 4 } }$ </td></tr><tr><td>@[.5:.95]</td><td></td><td></td><td> $^ { 2 . 4 3 _ { \pm 0 . 1 2 } } _ { 3 . 2 2 }$ </td><td></td></tr><tr><td>TnD</td><td>@.5</td><td> $\begin{array} { c } { 3 . 1 4 { \scriptstyle \pm 0 . 0 9 } } \\ { 3 . 4 9 { \scriptstyle \pm 0 . 0 3 } } \end{array}$ </td><td> $3 . 3 2 _ { \pm 0 . 0 4 }$ </td><td>3.10±0.04 2.92±0.02</td><td></td></tr><tr><td></td><td>(Masked) @[.5:.95]</td><td></td><td>3.50±0.03 3.42±0.02</td><td></td><td> $3 . 2 1 { \scriptstyle \pm 0 . 0 2 }$ </td></tr></table>

Table 2: For the 40 words to be “masked out" from teacher demonstrations, we repeat the experiment with 5 random seeds and compared their nAoA with standard errors to those observed in CLM and TnD baselines.

## 4.2 Corrective feedback helps knowledge distillation for smaller student models

ous observations by Nikolaus and Fourtassi (2021) that combining production and perception-based language learning from the start will deteriorate performance. We hypothesize that this discrepancy may result from using the BLEU score as a proxy for rewards from communicative feedback in their work, while our setup incorporates a more explicit form of corrective feedback.

The current age-based reward design assumes that the student and teacher models are the same size. This section investigates whether such a reward function could distill linguistic knowledge to smaller student models. The original student GPT-2 model has a dimension of $d = 7 6 8$ (12 attention heads each with a dimension of 64). We now keep all experimental setups untouched but smaller student models with dimensions of 588 $( 1 2 \times 4 9 )$ , 360 $( 1 0 \times 3 6 )$ , and 250 (10 × 25) respectively. Figure 6 shows that such efficient language learning can still be observed, even when the setting is translated to smaller models. Each TnD model outperforms the CLM baseline of the same size and even surpasses CLM baselines of large capacity in early steps.

## 4.3 Teacher's word preferences in demonstrations affect student

To explore how the teacher model's word selection impacts students' language development, we repeat the experiments where a chosen set of 40 words for each test vocabulary is excluded from teacher demonstrations. To ensure fluent generation, we maintain the presence of essential functional words so these words don't appear in the 40 chosen words. During the language generation process by the teacher model, if a word from this set is to be decoded, we select the next best alternative, ensuring these words were never presented in teacher demonstrations. We show the learning curves for these excluded words in Figure 7 and present the nAoA in Table 2 (in Appendix). Our findings indicate that the teacher model's word choices significantly influence the efficiency of word acquisition by the student model. The absence of words from teacher demonstrations leads to slower learning speed for student models, as evidenced by a higher nAoA, although the student models are ultimately able to learn these words from the corpus and their trials.

![](images/20e57669427ed3ebf0ba47e0535f5538142b1ec33005c3f70aa0de01a8edc72d.jpg)  
(a) The word “have."

![](images/063536eaa3e11f1febb420c43107532cc5d3da1b5137655a1616a472cc9bdcf5.jpg)  
(b) The word “now."  
Figure 8: Examples of per-word learning curves and cumulative word frequency in BookCorpus. The dashed lines mark the log frequency of words (left y-axis) from each source. The solid line and dots mark the word surprisal (right y-axis).

## 4.4 Practice makes perfect in trials

Finally, we conduct experiments to underscore the significance of the student's active trials in the process of word learning. A student model can learn a word from 3 sources: its own trials, teacher demonstrations, and exposure to the corpus. To determine which source contributes most significantly to learning, we begin by plotting the per-word learning curves alongside the cumulative frequency of word encounters in trials, demonstrations, and the corpus. We observe, interestingly, that the learning curves for certain words exhibit a pronounced correlation with the frequency of these words in trials, as exampled in Figure 8 and 9.

We speculate that this pattern may be associated with the part of speech (POS) of the word. Following Portelance et al. (2020), we delve into this phenomenon by focusing on specific subsets of words, including nouns, predicates, and functional words, to explore the relationship further. To evaluate the impact of each source of word acquisition, we consider the cumulative word frequency as a predictor of the word surprisal, and carry out linear regressions complemented by likelihood ratio tests to determine the beta weights for each predictor. Upon analysis, we identify significant collinearity between word frequency in teacher demonstrations and the corpus, as indicated by a high variance inflation factor (VIF), while the word frequency in student trials is less intertwined, exhibiting a moderate VIF below 10. We thus calculate the beta weights separately for pairs (trial, demo) and (trial, corpus), then compute the average beta weights for trials. Together with the Pearson correlation, we summarize the results in Table 3. Negative beta weights signify a negative correlation, with a larger absolute value denoting a stronger association and contribution. Our analysis reveals that the cumulative frequency of words encountered in trials plays a significant role in the acquisition of functional words and predicates. However, this significant contribution does not extend to nouns, indicating a potential impact of active trials on different POS within the learning process. This finding is linguistically intuitive, as function words and predicates are words that require other dependent words to fully express their meaning (Gleitman, 1990) and thus require more practice. We posit that grounding language in the non-linguistic world is essential for acquiring the meanings of words, particularly for concrete noun (Ma et al., 2023).

<table><tr><td rowspan=1 colspan=7>BabyLM Corpus        BookCorpusPOS Freq.    CMN       CDI       CMN       CDI $\beta$    r    β   r    β   r    β   r</td></tr><tr><td rowspan=1 colspan=2>trialnoun demo -0corpus</td><td rowspan=1 colspan=2>-0.36 -0.90.67 -0.93 -0-0.51 -0.93</td><td rowspan=1 colspan=1>-0.25 -0.85.73 -0.89 --0.53 -0.88</td><td rowspan=1 colspan=2>-0.38 -0.92 -0.31 -0.850.56 -0.93 -0.67 -0.87-0.56 -0.93 -0.60 -0.87</td></tr><tr><td rowspan=2 colspan=2>trialcorpus</td><td rowspan=1 colspan=2>-0.70 -0.90</td><td rowspan=1 colspan=1>-0.72 -0.86</td><td rowspan=2 colspan=1>-0.49 -0.92-0.45 -0.93</td><td rowspan=2 colspan=1>-0.54 -0.88-0.40 -0.87</td></tr><tr><td rowspan=1 colspan=2>-0.22 -0.93</td><td rowspan=1 colspan=1>-0.93</td><td rowspan=1 colspan=1>-0.19 -0.90</td></tr><tr><td rowspan=2 colspan=1>func</td><td rowspan=2 colspan=1>trialdemo -0corpus</td><td rowspan=1 colspan=2>-0.67 -0.93</td><td rowspan=1 colspan=1>-0.72 -0.92</td><td rowspan=1 colspan=1>-0.67 -0.94</td><td rowspan=2 colspan=1>-0.59 -0.93-0.37 -0.87-0.35 -0.90</td></tr><tr><td rowspan=1 colspan=2>.39 -0.92 --0.17 -0.92</td><td rowspan=1 colspan=1>0.21 -0.90-0.25 -0.91</td><td rowspan=1 colspan=1>-0.22 -0.91-0.35 -0.92</td></tr></table>

Table 3: For each POS category, we present the beta weights $\beta$ and Pearson correlation r between their mean surprisal and cumulative word frequency over the course of training. These metrics are evaluated based on the student's trials, teacher's demonstrations, and the overall corpus frequency up to the current training step.

## 5 Related Work and Discussions

## 5.1 Interaction in neural language learning

Researchers have emphasized the role of interaction in computational models of language (Bisk et al., 2020; Tsuji et al., 2021). Preliminary efforts have been conducted under specific constraints, such as in domain-specific scenarios (Qu and Chai, 2010; Weston, 2016; Bianchi et al., 2021; Stein et al., 2021; Madotto et al., 2021) or considering particular types of dialogue acts (Zhang et al., 2018; Yuan et al., 2019). More recently, a series of studies have approached language acquisition through the lens of multimodal referential games (Lazaridou et al., 2016; Zhu et al., 2022; Liu et al., 2022), emphasizing the importance of pragmatic inference and communicative feedback in speaker-listener interactions. Early work investigated scenarios where the teacher models actively select training data to optimally assist a passive student learner (ter Hoeve et al., 2022). Nikolaus and Fourtassi (2021) adopted setups where student models learn by producing language and receive feedback of communicative success using the BLEU score (Papineni et al., 2002). However, these models do not receive teachers’ demonstrations or corrections. Our work diverges from these studies as we focus on symmetric teacher-student interactions that require language production from the student and explicit corrective feedback from the teacher, which goes beyond the simple speakerlistener roles. We also adopt massive corpora to study modern transformer-based generative language models, a general approach without domain restrictions. Recent advancements in reinforcement learning from human feedback (RLHF) (Ouyang et al., 2022) mark a breakthrough in interactive language learning. This work leverages the implementation infrastructure of RLHF, but diverges in significant ways. Whereas RLHF aims to align a pretrained language model with human preferences, our objective is to “babysit" a language model from scratch. We refer to existing surveys (Kaufmann et al., 2023; Zheng et al., 2023) and discuss RLHF further in Appendix A.2.

## 5.2 Psychologically motivated analysis of language models

Although language models and human language learners differ in their inductive biases and data sources (Baroni, 2022; Warstadt and Bowman, 2022), several works have looked into the learnability, proficiency, and efficiency of neural language learning, for example, the relationships between word surprisal in language models to various psycholinguistic variables (Portelance et al., 2020; Ma et al., 2023). Recent efforts have shifted towards exploring the developmental trajectories of language models, rather than their end performance (Sellam et al., 2021; Blevins et al., 2022; Biderman et al., 2023; Xia et al., 2023), sparkling further investigations into the developmental aspects of psycholinguistics using computational approaches (Chang and Bergen, 2022; Chang et al.,

2023; Evanson et al., 2023). The scientific rationale behind this is that these models can serve as hypothesis generators or proofs of concept, verifying mechanisms that are practically effective for machines and potentially feasible for human learners (Portelance, 2022; Portelance and Jasbi, 2023). We echo that the benchmark outcomes of language models themselves are insufficient (Baroni, 2022; Portelance, 2022), and researchers need to control the factors that may have contributed to the model's learning process, particularly through conducting ablation studies (Warstadt and Bowman, 2022). In this study, we follow this spirit and investigate the role of trials and demonstrations through systematic computational experiments, assessing its role in neural word acquisition.

## 5.3 Interaction in human language learning

Social interactions are crucial in language acquisition, and the role of caregivers’ feedback has been extensively explored in the field of developmental psychology (Bates et al., 1975; Saxton et al., 2005; Warlaumont et al., 2014). While our findings suggest that corrective feedback could enhance efficient neural word learning, these results should not be generalized to human language learning. We discuss more about the role of interaction in human language learning in Appendix A.1.

## 6 Conclusion

This research introduces a trial-and-demonstration (TnD) learning framework to examine the effectiveness of corrective feedback in neural word acquisition through systematically controlled experiments, assessing how the interplay between student trials and teacher demonstrations contributes to learning efficiency in neural language models. We find that (1) TnD learning accelerates neural word acquisition across student models of different sizes; (2) the teacher's choices of words influence students'word-specific learning efficiency; and (3) a practice-makes-perfect effect is evident by a strong correlation between the frequency of words in trials and their respective learning curves. Our findings confirm the crucial role of interaction in efficient word learning with language models.

Additional experiments. Due to the limited space and primary research scope of this work, we present experiments about other possible baselines besides controllable ablation, downstream natural language understanding task performances, and the robustness of the findings under different hyperparameters in Appendix B.

## Limitations

Iterative setting. This experiment can be conducted iteratively by replacing the teacher model with the student model from previous iterations. While this iterative approach is intriguing, it introduces new complexities that require significant modifications to the current controlled ablation studies. We defer exploration of this approach to future work, as the current study focuses on examining the roles of trials and demonstrations.

The reward. Our study is limited by the use of a single reward model focused on corrective feedback. More realistic scenarios should also encompass communicative feedback, with the success of communication serving as a reward. Additionally, Thorndike (1911) proposed the idea that a child might instinctively feel satisfaction from producing a sound that echoes a meaningful memory. The design of such an intrinsic reward model is cognitively intriguing, and could aid in scaling student models without an external reward model, offering potential benefits for engineering applications.

The reward model. We employ a robust language model (LLaMA-2-7B) as a reward model to concentrate on the roles of trials and demonstrations without concerns about reward quality. Future research should explore the impact of using less accurate reward models.

The tokenizer. One limitation of our approach is the reliance on the Byte Pair Encoding (BPE) tokenizer (Sennrich et al., 2016) inherited from GPT-2. Ideally, the method should be tokenizer-free to facilitate the learning of early language elements such as sound effects and animal sounds, e.g., “baa-baa” in CDI (Fenson et al., 2006), which can be crucial for a more natural and foundational language acquisition process.

Other languages. The present study focuses on English as the subject of investigation due to the available corpus resources. Future research should consider exploring other languages.

## Acknowledgement

This work was supported in part by NSF IIS-1949634 and NSF SES-2128623. Ziqiao Ma is supported in part by the Weinberg Cognitive Science Fellowship. The authors extend their special appreciation to Susan Gelman and Freda Shi for their valuable feedback. The authors would like to thank Run Peng, Yichi Zhang, Jacob Sansom, Zheyuan Zhang, Xuejun Zhang for proofreading and their helpful input during discussions. We thank all anonymous reviewers for their feedback.

## References

Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al. 2022. Constitutional ai: Harmlessness from ai feedback. arXiv preprint arXiv:2212.08073.

Marco Baroni. 2022. On the proper role of linguistically-oriented deep net analysis in linguistic theorizing. Algebraic structures in natural language, pages 1–16.

Elizabeth Bates, Luigia Camaioni, and Virginia Volterra. 1975. The acquisition of performatives prior to speech. Merrill-Palmer quarterly of behavior and development, 21(3):205–226.

Federico Bianchi, Ciro Greco, and Jacopo Tagliabue. 2021. Language in a (search) box: Grounding language learning in real-world human-machine interaction. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 4409–4415.

Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O'Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, et al. 2023. Pythia: A suite for analyzing large language models across training and scaling. In International Conference on Machine Learning, pages 2397–2430. PMLR.

Yonatan Bisk, Ari Holtzman, Jesse Thomason, Jacob Andreas, Yoshua Bengio, Joyce Chai, Mirella Lapata, Angeliki Lazaridou, Jonathan May, Aleksandr Nisnevich, Nicolas Pinto, and Joseph Turian. 2020. Experience grounds language. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 8718–8735. Association for Computational Linguistics.

John Bitchener and Dana R Ferris. 2011. Written corrective feedback in second language acquisition and writing. Routledge, Taylor & Francis Group.

Terra Blevins, Hila Gonen, and Luke Zettlemoyer. 2022. Analyzing the mono-and cross-lingual pretraining dynamics of multilingual language models. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 3575–3590.

Manuel Bohn and Michael C Frank. 2019. The pervasive role of pragmatics in early language. Annual Review of Developmental Psychology, 1:223–249.

Mika Braginsky, Daniel Yurovsky, Virginia A. Marchman, and Mike Frank. 2016. From uh-oh to tomorrow: Predicting age of acquisition for early words across languages. In Cognitive Science.

Roger Brown. 1970. Derivational complexity and order of acquisition. Cognition and Development of Language.

Jerome Bruner. 1985. Child's talk: Learning to use language. Child Language Teaching and Therapy, 1(1):111–114.

Tyler A Chang and Benjamin K Bergen. 2022. Word acquisition in neural language models. Transactions of the Association for Computational Linguistics, 10:1– 16.

Tyler A Chang, Zhuowen Tu, and Benjamin K Bergen. 2023. Characterizing learning curves during language model pre-training: Learning, forgetting, and stability. arXiv preprint arXiv:2308.15419.

Michelle M Chouinard and Eve V Clark. 2003. Adult reformulations of child errors as negative evidence. Journal of child language, 30(3):637–669.

Eve V Clark. 2018. Conversation and language acquisition: A pragmatic approach. Language Learning and Development, 14(3):170–185.

Herbert H Clark. 1996. Using language. Cambridge university press.

Mounira El Tatawy. 2002. Corrective feedback in second language acquisition. Studies in Applied Linguistics and TESOL, 2(2).

Rod Ellis, Shawn Loewen, and Rosemary Erlam. 2006. Implicit and explicit corrective feedback and the acquisition of 12 grammar. Studies in second language acquisition, 28(2):339–368.

Linnea Evanson, Yair Lakretz, and Jean-Rémi King. 2023. Language acquisition: do children and language models follow similar learning stages? In Findings of the Association for Computational Linguistics: ACL 2023, pages 12205–12218.

MJ Farrar. 1992. Negative evidence and grammatical morphemes. Developmental Psychology, 28:91–99.

Larry Fenson, Virginia A Marchman, Donna J Thal, Phillip S Dale, J Steven Reznick, and Elizabeth Bates. 2006. Macarthur-bates communicative development inventories. PsycTESTS Dataset.

Lila Gleitman. 1990. The structural sources of verb meanings. Language acquisition, 1(1):3–55.

Michael Alexander Kirkwood Halliday. 1975. Learning how to mean. In Foundations of language development, pages 239–265. Elsevier

Ji He, Jianshu Chen, Xiaodong He, Jianfeng Gao, Lihong Li, Li Deng, and Mari Ostendorf. 2016. Deep reinforcement learning with a natural language action space. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1621–1630.

Sarah Hiller. 2016. Corrective feedback in first language acquisition. Master's thesis, Universiteit van Amsterdam.

Timo Kaufmann, Paul Weng, Viktor Bengs, and Eyke Hüllermeier. 2023. A survey of reinforcement learning from human feedback. arXiv preprint arXiv:2312.14925.

Jongwoo Ko, Sungnyun Kim, Tianyi Chen, and Se-Young Yun. 2024. DistiLLM: Towards streamlined distillation for large language models. In Forty-first International Conference on Machine Learning.

Patricia K Kuhl. 2004. Early language acquisition: cracking the speech code. Nature reviews neuroscience, 5(11):831–843.

Angeliki Lazaridou, Nghia The Pham, and Marco Baroni. 2016. Towards multi-agent communication-based language learning. arXiv preprint arXiv:1605.07133.

Harrison Lee, Samrat Phatale, Hassan Mansoor, Kellie Lu, Thomas Mesnard, Colton Bishop, Victor Carbune, and Abhinav Rastogi. 2023. Rlaif: Scaling reinforcement learning from human feedback with ai feedback. arXiv preprint arXiv:2309.00267.

Andy Liu, Hao Zhu, Emmy Liu, Yonatan Bisk, and Graham Neubig. 2022. Computational language acquisition with theory of mind. In The Eleventh International Conference on Learning Representations.

Ziqiao Ma, Jiayi Pan, and Joyce Chai. 2023. Worldto-words: Grounded open vocabulary acquisition through fast mapping in vision-language models. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 524–544.

Brian MacWhinney. 2000. The CHILDES project: The database, volume 2. Psychology Press.

Andrea Madotto, Mahdi Namazifar, Joost Huizinga, Piero Molino, Adrien Ecoffet, Huaixiu Zheng, Dian Yu, Alexandros Papangelis, Chandra Khatri, and Gokhan Tur. 2021. Exploration based language learning for text-based games. In Proceedings of the Twenty-Ninth International Conference on International Joint Conferences on Artificial Intelligence, pages 1488–1494.

Gary F Marcus. 1993. Negative evidence in language acquisition. Cognition, 46(1):53–85.

Karthik Narasimhan, Tejas D Kulkarni, and Regina Barzilay. 2015. Language understanding for textbased games using deep reinforcement learning. In Conference on Empirical Methods in Natural Language Processing, EMNLP 2015, pages 1–11. Association for Computational Linguistics (ACL).

Mitja Nikolaus and Abdellah Fourtassi. 2021. Modeling the interaction between perception-based and

production-based learning in children's early acquisition of semantic knowledge. In Proceedings of the 25th Conference on Computational Natural Language Learning, pages 391–407.

Mitja Nikolaus and Abdellah Fourtassi. 2023. Communicative feedback in language acquisition. New Ideas in Psychology, 68:100985.

OpenAI. 2022. Chatgpt: Optimizing language models for dialogue.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744.

Annemarie Sullivan Palincsar. 1986. The role of dialogue in providing scaffolded instruction. Educational psychologist, 21(1-2):73–98.

Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting of the Association for Computational Linguistics, pages 311–318.

Eva Portelance. 2022. Neural Network Approaches to the Study of Word Learning. Ph.D. thesis, Stanford University.

Eva Portelance, Judith Degen, and Michael C Frank. 2020. Predicting age of acquisition in early word learning using recurrent neural networks. In CogSci.

Eva Portelance, Michael C Frank, and Dan Jurafsky. 2024. Learning the meanings of function words from grounded language using a visual question answering model. Cognitive Science, 48(5):e13448.

Eva Portelance and Masoud Jasbi. 2023. The roles of neural networks in language acquisition. PsyArXiv.

Shaolin Qu and Joyce Yue Chai. 2010. Context-based word acquisition for situated dialogue in a virtual world. Journal of Artificial Intelligence Research 37:247-277.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. 2019. Language models are unsupervised multitask learners. OpenAl blog, 1(8):9.

Marc'Aurelio Ranzato, Sumit Chopra, Michael Auli, and Wojciech Zaremba. 2016. Sequence level training with recurrent neural networks. In International Conference on Learning Representations.

Swarnadeep Saha, Peter Hase, and Mohit Bansal. 2023. Can language models teach? teacher explanations improve student performance via personalization. In Thirty-seventh Conference on Neural Information Processing Systems.

Matthew Saxton. 2000. Negative evidence and negative feedback: Immediate effects on the grammaticality of child speech. First Language, 20(60):221–252.

Matthew Saxton, Phillip Backley, and Clare Gallaway. 2005. Negative input for grammatical errors: Effects after a lag of 12 weeks. Journal of child language, 32(3):643–672.

John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. 2016. High-dimensional continuous control using generalized advantage estimation. In Proceedings of the International Conference on Learning Representations (ICLR).

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.

Thibault Sellam, Steve Yadlowsky, Ian Tenney, Jason Wei, Naomi Saphra, Alexander D'Amour, Tal Linzen, Jasmijn Bastings, Iulia Raluca Turc, Jacob Eisenstein, et al. 2021. The multiberts: Bert reproductions for robustness analysis. In International Conference on Learning Representations.

Atsushi Senju and Gergely Csibra. 2008. Gaze following in human infants depends on communicative signals. Current biology, 18(9):668–671.

Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016. Neural machine translation of rare words with subword units. In 54th Annual Meeting of the Association for Computational Linguistics, pages 1715–1725. Association for Computational Linguistics (ACL).

Haoyue Shi, Jiayuan Mao, Kevin Gimpel, and Karen Livescu. 2019. Visually grounded neural syntax acquisition. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 1842–1861.

Linda Smith and Chen Yu. 2008. Infants rapidly learn word-referent mappings via cross-situational statistics. Cognition, 106(3):1558–1568.

Catherine E Snow, Barbara Alexander Pan, Alison Imbens-Bailey, and Jane Herman. 1996. Learning how to say what one means: A longitudinal study of children's speech act use. Social Development, 5(1):56–84.

Katharina Stein, Leonie Harter, and Luisa Geiger. 2021. Shapelurn: An interactive language learning game with logical inference. In Proceedings of the First Workshop on Interactive Learning for Natural Language Processing, pages 16–24.

Maartje ter Hoeve, Evgeny Kharitonov, Dieuwke Hupkes, and Emmanuel Dupoux. 2022. Towards interactive language modeling. In Proceedings of the 1st Workshop on Semiparametric Methods in NLP: Decoupling Logic from Knowledge.

Edward Thorndike. 1911. Animal intelligence: Experimental studies. New York: Macmillan.

Michael Tomasello. 2005. Constructing a language: A usage-based theory of language acquisition. Harvard university press.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Sho Tsuji, Alejandrina Cristia, and Emmanuel Dupoux. 2021. Scala: A blueprint for computational models of language acquisition in social context. Cognition, 213:104779.

Lev S Vygotsky. 1934. Thought and language. MIT press.

Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. 2019. Superglue: A stickier benchmark for general-purpose language understanding systems. Advances in neural information processing systems, 32.

Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. 2018. Glue: A multi-task benchmark and analysis platform for natural language understanding. In Proceedings of the 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP, pages 353-355.

Anne S Warlaumont, Jeffrey A Richards, Jill Gilkerson, and D Kimbrough Oller. 2014. A social feedback loop for speech development and its reduction in autism. Psychological science, 25(7):1314–1324.

Alex Warstadt and Samuel R Bowman. 2022. What artificial neural networks can tell us about human language acquisition. Algebraic Structures in Natural Language, pages 17–60.

Alex Warstadt, Aaron Mueller, Leshem Choshen, Ethan Wilcox, Chengxu Zhuang, Juan Ciro, Rafael Mosquera, Bhargavi Paranjabe, Adina Williams, Tal Linzen, et al. 2023. Findings of the babylm challenge: Sample-efficient pretraining on developmentally plausible corpora. In Proceedings of the BabyLM Challenge at the 27th Conference on Computational Natural Language Learning.

Jason E Weston. 2016. Dialog-based language learning. Advances in Neural Information Processing Systems, 29.

Mengzhou Xia, Mikel Artetxe, Chunting Zhou, Xi Victoria Lin, Ramakanth Pasunuru, Danqi Chen, Luke Zettlemoyer, and Ves Stoyanov. 2023. Training trajectories of language models across scales. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13711–13738.

Xingdi Yuan, Marc-Alexandre Côté, Jie Fu, Zhouhan Lin, Christopher Pal, Yoshua Bengio, and Adam Trischler. 2019. Interactive language learning by question answering. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 2796–2813.

Daniel Yurovsky and Michael C Frank. 2017. Beyond naïve cue combination: Salience and social cues in early word learning. Developmental Science, 20(2):e12349.

Haichao Zhang, Haonan Yu, and Wei Xu. 2018. Interactive language acquisition with one-shot visual concept learning through a conversational game. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2609–2619.

Rui Zheng, Shihan Dou, Songyang Gao, Yuan Hua, Wei Shen, Binghai Wang, Yan Liu, Senjie Jin, Qin Liu, Yuhao Zhou, et al. 2023. Secrets of rlhf in large language models part i: Ppo. arXiv preprint arXiv:2307.04964.

Hao Zhu, Yonatan Bisk, and Graham Neubig. 2022. Language learning from communicative goals and linguistic input. In Proceedings of the Annual Meeting of the Cognitive Science Society, volume 44.

Yukun Zhu, Ryan Kiros, Rich Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Torralba, and Sanja Fidler. 2015. Aligning books and movies: Towards story-like visual explanations by watching movies and reading books. In Proceedings of the IEEE international conference on computer vision, pages 19-27.

Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. 2019. Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593.

## A Trials-and-Demonstrations (TnD) Details and Discussions

## A.1 Interaction in human language learning

To unveil the role of social interactions in language acquisition, a prominent body of research has focused on pragmatic inference – that children exhibit the ability to refine their linguistic knowledge by inferring the communicative intents of others (Senju and Csibra, 2008; Yurovsky and Frank, 2017; Bohn and Frank, 2019). In addition, other researchers have argued for the importance of caregivers' feedback to the language produced by children (Warlaumont et al., 2014), in the form of descriptions, explanations, corrections, etc, to human language development. For example, extensive efforts have been made to examine the effects of communicative feedback on language acquisition, both in developmental psychology (Bates et al., 1975; Snow et al., 1996; Nikolaus and Fourtassi, 2023) and computational modeling (Nikolaus and Fourtassi, 2021; Liu et al., 2022). This type of feedback emphasizes the explicit negotiation of mutual understanding with the conversational partner to achieve and maintain common ground (Clark, 1996). While communicative feedback emphasizes the success and failure of communication, the feedback we study in this study is more akin to corrective feedback. This type of feedback involves responses from caregivers which offer corrections to possible errors in children's speech, including variants such as negative evidence, reformulations, or recasts (Farrar, 1992; Saxton, 2000; Chouinard and Clark, 2003; Saxton et al., 2005; Hiller, 2016). Although corrective feedback is shown to be helpful in second language acquisition (El Tatawy, 2002; Ellis et al., 2006; Bitchener and Ferris, 2011), researchers largely dispute its availability and effectiveness in human first language acquisition (Brown, 1970; Marcus, 1993). While our findings suggest that corrective feedback through demonstrations can enhance efficient neural language learning, these results should not be generalized to human language learning, where demonstrations are much less frequent in the noisy feedback that children typically receive.

## A.2 Relationship with RLHF

Reinforcement learning enables language systems to learn from feedback in the form of rewards from games (Narasimhan et al., 2015; He et al., 2016) or heuristic scores (Ranzato et al., 2016; Nikolaus and Fourtassi, 2021). Recent advancements in reinforcement learning from human feedback (RLHF) have generated considerable excitement, especially in its application to large language models such as ChatGPT (OpenAI, 2022). Reinforcement learning is employed to align the model's policy with human preferences, utilizing human-annotated preference data (Ziegler et al., 2019; Ouyang et al., 2022) or AI models acting as proxies for human judgment (Bai et al., 2022; Lee et al., 2023). We refer to Kaufmann et al. (2023) and Zheng et al. (2023) for more details. Our work leverages the implementation infrastructure of RLHF but diverges in significant ways: whereas RLHF aims to align an existing language model with human preferences, our objective is to “"babysit" a language model from scratch using reinforcement learning, specifically to model the process of receiving and integrating corrective feedback.

## A.3 Preliminaries

Inspired by recent work in RLHF, we use Proximal Policy Optimization (PPO) (Schulman et al., 2017) to train the student model. Consider a model $\pi _ { \theta }$ whose the current state $s _ { i }$ is a sequence of tokens $s _ { i } = t _ { 1 } , \cdots , t _ { i } .$ , and receives a reward $r _ { i }$ . We outline the key components and refer to Zheng et al. (2023) for more details.

Clipped surrogate objective. The clipped surrogate objective is defined as:

$$
\mathcal { L } _ { \theta } ^ { \mathrm { p g } } = \mathbb { E } _ { i } \big [ \operatorname* { m i n } \big ( \Pi _ { i } ( \theta ) , \mathrm { C L I P } \big ( \Pi _ { i } ( \theta ) , \epsilon \big ) \big ) A _ { i } \big ]\tag{3}
$$

where

$$
\Pi _ { i } ( \theta ) : = \frac { \pi _ { \theta } ( t _ { i + 1 } \mid s _ { i } ) } { \pi _ { \theta _ { \mathrm { o l d } } } ( t _ { i + 1 } \mid s _ { i } ) }\tag{4}
$$

The CLIP function clips the value within (1 — $\epsilon , 1 + \epsilon )$ , which regularizes the policy from drastic changes to ensure robustness.

Generalized advantage estimation. The advantage function $A _ { i }$ at step i is estimated by the Generalized Advantage Estimation (GAE) algorithm (Schulman et al., 2016) for a balanced bias-variance trade-off:

$$
A _ { i } = \sum _ { k = 0 } ^ { \infty } ( \gamma \lambda ) ^ { k } \delta _ { i + k }\tag{5}
$$

where $\gamma$ is the discount factor, λ is a hyperparameter controlling the trade-off between bias and variance, and $\delta _ { i } = r _ { i } + \gamma V ( s _ { i + 1 } ) - V ( s _ { i } )$ is the temporal difference (TD) error. The reward r is defined in Section 2.3.

Proximal policy optimization. During the same time, PPO also estimates and optimizes its value function $V _ { \theta _ { v h } }$ with MSE loss:

$$
\mathcal { L } _ { \theta _ { v h } } ^ { \mathrm { v a l u e } } = \mathbb { E } _ { i } \left[ \left( V _ { \theta _ { v h } } ( s _ { i } ) - \hat { V } _ { i } \right) ^ { 2 } \right]\tag{6}
$$

where $V _ { \theta _ { v h } } ( s _ { i } )$ is the estimated value, and $\hat { V _ { i } }$ is the target value from GAE. Differing from conventional implementation, our approach didn't employ a reference model for the KL penalty. This deviation emphasizes the language model's evolutionary nature during its training, enabling more significant updates and eliminating biases from overadherence to a reference model. The final reinforcement learning loss is a linear combination:

$$
\mathcal { L } _ { \theta } ^ { \mathrm { p p o } } = \mathcal { L } ^ { \mathrm { p g } } ( \theta ) + c \cdot \mathcal { L } _ { \theta _ { v h } } ^ { \mathrm { v a l u e } }\tag{7}
$$

where $c \in [ 0 , 1 ]$

## B Additional Experiments, Results, and Discussions

## B.1 Reproducibility

Test vocabulary. We include the list of words and the evaluation datasets within the code.2

Training details. We randomly initialize the GPT-2 student models. For each combination of corpus and baseline, the training process is conducted 5 times for 10k steps, each with a different random seed. We utilize the top-k decoding strategy for language generation, setting $k = 2 0$ . The learning rate for reinforcement learning is set to $2 e ^ { - 5 }$ . We follow the default setting for other PPO hyper-parameters, such as a clip range $( \epsilon = 0 . 2 )$ from the TRL library. All other hyper-parameters for causal language modeling, including the learning rate set at 1e−4 and a batch size of 128, remain consistent with the training setup of GPT-2 by Chang and Bergen (2022).

Checkpointing. We save the intermediate steps:   
[2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40,   
45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110,   
120, 130, 140, 150, 160, 170, 180, 190, 200, 210,   
220, 230, 240, 250, 260, 270, 280, 290, 300, 310,   
320, 330, 340, 350, 360, 370, 380, 390, 400, 410,   
420, 430, 440, 450, 460, 470, 480, 490, 500, 550,   
600, 650, 700, 750, 800, 850, 900, 950, 1000, 1500,   
2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500,   
6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500,   
10000, 20000, 30000, 40000, 50000, 60000, 70000,   
80000, 90000, 100000].

Computational resources. In TnD training, each experiment to train a student model using TnD is conducted on 2 A40 GPUs for 36 hours. In teacher model pre-training, we distribute the computation over 4 A40 GPUs with batch size 32 per device for 20 hours. To fine-tune the neural age predictor, we use a LLaMA-2-7B model (Touvron et al., 2023) with regression head on our developmental trajectory (text and step pairs) dataset. To ensure the prediction quality and save computation resources, we fine-tune our model using mixed precision on all parameters. The training is distributed over 8 A40 GPUs with batch size 8 per device with fullyshared data parallel. The learning rate is $5 e ^ { - 5 }$

## B.2 Additional Results

nAoA distributions. We present the ridgeline and scatter plot of words and their neural age of acquisition (nAoA) in BabyLM Corpus (Figure 13) and BookCorpus (Figure 14).

Surprisal and learning curves. We observe a similar pattern reported by Chang and Bergen (2022) that learning curves tend to level off at a local plateau, which aligns with the unigram surprisal. This phenomenon renders the singlesigmoid model unreliable for capturing the complexity, and we adopt a double-sigmoid function to fit the learning curve. We run regression between the plateau of double-sigmoid curves and the unigram surprisals calculated from all sources of word occurrences (Figure 10). We find a strong correspondence between the plateau and unigram frequency, suggesting that the double-sigmoid function is a better option than the single-sigmoid function to fit learning curves. To analyze longer learning curves of up to 1M steps, more complex functions such as linear GAMs have been adopted (Chang et al., 2023). For our purposes, a double-sigmoid function suffices.

## B.3 Teacher's word preferences in demonstrations affect student

To explore how the teacher model's word selection impacts students' language development, we repeat the experiments where a chosen set of 40 words for each test vocabulary is excluded from teacher demonstrations. To ensure fluent generation, we maintain the presence of essential functional words so these words don't appear in the 40 chosen words. During the language generation process by the teacher model, if a word from this set is to be decoded, we select the next best alternative, ensuring these words were never presented in teacher demonstrations. We depict the learning curves for these excluded words in Figure 7 and present the nAoA in Table 2. Our findings indicate that the teacher model's word choices significantly influence the efficiency of word acquisition by the student model. The absence of words from teacher demonstrations leads to slower learning speed for the student models, as evidenced by a higher nAoA, although the student models are ultimately able to learn these words from the corpus and their trials.

![](images/9589e812d448a172851834c1c009b7a75a7177f989fd84741e460944c7905377.jpg)  
(a) The word "hear."

![](images/5c0c0d534601ac07d3a872e256f69ec4033f9387571a156e663d4a199f6528c2.jpg)  
(b) The word “there."

![](images/0fb3bb65022729435402a9d6019e122be00230992792dbc9d5559b9c072eb3fe.jpg)  
(c) The word “go."

![](images/0697dd6e1becf4c59f319935f24e89bc41c16d278b03c37a3c101f1606c4b713.jpg)  
(d) The word “take."

![](images/8c4bcc3645ef5c938bbc41cba0e7ef238ac47a8a0609b7a228b8953fce70bfb0.jpg)  
(e) The word “if."

![](images/b8ddc4cb6a1e4677a8c05d6959b4c9506af26e512fd84b18efc354043d707ba1.jpg)  
(f) The word “back."

![](images/c8bec2904675b8d2896f976e79b722058b9fccf5acf51cc191838eef6ea7944b.jpg)  
(g) The word “light."

![](images/374b758cec2b69766c18d316ddeb25fe615e055f474e74e429c63bea586f674b.jpg)  
(h) The word “car."

Figure 9: Examples of per-word learning curves and cumulative word frequency in BookCorpus.  
![](images/23d6cad3c8b1efb1292424444276ee50c1b56e12931ae625432a562b91019833.jpg)  
(a) Plateau v.s. unigram surprisal in BookCorpus.

![](images/f65e06f1a59b543a710bdbe0c041537d9dbd2e67c8ce5cc664a92e903a575942.jpg)  
(b) Plateau v.s. unigram surprisal in BabyLM.  
Figure 10: Sigmoid plateau v.s. unigram surprisal.

Additional per-word learning curves. We report additional examples of per-word learning curves and cumulative word frequency over time in Figure 9. Predicates and function words, such as “back" and “go", have a high correlation between surprisal and cumulative word frequency on the student model's trial, shown in Figure 9(a-f). Nouns such as "light" and “car", however, depict a less correlation between the word's learning curve and trial's frequency, shown in Figure 9(g-h).

## B.4 Downstream evaluation on NLU tasks

We evaluate the final model on downstream natural language understanding (NLU) tasks. Specifically, we fine-tune the final CLM and TnD models (from the BabyLM corpus) on the BabyLM round 1 NLU evaluation set, which is based on (Super)GLUE (Wang et al., 2018, 2019). Table 4 shows that the TnD model performs on par with the CLM model, slightly better overall. We observe that the TnD model did significantly better on the Recognizing Textual Entailment (RTE, which requires determining inferential relationships between hypothesis and premise) task, but underperforms on the Question-Answering NLI (QNLI, which requires comprehending longer paragraphs).

## B.5 Discussion and experiment on other possible baselines

Causal language modeling using teachergenerated text (naive content distillation). One possible baseline is to re-run the CLM baseline using teacher/student-generated texts, rather than the corpus text, at a non-interactive step. This experiment is controlled over the language input to the CLM and TnD baselines, but is not a fair setting for CLM as the student's trials are usually of poor quality in initial steps. We replace the PPO updates in TnD with causal language modeling on both texts from the student model's trial and the teacher model's demonstration. We find that the resulting $\mathsf { C L M } _ { \mathsf { T n D } }$ baseline achieves an almost identical overall learning curve as CLM (Figure 11b).

Causal language modeling using teachergenerated text (naive content distillation). Our main difference from the existing knowledge distillation methods for LLMs is that we train a student model from scratch, rather than requiring a pretrained but weaker student model. For example, Ko et al. (2024) adopt the T5 language model as the student. Closest to our work is Nikolaus and Fourtassi (2021), which also adopts an ablation study setup. We did not compare to their method as a baseline as they noticed that combining production and perception-based language learning does not work from scratch. To the best of our knowledge, our TnD method is the only interactive language learning algorithm that functions well with a student model from scratch.

<table><tr><td>Model</td><td colspan="10">Average CoLA SST-2 MRPC QQP MNLI  $\mathrm { M N L I _ { m m } }$  QNLI RTE BoolQ MultiRC WSC</td></tr><tr><td> $\mathsf { C L M } + \mathsf { B a b y L M }$   $\mathsf { T n D } + \mathbf { B a b j } \mathbf { L M }$ </td><td>67.2 67.5</td><td>69.8 66.6</td><td>84.4 82.5</td><td>76.2 75.4</td><td>79.1 79.0</td><td>67.2 67.4</td><td>68.0 69.0</td><td>68.5 48.5 61.0 60.6</td><td>63.6 64.9</td><td>52.0 54.5</td><td>61.4 61.4</td></tr><tr><td> $\mathsf { C L M } + \mathsf { B o o k C o r p u s }$   $\mathsf { T n D } + \mathsf { B o o k C o r p u s }$ </td><td>65.5 65.8</td><td>67.6 67.5</td><td>85.4 83.3</td><td>73.2 77.3</td><td>78.3 77.5</td><td>66.0 66.7</td><td>67.1 67.4</td><td>60.1 45.5 66.4 45.5</td><td>65.3 67.5</td><td>51.0 51.2</td><td>61.4 53.0</td></tr></table>

Table 4: We present the NLU evaluation results based on the BabyLM Challenge, which consists of tasks selected from (Super)GLUE. We fine-tune the final models developed on both BookCorpus and BabyLM datasets.  
![](images/20eefc6e2c91ce8d6d394a274b6cd6a577b94bbc572fb1ce13553dfe508ad27f.jpg)  
(a) Overall surprisal v.s. training step on BabyLM.

![](images/84dacd6bfee5192459927ea91c35bac32d6735e8ac2eedbe8186f7de8831dcc6.jpg)  
(b) Surprisal v.s. training step on word “fine”.  
Figure 11: Comparison between CLM on model predicted text, CLM on train corpus, and TnD. Evaluated on CDI words on BabyLM.

Using the corpus sentences as demonstrations. While it is possible to directly use the ground truth sentences from the training corpus as demonstrations (ter Hoeve et al., 2022), it can be very difficult to adapt the teacher model's behaviors for our controlled studies. As a result, we use a pre-trained language model as the proxy for demonstration generation, rather than using the original text. In our preliminary experiments, using ground truth sentences from the training corpus as demonstrations do not lead to a noticeable difference from model-generated demonstrations.

## B.6 The robustness of results and findings over hyper-parameters

Learning rate To evaluate the robustness of results over different PPO learning rates, we run our experiment with learning rate =

![](images/d3e7e1d9067c9b012093eb574630174f1af1d659a27c6268e58625136c56b958.jpg)

![](images/9573d8e069250bf57725c87832b7013fa9c48006fc3c39e29134c0d60ef51afd.jpg)  
(a) Effect of PPO learn- (b) Effect of alternating freing rate on CDI words on quency on CDI words on BabyLM. BabyLM.  
Figure 12: The robustness of the learning curve results over other hyper-parameters, e.g., learning rates and alternating schedules.

$8 e ^ { - 6 } , 1 e ^ { - 5 } , 2 e ^ { - 5 } , 3 e ^ { - 5 }$ (Figure 12a). We find that a smaller learning rate results in a later age of acquisition compared to a higher learning rate. Nevertheless, our findings remain robust across different learning rates, although a higher learning rate can lead to unstable training and poorer end performance.

Alternating frequency We run our experiment on different alternating frequencies to study its impact on the TnD framework. To perform a systematic study, we group the alternating schedule [c, r] by their PPO/CLM steps ratio. We experiment with the settings when the ratio equals to 1 ([1, 1], [2, 2]), 2 ([2, 1], [4, 2]), 3 ([3, 1], [6, 2]), and 4 ([4, 1]), as presented in Figure 12b. We find that the alternating frequencies under the same ratio lead to similar performance. Our findings are robust over different alternating frequencies, except the ratio = 1 leading to unstable training and poor end performance.

## C Limitations, Licenses, and Risks

## C.1 Artifacts and licenses

Our work largely relies on publicly available datasets such as BookCorpus (Zhu et al., 2015) and BabyLM (Warstadt et al., 2023), and pre-trained models such as GPT-2 (Radford et al., 2019) and LLaMA-2 (Touvron et al., 2023). We strictly follow the LLaMA license and limit the scope of the LLaMA model to academic research only. We report a list of licenses for all datasets and models

<table><tr><td>Dataset</td><td>URL License</td></tr><tr><td>BabyLM</td><td>BookCorpus Link MIT License Link MIT License</td></tr><tr><td>Models</td><td>URL License</td></tr><tr><td>GPT-2</td><td>Link MIT License</td></tr><tr><td>LLaMA-2</td><td>Link Llama License</td></tr></table>

Table 5: License information for the code base in our experiment.  
used in our experiment in Table 5.

## C.2 Ethical concerns and risks

This work does not depend on human annotators or human subjects for interactive experiments. We leverage open datasets and model-generated content for training that could contain biases and sensitive contents inherited, which may cause fairness issues in the final model when applied to practical applications. Future research should be done to look into these issues, potentially by designing fairness-aware reward models.

![](images/0cfc7fda54ffe1f2b174f551805862a9c90a4bdae45c53d623d99f0c0121aedb.jpg)  
Figure 13: The ridgeline and scatter plot of words and their neural age of acquisition (nAoA) in BabyLM Corpus.

![](images/19fb818ee83ca7e66d450f854aac968ec10555e6e6dda939891462d3ce13bf76.jpg)  
Figure 14: The ridgeline and scatter plot of words and their neural age of acquisition (nAoA) in BookCorpus.