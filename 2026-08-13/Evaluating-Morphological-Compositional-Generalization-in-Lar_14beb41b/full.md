# Evaluating Morphological Compositional Generalization in Large Language Models

Mete Ismayilzada<sup>1,2</sup>, Defne Circi<sup>\*3</sup>, Jonne Sälevä<sup>\*4</sup>, Hale Sirin<sup>7</sup>, Abdullatif Köksal<sup>5,6</sup>, Bhuwan Dhingra<sup>3</sup>, Antoine Bosselut<sup>1</sup>, Duygu Ataman<sup>†9</sup>, Lonneke van der Plas <sup>†2,8</sup>

<sup>1</sup>EPFL, <sup>2</sup>Idiap Research Institute, <sup>3</sup>Duke University, <sup>4</sup>Brandeis University, <sup>5</sup>LMU Munich, <sup>6</sup>University of Cambridge, <sup>7</sup>Johns Hopkins University, <sup>8</sup>Università della Svizzera Italiana, <sup>9</sup>New York University

mete.ismayilzada@idiap.ch

## Abstract

Large language models (LLMs) have demonstrated significant progress in various natural language generation and understanding tasks. However, their linguistic generalization capabilities remain questionable, raising doubts about whether these models learn language similarly to humans. While humans exhibit compositional generalization and linguistic creativity in language use, the extent to which LLMs replicate these abilities, particularly in morphology, is under-explored. In this work, we systematically investigate the morphological generalization abilities of LLMs through the lens of compositionality. We define morphemes as compositional primitives and design a novel suite of generative and discriminative tasks to assess morphological productivity and systematicity. Focusing on agglutinative languages such as Turkish and Finnish, we evaluate several state-of-the-art instruction-finetuned multilingual models, including GPT-4 and Gemini. Our analysis shows that LLMs struggle with morphological compositional generalization particularly when applied to novel word roots, with performance declining sharply as morphological complexity increases. While models can identify individual morphological combinations better than chance, their performance lacks systematicity, leading to significant accuracy gaps compared to humans.

## 1 Introduction

Large language models (LLMs) have recently achieved remarkable advances in the broad domain of natural language generation and understanding tasks (Gemini, 2024; Zhao et al., 2023; Bubeck et al., 2023; Wei et al., 2022; Brown et al., 2020). However, these models have also been shown to lack strong linguistic generalization capabilities (Weissweiler et al., 2023; McCoy et al.,

![](images/7d0307f4ff0f38a20425bbc8bd4cb6c9fdeef98bb8fe3cf9945fdb62bea44974.jpg)  
Figure 1: Our morphological generalization tasks illustrated with an example in Turkish. ID and OOD refer to in-distribution and out-of-distribution respectively. English translations are not part of the task and only shown here for illustrative purposes.

2023; Goldman et al., 2022; Wilson et al., 2023; Linzen, 2020; Baroni, 2019). This discrepancy casts doubt on whether language models learn a language the same way as humans do. When learning a language which is essentially a finite set of words and rules, humans exhibit linguistic creativity (Chomsky, 1965; Bergs, 2019) and compositional generalization through productivity and systematicity (Fodor and Pylyshyn, 1988a; Chomsky, 1957). These abilities allow humans respectively to produce and understand novel combinations of familiar grammar units. While compositional generalization abilities of language models have been extensively studied (Lake and Baroni, 2018; Keysers et al., 2019; Kim and Linzen, 2020), the extent to which language models employ this ability in morphology however remains largely under-explored. Recent works evaluating morphological generalization in language models have only focused on the productivity aspect with a limited coverage of inflectional forms (Weissweiler et al., 2023; Anh et al., 2024).

In this work, we address this gap by systematically investigating the morphological generalization abilities of LLMs through the lens of compositionality. Following Keysers et al. (2019), we define the morphemes (smallest meaningful units in a language)<sup>1</sup> as the compositional primitives and design a novel suite of generative and discriminative language tasks based on the morphological combinations of these primitives. These tasks aim to test morphological productivity (ability to produce novel well-formed combinations of morphemes) and systematicity (ability to systematically understand novel combinations) respectively. Figure 1 illustrates an example of both tasks.

We evaluate several state-of-the-art instructionfinetuned large multilingual models on these tasks: GPT-4, Gemini-1.5, Aya-23 and Qwen-2.5. To ensure our findings are not language-specific, we experiment with two morphologically rich (i.e. characterized by a large number of inflectional and derivational forms) languages: Turkish and Finnish. Both languages share typological features (e.g. agglutination) despite being unrelated.

We find that LLMs lack human-like morphological compositional generalization ability in agglutinative languages despite their high performance on various tasks in these languages. Our analysis shows that morphological productivity, especially when applied to novel word roots is highly challenging for LLMs. Moreover, as the morphological complexity of words increases the model performance sharply decreases (to nearly zero) while human performance is not consistently affected. On the systematicity task, while models perform much better than chance in identifying the validity of individual morphological combinations, however, this behaviour is not robust or systematic i.e. models fail to consistently determine validity of several compositions made up of the same set of morphemes.

In summary, our contributions are as follows: 1) We design novel morphological generalization tasks that require compositional processing. 2) We prepare specific test suites in both Turkish and Finnish to measure morphological generalization and make these available for future research<sup>2</sup>. 3)

Using our novel tasks and test suites, we conduct a systematic analysis of morphological compositional generalization abilities of LLMs. 4) Our findings reveal a systematic gap in LLM’s ability compared to humans concerning morphological generalization in agglutinative languages that also requires compositionality.

## 2 Related Work

## 2.1 Compositional Generalization

Compositional generalization is the capacity to understand and produce novel compositions of seen primitives and is typically characterized by systematicity and productivity (Fodor and Pylyshyn, 1988a; Keysers et al., 2019). Systematicity refers to the ability to understand different combinations that are made up of the same known components such as John loves Mary and Mary loves John. Productivity, on the other hand, is the ability to produce potentially infinite novel combinations of a finite number of known building blocks such as using conjunctions to construct sentences Mary knows that John loves Mary and John heard that Mary knows that John loves Mary. Past work has developed several benchmarks to measure compositional generalization abilities of neural models both in fine-tuning and in-context learning settings and has shown this task to be highly challenging (Yang et al., 2024; Lake and Baroni, 2018; Keysers et al., 2019; Kim and Linzen, 2020; An et al., 2023; Dziri et al., 2023). These benchmarks have mainly focused on synthetic sequence matching, semantic parsing, question-answering and problem-solving tasks. Our work however, investigates compositional generalization in the context of morphology.

## 2.2 Morphological Generalization

Morphological generalization is the ability to understand words based on their constituent parts known as morphemes and combine them to derive new words (Wysocki and Jenkins, 1987). Morphemes are the smallest meaningful units of language that typically correspond to word roots and affixes (i.e. prefixes, infixes and suffixes). Composing these units to construct new words can be done through inflection and derivation tasks in morphology where derivation often changes the syntactic category of the words and inflection does not. These tasks have gained considerable attention as part of the SIGMORPHON’s shared tasks (Cotterell et al., 2016, 2018a; Vylomova et al., 2020;

Kodner and Khalifa, 2022; Goldman et al., 2023) and efforts to create a universal morphology (Mc-Carthy et al., 2020). While transformer-based mod els have been shown to achieve near-perfect accuracy on these tasks (Canby et al., 2020), recent work has also found that these results are inflated due to lemma overlap pointing to a lack of generalization (Goldman et al., 2022). Other works have recently investigated the morphological capabilities of LLMs using inflection tasks and reported similarly weak performance results (Anh et al., 2024; Weissweiler et al., 2023). Similar to our study, both of these works use the popular Wug test (Berko Gleason, 1958) to evaluate the morpho logical generalization, however, they only focus on the productivity aspect, and their coverage of inflectional and derivational forms is limited. For example, Weissweiler et al. (2023) considers only a handful of specific inflectional forms (e.g. first person singular agreement and past tense, second person plural agreement etc.) for each language and Anh et al. (2024) translates the original Wug test suite which is very small in size (23 samples) into different languages. On the other hand, we cast the inflection and derivation tasks into the form of a compositional generalization task and evaluate models on both productivity and systematicity aspects. While focus of other works is breadth (languages from different families), we instead con duct an in-depth analysis of morphological generalization in typologically similar but unrelated languages with a large test suite covering a wide and diverse range of inflectional and derivational combinations.

## 3 Methodology

## 3.1 Background

The important role of compositional processing in language understanding and generation has been extensively studied (Carnap, 1947; Chomsky, 1965; Fodor and Pylyshyn, 1988b; Zadrozny, 1994; Bauer, 2001; Aronoff and Lindsay, 2014). Past works have shown that new word formation is often a multi-level process that requires identifying the correct order of primary and secondary morphemes (Kiparsky, 1982a,b; Hockett, 1954), and while humans might memorize some frequent words and phrases as a whole, most of the expressive language generation relies on productive rules of grammar (O’Donnell, 2015). However, not all languages are equally productive, and more productive languages (e.g. agglutinative) tend to have complex inflectional morphology (Cotterell et al., 2019; Ackerman and Malouf, 2013). Moreover, these languages have been shown to be harder to model for n-gram and recurrent language models (Cotterell et al., 2018b; Czarnowska et al., 2019). Inspired by these works, we focus our study on two highly agglutinative languages and compositional tasks which we describe in detail below.

## 3.2 Tasks

Similar to works studying compositional abilities of neural networks (Goodwin et al., 2020; Lake and Baroni, 2018; Keysers et al., 2019), we design two novel and simple compositional probing tasks to test morphological abilities of models. First, a morphological productivity task which we define as a generative task where the model is given a word root, a list of affixes (not necessarily in the correct order) and is asked to derive a meaningful word by composing the root with the affixes in the correct order. Second, a morphological systematicity task which we define as a binary discriminative task where the model is again given a word root, a list of affixes and a word derived from the root using the given affixes (not necessarily a meaningful word) and is asked to determine the grammatical validity of the derived word. Figure 1 illustrates these tasks with an example in Turkish.

Additionally, to measure the morphological generalization capabilities of LLMs, we take inspiration from Berko’s Wug test (Berko Gleason, 1958) that is typically used to probe the inflectional and derivational morphological knowledge of children and design out-of-distribution (OOD) versions of our tasks using nonce word roots. More specifically, for each in-distribution (ID) word root in our test suite, we automatically generate a nonce word (i.e. word that does not exist in the given language) and use it in both tasks as the word root in place of the original one. However, since the model has never seen these words, to make sure the model understands the meaning of this new word, we provide the model with the original word root as a definition of the novel word root. Our generation of nonce words relies on the underlying morphophonological features and the frequency of each letter in a given language to make sure these words are plausible and inflected in the same way as the original root. Further details on nonce word generation can be found in Appendix B.

## 3.3 Data

We focus our study on two highly agglutinative languages, Turkish and Finnish, and prepare test suites specific for our tasks in these languages. We particularly choose these languages because they are characterized by a large number of morphemes and hence require a high degree of compositional generalization ability.

Turkish Turkic languages are well-known to be highly agglutinative where the word is composed of several morphemes in addition to a root. We select Turkish as a representative of this language family in our study. To prepare our test suite we use the Bilkent Turkish Writings Dataset<sup>3</sup> as our base corpus which contains 6, 844 creative writings of Turkish 101 and Turkish 102 courses between 2014- 2018 and hence, is full of morphologically complex words. Data statistics can be found in Appendix Table 1. We preprocess this dataset to extract words and the sentences they are found in. Then we employ a morphological analyzer for Turkish (Ozturel et al., 2019) to segment these words into a root and surface-level morphemes. To create a diverse and balanced test suite, we sample 150 examples per morpheme length 1 to 7 while maximizing the number of unique roots and morphemes (in total 1, 049 samples). Finally, we automatically generate a nonce word for each word in our test suite by relying on the fact that surface realizations of morphemes in Turkish are characterized by deterministic morphophonological processes such as vowel harmony, consonant assimilation and elision. Final data statistics and examples can be found in Appendix Tables 2 and 5 respectively. Further details on data collection can be found in Appendix D.

Finnish We first collect a 1,000,000 sentence subsample of the Finnish mC4 corpus (Xue et al., 2021). We then extract unique words from the text and morphologically segment them using omorfi (Pirinen, 2015) and UralicNLP (Hämäläinen, 2019). After excluding words that analyzers did not cover, we manually annotate the segmentations to identify prefixes, lemmas, and affixes among the segments. We then perform stratified sampling based on the number of affixes to ensure an even range of morphological complexity in our data set. Finally, we extract sentences corresponding to each analyzed word from mC4 and validate whether they make sense. In a significant portion of cases, we notice that the raw sentences are noisy; in these cases, we opt to generate synthetic sentences using ChatGPT, which we (authors) then manually validate to be grammatical. Final data statistics and examples can be found in Appendix Tables 6 and 7.

## 4 Experiments

Setup We treat the productivity task as an openended task in which the model is asked to derive a word from the given root and affixes and the systematicity task as a binary classification task in which the model is asked to determine whether the given derivation is grammatically correct. For the systematicity task, we generate negative examples by producing all the combinations of morphemes attached to the same root and choosing the top four compositions (two for morpheme lengths of 1 and 2)<sup>4</sup> that are closest to the original valid combination measured by the Levenshtein distance. We do this to ensure our incorrect combinations are challenging enough for the model as they will be deceptively close to a plausible derivation. We also experiment with other negative example selection strategies such as random selection and a heuristic selection based on the linguistic characteristics of the given language. We describe these settings in more detail and compare the results in Section 5.5. Finally, we (authors) manually verify all the generated negative examples and fix the label of false negatives.

Models We evaluate several state-of-the-art multilingual instruction-finetuned LLMs, namely, two open-weights models, Aya-23 (Aryabumi et al., 2024) and Qwen-2.5 (Team, 2024), and two closedsource models, Gemini-1.5 (Gemini, 2024) and GPT-4 (OpenAI, 2024). We evaluate all models on all languages except for Aya-23 which officially supports Turkish, but not Finnish.<sup>5</sup>. We also report the performance of a random baseline that generates a derivation with a random combination of given morphemes (productivity task) and randomly decides whether the derivation is grammatically correct, and a majority baseline, which selects the most frequent label (in our case "No") for the systematicity task (not applicable for the productivity task). All models are evaluated using few-shot (1, 3, and 5) in-context learning and greedy decoding since our tasks are deterministic by nature<sup>6</sup>. Unless otherwise specified, prompt instructions are in English, and number of shots is set to 5 for reported results<sup>7</sup>. Further details on model evaluation can be found in Appendix C.

![](images/b335df0994fb5bc45889e5048ab5c6fad6e1005a2e19d0ba8dcc53bcfe84147f.jpg)

![](images/a3cbcd9cc73c9f776493b82a88c061d4842d3bbdf5331de8aba62e547b88a4b7.jpg)

![](images/fec53f4da57d2b24cae36520c41fd28774141a00f2907afa5ff92c879b845959.jpg)  
Figure 2: Morphological productivity and systematicity task results for Turkish. Detailed results for all shots are in Appendix Table 8.

![](images/7756d9f017cbfd3af6a54ce7435e39299fbe94e0c4c3b8b2dff0bb2fb705e8d3.jpg)

![](images/4eeebef376cefcf6b69e4ffad20a2f2803f7c33dbcbdf45415fb8dd4101083e7.jpg)

![](images/4946d211604dbeecec58a8d70d197d551ad3cebe16d4177093f31c254f44df15.jpg)  
Figure 3: Morphological productivity and systematicity task results for Finnish. Detailed results for all shots are in Appendix Table 9.

Evaluation Metrics For the productivity task, we use Exact Match accuracy against the correct derivations. For the systematicity task, we report an average of Macro-F1 scores for each sample and a Coherence score that measures whether the model correctly and consistently identifies the validity (or invalidity) of all derivations for a given set of morphemes. Hence, coherence is defined as a binary score where the model gets a score of 1 for a given sample if and only if it correctly guesses the validity of all derivations pertinent to that sample, otherwise 0. We employ this stringent metric to test the robustness of model performance similar to (Storks and Chai, 2021).

Human Evaluation We evaluate human performance on both tasks using two native speakers<sup>8</sup> per language, who annotate 70 and 60 samples from the Turkish and Finnish test suites, respectively. To ensure our evaluation sample is a representative sample of the entire test suite, we randomly select 10 examples per morpheme length for each test distribution. Human annotators follow the same task instructions used for model prompts and were shown five examples. We report almost perfect or substantial inter-annotator agreement measured by Cohen’s kappa score (Cohen, 1960) for both tasks, languages, and test distributions (Appendix Tables 3, 4). Finally, for each task metric, we report the average score of annotators as the final human score.

Results Figure 2 and 3 summarize all model results for both morphological productivity and systematicity tasks evaluated respectively on the Turkish and Finnish data. We see that on the productivity task, all models except GPT-4 barely crack the random performance. While GPT-4 performs the best for both languages, it significantly lags behind the human performance ( 43% and 51% in Turkish and 40.8% and 48.9% for Finnish respectively for ID and OOD data). Moreover, the GPT-4 performance gap between the ID and OOD test suites for both languages is much larger than the human gap ( 10% vs. 3% in Turkish and 1.7% in Finnish). These results indicate that humans are much more compositionally productive in morphology and generalize more robustly to novel unseen words.

From the systematicity task results, we see that models perform much better than random and majority baselines with GPT-4 again in lead, however, the performance gap compared to humans is still significant, especially, on robustness as measured by coherence score ( 19.1% and 46.5% in Turkish and 8.8% and 25.2% in Finnish respectively for ID and OOD data). The ID and OOD performance gap is also significant for all models, especially when measured by coherence score (ranging from 9.5% to 23.3% in Macro-F1 and from 13.5% to 37.2% in Coherence) while this gap is very low ( 2%) for humans when measured by both metrics. These results show that humans are much more compositionally systematic and consistent in discriminating between correct and incorrect morphological forms made up of the same set of morphemes.

## 5 Analysis

## 5.1 Effect of Morphological Complexity

Recent works have shown that morphological complexity plays a crucial role in the morphological generalization abilities of LLMs (Anh et al., 2024; Czarnowska et al., 2019; Cotterell et al., 2018b). Morphological complexity is typically categorized into integrative (I-complexity) which refers to the predictability of inflected form and enumerative (Ecomplexity) complexity which refers to the number of cases and inflectional paradigms in language grammar (Ackerman and Malouf, 2013). While both languages we study are morphologically complex, our test suites include inflectional and derivational forms of varying length in the number of morphemes (1-7 in Turkish and 1-6 in Finnish). This allows us to study the effect of within-language E-complexity on the performance of our models. Figure 4 summarizes the GPT-4 performance for both tasks stratified by the number of bound morphemes on the Turkish data. On the productivity task, we observe a sharp downward trend (plummeting to nearly zero) in performance as the number of morphemes increases for both ID and OOD test suites with a relatively constant gap between ID and OOD performance while humans exhibit no such dependence on complexity (Appendix Tables 16, 25). This shows that humans learn their native language robustly and can easily produce and identify long novel words while models are quite sensitive to the morphological (E-) complexity.

On the systematicity task, Macro-F1 scores for ID and OOD remain mostly unchanged as complexity increases, but coherence scores show a negative correlation with the increasing morphological complexity. We also observe a surprisingly low performance on 1-morpheme OOD words which we attribute to the varying number of negative options by morpheme length and potential shortcuts in longer morpheme words, as discussed in Appendix A.3.

## 5.2 Effect of Context

While our core tasks are somewhat synthetic in nature, we do also experiment with more realistic versions where we provide the model a sentence as an additional context. Specifically, we frame them as sentence completion tasks where a sentence with a blank is provided and the model is asked to fill in the blank with the correct word derived from the given word root and affixes (productivity task) or determine if the given derivation is the correct option for the blank (systematicity task).

Figure 5 summarizes the results for both productivity and systematicity tasks evaluated on the Turkish data where we provide a sentence with a blank to the model as a context (i.e. sentence completion task). This results in some improvement on the productivity task, however, we observe significant decrease in performance on the systematicity task especially for smaller models such as Aya-23 and Qwen-2.5 series and in OOD setting. This could be due to the additional complexity introduced by the extra context, however, we should note that worse performance on this task implies even stronger generalization failure since this task is more real-world and closer to the next word prediction task compared to the original context-free setup.

## 5.3 Effect of Tokenization

Past work has shown that suboptimal tokenizers, especially byte-pair encoding (Sennrich et al., 2016) used in GPT-4 have generally a negative effect on the morphological abilities of language models (Meyer and Buys, 2023; Bostrom and Durrett, 2020; Hofmann et al., 2021). Whether the low performance of the model on the productivity task can be attributed to the suboptimal nature of the tokenization is of interest in particular because our tasks rely on the morphologically segmented morphemes while the model utilizes byte-level tokens that are mostly English. To measure the effect of the tokenization, we ran a version of the productivity task where the morphemes provided to the model are obtained by segmenting the final derivation based on the model’s own tokenizer instead of the morphologically-aligned units. Figure 6 compares the performance of the tokenizer-aligned morphemes with the morphologically-aligned morphemes on the ID test set.<sup>9</sup> We see that the performance in both cases is very similar to each other which points to a possibility that tokenization may not be the underlying issue behind the low performance. This finding is also consistent with some past work on exploring morphological capabilities of ChatGPT (Weissweiler et al., 2023)<sup>10</sup>

![](images/31e0c5905a23235cb5696c115d41629529daac618ea660940e7d508a89c8e765.jpg)

![](images/8ee3d43d8120e627bea92313f37cc1ef3827cb9f754f8a0b03de1a8c8ea3eb56.jpg)

![](images/320c6d91babcc91ec9d1bfc0adb67c921038cc562400c8382c6d41946a10bedb.jpg)  
Figure 4: GPT-4 morphological productivity and systematicity task results for Turkish stratified by number of bound morphemes. Detailed results are in Appendix Tables 16, 17, 18. Finnish results are in Appendix Figure 11.

![](images/9d61353e9cae7e558c5d797a2d493532e57ab8bfe33e31c3e2d6d093bd743207.jpg)

![](images/fa70a0aa4f08b5c470bf21513acf4e441fb26de1eed1d2ee0e6ddd5b2238bc6b.jpg)

![](images/22486148f03a1755f62a3cad8798e96a05c935897eddf22003c1b2f20fb4eb2b.jpg)  
Figure 5: Morphological productivity and systematicity task results for Turkish showing the effect of additional context. Detailed results are in Appendix Table 36. Results for Finnish are in Appendix Figure 13.

Morphological Productivity Turkish

![](images/f12ed7738a277445e95ffb6871c770964c4e33dd1885688720a4e0c7e0c38e00.jpg)  
Figure 6: GPT-4 productivity task results on the ID test suite for Turkish stratified by number of bound morphemes showing the effect of tokenization. Detailed results are in Appendix Table 42.

## 5.4 Effect of Morpheme Order

Since our goal is to study the ability of LLMs to combine the morphological units in the correct order, in all of our experiments we shuffle the order of the units in the prompts. However, given that models are sensitive to small prompt changes (Pezeshkpour and Hruschka, 2023; Zhu et al., 2024; Wang et al., 2023; Zhao et al., 2021), we also analyze the effect of changing the morpheme order on the performance of the model. To this end, we run our main experiments with all the morphemes in their correct order and report the results in Figure 7. We can see that this small change improves the performance across both tasks and models and especially, in the productivity task, the improvement can be up to 20%. This shows that models understand the tasks and can provide a correct answer by simply copying the morphemes when they are given in the correct order, however, they struggle to compose the correct order themselves. This further indicates that LLMs lack the necessary robust compositional generalization in morphology.

![](images/23e055aa0657ac1e3aaac86d7307c7294d123b9bdd8a9879b1585d2e152e4568.jpg)

![](images/656fb2aaf40c2796bb82dc53b62f0626d6c3cef801dab608d7c431e9e0ddafaf.jpg)

![](images/51fac64bd35041d8ce50cf2fa0fd700230cd6360a4d139097893976ef25aebf4.jpg)  
Figure 7: Morphological productivity and systematicity task results for Turkish showing the effect of the morpheme order. Detailed results are in Appendix Table 46. Results for Finnish are in Appendix Figure 14.

## 5.5 Effect of Negative Sample Selection

In our systematicity task, we generate negative samples (i.e. derived combinations that are not grammatically correct) by permuting the order of morphemes attached to the root. While the number of permutations is manageable for 2 or 3 morphemes (e.g., 2!=2, 3!=6), it grows rapidly with more morphemes (e.g., 6!=720). Evaluating all permutations would be ideal for robust systematicity testing, but this is infeasible due to high computational costs. Instead, we can select a subset of reasonable size to be a representative sample of all possible negative options. However, the strategy for which samples and how many to select can be somewhat arbitrary. Therefore, we experiment with three different selection strategies, and set the number of selections to four for simplicity: 1) random where we randomly select four negative options; 2) language-agnostic heuristic where we select the top four negative options that are closest to the positive option measured by Levenshtein distance (our default strategy); and 3) language-specific heuristic where we employ linguistic features of the tested language to filter out options that may be "too easy" for the model. We found one such heuristic for Turkish test suite based on the fact that Turkish phonology does not allow two adjacent vowels in morpheme combinations which we describe in Appendix E. We report the results of these different negative sample selection experiments in Figure 8. We see that the random selection has the highest performance on both ID and OOD test sets, followed by the language-agnostic and language-specific strategies. This implies that all our previous model results might be an upper bound and the true performance gap compared to humans is even larger than what we observe.

![](images/fd56a84be313afeb21239bdf227154b57a0b3fa8dc59ccda8e10fe3c7df8b0ad.jpg)

![](images/2eb2d6e0f4751a23bd927d598e140d723b726152b821582e8b08d6d33f7a19d4.jpg)  
Figure 8: Morphological systematicity task results for Turkish showing the effect of different negative sample selection strategies. Detailed results are in Appendix Table 52.

## 5.6 Error Analysis

In order to understand the limitations of language models on our tasks, we manually analyze 30 Turkish word derivations for each morpheme combina tion length (1-7) and for both productivity ID and OOD test sets resulting in a total of 178 and 185 derivations from GPT-4 that are incorrect. We annotate each generation on three criteria: 1) whether the generation is an invalid word (i.e. grammati cally incorrect word) 2) whether the generation is unfaithful (i.e. generation does not follow the productivity task constraints) and 3) whether the gener ation includes any hallucinations (i.e. whether the generation has extra morphemes not mentioned in the task prompt). Our analysis shows that while on the OOD test set, GPT-4 generates a grammatically incorrect word most of the time (79%), this proportion is significantly lower for the ID test set (31%). However, on the ID test set, we observe a high unfaithfulness and hallucination ratio (91% and 67%) meaning that most of the valid generations do not follow the task constraints. On the other hand, we see lower unfaithfulness and hallucination ra tios on the OOD test (75% and 52% respectively) which points to a real word bias also reported by (Weissweiler et al., 2023) where the model is biased toward generating frequent words for word roots existing in a given language irrespective of the underlying task. In other words, OOD setting forces the model to perform the true morphological generalization task which it fails as indicated by the higher percentage of invalid derivations. To identify the root causes of some of these errors, we analyze the GPT-4 chain-of-thought answers on the Turkish data and reveal several failure modes such as sequential dependency errors, semantic misinterpretations, lack of grammatical knowl edge, and unfaithful reasoning, all of which we detail with examples in Appendix A.4. Finally, we also analyze the few errors human annotators made and find that these errors are either trivial typos or failure to notice an extra letter in a long word.

## 6 Conclusion

In this paper, we proposed a novel experimental paradigm to test morphological generalization abilities of large language models through compositionality. Our tasks target measuring morphological productivity and systematicity in a given language. We applied these tasks on the morphologically complex languages of Turkish and Finnish and evaluated morphological compositional generalization abilities of several state-of-the-art large language models. Our experimental results and analysis reveal a significant gap in the performance of LLMs compared to humans with respect to generalization in morphology of agglutinative languages.

## Limitations

While our novel tasks are language, dataset, and model-independent, our study only focused on two agglutinative languages and a few large language models. Therefore, the applicability of our findings in other languages and models should be further studied. We also mainly focused on the grammatical validity of the words, whereas it would be equally interesting to study the capacity of LLMs to produce and understand novel semantically and pragmatically valid derivations. While we have also optimized our prompts to be as simple and maximally instructive and tested in multiple languages and in chain-of-thought setting, whether a different set of prompts would produce the same results is not clear. Finally, we mainly evaluate models using greedy decoding due to the deterministic nature of our tasks and additionally only experiment with temperature and top-p sampling, however, the effect of different decoding strategies needs to be explored.

## Acknowledgments

We gratefully acknowledge the support of the Swiss National Science Foundation (grant 205121\_207437: C - LING) and the Microsoft Accelerating Foundation Models Research Program. Defne would also like to acknowledge the support of the National Science Foundation under grant DGE-2022040. We also thank Mammad Hajili, Osman Batur Ince, Omer Goldman, members of the NLP Lab at EPFL and the CCL Group at Idiap Research Institute for their valuable feedback in the early stages of this project and Raghav Mantri for his help with the Gemini experiments.

## References

Farrell Ackerman and Robert Malouf. 2013. Morphological organization: The low conditional entropy conjecture. Language, 89:429–464.

Shengnan An, Zeqi Lin, Qiang Fu, Bei Chen, Nanning Zheng, Jian-Guang Lou, and Dongmei Zhang. 2023. How do in-context examples affect compositional generalization? In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 11027– 11052, Toronto, Canada. Association for Computational Linguistics.

Dang Anh, Limor Raviv, and Lukas Galke. 2024. Morphology matters: Probing the cross-linguistic morphological generalization abilities of large language models through a wug test. In Proceedings of the Workshop on Cognitive Modeling and Computational Linguistics, pages 177–188, Bangkok, Thailand. Association for Computational Linguistics.

Mark Aronoff and Mark Lindsay. 2014. 67productivity, blocking, and lexicalization. In The Oxford Handbook of Derivational Morphology. Oxford University Press.

Viraat Aryabumi, John Dang, Dwarak Talupuru, Saurabh Dash, David Cairuz, Hangyu Lin, Bharat Venkitesh, Madeline Smith, Jon Ander Campos, Yi Chern Tan, Kelly Marchisio, Max Bartolo, Sebastian Ruder, Acyr Locatelli, Julia Kreutzer, Nick Frosst, Aidan Gomez, Phil Blunsom, Marzieh Fadaee, Ahmet Üstün, and Sara Hooker. 2024. Aya 23: Open weight releases to further multilingual progress. Preprint, arXiv:2405.15032.

Marco Baroni. 2019. Linguistic generalization and compositionality in modern artificial neural networks. Philosophical Transactions of the Royal Society B, 375.

Laurie Bauer. 2001. Morphological Productivity. Cambridge Studies in Linguistics. Cambridge University Press.

Alexander Bergs. 2019. What, if anything, is linguistic creativity? Gestalt Theory, 41:173–183.

Jean Berko Gleason. 1958. The child’s learning of english morphology. Word, 14.

Kaj Bostrom and Greg Durrett. 2020. Byte pair encoding is suboptimal for language model pretraining. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 4617–4624, Online. Association for Computational Linguistics.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam Mc-Candlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. Preprint, arXiv:2005.14165.

Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, Harsha Nori, Hamid Palangi, Marco Tulio Ribeiro, and Yi Zhang. 2023. Sparks of artificial general intelligence: Early experiments with gpt-4. Preprint, arXiv:2303.12712.

Marc E. Canby, Aidana Karipbayeva, Bryan Lunt, Sahand Mozaffari, Charlotte Yoder, and J. Hockenmaier. 2020. University of illinois submission to the sigmorphon 2020 shared task 0: Typologically diverse morphological inflection. In Special Interest Group on Computational Morphology and Phonology Workshop.

Rudolf Carnap. 1947. Meaning and necessity: A study in semantics and modal logic, volume 30. University of Chicago Press.

N. Chomsky. 1957. Syntactic Structures. Janua linguarum (Mouton, Paris).: Series Minor. Mouton.

Noam Chomsky. 1965. Aspects of the Theory of Syntax, 50 edition. The MIT Press.

Jacob Cohen. 1960. A coefficient of agreement for nominal scales. Educational and Psychological Measurement, 20(1):37–46.

Ryan Cotterell, Christo Kirov, Mans Hulden, and Jason Eisner. 2019. On the complexity and typology of inflectional morphological systems. Transactions of the Associationfor Computational Linguistics, 7:327– 342.

Ryan Cotterell, Christo Kirov, John Sylak-Glassman, Géraldine Walther, Ekaterina Vylomova, Arya D. Mc-Carthy, Katharina Kann, Sabrina J. Mielke, Garrett Nicolai, Miikka Silfverberg, David Yarowsky, Jason Eisner, and Mans Hulden. 2018a. The CoNLL– SIGMORPHON 2018 shared task: Universal morphological reinflection. In Proceedings of the CoNLL–SIGMORPHON 2018 Shared Task: Universal Morphological Reinflection, pages 1–27, Brussels. Association for Computational Linguistics.

Ryan Cotterell, Christo Kirov, John Sylak-Glassman, David Yarowsky, Jason Eisner, and Mans Hulden. 2016. The sigmorphon 2016 shared task—morphological reinflection. In Special Interest Group on Computational Morphology and Phonology Workshop.

Ryan Cotterell, Sabrina J. Mielke, Jason Eisner, and Brian Roark. 2018b. Are all languages equally hard to language-model? In Proceedings of the 2018 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers), pages 536–541, New Orleans, Louisiana. Association for Computational Linguistics.

Paula Czarnowska, Sebastian Ruder, Edouard Grave, Ryan Cotterell, and Ann Copestake. 2019. Don’t forget the long tail! a comprehensive analysis of morphological generalization in bilingual lexicon induction. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 974–983, Hong Kong, China. Association for Computational Linguistics.

Nouha Dziri, Ximing Lu, Melanie Sclar, Xiang Lorraine Li, Liwei Jian, Bill Yuchen Lin, Peter West, Chandra Bhagavatula, Ronan Le Bras, Jena D. Hwang, Soumya Sanyal, Sean Welleck, Xiang Ren, Allyson Ettinger, Zaïd Harchaoui, and Yejin Choi. 2023. Faith and fate: Limits of transformers on compositionality. ArXiv, abs/2305.18654.

Jerry A. Fodor and Zenon W. Pylyshyn. 1988a. Connectionism and cognitive architecture: A critical analysis. Cognition, 28(1):3–71.

Jerry A Fodor and Zenon W Pylyshyn. 1988b. Connectionism and cognitive architecture: A critical analysis. Cognition, 28(1-2):3–71.

Team Gemini. 2024. Gemini: A family of highly capable multimodal models. Preprint, arXiv:2312.11805.

Omer Goldman, Khuyagbaatar Batsuren, Salam Khalifa, Aryaman Arora, Garrett Nicolai, Reut Tsarfaty, and Ekaterina Vylomova. 2023. Sigmorphon–unimorph 2023 shared task 0: Typologically diverse morphological inflection. In Special Interest Group on Computational Morphology and Phonology Workshop.

Omer Goldman, David Guriel, and Reut Tsarfaty. 2022. (un)solving morphological inflection: Lemma overlap artificially inflates models’ performance. In Proceedings ofthe 60th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 2: Short Papers), pages 864–870, Dublin, Ireland. Association for Computational Linguistics.

Emily Goodwin, Koustuv Sinha, and Timothy J. O’Donnell. 2020. Probing linguistic systematicity. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 1958– 1969, Online. Association for Computational Linguistics.

Mika Hämäläinen. 2019. Uralicnlp: An nlp library for uralic languages. Journal of Open Source Software, 4(37):1345.

Charles F Hockett. 1954. Two models of grammatical description. Word, 10(2-3):210–234.

Valentin Hofmann, Janet Pierrehumbert, and Hinrich Schütze. 2021. Superbizarre is not superb: Derivational morphology improves BERT’s interpretation of complex words. In Proceedings ofthe 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 3594–3608, Online. Association for Computational Linguistics.

Daniel Keysers, Nathanael Schärli, Nathan Scales, Hylke Buisman, Daniel Furrer, Sergii Kashubin, Nikola Momchev, Danila Sinopalnikov, Lukasz Stafiniak, Tibor Tihon, Dmitry Tsarkov, Xiao Wang, Marc van Zee, and Olivier Bousquet. 2019. Measuring compositional generalization: A comprehensive method on realistic data. ArXiv, abs/1912.09713.

Najoung Kim and Tal Linzen. 2020. COGS: A compositional generalization challenge based on semantic interpretation. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 9087–9105, Online. Association for Computational Linguistics.

Paul Kiparsky. 1982a. Lexical morphology and phonology. Linguistics in the Morning Calm/Hanshin.

Paul Kiparsky. 1982b. Word formation and the lexicon. In 1982 Mid America Linguistics Conference Papers/Dept. ofLing., Univ. ofKansas.

Jordan Kodner and Salam Khalifa. 2022. Sigmorphon–unimorph 2022 shared task 0: Modeling inflection in language acquisition. Proceedings ofthe 19th SIGMORPHON Workshop on Computational Research in Phonetics, Phonology, and Morphology.

Brenden M. Lake and Marco Baroni. 2018. Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks. Preprint, arXiv:1711.00350.

Jindˇrich Libovický, Helmut Schmid, and Alexander M. Fraser. 2021. Why don’t people use character-level machine translation? In Findings.

Tal Linzen. 2020. How can we accelerate progress towards human-like linguistic generalization? In Annual Meeting ofthe Associationfor Computational Linguistics.

Risto Luukkonen, Jonathan Burdge, Elaine Zosa, Aarne Talman, Ville Komulainen, Väinö Hatanpää, Peter Sarlin, and Sampo Pyysalo. 2024. Poro 34b and the blessing of multilinguality. Preprint, arXiv:2404.01856.

Arya D. McCarthy, Christo Kirov, Matteo Grella, Amrit Nidhi, Patrick Xia, Kyle Gorman, Ekaterina Vylomova, Sabrina J. Mielke, Garrett Nicolai, Miikka Silfverberg, Timofey Arkhangelskiy, Nataly Krizhanovsky, Andrew Krizhanovsky, Elena Klyachko, Alexey Sorokin, John Mansfield, Valts Erntreits, Yuval Pinter, Cassandra L. Jacobs, Ryan Cotterell, Mans Hulden, and David Yarowsky. 2020. Unimorph 3.0: Universal morphology. In International Conference on Language Resources and Evaluation.

R. Thomas McCoy, Paul Smolensky, Tal Linzen, Jianfeng Gao, and Asli Celikyilmaz. 2023. How much do language models copy from their training data? evaluating linguistic novelty in text generation using RAVEN. Transactions of the Association for Computational Linguistics, 11:652–670.

Francois Meyer and Jan Buys. 2023. Subword segmental machine translation: Unifying segmentation and target sentence generation. In Findings of the Association for Computational Linguistics: ACL 2023, pages 2795–2809, Toronto, Canada. Association for Computational Linguistics.

Timothy J O’Donnell. 2015. Productivity and reuse in language: A theory of linguistic computation and storage. MIT Press.

OpenAI. 2024. Gpt-4 technical report. Preprint, arXiv:2303.08774.

Adnan Ozturel, Tolga Kayadelen, and Demirsahin I. 2019. A syntactically expressive morphological analyzer for turkish. In Proceedings of the 14th International Conference on Finite-State Methods and Natural Language Processing, pages 65–75, Dresden, Germany. Association for Computational Linguistics.

Pouya Pezeshkpour and Estevam Hruschka. 2023. Large language models sensitivity to the order of options in multiple-choice questions. Preprint, arXiv:2308.11483.

Tommi A Pirinen. 2015. Development and use of computational morphology of finnish in the open source and open science era: Notes on experiences with omorfi development. SKY Journal of Linguistics, 28:381–393.

Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016. Neural machine translation of rare words with subword units. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1715–1725, Berlin, Germany. Association for Computational Linguistics.

Shane Storks and Joyce Chai. 2021. Beyond the tip of the iceberg: Assessing coherence of text classifiers. In Findings of the Association for Computational Linguistics: EMNLP 2021, pages 3169–3177, Punta Cana, Dominican Republic. Association for Computational Linguistics.

Qwen Team. 2024. Qwen2.5: A party of foundation models.

Cagri Toraman, Eyup Halit Yilmaz, Furkan ¸Sahinuç, and Oguzhan Ozcelik. 2022. Impact of tokenization on language models: An analysis for turkish. ACM Transactions on Asian and Low-Resource Language Information Processing, 22:1 – 21.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023. Llama: Open and efficient foundation language models. Preprint, arXiv:2302.13971.

Ekaterina Vylomova, Jennifer White, Elizabeth Salesky, Sabrina J. Mielke, Shijie Wu, Edoardo Maria Ponti, Rowan Hall Maudslay, Ran Zmigrod, Josef Valvoda, Svetlana Toldova, Francis Tyers, Elena Klyachko, Ilya Yegorov, Natalia Krizhanovsky, Paula Czarnowska, Irene Nikkarinen, Andrew Krizhanovsky, Tiago Pimentel, Lucas Torroba Hennigen, Christo Kirov, Garrett Nicolai, Adina Williams, Antonios Anastasopoulos, Hilaria Cruz, Eleanor

Chodroff, Ryan Cotterell, Miikka Silfverberg, and Mans Hulden. 2020. SIGMORPHON 2020 shared task 0: Typologically diverse morphological inflection. In Proceedings of the 17th SIGMORPHON Workshop on Computational Research in Phonetics, Phonology, and Morphology, pages 1–39, Online. Association for Computational Linguistics.

Jiongxiao Wang, Zichen Liu, Keun Hee Park, Zhuojun Jiang, Zhaoheng Zheng, Zhuofeng Wu, Muhao Chen, and Chaowei Xiao. 2023. Adversarial demonstration attacks on large language models. Preprint, arXiv:2305.14950.

Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed H. Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, and William Fedus. 2022. Emergent abilities of large language models. Preprint, arXiv:2206.07682.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. 2023. Chain-of-thought prompting elicits reasoning in large language models. Preprint, arXiv:2201.11903.

Leonie Weissweiler, Valentin Hofmann, Anjali Kantharuban, Anna Cai, Ritam Dutt, Amey Hengle, Anubha Kabra, Atharva Kulkarni, Abhishek Vijayakumar, Haofei Yu, Hinrich Schuetze, Kemal Oflazer, and David Mortensen. 2023. Counting the bugs in ChatGPT’s wugs: A multilingual investigation into the morphological capabilities of a large language model. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 6508–6524, Singapore. Association for Computational Linguistics.

Chris Wendler, Veniamin Veselovsky, Giovanni Monea, and Robert West. 2024. Do llamas work in English? on the latent language of multilingual transformers. In Proceedings of the 62nd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 15366–15394, Bangkok, Thailand. Association for Computational Linguistics.

Michael Wilson, Jackson Petty, and Robert Frank. 2023. How abstract is linguistic generalization in large language models? experiments with argument structure. Transactions of the Association for Computational Linguistics, 11:1377–1395.

Katherine Wysocki and Joseph R. Jenkins. 1987. Deriving word meanings through morphological generalization. Reading Research Quarterly, 22(1):66–81.

Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, and Colin Raffel. 2021. mT5: A massively multilingual pre-trained text-to-text transformer. In Proceedings ofthe 2021 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 483–498, Online. Association for Computational Linguistics.

Haoran Yang, Hongyuan Lu, Wai Lam, and Deng Cai. 2024. Exploring compositional generalization of large language models. In Proceedings ofthe 2024 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies (Volume 4: Student Research Workshop), pages 16–24.

Wlodek Zadrozny. 1994. From compositional to systematic semantics. Linguistics and philosophy, 17:329– 342.

Ruochen Zhang, Samuel Cahyawijaya, Jan Christian Blaise Cruz, Genta Winata, and Alham Aji. 2023. Multilingual large language models are not (yet) code-switchers. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 12567–12582, Singapore. Association for Computational Linguistics.

Tony Z. Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. 2021. Calibrate before use: Improving few-shot performance of language models. Preprint, arXiv:2102.09690.

Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, and Ji-Rong Wen. 2023. A survey of large language models. Preprint, arXiv:2303.18223.

Kaijie Zhu, Jindong Wang, Jiaheng Zhou, Zichen Wang, Hao Chen, Yidong Wang, Linyi Yang, Wei Ye, Yue Zhang, Neil Zhenqiang Gong, and Xing Xie. 2024. Promptrobust: Towards evaluating the robustness of large language models on adversarial prompts. Preprint, arXiv:2306.04528.

## A Additional Analysis

## A.1 Effect of Instruction Language

Since most LLMs are pre-trained on significantly more instruction data in English than other languages, we base most of our results on experiments where we use English as the prompt instruction language. However, as our data is in a different language, this results in a code-switched language which has been shown to be a challenge for large language models (Zhang et al., 2023). To measure the effect of the instruction language on the morphological generalization tasks, we run our experiments with Turkish and Finnish as the instruction language and report results for both tasks in Figure 9. We mostly observe a drop or no change in performance when the instruction language is other than English.

## A.2 Effect of Chain-of-thought Reasoning

Chain-of-thought prompting has been shown to be effective in eliciting strong reasoning capabilities from LLMs (Wei et al., 2023). In order to measure the effect of this reasoning technique on LLMs’ performance on our tasks, we evaluate GPT-4 (the best performing model) on both productivity and systematicity tasks in zero-shot and 5-shot chainof-thought settings. We report the results of these experiments compared with the 5-shot standard prompting in Figure 10. We observe that while 5-shot chain-of-thought performance is better than the zero-shot chain-of-thought, it is slightly worse than or similar to the 5-shot standard prompting. To identify the causes of these errors, we manually analyze the several chain-of-thought answers which we describe in Appendix A.4.

## A.3 Further details on the effect of morphological complexity

In Figure 4, we observe a surprisingly low performance ( 40% drop from ID performance) on the 1-morpheme OOD words, but we attribute this behaviour to the varying number of negative options available for each morpheme length and possible presence of shortcuts in larger morpheme words. We should note that we have different number of total options to discriminate for a given sample depending on the number of morphemes (for 1 and 2 morphemes, we have 2 options, for 3-7, we have 5 options). Hence, a single mistake is penalized more in the former case than in the latter. However, within the former category, we see a much higher performance for 2-morpheme examples which might seem surprising, however, we hypothesize that this could be due to the presence of potential shortcuts for the model to exploit in the 2-morpheme case. Indeed, if we analyze the proportion of errors in both cases, we find that in the 1-morpheme case, a significant portion of errors (64%) is false negative i.e. the model identifies a nonce root with a valid morpheme as grammatically incorrect, while this is not the case in the 2-morpheme case. However, in the 2-morpheme case, the model might be exploiting the correct order of morphemes as sole evidence for the validity of the derivation while in the 1-morpheme case, there is no such shortcut and the model should understand the applicability of the given morpheme to the given word root.

![](images/ff7baf1c00835af738505db5035abde3d6285f67aa34d28d78476ea3ce8f3a2c.jpg)

![](images/b2545273f56aead91bc732f123743e90d202b8590b8e8365d05b4745bab9b288.jpg)

![](images/eff000bd0ccf7a7c7c7981eeb92382e98bbbe3a28e180613dccbe7258c5c12fc.jpg)

Figure 9: Morphological productivity and systematicity task results for Turkish showing the effect of the instruction language. Detailed results are in Table 30. Results for Finnish can be found in Figure 12.  
![](images/ba0bd90dede78fef5d7cb0fb6bd352ec5de6e90efb78fba21b34e5fc500d536f.jpg)

![](images/7858c12f164bc08ba8ed6b12a90dd82706397d7291d82139e9d9ce2a18e65006.jpg)

![](images/e453fa13a72216933c263f9de59ef3eafab106925199b9b3600cf7b2b3d89de2.jpg)  
Figure 10: GPT-4 morphological productivity and systematicity task results for Turkish showing the effect of chain-of-thought reasoning. Detailed results are in Table 43.

## A.4 Chain-of-thought Error Analysis

We randomly sample 10 examples from the 5-shot chain-of-thought experiments on the Turkish evaluation data (per morpheme length and test distribution) where GPT-4 made an error and manually analyze its answers across both tasks. Our analysis reveals the following primary types of errors:

## 1. Sequential Dependency Errors

One common error we observe in the productivity task is due to the sequential processing of the given affixes by GPT-4. It typically starts applying the given affixes in the order they are given, however, since the affixes are typically given in shuffled order, this often results in an invalid word early on. The model, however, does not seem to realize its mistake and continues with the generation often confidently assigning meaning to the intermediate erroneous words. For example, given the word root "hedef" and affixes "-in", "-dig" and "-˘ le", it considers the affixes sequentially in this order by first producing "hedefin" which is valid, then "hedefindig" which is invalid, how-˘ ever, it interprets the generation as "which is the target" and finally produces "hedefindigle"˘ which it interprets as "with what is the target".

## 2. Semantic Misinterpretations

Another set of errors stems from GPT-4 misinterpreting the meaning of the individual morphemes or the whole derivation. For instance, in one example, where the given morphemes are "bagır" ("to shout"), "-sa" and "-k" and˘ GPT-4 is asked to determine the validity of the combination "bagırsak", it misinterprets˘ this derivation as meaning "intestine" (which is also written as "bagırsak") and argues that ˘ this derivation can not be made up of the given affixes. While this reasoning is correct, the model misses the other plausible meaning of this derivation ("as if we shout") that can be derived from the given morphemes. In another example, where the model is given the morphemes "oyna" and "-sana" and asked to produce a valid word, it misinterprets the meaning of the morpheme "-sana" as "to you" and argues that it can not be applied to the root "oyna" whereas "-sana" is a valid suffix added to verbs.

## 3. Lack of Grammatical Knowledge

Another common pattern we see can be attributed to the lack of proper grammatical knowledge. In one example, the model is given the morphemes "uyum", "-suz", "-luk" and "-ta" and asked to determine the validity of the derivation "uyumluktasuz" which is invalid, however, the model assesses the validity of each morpheme and concludes that the combination should also be valid. In another example, it tries to add a verb suffix to a noun ("yargıyoruz"). Yet in other examples, it argues that valid affixes do not exist in the language or a valid morphological combination is not possible.

## 4. Unfaithful Reasoning

Finally, we also observe a large set of reasoning errors due to inconsistent reasoning chains, hallucinations or unfaithful instruction following. For instance, in one example, GPT-4 concatenates the morphemes "unut" and "-alı" and yet derives "unutuluyor". In another example, it auto-corrects an invalid word ("kaldırınızdıgda") to a valid word˘ "kaldırdıgınızda" and argues that the original˘ derivation is correct.

## A.5 Effect of Decoding Strategies

We mainly experiment with greedy decoding (e.g. temperature is set to 0 and top\_p is set to 1) in all of our experiments as the nature of our tasks is deterministic. However, to check the sensitivity of our findings across diverse decoding settings, we additionally run our study with GPT-4 (the best performing model) on both tasks and languages with varying temperature and top\_p values and report the results in Tables 53, 54, 55 and 56 respectively. We find no significant or systematic differences across different decoding strategies which strengthens the robustness of our findings.

## A.6 Effect of Prompt Instructions

Due to the cost of LLM evaluation, we mainly experiment with one set of prompt instructions that we have found to be simple and effective through a moderate level of prompt engineering. However, to check the sensitivity of our findings across different prompt instructions, we additionally run our study with GPT-4 (the best performing model) on both tasks and languages with a paraphrased version of the original prompt instructions (found in Appendix F) and report the results in Tables 57 and 58. We find no significant or systematic differences across different prompts which strengthens the robustness of our findings.

## B Nonce word generation

Turkish To automatically generate novel nonce words in Turkish (out-of-distribution words that do not exist) that are inflected the same as the original word roots, we leverage the deterministic morphophonological features of Turkish. In particular, vowel harmony and consonant assimilation in Turkish completely determines which surface forms of the meta level morphemes would apply. Furthermore, these features depend only on the last vowel and the consonant. Hence, for a given word root in Turkish, we keep its last vowel and the consonant and randomly modify the other vowels and consonants with other vowels and consonants based on the frequency of each letter in Turkish to make sure we obtain words that would be plausible in this language. For example, if the given word root is "sanat", we keep the suffix "at" as is and modify the prefix "san" by randomly replacing each vowel in it with another vowel and consonant with another consonant. This makes sure that the words inflect the same and they are of the same length. However, if the word is too short (only two letters), and there is no prefix, we generate a random prefix of length three with vowels and consonants alternating (Turkish typically doesn’t allow dense consonant clusters)

Finnish The Finnish nonce word generation is done similarly to the Turkish nonce word generation, where we alter only the word root. All consonants are replaced with other consonants and vowels with other vowels that conform to the rules of Finnish vowel harmony.

<table><tr><td>#words #unique words #unique roots #unique meta affixes #unique affixes #unique meta affix compositions #unique affix compositions</td><td>3,775,470 348,173 9,576 103 372</td></tr></table>

Table 1: Statistics of BTWD dataset in Turkish. Meta affixes refer to the bound morphemes that are not surfacerealized.

<table><tr><td>#samples #unique roots #unique meta affixes #unique affixes #unique meta affix compositions #unique affix compositions</td><td>1,049 477 96 243</td></tr></table>

Table 2: Statistics of our final test suite in Turkish. Meta affixes refer to the bound morphemes that are not surface-realized.

<table><tr><td rowspan=1 colspan=1>Task</td><td rowspan=1 colspan=1>Test Distribution</td><td rowspan=1 colspan=1>κ</td></tr><tr><td rowspan=2 colspan=1>ProductivityProductivitySystematicitySystematicity</td><td rowspan=2 colspan=1>IDOODIDOOD</td><td rowspan=1 colspan=1>0.940.910.94</td></tr><tr><td rowspan=1 colspan=1>0.99</td></tr></table>

Table 3: Human inter-annotator agreement on Turkish test suite measured by Cohen’s κ score. We note that since the productivity task is an open-ended generative task, the chance agreement would be close to 0, hence κ score is equal to the raw agreement.

<table><tr><td>Task</td><td>Test Distribution</td><td>κ</td></tr><tr><td rowspan="3">Productivity Productivity Systematicity Systematicity</td><td>ID</td><td>0.77</td></tr><tr><td>OOD</td><td>0.78</td></tr><tr><td>ID OOD</td><td>0.75 0.84</td></tr></table>

Table 4: Human inter-annotator agreement on Finnish test suite measured by Cohen’s κ score. We note that since the productivity task is an open-ended generative task, the chance agreement would be close to 0, hence κ score is equal to the raw agreement.

## C Model Evaluation

We evaluate the following state-of-the-art multilingual instruction-finetuned LLMs:

• Aya-23 (Aryabumi et al., 2024) a powerful open-weights multilingual LLM serving 23 languages including Turkish. We evaluate both 8B and 35B sizes of this model series, but only on Turkish dataset as Aya-23 does not officially support Finnish yet.

• Qwen-2.5 (Team, 2024) recent open-weights multilingual LLM that has shown impressive results across various benchmarks and supports over 29 languages. We evaluate both 7B and 32B sizes of this model series in both languages.

• Gemini-1.5 (Gemini, 2024) a closed-source multilingual LLM that supports over 40 languages including Turkish and Finnish. We evaluate the gemini-1.5-flash version in both languages.

• GPT-4 (OpenAI, 2024) a closed-source multilingual LLM that supports many languages including Finnish and Turkish. We evaluate the 2024-02-15-preview version in both languages.

Models are evaluated using in-context few-shot learning where number of shots take values in {1,3,5}. We make sure each shot has the same number of morphemes as its corresponding task example. By default, all our prompt templates are in English since LLMs are quite proficient in following instructions in this language (Wendler et al., 2024), however, we also experiment with instruction templates in Turkish and Finnish which generally show worse performance (Appendix A.1). Similarly, while by default we use the standard prompting for all experiments, we also experiment with chain-of-thought prompting (Wei et al., 2023), but find very little difference in performance (Appendix A.2). Prompts for all tasks and languages can be found in Appendix F.

## D Data

Turkish Since the morphological analyzer we use to process the Turkish dataset (Ozturel et al., 2019) is based on a finite state machine relying on purely syntactic rules, it produces several alternative decompositions for some words (e.g. analyzer produces both decompositions “an+la+dıg+ımız” ˘ and “anla+dıg+ımız” for the word “anladı˘ gımız” ).˘ Hence, we further apply some language-specific heuristics to automatically filter out invalid decompositions. This preprocessing still leaves some words with multiple decompositions that can only be validated using semantics, hence, as a last step, we (authors) manually verify and determine the final segmentation of a word.

Productivity task prompt [ID root]   
You are given a word root and a list of affixes   
(separated by comma) in Turkish and your task   
is to generate a grammatically correct word   
from this root using all the given affixes. You   
are allowed to use only the given affixes and   
each affix only once. Answer with only the   
generated word.   
Example 1:   
Word root: bula¸s   
Affixes: ma, sa, tır, ydı, k   
Answer: bula¸stırmasaydık   
Example 2:   
Word root: bekle   
Affixes: me, di, z, n, e   
Answer:

## E Heuristic Negative Sample Selection For Turkish

Turkish phonology does not allow two vowels to occur together and typically employs "buffer" letters such as "y", "s" in between these vowels, however, blindly permuting the order of Turkish morphemes inevitably results in negative samples where two vowels may occur next to each other. We hypothesized that models might easily identify these options by exploiting the "no-two-vowel" shortcut and without considering the semantic order of morphemes. To check this hypothesis, we counted the number of GPT-4 mistakes corresponding to options that both have and don’t have two vowels occurring together and found that while the model makes a mistake in around 8% (in-distribution) and 16% (out-of-distribution) of all the negative options that do not have two vowels occurring together, these ratios are only 1% and 4% when we look at the negative options that have two adjacent vowels. Motivated by this discrepancy, we designed our third heuristic-based selection strategy for Turkish such that after ranking the options by their distance to the positive option, we select the top four negative options that do not have two adjacent vowels in their morpheme composition wherever possible.

## F Prompts

This section lists the instruction prompts for all tasks and language templates. We present examples in one-shot setting, templates for different shots are the same with more examples. For the English language template, we provide examples in Turkish, the templates are the same for Finnish with examples in Finnish.

## F.1 Templates in English

You are given a novel word root with its definition and a list of affixes (separated by comma) in Turkish and your task is to generate a grammatically correct word from this root using all the given affixes. You are allowed to use only the given affixes and each affix only once. Answer with only the generated word.

Example 1:

Word root: lıdı¸s

Definition: lıdı¸s means karı¸s in Turkish.

Affixes: sa, ydı, k, ma

Answer: lıdı¸smasaydık

Example 2:

Word root: ihek

Definition: ihek means emek in Turkish.

Affixes: in, imiz, ler, çi

Answer:

<table><tr><td rowspan=1 colspan=1>ID root (OOD root)</td><td rowspan=1 colspan=1>Affixes</td><td rowspan=1 colspan=1>ID Derivations</td></tr><tr><td rowspan=1 colspan=1>sohbet (şakşet)</td><td rowspan=1 colspan=1> $- \mathrm { l e r ~ o r ~ - y i n }$ </td><td rowspan=1 colspan=1>sohbetler √sohbetyin</td></tr><tr><td rowspan=1 colspan=1>sira (yova)</td><td rowspan=1 colspan=1> $- \mathrm { d a n , - m s }$ </td><td rowspan=1 colspan=1>sıradanmış√siramışdan</td></tr><tr><td rowspan=1 colspan=1>değer (diser)</td><td rowspan=1 colspan=1> $- \mathrm { l e n } , - \mathrm { d i r } , - \mathrm { i p }$ </td><td rowspan=1 colspan=1>değerlendirip √değeriplendirdeğerdirlenipdeğeripdirlendeğerlenipdir</td></tr><tr><td rowspan=1 colspan=1>endişe (ödlede)</td><td rowspan=1 colspan=1> $\mathrm { - l e n , - d i r , \mathrm { - m e , - m e k } }$ </td><td rowspan=1 colspan=1>endişelendirmemek√endişelendirmekmeendişemelendirmekendişelenmedirmekendişemedirlenmek</td></tr><tr><td rowspan=1 colspan=1>kişi (meşi)</td><td rowspan=1 colspan=1> $\mathbf { - l e } \ S , \mathbf { - t i r } , \mathbf { - m e } , \mathbf { - s i } , \mathbf { - n e }$ </td><td rowspan=1 colspan=1>kişileştirmesine √kişileştirnesimekişileştirmenesikişileşsitirmenekişileşmetirsine</td></tr><tr><td rowspan=1 colspan=1>hayal (rokal)</td><td rowspan=1 colspan=1> $\mathrm { - l e r , \mathrm { - i m , \mathrm { - d e , \mathrm { - k i , \mathrm { - l e r , \mathrm { - i } } } } } }$ </td><td rowspan=1 colspan=1>hayallerimdekileri √hayalleriimdekilerhayalilerimdekilerhayallerimdeikilerhayallerimdekiiler</td></tr><tr><td rowspan=1 colspan=1>sinıf (datıf)</td><td rowspan=1 colspan=1> $- \mathrm { l a n , - d u r , - 1 l , - m a , - l a r , - 1 , - n } 1$ </td><td rowspan=1 colspan=1>sınıflandırılmalarını√sıniflandırulmalarnısınıflardırılmalanınısınıflandırılmalarnusınıflandırılımalarnı</td></tr></table>

Table 5: Examples from our test suite in Turkish for each morpheme length from 1 to 7. OOD derivations can be obtained by replacing the ID root with the corresponding OOD root. Correct derivations are marked with ✓.

Morphological Productivity Finnish

![](images/719f7ad787ba7573ecc00264506233b2957ec1bbbe6a9dff54cc010173d52b0c.jpg)

![](images/4968333342651650015e8c4394876eafd7a59caac4eedfda875685888f123a0f.jpg)

![](images/eaf6b2575699955136278b8620940b42d706750840823a9cb3fd606528a6c055.jpg)

Figure 11: GPT-4 morphological productivity and systematicity task results for Finnish stratified by number of bound morphemes. Detailed results are in Tables 25, 26, 27.  
![](images/ed6739d3e9761bf70ddd151e40ecbb85df3f5ed4681def272cf0f8eb40dc27d1.jpg)

![](images/9a477d365915a92e51ce9c7d14a3878aab1b52a5bf8baace062d8bbe6347c9f4.jpg)

![](images/4d73aa8c3425d1b46d419f77296278500ec190feca028eb5c98d29dc22d6af0d.jpg)

Figure 12: Morphological productivity and systematicity task results for Finnish showing the effect of the instruction language. Detailed results are in Tables 33.
<table><tr><td>#samples</td><td>480 406</td></tr><tr><td>#unique roots</td><td>386</td></tr><tr><td>#unique affixes #unique affix compositions</td><td>365</td></tr></table>

Table 6: Statistics of our final test suite in Finnish.

## Systematicity task prompt [ID root]

You are given a word root, a list of affixes (separated by comma) and a word in Turkish that is derived from the given word root using the given affixes. Your task is to determine whether the derived word is grammatically correct. Answer only with Yes or No.

Example 1:

Word root: küçük

Affixes: ümüz, lüg, den ˘

Derived word: küçüklügümüzden ˘

Answer: Yes

Example 2:

Word root: evren

Affixes: sel, e, lig˘

Derived word: evrenesellig˘

Answer:

## Systematicity task prompt [OOD root]

You are given a novel word root with its definition, a list of affixes (separated by comma) and a word in Turkish that is derived from the given word root using the given affixes. Your task is to determine whether the derived word is grammatically correct. Answer only with Yes or No.

Example 1:

Word root: ene¸silvöte

Definition: ene¸silvöte means üniversite in Turkish.

Affixes: niz, yse, de

Derived word: ene¸silvötedeyseniz

Answer: Yes

## Example 2:

Word root: yivek

Definition: yivek means yürek in Turkish.

Affixes: den, ler, iniz

Derived word: yiveklerdeniniz

Answer:

<table><tr><td rowspan=1 colspan=1>ID root (OOD root)</td><td rowspan=1 colspan=1>Affixes</td><td rowspan=1 colspan=1>ID Derivations</td></tr><tr><td rowspan=1 colspan=1>yöpaikka (äydainca)</td><td rowspan=1 colspan=1>-nne or -ksi</td><td rowspan=1 colspan=1>yöpaikkanne√yöpaikkaksi</td></tr><tr><td rowspan=1 colspan=1>sano (tato)</td><td rowspan=1 colspan=1>-taan, -pas</td><td rowspan=1 colspan=1>sanotaanpas√sanopastaan</td></tr><tr><td rowspan=1 colspan=1>petoks (seloks)</td><td rowspan=1 colspan=1>-i, -ne, -en</td><td rowspan=1 colspan=1>petoksineen √petoksneienpetoksneenipetoksiennepetoksennei</td></tr><tr><td rowspan=1 colspan=1>olosuhte (olanajke)</td><td rowspan=1 colspan=1>-kuvaus, -i, -lta, -an</td><td rowspan=1 colspan=1>kuvausolosuhteiltaan √kuvausolosuhteltaiankuvausolosuhteltaanikuvausolosuhteianltakuvausolosuhteanilta</td></tr><tr><td rowspan=1 colspan=1>palvelu (sapsevu)</td><td rowspan=1 colspan=1>-laina, -n, -välitys, -j, -a</td><td rowspan=1 colspan=1>lainanvälityspalveluja√lainanvälityspalveluajnlainavälityspalvelujalainavälitysnpalvelujalainavälitysnpalveluaj</td></tr><tr><td rowspan=1 colspan=1>salaisuuks (noraekauks)</td><td rowspan=2 colspan=1>-motivaatio, -n, -nostatus, -i, -a, -ni</td><td rowspan=2 colspan=1>motivaationnostatussalaisuuksiani √motivaationnostatussalaisuuksiniamotivaationnostatussalaisuuksainimotivaationnostatussalaisuuksniaimotivaationostatusnsalaisuuksiani</td></tr><tr><td rowspan=1 colspan=1></td></tr></table>

Table 7: Examples from our test suite in Finnish for each morpheme length from 1 to 6. OOD derivations can be obtained by replacing the ID root with the corresponding OOD root. Correct derivations are marked with ✓.

![](images/f5f903a8569173438dd95f7956f242cb92ba3a857c61e2a8ae8f4296b15ebe10.jpg)  
Model

![](images/059d147d53e7efe6bdc28577b1953d5aaa791c867f428741b2c9dce210fe1ccf.jpg)  
Model

![](images/bdebb407abe824df7d629e9ec80b356509355e71b8d3a08db9c90c6d72f439fe.jpg)  
Model

Figure 13: Morphological productivity and systematicity task results for Finnish showing the effect of additional context. Detailed results are in Table 39.  
![](images/bb2aa4273f66f430636fb98767a41f12549fa90dacff800e2733f0f93ad797d5.jpg)

![](images/c9c8db0b98a487ceb5adb51d59c4af546e92c4aa5d2e686ec5650117462aa684.jpg)

![](images/2ea17fbd2b3b7a5ea812ff823a2e9ba53b48b6e5f22df6ef712b44bc186607de.jpg)  
Model  
Figure 14: Morphological productivity and systematicity task results for Finnish showing the effect of the morpheme order. Detailed results are in Table 49.

## Productivity task prompt [ID root] (with context)

You are given a word root, a list of affixes (separated by comma) and a sentence with a blank (\_\_\_) in Turkish and your task is to fill in the blank by generating a grammatically correct word from this root using all the given affixes. You are allowed to use only the given affixes and each affix only once. Answer with only the generated word.

Word root: kal

Affixes: an, lar

Sentence: giden geminin yokluguna bir türlü ˘ inandıramaz kendilerini limanda

Answer: kalanlar

Example 2:

Word root: kurtar

Affixes: ecek, abil

Sentence: göç ettikten sonra diger hem¸ser- ˘ ileri gibi mal, mülk pe¸sinde olsa belki annesini parasızlıktan belki de kızı bir fabrika kö¸sesinde çalı¸smak zorunda kalmayıp daha uzun ya¸sayabilecekti

Systematicity task prompt [ID root] (with context) You are given a word root, a list of affixes (separated by comma), a sentence with a blank (\_\_\_) and a word in Turkish that is derived from the given word root using the given affixes. Your task is to determine whether the derived word is the correct option to fill in the blank. Answer only with Yes or No.

Word root: küçük

Affixes: ümüz, den, lüg˘

Derived word: küçüklügümüzden ˘

Affixes: lan, ız, acag˘

Sentence: bir ¸seyler ya¸sadıktan sonra mı hep

Derived word: akılacagızlan ˘

<table><tr><td rowspan="2">Models</td><td colspan="2">Morph. Productivity (accuracy)</td><td colspan="2">Morph. Systematicity (macro-F1)</td><td colspan="2">Morph. Systematicity (coherence)</td></tr><tr><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td></tr><tr><td>majority</td><td>0.0 / 0.0 / 0.0</td><td>0.0 / 0.0 / 0.0</td><td>41.2 / 41.2 / 41.2</td><td>41.2 / 41.2 / 41.2</td><td>0.0 / 0.0 / 0.0</td><td>0.0 / 0.0 / 0.0</td></tr><tr><td>random</td><td>24.6 / 25.0 / 25.0</td><td>24.7 / 24.6 / 24.2</td><td>41.8 / 41.8 / 43.5</td><td>43.0 / 42.3 / 42.3</td><td>9.1 / 9.0 / 9.4</td><td>8.5 / 9.6 / 9.0</td></tr><tr><td>aya-23-8b</td><td>12.8 / 13.7 / 13.3</td><td>8.8 / 11.5 / 12.3</td><td>62.0 / 64.6 / 67.5</td><td>53.9 / 49.3 / 51.5</td><td>27.9 / 31.4 / 36.0</td><td>19.1 / 15.7 / 18.4</td></tr><tr><td>aya-23-35b</td><td>17.4 / 19.8 / 21.0</td><td>14.6 / 17.7 / 19.3</td><td>69.9 / 80.1 / 81.8</td><td>64.6 / 71.0 / 72.1</td><td>36.8 / 52.6 / 55.8</td><td>29.2 / 39.9 / 41.8</td></tr><tr><td>qwen-2.5-7b</td><td>15.0 / 14.9 / 15.8</td><td>13.2 / 12.9 / 12.9</td><td>71.1 / 73.6 / 74.6</td><td>65.7 / 66.8 / 66.0</td><td>40.5 / 44.3 / 45.1</td><td>33.5 / 33.9 / 33.1</td></tr><tr><td>qwen-2.5-32b</td><td>22.6 / 23.7 / 24.1</td><td>21.7 / 21.8 / 21.8</td><td>77.3 / 84.7 / 85.9</td><td>53.1 / 71.3 / 75.3</td><td>56.7 / 66.3 / 66.8</td><td>18.5 / 45.7 / 48.3</td></tr><tr><td>gemini-1.5-flash</td><td>28.8 / 30.5 / 30.7</td><td>24.9 / 25.7 / 25.1</td><td>60.8 / 80.8 / 85.4</td><td>41.4 / 52.8 / 62.1</td><td>32.2 / 63.6 / 70.7</td><td>0.4 / 19.3 / 33.3</td></tr><tr><td>gpt-4</td><td>49.0 / 52.1 / 54.2</td><td>36.7 / 40.5 / 43.9</td><td>85.5 / 90.2 / 91.6</td><td>61.9 / 77.7 / 78.8</td><td>71.4 / 76.8 / 76.6</td><td>33.5 / 55.9 / 51.4</td></tr><tr><td>human*</td><td>97.1</td><td>95.0</td><td>98.8</td><td>99.1</td><td>95.7</td><td>97.9</td></tr></table>

Table 8: 1-shot / 3-shot / 5-shot results for Turkish in English template for all examined models across tasks. ∗Due to the cost of evaluation, our human study is only evaluated on 70 randomly sampled instances per task and test distribution.
<table><tr><td rowspan="2">Models</td><td colspan="2">Morph. Productivity (accuracy)</td><td colspan="2">Morph. Systematicity (macro-F1)</td><td colspan="2">Morph. Systematicity (coherence)</td></tr><tr><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td></tr><tr><td>majority</td><td>0.0 / 0.0 / 0.0</td><td>0.0 / 0.0 / 0.0</td><td>40.7 / 40.7 / 40.7</td><td>40.7 / 40.7 / 40.7</td><td>0.0 / 0.0 / 0.0</td><td>0.0 / 0.0 / 0.0</td></tr><tr><td>random</td><td>29.4 / 32.3 / 29.8</td><td>32.7 / 30.2 / 30.2</td><td>42.4 / 43.3 / 42.4</td><td>42.4 / 43.7 / 42.5</td><td>10.2 / 11.7 / 10.8</td><td>10.8 / 10.8 / 9.6</td></tr><tr><td>qwen-2.5-7b</td><td>13.5 / 13.5 / 16.0</td><td>10.2 / 11.7 / 14.4</td><td>61.3 / 65.4 / 68.3</td><td>54.6 / 57.3 / 59.4</td><td>31.2 / 35.8 / 39.2</td><td>21.9 / 25.8 / 27.7</td></tr><tr><td>qwen-2.5-32b</td><td>22.5 / 21.9 / 22.3</td><td>19.2 / 19.8 / 21.3</td><td>52.0 / 65.9 / 69.0</td><td>43.6 / 54.7 / 62.2</td><td>19.0 / 39.8 / 42.5</td><td>5.2 / 22.1 / 33.1</td></tr><tr><td>gemini-1.5-flash</td><td>22.5 / 26.9 / 28.1</td><td>20.6 / 22.9 / 24.0</td><td>49.4 / 71.2 / 77.7</td><td>40.7 / 50.3 / 56.8</td><td>14.2 / 48.1 / 52.3</td><td>0.0 / 15.4 / 25.0</td></tr><tr><td>gpt-4</td><td>37.7 / 40.6 / 44.2</td><td>31.5 / 35.0 / 34.4</td><td>70.0 / 83.1 / 85.2</td><td>42.2 / 65.6 / 74.8</td><td>47.5 / 65.4 / 66.2</td><td>2.7 / 39.8 / 50.6</td></tr><tr><td>human*</td><td>85.0</td><td>83.3</td><td>89.4</td><td>91.7</td><td>75.0</td><td>75.8</td></tr></table>

Table 9: 1-shot / 3-shot / 5-shot results for Finnish in English template for all examined models across tasks. ∗Due to the cost of evaluation, our human study is only evaluated on 60 randomly sampled instances per task and test distribution.

Productivity task prompt [ID root] (CoT) You are given a word root and a list of affixes (separated by comma) in Turkish. Your task is to construct a grammatically correct word by appending the given affixes to the root. Use each affix exactly once. After forming a word, list each affix used in the construction of that word to verify adherence to the rules. Check the following: Ensure no affix is used more than once, confirm that all provided affixes are used, verify that no extra affixes outside the provided list are included. Think step by step and then provide your final answer within the tags <Answer>correctword</Answer>.

Productivity task prompt [OOD root] (CoT) You are provided with a novel word root with its definition, and a list of affixes (separated by comma) in Turkish. Your task is to construct a grammatically correct word by appending the given affixes to the root. Use each affix exactly once. After forming a word, list each affix used in the construction of that word to verify adherence to the rules. Check the following: Ensure no affix is used more than once, confirm that all provided affixes are used, verify that no extra affixes outside the provided list are included. Think step by step and then provide your final answer within the tags <Answer>correctword</Answer>.

Example 1:

Word root: kuru

Affixes: t, mu¸s

Answer: First, let’s append the affixes to the root "kuru" in a grammatically correct order:

...<explaining the correct order of morphemes>...

Example 2:

Word root: mana

Example 1:

Affixes: sız, dır

Word root: doru

Definition: doru means kuru in Turkish.

Affixes: t, mu¸s

Answer: ...<explanation>...

Example 2:

Word root: çokan

Definition: çokan means yalan in Turkish.

Answer:

Affixes: la, lar

Answer:

<table><tr><td rowspan="2">Models</td><td colspan="7">Number of morphemes (excl. root)</td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td></tr><tr><td>majority</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td></tr><tr><td>random</td><td>100.0 / 100.0</td><td>48.0 / 46.7</td><td>20.0 / 18.0</td><td>3.3 / 6.7</td><td>0.0 / 1.3</td><td>0.7 / 0.0</td><td>0.0 / 0.0</td></tr><tr><td>aya-23-8b</td><td>60.0 / 52.7</td><td>22.7 / 8.7</td><td>5.3 / 0.0</td><td>0.7 / 0.0</td><td>0.7 / 0.0</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td></tr><tr><td>aya-23-35b</td><td>72.7 / 69.3</td><td>35.3 / 20.7</td><td>7.3 / 6.7</td><td>4.0 / 4.0</td><td>2.0 / 1.3</td><td>0.0 / 0.0</td><td>0.7 / 0.0</td></tr><tr><td>qwen-2.5-7b</td><td>63.3 / 64.0</td><td>26.7 / 22.0</td><td>13.3 / 6.0</td><td>0.0 / 0.7</td><td>0.0 / 0.0</td><td>0.7 / 0.0</td><td>0.7 / 0.0</td></tr><tr><td>qwen-2.5-32b</td><td>82.0 / 85.3</td><td>46.3 / 42.3</td><td>18.7 / 15.3</td><td>6.0 / 6.7</td><td>3.3 / 1.3</td><td>1.3 / 0.7</td><td>0.7 / 0.0</td></tr><tr><td>gemini-1.5-flash</td><td>86.7 / 80.0</td><td>52.7 / 44.7</td><td>36.0 / 30.7</td><td>12.0 / 10.7</td><td>7.3 / 7.3</td><td>4.7 / 0.7</td><td>2.0 / 0.0</td></tr><tr><td>gpt-4</td><td>95.3 / 96.7</td><td>80.7 / 65.3</td><td>62.7 / 43.3</td><td>43.8 / 31.3</td><td>27.3 / 17.6</td><td>19.3 / 0.7</td><td>13.8 / 2.1</td></tr></table>

Table 10: Morphological productivity 1-shot ID / OOD accuracy results for Turkish in English template for all examined models.
<table><tr><td rowspan="2">Models</td><td colspan="7">Number of morphemes (excl. root)</td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td></tr><tr><td>majority</td><td>33.3 / 33.3</td><td>33.1 / 33.1</td><td>44.3 / 44.3</td><td>44.4 / 44.4</td><td>44.4 / 44.4</td><td>44.4 / 44.4</td><td>44.4 / 44.4</td></tr><tr><td>random</td><td>44.9 / 42.4</td><td>37.6 /39.3</td><td>40.2 / 41.4</td><td>43.5 / 44.5</td><td>42.6 / 46.4</td><td>41.9 / 42.4</td><td>42.3 / 44.8</td></tr><tr><td>aya-23-8b</td><td>72.9 / 54.7</td><td>68.0 / 48.7</td><td>66.1 / 50.0</td><td>58.8 / 54.8</td><td>59.0 / 56.5</td><td>54.3 / 55.3</td><td>55.1 / 57.5</td></tr><tr><td>aya-23-35b</td><td>70.4 / 60.2</td><td>82.7 / 70.2</td><td>83.1 / 73.0</td><td>65.0 / 58.9</td><td>63.5 / 65.5</td><td>61.5 / 61.9</td><td>63.2 / 62.6</td></tr><tr><td>qwen-2.5-7b</td><td>71.8 / 53.1</td><td>72.9 / 64.7</td><td>73.7 / 64.4</td><td>77.7 / 74.0</td><td>73.0 / 72.3</td><td>65.5 / 66.9</td><td>63.3 / 64.4</td></tr><tr><td>qwen-2.5-32b</td><td>65.6 / 34.2</td><td>57.0 /35.1</td><td>75.6 / 57.6</td><td>87.4 / 61.5</td><td>86.8 / 62.1</td><td>84.2 / 57.3</td><td>84.7 / 63.9</td></tr><tr><td>gemini-1.5-flash</td><td>62.4/ 33.3</td><td>58.2 / 34.0</td><td>60.1 / 44.2</td><td>59.0 / 44.8</td><td>61.9 / 44.4</td><td>56.9 / 44.3</td><td>67.2 / 44.7</td></tr><tr><td>gpt-4</td><td>86.7 / 36.2</td><td>69.1 / 43.8</td><td>82.5 / 61.8</td><td>88.4 / 64.2</td><td>92.2 / 78.7</td><td>88.7 / 72.0</td><td>90.6 / 76.9</td></tr></table>

Table 11: Morphological systematicity 1-shot ID / OOD macro-F1 results for Turkish in English template for all examined models.

Systematicity task prompt [ID root] (CoT) You are given a word root, a list of affixes (separated by comma) and a word in Turkish that is derived from the given word root using the given affixes. Your task is to determine whether the derived word is grammatically correct. First, analyze how the affixes interact with the word root. Then, assess the order in which the affixes are applied and verify that this order adheres to the language’s rules. Think step by step and then provide your final answer within the tags <Answer>Yes/No</Answer>.   
Example 1:   
Word root: kuru   
Affixes: t, mu¸s   
Derived word: kurutmu¸s   
Answer: To analyze the derived word "kurutmu¸s," we need to look at the affixes and how they interact with the word root "kuru."   
...<explaining the correct order of morphemes>...   
Example 2:   
Word root: etki   
Affixes: yici, le   
Derived word: etkileyici   
Answer: Systematicity task prompt [OOD root] (CoT) You are given a novel word root with its definition, a list of affixes (separated by comma) and a word in Turkish that is derived from the given word root using the given affixes. Your task is to determine whether the derived word is grammatically correct. First, analyze how the affixes interact with the word root. Then, assess the order in which the affixes are applied and verify that this order adheres to the language’s rules. Think step by step and then provide your final answer within the tags <Answer>Yes/No</Answer>. Example 1:   
Word root: doru   
Definition: doru means kuru in Turkish.   
Affixes: t, mu¸s   
Derived word: dorutmu¸s   
Answer: ...<explain the correct order of morphemes based on the definition>...   
Example 2:   
Word root: imli   
Definition: imli means etki in Turkish.   
Affixes: yici, le   
Derived word: imlileyici   
Answer:

<table><tr><td rowspan="2">Models</td><td colspan="7">Number of morphemes (excl. root)</td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td></tr><tr><td>majority</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td></tr><tr><td>random</td><td>28.0 / 25.3</td><td>18.7 / 22.0</td><td>2.0 / 1.3</td><td>4.7 / 2.0</td><td>2.7 / 2.7</td><td>4.0 / 2.7</td><td>4.0 / 3.4</td></tr><tr><td>aya-23-8b</td><td>62.0 / 38.0</td><td>53.3 / 26.7</td><td>20.0 / 8.7</td><td>16.7 / 12.7</td><td>18.0 / 15.3</td><td>10.7 / 13.3</td><td>14.8 / 18.8</td></tr><tr><td>aya-23-35b</td><td>56.7 / 45.3</td><td>74.0 / 57.3</td><td>47.3 / 29.3</td><td>19.3 / 14.0</td><td>16.0 / 20.7</td><td>20.0 / 17.3</td><td>24.2 / 20.1</td></tr><tr><td>qwen-2.5-7b</td><td>60.7 / 36.0</td><td>60.7 / 49.3</td><td>39.3 / 29.3</td><td>42.0 / 34.7</td><td>36.7 / 34.7</td><td>21.3 / 24.0</td><td>22.8 / 26.2</td></tr><tr><td>qwen-2.5-32b</td><td>49.3 / 2.7</td><td>35.6 / 2.7</td><td>54.0 / 20.7</td><td>71.3 / 28.7</td><td>66.7 / 28.7</td><td>63.3 / 19.3</td><td>56.4 / 26.8</td></tr><tr><td>gemini-1.5-flash</td><td>44.0 / 0.0</td><td>38.7 / 1.3</td><td>26.7 / 0.0</td><td>24.7 / 0.7</td><td>30.7 / 0.0</td><td>22.7 / 0.0</td><td>38.3 / 0.7</td></tr><tr><td>gpt-4</td><td>80.0 / 4.7</td><td>54.0 / 16.0</td><td>66.7 / 30.0</td><td>76.0 / 34.7</td><td>82.7 / 58.7</td><td>68.7 / 41.3</td><td>71.8 / 49.0</td></tr></table>

Table 12: Morphological systematicity 1-shot ID / OOD coherence results for Turkish in English template for all examined models.
<table><tr><td rowspan="2">Models</td><td colspan="7">Number of morphemes (excl. root)</td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td></tr><tr><td>majority random</td><td>0.0 / 0.0</td><td>0.0 / 0.0 46.7 / 50.7</td><td>0.0 / 0.0 20.7 / 16.0</td><td>0.0 / 0.0 5.3 / 4.7</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td></tr><tr><td rowspan="2">aya-23-8b</td><td>100.0 / 100.0</td><td></td><td></td><td></td><td>2.7 / 0.7</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td></tr><tr><td>58.0 / 64.7</td><td>29.3 / 13.3</td><td>5.3 / 2.0</td><td>2.0 / 0.0</td><td>1.3 / 0.7</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td></tr><tr><td>aya-23-35b</td><td>73.3 / 84.0</td><td>43.3 / 29.3</td><td>12.0 /6.7</td><td>5.3 / 4.0</td><td>1.3 / 0.0</td><td>1.3 / 0.0</td><td>2.0 / 0.0</td></tr><tr><td>qwen-2.5-7b</td><td>68.7 / 61.3</td><td>24.7 / 22.7</td><td>6.7 / 5.3</td><td>2.0 / 0.7</td><td>1.3 / 0.0</td><td>0.7 / 0.0</td><td>0.0 / 0.0</td></tr><tr><td>qwen-2.5-32b</td><td>84.7 / 80.7</td><td>45.0 / 38.9</td><td>21.3 / 18.7</td><td>6.7 / 11.3</td><td>4.7 / 3.3</td><td>0.7 / 0.0</td><td>2.7 / 0.0</td></tr><tr><td>gemini-1.5-flash</td><td>84.7 / 80.0</td><td>57.3 / 50.0</td><td>37.3 / 29.3</td><td>16.7 / 9.3</td><td>10.7 / 7.3</td><td>4.0 / 2.7</td><td>2.7 / 1.3</td></tr><tr><td>gpt-4</td><td>94.7 / 94.7</td><td>81.3 / 68.7</td><td>64.0 / 45.2</td><td>49.3 / 34.2</td><td>30.7 / 17.6</td><td>25.3 / 11.6</td><td>19.3 / 11.7</td></tr></table>

Table 13: Morphological productivity 3-shot ID / OOD accuracy results for Turkish in English template for all examined models.

You are provided with a word root and a set of affixes (comma-separated) in language. Your task is to create a grammatically correct word using this root and all the provided affixes. You must use only the given affixes, and each affix can be used only once. Respond with the final word only.

Example 1:

Word root: bula¸s

Affixes: ma, sa, tır, ydı, k

Answer: bula¸stırmasaydık

Example 2:

Word root: bekle

Affixes: me, di, z, n, e

Answer:

You are given a new word root along with its definition, and a set of affixes (commaseparated) in language. Assuming that the new word root is a valid language word, your task is to form a grammatically correct word using this root and all the provided affixes. You must use only the given affixes, and each one can be used just once. Provide only the generated word as your answer.

Example 1:

Word root: lıdı¸s

Definition: lıdı¸s means karı¸s in Turkish.

Affixes: sa, ydı, k, ma

Answer: lıdı¸smasaydık

Example 2:

Word root: ihek

Definition: ihek means emek in Turkish.

Affixes: in, imiz, ler, çi

Answer:

<table><tr><td>Systematicity task prompt [ID root][paraphrased] You are provided with a word root, a set of affixes (comma-separated), and a word in language that is derived from the given root using the provided affixes. Your task is to verify whether the derived word is gram- matically correct. Respond with only Yes or No. Example 1: Word root: küçük Affixes: ümüz, lüğ, den Derived word: küçüklüğümüzden Answer: Yes Example 2: Word root: evren Affixes: sel, e, liğ Derived word: evreneselliğ Answer:</td><td>Systematicity task prompt [OOD root][paraphrased] You are provided with a new word root along with its definition, a set of affixes (comma-separated), and a word in language that is derived from the given root using the provided affixes. Assuming that the new word root is a valid language word, your task is to verify whether the derived word is gram- matically correct. Respond with only Yes or No. Example 1: Word root: eneşilvöte Definition: eneşilvöte means üniversite in Turkish. Affixes: niz, yse, de Derived word: eneşilvötedeyseniz Answer: Yes Example 2: Word root: yivek Definition: yivek means yürek in Turkish. Affixes: den, ler, iniz Derived word: yiveklerdeniniz Answer:</td></tr></table>

<table><tr><td rowspan="2">Models</td><td colspan="7">Number of morphemes (excl. root)</td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td></tr><tr><td>majority random</td><td>33.3 / 33.3 40.0 / 45.3</td><td>33.1 / 33.1 40.2 / 43.1</td><td>44.3 / 44.3</td><td>44.4 / 44.4</td><td>44.4 / 44.4</td><td>44.4 / 44.4</td><td>44.4 / 44.4</td></tr><tr><td></td><td></td><td></td><td>44.4 / 41.5</td><td>40.2 / 43.4</td><td>43.9 / 40.3</td><td>43.9 / 41.8</td><td>39.8 / 41.0</td></tr><tr><td>aya-23-8b</td><td>75.3 / 51.8</td><td>68.0 / 43.3</td><td>64.5 / 34.2</td><td>60.3 / 44.2</td><td>64.3 / 52.7</td><td>55.6 / 55.3</td><td>64.0 / 63.3</td></tr><tr><td>aya-23-35b</td><td>74.9 / 57.3</td><td>83.6 / 68.2</td><td>86.2 / 78.1</td><td>79.5 / 75.3</td><td>77.3 / 77.3</td><td>78.8 / 72.8</td><td>80.3 / 68.2</td></tr><tr><td>qwen-2.5-7b</td><td>60.7 / 56.9</td><td>75.3 / 62.9</td><td>76.9 / 72.2</td><td>78.4 / 67.9</td><td>74.1 / 72.3</td><td>74.9 / 67.3</td><td>74.9 / 68.4</td></tr><tr><td>qwen-2.5-32b</td><td>76.4 / 53.8</td><td>74.0 / 60.2</td><td>87.3 / 75.6</td><td>88.6 / 78.0</td><td>91.0 / 76.6</td><td>89.0 / 76.2</td><td>86.5 / 78.7</td></tr><tr><td>gemini-1.5-flash</td><td>86.0 / 45.3</td><td>81.6 / 50.4</td><td>79.1 / 55.0</td><td>71.2 /53.9</td><td>81.2 / 55.7</td><td>79.1 / 55.1</td><td>87.6 / 54.1</td></tr><tr><td>gpt-4</td><td>89.6 /59.1</td><td>81.8 / 62.9</td><td>92.5 / 84.9</td><td>94.9 / 85.7</td><td>92.2 / 84.7</td><td>88.0 / 81.8</td><td>92.4 / 84.7</td></tr></table>

Table 14: Morphological systematicity 3-shot ID / OOD macro-F1 results for Turkish in English template for all examined models.
<table><tr><td rowspan="2">Models</td><td colspan="7">Number of morphemes (excl. root)</td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td></tr><tr><td>majority random</td><td>0.0 / 0.0 25.3 / 29.3</td><td>0.0 / 0.0 23.3 / 24.7</td><td>0.0 / 0.0 2.7 / 2.7</td><td>0.0 / 0.0 0.7 / 2.7</td><td>0.0 / 0.0 3.3 / 2.0</td><td>0.0 / 0.0 6.0/ 1.3</td><td>0.0 / 0.0 2.0 / 4.7</td></tr><tr><td>aya-23-8b</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>aya-23-35b</td><td>64.7 / 34.0</td><td>52.7 / 19.3</td><td>22.0 / 2.0</td><td>22.7 / 6.7</td><td>24.0 / 12.0</td><td>13.3 / 12.7</td><td>20.1 / 23.5</td></tr><tr><td></td><td>64.0 / 39.3</td><td>75.3 / 54.7</td><td>56.7 / 43.3</td><td>41.3 / 34.7</td><td>39.3 / 41.3</td><td>43.3 / 36.0</td><td>48.3 / 30.2</td></tr><tr><td>qwen-2.5-7b</td><td>44.7 / 41.3</td><td>64.7 / 46.7</td><td>39.3 / 34.0</td><td>45.3 / 26.0</td><td>36.0 / 32.7</td><td>40.0 / 28.7</td><td>40.3 / 28.2</td></tr><tr><td>qwen-2.5-32b</td><td>65.3 / 32.7</td><td>61.1 / 40.9</td><td>71.3 / 49.3</td><td>65.3 / 52.0</td><td>72.0 / 51.3</td><td>66.7 / 46.0</td><td>62.4 / 47.7</td></tr><tr><td>gemini-1.5-flash</td><td>79.3 / 20.0</td><td>72.7 / 26.0</td><td>58.0 / 18.7</td><td>45.3 / 16.0</td><td>62.0 / 18.7</td><td>56.7 / 18.0</td><td>71.1 / 17.4</td></tr><tr><td>gpt-4</td><td>84.7 / 40.0</td><td>72.7 / 45.3</td><td>84.0 / 65.3</td><td>84.0 / 70.0</td><td>74.7 / 61.3</td><td>63.3 / 52.7</td><td>74.5 / 56.4</td></tr><tr><td>majority</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td></tr><tr><td>random</td><td>100.0 / 100.0</td><td>44.0 / 48.7</td><td>22.7 / 15.3</td><td>7.3 / 4.7</td><td>0.7 / 0.7</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td></tr><tr><td>aya-23-8b</td><td>60.0 / 64.0</td><td>26.7 / 19.3</td><td>3.3 / 2.7</td><td>0.7 / 0.0</td><td>1.3 / 0.0</td><td>0.7 / 0.0</td><td>0.7 / 0.0</td></tr><tr><td>aya-23-35b</td><td>76.0 / 87.3</td><td>46.0 / 34.7</td><td>14.7 / 10.0</td><td>5.3 / 3.3</td><td>1.3 / 0.0</td><td>2.0 / 0.0</td><td>1.3 / 0.0</td></tr><tr><td>qwen-2.5-7b</td><td>70.7 / 65.3</td><td>27.3 / 16.7</td><td>8.0 / 6.7</td><td>3.3 / 0.7</td><td>0.7 / 0.7</td><td>0.0 / 0.0</td><td>0.7 / 0.0</td></tr><tr><td>qwen-2.5-32b</td><td>83.3 / 86.7</td><td>48.3 / 39.6</td><td>20.7 / 14.7</td><td>12.0 / 10.0</td><td>2.7 / 2.0</td><td>0.0 / 0.0</td><td>2.0 / 0.0</td></tr><tr><td>gemini-1.5-flash</td><td>91.3 / 79.3</td><td>56.7 / 51.3</td><td>39.3 / 29.3</td><td>12.0 / 8.7</td><td>10.0 / 6.0</td><td>2.7 / 1.3</td><td>2.7 / 0.0</td></tr><tr><td>gpt-4</td><td>96.0 / 96.7</td><td>85.3 / 72.0</td><td>66.0 / 55.5</td><td>43.7 / 37.3</td><td>40.0 / 23.9</td><td>28.0 / 11.6</td><td>20.6 / 10.1</td></tr><tr><td>human</td><td>100.0 / 100.0</td><td>100.0 / 95.0</td><td>100.0 / 95.0</td><td>100.0 / 100.0</td><td>100.0 / 95.0</td><td>90.0 / 100.0</td><td>90.0 / 80.0</td></tr></table>

Table 15: Morphological systematicity 3-shot ID / OOD coherence results for Turkish in English template for all examined models.

Table 16: Morphological productivity 5-shot ID / OOD accuracy results for Turkish in English template for all examined models. 1-shot and 3-shot results can be found in Tables 10, 13 respectively.
<table><tr><td rowspan="2">Models</td><td colspan="7">Number of morphemes (excl. root)</td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td></tr><tr><td>majority</td><td>33.3 / 33.3</td><td>33.1 / 33.1</td><td>44.3 / 44.3</td><td>44.4 / 44.4</td><td>44.4 / 44.4</td><td>44.4 / 44.4</td><td>44.4 / 44.4</td></tr><tr><td>random</td><td>42.9 / 46.4</td><td>43.8 / 37.6</td><td>43.6 / 42.2</td><td>40.4 / 40.4</td><td>44.5 / 43.9</td><td>43.1 / 44.0</td><td>46.0 / 41.4</td></tr><tr><td>aya-23-8b</td><td>74.0 / 49.6</td><td>71.3 / 49.8</td><td>67.7 / 44.3</td><td>65.6 / 52.8</td><td>69.7 / 56.7</td><td>61.3 / 48.0</td><td>63.3 / 59.4</td></tr><tr><td>aya-23-35b</td><td>77.3 / 63.6</td><td>87.1 / 68.2</td><td>85.7 / 79.4</td><td>82.2 / 75.2</td><td>82.1 / 76.7</td><td>80.5 / 70.5</td><td>77.8 / 71.2</td></tr><tr><td>qwen-2.5-7b</td><td>66.0 / 58.0</td><td>76.2 / 62.7</td><td>76.2/ 71.1</td><td>78.6 / 70.4</td><td>74.4 / 68.6</td><td>76.6 / 67.1</td><td>74.5 / 63.8</td></tr><tr><td>qwen-2.5-32b</td><td>80.4 / 58.4</td><td>78.1 / 68.9</td><td>89.5 / 79.1</td><td>90.3 / 80.8</td><td>89.7 / 80.8</td><td>88.1 / 81.0</td><td>85.5 / 77.9</td></tr><tr><td>gemini-1.5-flash</td><td>86.9 / 53.3</td><td>82.9 / 55.8</td><td>83.4 / 66.3</td><td>85.5 / 60.0</td><td>84.8 / 60.5</td><td>85.5 / 66.3</td><td>89.2 / 72.3</td></tr><tr><td>gpt-4</td><td>92.0 / 52.7</td><td>90.9 / 82.9</td><td>94.4 / 86.1</td><td>93.9 / 84.7</td><td>90.8 / 80.6</td><td>89.1 / 82.9</td><td>90.2 / 81.4</td></tr><tr><td>human</td><td>100.0 / 100.0</td><td>100.0 / 96.7</td><td>100.0 / 100.0</td><td>97.2 / 100.0</td><td>100.0 / 100.0</td><td>98.8 / 97.6</td><td>95.2 / 100.0</td></tr></table>

Table 17: Morphological systematicity 5-shot ID / OOD macro-F1 results for Turkish in English template for all examined models. 1-shot and 3-shot results can be found in Tables 11, 14 respectively.

## F.2 Templates in Turkish

Size Türkçe bir kök ve bir ek listesi (virgülle ayrılmı¸s) verilecek ve sizden bu kökten verilen tüm ekleri kullanarak dilbilgisel olarak dogru˘ bir kelime üretmeniz istenecek. Sadece verilen ekleri kullanabilirsiniz ve her bir ek sadece bir kez kullanılabilir. Sadece üretilen kelimeyi çıktı olarak verin.

Örnek 1:

Kök: küçük

Ekler: ümüz, lüg, den ˘

Cevap: küçüklügümüzden ˘

Örnek 2:

Kök: sevgi

Ekler: in, li, m

Cevap:

Productivity task prompt [OOD root] Size Türkçe yeni bir kök, onun tanımlaması ve bir ek listesi (virgülle ayrılmı¸s) verilecek ve sizden bu kökten verilen tüm ekleri kullanarak dilbilgisel olarak dogru bir kelime üretmeniz is-˘ tenecek. Sadece verilen ekleri kullanabilirsiniz ve her bir ek sadece bir kez kullanılabilir. Sadece üretilen kelimeyi çıktı olarak verin.

Örnek 1:

Kök: nıtal

Tanım: nıtal Türkçe kal anlamına gelir.

Örnek 2:

Kök: rarcu

Tanım: rarcu Türkçe vurgu anlamına gelir.

Ekler: la, mı¸s

Cevap:

<table><tr><td rowspan="2">Models</td><td colspan="7">Number of morphemes (excl. root)</td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td></tr><tr><td>majority</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td></tr><tr><td>random</td><td>24.7 / 30.0</td><td>26.0 / 20.0</td><td>2.0 / 1.3</td><td>4.7 / 1.3</td><td>2.0 / 4.7</td><td>4.0 / 4.0</td><td>2.7 / 2.0</td></tr><tr><td>aya-23-8b</td><td>62.7 / 32.0</td><td>58.7 / 30.7</td><td>24.7 / 7.3</td><td>30.0 / 14.7</td><td>34.0 / 15.3</td><td>18.0 / 8.0</td><td>24.2 / 20.8</td></tr><tr><td>aya-23-35b</td><td>68.0 / 50.7</td><td>80.7 / 53.3</td><td>56.0 / 49.3</td><td>47.3 / 36.0</td><td>49.3 / 38.7</td><td>46.0 / 30.0</td><td>43.0 / 34.9</td></tr><tr><td>qwen-2.5-7b</td><td>51.3 / 40.7</td><td>66.7 / 46.0</td><td>34.0 /30.7</td><td>43.3 / 30.7</td><td>37.3 / 31.3</td><td>39.3 / 27.3</td><td>43.6 / 24.8</td></tr><tr><td>qwen-2.5-32b</td><td>71.3 / 40.7</td><td>67.1 / 54.4</td><td>72.0 /48.0</td><td>72.0 / 50.0</td><td>67.3 / 53.3</td><td>62.0 / 49.3</td><td>55.7 / 42.3</td></tr><tr><td>gemini-1.5-flash</td><td>80.7 / 31.3</td><td>75.3 / 34.7</td><td>65.3 / 34.7</td><td>69.3 / 26.7</td><td>64.7 / 27.3</td><td>68.0 / 34.7</td><td>71.8 / 43.6</td></tr><tr><td>gpt-4</td><td>88.0 /31.3</td><td>86.7 / 74.7</td><td>86.0 /57.3</td><td>78.0 / 56.7</td><td>68.7 / 46.0</td><td>64.7 / 50.7</td><td>64.4 / 43.0</td></tr><tr><td>human</td><td>100.0 / 100.0</td><td>100.0 / 95.0</td><td>100.0 / 100.0</td><td>95.0 / 100.0</td><td>100.0 / 100.0</td><td>95.0 / 90.0</td><td>80.0 / 100.0</td></tr></table>

Table 18: Morphological systematicity 5-shot ID / OOD coherence results for Turkish in English template for all examined models. 1-shot and 3-shot results can be found in Tables 12, 15 respectively.
<table><tr><td rowspan="2">Models</td><td colspan="6">Number of morphemes (excl. root)</td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td></tr><tr><td>majority random</td><td>0.0 / 0.0 100.0 / 100.0</td><td>0.0 / 0.0 43.8 / 67.5</td><td>0.0 / 0.0 18.8 / 10.0</td><td>0.0 / 0.0 7.5 / 10.0</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td></tr><tr><td>qwen-2.5-7b</td><td></td><td></td><td></td><td></td><td>5.0 / 5.0</td><td>1.2 / 3.8</td></tr><tr><td>qwen-2.5-32b</td><td>55.0 / 45.0</td><td>22.5 / 15.0</td><td>2.5 / 1.2</td><td>1.2 / 0.0</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td></tr><tr><td></td><td>72.5 / 71.2</td><td>41.2 / 32.5</td><td>18.8 / 8.8</td><td>2.5 / 2.5</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td></tr><tr><td>gemini-1.5-flash</td><td>75.0 / 73.8</td><td>38.8 / 36.2</td><td>12.5 / 8.8</td><td>6.2 / 1.2</td><td>2.5 / 3.8</td><td>0.0 / 0.0</td></tr><tr><td>gpt-4</td><td>81.2 / 86.2</td><td>76.2 / 58.8</td><td>26.2 / 21.2</td><td>15.0 / 6.2</td><td>18.8 / 8.8</td><td>8.8 / 7.5</td></tr></table>

Table 19: Morphological productivity 1-shot ID / OOD accuracy results for Finnish in English template for all examined models.

Systematicity task prompt [ID root] Systematicity task prompt [OOD root]   
Size Türkçe bir kök, bir ek listesi (virgülle Size Türkçe yeni bir kök, onun tanımlaması,   
ayrılmı¸s) ve bu ekleri kullanarak türetilmi¸s bir ek listesi (virgülle ayrılmı¸s) ve bu ekleri   
bir kelime verilecek. Sizden bu kelimenin kullanarak türetilmi¸s bir kelime verilecek.   
dilbilgisel olarak dogru olup olmadı˘ gını˘ Sizden bu kelimenin dilbilgisel olarak dogru˘   
belirlemeniz istenecek. Sadece Evet veya Hayır olup olmadıgını belirlemeniz istenecek. Sadece˘   
ile cevap verin. Evet veya Hayır ile cevap verin.   
Örnek 1: Örnek 1:   
Kök: küçük Kök: yivük   
Ekler: ümüz, lüg, den˘ Tanım: yivük Türkçe küçük anlamına gelir.   
Türetilmi¸s kelime: küçüklügümüzden˘ Ekler: den, lüg, ümüz˘   
Cevap: Evet Türetilmi¸s kelime: yivüklügümüzden˘   
Cevap: Evet   
Örnek 2:   
Kök: sahip Örnek 2:   
Ekler: iniz, dig, len˘ Kök: minlek   
Türetilmi¸s kelime: sahipdiginizlen˘ Tanım: minlek Türkçe gerçek anlamına gelir.   
Cevap: Ekler: le¸s, di, me   
Türetilmi¸s kelime: minlekle¸smedi   
Cevap:

<table><tr><td></td><td colspan="6">Number of morphemes (excl. root)</td></tr><tr><td>Models</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td></tr><tr><td>majority</td><td>33.3 / 33.3</td><td>33.3 / 33.3</td><td>44.4 / 44.4</td><td>44.4 / 44.4</td><td>44.4 / 44.4</td><td>44.4 / 44.4</td></tr><tr><td>random</td><td>41.7 / 37.1</td><td>42.1 / 46.7</td><td>42.3 / 42.7</td><td>40.7 / 41.6</td><td>42.5 / 42.2</td><td>45.0 / 44.4</td></tr><tr><td>qwen-2.5-7b</td><td>65.8 / 57.1</td><td>77.5 / 71.2</td><td>61.5 / 57.2</td><td>53.3 / 47.7</td><td>53.5 / 47.2</td><td>56.0 / 47.2</td></tr><tr><td>qwen-2.5-32b</td><td>39.2 / 34.2</td><td>58.8 / 41.7</td><td>56.6 / 49.3</td><td>55.4 / 46.5</td><td>52.2 / 45.1</td><td>49.7 / 45.1</td></tr><tr><td>gemini-1.5-flash</td><td>50.8 / 33.3</td><td>45.0 / 33.3</td><td>54.7 / 44.4</td><td>49.7 / 44.4</td><td>46.5 / 44.4</td><td>49.7 / 44.4</td></tr><tr><td>gpt-4</td><td>79.2 / 34.2</td><td>73.3 / 36.7</td><td>73.9 / 47.9</td><td>64.4 / 45.8</td><td>71.2 / 44.4</td><td>58.2 / 44.4</td></tr></table>

Table 20: Morphological systematicity 1-shot ID / OOD macro-F1 results for Finnish in English template for all examined models.

<table><tr><td rowspan="2">Models</td><td colspan="6">Number of morphemes (excl. root)</td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td></tr><tr><td>majority random</td><td>0.0 / 0.0 23.8 / 21.2</td><td>0.0 / 0.0 25.0 / 30.0</td><td>0.0 / 0.0 3.8 / 1.2</td><td>0.0 / 0.0 0.0 / 5.0</td><td>0.0 / 0.0 6.2 / 0.0</td><td>0.0 / 0.0 2.5 / 7.5</td></tr><tr><td>qwen-2.5-7b qwen-2.5-32b gemini-1.5-flash</td><td>50.0 / 41.2 11.2 / 3.8 28.7 / 0.0</td><td>67.5 / 57.5 38.8 / 12.5 17.5 / 0.0</td><td>26.2 / 17.5 22.5 / 8.8 16.2 / 0.0</td><td>12.5 / 5.0 18.8 / 3.8 10.0 / 0.0</td><td>15.0/ 5.0 13.8 / 1.2 3.8 / 0.0</td><td>16.2 / 5.0 8.8 / 1.2</td></tr></table>

Table 21: Morphological systematicity 1-shot ID / OOD coherence results for Finnish in English template for all examined models.

Productivity task prompt [ID root] (with Systematicity task prompt [ID root] (with   
context) context)   
Size Türkçe bir kök, bir ek listesi (virgülle Size Türkçe bir kök, bir ek listesi (virgülle   
ayrılmı¸s) ve bo¸sluklu (\_\_\_) bir cümle verilecek ayrılmı¸s), bo¸sluklu (\_\_\_) bir cümle ve bu   
ve sizden bo¸slugu doldurmak için bu kökten˘ ekleri kullanarak türetilmi¸s bir kelime verilecek.   
verilen tüm ekleri kullanarak dilbilgisel olarak Sizden bo¸slugu doldurmak için bu kelimenin˘   
dogru bir kelime üretmeniz istenecek. Sadece˘ dilbilgisel olarak dogru olup olmadı˘ gını˘   
verilen ekleri kullanabilirsiniz ve her bir ek belirlemeniz istenecek. Sadece Evet veya Hayır   
sadece bir kez kullanılabilir. Sadece üretilen ile cevap verin.   
kelimeyi çıktı olarak verin.   
Örnek 1:   
Örnek 1: Kök: karı¸s   
Kök: küçük Ekler: ma, sa, k, ydı   
Ekler: den, ümüz, lüg˘ Cümle: gerçek ¸su ki anlayamadıgımız ¸seylere˘   
Cümle: kalma bir oyuna dönü¸stürdük mucize deyip \_\_\_, bugünlere belki de hiç   
hayatımızı ula¸samayacaktık   
Cevap: küçüklügümüzden˘ Türetilmi¸s kelime: karı¸smasaydık   
Cevap: Evet   
Örnek 2:   
Kök: ilkokul Örnek 2:   
Ekler: da, m, ydı Kök: sanat   
Cümle: Ilk kez onun bir ¸siirini okuyabilme fır- Ekler: ı, çı, lar, ndan   
satı buldugumda, henüz daha \_\_\_ ve bu kadar˘ Cümle: tüm bu deneyimlerime ev sahipligi˘   
farklı bir tarzla kar¸sıla¸smak beni oldukça heye- yapan ülke ise dünyanın en ünlü ve en çok   
canlandırmı¸stı begenilen \_\_\_ biri olan van gogh’un do ˘ gup ˘   
Cevap: büyüdügü hollanda’dan ba¸ska bir yer de˘ gil˘   
Türetilmi¸s kelime: sanatçılarndanı   
Cevap:

<table><tr><td rowspan="2">Models</td><td colspan="6">Number of morphemes (excl. root)</td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td></tr><tr><td>majority random</td><td>0.0 / 0.0 100.0 / 100.0</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td></tr><tr><td></td><td></td><td>56.2 / 46.2</td><td>12.5 / 17.5</td><td>7.5 / 12.5</td><td>13.8 / 3.8</td><td>3.8 / 1.2</td></tr><tr><td>qwen-2.5-7b</td><td>58.8 / 55.0</td><td>15.0 / 11.2</td><td>5.0 / 2.5</td><td>1.2 / 0.0</td><td>1.2 / 1.2</td><td>0.0 / 0.0</td></tr><tr><td>qwen-2.5-32b</td><td>70.0 / 67.5</td><td>41.2 / 36.2</td><td>12.5 / 10.0</td><td>3.8 / 2.5</td><td>1.2 / 2.5</td><td>2.5 / 0.0</td></tr><tr><td>gemini-1.5-flash</td><td>80.0 / 76.2</td><td>51.2 / 42.5</td><td>13.8 /7.5</td><td>5.0 / 1.2</td><td>6.2 / 6.2</td><td>5.0 / 3.8</td></tr><tr><td>gpt-4</td><td>81.2 / 90.0</td><td>73.8 / 62.5</td><td>32.5 / 20.0</td><td>20.0 / 13.8</td><td>26.2 / 11.2</td><td>10.0 / 12.5</td></tr></table>

Table 22: Morphological productivity 3-shot ID / OOD accuracy results for Finnish in English template for all examined models.
<table><tr><td rowspan="2">Models</td><td colspan="6">Number of morphemes (excl. root)</td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td></tr><tr><td rowspan="2">majority random</td><td>33.3 / 33.3</td><td>33.3 / 33.3</td><td>44.4 / 44.4</td><td>44.4 / 44.4</td><td>44.4 / 44.4</td><td>44.4 / 44.4</td></tr><tr><td>40.0 / 43.8</td><td>43.8 / 37.9</td><td>45.9 / 48.5</td><td>46.1 / 42.9</td><td>44.7 / 45.3</td><td>39.3 / 44.0</td></tr><tr><td>qwen-2.5-7b</td><td>62.9 / 56.7</td><td>81.2 / 77.1</td><td>67.2 / 55.4</td><td>62.8 / 52.5</td><td>57.7 / 52.0</td><td>60.4 / 50.4</td></tr><tr><td>qwen-2.5-32b</td><td>52.9 / 37.5</td><td>70.0 / 57.5</td><td>71.6 / 59.4</td><td>66.3 / 54.7</td><td>65.7 / 56.1</td><td>68.7 / 62.8</td></tr><tr><td>gemini-1.5-flash</td><td>84.2 / 50.8</td><td>70.4 /52.1</td><td>72.1 / 55.1</td><td>64.9 / 48.8</td><td>66.8 / 46.4</td><td>69.0 / 48.6</td></tr><tr><td>gpt-4</td><td>81.7 / 52.5</td><td>87.9 / 65.8</td><td>84.4 / 67.7</td><td>83.0 / 68.9</td><td>78.4 / 69.7</td><td>83.1 / 69.2</td></tr></table>

Table 23: Morphological systematicity 3-shot ID / OOD macro-F1 results for Finnish in English template for all examined models.

## F.3 Templates in Finnish

Productivity task prompt [ID root]   
Sinulle annetaan sanan sananvartalo ja luettelo   
pilkulla erotettuja päätteitä kielellä suomi.   
Tehtäväsi on luoda tästä juuresta kieliopillisesti   
oikea sana käyttämällä kaikkia annettuja   
päätteitä. Voit käyttää vain annettuja päätteitä   
ja kutakin päätettä vain kerran. Vastaa vain   
luodulla sanalla.   
Esimerkki 1:   
Sananvartalo: markiise   
Päätteet: j, a   
Vastaus: markiiseja   
Esimerkki 2:   
Sananvartalo: kasvattamis   
Päätteet: si, ta   
Vastaus:

Productivity task prompt [OOD root]   
Sinulle annetaan uusi sananvartalo, sen   
määritelmä sekä pilkulla eroteltu luettelo   
päätteitä kielellä suomi. Tehtäväsi on luoda   
juuresta kieliopillisesti oikea sana käyttämällä   
kaikkia annettuja päätteitä. Käyttä vain   
annettuja päätteitä ja kutakin päätettä vain   
kerran. Vastaa vain luodulla sanalla.   
Esimerkki 1:   
Sananvartalo: seloks   
Määritelmä: seloks tarkoittaa petoks kielellä   
suomi.   
Päätteet: ne, en, i   
Vastaus: seloksineen   
Esimerkki 2:   
Sananvartalo: osivma   
Määritelmä: osivma tarkoittaa ohitta kielellä   
suomi.   
Päätteet: han, ko, a   
Vastaus:

<table><tr><td></td><td colspan="6">Number of morphemes (excl. root)</td></tr><tr><td>Models</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td></tr><tr><td>majority</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td></tr><tr><td>random</td><td>26.2 / 26.2</td><td>27.5 / 18.8</td><td>5.0 / 8.8</td><td>5.0 /2.5</td><td>3.8 / 6.2</td><td>2.5 / 2.5</td></tr><tr><td>qwen-2.5-7b</td><td>45.0 / 41.2</td><td>72.5 / 66.2</td><td>30.0 / 15.0</td><td>27.5 / 11.2</td><td>22.5 / 12.5</td><td>17.5 / 8.8</td></tr><tr><td>qwen-2.5-32b</td><td>31.2 / 8.8</td><td>56.2 / 36.2</td><td>45.0 / 25.0</td><td>35.0 / 16.2</td><td>31.2 / 17.5</td><td>40.0 / 28.7</td></tr><tr><td>gemini-1.5-flash</td><td>76.2 / 28.7</td><td>56.2 / 28.7</td><td>43.8 / 16.2</td><td>36.2 / 7.5</td><td>37.5 / 3.8</td><td>38.8 / 7.5</td></tr><tr><td>gpt-4</td><td>72.5 / 32.5</td><td>82.5 / 48.8</td><td>63.7 / 38.8</td><td>61.3 / 38.8</td><td>53.8 / 41.2</td><td>58.8 / 38.8</td></tr></table>

Table 24: Morphological systematicity 3-shot ID / OOD coherence results for Finnish in English template for all examined models.

<table><tr><td rowspan="2">Models</td><td colspan="6">Number of morphemes (excl. root)</td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td></tr><tr><td>majority</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td></tr><tr><td>random</td><td>100.0 / 100.0</td><td>50.0 / 46.2</td><td>13.8 / 17.5</td><td>7.5 / 11.2</td><td>6.2 / 2.5</td><td>1.2 / 3.8</td></tr><tr><td>qwen-2.5-7b</td><td>62.5 / 60.0</td><td>26.2 / 17.5</td><td>3.8 / 6.2</td><td>1.2 / 1.2</td><td>1.2 / 1.2</td><td>1.2 / 0.0</td></tr><tr><td>qwen-2.5-32b</td><td>72.5 / 77.5</td><td>43.8 / 33.8</td><td>10.0 / 11.2</td><td>3.8 / 2.5</td><td>1.2 / 1.2</td><td>2.5 / 1.2</td></tr><tr><td>gemini-1.5-flash</td><td>81.2 / 73.8</td><td>51.2 / 45.0</td><td>17.5 / 10.0</td><td>5.0 /3.8</td><td>8.8 / 7.5</td><td>5.0 /3.8</td></tr><tr><td>gpt-4</td><td>83.8 / 90.0</td><td>73.8 / 66.2</td><td>40.0 / 23.8</td><td>22.5 / 5.0</td><td>30.0 / 12.5</td><td>15.0 / 8.8</td></tr><tr><td>human</td><td>90.0 / 95.0</td><td>90.0 / 90.0</td><td>100.0 / 95.0</td><td>80.0 / 80.0</td><td>60.0 / 80.0</td><td>90.0 / 60.0</td></tr></table>

Table 25: Morphological productivity 5-shot ID / OOD accuracy results for Finnish in English template for all examined models. 1-shot and 3-shot results can be found in Tables 19, 22 respectively.

Systematicity task prompt [ID root] Systematicity task prompt [OOD root]   
Sinulle annetaan sananvartalo, pilkulla eroteltu Sinulle annetaan uusi sananvartalo, sen   
luettelo päätteistä sekä annettuja päätteitä määritelmä sekä pilkulla eroteltu luettelo   
käyttämällä vartalosta johdettu sana kielellä päätteistä sekä uusi sana kielellä suomi, joka on   
suomi. Tehtäväsi on selvittää, onko johdettu johdettu annetusta sananvartalosta annettujen   
sana kieliopillisesti oikein. Vastaa vain Kyllä päätteiden avulla. Tehtäväsi on selvittää, onko   
tai Ei. johdettu sana kieliopillisesti oikein. Vastaa vain   
Kyllä tai Ei.   
Esimerkki 1:   
Sananvartalo: palauttaminen Esimerkki 1:   
Päätteet: n, mi, elee Sananvartalo: sätletjimsä   
Johdettu sana: mieleenpalauttaminen Määritelmä: sätletjimsä tarkoittaa järjestelmä   
Vastaus: Kyllä kielellä suomi.   
Päätteet: laadu, hallinta, n, n   
Esimerkki 2: Johdettu sana: laadunhallintasätletjimsän   
Sananvartalo: näkyv Vastaus: Kyllä   
Päätteet: imp, in, i   
Johdettu sana: näkyvimpiin Esimerkki 2:   
Vastaus: Sananvartalo: olanajke   
Määritelmä: olanajke tarkoittaa olosuhte   
kielellä suomi.   
Päätteet: i, kuvaus, an, lta   
Johdettu sana: kuvausolanajkeanilta   
Vastaus:

<table><tr><td></td><td colspan="6">Number of morphemes (excl. root)</td></tr><tr><td>Models</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td></tr><tr><td>majority</td><td>33.3 / 33.3</td><td>33.3 / 33.3</td><td>44.4 / 44.4</td><td>44.4 / 44.4</td><td>44.4 / 44.4</td><td>44.4 / 44.4</td></tr><tr><td>random</td><td>43.8 / 42.5</td><td>40.4 / 42.9</td><td>43.7 / 39.8</td><td>41.6 / 47.3</td><td>39.4 / 42.4</td><td>45.7 / 40.0</td></tr><tr><td>qwen-2.5-7b</td><td>65.8 / 55.4</td><td>86.7 / 75.4</td><td>65.0 / 58.1</td><td>66.2 / 54.2</td><td>62.5 / 57.7</td><td>63.5 / 55.9</td></tr><tr><td>qwen-2.5-32b</td><td>57.1 / 40.0</td><td>72.9 / 66.7</td><td>71.7 / 69.8</td><td>72.8 / 63.6</td><td>70.7 / 71.1</td><td>68.6 / 62.3</td></tr><tr><td>gemini-1.5-flash</td><td>84.2 / 49.6</td><td>80.8 / 65.4</td><td>79.6 / 62.0</td><td>75.4 / 52.8</td><td>74.9 / 53.5</td><td>71.4/57.7</td></tr><tr><td>gpt-4</td><td>84.2 / 61.3</td><td>89.2 / 78.8</td><td>84.2 / 83.8</td><td>88.0 / 76.7</td><td>83.0 / 76.5</td><td>82.7 / 71.6</td></tr><tr><td>human</td><td>73.3 / 86.7</td><td>93.3 / 100.0</td><td>93.2 / 97.6</td><td>95.5 / 90.9</td><td>91.2 / 90.5</td><td>89.7 / 84.4</td></tr></table>

Table 26: Morphological systematicity 5-shot ID / OOD macro-F1 results for Finnish in English template for all examined models. 1-shot and 3-shot results can be found in Tables 20, 23 respectively.
<table><tr><td rowspan="2">Models</td><td colspan="6">Number of morphemes (excl. root)</td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td></tr><tr><td>majority</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td></tr><tr><td>random</td><td>26.2 / 26.2</td><td>25.0 / 23.8</td><td>3.8 / 0.0</td><td>1.2 / 5.0</td><td>2.5 / 1.2</td><td>6.2 / 1.2</td></tr><tr><td>qwen-2.5-7b</td><td>48.8 / 36.2</td><td>81.2 / 65.0</td><td>28.7 / 16.2</td><td>31.2 / 15.0</td><td>22.5 / 18.8</td><td>22.5 / 15.0</td></tr><tr><td>qwen-2.5-32b</td><td>37.5 / 13.8</td><td>61.3 / 51.2</td><td>41.2 / 38.8</td><td>46.2 / 28.7</td><td>33.8 / 41.2</td><td>35.0 / 25.0</td></tr><tr><td>gemini-1.5-flash</td><td>76.2 / 28.7</td><td>71.2 / 48.8</td><td>47.5 / 26.2</td><td>43.8 / 12.5</td><td>42.5 / 12.5</td><td>32.5 / 21.2</td></tr><tr><td>gpt-4</td><td>76.2 / 45.0</td><td>83.8 / 68.8</td><td>56.2 / 61.3</td><td>68.8 / 48.8</td><td>57.5 / 40.0</td><td>55.0 / 40.0</td></tr><tr><td>human</td><td>60.0 / 80.0</td><td>90.0 / 100.0</td><td>75.0 / 90.0</td><td>85.0 / 75.0</td><td>70.0 / 60.0</td><td>70.0 / 50.0</td></tr></table>

Table 27: Morphological systematicity 5-shot ID / OOD coherence results for Finnish in English template for all examined models. 1-shot and 3-shot results can be found in Tables 21, 24 respectively.

Allaolevassa lauseessa (kirjoitettu kielellä suomi) on tyhjä kohta (\_\_\_) joka tulee täyttää kieliopillisesti oikealla sanalla. Alla on myös sananvartalo sekä pilkulla eroteltu luettelo päätteistä. Tehtäväsi on käyttää vartaloa sekä päätteitä ja johtaa niistä kieliopillisesti oikein taivutetu sana joka sopii tyhjään kohtaan lausessaa asiayhteys/konteksti huomioonottaen. Käytä jokaista päätettä vain kerran. Vastaa vain generoidulla sanalla, älä sano mitään muuta.

Esimerkki 1:

Sananvartalo: markiise

Päätteet: a, j

Lause: \_\_\_ saatavana yksivärisinä, raidallisina ja voit myös valita haluatko markiisisi veivivai sähkökäyttöisenä.

Vastaus: markiiseja

Esimerkki 2:

Sananvartalo: suhteutet

Päätteet: na, tu

Lause: \_\_\_ väkilukuun, suomessa on enemmän

metsää kuin missään muussa euroopan maassa.

Allaolevassa lauseessa on tyhjä kohta (\_\_\_) joka tulee täyttää kieliopillisesti oikealla sanalla. Alla on myös sananvartalo, pilkulla eroteltu luettelo päätteistä sekä niitä käyttäen annetusta vartalosta johdettu sana kielellä suomi. Tehtäväsi on päätellä, onko johdettu sana kieliopillisesti oikein, jos sen asettaa lauseen tyhjään kohtaan eli onko sana kieliopillisesti oikein taivutetu asiayhteys/konteksti huomioonottaen. Vastaa joko Kyllä tai Ei.

Sananvartalo: petoks

Lause: hän paljasti koko korruptoituneen jär-

Johdettu sana: petoksineen

Esimerkki 2:

Sananvartalo: kannatta

Päätteet: isi, han, ko

Lause: \_\_\_ minun opiskella suomea?

Johdettu sana: kannattakoisihan

Vastaus:

<table><tr><td>Models</td><td colspan="2">Morph. Productivity (accuracy)</td><td colspan="2">Morph. Systematicity (macro-F1)</td><td colspan="2">Morph. Systematicity (coherence)</td></tr><tr><td></td><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td></tr><tr><td>aya-23-8b</td><td>12.8 / 8.4</td><td>8.8 / 5.3</td><td>62.0 / 62.4</td><td>53.9 / 44.3</td><td>27.9 / 31.9</td><td>19.1 / 4.5</td></tr><tr><td>aya-23-35b</td><td>17.4 / 14.6</td><td>14.6 / 11.7</td><td>69.9 / 79.6</td><td>64.6 / 48.9</td><td>36.8 / 59.2</td><td>29.2 / 12.2</td></tr><tr><td>qwen-2.5-7b</td><td>15.0 / 9.3</td><td>13.2 / 10.0</td><td>71.1 / 69.9</td><td>65.7 / 59.1</td><td>40.5 / 38.1</td><td>33.5 / 23.6</td></tr><tr><td>qwen-2.5-32b</td><td>22.6 / 22.9</td><td>21.7 / 20.4</td><td>77.3 / 78.5</td><td>53.1 / 44.5</td><td>56.7 / 60.0</td><td>18.5 / 5.7</td></tr><tr><td>gemini-1.5-flash</td><td>28.8 / 21.8</td><td>24.9 / 19.3</td><td>60.8 / 45.1</td><td>41.4 /41.2</td><td>32.2 / 5.9</td><td>0.4 / 0.0</td></tr><tr><td>gpt-4</td><td>49.0 / 48.5</td><td>36.7 / 38.1</td><td>85.5 / 76.3</td><td>61.9 / 50.4</td><td>71.4 / 58.6</td><td>33.5 / 15.7</td></tr></table>

Table 28: 1-shot English / Turkish template results for Turkish for all examined models across tasks.

<table><tr><td>Models</td><td>Morph. Productivity (accuracy) ID</td><td>OOD</td><td>Morph. Systematicity (macro-F1) ID</td><td>OOD</td><td>Morph. Systematicity (coherence) ID</td><td>OOD</td></tr><tr><td>aya-23-8b</td><td>13.7 / 7.0</td><td>11.5 / 9.6</td><td>64.6 / 69.0</td><td>49.3 / 46.4</td><td>31.4 / 38.7</td><td>15.7 / 8.0</td></tr><tr><td>aya-23-35b</td><td>19.8 / 17.1</td><td>17.7 / 16.9</td><td>80.1 / 81.6</td><td>71.0 /57.1</td><td>52.6 / 59.6</td><td>39.9 / 24.7</td></tr><tr><td>qwen-2.5-7b</td><td>14.9 / 11.2</td><td>12.9 / 11.4</td><td>73.6 / 73.7</td><td>66.8 / 61.4</td><td>44.3 / 44.8</td><td>33.9 / 28.2</td></tr><tr><td>qwen-2.5-32b</td><td>23.7 / 20.7</td><td>21.8 / 19.9</td><td>84.7 / 86.1</td><td>71.3 / 65.2</td><td>66.3 / 70.5</td><td>45.7 / 36.8</td></tr><tr><td>gemini-1.5-flash</td><td>30.5 / 23.0</td><td>25.7 / 20.7</td><td>80.8 / 56.9</td><td>52.8 / 41.2</td><td>63.6 / 25.3</td><td>19.3 / 0.0</td></tr><tr><td>gpt-4</td><td>52.1 / 51.5</td><td>40.5 / 40.6</td><td>90.2 / 88.6</td><td>77.7 / 66.3</td><td>76.8 / 76.3</td><td>55.9 / 39.9</td></tr></table>

Table 29: 3-shot English / Turkish template results for Turkish for all examined models across tasks.

<table><tr><td>Models</td><td>Morph. Productivity (accuracy) ID</td><td>OOD</td><td>Morph. Systematicity (macro-F1) ID</td><td>OOD</td><td>ID</td><td>Morph. Systematicity (coherence) OOD</td></tr><tr><td>aya-23-8b</td><td>13.3 / 8.6</td><td>12.3 / 8.1</td><td>67.5 / 62.0</td><td>51.5 / 47.9</td><td>36.0 / 30.6</td><td>18.4 / 15.8</td></tr><tr><td>aya-23-35b</td><td>21.0 / 17.1</td><td>19.3 / 18.0</td><td>81.8 / 81.7</td><td>72.1 / 64.3</td><td>55.8 / 57.7</td><td>41.8 / 30.4</td></tr><tr><td>qwen-2.5-7b</td><td>15.8 / 13.0</td><td>12.9 / 12.3</td><td>74.6 / 73.5</td><td>66.0 / 65.2</td><td>45.1 / 42.4</td><td>33.1 / 32.3</td></tr><tr><td>qwen-2.5-32b</td><td>24.1 / 23.3</td><td>21.8 / 21.9</td><td>85.9 / 86.9</td><td>75.3 / 70.7</td><td>66.8 / 70.5</td><td>48.3 / 44.2</td></tr><tr><td>gemini-1.5-flash</td><td>30.7 / 25.8</td><td>25.1 / 22.1</td><td>85.4 / 61.6</td><td>62.1 / 41.6</td><td>70.7 / 32.9</td><td>33.3 / 0.7</td></tr><tr><td>gpt-4</td><td>54.2 / 53.1</td><td>43.9 / 40.7</td><td>91.6 / 92.7</td><td>78.8 / 76.3</td><td>76.6 / 82.1</td><td>51.4 / 51.9</td></tr></table>

Table 30: 5-shot English / Turkish template results for Turkish for all examined models across tasks. Results for 1-shot and 3-shot can be found in Tables 28 and 29.

<table><tr><td>Models</td><td colspan="2">Morph. Productivity (accuracy)</td><td colspan="2">Morph. Systematicity (macro-F1)</td><td colspan="2">Morph. Systematicity (coherence)</td></tr><tr><td></td><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td></tr><tr><td>qwen-2.5-7b</td><td>13.5 / 7.9</td><td>10.2 / 10.8</td><td>61.3 / 56.8</td><td>54.6 / 41.2</td><td>31.2 / 23.8</td><td>21.9 / 2.1</td></tr><tr><td>qwen-2.5-32b</td><td>22.5 / 16.9</td><td>19.2 / 11.9</td><td>52.0 / 54.5</td><td>43.6 / 44.4</td><td>19.0 / 21.9</td><td>5.2 / 5.8</td></tr><tr><td>gemini-1.5-flash</td><td>22.5 / 25.2</td><td>20.6 / 21.2</td><td>49.4 / 50.2</td><td>40.7 / 40.9</td><td>14.2 / 15.6</td><td>0.0 / 0.2</td></tr><tr><td>gpt-4</td><td>37.7 / 37.3</td><td>31.5 / 27.7</td><td>70.0 / 59.6</td><td>42.2 / 41.6</td><td>47.5 / 30.4</td><td>2.7 / 1.2</td></tr></table>

Table 31: 1-shot English / Finnish template results for Finnish for all examined models across tasks.

<table><tr><td rowspan="2">Models</td><td colspan="2">Morph. Productivity (accuracy)</td><td colspan="2">Morph. Systematicity (macro-F1)</td><td colspan="2">Morph. Systematicity (coherence)</td></tr><tr><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td></tr><tr><td>qwen-2.5-7b</td><td>13.5 / 10.4</td><td>11.7 / 10.2</td><td>65.4 / 65.0</td><td>57.3 / 55.7</td><td>35.8 / 32.5</td><td>25.8 / 22.5</td></tr><tr><td>qwen-2.5-32b</td><td>21.9 / 21.0</td><td>19.8 / 17.3</td><td>65.9 / 60.9</td><td>54.7 / 56.4</td><td>39.8 / 31.7</td><td>22.1 / 24.8</td></tr><tr><td>gemini-1.5-flash</td><td>26.9 / 24.6</td><td>22.9 / 19.2</td><td>71.2 / 74.7</td><td>50.3 / 48.0</td><td>48.1 / 51.7</td><td>15.4 / 11.9</td></tr><tr><td>gpt-4</td><td>40.6 / 40.8</td><td>35.0 / 32.7</td><td>83.1 / 72.0</td><td>65.6 / 45.9</td><td>65.4 / 51.0</td><td>39.8 / 8.5</td></tr></table>

Table 32: 3-shot English / Finnish template results for Finnish for all examined models across tasks.

<table><tr><td rowspan="2">Models</td><td colspan="2">Morph. Productivity (accuracy)</td><td colspan="2">Morph. Systematicity (macro-F1)</td><td colspan="2">Morph. Systematicity (coherence)</td></tr><tr><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td></tr><tr><td>qwen-2.5-7b</td><td>16.0 / 13.1</td><td>14.4 / 13.3</td><td>68.3 / 66.6</td><td>59.4 / 60.1</td><td>39.2 / 35.2</td><td>27.7 / 28.1</td></tr><tr><td>qwen-2.5-32b</td><td>22.3 / 20.6</td><td>21.3 / 20.6</td><td>69.0 / 66.5</td><td>62.2 / 59.1</td><td>42.5 / 39.4</td><td>33.1 / 29.0</td></tr><tr><td>gemini-1.5-flash</td><td>28.1 / 28.1</td><td>24.0 / 21.7</td><td>77.7 / 76.2</td><td>56.8 / 59.2</td><td>52.3 / 51.9</td><td>25.0 / 28.1</td></tr><tr><td>gpt-4</td><td>44.2 / 42.9</td><td>34.4 / 34.4</td><td>85.2 / 81.1</td><td>74.8 / 57.2</td><td>66.2 / 65.0</td><td>50.6 / 26.2</td></tr><tr><td>aya-23-8b</td><td>12.8 / 13.1</td><td>8.8 / 6.3</td><td>62.0 / 49.8</td><td>53.9 / 40.3</td><td>27.9 / 16.8</td><td>19.1 / 7.5</td></tr><tr><td>aya-23-35b</td><td>17.4 / 21.0</td><td>14.6 / 13.1</td><td>69.9 / 66.3</td><td>64.6 / 55.8</td><td>36.8 / 37.3</td><td>29.2 / 22.0</td></tr><tr><td>qwen-2.5-7b</td><td>15.0 / 13.0</td><td>13.2 / 9.8</td><td>71.1 /61.9</td><td>65.7 / 53.6</td><td>40.5 / 29.8</td><td>33.5 / 19.7</td></tr><tr><td>qwen-2.5-32b</td><td>22.6 / 23.7</td><td>21.7 / 19.3</td><td>77.3 / 44.0</td><td>53.1 / 41.4</td><td>56.7 / 4.7</td><td>18.5 / 0.3</td></tr><tr><td>gemini-1.5-flash</td><td>28.8 / 36.1</td><td>24.9 / 24.1</td><td>60.8 / 57.4</td><td>41.4 / 43.5</td><td>32.2 / 26.2</td><td>0.4 / 3.8</td></tr><tr><td>gpt-4</td><td>49.0 / 59.6</td><td>36.7 / 39.9</td><td>85.5 / 71.0</td><td>61.9 / 54.5</td><td>71.4 / 49.6</td><td>33.5 / 21.4</td></tr></table>

Table 33: 5-shot English / Finnish template results for Finnish for all examined models across tasks. 1-shot and 3-shot results can be found in Tables 31 and 32.

Table 34: 1-shot No context / With context results for Turkish in English template for all examined models across tasks.
<table><tr><td>Models</td><td>Morph. Productivity (accuracy) ID</td><td>OOD</td><td>Morph. Systematicity (macro-F1) ID</td><td>OOD</td><td>Morph. Systematicity (coherence) ID</td><td>OOD</td></tr><tr><td>aya-23-8b</td><td>13.7 / 13.6</td><td>11.5 / 9.6</td><td>64.6 / 50.3</td><td>49.3 / 42.5</td><td>31.4 / 16.8</td><td>15.7 / 9.6</td></tr><tr><td>aya-23-35b</td><td>19.8 / 24.3</td><td>17.7 / 17.8</td><td>80.1 / 66.3</td><td>71.0 /51.0</td><td>52.6 / 34.9</td><td>39.9 / 16.1</td></tr><tr><td>qwen-2.5-7b</td><td>14.9 / 13.5</td><td>12.9 / 9.8</td><td>73.6 / 63.4</td><td>66.8 / 59.3</td><td>44.3 / 30.3</td><td>33.9 / 25.2</td></tr><tr><td>qwen-2.5-32b</td><td>23.7 / 24.2</td><td>21.8 / 19.3</td><td>84.7 / 61.0</td><td>71.3 / 47.3</td><td>66.3 / 33.2</td><td>45.7 / 10.2</td></tr><tr><td>gemini-1.5-flash</td><td>30.5 / 38.4</td><td>25.7 / 26.5</td><td>80.8 / 75.6</td><td>52.8 / 60.1</td><td>63.6 / 49.7</td><td>19.3 / 27.2</td></tr><tr><td>gpt-4</td><td>52.1 / 59.8</td><td>40.5 / 45.6</td><td>90.2 / 85.2</td><td>77.7 / 67.1</td><td>76.8 / 71.9</td><td>55.9 / 41.5</td></tr></table>

Table 35: 3-shot No context / With context results for Turkish in English template for all examined models across tasks.
<table><tr><td>Models</td><td>Morph. Productivity (accuracy) ID</td><td>OOD</td><td>Morph. Systematicity (macro-F1) ID</td><td>OOD</td><td>Morph. Systematicity (coherence) ID</td><td>OOD</td></tr><tr><td>aya-23-8b</td><td>13.3 / 14.8</td><td>12.3 / 9.9</td><td>67.5 / 51.4</td><td>51.5 / 42.9</td><td>36.0 / 19.3</td><td>18.4 / 10.6</td></tr><tr><td>aya-23-35b</td><td>21.0 / 26.5</td><td>19.3 / 18.7</td><td>81.8 / 66.2</td><td>72.1 / 47.6</td><td>55.8 / 35.4</td><td>41.8 / 14.3</td></tr><tr><td>qwen-2.5-7b</td><td>15.8 / 15.0</td><td>12.9 / 11.4</td><td>74.6 / 63.9</td><td>66.0 / 58.8</td><td>45.1 / 30.3</td><td>33.1 / 23.1</td></tr><tr><td>qwen-2.5-32b</td><td>24.1 / 26.2</td><td>21.8 / 21.4</td><td>85.9 / 68.4</td><td>75.3 / 52.0</td><td>66.8 / 44.8</td><td>48.3 / 18.1</td></tr><tr><td>gemini-1.5-flash</td><td>30.7 / 41.7</td><td>25.1 / 28.7</td><td>85.4 / 77.6</td><td>62.1 / 65.3</td><td>70.7 / 51.6</td><td>33.3 / 32.2</td></tr><tr><td>gpt-4</td><td>54.2 / 60.0</td><td>43.9 / 46.3</td><td>91.6 / 88.4</td><td>78.8 / 72.1</td><td>76.6 / 77.2</td><td>51.4 / 48.4</td></tr></table>

Table 36: 5-shot No context / With context results for Turkish in English template for all examined models across tasks. 1-shot and 3-shot results can be found in Tables 34 and 35.
<table><tr><td rowspan="2">Models</td><td colspan="2">Morph. Productivity (accuracy)</td><td colspan="2">Morph. Systematicity (macro-F1)</td><td colspan="2">Morph. Systematicity (coherence)</td></tr><tr><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td></tr><tr><td>qwen-2.5-7b</td><td>13.5 / 10.6</td><td>10.2/ 9.6</td><td>61.3 / 61.5</td><td>54.6 / 53.2</td><td>31.2 / 30.4</td><td>21.9 / 19.0</td></tr><tr><td>qwen-2.5-32b</td><td>22.5 / 22.3</td><td>19.2 / 17.3</td><td>52.0 / 43.6</td><td>43.6 / 40.7</td><td>19.0 / 4.8</td><td>5.2 / 0.0</td></tr><tr><td>gemini-1.5-flash</td><td>22.5 / 26.7</td><td>20.6 / 21.9</td><td>49.4 / 54.7</td><td>40.7 / 41.1</td><td>14.2 / 22.1</td><td>0.0 / 0.6</td></tr><tr><td>gpt-4</td><td>37.7 / 46.7</td><td>31.5 / 31.7</td><td>70.0 / 74.6</td><td>42.2 / 49.6</td><td>47.5 / 53.1</td><td>2.7 / 14.4</td></tr></table>

Table 37: 1-shot No context / With context results for Finnish in English template for all examined models across tasks.
<table><tr><td rowspan="2">Models</td><td colspan="2">Morph. Productivity (accuracy)</td><td colspan="2">Morph. Systematicity (macro-F1)</td><td colspan="2">Morph. Systematicity (coherence)</td></tr><tr><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td></tr><tr><td>qwen-2.5-7b</td><td>13.5 / 11.0</td><td>11.7 / 10.8</td><td>65.4 / 63.5</td><td>57.3 / 55.3</td><td>35.8 / 34.0</td><td>25.8 / 21.5</td></tr><tr><td>qwen-2.5-32b</td><td>21.9 / 22.5</td><td>19.8 / 16.0</td><td>65.9 / 62.0</td><td>54.7 / 47.3</td><td>39.8 / 33.8</td><td>22.1 / 10.6</td></tr><tr><td>gemini-1.5-flash</td><td>26.9 / 32.3</td><td>22.9 / 23.3</td><td>71.2 / 72.3</td><td>50.3 / 57.3</td><td>48.1 / 43.3</td><td>15.4 / 23.3</td></tr><tr><td>gpt-4</td><td>40.6 / 52.3</td><td>35.0 / 33.5</td><td>83.1 / 83.8</td><td>65.6 / 65.2</td><td>65.4 / 67.9</td><td>39.8 / 39.2</td></tr></table>

Table 38: 3-shot No context / With context results for Finnish in English template for all examined models across tasks.
<table><tr><td rowspan="2">Models</td><td colspan="2">Morph. Productivity (accuracy)</td><td colspan="2">Morph. Systematicity (macro-F1)</td><td colspan="2">Morph. Systematicity (coherence)</td></tr><tr><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td></tr><tr><td>qwen-2.5-7b</td><td>16.0 / 13.5</td><td>14.4 / 12.7</td><td>68.3 / 63.0</td><td>59.4 / 54.5</td><td>39.2 / 32.3</td><td>27.7 / 21.9</td></tr><tr><td>qwen-2.5-32b</td><td>22.3 / 23.8</td><td>21.3 / 20.0</td><td>69.0 / 65.5</td><td>62.2 / 54.5</td><td>42.5 / 37.9</td><td>33.1 / 20.6</td></tr><tr><td>gemini-1.5-flash</td><td>28.1 / 32.7</td><td>24.0 / 24.4</td><td>77.7 / 72.4</td><td>56.8 / 53.8</td><td>52.3 / 42.7</td><td>25.0 / 19.8</td></tr><tr><td>gpt-4</td><td>44.2 / 53.1</td><td>34.4 / 32.3</td><td>85.2 / 85.8</td><td>74.8 / 68.5</td><td>66.2 / 68.1</td><td>50.6 / 41.5</td></tr></table>

Table 39: 5-shot No context / With context results for Finnish in English template for all examined models across tasks. 1-shot and 3-shot results can be found in Tables 37 and 38.

<table><tr><td rowspan="2">Models</td><td colspan="7">Number of morphemes (excl. root)</td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td></tr><tr><td>gpt-4</td><td>95.3 / 84.0</td><td>80.7 / 67.8</td><td>62.7 / 52.7</td><td>43.8 / 42.1</td><td>27.3 / 32.0</td><td>19.3 / 26.5</td><td>13.8 / 13.9</td></tr></table>

Table 40: GPT-4 morphological productivity 1-shot morphologically aligned / tokenizer aligned accuracy results on the ID test set for Turkish in English template.
<table><tr><td rowspan="2">Models</td><td colspan="7">Number of morphemes (excl. root)</td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td></tr><tr><td>gpt-4</td><td>94.7 / 84.7</td><td>81.3 / 72.5</td><td>64.0 / 58.0</td><td>49.3 / 48.2</td><td>30.7 / 39.5</td><td>25.3 / 36.8</td><td>19.3 / 23.8</td></tr></table>

Table 41: GPT-4 morphological productivity 3-shot morphologically aligned / tokenizer aligned accuracy results on the ID test set for Turkish in English template.
<table><tr><td rowspan="2">Models</td><td colspan="7">Number of morphemes (excl. root)</td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td></tr><tr><td>gpt-4</td><td>96.0 / 88.0</td><td>85.3 / 69.8</td><td>66.0 / 64.0</td><td>43.7 / 44.8</td><td>40.0 / 42.2</td><td>28.0 / 32.4</td><td>20.6 / 25.7</td></tr></table>

Table 42: GPT-4 morphological productivity 5-shot morphologically aligned / tokenizer aligned accuracy results on the ID test set for Turkish in English template. 1-shot and 3-shot results can be found in Tables 40 and 41.
<table><tr><td rowspan="2">Models gpt-4</td><td colspan="2">Morph. Productivity (accuracy)</td><td colspan="2">Morph. Systematicity (macro-F1)</td><td colspan="2">Morph. Systematicity (coherence)</td></tr><tr><td>ID 54.2 / 36.4 / 46.8</td><td>OOD 43.9 / 31.2 / 45.8</td><td>ID 91.6 / 85.1 / 88.8</td><td>OOD 78.8 / 70.3 / 83.0</td><td>ID 76.6 / 63.8 / 72.5</td><td>OOD 51.4 /38.1 /61.1</td></tr></table>

Table 43: GPT-4 5-shot / 0-shot-cot / 5-shot-cot results for Turkish in English template across tasks.
<table><tr><td rowspan="2">Models</td><td colspan="2">Morph. Productivity (accuracy)</td><td colspan="2">Morph. Systematicity (macro-F1)</td><td colspan="2">Morph. Systematicity (coherence)</td></tr><tr><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td></tr><tr><td>aya-23-8b</td><td>12.8 / 14.2</td><td>8.8 / 9.4</td><td>62.0 / 62.0</td><td>53.9 / 53.5</td><td>27.9 / 28.0</td><td>19.1 / 19.4</td></tr><tr><td>aya-23-35b</td><td>17.4 / 22.3</td><td>14.6 / 17.9</td><td>69.9 / 69.4</td><td>64.6 / 64.3</td><td>36.8 / 36.2</td><td>29.2 / 28.2</td></tr><tr><td>qwen-2.5-7b</td><td>15.0 / 16.7</td><td>13.2 / 14.9</td><td>71.1 /73.1</td><td>65.7 / 67.8</td><td>40.5 / 42.7</td><td>33.5 / 36.5</td></tr><tr><td>qwen-2.5-32b</td><td>22.6 / 30.0</td><td>21.7 / 29.2</td><td>77.3 / 80.4</td><td>53.1 / 58.1</td><td>56.7 / 61.0</td><td>18.5 / 25.9</td></tr><tr><td>gemini-1.5-flash</td><td>28.8 / 37.2</td><td>24.9 / 32.1</td><td>60.8 / 65.4</td><td>41.4 /41.5</td><td>32.2 / 39.6</td><td>0.4 / 0.7</td></tr><tr><td>gpt-4</td><td>49.0 / 63.0</td><td>36.7 / 54.5</td><td>85.5 / 88.6</td><td>61.9 / 68.3</td><td>71.4 / 77.5</td><td>33.5 / 43.7</td></tr></table>

Table 44: 1-shot Shuffled / Correct morpheme order results for Turkish in English template for all examined models across tasks.
<table><tr><td rowspan="2">Models</td><td colspan="2">Morph. Productivity (accuracy)</td><td colspan="2">Morph. Systematicity (macro-F1)</td><td colspan="2">Morph. Systematicity (coherence)</td></tr><tr><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td></tr><tr><td>aya-23-8b</td><td>13.7 / 14.9</td><td>11.5 / 12.5</td><td>64.6 / 63.8</td><td>49.3 / 49.1</td><td>31.4 /30.7</td><td>15.7 / 16.4</td></tr><tr><td>aya-23-35b</td><td>19.8 / 25.7</td><td>17.7 / 23.7</td><td>80.1 / 80.2</td><td>71.0 / 73.0</td><td>52.6 / 52.2</td><td>39.9 / 41.9</td></tr><tr><td>qwen-2.5-7b</td><td>14.9 / 17.8</td><td>12.9 / 15.5</td><td>73.6 / 76.1</td><td>66.8 / 69.4</td><td>44.3 / 47.9</td><td>33.9 / 38.0</td></tr><tr><td>qwen-2.5-32b</td><td>23.7 / 30.7</td><td>21.8 / 32.1</td><td>84.7 / 86.9</td><td>71.3 / 75.7</td><td>66.3 / 70.3</td><td>45.7 / 53.0</td></tr><tr><td>gemini-1.5-flash</td><td>30.5 / 39.6</td><td>25.7 / 33.4</td><td>80.8 / 85.4</td><td>52.8 / 58.0</td><td>63.6 / 71.6</td><td>19.3 / 27.4</td></tr><tr><td>gpt-4</td><td>52.1 / 70.3</td><td>40.5 / 63.5</td><td>90.2 / 92.9</td><td>77.7 / 81.0</td><td>76.8 / 82.0</td><td>55.9 / 59.6</td></tr></table>

Table 45: 3-shot Shuffled / Correct morpheme order results for Turkish in English template for all examined models across tasks.
<table><tr><td>Models</td><td>Morph. Productivity (accuracy) ID</td><td>OOD</td><td>Morph. Systematicity (macro-F1) ID</td><td>OOD</td><td>Morph. Systematicity (coherence) ID</td><td>OOD</td></tr><tr><td>aya-23-8b</td><td>13.3 / 15.0</td><td>12.3 / 13.0</td><td>67.5 / 66.4</td><td>51.5 / 51.8</td><td>36.0 / 34.7</td><td>18.4 / 18.1</td></tr><tr><td>aya-23-35b</td><td>21.0 / 28.5</td><td>19.3 / 25.8</td><td>81.8 / 81.3</td><td>72.1 / 72.3</td><td>55.8 / 55.2</td><td>41.8 / 42.3</td></tr><tr><td>qwen-2.5-7b</td><td>15.8 / 19.2</td><td>12.9 / 16.9</td><td>74.6 / 76.4</td><td>66.0 / 68.5</td><td>45.1 / 47.0</td><td>33.1 / 35.7</td></tr><tr><td>qwen-2.5-32b</td><td>24.1 / 32.3</td><td>21.8 / 36.4</td><td>85.9 / 87.5</td><td>75.3 / 78.2</td><td>66.8 / 69.6</td><td>48.3 / 52.2</td></tr><tr><td>gemini-1.5-flash</td><td>30.7 / 43.3</td><td>25.1 / 35.1</td><td>85.4 / 88.6</td><td>62.1 / 66.1</td><td>70.7 / 74.5</td><td>33.3 / 39.5</td></tr><tr><td>gpt-4</td><td>54.2 / 73.0</td><td>43.9 / 66.7</td><td>91.6 /93.7</td><td>78.8 / 82.6</td><td>76.6 / 82.2</td><td>51.4 /58.3</td></tr><tr><td rowspan="2">Models</td><td colspan="2">Morph. Productivity (accuracy)</td><td colspan="2">Morph. Systematicity (macro-F1)</td><td colspan="2">Morph. Systematicity (coherence)</td></tr><tr><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td></tr><tr><td>qwen-2.5-7b</td><td>13.5 / 15.6</td><td>10.2 / 12.3</td><td>61.3 / 62.9</td><td>54.6 / 55.2</td><td>31.2 / 34.2</td><td>21.9 / 22.5</td></tr><tr><td>qwen-2.5-32b</td><td>22.5 / 24.8</td><td>19.2 / 23.1</td><td>52.0 / 53.1</td><td>43.6 / 44.0</td><td>19.0 / 20.4</td><td>5.2 / 5.8</td></tr><tr><td>gemini-1.5-flash</td><td>22.5 / 26.7</td><td>20.6 / 25.6</td><td>49.4 / 51.2</td><td>40.7 / 40.9</td><td>14.2 / 17.5</td><td>0.0 / 0.2</td></tr><tr><td>gpt-4</td><td>37.7 / 46.0</td><td>31.5 / 39.6</td><td>70.0 / 70.7</td><td>42.2 / 43.7</td><td>47.5 / 48.3</td><td>2.7 / 5.0</td></tr></table>

Table 46: 5-shot Shuffled / Correct morpheme order results for Turkish in English template for all examined models across tasks. 1-shot and 3-shot results can be found in Tables 44 and 45.

Table 47: 1-shot Shuffled / Correct morpheme order results for Finnish in English template for all examined models across tasks.
<table><tr><td>Models</td><td>Morph. Productivity (accuracy) ID</td><td>OOD</td><td>Morph. Systematicity (macro-F1) ID</td><td>OOD</td><td>Morph. Systematicity (coherence) ID</td><td>OOD</td></tr><tr><td>qwen-2.5-7b</td><td>13.5 / 16.5</td><td>11.7 / 15.2</td><td>65.4 / 67.5</td><td>57.3 / 59.6</td><td>35.8 / 39.8</td><td>25.8 / 29.0</td></tr><tr><td>qwen-2.5-32b</td><td>21.9 / 25.2</td><td>19.8 / 25.0</td><td>65.9 / 66.6</td><td>54.7 / 55.4</td><td>39.8 / 41.5</td><td>22.1 / 23.1</td></tr><tr><td>gemini-1.5-flash</td><td>26.9 / 35.0</td><td>22.9 / 28.7</td><td>71.2 / 73.2</td><td>50.3 / 51.1</td><td>48.1 / 50.6</td><td>15.4 / 16.5</td></tr><tr><td>gpt-4</td><td>40.6/ 56.0</td><td>35.0 / 50.4</td><td>83.1 / 83.2</td><td>65.6 / 67.4</td><td>65.4 / 67.3</td><td>39.8 / 42.7</td></tr></table>

Table 48: 3-shot Shuffled / Correct morpheme order results for Finnish in English template for all examined models across tasks.
<table><tr><td rowspan="2">Models</td><td colspan="2">Morph. Productivity (accuracy)</td><td colspan="2">Morph. Systematicity (macro-F1)</td><td colspan="2">Morph. Systematicity (coherence)</td></tr><tr><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td></tr><tr><td>qwen-2.5-7b</td><td>16.0 / 18.8</td><td>14.4 / 17.1</td><td>68.3 / 68.7</td><td>59.4 / 61.6</td><td>39.2 / 40.2</td><td>27.7 / 29.4</td></tr><tr><td>qwen-2.5-32b</td><td>22.3 / 27.9</td><td>21.3 / 31.7</td><td>69.0 / 71.3</td><td>62.2 / 64.7</td><td>42.5 / 46.7</td><td>33.1 / 37.3</td></tr><tr><td>gemini-1.5-flash</td><td>28.1 / 35.0</td><td>24.0 / 30.6</td><td>77.7 / 79.9</td><td>56.8 / 57.7</td><td>52.3 / 58.5</td><td>25.0 / 26.9</td></tr><tr><td>gpt-4</td><td>44.2 / 59.6</td><td>34.4 / 52.1</td><td>85.2 / 86.9</td><td>74.8 / 78.4</td><td>66.2 / 70.0</td><td>50.6 / 57.1</td></tr></table>

Table 49: 5-shot Shuffled / Correct morpheme order results for Finnish in English template for all examined models across tasks. 1-shot and 3-shot results can be found in Tables 47 and 48.
<table><tr><td rowspan="2">Models</td><td colspan="2">Morph. Systematicity (macro-F1)</td><td colspan="2">Morph. Systematicity (coherence)</td></tr><tr><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td></tr><tr><td>aya-23-8b</td><td>74.8 / 62.0 / 60.2</td><td>59.7 / 53.9 / 51.1</td><td>46.4 / 27.9 / 25.4</td><td>26.9 / 19.1 / 16.1</td></tr><tr><td>aya-23-35b</td><td>83.5 / 69.9 / 67.9</td><td>74.2 / 64.6 / 61.6</td><td>59.5 / 36.8 / 33.3</td><td>43.3 / 29.2 / 26.0</td></tr><tr><td>qwen-2.5-7b</td><td>81.7 / 71.1 / 68.6</td><td>75.1 / 65.7 / 63.7</td><td>64.1 / 40.5 / 36.3</td><td>54.1 / 33.5 / 29.4</td></tr><tr><td>qwen-2.5-32b</td><td>79.7 / 77.3 / 76.6</td><td>54.0 / 53.1 / 52.8</td><td>65.5 / 56.7 / 53.8</td><td>22.0 / 18.5 / 17.7</td></tr><tr><td>gemini-1.5-flash</td><td>62.3 / 60.8 / 60.6</td><td>41.5 / 41.4 /41.3</td><td>35.0 / 32.2 / 31.6</td><td>0.6 / 0.4 / 0.4</td></tr><tr><td>gpt-4</td><td>85.8 / 85.5 / 83.2</td><td>63.3 / 61.9 / 60.9</td><td>75.7 / 71.4 / 66.9</td><td>38.5 / 33.5 / 30.4</td></tr></table>

Table 50: 1-shot Random / Language-agnostic / Language-specific negative sample selection results for Turkish in English template for all examined models across tasks.
<table><tr><td rowspan="2">Models</td><td colspan="2">Morph. Systematicity (macro-F1)</td><td colspan="2">Morph. Systematicity (coherence)</td></tr><tr><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td></tr><tr><td>aya-23-8b</td><td>74.4 / 64.6 / 61.9</td><td>53.0 / 49.3 / 47.1</td><td>45.6 / 31.4 / 28.1</td><td>21.5 / 15.7 / 13.4</td></tr><tr><td>aya-23-35b</td><td>88.2 / 80.1 / 78.8</td><td>80.4 / 71.0 / 71.0</td><td>72.9 / 52.6 / 48.9</td><td>60.5 / 39.9 / 38.1</td></tr><tr><td>qwen-2.5-7b</td><td>81.2 / 73.6 / 71.6</td><td>75.3 / 66.8 / 65.6</td><td>63.8 / 44.3 / 38.8</td><td>53.5 / 33.9 / 32.2</td></tr><tr><td>qwen-2.5-32b</td><td>88.3 / 84.7 / 83.4</td><td>74.5 / 71.3 / 69.8</td><td>78.3 / 66.3 / 63.6</td><td>55.4 / 45.7 / 42.4</td></tr><tr><td>gemini-1.5-flash</td><td>80.2 / 80.8 / 79.9</td><td>51.7 / 52.8 / 51.5</td><td>65.3 / 63.6 / 60.8</td><td>17.8 / 19.3 / 17.3</td></tr><tr><td>gpt-4</td><td>93.7 / 90.2 / 89.1</td><td>82.4 / 77.7 / 74.1</td><td>88.1 / 76.8 / 72.7</td><td>66.8 / 55.9 / 45.8</td></tr></table>

Table 51: 3-shot Random / Language-agnostic / Language-specific negative sample selection results for Turkish in English template for all examined models across tasks.
<table><tr><td>Models</td><td colspan="2">Morph. Systematicity (macro-F1)</td><td colspan="2">Morph. Systematicity (coherence)</td></tr><tr><td></td><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td></tr><tr><td>aya-23-8b</td><td>77.4 / 67.5 / 66.6</td><td>58.1 / 51.5 / 50.6</td><td>51.0 / 36.0 / 33.5</td><td>25.5 / 18.4 / 17.3</td></tr><tr><td>aya-23-35b</td><td>89.6 / 81.8 / 80.5</td><td>80.7 / 72.1 / 70.9</td><td>75.8 / 55.8 / 51.8</td><td>60.5 / 41.8 / 38.8</td></tr><tr><td>qwen-2.5-7b</td><td>83.1 / 74.6 / 72.1</td><td>76.0 / 66.0 / 65.0</td><td>64.3 / 45.1 / 39.7</td><td>52.9 / 33.1 / 30.3</td></tr><tr><td>qwen-2.5-32b</td><td>90.1 / 85.9 / 83.9</td><td>81.3 / 75.3 / 73.6</td><td>80.3 / 66.8 / 61.8</td><td>64.7 / 48.3 / 45.8</td></tr><tr><td>gemini-1.5-flash</td><td>87.8 / 85.4 / 85.3</td><td>61.6 / 62.1 / 59.4</td><td>78.3 / 70.7 / 68.6</td><td>34.5 / 33.3 / 29.1</td></tr><tr><td>gpt-4</td><td>95.4 / 91.6 / 89.4</td><td>83.7 / 78.8 / 72.1</td><td>89.2 / 76.6 / 70.8</td><td>64.7 / 51.4 / 38.7</td></tr></table>

Table 52: 5-shot Random / Language-agnostic / Language-specific negative sample selection results for Turkish in English template for all examined models across tasks.

<table><tr><td>Models</td><td>ID</td><td>Morph. Productivity (accuracy) OOD</td><td>Morph. Systematicity (macro-F1) ID</td><td>OOD</td><td>Morph. Systematicity (coherence) ID</td><td>OOD</td></tr><tr><td>gpt-4 (temp=0)*</td><td>54.0</td><td>44.0</td><td>92.0</td><td>79.0</td><td>77.0</td><td>51.0</td></tr><tr><td>gpt-4 (temp=0.3)</td><td>53.0</td><td>43.0</td><td>92.0</td><td>79.0</td><td>80.0</td><td>53.0</td></tr><tr><td>gpt-4 (temp=0.5)</td><td>55.0</td><td>43.0</td><td>92.0</td><td>80.0</td><td>80.0</td><td>53.0</td></tr><tr><td>gpt-4 (temp=0.7)</td><td>53.0</td><td>43.0</td><td>92.0</td><td>80.0</td><td>80.0</td><td>53.0</td></tr><tr><td>gpt-4 (temp=0.9)</td><td>53.0</td><td>42.0</td><td>92.0</td><td>79.0</td><td>80.0</td><td>51.0</td></tr></table>

Table 53: 5-shot results for Turkish in English template for GPT-4 across tasks and different temperature values. ∗Corresponds to default decoding setting for main results.

<table><tr><td>Models</td><td>ID</td><td>Morph. Productivity (accuracy) OOD</td><td>ID</td><td>Morph. Systematicity (macro-F1) OOD</td><td>Morph. Systematicity (coherence) ID OOD</td></tr><tr><td>gpt-4 (top_p=1)*</td><td>54.0</td><td>44.0</td><td>92.0</td><td>79.0</td><td></td></tr><tr><td>gpt-4 (top_p=0.95)</td><td>53.0</td><td>43.0</td><td>92.0</td><td>78.0</td><td></td></tr><tr><td>gpt-4 (top_p=0.9)</td><td>54.0</td><td>42.0</td><td>92.0</td><td>78.0</td><td>51.0 52.0</td></tr></table>

Table 54: 5-shot results for Turkish in English template for GPT-4 across tasks and different top\_p values. ∗Corresponds to default decoding setting for main results.

<table><tr><td>Models</td><td>Morph. Productivity (accuracy) ID</td><td>OOD</td><td>Morph. Systematicity (macro-F1) ID</td><td>OOD</td><td>Morph. Systematicity (coherence) ID</td><td>OOD</td></tr><tr><td>gpt-4 (temp=0)*</td><td>44.0</td><td>34.0</td><td>85.0</td><td>75.0</td><td>66.0</td><td>51.0</td></tr><tr><td>gpt-4 (temp=0.3)</td><td>45.0</td><td>36.0</td><td>85.0</td><td>74.0</td><td>64.0</td><td>48.0</td></tr><tr><td>gpt-4 (temp=0.5)</td><td>45.0</td><td>34.0</td><td>85.0</td><td>73.0</td><td>64.0</td><td>45.0</td></tr><tr><td>gpt-4 (temp=0.7)</td><td>44.0</td><td>36.0</td><td>86.0</td><td>73.0</td><td>65.0</td><td>46.0</td></tr><tr><td>gpt-4 (temp=0.9)</td><td>44.0</td><td>33.0</td><td>84.0</td><td>71.0</td><td>63.0</td><td>42.0</td></tr></table>

Table 55: 5-shot results for Finnish in English template for GPT-4 across tasks and different temperature values. ∗Corresponds to default decoding setting for main results.

<table><tr><td>Models</td><td>Morph. Productivity (accuracy) ID</td><td>OOD</td><td>ID</td><td>Morph. Systematicity (macro-F1) OOD</td><td>ID</td><td>Morph. Systematicity (coherence) OOD</td></tr><tr><td>gpt-4 (top_p=1)*</td><td>44.0</td><td>34.0</td><td>85.0</td><td>75.0</td><td>66.0</td><td>51.0</td></tr><tr><td>gpt-4 (top_p=0.95)</td><td>43.0</td><td>34.0</td><td>86.0</td><td>73.0</td><td>65.0</td><td>46.0</td></tr><tr><td>gpt-4 (top_p=0.9)</td><td>43.0</td><td>34.0</td><td>85.0</td><td>79.0</td><td>64.0</td><td>44.0</td></tr></table>

Table 56: 5-shot results for Finnish in English template for GPT-4 across tasks and different top\_p values. ∗Corresponds to default decoding setting for main results.

<table><tr><td>Models</td><td>Morph. Productivity (accuracy) ID</td><td>OOD</td><td>Morph. Systematicity (macro-F1) ID</td><td>OOD</td><td>Morph. Systematicity (coherence) ID</td><td>OOD</td></tr><tr><td>gpt-4 (original)*</td><td>54.0</td><td>44.0</td><td>92.0</td><td>79.0</td><td>77.0</td><td>51.0</td></tr><tr><td>gpt-4 (paraphrased)</td><td>56.0</td><td>46.0</td><td>93.0</td><td>80.0</td><td>81.0</td><td>54.0</td></tr></table>

Table 57: 5-shot results for Turkish in English template for GPT-4 across tasks and different prompt instructions. ∗Corresponds to default prompt instructions for main results.

<table><tr><td>Models</td><td>ID</td><td>Morph. Productivity (accuracy) OOD</td><td>Morph. Systematicity (macro-F1) ID</td><td>OOD</td><td>Morph. Systematicity (coherence) ID</td><td>OOD</td></tr><tr><td>gpt-4 (original)*</td><td>44.0</td><td>34.0</td><td>85.0</td><td>75.0</td><td>66.0</td><td>51.0</td></tr><tr><td>gpt-4 (paraphrased)</td><td>46.0</td><td>37.0</td><td>84.0</td><td>73.0</td><td>62.0</td><td>45.0</td></tr></table>

Table 58: 5-shot results for Finnish in English template for GPT-4 across tasks and different prompt instructions. ∗Corresponds to default prompt instructions for main results.