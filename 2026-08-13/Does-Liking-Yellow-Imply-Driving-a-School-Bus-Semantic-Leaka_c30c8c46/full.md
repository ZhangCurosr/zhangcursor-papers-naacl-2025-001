# Does Liking Yellow Imply Driving a School Bus? Semantic Leakage in Language Models

Hila Gonen<sup>1</sup> Terra Blevins<sup>1</sup> Alisa Liu<sup>1</sup> Luke Zettlemoyer<sup>1</sup> Noah A. Smith<sup>1,2</sup> <sup>1</sup>Paul G. Allen School of Computer Science & Engineering, University of Washington <sup>2</sup>Allen Institute for Artificial Intelligence hilagnn@gmail.com {blvns,alisaliu,lsz,nasmith}@cs.washington.edu

## Abstract

Despite their wide adoption, the biases and unintended behaviors of language models remain poorly understood. In this paper, we identify and characterize a phenomenon never discussed before, which we call semantic leakage, where models leak irrelevant information from the prompt into the generation in unexpected ways. We propose an evaluation setting to detect semantic leakage both by humans and automatically, curate a diverse test suite for diagnosing this behavior, and measure significant semantic leakage in 13 flagship models. We also show that models exhibit semantic leakage in languages besides English and across different settings and generation scenarios. This discovery highlights yet another type of bias in language models that affects their generation patterns and behaviour.

## 1 Introduction

As language models (LMs) become more prevalent (Touvron et al., 2023; Anil et al., 2023; Achiam et al., 2023; Scao et al., 2022), we are steadily learning more about their peculiarities and the unique and often unexpected properties of their behavior. Phenomena ranging from hallucinations (Ji et al., 2023) to sycophancy (Sharma et al., 2024) and many types of biases (Navigli et al., 2023) have been revealed in these models’ outputs. Each such discovery leads to a cycle of in-depth study and development of new methods to mitigate these behaviors as much as possible.

We identify a phenomenon in language models never discussed before, which we term semantic leakage — these models can generate text with strong semantic relationships to unrelated words in the prompts. For example, when given the prompt “He likesyellow. He works as a”, GPT4O<sup>1</sup> generates the output “school bus driver” (Figure 1). Here we say that the word yellow has leaked into the generation in a way that unintentionally influences the generated occupation. Figure 1 shows three additional examples of prompt-generation pairs (using GPT4O). In each example, the leakage from the semantic meaning of the underlined word in the prompt is apparent in the generation.

![](images/2fe2c7656fe4083c5513b84e7667f8680dccc5ed7f7dce6e92c9ed0dfd9ae887.jpg)  
Figure 1: Examples of semantic leakage in GPT4O. The leaking concept is underlined.

We define semantic leakage in a generation as an undue influence of semantic features from words in the prompt on the generation, “undue” in the sense that the semantic relatedness between the prompt and the generation is stronger than would be expected in natural distributions. Often semantic leaks read as forced, overwrought, even nonsensical generations, like those found in children’s stories.

In this paper, we introduce an evaluation metric for measuring semantic leakage. We examine semantic leakage with 109 examples of different semantic categories (animals, food, music, etc.) and demonstrate that it exists across 13 models and

4 temperature sampling values, as well as in additional generation settings (e.g., open-ended generation and multilingual settings). Our analysis shows that finetuned/instruction-tuned models tend to leak more, and that semantic leakage also happens across languages.

Semantic leakage is closely related to different types of biases models exhibit, ranging from gender, racial and cultural biases (Bolukbasi et al., 2016; Caliskan et al., 2017; Gonen and Goldberg, 2019; Nadeem et al., 2021) to cognitive and psychological biases (Jones and Steinhardt, 2022; Macmillan-Scott and Musolesi, 2024; Hagendorff et al., 2023), in which associations between different concepts are learned by the model during training and exposed as bias during generation (Gonen and Webster, 2020; Schick et al., 2021). As an example, consider the prompt “She works at the hospital as a”, and the prompt “He works at the hospital as a”. Given the generations “nurse” and “doctor”, respectively, which is a typical biased behavior, we can think of the word “she” as the concept that leaks the property of the female gender into the generation “nurse”, a stereotypically female occupation. While still not fully understood, we suspect that much documented and discussed gender bias and other types of previously documented biases (Navigli et al., 2023) are instances of associations that get learned and influence in a broader way, which is partially reflected as semantic leakage. Specifically, here we are looking at larger semantic classes (i.e., compared to gender or race), and we seek to quantify and study learned associations and their effect on model generation settings.

Given the nature of models in learning associations during training, whether semantic leakage is surprising or not is a point of contention. In addition, the implications of this behaviour depend on the user and application context, and are not necessarily either good or bad. At the same time, we strongly believe that it is an interesting and important behavior to investigate as it may be a broad enough umbrella to encompass many other associations that are studied as more specific cases. Studying the broader family may be easier and more beneficial, as the more general class may include cases more resistant to mitigation strategies and more prevalent in model outputs.

Finally, in addition to typical and previously discussed biases of different types that fall under this broad class of semantic leakage, we point to additional potential ramifications of this phenomenon. First, semantic leakage might hinder the performance of models via the overshadowing mechanism, introduced in Zhang et al. (2024), in which strong associations in a question override more important and relevant parts of the question to a point where the latter are ignored, resulting in a wrong answer. Semantic leakage might also be used in adversarial settings, similar to attacks presented in Wallace et al. (2019), where specific words that are added to the prompt negatively influence the output of the model. Other potential implications include hindering performance in creative writing (Ippolito et al., 2022), where diversity and originality are especially important, but semantic leakage might significantly hurt the output quality in unexpected ways.

Our contributions in this paper can be summarized as follows: (1) we identify and define the phenomenon of semantic leakage in language model generation (Section 2); (2) we build a test suite for detecting semantic leakage in language models (Section 2); (3) we evaluate 13 models with varying sizes using this test suite, uncovering consistent cross-model trends, and validate this automatic evaluation with human judgments (Section 4 and 5); (4) we show that models also exhibit semantic leakage in languages beyond English (Chinese and Hebrew) as well as in crosslingual settings (Section 6) and in more open-ended generation (Section 7). By characterizing semantic leakage, we demonstrate yet another property language models exhibit in text generation, and highlight how choices in prompt construction can inadvertently affect model output.

## 2 Semantic Leakage

## 2.1 Overview and Definitions

When producing text, language models can draw on semantic associations with words from the input, or prompt, that are not required or expected, and sometimes even violate rules of logic or common sense. For example, given the prompt “He likes koalas. His favorite food is” GPT4O generates the output “eucalyptus leaves”. Here, we say that the semantic association with “koalas” and the foods they eat “leaks” into the generation, despite the fact that a person’s favorite food and their opinion on koalas are unrelated in the real world. We call this phenomenon semantic leakage.

A related phenomenon of conceptual leakage has been reported in image generation (Rassin et al., 2022). There, they find that visual properties of one object leak into other objects in the image (for example, the prompt $^ { 6 6 } a$ zebra and a street” generates an image of a zebra next to a zebra crossing), which resembles examples we show (e.g., yellow leaking into the occupation through “school bus driver”, see first example in Figure 1). While this behaviour might be seen as subjective, the cases we consider in this paper are, we believe, beyond debate.

Semantic leakage in text generation can also manifest in more subtle ways: for the prompt $^ { * } H e$ likes green. He works as $\boldsymbol { a } ^ { \flat }$ , GPT4O generates “landscape architect” as a response. In other cases, the model may leak semantics that are not even used in the prompt: For example, when prompted with an idiom, a model can leak the literal semantic meaning of that phrase (that is not actually being used): for instance, when prompted with “She gave him the green light for the new project. A day later he sent an invitation to everyone by mail, with an envelope colored”, GPT3.5 generates the response “bright green to match the theme of the project.” This is similar to another observation made by Rassin et al. (2022) where the authors demonstrate that sense-ambiguous words are hard for the model to isolate, and the generated images often exhibit the unintended sense together with the intended one.

## 2.2 Operationalizing the Measurement of Semantic Leakage

We define the different elements of semantic leakage as follows. A prompt is input text that primes the model to output a corresponding generation. We consider two types of prompts: control prompts, which do not include any spurious semantic signal (“His favorite food $i s ^ { \prime \prime } )$ , and test prompts $( ^ { \ast } H e$ likes koalas. His favorite food $i s ^ { \prime \prime } )$ , which mirror the control prompt but add a semantically unrelated concept (“koalas”) to the input, leading to a different, test generation. While it is known that changing the surface form of the prompt often alters model output (Gonen et al., 2023; Sclar et al., 2024), these new test generations are frequently much more semantically similar to the concept than the control generations (Section 4).

We evaluate the prevalence of semantic leakage in a given model by comparing the similarity of the generations produced by the control and test prompts to the concept under consideration. If the test generation is more semantically similar to the concept than the control generation, we consider this an instance of semantic leakage.

To quantify the prevalence of semantic leakage, we design an evaluation setting that is motivated by the definition of semantic leakage, as having stronger connections of the concept to the test generation. The goal is to compare the similarity of the generations produced by the control and test prompts to the concept, while making sure no other factors are taken into consideration. We use the following similarities to then derive the evaluation metric detailed below:

$$
\begin{array} { c } { { \sin _ { \mathrm { c o n t r o l } } = s i m i l a r i t y ( \mathrm { c o n c e p t , c o n t r o l } ) } } \\ { { \sin _ { \mathrm { t e s t } } = s i m i l a r i t y ( \mathrm { c o n c e p t , t e s t } ) } } \end{array}
$$

Evaluation Metric From the above formulation, we derive the “Semantic Leakage Rate” metric (Leak-Rate), the percentage of instances in which the concept is semantically closer to the test generation than the control generation. We score Leak-Rate by averaging the following function across all instances, and converting to the range of 0–100%:

$$
\mathrm { L e a k - R a t e ( t e s t , c o n t r o l ) } = \left\{ \begin{array} { l l } { { 1 } } & { { \mathrm { s i m } _ { \mathrm { t e s t } } > \mathrm { s i m } _ { \mathrm { c o n t r o l } } } } \\ { { 0 } } & { { \mathrm { s i m } _ { \mathrm { t e s t } } < \mathrm { s i m } _ { \mathrm { c o n t r o l } } } } \\ { { 0 . 5 } } & { { \mathrm { s i m } _ { \mathrm { t e s t } } = \mathrm { s i m } _ { \mathrm { c o n t r o l } } } } \end{array} \right.\tag{1}
$$

This evaluation setting can be implemented as automatic evaluation (Section 4) or as human evaluation (Section 5), with no required adjustments. For automatic evaluation, we embed the texts (concept, control generation, test generation) with an embedding method (see Section 3.3) and compute similarity measures; for human evaluation, we ask annotators to judge which of the two pairs is more semantically similar.

When the model does not exhibit semantic leakage at all, we expect a Leak-Rate of 50%, i.e., an even split between test vs. control having higher similarity, for those that show a difference. We expect Leak-Rate higher than 50% when the model exhibits semantic leakage. The Leak-Rate scores are comparable across models for a given embedding method and test set of prompt pairs.

## 2.3 Building a Test Suite

We build a test set of prompts containing categories and concepts with clear semantic associations. We also focus on categories and prompt settings that are likely to allow uncontroversial evaluation (by priming for short model outputs), though later we also consider open-ended generation (Section 7). In addition, a subset of the test suite considers idioms, which have both literal and figurative interpretations, as concepts in the test prompts.

We manually create 109 prompts with concepts from categories such as colors, food, animals, songs, occupations and more. A few examples are listed in Table 1.<sup>2</sup> Each prompt in our test suite is matched with a control prompt as explained above. When designing the control prompts, we prioritized keeping them as basic and simple as possible to avoid confounding factors. In some cases, more elaborate prompts required us to include richer control prompts that are more similar to the test prompts in their length and structure.<sup>3</sup> Both types of control prompts are included in our evaluation.

## 3 Experimental Setup

We present the experimental setup for automatically evaluating semantic leakage in a wide range of models and model sizes.

## 3.1 Experimental Details

We evaluate semantic leakage in multiple language models from two families: GPT<sup>4</sup> and LLAMA models (Touvron et al., 2023), as detailed below. For all models, we explore several temperature values (0, 0.5, 1, 1.5), and run each prompt 10 times to get variation in the generations, when possible.

Before evaluation, for cases where the prompt is repeated in the model generation, we remove the repeated prompt. We also truncate the generations after the first period since the main piece of information is generated before it, and because LLAMA models tend to generate unrelated sentences or phrases that might interfere with the evaluation.

## 3.2 Models

We experiment with 13 models of two different state-of-the-art model families to explore semantic leakage in a diverse inventory of models.

GPT models We use OpenAI’s API and send requests to GPT models by calling GPT3.5, GPT4 (Achiam et al., 2023), and GPT4O.<sup>5</sup> With GPT models, for sentence completion prompts we prepend “Complete the sentence:”, as we find the model performs the task better this way.

LLAMA models We run all LLAMA variations using Huggingface (Wolf et al., 2019).<sup>6</sup> We cap the generation in LLAMA models at 100 tokens (300 tokens for open generation, Section 7).

## 3.3 Embedding Methods

For automatic evaluation, we aim to use basic embedding methods that are able to detect and reflect semantic similarities, and are ideally detached from the models we evaluate to avoid confounding factors. We consider the following embedding methods. BERT-SCORE directly provides a similarity score, for the others we apply cosine-similarity.

BERT-SCORE (BS) BERT-SCORE (Zhang et al., 2020) is an automatic evaluation metric for text generation, that computes a similarity score for each token in the candidate sentence with each token in the reference sentence, where token similarity is computed using contextual embeddings. We use the distilbert-base-uncased model.<sup>7</sup>

For the multilingual experiments, where we expect generations in non-English languages (Section 6), we use the respective models: bert-base-chinese for Chinese, and bert-base-multilingual-cased for Hebrew. Crosslingual settings still use the English model as the generations there are mainly in English.

SENTENCEBERT EMBEDDINGS (SB) SEN-TENCEBERT (Reimers and Gurevych, 2019) is a modification of BERT (Devlin et al., 2019) that uses Siamese and triplet network structures to derive semantically meaningful sentence embeddings that can be compared using cosine-similarity. We use the huggingface implementation.<sup>8</sup>

OPENAI EMBEDDINGS (OAI) We use text-embedding-3-large,<sup>9</sup> OpenAI’s best performing embeddings. There is no public documentation of the model training.

<table><tr><td>Model</td><td>Prompt and generation</td></tr><tr><td>LLAMA2</td><td>John likes dolphins. John&#x27;s father is working as a marine biologist</td></tr><tr><td>7B-CHAT</td><td>John likes tigers. John&#x27;s father is working as a tiger in a zoo</td></tr><tr><td>LLAMA2</td><td>He likes pandas. His favorite food is bamboo</td></tr><tr><td>13B-CHAT</td><td>He likes mice. His favorite food is cheese</td></tr><tr><td>LLAMA2</td><td>His name is Cedar. His friend lives in a treehouse</td></tr><tr><td>70B-CHAT</td><td>His name is Ocean. His friend lives in a houseboat</td></tr><tr><td>LLAMA3</td><td>He watched a polka concert. For dinner he had a plate of pierogies and a side of sauerkraut</td></tr><tr><td>8B-INS</td><td>He watched a flamenco concert. For dinner he had a delicious paella</td></tr><tr><td>LLAMA3</td><td>Mercedes has a friend. The friend&#x27;s hobby is collecting Mercedes cars because he loves the Mercedes brand</td></tr><tr><td>70B-INS</td><td>Forest has a friend. The friend&#x27;s hobby is gardening, to her home in a way which makes it a forest</td></tr><tr><td>GPT3.5</td><td>He is a firefighter. His favorite song is “Ring of Fire&quot; by Johnny Cash</td></tr><tr><td></td><td>He is a postman. His favorite song is &quot;Signed, Sealed, Delivered&quot; by Stevie Wonder The woman was as cold as ice. Afterwards, I went to the library and read about the fascinating science of</td></tr><tr><td>GPT4</td><td>cryogenics, which explores the process of freezing and preserving organic matter at extremely low temperatures It was her turn to speak and she got cold feet. A day later she stayed at home because she got hurt in her ankle during a morning jog</td></tr><tr><td>GPT40</td><td>She is a music lover. Her nephew was accused of stealing her prized vinyl collection</td></tr><tr><td></td><td>She is a cinema lover. Her nephew was accused of pirating movies</td></tr></table>

Table 1: Examples of instances that lead to semantic leakage, taken from different models. The prompt is in green (italic) and the generation is in blue.

In addition to these metrics, we validate our experiments with a manual evaluation of semantic leakage on a subset of the models (Section 5). This also serves as a validation of our automatic metrics.

## 4 Results

Significant semantic leakage across various use cases and models. Table 2 depicts the average leakage for each model, across multiple samplings and temperature values, as detailed in Section 3.1. We see that semantic leakage is exhibited by all model variations, and is detected by all embedding models we use. Leak-Rate values are all well above the 50% random mark and statistically significant<sup>10</sup> with $p < 1 0 ^ { - 1 0 0 }$ , validating the semantic relatedness of the prompt and the test generation.

Table 1 lists a few examples of semantic leakage from the different models, showcasing leakage in diverse use cases and styles and with respect to a variety of leaking concepts. In many cases we explore, the generations do not make sense in the context, or are very limited and focused on the leaking concept from the prompt.

Leakage is more pronounced in certain model variations. The results in Table 2 show that certain model variations tend to exhibit more semantic leakage than their counterparts. We now analyze the differences within the 2 model families.

<table><tr><td rowspan=4 colspan=1>Model</td><td rowspan=4 colspan=3>BS   SB</td><td></td></tr><tr><td rowspan=1 colspan=3></td></tr><tr><td rowspan=1 colspan=2>Rate</td></tr><tr><td rowspan=1 colspan=1>OAI</td></tr><tr><td rowspan=1 colspan=1>GPT3.5</td><td rowspan=1 colspan=1>74.3</td><td rowspan=1 colspan=2>68.6</td><td rowspan=1 colspan=1>85.5</td></tr><tr><td rowspan=2 colspan=1>GPT4GPT40</td><td rowspan=1 colspan=1>70.8</td><td rowspan=1 colspan=2>61.2</td><td rowspan=1 colspan=1>84.4</td></tr><tr><td rowspan=1 colspan=1>76.9</td><td rowspan=1 colspan=2>70.4</td><td rowspan=1 colspan=1>85.0</td></tr><tr><td rowspan=1 colspan=1>2-7b</td><td rowspan=1 colspan=1>66.8</td><td rowspan=1 colspan=2>64.9</td><td rowspan=1 colspan=1>72.8</td></tr><tr><td rowspan=1 colspan=1>2-7b-chat</td><td rowspan=1 colspan=1>72.6</td><td rowspan=1 colspan=2>71.7</td><td rowspan=1 colspan=1>77.8</td></tr><tr><td rowspan=1 colspan=1>2-13b</td><td rowspan=1 colspan=1>70.4</td><td rowspan=1 colspan=2>65.1</td><td rowspan=1 colspan=1>73.6</td></tr><tr><td rowspan=1 colspan=1>2-13b-chat</td><td rowspan=1 colspan=1>71.5</td><td rowspan=1 colspan=2>65.2</td><td rowspan=1 colspan=1>78.4</td></tr><tr><td rowspan=1 colspan=1>2-70b</td><td rowspan=1 colspan=1>71.9</td><td rowspan=1 colspan=2>63.2</td><td rowspan=1 colspan=1>75.9</td></tr><tr><td rowspan=1 colspan=1>2-70b-chat</td><td rowspan=1 colspan=1>75.4</td><td rowspan=1 colspan=2>66.8</td><td rowspan=1 colspan=1>78.2</td></tr><tr><td rowspan=1 colspan=1>3-8b</td><td rowspan=1 colspan=1>69.6</td><td rowspan=1 colspan=2>65.9</td><td rowspan=1 colspan=1>75.5</td></tr><tr><td rowspan=1 colspan=1>3-8b-ins</td><td rowspan=1 colspan=1>78.1</td><td rowspan=1 colspan=2>68.8</td><td rowspan=1 colspan=1>81.5</td></tr><tr><td rowspan=1 colspan=1>3-70b</td><td rowspan=1 colspan=1>71.6</td><td rowspan=1 colspan=2>68.1</td><td rowspan=1 colspan=1>75.2</td></tr><tr><td rowspan=1 colspan=1>3-70b-ins</td><td rowspan=1 colspan=1>76.3</td><td rowspan=1 colspan=2>71.2</td><td rowspan=1 colspan=1>77.3</td></tr></table>

Table 2: Semantic Leak-Rate averaged across 10 samples for each of 4 temperature values. No semantic leakage would correspond to a Leak-Rate of 50% (random guessing), with higher values indicating more leakage. The bottom sections present the LLAMA2 and LLAMA3 models, respectively. The model showing the most leakage in each <model family, metric> setting is bolded.

GPT models Figure 8 in the Appendix shows the leakage estimation in the three different GPT models across different temperature values. GPT4O consistently leaks more than GPT4 and GPT-3.5.<sup>11</sup>

LLAMA models For LLAMA models we consistently see that the instruction-tuned models (CHAT version in LLAMA2 and INSTRUCT version in LLAMA3) leak more than their pretrained-only counterparts. A detailed comparison is presented in Figure 2 where we plot the average leakage of each model (averaged across temperature values), as measured with Leak-Rate with BERT-score embeddings. All the differences are statistically significant<sup>12</sup> with $p < 0 . 0 0 2$ except for Llama-2-13b. We see similar trends with all other metrics as well.

![](images/20e583da2e67bdf1ccaa43d8a084267ac92886af558878ed540512b5cd2197ea.jpg)  
Figure 2: Semantic leakage in LLAMA models, averaged across temperature values (measured with Leak-Rate using BERT-score).

Leakage across different temperatures. We now inspect the way sampling temperature affects semantic leakage. For the GPT models, we see no clear trends (Figure 8 in the Appendix).<sup>13</sup>

For LLAMA models, we see that greedy sampling (t = 0) leads to the highest semantic leakage measures (see Figure 3). Generally, lower temperature values lead to more leakage—this is consistent for most models and across all metrics.

## 5 Human Evaluation

We now perform a manual evaluation of semantic leakage, which will also validate our automatic metrics and experiments.

![](images/e77a43c0549d76a1ce33d957ef421788a19699fd6ea79d4b9934b2a205a40798.jpg)  
Figure 3: Semantic leakage in LLAMA at different temperatures (measured with Leak-Rate using BERT-score).

Human Evaluation Setup Human evaluation is based on the same setting as automatic evaluation, described in Section 2.2.

We recruit two native English speakers who are not involved with the project. These annotators are not provided with the objective of this experiment, but instead are given the following annotation guidelines: Consider the word or phrase X. Which of the following texts (A or B) is more semantically related to X? (A/B/Neither). We map the annotator’s choice of (A/B/Neither) to (test, control, neither) and then score Leak-Rate (Eq. 1).

Using these guidelines, the annotators are asked to label 109 test-control generation pairs from each model (ordered randomly). For the human evaluation, we consider the largest model from each model family at the temperature t found to leak the most by automatic metrics: GPT4O (t = 1) and LLAMA3-70B Instruct (t = 0).

Analysis Figure 4 compares the semantic leakage detected by the human evaluation for GPT4O against the automatic metric. The values in the human evaluation row are percentages for each category: test is more similar to the concept, control is more similar to the concept, or neither.

We expect humans to have higher tolerance for similar scores, i.e., more cases falling under $\mathrm { s i m } _ { \mathrm { t e s t } } = \mathrm { s i m } _ { \mathrm { c o n t r o l } }$ in the human evaluation than in the automatic evaluation, where it occurs almost solely when the test and control generations are the same (see Equation 1). To visually account for this difference we plot the automatic metric results by using colored gradient to depict the difference in similarity values sim<sub>test</sub> sim<sub>control</sub>, with positive values (shades of blue on the right) implying semantic leakage. The results account for all ten generations sampled from each model.

We find similar trends on LLAMA3-70B Instruct with an average human-annotated Leak-Rate of 66.7 and automatic evaluation of Leak-Rate that range from 71.2 to 77.3 across the different embedding types.

We also calculate Kendall’s τ on the human annotations. We find high interannotator agreement between the human annotators (τ = 0.68), indicating that humans generally agree on the cases that constitute semantic leakage. We also compute Kendall’s τ on the human evaluation vs. similarity differences calculated using BERTScore embeddings to evaluate how well these methods correlate,<sup>14</sup> and get a moderate correlation of τ = 0.39 when averaged between the two annotators. 15

![](images/7d9ba30311e0975c344ea2e5db198b7ca9d0c1f8cb50ba1e3ec5d9eb0e991cab.jpg)  
Figure 4: Human detection of semantic leakage compared to automatic methods. Leak-Rate is reported on the right for each method.

## 6 Multilingual and Crosslingual Semantic Leakage

In the previous sections we established that semantic leakage is exhibited in various scenarios in English, across different models and model sizes. We now inspect semantic leakage in languages outside of English, as well as in crosslingual settings, by focusing on Hebrew and Chinese, Hebrew-English and Chinese-English. As in the previous section, we consider the largest model from each model family at the temperature t found to leak the most: GPT4O (t = 1) and LLAMA3-70B Instruct (t = 0). Since we found that LLAMA is struggling with Hebrew, especially with Hebrew generation, we exclude these results.

Multilingual Semantic Leakage We translate all the prompts into Chinese and Hebrew (see examples in Table 4 in the Appendix, rows 1, 3) with the help of native speakers (including the authors). The translation is mostly straightforward with the exception of idioms and names. For English idioms that do not exist in the target language, we write a new example using an idiom in that language (together with a matching context and a control prompt). For names, in Chinese we choose a plausible Chinese name with that meaning (e.g., 小 麦, meaning little wheat, for Rye). In Hebrew, we make an attempt to translate to an existing frequent name in Hebrew, or to substitute it with a name with a related meaning (e.g., Lake instead of River). If that is not an option (e.g., Rye), we come up with a different name that has a clear semantic field.

Crosslingual Semantic Leakage We create mixed prompts that begin with Chinese/Hebrew and end with English (see examples in Table 4 in the Appendix, rows 2, 4). We use the translations to Chinese and Hebrew, and mix with English: for the first part of the prompt we use the translation to Chinese/Hebrew, and for the rest of it we use the original part in English. For evaluation purposes, we use the English concepts, since we expect the generation to be in English. For names or idioms that are used in Chinese/Hebrew parts of the prompt, we use their literal translations into English.

Results Figure 5 shows the human evaluation results for the multilingual and crosslingual experiments (with GPT4O). We can see significant semantic leakage in multilingual and crosslingual settings, with Leak-Rate values that range from 70.6 to 78.4 for the 4 different settings, similar to the Leak-Rate values we got for English.<sup>16</sup>

The generation quality with the LLAMA model is generally much lower, and we get Leak-Rates of 66.5 and 61.5 for Chinese and Chinese-English, respectively, according to human annotation. As noted above, the quality of generation in Hebrew and Hebrew-English did not allow for evaluation of semantic leakage.

Table 3 shows the automatic evaluation of the multilingual and crosslingual experiments. The results for the multilingual settings are not as reliable as in the English setting since we cannot determine the quality of the underlying embedding methods for Hebrew and Chinese.<sup>17</sup> In the crosslingual setting we mainly compare English generations with English concepts, thus the results are more reliable.

![](images/e8b53014789c8054e8ab1b3247e7045d943a2b8c1c77002fad79c1d868a83577.jpg)  
Figure 5: Human detection of semantic leakage in multilingual and crosslingual settings.

<table><tr><td rowspan=2 colspan=1>Model</td><td rowspan=2 colspan=1>Language</td><td rowspan=2 colspan=3>Leak-RateBS</td></tr><tr><td rowspan=1 colspan=1>SB</td><td rowspan=1 colspan=1>OAI</td></tr><tr><td rowspan=4 colspan=1>GPT40</td><td rowspan=4 colspan=1>HebHeb-EnZhZh-En</td><td rowspan=1 colspan=1>60.6</td><td rowspan=1 colspan=1>53.5</td><td rowspan=1 colspan=1>67.2</td></tr><tr><td rowspan=1 colspan=1>62.1</td><td rowspan=1 colspan=1>58.2</td><td rowspan=1 colspan=1>74.9</td></tr><tr><td rowspan=2 colspan=1>67.661.6</td><td rowspan=2 colspan=1>48.960.5</td><td rowspan=1 colspan=1>80.8</td></tr><tr><td rowspan=1 colspan=1>71.3</td></tr><tr><td rowspan=1 colspan=1>3-70b-ins</td><td rowspan=1 colspan=1>ZhZh-En</td><td rowspan=1 colspan=1>73.479.4</td><td rowspan=1 colspan=1>54.181.2</td><td rowspan=1 colspan=1>82.885.6</td></tr></table>

Table 3: Semantic leakage scores for multilingual and crosslingual setting, averaged across 10 samplings and measured by Leak-Rate.

## 7 Open-Ended Generation

We also examine semantic leakage in open-ended scenarios that encourage the model to generate multiple sentences or paragraphs. It is less clear how to quantify the semantic leakage in this setting, as it can manifest in more ways within the longer output; we therefore rely on more qualitative analysis in addition to the automatic metrics. This section focuses on generations from GPT4O with t = 1, as this model with this temperature was most prone to leakage in previous experiments (Section 4).

Stories A popular setting for open-ended text generation is storytelling: we ask the model to write a story about a child, with the child’s name being the concept we expect to leak. Figure 6 gives the story that GPT4O generates when prompted to create a story about a child named Coral (and also Melody in Figure 9 in the Appendix). These generated stories contain many words and elements directly related to the semantics of each name, and the narrative of each story is tailored to those meanings. When measuring semantic leakage in this setting automatically for 23 different names,<sup>18</sup> we get Leak-Rates of 78.7, 60.9 and 76.1 with BS, SB and OAI embeddings respectively, suggesting similar rates of leakage to the other examples.

![](images/9436770cbc01e76009302e7ebf0a35545bb8bdc5f0ad71c6a5910d45f584d11a.jpg)  
Figure 6: Example of story generation, exhibiting semantic leakage based on the name in the prompt.

Recipes We also consider open-ended recipe generation. In Figure 7, we present GPT4O’s generation when asked to suggest a recipe for use with a blue pan (or a white pan in Figure 10 in the Appendix), where we consider each color to be the potentially leaking concept. The model suggests to make blueberry pancakes and chicken alfredo, respectively, with the foods matching the color of the pan. When prompted with a red pan, ingredients like peppers, tomatoes and paprika are far more frequent, same as with asparagus in the case of a green pan and lemon with a yellow pan.

We experiment with 5 different colors (10 samples each) that are likely to be related to types of food (blue, green, red, white, and yellow). Automatic evaluation in this setting is more challenging since the concept might appear in the generation in a leaky way (“red pepper”) or just as part of the answer, with no attributed leakage (“red pan”) and these two cases are difficult to differentiate automatically. We decide to remove all occurrences of “<color> pan” from the generations when using automatic metrics, and get Leak-Rates of 74.0/58.0/74.0 with BS/SB/OAI embeddings, again suggesting semantic leakage here as well.

![](images/47601286708cea3177d7cb3afc94e2e5be2b4cf6e963559024c9acd8f0472112.jpg)  
Figure 7: Example of recipe generation, exhibiting semantic leakage based on the color in the prompt.

## 8 Related Work

Rassin et al. (2022) show similar conceptual leakage in image generation with DALLE-2 (Ramesh et al., 2022), a text-to-image model by OpenAI. They find that sense-ambiguous words are hard for the model to isolate—similar to our observation that idioms in the prompt can leak the literal meaning into the generation. Rassin et al. (2024) try to mitigate this issue by encouraging the crossattention maps to agree with the linguistic binding reflected by the prompt syntax. Leakage of properties in DALLE-2 has also been discussed by Ramesh et al. (2022) and Leivada et al. (2023).

As models have improved, a related line of research has asked whether they match human behavior, particularly with respect to cognitive biases and priming. Prior work compares models’ behavior to documented behavior of humans in psychological experiments, highlighting some similarities as well as differences from human response to given information (Jones and Steinhardt, 2022; Macmillan-Scott and Musolesi, 2024; Hagendorff et al., 2023). A known psychological phenomenon that is very related to models’ semantic leakage as we present it here, is that of priming, and specifically semantic priming (Meyer and Schvaneveldt, 1971; Neely, 1976; Tulving et al., 1982; Tulving and Schacter, 1990), where decisions and reactions about specific words change according to given semantic stimuli provided before the target word. We plan to investigate this relationship further in future work.

Biases in Language Models There is an extensive line of work investigating bias and stereotypes of different demographic groups in NLP models, ranging from word embeddings (Bolukbasi et al., 2016; Caliskan et al., 2017; Gonen and Goldberg, 2019) all the way to LMs (Nadeem et al., 2021;

Kotek et al., 2023; Cheng et al., 2023), sometimes focusing on specific end tasks (Gonen and Webster, 2020; Stanovsky et al., 2019; Kiritchenko and Mohammad, 2018; Zhao et al., 2018). Existing works in this space usually investigate specific bias types with respect to concrete socio-demographic groups (Kirk et al., 2021; Venkit et al., 2023, 2022; Navigli et al., 2023), as well as toxicity in model generation (Gehman et al., 2020; Nozza et al., 2021). Some works also point to specific input words that cause harmful generation, such as first names (De-Arteaga et al., 2019; Maudslay et al., 2019) among others (Röttger et al., 2024; Attanasio et al., 2022). In this context, semantic leakage serves as a broader class of association bias, potentially pointing to the underlying mechanism that causes biases in model generation by latching onto concrete properties in the input, resulting in toxic or stereotyped generation.

## 9 Conclusion

We identified a phenomenon in generative language models never discussed before, which we term semantic leakage: language models are prone to generating text that reflects unrelated semantic information from the prompt, leading to peculiar and sometimes unreasonable outputs. We defined and measured semantic leakage in a range of models, and together with human evaluation show that it is prevalent and consistent across all models we test. We also found that semantic leakage occurs in many different generation settings, including multilingual and crosslingual ones.

This leakage reflects associations learned by the model, similar to how different types of biases are learned; therefore, our characterization of semantic leakage broadens the scope we should consider with respect to potential ramifications of learned associations. While the implications of semantic leakage are not fully clear, it is a consistent behavior pattern in LMs that merits further study.

The finding that instruction-tuned models leak more is of special interest, given that they are the most popular and best performing model variations currently used. We hypothesize that semantic leakage is more dominant in these models because the leaking generations are less generic and seem to provide more information/content, which might be a property that is incentivized under these finetuning processes. We plan to explore this hypothesis more formally in future work.

## Limitations

While our experimental setup spans 13 models of different types and sizes, and explores different sampling temperature values, the scale of the prompts in our test suite remains limited due to the difficulty of manually creating prompts that are likely to leak in a way that we can detect and evaluate. This is also often the case when measuring other language model biases. In addition, though the results are consistent across all models and languages we experiment with, the trends might be different with other models or languages we have not tested.

Finally, we cannot guarantee that the automatic evaluations do not include noise: in some cases, even after our automatic post-processing of the generations, our automatic metrics might consider non-leaking instance as leaking (e.g., in cases of occurrences of the concept in the generation due to a partial repetition of the prompt). However, the human evaluation we conduct generally agrees with the findings we get with the automatic detection, supporting their reliability.

## Ethical Considerations

Semantic leakage is a broad umbrella that may encompass many types of biases, and it broadens our perception and understanding about association biases that are being learned by models and then exhibited in generation. As such, semantic leakage might manifest at times as a concrete type of bias, e.g., cultural, gender or racial biases. This depends on the specific setting and the chosen concept that leaks; while our prompts and concepts are fairly general, these biases and stereotypes could potentially surface in our evaluation.<sup>19</sup>

However, our main goal in this paper is not to focus on specific types of biases (or biases that are particularly harmful) but to characterize and quantify the broader class of semantic leakage. We believe that a better understanding of the greater picture will in turn arm us with better tools to combat the subset of biases and behaviors that do have harmful consequences in practice.

Apart from that, in this paper the annotations and translations were done by several volunteers that were willing to help for the project. The load of this work was very low.

## Acknowledgements

This research received support through Schmidt Sciences, LLC. We thank Shauli Ravfogel and Ido Levin for helpful discussions and ideas. We also thank the translators and annotators for their contribution to the paper (Weijia Shi, Jacqueline He, Jacob Schreiber) and the ARK lab members for their valuable feedback. Finally, we would like to express our gratitude to the reviewers of this paper for their useful comments and suggestions.

## References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. 2023. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805.

Giuseppe Attanasio, Debora Nozza, Dirk Hovy, and Elena Baralis. 2022. Entropy-based attention regularization frees unintended bias mitigation from lists. arXiv preprint arXiv:2203.09192.

Tolga Bolukbasi, Kai-Wei Chang, James Y Zou, Venkatesh Saligrama, and Adam T Kalai. 2016. Man is to computer programmer as woman is to homemaker? debiasing word embeddings. In Advances in Neural Information Processing Systems.

Aylin Caliskan, Joanna J Bryson, and Arvind Narayanan. 2017. Semantics derived automatically from language corpora contain human-like biases. Science, 356(6334):183–186.

Myra Cheng, Esin Durmus, and Dan Jurafsky. 2023. Marked personas: Using natural language prompts to measure stereotypes in language models. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1504–1532, Toronto, Canada. Association for Computational Linguistics.

Maria De-Arteaga, Alexey Romanov, Hanna Wallach, Jennifer Chayes, Christian Borgs, Alexandra Chouldechova, Sahin Geyik, Krishnaram Kenthapadi, and Adam Tauman Kalai. 2019. Bias in bios: A case study of semantic representation bias in a high-stakes setting. In proceedings ofthe Conference on Fairness, Accountability, and Transparency, pages 120–128.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the

North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers).

Samuel Gehman, Suchin Gururangan, Maarten Sap, Yejin Choi, and Noah A. Smith. 2020. RealToxicityPrompts: Evaluating neural toxic degeneration in language models. In Findings ofthe Association for Computational Linguistics: EMNLP 2020, pages 3356–3369, Online. Association for Computational Linguistics.

Hila Gonen and Yoav Goldberg. 2019. Lipstick on a pig: Debiasing methods cover up systematic gender biases in word embeddings but do not remove them. In Proceedings ofNAACL-HLT.

Hila Gonen, Srini Iyer, Terra Blevins, Noah A Smith, and Luke Zettlemoyer. 2023. Demystifying prompts in language models via perplexity estimation. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 10136–10148.

Hila Gonen and Kellie Webster. 2020. Automatically identifying gender issues in machine translation using perturbations. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 1991–1995.

Thilo Hagendorff, Sarah Fabi, and Michal Kosinski. 2023. Human-like intuitive behavior and reasoning biases emerged in large language models but disappeared in chatgpt. Nature Computational Science, 3(10):833–838.

Daphne Ippolito, Ann Yuan, Andy Coenen, and Sehmon Burnam. 2022. Creative writing with an ai-powered writing assistant: Perspectives from professional writers. arXiv preprint arXiv:2211.05030.

Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. 2023. Survey of hallucination in natural language generation. ACM Computing Surveys, 55(12):1–38.

Erik Jones and Jacob Steinhardt. 2022. Capturing failures of large language models via human cognitive biases. Advances in Neural Information Processing Systems, 35:11785–11799.

Svetlana Kiritchenko and Saif Mohammad. 2018. Examining gender and race bias in two hundred sentiment analysis systems. In Proceedings of the Seventh Joint Conference on Lexical and Computational Semantics, pages 43–53, New Orleans, Louisiana. Association for Computational Linguistics.

Hannah Rose Kirk, Yennie Jun, Filippo Volpin, Haider Iqbal, Elias Benussi, Frederic Dreyer, Aleksandar Shtedritski, and Yuki Asano. 2021. Bias out-of-thebox: An empirical analysis of intersectional occupational biases in popular generative language models. Advances in neural information processing systems, 34:2611–2624.

Hadas Kotek, Rikker Dockum, and David Sun. 2023. Gender bias and stereotypes in large language models. In Proceedings of the ACM collective intelligence conference, pages 12–24.

Evelina Leivada, Elliot Murphy, and Gary Marcus. 2023. Dall· e 2 fails to reliably capture common syntactic processes. Social Sciences & Humanities Open, 8(1):100648.

Olivia Macmillan-Scott and Mirco Musolesi. 2024. (ir) rationality and cognitive biases in large language models. Royal Society Open Science, 11(6):240255.

Rowan Hall Maudslay, Hila Gonen, Ryan Cotterell, and Simone Teufel. 2019. It‘s all in the name: Mitigating gender bias with name-based counterfactual data substitution. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 5267–5275, Hong Kong, China. Association for Computational Linguistics.

David E Meyer and Roger W Schvaneveldt. 1971. Facilitation in recognizing pairs of words: evidence of a dependence between retrieval operations. Journal of experimental psychology, 90(2):227.

Moin Nadeem, Anna Bethke, and Siva Reddy. 2021. Stereoset: Measuring stereotypical bias in pretrained language models. In Proceedings ofthe 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 5356–5371.

Roberto Navigli, Simone Conia, and Björn Ross. 2023. Biases in large language models: origins, inventory, and discussion. ACM Journal ofData and Information Quality.

James H Neely. 1976. Semantic priming and retrieval from lexical memory: Evidence for facilitatory and inhibitory processes. Memory & cognition, 4(5):648– 654.

Debora Nozza, Federico Bianchi, and Dirk Hovy. 2021. HONEST: Measuring hurtful sentence completion in language models. In Proceedings of the 2021 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies, pages 2398–2406, Online. Association for Computational Linguistics.

Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. 2022. Hierarchical textconditional image generation with clip latents. arXiv preprint arXiv:2204.06125.

Royi Rassin, Eran Hirsch, Daniel Glickman, Shauli Ravfogel, Yoav Goldberg, and Gal Chechik. 2024. Linguistic binding in diffusion models: Enhancing attribute correspondence through attention map alignment. Advances in Neural Information Processing Systems, 36.

Royi Rassin, Shauli Ravfogel, and Yoav Goldberg. 2022. Dalle-2 is seeing double: Flaws in word-to-concept mapping in text2image models. In Proceedings of the Fifth BlackboxNLP Workshop on Analyzing and Interpreting Neural Networks for NLP, pages 335– 345.

Nils Reimers and Iryna Gurevych. 2019. Sentence-bert: Sentence embeddings using siamese bert-networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing.

Paul Röttger, Hannah Kirk, Bertie Vidgen, Giuseppe Attanasio, Federico Bianchi, and Dirk Hovy. 2024. XSTest: A test suite for identifying exaggerated safety behaviours in large language models. In Proceedings ofthe 2024 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 5377–5400, Mexico City, Mexico. Association for Computational Linguistics.

Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilic, Daniel Hesslow, Roman´ Castagné, Alexandra Sasha Luccioni, François Yvon, et al. 2022. Bloom: A 176b-parameter openaccess multilingual language model. arXiv preprint arXiv:2211.05100.

Timo Schick, Sahana Udupa, and Hinrich Schütze. 2021. Self-diagnosis and self-debiasing: A proposal for reducing corpus-based bias in nlp. Transactions of the Association for Computational Linguistics, 9:1408– 1424.

Melanie Sclar, Yejin Choi, Yulia Tsvetkov, and Alane Suhr. 2024. Quantifying language models’ sensitivity to spurious features in prompt design or: How i learned to start worrying about prompt formatting. In Proceedings ofInternational Conference on Learning Representations.

Mrinank Sharma, Meg Tong, Tomasz Korbak, David Duvenaud, Amanda Askell, Samuel R Bowman, Newton Cheng, Esin Durmus, Zac Hatfield-Dodds, Scott R Johnston, et al. 2024. Towards understanding sycophancy in language models. In Proceedings ofInternational Conference on Learning Representations.

Gabriel Stanovsky, Noah A. Smith, and Luke Zettlemoyer. 2019. Evaluating gender bias in machine translation. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 1679–1684, Florence, Italy. Association for Computational Linguistics.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Endel Tulving and Daniel L Schacter. 1990. Priming and human memory systems. Science, 247(4940):301–306.

Endel Tulving, Daniel L Schacter, and Heather A Stark. 1982. Priming effects in word-fragment completion are independent of recognition memory. Journal of experimental psychology: learning, memory, and cognition, 8(4):336.

Pranav Narayanan Venkit, Sanjana Gautam, Ruchi Panchanadikar, Ting-Hao’Kenneth’ Huang, and Shomir Wilson. 2023. Nationality bias in text generation. arXiv preprint arXiv:2302.02463.

Pranav Narayanan Venkit, Mukund Srinath, and Shomir Wilson. 2022. A study of implicit bias in pretrained language models against people with disabilities. In Proceedings of the 29th International Conference on Computational Linguistics, pages 1324–1332, Gyeongju, Republic of Korea. International Committee on Computational Linguistics.

Eric Wallace, Shi Feng, Nikhil Kandpal, Matt Gardner, and Sameer Singh. 2019. Universal adversarial triggers for attacking and analyzing NLP. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 2153–2162, Hong Kong, China. Association for Computational Linguistics.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, et al. 2019. Huggingface’s transformers: State-ofthe-art natural language processing. arXiv preprint arXiv:1910.03771.

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q Weinberger, and Yoav Artzi. 2020. Bertscore: Evaluating text generation with bert. In Proceedings of International Conference on Learning Representations.

Yuji Zhang, Sha Li, Jiateng Liu, Pengfei Yu, Yi R Fung, Jing Li, Manling Li, and Heng Ji. 2024. Knowledge overshadowing causes amalgamated hallucination in large language models. arXiv preprint arXiv:2407.08039.

Jieyu Zhao, Tianlu Wang, Mark Yatskar, Vicente Ordonez, and Kai-Wei Chang. 2018. Gender bias in coreference resolution: Evaluation and debiasing methods. In Proceedings of the 2018 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers), pages 15–20, New Orleans, Louisiana. Association for Computational Linguistics.

## A Semantic Leakage in GPT Models

Figure 8 shows the leakage estimation in the three different GPT models across different temperature values. GPT4O consistently leaks more than GPT4 and GPT-3.5.

![](images/0d73cafebace73305f1e1fcf1b8fdb4c9b514673e12d52c1f3230ae729c554fd.jpg)  
Figure 8: Semantic leakage in GPT models using different temperature values (measured with Leak-Rate using BERT-score).

## B Multilingual and Crosslingual Semantic Leakage

Table 4 shows example multilingual and crosslingual prompts and generations.

## C Open-Ended Generation

Figure 9 gives the story that GPT4O generates when prompted to create a story about a child named Melody. The generated story contains many words and elements directly related to the semantics of the name, and the narrative of the story is tailored to that meaning.

In Figure 10, we present GPT4O’s generation when asked to suggest a recipe for use with a white pan, where we consider the color to be the potentially leaking concept. The model suggests to make chicken alfredo, matching the color of the pan.

![](images/f40441e5c5bf7a7c88714c4d924703b4a3538fdb19478974262c82af0a457782.jpg)  
Figure 9: Example of story generation, exhibiting semantic leakage based on the name in the prompt.

![](images/2cc75733ff2f4e6b71f717a655d17eb7e42fb54d18318ac139a48e0bf70719b5.jpg)  
Figure 10: Example of recipe generation, exhibiting semantic leakage based on the color in the prompt.

<table><tr><td>Language</td><td>Prompt</td><td>Generation</td></tr><tr><td>Heb</td><td>N y 2N SN .NP 27 2N N</td><td>DIDSPN</td></tr><tr><td>Heb-En</td><td>His favorite food is. ユ </td><td>eucalyptus leaves</td></tr><tr><td> $\mathrm { Z h }$ </td><td>他喜欢考拉。他最喜欢吃</td><td>桉树叶</td></tr><tr><td>Zh-En</td><td>他喜欢考拉。His favorite food is</td><td>eucalyptus leaves</td></tr></table>

Table 4: Example prompts and generations in Hebrew, Hebrew-English, Chinese and Chinese-English, by this order. All are translations of the prompt: “He likes koalas. His favorite food $i s ^ { \prime \prime }$ , and the Hebrew/Chinese generations can be translated to “eucalyptus”/“eucalyptus leaves”, respectively.