# The Impact of Visual Information in Chinese Characters: Evaluating Large Models’ Ability to Recognize and Utilize Radicals

Xiaofeng Wu<sup>1</sup> Karl Stratos<sup>2</sup> Wei Xu<sup>1</sup>

<sup>1</sup>Georgia Institute of Technology <sup>2</sup>Rutgers University xwu414@gatech.edu, karl.stratos@rutgers.edu, wei.xu@cc.gatech.edu

## Abstract

The glyphic writing system of Chinese incorporates information-rich visual features in each character, such as radicals that provide hints about meaning or pronunciation. However, there has been no investigation into whether contemporary Large Language Models (LLMs) and Vision-Language Models (VLMs) can harness these sub-character features in Chinese through prompting. In this study, we establish a benchmark<sup>1</sup> to evaluate LLMs’ and VLMs’ understanding of visual elements in Chinese characters, including radicals, composition structures, strokes, and stroke counts. Our results reveal that models surprisingly exhibit some, but still limited, knowledge of the visual information, regardless of whether images of characters are provided. To incite models’ ability to use radicals, we further experiment with incorporating radicals into the prompts for Chinese language processing (CLP) tasks. We observe consistent improvement in Part-Of-Speech tagging when providing additional information about radicals, suggesting the potential to enhance CLP by integrating sub-character information.

## 1 Introduction

Visual information embedded in Chinese characters is important, as most Chinese characters convey a meaning equivalent to an entire word in English with a complex glyphic structure. Multiple writing strokes form the radicals,<sup>2</sup> which often carry information about semantic meaning and pronunciation; the radicals are then visually combined to form Chinese characters. When encountering unfamiliar characters, Chinese speakers rely on semantic and phonetic hints from radicals, much like how English speakers use sub-words such as prefixes or suffixes, to approximate the meaning and pronunciation of unfamiliar words. For example, the Chinese character “花” (meaning “flower”; pronounced as “hua”) in Figure ¯ 1 has “艹” (meaning “herbal”) on the top, contributing to its semantic meaning, and “化” (pronounces as “huà”) on the bottom, indicating its pronunciation. By utilizing the radical information, one can infer that “花” is related to herbs and has a pronunciation similar to “huà” without prior knowledge of the character.

![](images/f3829e26c08b45a267c07a7aa74a5b3a0873cf31dad7e885dd3d584f1ff52e7a.jpg)  
Figure 1: Chinese character “花” displayed at the character, radical, and stroke levels from left to right. Different radicals are shown in green, yellow, and pink colors, while the writing order of the strokes is indicated by red (current), gray (upcoming), and black (completed).

Although radicals contain rich information, they have received little attention in digital text processing. Contemporary typefaces treat Chinese characters, radicals, and strokes as indivisible units, disregarding their compositional relationships. Consequently, most language models follow this approach, under-utilizing the rich visual and semantic information embedded in Chinese characters. While limited prior works (Sun et al., 2021; Si et al., 2021; Stratos, 2017) have attempted to address this issue by incorporating visual embeddings, such as strokes or font images, into smaller-scale models (i.e., BERT or Neural MT), there remains a lack of research investigating whether these visual features can be recognized and utilized by models in light of the significant advancements in LLMs and VLMs, especially in inference methods (e.g., prompting).

To determine whether pre-trained LLMs recognize or can acquire the visual knowledge embedded in Chinese characters, we establish a benchmark by collecting over 14,000 Chinese characters from the Chinese, Japanese, and Korean (CJK) Unified Ideographs,<sup>3</sup> considering four visual elements: radicals, composition structures, strokes, and stroke counts. As shown in Figure 3, the composition structure refers to the visual arrangement of a character’s radicals (e.g., top-to-bottom or left-to-right). Stroke composition provides an essential way to represent not typable radicals; As shown in Figure 2, some radicals cannot be typed using standard input methods but can still be accurately depicted through stroke compositions (see more details in Appendix D.2). Lastly, Stroke count offers a measure for Chinese characters’ visual complexity, similar to word length in English.

We conduct experiments on four tasks: structure recognition, radical recognition, stroke count identification, and stroke identification. We evaluated a series of LLMs and VLMs (e.g., GPT-4, Gemini-1.5, Ernie-4, Aya-command, QWen-7B, etc.), and found that all models possess some visual knowledge of Chinese characters, even without image inputs; however, it is only to a limited extent. In particular, the models tend to perform well in recognizing the first radical of a Chinese character, such as “艹” (herbal) in “花”(flower), but often fail with subsequent ones. We also demonstrate that the pixel-based encoder PIXEL (Razzhigaev et al., 2022) has the ability to capture structural information effectively after fine-tuning. As a language model pre-trained only on an English corpus, PIXEL achieved an F1 score of 84.57, significantly higher than the second-best score of 54.30 achieved by Ernie-4<sup>4</sup> and 23.29 by GPT-4 when provided with images of characters, indicating its potential for CLP as it naturally captures visual information.

We further investigate whether models can utilize radicals to improve performance on understanding tasks (e.g., POS tagging and NER) by prompting them to use radicals when encountering unfamiliar words. Our experiments show that radical information yields promising results in downstream tasks, particularly in POS tagging. We observe consistent improvement across models and datasets when the information about radicals is provided. Notably, Ernie-Lite-8K’s POS tagging F1 score on GSD(Qi and Yasuoka, 2023) decreases by 2.1 points when recognizing radicals on its own, but increases by 5.7 points when provided with correct radicals. For Name Entity Recognition (NER), we also observe an improvement in three out of six models. Analyzing the cases where incorporating radical degrades the model performance, we see that incorrect answers often occur when the model fails to identify unfamiliar words and bypasses the radical information process, indicating the decrease is likely due to long prompts. When evaluating only sentences where the model detects unfamiliar words, performance on NER generally improves. Our work demonstrates that models possess the ability to recognize and utilize radical information only to a certain limit, suggesting that deeper integration, such as additional training on radicals or improvement in the Chinese digital system to incorporate radicals, could unlock further potential.

![](images/bbe71d31b8c70fbc7f11e31019202e58f8373216516cbd1cc87bea64abbd20ed.jpg)  
Figure 2: Example images of untypable radicals alongside example characters and their corresponding Unicode values, if available.

## 2 Related Work

Chinese Character Decomposition in Computer Vision. The task of decomposing Chinese characters into constituent components has majorly been studied in the field of computer vision. Research within this domain, such as the studies by Ma et al. (2021), Xia (1994), and Liu et al. (2021), has explored analogous challenges. The work by Zhang et al. (2018) employs a methodical approach by categorizing characters into structured types and further decomposing sub-components according to their spatial arrangements—akin to the layered structural analysis that we adopt in this paper.

Chinese Decomposition Datasets. A comprehensive dataset (Kawabata et al., 2018) that offers decompositions for the CJK Unified Ideographs. Although this collection overlaps with our dataset, it does not cite any authoritative sources for its data. This omission leads to ambiguity due to multiple decomposition sequences for individual characters.

Our approach utilizes sources from authoritative Chinese dictionaries, such as the Kangxi Dictionary (康熙字典) and the Xinhua Dictionary (新华 字典), ensuring a validated framework for visual information. Additionally, our dataset contains standard stroke orders for all 14,648 characters, which the aforementioned dataset lacks.

Glyphic Embedding Strategies in LMs. Few prior works have utilized the idea of adding additional input embedding with Chinese visual features. For instance, (Shi et al., 2015) attempted to add radical embedding in the pre-transformer era. (Sun et al., 2021) introduced font images into embedding, and (Si et al., 2021) experimented with stroke among other glyph-based embeddings such as Cangjie<sup>5</sup> (仓颉). Another interesting approach is PIXEL (Razzhigaev et al., 2022), which uses a pixel-based encoder to transform input into images. Our experiment results in §4 highlights the potential of pixel-based language models in CLP.

## 3 Chinese Character Dataset (CCD)

To evaluate contemporary LLMs and VLMs’ proficiency with visual information in Chinese characters, we compile a dataset using characters from CJK Unified Ideographs with visual features collected from the digitized Kangxi Dictionary and Xinhua Dictionary. Our dataset includes 14,648 Chinese characters and details their corresponding radicals, strokes, and stroke count. A subset of 4,651 Simplified Chinese characters also contains structural composition information. The detailed statistics are provided in Table 1 with three tiers of character frequency based on the Table of General Standard Chinese Characters (通用规范汉字表).<sup>6</sup>

Structure of Chinese Characters. According to the digitized Kangxi Dictionary,<sup>7</sup> we categorize 4651 simplified Chinese characters into eight major structural arrangements, with examples of each structure illustrated in Figure 3: top-bottom, left-right, top-mid-bottom, left-mid-right, wrapping, inlay, triple-stack, as well as single structure, which refers to characters that cannot be further segmented. The structure of Chinese characters can be rather complex. For example, the character “花”, shown in Figure 1, has a top-bottom structure, consisting of “艹” and “化”. Furthermore, “化”

exhibits a left-right structure which can be further decomposed into “亻” and “七.” For the sake of clarity, we categorize all characters based on their top-level structure (i.e., top-bottom for “花”).

Radicals of Chinese Characters. Based on the aforementioned structural composition, we collect radicals by decomposing each Chinese character into meaningful components that have corresponding Unicode representations. The radicals are then ordered according to specific rules: from top to bottom, left to right, outside to inside, and main parts before inlay parts, as illustrated in Figure 3. We avoid decomposition for simple Chinese characters that would reduce to meaningless strokes. For example, while the character “八” could be segmented as a left-right structure, we classify it as a single structure with only one radical to prevent it from being reduced to meaningless strokes.

The initial collection of radicals was performed using APISpace’s Chinese character segmentation API,<sup>8</sup> which analyzes characters’ stroke compositions and extracts radicals based on the optimal subsequences of strokes. After the automated annotation, we conducted a thorough two-round manual review to ensure accuracy. More than 1,000 characters required manual adjustments due to missing or incorrect radicals; in addition, another 500 adjustments were made to prevent unnecessary reduction of characters into strokes by one of the native Chinese-speaking authors. Four native Chinese speakers further reviewed the entire dataset and collectively corrected for about 2% annotations. Further details are provided in the Appendix B.

Stroke Composition of Chinese Characters. Stroke composition refers to the sequence of writing order of a Chinese character’s strokes. Chinese dictionaries categorize all Chinese strokes into five basic stroke types: “一”, “丨”, “ノ”, “丶”, and “フ’, which our dataset adopts. We first utilized the Xinhua Dictionary API to automatically annotate. For characters not found in the API, we attempted to concatenate the stroke composition of their radicals in order. We then manually reviewed all stroke compositions to ensure accuracy.

Stroke Count of Chinese Characters. The number of strokes required to write a Chinese character is also present in our dataset, which offers a measure of word complexity. Unlike the alphabetic writing system, where word length can hint at complexity, Chinese characters occupy a uniform length of one, making stroke count a valuable indicator of intricacy. The statistics for strokes are provided in Table 1 with illustrations in Figure 3.

![](images/65818a48752dc23f6fdd4067f9ddae27f8c8bfb9f20f621d527e9ecab2b58264.jpg)  
Figure 3: Examples of composition structures with radicals in order of black, red, yellow and four different tasks.

<table><tr><td>Character Level</td><td>Statistic</td></tr><tr><td># of total Chinese characters - Commonly used (tier 1): - Less commonly used (tier 2): - Terminology used (tier 3): - Hardly ever used (others):</td><td>14,648 3,500 (24.1%) 3,000 (20.6%) 1,605 (11.0%)</td></tr><tr><td>- w/ structure information:</td><td>5,543 (37.8%) 4,651 (31.8%)</td></tr><tr><td>Radical Level</td><td></td></tr><tr><td># of unique radical # of single-appearance radical</td><td>2,132 692</td></tr><tr><td>Stroke Level</td><td></td></tr><tr><td># of unique stroke composition</td><td>13,740</td></tr><tr><td># of strokes per character (µ)</td><td>11.51</td></tr><tr><td># of strokes per character (σ) # of strokes per character (min)</td><td>3.92 1 39</td></tr></table>

Table 1: Key statistics of our Chinese character dataset.

## 4 Evaluation on Recognizing Visual Information of Chinese Characters

To evaluate whether language models contain or can learn the visual information embedded in Chinese characters, we establish a benchmark by set-

ting up a series of tasks (see examples of each task in Figure 3) derived from our dataset.

## 4.1 Chinese Character Tasks

Structure Recognition of Chinese Characters. We assess LLMs and VLMs’ proficiency in identifying the correct structural arrangements of Chinese characters using a multiple-choice format. We present the character with eight structure types and evaluate the model’s answer using the F1 score.

Radical Recognition of Chinese Characters. We evaluate LLMs and VLMs’ ability to recognize radical information in two tasks: character-toradical and radical-to-character. For the first task, models receive character and order guidelines and are prompted to identify their radicals in sequence. Performance is measured by the accuracy of each radical in order and the overall F1 score. For the second task, models are given radicals and their relative positions and asked to identify the correct characters, with accuracy as the metric.

Stroke Count Identification of Chinese Characters. We evaluate models’ ability to identify the number of strokes required to write query characters with performance measured using Mean Absolute Error (MAE) and Mean Squared Error (MSE).

Stroke Decomposition of Chinese Characters. Similar to the radical-to-character task, we evaluate

<table><tr><td rowspan=3 colspan=3>StructureModelF1  H↑   ↓</td><td rowspan=2 colspan=6>Radicals1st 2nd 3rd  F1  H</td><td rowspan=1 colspan=2>Stroke Ct.</td><td rowspan=1 colspan=5>Stroke Composition</td></tr><tr><td rowspan=1 colspan=1>Acc</td><td rowspan=1 colspan=2>MSE MAE</td><td rowspan=2 colspan=5>1st 2nd 3rd  F1  HAccAcc Acc  ↑  ↓</td></tr><tr><td rowspan=1 colspan=5>Acc Acc Acc  ↑   ↓</td><td rowspan=1 colspan=1>↑</td><td rowspan=1 colspan=2>↓    ↓</td></tr><tr><td rowspan=1 colspan=16>Vision Language Models (VLMs)</td></tr><tr><td rowspan=1 colspan=3>文Ernie-4            54.30</td><td rowspan=1 colspan=1>41.03</td><td rowspan=1 colspan=1>34.21</td><td rowspan=1 colspan=1>12.50</td><td rowspan=1 colspan=2>41.67</td><td rowspan=1 colspan=1>71.79</td><td rowspan=1 colspan=1>12.54</td><td rowspan=1 colspan=1>1.78</td><td rowspan=1 colspan=1>53.85</td><td rowspan=1 colspan=1>35.90</td><td rowspan=1 colspan=1>47.37</td><td rowspan=1 colspan=1>30.90</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>文Kimi-v1            45.60</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>36.73</td><td rowspan=1 colspan=1>19.15</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=2>32.93</td><td rowspan=1 colspan=1>42.86</td><td rowspan=1 colspan=1>15.32</td><td rowspan=1 colspan=1>2.68</td><td rowspan=1 colspan=1>30.61</td><td rowspan=1 colspan=1>26.53</td><td rowspan=1 colspan=1>16.67</td><td rowspan=1 colspan=1>20.70</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>Claude-3             23.70</td><td rowspan=1 colspan=1>0.54</td><td rowspan=1 colspan=1>8.80</td><td rowspan=1 colspan=1>0.61</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=2>2.441.09</td><td rowspan=1 colspan=1>57.30</td><td rowspan=1 colspan=1>5.93</td><td rowspan=1 colspan=1>1.22</td><td rowspan=1 colspan=1>15.40</td><td rowspan=1 colspan=1>19.60</td><td rowspan=1 colspan=1>26.80</td><td rowspan=1 colspan=1>19.62</td><td rowspan=1 colspan=1>1.22</td></tr><tr><td rowspan=1 colspan=2>Gemini-1.5           27.15</td><td rowspan=1 colspan=1>0.36</td><td rowspan=1 colspan=1>3.00</td><td rowspan=1 colspan=1>0.41</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>1.53</td><td rowspan=1 colspan=1>1.12</td><td rowspan=1 colspan=1>27.08</td><td rowspan=1 colspan=1>8.83</td><td rowspan=1 colspan=1>2.28</td><td rowspan=1 colspan=1>29.60</td><td rowspan=1 colspan=1>16.80</td><td rowspan=1 colspan=1>22.00</td><td rowspan=1 colspan=1>22.04</td><td rowspan=1 colspan=1>1.00</td></tr><tr><td rowspan=1 colspan=3>GPT-4               23.280.46</td><td rowspan=1 colspan=1>10.20</td><td rowspan=1 colspan=1>0.41</td><td rowspan=1 colspan=2>0.00 9.22</td><td rowspan=1 colspan=1>0.95</td><td rowspan=1 colspan=1>24.18</td><td rowspan=1 colspan=1>7.96</td><td rowspan=1 colspan=1>1.64</td><td rowspan=1 colspan=1>24.00</td><td rowspan=1 colspan=1>19.60</td><td rowspan=1 colspan=1>23.80</td><td rowspan=1 colspan=1>21.96</td><td rowspan=1 colspan=1>1.34</td></tr><tr><td rowspan=1 colspan=3>GPT-40              26.66</td><td rowspan=1 colspan=2>6.000.40</td><td rowspan=1 colspan=3>0.00 6.620.74</td><td rowspan=1 colspan=1>67.39</td><td rowspan=1 colspan=1>10.35</td><td rowspan=1 colspan=1>1.72</td><td rowspan=1 colspan=2>46.4031.40</td><td rowspan=1 colspan=1>34.80</td><td rowspan=1 colspan=1>30.85</td><td rowspan=1 colspan=1>0.70</td></tr><tr><td rowspan=1 colspan=16>Close-Sourced Models (LLMs)</td></tr><tr><td rowspan=1 colspan=2>Ernie-Lite-8K       7.19</td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1>18.92</td><td rowspan=1 colspan=1>3.52</td><td rowspan=1 colspan=1>0.13</td><td rowspan=1 colspan=1>11.99</td><td rowspan=1 colspan=1>1.89</td><td rowspan=1 colspan=1>3.72</td><td rowspan=1 colspan=1>44.53</td><td rowspan=1 colspan=1>5.34</td><td rowspan=1 colspan=1>29.30</td><td rowspan=1 colspan=1>23.28</td><td rowspan=1 colspan=1>20.78</td><td rowspan=1 colspan=1>23.34</td><td rowspan=1 colspan=1>1.11</td></tr><tr><td rowspan=1 colspan=2>文Kimi-v1            24.51</td><td rowspan=1 colspan=1>0.83</td><td rowspan=1 colspan=1>7.24</td><td rowspan=1 colspan=1>0.33</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>1.10</td><td rowspan=1 colspan=1>0.72</td><td rowspan=1 colspan=1>50.16</td><td rowspan=1 colspan=1>19.05</td><td rowspan=1 colspan=1>3.12</td><td rowspan=1 colspan=1>33.12</td><td rowspan=1 colspan=1>21.56</td><td rowspan=1 colspan=1>19.72</td><td rowspan=1 colspan=1>22.99</td><td rowspan=1 colspan=1>1.07</td></tr><tr><td rowspan=1 colspan=2>Aya-command        12.56</td><td rowspan=1 colspan=1>0.16</td><td rowspan=1 colspan=1>35.72</td><td rowspan=1 colspan=1>2.16</td><td rowspan=1 colspan=1>0.26</td><td rowspan=1 colspan=2>20.130.73</td><td rowspan=1 colspan=1>5.65</td><td rowspan=1 colspan=1>13.20</td><td rowspan=1 colspan=1>2.79</td><td rowspan=1 colspan=1>28.24</td><td rowspan=1 colspan=1>23.48</td><td rowspan=1 colspan=1>19.44</td><td rowspan=1 colspan=1>21.43</td><td rowspan=1 colspan=1>0.37</td></tr><tr><td rowspan=1 colspan=2>Claude-3             23.70</td><td rowspan=1 colspan=1>0.54</td><td rowspan=1 colspan=1>70.02</td><td rowspan=1 colspan=1>5.64</td><td rowspan=1 colspan=1>0.43</td><td rowspan=1 colspan=1>45.57</td><td rowspan=1 colspan=1>1.09</td><td rowspan=1 colspan=1>40.40</td><td rowspan=1 colspan=1>7.78</td><td rowspan=1 colspan=1>1.32</td><td rowspan=1 colspan=1>28.64</td><td rowspan=1 colspan=1>19.02</td><td rowspan=1 colspan=1>31.19</td><td rowspan=1 colspan=1>22.91</td><td rowspan=1 colspan=1>0.88</td></tr><tr><td rowspan=1 colspan=2>Gemini-1.5           23.04</td><td rowspan=1 colspan=1>0.56</td><td rowspan=1 colspan=1>4.20</td><td rowspan=1 colspan=1>0.04</td><td rowspan=1 colspan=1>0.38</td><td rowspan=1 colspan=1>1.37</td><td rowspan=1 colspan=1>1.16</td><td rowspan=1 colspan=1>11.26</td><td rowspan=1 colspan=1>13.23</td><td rowspan=1 colspan=1>2.76</td><td rowspan=1 colspan=1>26.66</td><td rowspan=1 colspan=1>24.52</td><td rowspan=1 colspan=1>15.14</td><td rowspan=1 colspan=1>20.24</td><td rowspan=1 colspan=1>0.81</td></tr><tr><td rowspan=1 colspan=2>Few-shot GPT-3.5     22.82</td><td rowspan=1 colspan=1>0.88</td><td rowspan=1 colspan=1>54.14</td><td rowspan=1 colspan=1>7.37</td><td rowspan=1 colspan=1>0.30</td><td rowspan=1 colspan=1>34.60</td><td rowspan=1 colspan=1>1.21</td><td rowspan=1 colspan=1>23.12</td><td rowspan=1 colspan=1>7.96</td><td rowspan=1 colspan=1>1.65</td><td rowspan=1 colspan=1>27.86</td><td rowspan=1 colspan=1>22.70</td><td rowspan=1 colspan=1>30.23</td><td rowspan=1 colspan=1>25.62</td><td rowspan=1 colspan=1>1.13</td></tr><tr><td rowspan=1 colspan=2>Zero-shot GPT-3.5    15.43</td><td rowspan=1 colspan=1>0.69</td><td rowspan=1 colspan=1>52.14</td><td rowspan=1 colspan=1>4.33</td><td rowspan=1 colspan=1>0.20</td><td rowspan=1 colspan=1>31.66</td><td rowspan=1 colspan=1>1.30</td><td rowspan=1 colspan=1>17.45</td><td rowspan=1 colspan=1>10.80</td><td rowspan=1 colspan=1>2.17</td><td rowspan=1 colspan=1>30.70</td><td rowspan=1 colspan=1>21.92</td><td rowspan=1 colspan=1>26.97</td><td rowspan=1 colspan=1>25.09</td><td rowspan=1 colspan=1>0.98</td></tr><tr><td rowspan=1 colspan=1>Fine-tune GPT-3.5</td><td rowspan=1 colspan=1>27.14</td><td rowspan=1 colspan=1>0.33</td><td rowspan=1 colspan=1>4.12</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>1.23</td><td rowspan=1 colspan=1>1.11</td><td rowspan=1 colspan=1>71.66</td><td rowspan=1 colspan=1>7.36</td><td rowspan=1 colspan=1>1.46</td><td rowspan=1 colspan=1>47.50</td><td rowspan=1 colspan=1>44.58</td><td rowspan=1 colspan=1>32.67</td><td rowspan=1 colspan=1>28.64</td><td rowspan=1 colspan=1>1.08</td></tr><tr><td rowspan=1 colspan=1>CoT GPT-3.5</td><td rowspan=1 colspan=1>38.08</td><td rowspan=1 colspan=1>1.25</td><td rowspan=1 colspan=1>5.24</td><td rowspan=1 colspan=1>0.16</td><td rowspan=1 colspan=1>0.11</td><td rowspan=1 colspan=1>1.63</td><td rowspan=1 colspan=1>1.05</td><td rowspan=1 colspan=1>24.41</td><td rowspan=1 colspan=1>8.93</td><td rowspan=1 colspan=1>1.92</td><td rowspan=1 colspan=1>31.06</td><td rowspan=1 colspan=1>22.22</td><td rowspan=1 colspan=1>26.85</td><td rowspan=1 colspan=1>25.60</td><td rowspan=1 colspan=1>0.83</td></tr><tr><td rowspan=1 colspan=1>Few-shot GPT-4</td><td rowspan=1 colspan=1>45.28</td><td rowspan=1 colspan=1>0.48</td><td rowspan=1 colspan=1>58.44</td><td rowspan=1 colspan=1>6.45</td><td rowspan=1 colspan=1>0.31</td><td rowspan=1 colspan=1>41.66</td><td rowspan=1 colspan=1>0.84</td><td rowspan=1 colspan=1>38.01</td><td rowspan=1 colspan=1>7.96</td><td rowspan=1 colspan=1>1.65</td><td rowspan=1 colspan=1>24.18</td><td rowspan=1 colspan=1>18.22</td><td rowspan=1 colspan=1>21.90</td><td rowspan=1 colspan=1>20.87</td><td rowspan=1 colspan=1>1.37</td></tr><tr><td rowspan=1 colspan=3>Zero-shot GPT-4      35.400.54</td><td rowspan=1 colspan=1>57.86</td><td rowspan=1 colspan=1>6.28</td><td rowspan=1 colspan=1>0.20</td><td rowspan=1 colspan=1>41.42</td><td rowspan=1 colspan=1>0.88</td><td rowspan=1 colspan=1>38.76</td><td rowspan=1 colspan=1>12.17</td><td rowspan=1 colspan=1>1.99</td><td rowspan=1 colspan=1>27.04</td><td rowspan=1 colspan=1>21.16</td><td rowspan=1 colspan=1>21.99</td><td rowspan=1 colspan=1>22.18</td><td rowspan=1 colspan=1>1.21</td></tr><tr><td rowspan=1 colspan=16>Open-Sourced Models (LLMs)</td></tr><tr><td rowspan=1 colspan=2>Baichuan-13B      11.17</td><td rowspan=1 colspan=1>0.88</td><td rowspan=1 colspan=1>33.20</td><td rowspan=1 colspan=1>2.05</td><td rowspan=1 colspan=1>0.60</td><td rowspan=1 colspan=1>22.62</td><td rowspan=1 colspan=1>1.20</td><td rowspan=1 colspan=1>13.67</td><td rowspan=1 colspan=2>32.70 4.31</td><td rowspan=1 colspan=1>27.68</td><td rowspan=1 colspan=1>21.42</td><td rowspan=1 colspan=2>15.9222.74</td><td rowspan=1 colspan=1>1.56</td></tr><tr><td rowspan=1 colspan=2>文ChatGLM-6B       10.30</td><td rowspan=1 colspan=1>0.68</td><td rowspan=1 colspan=1>6.94</td><td rowspan=1 colspan=1>0.50</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>6.33</td><td rowspan=1 colspan=1>1.35</td><td rowspan=1 colspan=1>1.38</td><td rowspan=1 colspan=2>29.68 4.25</td><td rowspan=1 colspan=1>26.88</td><td rowspan=1 colspan=1>12.60</td><td rowspan=1 colspan=1>12.43</td><td rowspan=1 colspan=1>27.28</td><td rowspan=1 colspan=1>0.96</td></tr><tr><td rowspan=1 colspan=2>Chinese-LLaMA-7B5.13</td><td rowspan=1 colspan=1>0.97</td><td rowspan=1 colspan=1>9.26</td><td rowspan=1 colspan=1>0.64</td><td rowspan=1 colspan=1>0.17</td><td rowspan=1 colspan=1>6.32</td><td rowspan=1 colspan=1>1.92</td><td rowspan=1 colspan=1>0.32</td><td rowspan=1 colspan=1>15.83</td><td rowspan=1 colspan=1>3.00</td><td rowspan=1 colspan=1>26.26</td><td rowspan=1 colspan=1>24.86</td><td rowspan=1 colspan=1>13.42</td><td rowspan=1 colspan=1>22.32</td><td rowspan=1 colspan=1>0.93</td></tr><tr><td rowspan=1 colspan=1>&amp;InternLM-7B</td><td rowspan=1 colspan=1>9.68</td><td rowspan=1 colspan=1>1.05</td><td rowspan=1 colspan=1>12.08</td><td rowspan=1 colspan=1>0.34</td><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>8.89</td><td rowspan=1 colspan=1>1.50</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>45.38</td><td rowspan=1 colspan=1>5.50</td><td rowspan=1 colspan=1>28.82</td><td rowspan=1 colspan=1>24.66</td><td rowspan=1 colspan=1>13.38</td><td rowspan=1 colspan=1>22.01</td><td rowspan=1 colspan=1>0.95</td></tr><tr><td rowspan=1 colspan=1>文Yi-6B</td><td rowspan=1 colspan=1>8.86</td><td rowspan=1 colspan=1>0.70</td><td rowspan=1 colspan=1>14.18</td><td rowspan=1 colspan=1>1.05</td><td rowspan=1 colspan=1>0.21</td><td rowspan=1 colspan=2>12.141.40</td><td rowspan=1 colspan=1>0.32</td><td rowspan=1 colspan=1>29.49</td><td rowspan=1 colspan=1>4.24</td><td rowspan=1 colspan=1>28.56</td><td rowspan=1 colspan=1>22.40</td><td rowspan=1 colspan=1>7.76</td><td rowspan=1 colspan=2>24.170.85</td></tr><tr><td rowspan=1 colspan=1>Bloom-7B</td><td rowspan=1 colspan=1>9.81</td><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>3.48</td><td rowspan=1 colspan=1>0.54</td><td rowspan=1 colspan=1>0.04</td><td rowspan=1 colspan=2>4.151.70</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=2>46.76 4.05</td><td rowspan=1 colspan=1>27.92</td><td rowspan=1 colspan=1>24.96</td><td rowspan=1 colspan=1>14.47</td><td rowspan=1 colspan=2>23.190.87</td></tr><tr><td rowspan=1 colspan=1>Qwen-7B</td><td rowspan=1 colspan=1>5.25</td><td rowspan=1 colspan=1>1.16</td><td rowspan=1 colspan=1>17.30</td><td rowspan=1 colspan=1>0.85</td><td rowspan=1 colspan=1>0.23</td><td rowspan=1 colspan=2>12.411.50</td><td rowspan=1 colspan=1>1.59</td><td rowspan=1 colspan=2>34.16 4.62</td><td rowspan=1 colspan=1>25.02</td><td rowspan=1 colspan=1>20.20</td><td rowspan=1 colspan=1>21.92</td><td rowspan=1 colspan=2>23.301.30</td></tr><tr><td rowspan=1 colspan=1>Qwen-2-7B</td><td rowspan=1 colspan=2>6.761.50</td><td rowspan=1 colspan=1>15.42</td><td rowspan=1 colspan=1>0.68</td><td rowspan=1 colspan=3>0.2210.701.75</td><td rowspan=1 colspan=1>0.42</td><td rowspan=1 colspan=2>44.48 5.39</td><td rowspan=1 colspan=2>23.1618.50</td><td rowspan=1 colspan=1>21.54</td><td rowspan=1 colspan=1>22.68</td><td rowspan=1 colspan=1>1.40</td></tr><tr><td rowspan=1 colspan=3>Orion-14B            9.001.04</td><td rowspan=1 colspan=1>5.27</td><td rowspan=1 colspan=1>0.18</td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=2>9.461.11</td><td rowspan=1 colspan=1>3.39</td><td rowspan=1 colspan=2>31.45 4.45</td><td rowspan=1 colspan=5>28.4022.8219.3824.810.90</td></tr><tr><td rowspan=1 colspan=1>Fine-tune PIXEL</td><td rowspan=1 colspan=2>84.57</td><td rowspan=1 colspan=5></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=5></td></tr><tr><td rowspan=1 colspan=3>Majority Baseline     52.86</td><td rowspan=1 colspan=5>5.61 0.97 7.55 0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=2>15.72 3.12</td><td rowspan=1 colspan=5>31.8330.3441.610.00</td></tr></table>

Table 2: Model performance on Chinese character visuals on four different tasks (§4.1). H: Entropy, : Chinese-English bilingual models. The top scores for each type of models (VLM/close-sourced LLM/open-sourced LLM) and all models are highlighted in blue and green , respectively.

LLMs and VLMs’ ability to identify the sequence of strokes required to write the query character. Performance is measured by the accuracy of each stroke in order and the overall F1 score.

## 4.2 Experimental Setup

We assess the visual information of Chinese characters using multilingual, bilingual, and opensource LLMs and VLMs. Multilingual LLMs include Aya-command (Üstün et al., 2024), Claude-3 (Anthropic, 2024), Gemini-1.5, GPT-3.5 Turbo (OpenAI, 2024a), and GPT-4 (OpenAI, 2023). Chinese-English bilingual LLMs include ERNIE-Lite (Baidu, 2024a), Kimi-v1 (MoonshotAI, 2024), and open-source LLMs such as Baichuan-13B (BaichuanInc, 2024), BLOOM-7B (BigScience, 2024), ChatGLM-6B (Zeng et al., 2023), Chinese-

LLaMA-7B (HFL, 2024), InternLM-7B (InternLM, 2024), Orion-14B (Chen et al., 2024), Qwen-7B (Bai et al., 2023), Qwen-2-72B, and Yi-6B (01.AI et al., 2024). We also evaluate VLMs by providing images of query characters in the Microsoft YaHei<sup>9</sup> font to vision-capable models, including Claude-3, Gemini-1.5, GPT-4, GPT-4o (OpenAI, 2024b), and bilingual models Ernie-4 (Baidu, 2024b) and Kimiv1. Additionally, we assess the pixel-based encoder model, PIXEL (Rust et al., 2023). Since PIXEL is limited to specific tasks such as span-based QA, it is evaluated only on the multiple-choice structure recognition task after fine-tuning. To explore the models’ ability to learn Chinese visual information, we apply Chain-of-Thought (CoT) prompting and fine-tuning to GPT-3.5, and few-shot settings to both GPT-3.5 and GPT-4. The remaining models are evaluated in a zero-shot setting. We repeat the evaluation for each task five times to compute entropy, using it as an indicator of the models’ confidence. Detailed prompting and fine-tuning procedures are provided in Appendix C.1.

## 4.3 Experimental Results

As shown in Table 2, the majority of models demonstrate only a vague understanding of Chinese character-visual tasks. Among the evaluated models, Chinese-English bilingual VLMs achieve the highest overall performance, effectively leveraging visual information from the images. Multilingual VLMs, however, perform similarly to their LLM counterparts, with both groups achieving better-than-random-guess results. The performance of vision-lacking LLMs suggests that they have likely encountered textual data discussing knowledge about radicals during pre-training as shown in Appendix C.4. In contrast, open-source LLMs often perform worse than random guesses.

Structure Recognition Task. Most models tend to struggle with the structural arrangement of Chinese characters, with F1 scores below 50%. A notable exception is PIXEL, which achieves an outstanding F1 score of 84.57. Despite being pre-trained solely on an English corpus (English Wikipedia and BookCorpus) and exposed to Chinese only during fine-tuning, PIXEL (Razzhigaev et al., 2022) demonstrates strong potential for Chinese language processing by naturally capturing visually embedded information. GPT-3.5 also saw a 75% performance increase from zero-shot to finetuning settings. To better understand the performance boost after applying learning methods, we further examine the impact of Chinese character encoding and potential mapping patterns between a character’s structure and Unicode in Appendix D.

Radical Recognition Task. In the character-toradical task, a clear trend emerges where model performance is the highest for the first component and sharply decreases for subsequent ones. For example, Claude-3 achieves an F1 score of 70.02 for the first component, but this drops to 5.64 for the second component and nearly zero for the third. This pattern suggests that models could possibly associate the meaning of the radical with the character, as the first radical often relates to the semantic attribute of the character, such as “艹” in “花.”

![](images/ab95633155174fca7c3e14eb1eb3368b5e118c7c611c5d561153ef5155639ba2.jpg)  
Figure 4: Example responses generated by Ernie-4 with vision, translated using Google Translate.

Interestingly, fine-tuning, CoT prompting, and the addition of vision in multilingual models drastically decreased performance of character-to-radical task to nearly zero. However, in the radical-to-character task, fine-tuning GPT-3.5 results in a significant improvement, achieving an F1 score of 71.66. The reason for this disparity, aside from the inherent difficulties between the two tasks, could be that fine-tuning and query characters for the radical-tocharacter task come from a subset of more common characters, for which we annotated structure information. In contrast, the character-to-radical task includes more complex and rarely used characters, potentially leading to catastrophic learning failure.

Stroke Decomposition and Count Tasks. Overall, most models struggle with identifying specific stroke compositions but demonstrate a general understanding of stroke count. For instance, Claude-3 achieves the lowest MSE among all LLMs at 7.78, significantly lower than the dataset’s average stroke count of 11.51. Both tasks benefit from the applied learning methods. For stroke composition, finetuning yields the best results, while for stroke count, all methods show similar levels of improvement.

## 4.4 Error Analysis of Bilingual VLMs

To better understand the superior performance of bilingual VLMs, we conducted an error analysis on Ernie-4 and Kimi-v1 with vision. Both models exhibit common patterns of mistakes across several types of characters. First, complex and dense characters are often misrecognized as some other more frequently used characters that look similar. As the complexity of characters increases, the individual radicals become more compressed within the available space, which can lead to misrecognition between similar characters, such as “黄” and “ ”. Second, characters with only a single stroke difference are frequently mistaken for their more common variant. Third, when dealing with rare characters, Ernie-4 often fails to detect any character in the image, while Kimi-v1 may refuse to allow the user to send the prompt if it cannot extract the character. Additionally, both models sometimes mistake one radical for the entire character or confuse characters with black-and-white photos as shown in Figure 4. More examples of Kimi-v1 and Ernie-4 are provided in Appendix C.3.

<table><tr><td rowspan="2">Models</td><td colspan="5">Name Entity Recognition</td></tr><tr><td colspan="3">People&#x27;s Daily</td><td colspan="2">Weibo</td></tr><tr><td></td><td>B</td><td>RP</td><td></td><td>B</td><td>RP</td></tr><tr><td>Aya</td><td>52.00</td><td>54.61</td><td>(+2.6)</td><td>24.78</td><td>16.00 (-8.8)</td></tr><tr><td>Claude-3</td><td>68.54</td><td>70.48</td><td>(+1.9)</td><td>41.08 41.67</td><td>(+1.6)</td></tr><tr><td>ERNIE-Lite</td><td>7.55</td><td>21.05</td><td>(+14)</td><td>6.25 14.81</td><td>(+8.6)</td></tr><tr><td>GPT-3.5</td><td>55.74</td><td>55.96</td><td>(+0.2)</td><td>38.37 44.87</td><td>(+12)</td></tr><tr><td>GPT-4</td><td>65.23</td><td>65.96</td><td>(+0.7)</td><td>38.59 40.34</td><td>(+1.8)</td></tr><tr><td>QWen 72B</td><td>58.81</td><td>58.94</td><td>(+0.1)</td><td>29.39 33.17</td><td>(+3.8)</td></tr></table>

Table 3: Model performances for NER evaluated solely on samples where the model identifies unfamiliar words.

## 5 Evaluation on Utilizing Radicals

We evaluated LLMs on downstream tasks, specifically examining performance differences when models are prompted vs. not prompted to use their knowledge of radicals to infer the meaning of unfamiliar words as illustrated in Figure 5.

## 5.1 Downstream NLP Tasks

Although LLMs may not match supervised LMs in traditional tasks, we chose these as key indicators of Chinese understanding to track improvements when models utilize radical information.

Part-of-Speech (POS) tagging. For the POS tagging task, we selected a 5-word window containing at most one punctuation mark and tasked the model with identifying the POS tag of the central word. The model’s performance was evaluated using the F1 score. To cover a diverse range of sentences, we utilized three datasets: the GSD Simplified dataset (Qi and Yasuoka, 2023), the Parallel Universal Dependencies (PUD) dataset (McDonald et al., 2023), and a new dataset of 500 sentences from Tang Dynasty poems (written between 618 and 907 CE), processed using Classical Chinese RoBERTa (Yasuoka, 2023). We created and annotated this poetry dataset to evaluate how well radicals perform in Classical Chinese (文言), which is characterized by compact and precise language where more information is carried by each character. Additionally, we conducted an ablation study with varied input window sizes, detailed in Appendix F.2.

![](images/b181d14a5ab91cb30cfd8a99a1cc60ec20a3cab0622e9482ffec55967f25e95f.jpg)  
Figure 5: Example of model’s answers for part-ofspeech (POS) tagging with an unfamiliar Chinese word using radical prompting.

Named Entity Recognition (NER). We tasked the model with identifying three types of entities — PER (person), LOC (location), and ORG (organization) — at the character level, using the BIO tagging standard. We excluded nominal entities provided in some datasets to streamline the analysis. The model’s performance was evaluated using the F1 score. We use two distinct datasets for the NER task: the People’s Daily dataset (Chen, 2023), which focuses on formal Chinese text, and the Weibo NER dataset (Peng and Dredze, 2015), which is oriented towards casual and online text.

Chinese Word Segmentation (CWS). CWS is a unique task in Chinese language processing. Distinguished from many other languages, Chinese does not use delimiters such as white spaces to separate words within sentences. Accurately segmenting text could be beneficial for many CLP applications. For this task, we provide sentences from the GSD and PUD and ask models to segment them into words. Answers are evaluated using the F1 score.

## 5.2 Experimental Setup

We select a series of LLMs for evaluation, including Aya-command, Claude-3, ERNIE-Lite-8K, GPT-3.5, GPT-4, and QWen-1.5 72B Chat. The models are instructed to return answers in JSON format, with target sentences annotated in a manner similar to Blevins et al. (2023). Each task and dataset is evaluated using 2,000 sample sentences five times. Due to higher costs, Claude-3 and GPT-4 are evaluated with 1,000 samples. Additionally, we experiment with o1-mini (OpenAI, 2024c) using 100 samples, repeated three times. We experiment with three different prompting methods:

<table><tr><td rowspan="3">Models</td><td colspan="10">Part-Of-Speech Tagging</td><td colspan="4"></td></tr><tr><td colspan="4">GSD</td><td colspan="4"></td><td colspan="2"></td><td colspan="4">Poems</td></tr><tr><td>B</td><td>RP</td><td></td><td>RP (Oracle)</td><td></td><td>B</td><td>RP</td><td></td><td>RP (Oracle)</td><td></td><td>B</td><td>RP</td><td></td><td>RP (Oracle)</td></tr><tr><td>Aya-command</td><td></td><td>68.8668.91</td><td>(+0.1)</td><td>70.41</td><td>(+1.6)</td><td>73.8777.21</td><td></td><td>(+3.3)</td><td>76.95</td><td>(+3.1)</td><td>64.71</td><td>64.72</td><td>(+0.0)</td><td>65.54 (+0.8)</td></tr><tr><td>Claude-3</td><td></td><td>69.3770.68</td><td>(+1.3)</td><td>70.45</td><td>(+1.1)</td><td>69.3770.45</td><td></td><td>(+1.1)</td><td>70.68</td><td>(+1.3)</td><td>65.53</td><td>66.20</td><td>(+0.7)</td><td>66.71 (+1.2)</td></tr><tr><td>ERNIE-Lite-8K</td><td></td><td>27.0624.97</td><td>(-2.1)</td><td>32.73</td><td>(+5.7)</td><td>30.35 30.29</td><td></td><td>(-0.0)</td><td>41.29</td><td>(+10.9)</td><td>44.1942.17</td><td></td><td>(-2.0)</td><td>49.07 (+4.9)</td></tr><tr><td>GPT-3.5</td><td>59.08</td><td>64.62</td><td>(+5.5)</td><td>67.56</td><td>(+8.5)</td><td>62.61 69.90</td><td></td><td>(+7.3)</td><td>73.46</td><td>(+10.9)</td><td>53.51 59.22</td><td></td><td>(+5.7)</td><td>61.39 (+7.9)</td></tr><tr><td>GPT-4</td><td>71.55</td><td>72.14</td><td>(+0.6)</td><td>72.95</td><td>(+1.4)</td><td>76.2076.72</td><td></td><td>(+0.5)</td><td>77.35</td><td>(+1.2)</td><td>66.94</td><td>67.11</td><td>(+0.2)</td><td>67.57 (+0.6)</td></tr><tr><td>01-mini</td><td>63.24</td><td>67.96</td><td>(+4.7)</td><td>64.31</td><td>(+1.1)</td><td>70.3771.42</td><td></td><td>(+1.1)</td><td>75.49</td><td>(+4.1)</td><td>47.73</td><td>50.04</td><td>(+2.3)</td><td>49.00 (+1.3)</td></tr><tr><td>QWen-72B</td><td></td><td>62.2065.38</td><td>(+3.2)</td><td>67.32</td><td>(+5.1)</td><td>60.0964.70</td><td></td><td>(+4.6)</td><td>66.90</td><td>(+6.8)</td><td>55.6357.78</td><td></td><td>(+2.2)</td><td>59.54 (+3.9)</td></tr></table>

Table 4: Model performances for POS tagging with baseline (B), radical prompting without golden components (RP), and radical prompting with oracle information (RP (Oracle)). Performance change relative to baseline is highlighted with green for increase and red for decrease.
<table><tr><td rowspan="3">Models</td><td colspan="6">Name Entity Recognition</td><td colspan="6">Chinese Word Segmentation</td></tr><tr><td colspan="3">People&#x27;s Daily</td><td colspan="3">Weibo</td><td colspan="3">GSD</td><td colspan="3">PUD</td></tr><tr><td>B</td><td colspan="2">RP</td><td>B</td><td colspan="2">RP</td><td>B</td><td></td><td>RP</td><td>B</td><td>RP</td><td></td></tr><tr><td>Aya-command</td><td>38.24</td><td>36.36</td><td>(-1.88)</td><td>37.88</td><td>30.83</td><td>(-7.05)</td><td>87.98</td><td>89.08</td><td>(+1.10)</td><td>88.68</td><td>91.05</td><td>(+2.37)</td></tr><tr><td>Claude-3</td><td>69.74</td><td>73.79</td><td>(+4.05)</td><td>45.64</td><td>46.86</td><td>(+1.22)</td><td>94.90</td><td>95.16</td><td>(+0.26)</td><td>94.12</td><td>94.96</td><td>(+0.84)</td></tr><tr><td>ERNIE-Lite-8K</td><td>12.10</td><td>12.99</td><td>(+0.89)</td><td>6.72</td><td>6.90</td><td>(+0.19)</td><td>88.04</td><td>88.70</td><td>(+0.66)</td><td>69.54</td><td>73.57</td><td>(+4.03)</td></tr><tr><td>GPT-3.5</td><td>56.89</td><td>55.97</td><td>(-0.92)</td><td>36.65</td><td>36.64</td><td>(-0.01)</td><td>95.68</td><td>94.87</td><td>(-0.81)</td><td>93.91</td><td>93.70</td><td>(-0.21)</td></tr><tr><td>GPT-4</td><td>66.04</td><td>68.05</td><td>(+2.01)</td><td>43.83</td><td>44.68</td><td>(+0.85)</td><td>94.21</td><td>94.88</td><td>(+0.67)</td><td>94.24</td><td>95.63</td><td>(+1.39)</td></tr><tr><td>o1-mini</td><td>84.21</td><td>91.67</td><td>(+7.46)</td><td>56.37</td><td>69.70</td><td>(+13.3)</td><td>97.21</td><td>100.0</td><td>(+2.79)</td><td>93.65</td><td>97.00</td><td>(+3.35)</td></tr><tr><td>QWen 72B</td><td>62.73</td><td>59.59</td><td>(-3.14)</td><td>31.78</td><td>35.83</td><td>(+4.05)</td><td>96.59</td><td>95.57</td><td>(-1.02)</td><td>89.79</td><td>91.94</td><td>(+2.15)</td></tr></table>

Table 5: Model performances for NER and CWS tasks with baseline (B) and radical prompting (RP).

Baseline. Our baseline employs the CoT prompting framework with steps that guide the model to execute tasks. See prompts in Appendix F.4.

Radical Prompting. We incorporate the radical information into the input prompt as steps within the CoT framework. The process begins with the model identifying any unclear words within a given context. Then, the model is instructed to dissect these words into their constituent radicals and attempt to utilize useful radicals to aid the task. Steps are then provided to guide the model in executing specific tasks, identical to the baseline, with three examples. When using radical prompting, it is important to guide models to critically assess information from character components to avoid being misguided. Therefore, one of the three examples intentionally includes radical that can be misleading, helping the model learn to discern when to use radical information. Prompt lines of radical prompting are listed in Appendix F.3.

Radical Prompting (Oracle). Similar to the radical prompting method, instead of instructing the model to decompose characters, we directly provided the correct radicals in the input prompt. This method was applied only to the POS tagging task, as it required supplying the radical of just the central word. For the other tasks, it is impractical to provide radicals for all characters in the sentence.

## 5.3 Experimental Results

Our results suggest that radicals have strong potential if models can properly understand and utilize them. Results for POS tagging are shown in Table 4 with qualitative analysis in Appendix F.1. In the POS tagging task, models consistently show improvement across datasets, especially when the correct radicals are provided. Notably, in the PUD dataset, ERNIE-Lite-8K exhibits a slight decrease in performance without the correct radicals but shows an increase of approximately 11 F1 points when the correct radicals are included.

For the NER and CWS tasks, the initial results are mixed as shown in Table 5. However, our error analysis reveals that with the radical prompting method, incorrect answers often occur when the model bypasses the use of radicals and asserts that there are no ambiguous words in the sentence being examined. This suggests that the negative effect may be attributed to the longer prompts, as more robust models, such as Claude-3 and GPT-4, still demonstrate improvement in performance across datasets. When evaluating only the samples where the model identifies ambiguous words in the radical prompt setting, we find that the models genuinely perform better, as shown in Table 3. However, a notable exception is the Aya-command, whose performance drops significantly on the Weibo dataset. Upon closer examination, we find that Aya has a strong tendency to split words into individual characters rather than into radicals. Examples of such output are shown in Appendix F.5.

## 6 Conclusion

In this paper, we create a comprehensive benchmark on visual information embedded in Chinese characters. Our evaluation of the benchmark highlights the suboptimal performance of LLMs and VLMs in handling information below the character level. Despite this, our experiments with radical prompting demonstrate that these sub-character features can still be beneficial. The results show consistent improvements in POS tagging when correct radicals are provided, and promising results in NER on sentences containing unfamiliar words. Our work highlights the potential of radical knowledge for Chinese NLP applications and advocates attention to help models leverage it, including additional training on radicals or improving Chinese digital systems to more effectively integrate radical structures.

## Limitations

While our study offers valuable insights into the integration of radical prompting in Chinese language models, it also highlights areas for further exploration. First, the dataset used in this research does not represent the full range of Chinese characters, as the majority are sourced from simplified Chinese.

Moreover, the study primarily evaluates radical prompting on a limited selection of models and tasks, which may not fully capture its potential across a wider range of models and language processing applications.

Lastly, an area for improvement in our methodology involves the exclusive use of English in our prompt lines. Incorporating Chinese in the prompting strategy could further enhance the relevance and effectiveness of the prompts, better aligning with the linguistic context of the target language.

## Acknowledgements

We extend our gratitude to Dengchun Yuan for his inspiring discussions on this work and to Yao Dao for offering valuable advice on prompting design. We also thank Duong Minh Le for his helpful guidance on writing. Finally, we appreciate the efforts of Geyang Guo, Shu Zhu, and Wei Zhou for their assistance with the validation process.

## References

01.AI, Alex Young, Bei Chen, Chao Li, Chengen Huang, Ge Zhang, Guanwei Zhang, Heng Li, Jiangcheng Zhu, Jianqun Chen, Jing Chang, Kaidong Yu, Peng Liu, Qiang Liu, Shawn Yue, Senbin Yang, Shiming Yang, Tao Yu, Wen Xie, Wenhao Huang, Xiaohui Hu, Xiaoyi Ren, Xinyao Niu, Pengcheng Nie, Yuchi Xu, Yudong Liu, Yue Wang, Yuxuan Cai, Zhenyu Gu, Zhiyuan Liu, and Zonghong Dai. 2024. Yi: Open Foundation Models by 01.AI.

Anthropic. 2024. The Claude 3 Model Family: Opus, Sonnet, Haiku. Accessed: 2024-06-09.

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. 2023. Qwen Technical Report. arXiv preprint arXiv:2309.16609.

BaichuanInc. 2024. Baichuan-13B-Base. Accessed: 2024-06-11.

Baidu. 2024a. Introducing ERNIE 3.5: Baidu’s Knowledge-Enhanced Foundation Model Takes a Giant Leap Forward. Accessed: 2024-06-11.

Baidu. 2024b. Yiyan. Accessed: 2024-06-11.

BigScience. 2024. BLOOM-7B1. Accessed: 2024-06- 11.

Terra Blevins, Hila Gonen, and Luke Zettlemoyer. 2023. Prompting language models for linguistic structure. In Proceedings of the 61st Annual Meeting of the

Associationfor Computational Linguistics (Volume 1: Long Papers), pages 6649–6663, Toronto, Canada. Association for Computational Linguistics.

Du Chen, Yi Huang, Xiaopu Li, Yongqiang Li, Yongqiang Liu, Haihui Pan, Leichao Xu, Dacheng Zhang, Zhipeng Zhang, and Kun Han. 2024. Orion-14B: Open-source Multilingual Large Language Models. arXiv preprint arXiv:2401.12246.

Han Chen. 2023. People’s Daily (RenMin Daily) Named Entity Recognition Dataset. http://paper. people.com.cn/. A comprehensive dataset from the People’s Daily, covering news from 2021/01/01 to 2023/12/05, for Named Entity Recognition with news segments labeled for LOC, ORG, PER entities using BIO tagging strategy. License: CC0: Public Domain.

HFL. 2024. Chinese llama-2-7b. Accessed: 2024-06- 11.

InternLM. 2024. Internlm-7b.

Kawabata Kawabata, Masaya Nakamura, and Huáng Jùnliàng. 2018. CJKVI-IDS: Ideographic Description Sequences for CJK Unified Ideographs. https://github.com/cjkvi/cjkvi-ids. Accessed: 2024-4-4.

Xiaodong Liu, David Wisniewski, L. Vermeylen, Ana F. Palenciano, Wenjie Liu, and M. Brysbaert. 2021. The Representations of Chinese Characters: Evidence from Sublexical Components. Journal of Neuroscience, 42(1):135.

Jiefeng Ma, Zirui Wang, and Jun Du. 2021. An Open-Source Library of 2D-GMM-HMM Based on Kaldi Toolkit and Its Application to Handwritten Chinese Character Recognition. Lecture Notes in Computer Science, 12888.

Ryan McDonald, Joakim Nivre, Yvonne Quirmbach-Brundage, Yoav Goldberg, Dipanjan Das, Kuzman Ganchev, Keith Hall, Slav Petrov, Hao Zhang, Oscar Tackstrom, Claudia Bedini, Nuria Bertomeu Castello, and Jungmee Lee. 2023. Parallel Universal Dependencies (PUD) Treebanks for Multilingual Parsing. Available for the CoNLL 2017 shared task on Multilingual Parsing from Raw Text to Universal Dependencies. Annotations provided by Google and converted to UD v2 guidelines by the UD community.

MoonshotAI. 2024. Kimi. Accessed: 2024-06-11.

OpenAI. 2023. GPT-4. Accessed: 2024-06-11.

OpenAI. 2024a. Gpt-3.5 turbo. Accessed: 2024-06-11.

OpenAI. 2024b. Gpt-4o system card. Accessed: 2024- 10-16.

OpenAI. 2024c. Openai o1 mini: Advancing costefficient reasoning. Accessed: 2024-10-16.

Nanyun Peng and Mark Dredze. 2015. Named Entity Recognition for Chinese Social Media with Jointly Trained Embeddings. In Proceedings ofthe Human Language Technology Center of Excellence, Baltimore, MD. Johns Hopkins University.

Peng Qi and Koichi Yasuoka. 2023. Simplified Chinese Universal Dependencies Version 2.13. Universal Dependencies (UD) Chinese GSDSimp treebank. Available from GitHub: UD\_Chinese-GSDSimp.

Anton Razzhigaev, Anton Voronov, Andrey Kaznacheev, Andrey Kuznetsov, Denis Dimitrov, and Alexander Panchenko. 2022. Pixel-level BPE for autoregressive image generation. In Proceedings of the First Workshop on Performance and Interpretability Evaluations ofMultimodal, Multipurpose, Massive-Scale Models, pages 26–30, Virtual. International Conference on Computational Linguistics.

Phillip Rust, Jonas F. Lotz, Emanuele Bugliarello, Elizabeth Salesky, Miryam de Lhoneux, and Desmond Elliott. 2023. Language Modelling with Pixels. In The Eleventh International Conference on Learning Representations.

Xinlei Shi, Junjie Zhai, Xudong Yang, Zehua Xie, and Chao Liu. 2015. Radical Embedding: Delving Deeper into Chinese Radicals. In Proceedings of the Association for Computational Linguistics (ACL). Sogou Technology Inc., Beijing, China.

Chenglei Si, Zhengyan Zhang, Yingfa Chen, Fanchao Qi, Xiaozhi Wang, Zhiyuan Liu, Yasheng Wang, Qun Liu, and Maosong Sun. 2021. Sub-Character Tokenization for Chinese Pretrained Language Models. Transactions of the Association for Computational Linguistics, 9:634–649.

Karl Stratos. 2017. A Sub-Character Architecture for Korean Language Processing. In Proceedings ofthe 2017 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 2420–2430, Copenhagen, Denmark. Association for Computational Linguistics.

Zijun Sun, Xiaoya Li, Xiaofei Sun, Yuxian Meng, Xiang Ao, Qing He, Fei Wu, and Jiwei Li. 2021. ChineseBERT: Chinese Pretraining Enhanced by Glyph and Pinyin Information. In Proceedings of the Associationfor Computational Linguistics (ACL). Shannon.AI; Zhejiang University; Key Lab of Intelligent Information Processing of Chinese Academy of Sciences.

Franck Xia. 1994. Knowledge-based sub-pattern segmentation: decompositions of Chinese characters. Proceedings ofthe International Conference on Image Processing.

Koichi Yasuoka. 2023. RoBERTa Model Pre-trained on Classical Chinese Texts. https://huggingface.co/KoichiYasuoka/ roberta-classical-chinese-large-char. Derived from GuwenBERT-large with characterembeddings for traditional/simplified characters.

Suitable for tasks like sentence-segmentation, POS-tagging, dependency-parsing.

Aohan Zeng, Xiao Liu, Zhengxiao Du, Zihan Wang, Hanyu Lai, Ming Ding, Zhuoyi Yang, Yifan Xu, Wendi Zheng, Xiao Xia, Weng Lam Tam, Zixuan Ma, Yufei Xue, Jidong Zhai, Wenguang Chen, Zhiyuan Liu, Peng Zhang, Yuxiao Dong, and Jie Tang. 2023. GLM-130B: An Open Bilingual Pre-trained Model. In The Eleventh International Conference on Learning Representations (ICLR).

Jianshu Zhang, Yixing Zhu, Jun Du, and Lirong Dai. 2018. Radical Analysis Network for Zero-Shot Learning in Printed Chinese Character Recognition. In Proceedings of the IEEE International Conference on Multimedia and Expo (ICME), Hefei, Anhui, P.R. China. IEEE.

Ahmet Üstün, Viraat Aryabumi, Zheng-Xin Yong, Wei-Yin Ko, Daniel D’souza, Gbemileke Onilude, Neel Bhandari, Shivalika Singh, Hui-Lee Ooi, Amr Kayid, Freddie Vargus, Phil Blunsom, Shayne Longpre, Niklas Muennighoff, Marzieh Fadaee, Julia Kreutzer, and Sara Hooker. 2024. Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model. arXiv preprint arXiv:2402.07827.

## A General Experiment Details

Model Snapshots The experiments incorporated different versions of widely recognized models to evaluate their performance in processing Chinese characters. The specific snapshots used for each model are as follows:

• GPT-3.5 and GPT-4 were used with the snapshot dated 2023-11-06.

• Claude model’s evaluation utilized the 2024- 02-29 snapshot.

• Ernie-Lite-8K was tested using the 2023-09- 22 snapshot.

## Temperature Settings

• Aya, Yi-6B, Qwen-7B-Chat, Baichuan-13B, and Mistral-7B were set at a lower temperature of 0.3 as recommended.

• For other models not specifically mentioned, a temperature setting of 0.7 was used.

## B Validation on Radical Annotation

To ensure the quality of our radical annotations, we conducted a validation process with a team of four annotators. This team included three volunteer native Chinese speakers and one native Chinesespeaking author. Prior to the formal annotation process, each annotator underwent a brief practice session to familiarize themselves with the task. Following this, the speed of checking 100 rows is around two and a half minutes per annotator.

For the validation task, we randomly assigned 3,500 rows to each annotator from our dataset, ensuring that the task could be completed within roughly two hours. Each annotator was instructed to review the assigned rows and flag any errors they identified in the radical annotations. The errors flagged were then collectively analyzed to compute the error rate and guide the necessary revisions.

## C Details on Visual Info Evaluation

## C.1 Detail Settings

For our evaluation, we use different sampling methods and settings based on the type of model. For LLMs, a random sample of 1,000 characters is selected for each task and model. Due to higher costs, the number of samples for VLMs is reduced to 500. ERNIE-V and Kimi-V, which lack API access, are tested manually with only 100 samples. We incorporate few-shot learning by providing models with three examples for each task, except for the structure recognition task, where one example per structure type is given. In the Chain-of-Thought (CoT) setting, models are prompted to break down their reasoning process step-by-step, with detailed prompts provided in the Appendix C.2. Models with fine-tuning are trained in OpenAI’s platform using cross-entropy loss with a 7:3 split and tested using 1,000 samples randomly selected from the CCD dataset. To assess consistency and model entropy, each question is asked five times, and the best trial out of the five for each task is selected to calculate the overall results.

To adapt answers from models generating long responses conventionally, we first let models generate responses freely without a specific answer format. Then, we use GPT-3.5 Turbo to extract answers from various model responses. For opensource models and extraction-used GPT-3.5 Turbo, a temperature of 0.3 is applied. Closed-source models generally use a temperature of 0.7 unless otherwise recommended by model documentation.

## C.2 CoT Prompting

We present the prompt lines used for visual info evaluation in Figure 9.

## C.3 Chinese VLMs Behavior

Examples of VLMs misrecognizing images are shown in Figure 17, 18, 19, 20, and 21.

## C.4 GPT-4o and GPT-o1 Behavior

When simply prompting GPT-4o and GPT-4.1 with the question "What does this character mean?" for unfamiliar characters, both models provide responses with radical information, but they do so incorrectly (as shown in Figures 22). This suggests that while the models have been exposed to radicals in their training corpus, they are still unable to correctly identify or interpret the appropriate radicals.

## D Analysis on Chinese Character Encoding

## D.1 Experiment on Encoding

To further investigate why models after fine-tuning perform exceptionally well on structure tasks but show decreased performance on other Chinese visual tasks, we conducted a side experiment on different encoding systems to determine if they learn some sort of implicit pattern from the encoding.

Setup. We fine-tuned GPT-3.5 by explicitly switching all Chinese characters in the training and testing documents to various encodings—namely, Unicode, stroke, Pinyin<sup>10</sup>, Wubi, and Cangjie<sup>11</sup>—and evaluated them on the structure recognition task to assess the impact of these representations on the model’s learning ability with visual knowledge of Chinese characters.

Results. The results shown in Table 7 indicate that Unicode encoding performs comparably to vision-rich stroke encoding and significantly outperforms Pinyin encoding, which is limited to phonetic information. Upon further investigation, we found that the order of Chinese characters in Unicode is closely related to the stroke count and structure of the characters: Unicode is ordered by the stroke count of their indexing radical and the stroke count of the remaining parts. However, the full potential of Unicode is diminished by numerous exceptions and a broad spectrum of extensions that complicate its utility in conveying visual knowledge, where similar structures are likely grouped together with stroke counts in incremental order, as detailed in Table 6.

→鹿 廌  
→寨 塞 蹇 搴 謇 寋 褰 寒 襄 骞 赛 壌 嚢 囊 弿 醸  
→袗 珍 殄 抮 跈 翏 畛 诊 轸 昣 聄  
→追 桘 垖 辥 蛗 峊 薛  
→舋 爨 爂  
→岛 枭 捣  
→黉 喾 应 觉 尝 学 举 泶 鲎 鸴 単  
Figure 6: Examples of untypable radicals

## D.2 Challenges on Radical Encoding

None of the encoding systems mentioned above can fully exploit the potential of radicals. Strokebased systems over-decompose characters into individual strokes, losing meaningful structure, while glyph-based input methods like Wubi and Cangjie oversimplify and over-categorize characters to prioritize efficiency as input methods.

However, a significant challenge lies in developing a radical-based encoding system. While some radicals have corresponding Unicode representations, they cannot be typed using standard input methods. With around 100,000 CJK ideographs in Unicode, the task becomes even more difficult, as identifying the correct representation requires manual searching by sight, since there is no way to input the radical for automated searching, as shown in Figure 6. This limitation forces us to decompose some radicals further to maintain the integrity of character representation. Unfortunately, this results in a loss of meaning in certain cases, as the radicals become fragmented beyond their functional roles within characters.

## E Discussion on Chinese Characters

To investigate the importance of Chinese radicals, we selected a sample of 100 Chinese characters from our dataset and annotated them to determine whether the radicals directly contribute to the meaning or pronunciation of the character, as shown in Figures 7. Although the majority of characters have clues derived from the radicals, we found that most characters contain a combination of only one meaningful radical with other radicals hinting at pronunciation. For example, in the character “花,” we can infer that it is related to herbs from the radical “艹,” while “化” only provides a pronunciation hint, resulting in only a vague idea of character’s meaning. In 12 out of the 100 characters, none of the radicals were helpful.

<table><tr><td>Unicode</td><td>Character</td><td>Structure</td><td>Unicode</td><td>Character</td><td>Structure</td></tr><tr><td>U+4EBF</td><td>亿</td><td>Left-Right</td><td>U+4ED9</td><td>仙</td><td>Left-Right</td></tr><tr><td>U+4EC0</td><td>什</td><td>Left-Right</td><td>U+4EE3</td><td>代</td><td>Left-Right</td></tr><tr><td>U+4EC1</td><td>仁</td><td>Left-Right</td><td>U+4EEA</td><td>仪</td><td>Left-Right</td></tr><tr><td>U+4EC3</td><td>仃</td><td>Left-Right</td><td>U+4EEB</td><td>仫</td><td>Left-Right</td></tr><tr><td>U+4EC4</td><td>仄</td><td>Wrapping</td><td>U+4EF0</td><td>仰</td><td>Left-Right</td></tr><tr><td>U+4EC7</td><td>仇</td><td>Left-Right</td><td>U+4EF2</td><td>仲</td><td>Left-Right</td></tr><tr><td>U+4ECE</td><td>从</td><td>Left-Right</td><td>U+4EF5</td><td>仵</td><td>Left-Right</td></tr><tr><td>U+4ED1</td><td>仑</td><td>Top-Bottom</td><td>U+4EFB</td><td>任</td><td>Left-Right</td></tr><tr><td>U+4ED3</td><td>仓</td><td>Top-Bottom</td><td>U+4EFD</td><td>份</td><td>Left-Right</td></tr><tr><td>U+4ED5</td><td>仕</td><td>Left-Right</td><td>U+4F01</td><td>企</td><td>Top-Bottom</td></tr><tr><td>U+4ED6</td><td>他</td><td>Left-Right</td><td>U+4F0A</td><td>伊</td><td>Left-Right</td></tr><tr><td>U+4ED7</td><td>仗</td><td>Left-Right</td><td>U+4F0D</td><td>伍</td><td>Left-Right</td></tr><tr><td>U+4ED8</td><td>付</td><td>Left-Right</td><td>U+4F0E</td><td>伎</td><td>Left-Right</td></tr></table>

Table 6: This table showcases a randomly selected range of Unicode characters in the dataset along with their respective structures. This representation provides a snapshot of the structural information inherent in the Unicode.

<table><tr><td>Encoding</td><td>Structure Acc</td></tr><tr><td>Unicode</td><td>39.80</td></tr><tr><td>Stroke</td><td>43.80</td></tr><tr><td>PinYin</td><td>13.85</td></tr><tr><td>WuBi</td><td>11.81</td></tr><tr><td>CangJie</td><td>11.66</td></tr></table>

Table 7: GPT-3.5 fine-tuning’ performance on different ways of encoding.

![](images/dab552dba8316121c6f63e56691f63ad8f0050cabce0e0b372efdcb883592ef1.jpg)  
Figure 7: Distribution of Chinese characters with meaning (M) or pronunciation (P) hint from their radicals. The smaller circle on the right shows the distribution among all characters containing radicals with meaning (sum of Characters M only and Characters M & P).

![](images/da8e7e85c40959728263e987e0e176b8ca89b0d2049baafeb80f753fccf406e6.jpg)  
Radicals \~ M Radicals ∼ P Radicals \~ M & P Radicals\~ N/A  
Figure 8: Sampled distribution of radicals with meaning (M) or Pronunciation (P) hint.

This is due to the evolution of the language, where historically, a single Chinese character often conveyed the meaning of a full word. However, more words are now composed of two or more characters, leading to individual characters losing their original meanings. For example, the Chinese character “况” is now commonly used to mean “situation” in words like “情况” or “状况”. However, the original meaning of the character is “cold water” unexpectedly, which is closely related to the radical “冫”, referring to cold water.

## F Detailed Radical Prompting Result

## F.1 Quantitative Analysis on POS tagging Accuracy

We provide a case analysis for POS tagging in Table 8.
<table><tr><td>Category</td><td>Baseline</td><td>RP (Oracle)</td></tr><tr><td>Correct&amp; utilize Radical</td><td></td><td>81.2 (+81.2)</td></tr><tr><td>Correct without</td><td>608.6</td><td>611.2 (+2.6)</td></tr><tr><td>Incorrect &amp; utilize Radical</td><td></td><td>41.8 (+41.8)</td></tr><tr><td>Incorrect without</td><td>391.4</td><td>265.8 (-125.6)</td></tr></table>

Table 8: Quantitative analysis of GPT-3.5-Turbo’s POS tagging accuracy on the number of correct and incorrect predictions with and without the examination of components using radical prompting compared to the baseline. Improvement is shown in green.

## F.2 Window size’s impact on POS tagging

We evaluate the impact of different window sizes in POS tagging with GPT-3.5-Turbo in Table 9.

<table><tr><td rowspan="3">Window Size</td><td colspan="3">Part-Of-Speech Tagging</td></tr><tr><td colspan="3">GPT-3.5-Turbo with GSD</td></tr><tr><td>B</td><td>RP</td><td>RP (Oracle)</td></tr><tr><td>5</td><td>59.08</td><td>64.62 (+5.5)</td><td>67.56 (+8.5)</td></tr><tr><td>7</td><td>60.17</td><td>66.55 (+6.38)</td><td>66.73 (+6.56)</td></tr><tr><td>9</td><td>60.38</td><td>67.03 (+6.65)</td><td>67.23 (+6.85)</td></tr></table>

Table 9: Model performance for POS tagging with different word window sizes

## F.3 Radical Prompting Prompts

We provide our radical prompting lines for POS tagging, NER, and CWS tasks in Figure 11, 12, and 13, respectively.

## F.4 Base Prompting Prompts

We provide our base prompting lines for POS tagging, NER, and CWS tasks in Figure 14, 15, and 16, respectively.

## F.5 Aya Model Behavior

Examples of Aya decomposing radicals incorrectly are shown in Figure 10.

## G Responsible NLP Miscellanea

## G.1 Intent usage

In response to potential inquiries regarding the scope and legitimacy of our experiments, it is important to clarify that all aspects of our research strictly adhere to the intended use cases of the Large Language Models (LLMs) and the NLP task datasets employed. Furthermore, our use of these models and datasets complies fully with the usage policies of the APIs for each model involved. We note that the use of rare Chinese words triggered some safety mechanisms in models such as Gemini-1.5. However, our intent complies fully with the ethical guidelines and usage policies provided by the API providers.

## G.2 Computational Experiments Cost

In our research, we utilized vLLMs for evaluation on Yi 6B, Mistral 7B, Baichuan 13B, and Qwen 7B with a single a40 GPU. For other models, we accessed them through their respective APIs. The cost and running time for each model varied significantly. Specifically, the time required to run a single evaluation ranged from approximately 2 to 8 hours.

## G.3 Avoid Data Leakage

For all NLP tasks assessed in this study, evaluations were exclusively conducted on the development sets of the respective datasets to prevent data leakage.

## G.4 Personally Identifying Info

The dataset we created for evaluating the visual information of Chinese characters does not contain any offensive content or personally identifying information. However, we acknowledge the presence of individual names in the Weibo NER dataset that we use for evaluation.

## G.5 Evaluation Tools and Methodologies

To evaluate our Named Entity Recognition (NER) tasks, we used a Perl script: conlleval.pl.

For other tasks, we calculated the F1 score using Scikit-learn.

## G.6 AI Assistants

We acknowledge the use of GPT-4 for grammar checking and word polishing.

![](images/7ab481c409610521e00a3913c69d7ea1b73ae4ab2a5856730062e406906e7814.jpg)  
Figure 9: Prompt Line of Visual Tasks. Sub-windows are organized as follows: Structure (Top Left), Radical (Bottom Left), Stroke Count (Top Right), and Stroke Decomposition (Bottom Right). The red section indicates the Chain-of-Thought (CoT) prompting portion.

![](images/2f3633813fd1d1b46e1e9b57eb7bafc25bb2892258c1e92c7ba20719d53ad958.jpg)  
Figure 10: Example of Aya decomposes incorrectly.

![](images/6042cef1d9d449704b26d432ec10981178c49dbbbe50efdb0b8d533bb8a38156.jpg)  
Figure 11: Radical Prompting Prompt Line of POS tagging.

![](images/41076727b828c3f32bda454ecf71c97fa491450a6a46fe9a36ca65dd8484fe63.jpg)  
Figure 12: Radical Prompting Prompt Line of NER.

![](images/d9f2d962cab4710859a7c7abb5270b979a52f08236219620b78c202b19947459.jpg)  
Figure 13: Radical Prompting Prompt line for CWS.

![](images/ec00cfeffda19f74ca4a386b8ef362bb5f138d41c8c06224d8fbd573f30820f5.jpg)  
Figure 14: Base Prompt line for POS tagging.

![](images/b9a7c546731497596599964f1b93d9dd6230aa79ea865864eff33514567fe363.jpg)  
Figure 15: Base Prompt line for NER.

![](images/d0b2581bee7c5236c94e9fd9c86d741935f76d1ea8d6bfe7d0aa91ee71e26378.jpg)  
Figure 16: Base Prompt line for CWS.

![](images/622c806b1150903a0cb5fdb473d4513d6a86ab9ad57f1ff7055b270362c4d0a1.jpg)  
Figure 17: Example of Ernie-4 with vision response to rare character with English translation.

![](images/6a8a645ee84048bbe388256d4e2f9361f92dfdc9c630a2f258bcb0e44ae06b28.jpg)  
Figure 18: Example of Ernie-4 with vision response to extremely similar character with English translation.

![](images/ca50376053f9eccf584502ac045f93949aea77070c658e797a96bdf3a8c94f28.jpg)  
Figure 19: Example of Ernie-4 with vision response to part of the character as an answer with English translation.

![](images/56941a31a1baa01c31b36db89c0f106f8706d6217e5e8edf9663c6478ca59354.jpg)  
Figure 20: Example of Ernie-4 with vision response a character with different component parts as an answer with English translation.

![](images/6047935c951ab6f06bb5d65cab80172b423e1982f59ca171e8f4ea37643509e7.jpg)  
Figure 21: Example of Kimi-v1 with vision reject rarely used character with English translation.

![](images/44d022e72e011bac12c1a1991202de39706ea5825d1014d01913b1d51e35f528.jpg)  
Figure 22: An example of GPT-4o and o1-preview’s response, with incorrect statements highlighted in red.