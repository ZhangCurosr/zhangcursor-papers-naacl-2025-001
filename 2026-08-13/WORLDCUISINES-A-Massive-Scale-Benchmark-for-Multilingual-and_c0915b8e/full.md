# WORLDCUISINES: A Massive-Scale Benchmark for Multilingual and Multicultural Visual Question Answering on Global Cuisines

Genta Indra Winata∗♠<sup>,1,2</sup>, Frederikus Hudi∗♠<sup>,2,3</sup>, Patrick Amadeus Irawan∗♠<sup>,2,4</sup>,   
David Anugraha∗♠<sup>,5</sup>, Rifki Afina Putri∗♠<sup>,2,6</sup>, Yutong Wang♠<sup>,7</sup>, Adam Nohejl♠<sup>,3</sup>,   
Ubaidillah Ariq Prathama♠<sup>,4</sup>, Nedjma Ousidhoum♠<sup>,8</sup>, Afifa Amriani<sup>9</sup>,   
Anar Rzayev<sup>6</sup>, Anirban Das<sup>1</sup>, Ashmari Pramodya<sup>3</sup>, Aulia Adila<sup>7</sup>, Bryan Wilie<sup>10</sup>,   
Candy Olivia Mawalim<sup>7</sup>, Ching Lam Cheng<sup>11</sup>, Daud Abolade<sup>12,13</sup>,   
Emmanuele Chersoni<sup>14</sup>, Enrico Santus<sup>9</sup>, Fariz Ikhwantri<sup>9</sup>, Garry Kuwanto<sup>15</sup>,   
Hanyang Zhao<sup>16</sup>, Haryo Akbarianto Wibowo<sup>17</sup>, Holy Lovenia<sup>2</sup>,   
Jan Christian Blaise Cruz<sup>2,17</sup>, Jan Wira Gotama Putra<sup>9</sup>, Junho Myung<sup>6</sup>,   
Lucky Susanto<sup>18</sup>, Maria Angelica Riera Machin<sup>3</sup>, Marina Zhukova<sup>19</sup>,   
Michael Anugraha<sup>9</sup>, Muhammad Farid Adilazuarda<sup>2,17</sup>, Natasha Santosa<sup>20</sup>,   
Peerat Limkonchotiwat<sup>2,21</sup>, Raj Dabre<sup>22</sup>, Rio Alexander Audino<sup>4</sup>   
Samuel Cahyawijaya<sup>2,23</sup>, Shi-Xiong Zhang<sup>1</sup>, Stephanie Yulia Salim<sup>7</sup>, Yi Zhou<sup>8</sup>,   
Yinxuan Gui<sup>11</sup>, David Ifeoluwa Adelani♣<sup>,12,24,25,26</sup>, En-Shiun Annie Lee♣<sup>,5,27</sup>,   
Shogo Okada♣<sup>,7</sup>, Ayu Purwarianti♣<sup>,2,4</sup>, Alham Fikri Aji♣<sup>,2,17,18</sup>, Taro Watanabe♣<sup>,3</sup>,   
Derry Tanti Wijaya♣<sup>,15,18</sup>, Alice Oh♣<sup>,6</sup>, Chong-Wah Ngo♣<sup>,11</sup>,   
<sup>1</sup>Capital One <sup>2</sup>SEACrowd <sup>3</sup>NAIST <sup>4</sup>ITB <sup>5</sup>UofT <sup>6</sup>KAIST <sup>7</sup>JAIST <sup>8</sup>Cardiff University   
<sup>9</sup>Independent <sup>10</sup>HKUST <sup>11</sup>SMU <sup>12</sup>Masakhane <sup>13</sup>University of Lagos <sup>14</sup>HK PolyU   
<sup>15</sup>Boston University <sup>16</sup>Columbia University <sup>17</sup>MBZUAI <sup>18</sup>Monash University <sup>19</sup>UCSB   
<sup>20</sup>Tokyo Tech <sup>21</sup>AI Singapore <sup>22</sup>NICT <sup>23</sup>Cohere <sup>24</sup>McGill <sup>25</sup>MILA   
<sup>26</sup>Canada CIFAR AI Chair <sup>27</sup>Ontario Tech   
♠Main Authors ♣Senior Authors

## Abstract

Vision Language Models (VLMs) often struggle with culture-specific knowledge, particularly in languages other than English and in underrepresented cultural contexts. To evaluate their understanding of such knowledge, we introduce WORLDCUISINES, a massivescale benchmark for multilingual and multicultural, visually grounded language understanding. This benchmark includes a visual question answering (VQA) dataset with text-image pairs across 30 languages and dialects, spanning 9 language families and featuring over 1 million data points, making it the largest multicultural VQA benchmark to date. It includes tasks for identifying dish names and their origins. We provide evaluation datasets in two sizes (12k and 60k instances) alongside a training dataset (1 million instances). Our findings show that while VLMs perform better with correct location context, they struggle with adversarial contexts and predicting specific regional cuisines and languages. To support future research, we release a knowledge base with annotated food entries and images along with the VQA data.

![](images/44f4ffb47f015873226b29d88cf3be7e6648061991c0986cd169755446bbdf0e.jpg)  
Figure 1: Images of stuffed pasta and dumplings from our dataset showcase a similar culinary concept across different cultures: wrapping meat, dairy (such as cheese), or vegetables in dough. These dishes can be prepared in various ways, including pan-frying, deepfrying, steaming, or boiling.

## 1 Introduction

Food is an essential medium for the exchange of regional cultures, serving to connect diverse peoples and traditions (Wahlqvist, 2007). Analyzing various culinary practices provides valuable insights into the cultural values, historical narratives, and social customs of the communities that produce and consume these foods (Holtzman, 2006). Furthermore, food plays a significant role in shaping language, which serves as a proxy for cultural knowledge (Freedman, 2021). Food choices often reflect intricate community histories, societal transformations, and both individual and collective identities, thereby creating a rich tapestry of cultural expression (Almerico, 2014). The relationship between culture and food is dynamic; both evolve in tandem over time, resulting in the emergence of new dishes that are influenced by historical culinary traditions (Anderson, 2014).

<table><tr><td></td><td># VQA</td><td># Lang./Dialect†</td><td># Countries</td><td># Food Entries</td><td># Images</td><td>Parallel Data</td><td>License</td></tr><tr><td>FoodieQA (Li et al., 2024b)</td><td>659</td><td>2</td><td>1</td><td>60</td><td>389</td><td>X</td><td>CC BY-NC-ND 4.0</td></tr><tr><td>World Wide Dishes (Magomere et al., 2024)</td><td>765</td><td>131</td><td>63</td><td>765</td><td>301</td><td>X</td><td>CC-BY 4.0</td></tr><tr><td>xGQA (Pfeiffer et al., 2022)</td><td>12,578</td><td>8</td><td>8</td><td>N/A</td><td>398</td><td>√</td><td>CC-BY 4.0</td></tr><tr><td>MaXM‡ (Changpinyo et al., 2023)</td><td>2,142</td><td>7</td><td>7</td><td>N/A</td><td>335</td><td>×</td><td>Custom</td></tr><tr><td>EVJVQA (Nguyen et al., 2023)</td><td>33,790</td><td>3</td><td>1</td><td>N/A</td><td>4,909</td><td>×</td><td>N/A</td></tr><tr><td>CulturalVQA (Nayak et al., 2024)</td><td>2,378</td><td>1</td><td>11</td><td>N/A</td><td>2,328</td><td>X</td><td>N/A</td></tr><tr><td>SEA-VQA (Urailertprasert et al., 2024)</td><td>1,999</td><td>1</td><td>8</td><td>N/A</td><td>515</td><td>×</td><td>Custom</td></tr><tr><td>CVQA (Romero et al., 2024)</td><td>9,000</td><td>26</td><td>28</td><td>1,834</td><td>4,560</td><td>√</td><td>Various</td></tr><tr><td>IndiFoodVQA (Agarwal et al., 2024)</td><td>16,716</td><td>1</td><td>1</td><td>255</td><td>414</td><td>×</td><td>N/A</td></tr><tr><td>WC-VQA</td><td>1,152,000</td><td>30</td><td>189</td><td>2,414</td><td>6,045</td><td>√</td><td>CC BY-SA 4.0</td></tr></table>

Table 1: Data statistics for WC-VQA compared to existing VQA datasets. The data samples are sourced from their respective publications. ‡The reported numbers are based on their human-annotated test set. †This entry includes the language variations we collected for all languages.

As a result, similar food concepts can be found across different countries, reflecting a shared human culinary heritage. Researchers use food as a proxy to model and analyze cultural dynamics, helping to quantify cultural differences across regions (Adilazuarda et al., 2024). Many cultures have developed their own versions of “stuffed pasta” or “dumplings”, each with unique ingredients and preparation methods, often known by different names (Gallani, 2015), as illustrated in Figure 1. Small details like how the dumpling is shaped can signal the cultural background. Conversely, some dishes share the same name but have different meanings; for example, “jelly” in the U.S. refers to a fruit spread, while in the U.K. and parts of Asia, it refers to a gelatinous dessert (Poppe, 1992; Abe, 2013). This culinary diversity presents a challenge for Vision Language Models (VLMs), which must accurately recognize and differentiate food items based on cultural context for applications like food recognition. These models navigate the complexities of names, ingredients, and preparation styles that vary widely across regions. VLMs have shown effectiveness in text captioning (Liu et al., 2024b,c) and have been adapted to support multiple languages (Geigle et al., 2023; Shin et al., 2024).

However, there is limited research on evaluating the multicultural capabilities of VLMs, particularly in terms of multilinguality. The study by Romero et al. (2024) introduce visual question answering (VQA) from a multicultural perspective, but it mainly focuses on knowledge and situational context at a specific moment, which does not fully assess the ability of VLMs to reason and differentiate between cultures within a single question. Moreover, another study on food VQA is limited to Chinese culture and does not explore the broader spectrum of global cultures (Li et al., 2024b). An earlier investigation into cultural bias in language models also found that cultural knowledge is lacking (Naous et al., 2023). Therefore, further research is necessary to address these limitations and enhance our understanding of VLMs’ multicultural and multilingual capabilities.

To facilitate a comprehensive analysis of multilingual and multicultural research, we develop resources for evaluating VLMs. Table 1 summarizes how our work compares to previous studies. Our benchmark stands out for its cultural diversity, offering more VQA datasets and broader language and dialect coverage. Our major contributions can be summarized in three-fold:

• We present WORLDCUISINES, the first massive scale benchmark consisting of 1 million high-quality multilingual and multicultural text-image pairs annotated by native speakers in their local languages. We publicly release our resources, i.e., datasets, <sup>1</sup> code,<sup>2</sup> and leaderboard<sup>3</sup> to advance future research in this rapidly evolving field.

• We evaluate open-source and commercial

![](images/952bc3e258fb1fc1829f833bbe7844aa050982c340015f8cbe5560b544b9e75a.jpg)

Task 1b: Contextualized dish name prediction  
Task 1c: Dish name prediction with adversarial context  
![](images/b14e5d3699ac96986cd7067108c512a3824d8897186256f89d5c28ac76c3472e.jpg)

Task 2: Location prediction  
![](images/073e54b5e843d5102f77f7e6c1c1bf13fc91f1956276e64fc91e5afba009645e.jpg)  
Figure 2: WC-VQA in WORLDCUISINES comprises two primary tasks: (1) predicting dish names and (2) predicting regional cuisines. Task 1 is further divided into three subtasks: (a) no-context, (b) contextualized, and (c) adversarial. We also include two answer types: multiple-choice question (MCQ) and open-ended question (OEQ).

VLMs for cultural awareness through two VQA tasks: predicting dish names from images and context, and identifying their geographical origin. We also assess the impact of context, including adversarial scenarios.

• We create multilingual templates for queries and context (such as the questions in QA pairs) while preserving language varieties, including dialects and registers. This is achieved by creating translations that incorporate different inflections, articles, and contractions. Our goal is to ensure naturalness in each translation and to use appropriate inflections for place names.

## 2 WORLDCUISINES

We propose WORLDCUISINES, an open-source benchmark designed to evaluate the cultural relevance and understanding of VLMs. Figure 2 displays VQA examples in English, alongside selected parallel translations in Japanese and French.

## 2.1 Overview

We develop both a VQA dataset (WC-VQA) and a curated KB for world cuisines (WC-KB). The WC-VQA dataset is constructed using WC-KB, which serves as the primary data source. We design two tasks as follows:

• Task 1: Dish Name Prediction. This task involves predicting the name of a dish based on its image, a question, and contextual information. It comprises three subtasks, each with distinct query types: (a) no-context question, (b) contextualized question, and (c) adversarial contextualized question.

• Task 2: Location prediction. The task is to predict location where the food is commonly consumed and originated given the dish image, question, and a context.

WC-KB. A KB encompassing 2,414 dishes worldwide includes 6,045 images and metadata, covering both coarse-grained (e.g., stew) and finegrained categories (e.g., beef stew), locations, and regional cuisines. It also features multilingual translations of 90 crowd-sourced prompt templates and 401 parallel data entries (i.e., multilingual information) for location and regional cuisine information.

![](images/fdc23dba2383240c4b8285d0ebc7fd52bac05aa6ef1205d57f0fc86b7fa6b7c4.jpg)

Figure 3: WORLDCUISINES distribution of food entries by country in the World Map. The food entries are distributed across 189 countries, with the highest concentration found in Asia, Europe, and North America. There are also some entries from the continents of Africa, Oceania, and Central and South America.  
![](images/d9b044dad9636391a5652aab528a367bd6337da1b4469d912112dcb7a7c612cd.jpg)  
Figure 4: Countries by number of assigned dishes, showing the top 50 countries.

WC-VQA. A multilingual parallel VQA dataset with 1 million samples encompassing over 30 languages and dialects, including various varieties and registers, such as formal and casual styles, with high-quality human annotations. The VQA is designed to evaluate models’ ability to understand cultural food names and their origins.

## 2.2 WC-KB Construction

Our data sources are gathered from Wikipedia<sup>4</sup> and Wikimedia Commons<sup>5</sup> to ensure they can be easily redistributed under an accepted open-source license. The data construction process involves four key steps: (1) dish selection, (2) metadata annotation, (3) quality assurance, and (4) data compilation. Figure 3 provides statistics on the regions covered in our dataset, with detailed information available in Table 9 in the Appendix. Figure 4 shows the distribution of dish frequencies, highlighting the top 50 countries with the most dishes.

## 2.2.1 Dish Selection

We compile a comprehensive list of dish names sourced from Wikipedia. We manually review pages that feature lists of dishes to determine whether each dish is a specialty unique to a specific culture, as we aim to focus on dishes that have distinct cultural significance. We exclude generic categories, such as ice cream, which lacks a specific cultural association. We ensure that each dish on our list has its own dedicated Wikipedia page. If a dish does not have a Wikipedia page, it is also excluded from our compilation. This meticulous approach ensures that our dataset is both culturally relevant and well-documented.

<table><tr><td rowspan="3">Data Split</td><td colspan="6">Task 1 (Dish Name)</td><td rowspan="2" colspan="2">Task 2 (Location)</td><td rowspan="3">Total # VQA</td></tr><tr><td colspan="2">(a) no-context</td><td colspan="2">(b) contextualized</td><td colspan="2">(c) adversarial</td></tr><tr><td>#VQA</td><td># Images</td><td>#VQA</td><td># Images</td><td>#VQA # Images</td><td>#VQA</td><td># Images</td></tr><tr><td>Train (1M)</td><td>270,300</td><td>3,383</td><td>267,930</td><td>3,555</td><td>271,770</td><td>3,589</td><td>270,000</td><td>3,361</td><td>1,080,000</td></tr><tr><td>Test Small (12k)</td><td>3,000</td><td>100</td><td>3,000</td><td>100</td><td>3,000</td><td>100</td><td>3,000</td><td>100</td><td>12,000</td></tr><tr><td>Test Large (60k)</td><td>15,000</td><td>500</td><td>15,000</td><td>500</td><td>15,000</td><td>499</td><td>15,000</td><td>499</td><td>60,000</td></tr></table>

Table 2: Dataset statistics for WC-VQA tasks for train, test small, and test large data splits. Total #VQA represents the total number of VQA from Task 1 and Task 2.

## 2.2.2 Metadata Annotation

Given a dish name and its corresponding Wikipedia page link, we then ask annotators to manually compile metadata based on the provided information. This metadata includes:

• Visual Representation: Images sources from Wikimedia Commons are included, along with their license information.

• Categorization: Dishes are classified into both coarse-grained (e.g., rice, bread) and finegrained (e.g., fried rice, flatbread) categories.

• Description: Annotators provide a description of each dish based on the content from its Wikipedia page, avoiding the use of the dish’s name, origin, or any distinctive keywords that uniquely identify the dish.

• Cuisine: The dish’s origin cuisine and any cuisines with which it is strongly associated.

• Geographic Distribution: This includes the dish’s associated countries, area (city or region), and broader continental region.

The metadata description, along with the example, is further elaborated in the Appendix Table 4.

## 2.2.3 Quality Assurance

Before starting the quality assurance process, we first identify common issues that arise during the annotation and develop automated rules to detect easily identifiable annotation errors, such as incorrect string formatting. Annotators are then asked to correct these errors. To further ensure data quality and validity, we conduct several rounds of quality assurance. Initially, we focus on image quality by removing instances where images are blurry, dark, or contain distracting elements such as people or other dishes. We also verify image licenses by cross-referencing them with information on Wikimedia Commons. Next, we refine the dish categorization and descriptions, ensuring consistency in category assignments and maintaining descriptions free from “information breaches” (e.g., excluding regional details from the description). We standardize cuisine names and eliminate any redundancies. Finally, we meticulously review all country and area information to ensure its accuracy. This comprehensive approach guarantees the integrity and reliability of our dataset.

## 2.2.4 Data Compilation

In this phase, we verify the overall quality check done by annotators, and identify any potential inconsistencies that are missed during the quality assurance. Then, we compile the dataset by collecting the metadata into a single file.

## 2.3 VQA Generation

In this phase, we generate VQA data by sampling from WC-KB. An entry of VQA data comprises visual image, question text, and answer text. This process involves four stages: (1) conducting a similarity search for dish names, (2) constructing questions and contexts, (3) translating these elements into multiple languages, and (4) generating the VQA triplets.

## 2.3.1 Dish Names Similarity Search

To identify similar dishes in our dataset, we follow the approach from Winata et al. (2024) to employ a multilingual model ${ \mathrm { E } } 5 _ { \mathrm { L A R G E } }$ Instruct (Wang et al., 2024) for computing text embedding. Formally, given a dish x with name $x _ { \mathrm { n a m e } }$ and text description $x _ { \mathrm { d e s c } } ,$ we use a multilingual model θ to compute the embedding vector $v _ { x } = \theta ( \{ x _ { \mathrm { n a m e } } ; x _ { \mathrm { d e s c } } \} )$ then apply cosine similarity to compute a score $s = { \mathrm { s i m i l a r i t y } } ( v _ { i } , v _ { j } )$ between dish i and dish $j .$ For each dish, we consider the top-k most similar dishes to generate distractors in the multiple choice question.

## 2.3.2 Question and Context Construction

Dish name prediction (Task 1) is divided into three question variations depending on the context: (1a)

<table><tr><td rowspan="3">Model</td><td colspan="6">Task 1 (Dish Name)</td><td rowspan="2" colspan="2">Task 2</td><td colspan="2">Average</td></tr><tr><td colspan="2">(a) no-context</td><td colspan="2">(b) contextualized</td><td colspan="2">(c) adversarial</td><td colspan="2">(Location)</td></tr><tr><td>MCQ</td><td>OEQ</td><td>MCQ</td><td>OEQ</td><td>MCQ</td><td>OEQ</td><td>MCQ</td><td>OEQ</td><td>MCQ</td><td>OEQ</td></tr><tr><td>Open-Source</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Llava1.6 Vicuna 7B</td><td>34.57</td><td>1.59</td><td>43.48</td><td>4.03</td><td>34.84</td><td>1.41</td><td>32.24</td><td>9.29</td><td>36.28</td><td>4.08</td></tr><tr><td>Llava1.6 Vicuna 13B</td><td>40.17</td><td>2.79</td><td>48.17</td><td>5.85</td><td>39.05</td><td>2.57</td><td>37.79</td><td>10.16</td><td>41.30</td><td>5.34</td></tr><tr><td>Qwen2 VL Instruct 2B</td><td>41.65</td><td>7.98</td><td>42.29</td><td>8.13</td><td>39.69</td><td>6.74</td><td>47.85</td><td>14.55</td><td>42.87</td><td>9.35</td></tr><tr><td>Qwen2 VL Instruct 7B</td><td>61.48</td><td>6.76</td><td>67.85</td><td>10.36</td><td>53.52</td><td>6.12</td><td>55.90</td><td>21.03</td><td>59.69</td><td>11.07</td></tr><tr><td>Qwen2 VL Instruct 72B</td><td>74.19</td><td>12.67</td><td>80.79</td><td>21.31</td><td>62.43</td><td>8.37</td><td>61.90</td><td>27.27</td><td>69.83</td><td>17.40</td></tr><tr><td>Llama 3.2 Instruct 11B</td><td>59.93</td><td>18.75</td><td>64.12</td><td>22.96</td><td>53.17</td><td>13.39</td><td>57.93</td><td>31.58</td><td>58.79</td><td>21.67</td></tr><tr><td>Llama 3.2 Instruct 90B</td><td>77.69</td><td>16.93</td><td>82.92</td><td>23.60</td><td>63.96</td><td>10.87</td><td>67.87</td><td>31.31</td><td>73.11</td><td>20.68</td></tr><tr><td>Molmo-E 1B</td><td>18.81</td><td>0.01</td><td>24.22</td><td>0.23</td><td>19.55</td><td>0.01</td><td>18.97</td><td>1.54</td><td>20.39</td><td>0.45</td></tr><tr><td>Molmo-D 7B</td><td>46.01</td><td>2.89</td><td>55.95</td><td>3.66</td><td>41.61</td><td>2.31</td><td>33.35</td><td>11.45</td><td>44.23</td><td>5.08</td></tr><tr><td>Molmo-O 7B</td><td>39.96</td><td>5.15</td><td>44.93</td><td>6.03</td><td>38.41</td><td>3.51</td><td>29.81</td><td>10.07</td><td>38.28</td><td>6.19</td></tr><tr><td>Pangea 7B‡</td><td>52.35</td><td>1.52</td><td>63.07</td><td>2.73</td><td>49.17</td><td>1.57</td><td>48.71</td><td>20.15</td><td>53.33</td><td>6.49</td></tr><tr><td>Aria 25B</td><td>58.61</td><td>4.99</td><td>69.29</td><td>9.17</td><td>52.82</td><td>3.39</td><td>42.82</td><td>16.20</td><td>55.89</td><td>8.44</td></tr><tr><td>Phi-3.5 Vision 4B</td><td>43.37</td><td>2.91</td><td>48.71</td><td>4.23</td><td>40.87</td><td>2.07</td><td>35.01</td><td>9.22</td><td>41.99</td><td>4.61</td></tr><tr><td>Pixtral 12B</td><td>56.65</td><td>1.22</td><td>70.69</td><td>2.94</td><td>52.12</td><td>1.09</td><td>46.67</td><td>14.43</td><td>56.53</td><td>4.92</td></tr><tr><td>NVLM-D 72B</td><td>69.82</td><td>4.71</td><td>78.93</td><td>10.29</td><td>52.12</td><td>2.89</td><td>51.97</td><td>16.68</td><td>63.21</td><td>8.64</td></tr><tr><td>Proprietary</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-40</td><td>88.45</td><td>21.88</td><td>91.57</td><td>37.51</td><td>82.29</td><td>14.79</td><td>66.52</td><td>37.13</td><td>82.21</td><td>27.83</td></tr><tr><td>GPT-4o Mini</td><td>72.80</td><td>10.28</td><td>81.65</td><td>20.87</td><td>57.76</td><td>5.72</td><td>52.37</td><td>25.79</td><td>66.14</td><td>15.66</td></tr><tr><td>Gemini 1.5 Flash</td><td>77.05</td><td>12.81</td><td>80.97</td><td>15.16</td><td>69.13</td><td>6.46</td><td>71.53</td><td>30.03</td><td>74.67</td><td>16.12</td></tr></table>

Table 3: Accuracy (%) results of WC-VQA for Test Large (60k). MCQ and OEQ indicate multiple-choice question and open-ended question, respectively. Best and second-best are bolded and underlined, respectively. ‡We employ an optimized prompt provided by the authors (see Subsection E.1 in the Appendix for further details).

no-context question, where we simply ask for the name of the dish without any provided context; (1b) contextualized question where we provide additional information related to cuisine or location; and (1c) adversarial contextualized question which are similar to the contextualized questions but may include misleading location information to assess the model’s robustness to irrelevant details.

For example, consider coxinha from Brazil, shown in Figure 2 (1b). A query with additional context here would be: “What is the common name for this dish in Brazil?" Here, the origin of coxinha, Brazil, serves as the context. In contrast, adversarial context involves providing misleading or irrelevant information in terms of location or type of cuisine to assess the model’s robustness to such distractions. For instance, in the case of eggs benedict shown in Figure 2 (1c), an adversarial context would be: “Yesterday I had a nice lunch at a Korean restaurant. I am about to have this dish now. What is this dish called?” In this scenario, the model should ignore the irrelevant detail (“nice lunch at a Korean restaurant”) and focus solely on the image and the question.

Only basic question without any provided context is available for regional cuisine prediction (Task 2). The data statistics for each task are presented in Table 2.

## 2.3.3 Multiple Language Translation

Question and Context. All questions and contexts are initially collected in English, which are then carefully translated by native human speakers into 30 language varieties: 23 different languages with 7 languages having two different varieties each. We instructed the translators to prioritize the naturalness, and then followed by the diversity of translations when the duplication occurs.

Food Name Alias. Using Wikipedia pages as our primary source, we can verify if the English page has translations available in other languages. This enables us to extract dish names in multiple languages and compile them as translations for each dish. We utilize both the Wikipedia page titles in various languages and the alias text found on the English page. These translations are especially valuable for multilingual prompt translation, as they allow us to use the dish’s native name instead of its English equivalent, enhancing cultural relevance and accuracy. We use the English name as default when the translation is unavailable.

Locations and Cuisines. As there are more than 400 unique locations, including countries, cities, and areas, we first translate the English locations into other languages by using GPT-4o, followed by proofreading each translation by the native speakers. The string values for the regional cuisines, i.e., the adjective form of the location in English, are translated in the same manner as location.

![](images/d5dbfc02d13ec564817deb3aed7a7060ff272b234938f5ebcce644d4f64e4285.jpg)  
Figure 5: Accuracy (%) categorized by language (left), language vitality (center), and language family (right). We classify the language vitality by following the classification proposed by Joshi et al. (2020).

Morphological Inflections. Indo-European languages, such as Czech or Spanish, are rich in inflectional morphology which involves word modification to express different grammatical categories, such as number, gender, or case. For example, the equivalents of the phrases “in Japan” and “from Japan” in Czech are “v Japonsku” and “z Japonska”, respectively. We provide a framework for the human translators to use the inflections in the prompt template to prioritize the naturalness while keeping the inflections as few as possible.

## 2.3.4 Generating VQA Triplets

To ensure no overlap in train and test subsets, we split the dishes and the multilingual-questions into two subsets each, to ensure no dish or multilingual questions leakage between train and test. For every subset, we apply random sampling to get a pair of dish and its multilingual-questions. We use the dish entry in our WorldCuisines KB dataset to pick the image and the location to be injected to the context, if any. The answer candidates for multiple-choice were picked by utilizing similarity search (Section 2.3.1). We repeat this process until we reach the desired number of training or test samples, or until all possible dish and question combinations are used, discarding any duplicates.

## 3 Experiments

## 3.1 Experimental Setup

Metrics. We use accuracy as the primary metric to evaluate predictions. For Task 2 (openended), we employ BERTScore (Zhang et al., 2019) with XLM-R Large (Conneau and Lample, 2019) as a secondary metric to determine if the modelgenerated content includes food names similar to those in the gold labels. For open-ended questions, we compute the accuracy of each test sample against multiple references, including translations of the dish in different languages. This approach allows us to accommodate predictions that may not be in the expected language.

Models. We evaluate our benchmark on various available VLMs, including 15 open-source models and 3 proprietary models. During the inference of the open-source model, we use 16-bit floating point and employ greedy decoding. We access the proprietary models through API. The complete list of the models is available in Table 3.

## 4 Results and Discussion

## 4.1 Overall Results

The results for WC-VQA are summarized in Table 3. The multiple-choice question (MCQ) results without any context exhibit significant variability, ranging from 30% to 80%, indicating considerable differences in model performance. This variability indicates that predicting MCQs remains a challenging task for many models. Notably, proprietary models, particularly GPT-4o, demonstrate exceptional performance, outperforming all other models. In the open-ended question (OEQ) setting, the task proves even more difficult than the MCQ, with models achieving a maximum accuracy of less than 20% for dish name predictions and slightly higher for location predictions when no context is provided. However, incorporating context enhances performance across all settings, highlighting that context effectively guides the models in making better predictions. Interestingly, when the adversarial context is introduced, it misleads the models, leading to incorrect predictions and adding further complexity to the task. Among the models evaluated, Llama 3.2 Instruct significantly outperforms other open-source model families, while Qwen2 performs relatively better than Llava 1.6 and Molmo, despite having smaller model sizes.

## 4.2 The Role of Context

For dish name prediction (Task 1), incorporating more relevant context significantly enhances performance across all language families. However, when adversarial context is introduced, performance drops significantly. The adversarial context included in the prompt significantly affects the prediction. Instead of relying solely on the image input, the model often sways and makes predictions based on incorrect location or cuisine information, even when the context is unrelated to the query. This observation is particularly intriguing, as it signifies that such prompts can shift the model’s attention and influence its generation process.

## 4.3 Results by Language

In Task 1 with OEQ setting (Figure 5b), some languages with non-Latin scripts, such as Arabic,

Korean, Japanese, and Marathi, tend to perform poorly, with the exception of Chinese. For Task 2 with OEQ setting, most models struggle with Sino-Tibetan languages (i.e., Chinese, Cantonese, and Hokkien) and Niger-Congo languages (i.e., Yoruba). In contrast, the models demonstrate relatively strong performance with Japonic, Koreanic, Kra-Dai (i.e., Thai), and Turkic (i.e., Azerbaijani) languages. We also observe that answering OEQs in underrepresented languages remains particularly challenging for the models, as shown by the relatively lower results for the “left behind”, “scraping by”, and “hopeful” languages. Interestingly, lower performance in the OEQ does not necessarily translate to the lower performance in the MCQ setting (Figure 5a) where the performance gap between language categories is less pronounced. The gap between OEQ and MCQ, especially for underrepresented languages, suggests that the bottleneck might lie in the factors beyond cultural understanding, such as text generation capabilities.

## 4.4 Scaling Law

It is evident that large models perform better than smaller ones, showing the scaling law still exists in this experiment, as shown in Figure 6. It is very interesting to see the same trend across different model families (e.g., Llava, Qwen, and even GPT-4o series). However, it is pretty clear for opensource models, Llama 3.2 Instruct has the lead for overall performance, which may be due the coverage of multilingual data used in its training, although it is still unclear since there is no evidence or supporting information that can back up the finding. Regardless, NVLM-D model does not perform as good as their base model Qwen2 VL Instruct in our benchmark. One reason could be the NVLM model is highly tuned for English, but not in languages other than English.

## 5 Related Work

Cultural VQA. Several prior studies have focused on developing culturally relevant VQA benchmarks, including FM-IQA (Gao et al., 2015), MCVQA (Gupta et al., 2020), xGQA (Pfeiffer et al., 2022), MaXM (Changpinyo et al., 2023), MTVQA (Tang et al., 2024), MABL (Kabra et al., 2023), MAPS (Liu et al., 2024a), and MaRVL (Liu et al., 2021). Additionally, CVQA (Romero et al., 2024) and CulturalVQA (Nayak et al., 2024) provide VQA datasets that cover various regions and diverse topics, including food, with CVQA also offering questions in multiple languages alongside English translations. SEA-VQA (Urailertprasert et al., 2024) specifically benchmarks the South East Asian region. In contrast, FoodieQA (Li et al., 2024b) and World Wide Dishes (Magomere et al., 2024) are benchmark focusing on food. Our work is similarly motivated by using food as a cultural proxy, but it distinguishes itself with a significantly larger dataset and broader coverage of languages and cultures.

![](images/7935b096831f89f5af7933f3ec3275f8526f1c9c45f9713ebd2235a05a2f6ed5.jpg)  
(a) MCQ Accuracy vs. Parameters.

![](images/30de4bf897f683816e94eca03d273b2f99163e4233370afb2034d28cbbcf076c.jpg)  
(b) OEQ Accuracy vs. Parameters.  
Figure 6: Scaling matters for MCQ (6a) and OEQ (6b).

Multi-modal LLMs. Recent advancements in VLMs have led to the emergence of multi-modal LLMs that can process both images and text. LLaVA (Liu et al., 2024c) exemplifies this approach by utilizing Vicuna (Zheng et al., 2023) as an image encoder, thereby enhancing visual understanding. This architecture has set a precedent for other VLMs, including Qwen2-VL (Bai et al., 2023), Llama 3.2 (Dubey et al., 2024), Pixtral (Agrawal et al., 2024), Phi-3.5 Vision (Abdin et al., 2024), Molmo (Deitke et al., 2024), Aria (Li et al., 2024a), Pangea (Yue et al., 2024), and NVLM (Dai et al., 2024), each leveraging their respective large language models for multi-modal tasks. In a specialized application, FoodLMM (Yin et al., 2023) focuses specifically on the food domain, training on publicly available food datasets and conversational data generated by GPT-4 (Achiam et al., 2023). Our work evaluates the capabilities of these models within the food domain, offering insights into their performance and potential applications in culinary-related tasks across multicultural settings.

## 6 Conclusion

We introduce WORLDCUISINES, an open-source, large-scale benchmark designed for multilingual and multicultural, visually grounded language understanding. It comprises over 1 million data points across 30 languages and dialects. Our findings reveal that this benchmark remains challenging for VLMs, particularly with dishes from specific regions and in low-resource languages. This provides insight into how well models understand regional cuisines. To enhance usability, we offer a dedicated evaluation split with two datasets of varying sizes. Our evaluation shows that while VLMs perform better with the correct context, they struggle with adversarial contexts intended to mislead them. Additionally, we are releasing a comprehensive knowledge base, VQA dataset, code, and leaderboard as open-source resources to support future research.

## Acknowledgements

We extend our gratitude to everyone who has supported our project, especially the numerous annotators who provided meticulous and comprehensive annotations and conducted thorough quality checks. Special thanks to Francesca Porcu for her assistance with the Sardinian language and to Shintaro Ozaki for his help with Japanese. We are also deeply appreciative of Nayeon Lee and Wenliang Dai for their insightful discussions and for integrating NVLM into our benchmark. Additionally, we thank Xiang Yue and Yueqi Song for their help in integrating Pangea into our benchmark.

## Limitations

In this paper, we limit our investigation to avoid exhaustively evaluating all possible models due to resource constraints. Our primary focus is on developing a benchmark that facilitates exploration for future research. We also provide a training data split for reference, allowing other researchers to utilize it to enhance their VLMs and evaluate their models against our test sets. Currently, we include 30 different languages and dialects, establishing one of the largest and most diverse benchmarks for comprehensive multilingual VQA. We aim to extend this benchmark to encompass additional languages in the future, making it more inclusive and representative of a broader range of linguistic diversity.

It is important to note that our food entries are currently sourced from English Wikipedia. Although we aim to include as many diverse dishes as possible, we acknowledge that this approach limits the coverage of some regions. This is due to language affects commonsense and its specific knowledge (Sakai et al., 2024), which in turns suggesting insufficiency of sourcing only English Wikipedia. Nevertheless, our dataset serves as a valuable starting point. In future work, we plan to incorporate entries from non-English Wikipedia pages to improve regional representation and cultural diversity. For evaluation purposes, we include accuracy metrics for overall model performance and BERTScore for more detailed analysis. However, we recognize that evaluating VQA model performance on multicultural data remains an open challenge. Appropriate evaluation metrics are needed to effectively model the diversity of cultural contexts and linguistic variations. Addressing this issue will be a key focus of our future research efforts.

## Ethical Considerations

Our research focuses on evaluating VLMs within the context of multilingual and multicultural VQA, a field that holds significant implications for diverse multilingual communities. We are committed to conducting our data collection and evaluations with the highest standards of transparency and fairness. To achieve this, we have adopted a crowd-sourcing approach for the annotation process, inviting volunteers to contribute and become co-authors if they provide significant contributions. We follow the guidelines from ACL for authorship eligibility as shown in https://www.aclweb.org/adminwiki/index. php/Authorship\_Changes\_Policy\_for\_ACL\_ Conference\_Papers. In line with our commitment to openness and collaboration, we will release our dataset under an open-source license, CC-BY-SA 4.0.

## References

Marah Abdin, Sam Ade Jacobs, Ammar Ahmad Awan, Jyoti Aneja, Ahmed Awadallah, Hany Awadalla, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Harkirat Behl, et al. 2024. Phi-3 technical report: A highly capable language model locally on your phone. arXiv preprint arXiv:2404.14219.

Marié Abe. 2013. Tokyo, japan. In The Ethnomusicologists’ Cookbook, pages 40–45. Routledge.

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Muhammad Farid Adilazuarda, Sagnik Mukherjee, Pradhyumna Lavania, Siddhant Singh, Ashutosh Dwivedi, Alham Fikri Aji, Jacki O’Neill, Ashutosh Modi, and Monojit Choudhury. 2024. Towards measuring and modeling" culture" in llms: A survey. arXiv preprint arXiv:2403.15412.

Pulkit Agarwal, Settaluri Sravanthi, and Pushpak Bhattacharyya. 2024. Indifoodvqa: Advancing visual question answering and reasoning with a knowledgeinfused synthetic data generation pipeline. In Findings of the Association for Computational Linguistics: EACL 2024, pages 1158–1176.

Pravesh Agrawal, Szymon Antoniak, Emma Bou Hanna, Devendra Chaplot, Jessica Chudnovsky, Saurabh Garg, Theophile Gervet, Soham Ghosh, Amélie Héliou, Paul Jacob, et al. 2024. Pixtral 12b. arXiv preprint arXiv:2410.07073.

Gina M Almerico. 2014. Food and identity: Food studies, cultural, and personal identity. Journal ofInternational Business and Cultural Studies, 8:1.

Eugene Newton Anderson. 2014. Everyone eats: Understandingfood and culture. NYU Press.

Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. 2023. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966.

Lyle Campbell and Verónica Grondona. 2008. Ethnologue: Languages of the world. Language, 84(3):636–641.

Soravit Changpinyo, Linting Xue, Michal Yarom, Ashish Thapliyal, Idan Szpektor, Julien Amelot, Xi Chen, and Radu Soricut. 2023. Maxm: Towards multilingual visual question answering. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 2667–2682.

Alexis Conneau and Guillaume Lample. 2019. Crosslingual language model pretraining. Advances in neural information processing systems, 32.

Wenliang Dai, Nayeon Lee, Boxin Wang, Zhuoling Yang, Zihan Liu, Jon Barker, Tuomas Rintamaki, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. 2024. Nvlm: Open frontier-class multimodal llms. arXiv preprint arXiv:2409.11402.

Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, et al. 2024. Molmo and pixmo: Open weights and open data for state-of-the-art multimodal models. arXiv preprint arXiv:2409.17146.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Paul Freedman. 2021. Why Food Matters. Yale University Press.

Barbara Gallani. 2015. Dumplings: A global history. Reaktion Books.

Haoyuan Gao, Junhua Mao, Jie Zhou, Zhiheng Huang, Lei Wang, and Wei Xu. 2015. Are you talking to a machine? dataset and methods for multilingual image question. Advances in neural information processing systems, 28.

Gregor Geigle, Abhay Jain, Radu Timofte, and Goran Glavaš. 2023. mblip: Efficient bootstrapping of multilingual vision-llms. arXiv preprint arXiv:2307.06930.

Deepak Gupta, Pabitra Lenka, Asif Ekbal, and Pushpak Bhattacharyya. 2020. A unified framework for multilingual and code-mixed visual question answering. In Proceedings ofthe 1st conference ofthe Asia-Pacific chapter ofthe associationfor computational linguistics and the 10th internationaljoint conference on natural language processing, pages 900–913.

Jon D Holtzman. 2006. Food and memory. Annu. Rev. Anthropol., 35(1):361–378.

Pratik Joshi, Sebastin Santy, Amar Budhiraja, Kalika Bali, and Monojit Choudhury. 2020. The state and fate of linguistic diversity and inclusion in the nlp world. In Proceedings ofthe 58th Annual Meeting of the Association for Computational Linguistics, pages 6282–6293.

Anubha Kabra, Emmy Liu, Simran Khanuja, Alham Fikri Aji, Genta Winata, Samuel Cahyawijaya, Anuoluwapo Aremu, Perez Ogayo, and Graham Neubig. 2023. Multi-lingual and multi-cultural figurative language understanding. In Findings ofthe Associationfor Computational Linguistics: ACL 2023, pages 8269–8284.

Dongxu Li, Yudong Liu, Haoning Wu, Yue Wang, Zhiqi Shen, Bowen Qu, Xinyao Niu, Guoyin Wang, Bei Chen, and Junnan Li. 2024a. Aria: An open multimodal native mixture-of-experts model. arXiv preprint arXiv:2410.05993.

Wenyan Li, Xinyu Zhang, Jiaang Li, Qiwei Peng, Raphael Tang, Li Zhou, Weijia Zhang, Guimin Hu, Yifei Yuan, Anders Søgaard, et al. 2024b. Foodieqa: A multimodal dataset for fine-grained understanding of chinese food culture. arXiv preprint arXiv:2406.11030.

Chen Liu, Fajri Koto, Timothy Baldwin, and Iryna Gurevych. 2024a. Are multilingual llms culturallydiverse reasoners? an investigation into multicultural proverbs and sayings. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 2016–2039.

Fangyu Liu, Emanuele Bugliarello, Edoardo Maria Ponti, Siva Reddy, Nigel Collier, and Desmond Elliott. 2021. Visually grounded reasoning across languages and cultures. In Proceedings ofthe 2021 Conference on Empirical Methods in Natural Language Processing, pages 10467–10485.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2024b. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26296–26306.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2024c. Visual instruction tuning. Advances in neural information processing systems, 36.

Jabez Magomere, Shu Ishida, Tejumade Afonja, Aya Salama, Daniel Kochin, Foutse Yuehgoh, Imane Hamzaoui, Raesetje Sefala, Aisha Alaagib, Elizaveta Semenova, et al. 2024. You are what you eat? feeding foundation models a regionally diverse food dataset of world wide dishes. arXiv preprint arXiv:2406.09496.

Tarek Naous, Michael J Ryan, Alan Ritter, and Wei Xu. 2023. Having beer after prayer? measuring cultural bias in large language models. arXiv preprint arXiv:2305.14456.

Shravan Nayak, Kanishk Jain, Rabiul Awal, Siva Reddy, Sjoerd van Steenkiste, Lisa Anne Hendricks, Karolina Stanczak, and Aishwarya Agrawal. 2024.´ Benchmarking vision language models for cultural understanding. arXiv preprint arXiv:2407.10920.

Ngan Luu-Thuy Nguyen, Nghia Hieu Nguyen, Duong TD Vo, Khanh Quoc Tran, and Kiet Van Nguyen. 2023. Vlsp2022-evjvqa challenge: Multilingual visual question answering. arXiv preprint arXiv:2302.11752.

Jonas Pfeiffer, Gregor Geigle, Aishwarya Kamath, Jan-Martin Steitz, Stefan Roth, Ivan Vulic, and Iryna´ Gurevych. 2022. xgqa: Cross-lingual visual question answering. In Findings ofthe Associationfor Computational Linguistics: ACL 2022, pages 2497–2511.

J Poppe. 1992. Gelatin. In Thickening and gelling agents for food, pages 98–123. Springer.

David Romero, Chenyang Lyu, Haryo Akbarianto Wibowo, Teresa Lynn, Injy Hamed, Aditya Nanda Kishore, Aishik Mandal, Alina Dragonetti, Artem Abzaliev, Atnafu Lambebo Tonja, et al. 2024. Cvqa: Culturally-diverse multilingual visual question answering benchmark. arXiv preprint arXiv:2406.05967.

Yusuke Sakai, Hidetaka Kamigaito, and Taro Watanabe. 2024. mcsqa: Multilingual commonsense reasoning dataset with unified creation strategy by language models and humans. In Findings ofthe Association for Computational Linguistics: ACL 2024, pages 14182–14214. Association for Computational Linguistics.

DongJae Shin, HyeonSeok Lim, Inho Won, ChangSu Choi, Minjun Kim, SeungWoo Song, HanGyeol Yoo, SangMin Kim, and KyungTae Lim. 2024. X-llava: Optimizing bilingual large vision-language alignment. In Findings of the Association for Computational Linguistics: NAACL 2024, pages 2463–2473.

Jingqun Tang, Qi Liu, Yongjie Ye, Jinghui Lu, Shu Wei, Chunhui Lin, Wanqing Li, Mohamad Fitri Faiz Bin Mahmood, Hao Feng, Zhen Zhao, et al. 2024. Mtvqa: Benchmarking multilingual text-centric visual question answering. arXiv preprint arXiv:2405.11985.

Norawit Urailertprasert, Peerat Limkonchotiwat, Supasorn Suwajanakorn, and Sarana Nutanong. 2024. Sea-vqa: Southeast asian cultural context dataset for visual question answering. In Proceedings of the 3rd Workshop on Advances in Language and Vision Research (ALVR), pages 173–185.

Mark L Wahlqvist. 2007. Regional food culture and development. Asia Pacific journal of clinical nutrition, 16:2.

Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. 2024. Multilingual e5 text embeddings: A technical report. arXiv preprint arXiv:2402.05672.

Genta Indra Winata, Ruochen Zhang, and David Ifeoluwa Adelani. 2024. Miners: Multilingual language models as semantic retrievers. arXiv preprint arXiv:2406.07424.

Yuehao Yin, Huiyan Qi, Bin Zhu, Jingjing Chen, Yu-Gang Jiang, and Chong-Wah Ngo. 2023. Foodlmm: A versatile food assistant using large multi-modal model. arXiv preprint arXiv:2312.14991.

Xiang Yue, Yueqi Song, Akari Asai, Seungone Kim, Jean de Dieu Nyandwi, Simran Khanuja, Anjali Kantharuban, Lintang Sutawika, Sathyanarayanan Ramamoorthy, and Graham Neubig. 2024. Pangea: A fully open multilingual multimodal llm for 39 languages. arXiv preprint arXiv:2410.16153.

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q Weinberger, and Yoav Artzi. 2019. Bertscore: Evaluating text generation with bert. arXiv preprint arXiv:1904.09675.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. 2023. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in Neural Information Processing Systems, 36:46595–46623.

## A Data Statement

## A.1 Executive Summary

WORLDCUISINES is a vision-language benchmark comprised of two resources: (1) WC-VQA, a multilingual parallel question answering dataset covering 30 languages and dialects where each dish image is accompanied by questions and context constructed through human translation; and (2) WC-KB, a knowledge base containing images and metadata associated with the dishes.

## A.2 Curation Rationale

The goal of WORLDCUISINES is to evaluate the cultural understanding of vision-language models (VLMs) within the food domain. To achieve this, we develop WC-VQA and WC-KB. Dish names and their information are collected from English Wikipedia, and the images are selected from Wikimedia Commons to ensure a permissive license, with an emphasis on representing a wide range of food categories and geographic origins (or where the dish is popular). This selection strategy aims to provide insights into the VLMs’ ability to generalize across diverse culinary and cultural contexts.

## A.3 Language Variety

WORLDCUISINES covers 30 languages and dialects spoken across diverse countries and regions. The complete list of languages and dialects is shown in Table 5. An example of the multilingual prompt is shown in Table 6.

## A.4 Annotator Demographic

Over 30 annotators are involved in building WORLDCUISINES, specifically in translating the query and context for the WC-VQA dataset. Most annotators are native speakers of the target languages or dialects included in our data; some are L2 speakers with more than 10 years of study in their respective languages. The detailed demographics for each language are elaborated below.

## A.4.1 Austronesian

Indonesian Two native Indonesian speakers are involved as translators. One is in the 26–35 age range, and the other is in the 16–25 age range.

<table><tr><td>Attribute</td><td>Value</td><td>Description</td><td>Example</td></tr><tr><td>Name</td><td>String</td><td>Name of the dish.</td><td>&quot;Dorayaki&quot;</td></tr><tr><td>Alias</td><td>List&lt;Dict&gt;</td><td>Name alias, i.e. the name in the original language.</td><td>[{“どら焼き”:“Japanese”}]</td></tr><tr><td>Coarse-grained categories</td><td>List&lt;String&gt;</td><td>Coarse-level categories.</td><td>[“Pancake&quot;, &quot;Dessert&quot;]</td></tr><tr><td>Fine-grained categories</td><td>List&lt;String&gt;</td><td>Fine-level categories.</td><td>[“Wagashi Pancake&quot;]</td></tr><tr><td>Cuisines</td><td>String</td><td>Name of cuisine.</td><td>“Japanese&quot;</td></tr><tr><td>Associated Cuisines</td><td>String</td><td>Associated cuisines to the dish.</td><td>&quot;Japanese&quot;</td></tr><tr><td>Area</td><td>String</td><td>Specific region where the dish is originated</td><td>&quot;Ueno&quot;</td></tr><tr><td>Countries</td><td>String</td><td>Specific region where the dish is originated</td><td>&quot;Japan&quot;</td></tr><tr><td>Region[1..5]</td><td>String</td><td>Specific continent where the dish is originated</td><td>“Eastern Asia&quot; “The dish consists of two small pancake-like</td></tr><tr><td>Text Description</td><td>String</td><td>Short description of the dish, including the ingredients used to prepare the dish or the cooking method</td><td>patties made from castella wrapped around a filling of sweet bean paste.&#x27;</td></tr><tr><td>Image[1..8] URL</td><td>String</td><td>Image link to Wikimedia Commons.</td><td>9 “.../commons/9/9c/Dorayaki_001_(3).jpg&#x27;</td></tr><tr><td>Image[1..8] License</td><td>String</td><td>License of the image</td><td>“CC BY-SA 3.0”</td></tr></table>

Table 4: WC-KB attributes in WORLDCUISINES.
<table><tr><td>Language Name</td><td>Language Vitality†</td><td>Resource Classification‡</td><td>Linguistic Register</td><td>Additional Notes</td></tr><tr><td>Austronesian</td><td></td><td></td><td></td><td></td></tr><tr><td>Indonesian</td><td>Institutional</td><td>3 - Rising Star</td><td>Formal Casual</td><td></td></tr><tr><td>Tagalog</td><td>Institutional</td><td>3 - Rising Star</td><td></td><td></td></tr><tr><td>Sundanese</td><td>Stable</td><td>1 - Scraping by</td><td>Loma</td><td>Common speech form</td></tr><tr><td>Javanese</td><td>Institutional</td><td>1 - Scraping by</td><td>Krama Ngoko</td><td>Central-Java dialect, polite form Central-Java dialect, casual form</td></tr><tr><td>Japonic</td><td></td><td></td><td></td><td></td></tr><tr><td>Japanese</td><td>Institutional</td><td>5 - Winners</td><td>Formal Casual</td><td>Polite form or teinei-go</td></tr><tr><td>Sino-Tibetan</td><td></td><td></td><td></td><td>Daily conversation</td></tr><tr><td>Chinese</td><td>Institutional</td><td>5 - Winners</td><td></td><td>Standard Mandarin</td></tr><tr><td>Cantonese</td><td>Institutional</td><td>1 - Scraping by</td><td></td><td></td></tr><tr><td>Hokkien</td><td>Institutional</td><td>0 - Left Behind</td><td>Written</td><td>Medan dialect</td></tr><tr><td></td><td></td><td></td><td>Spoken</td><td>Medan dialect</td></tr><tr><td>Koreanic</td><td></td><td></td><td>Formal</td><td></td></tr><tr><td>Korean</td><td>Institutional</td><td>4 - Underdog</td><td>Casual</td><td></td></tr><tr><td>Kra-Dai</td><td></td><td></td><td></td><td></td></tr><tr><td>Thai</td><td>Institutional</td><td>3 - Rising Star</td><td></td><td></td></tr><tr><td>Indo-European</td><td></td><td></td><td></td><td></td></tr><tr><td>English</td><td>Institutional</td><td>5 - Winners</td><td></td><td></td></tr><tr><td>Spanish</td><td>Institutional</td><td>5 - Winners</td><td></td><td>Latin-American dialect</td></tr><tr><td>French</td><td>Institutional</td><td>5 - Winners</td><td></td><td></td></tr><tr><td>Russian</td><td>Institutional</td><td>4 - Underdog</td><td>Formal Casual</td><td></td></tr><tr><td>Czech</td><td>Institutional</td><td>4 - Underdog</td><td></td><td></td></tr><tr><td>Italian</td><td>Institutional</td><td>4 - Underdog</td><td></td><td></td></tr><tr><td>Hindi</td><td>Institutional</td><td>4 - Underdog</td><td></td><td></td></tr><tr><td>Bengali</td><td>Institutional</td><td>3 - Rising Star</td><td></td><td></td></tr><tr><td>Marathi</td><td>Institutional</td><td>2 - Hopeful</td><td></td><td></td></tr><tr><td>Sardinian</td><td>Endangered</td><td>1 - Scraping by</td><td></td><td>Logudorese (src)</td></tr><tr><td>Sinhala</td><td>Institutional</td><td>0 - Left Behind</td><td>Formal</td><td>Spoken form</td></tr><tr><td>Afro-Asiatic</td><td></td><td></td><td></td><td></td></tr><tr><td>Arabic (MSA)</td><td>Institutional</td><td>5 - Winners</td><td></td><td></td></tr><tr><td>Niger-Congo</td><td></td><td></td><td></td><td></td></tr><tr><td>Yoruba</td><td>Institutional</td><td>2 - Hopeful</td><td></td><td></td></tr><tr><td>Turkic</td><td></td><td></td><td></td><td></td></tr><tr><td>Azerbaijani</td><td>Institutional</td><td>1 - Scraping by</td><td></td><td>North Variety (azj)</td></tr></table>

Table 5: The details of languages used in the prompt generation for our VQA dataset. †Taken from Ethnologue (Campbell and Grondona, 2008). ‡Based on Joshi et al. (2020).

Tagalog One native Tagalog speaker in the 16–25 age range is involved as a translator.

Sundanese Two L2 Sundanese speakers contribute to the translation. One, in the 16–25 age range with 15 years of experience with the Sundanese language, assists with translation. The other, in the 26–35 age range with 25 years of experience with the language, primarily serves as the proofreader.

<table><tr><td rowspan="2">Language</td><td rowspan="2">Multi-choice question (MCQ)</td><td rowspan="2">Question Prompt</td><td rowspan="2">Open-ended question (OEQ) ID</td><td rowspan="2">Answer Text</td></tr><tr><td>Yesterday I had a nice lunch at a Japanese restaurant. I am about to have this dish now. What is this dish called?</td></tr><tr><td>English</td><td>1. Hangtown fry 2. Zucchini slice 3. Chawanmushi 4. Rolex 5. Egg foo young Print only the answer with a single answer id (1,2,3,4,5).</td><td>Yesterday I had a nice lunch at a Japanese restaurant. Print only the answer.</td><td>I am about to have this dish now. What is this dish called?</td><td>5 Egg foo young</td></tr><tr><td>French</td><td>Je suis sur le point de manger ce plat maintenant. Comment appelle-t-on ce plat ? 1. Hangtown fry 2. Zucchini slice 3. Chawanmushi 4. Rolex 5. Fu yung hai Print only the answer with a single answer id (1,2,3,4,5). Kemarin, saya menyantap makan siang yg nikmat di restoran Jepang.</td><td>Je suis sur le point de manger ce plat maintenant. Comment appelle-t-on ce plat ? Print only the answer.</td><td>Hier, j’ai pris un bon déjeuner dans un restaurant japonais.</td><td>5 Fu yung hai</td></tr><tr><td>Indonesian (Formal)</td><td>Sekarang saya akan menyantap hidangan ini. Disebut apakah hidangan ini? 1. Hangtown fry 2. Zucchini slice 3. Chawanmushi 4. Rolex 5. Puyunghai Print only the answer with a single answer id (1,2,3,4,5). Kemarin aku makan siang enak di restoran Jepang.</td><td>Disebut apakah hidangan ini? Print only the answer.</td><td>Kemarin, saya menyantap makan siang yg nikmat di restoran Jepang. Sekarang saya akan menyantap hidangan ini. 5</td><td>Puyunghai</td></tr><tr><td>Indonesian (Casual)</td><td>Sekarang mau makan makanan ini. Makanan ini disebut apa? 1. Hangtown fry 2. Zucchini slice 3. Chawanmushi 4. Rolex 5. Puyunghai Print only the answer with a single answer id (1,2,3,4,5).</td><td>Makanan ini disebut apa? Print only the answer.</td><td>Kemarin aku makan siang enak di restoran Jepang. Sekarang mau makan makanan ini.</td><td>5 Puyunghai</td></tr><tr><td>Japanese (Formal)</td><td>昨日、私は日本料理店で美味しい昼食を食べました。 今まさにこの料理を食べようとしています。 この料理の名前は何ですか? 1. Hangtown fry 2. Zucchini slice 3. 茶碗蒸し 4. Rolex 5. 芙蓉蛋</td><td>この料理の名前は何ですか? Print only the answer.</td><td>昨日、私は日本料理店で美味しい昼食を食べました。 今まさにこの料理を食べようとしています。</td><td>5 芙蓉蛋</td></tr><tr><td>Japanese (Casual)</td><td>Print only the answer with a single answer id (1,2,3,4,5). 昨日日本料理のお店で美味しいランチを食べたんだけど、 今まさに食べてるこの料理の名前は何？ 1. Hangtown fry 2. Zucchini slice 3. 茶碗蒸し 4. Rolex 5. 芙蓉蛋 Print only the answer with a single answer id (1,2,3,4,5).</td><td>Print only the answer.</td><td>昨日日本料理のお店で美味しいランチを食べたんだけど、 今まさに食べてるこの料理の名前は何? 5</td><td>芙蓉蛋</td></tr><tr><td>Javanese (Krama)</td><td>Kaping wingi kula nedha nikmat ing restoran Jepang. Kula kepengin nedha menika malih sakmenika Naminipun nopo dhaharan menika? 1. Hangtown fry 2. Zucchini slice 3. Chawanmushi 4. Rolex 5. Endhog foo young</td><td>Kaping wingi kula nedha nikmat ing restoran Jepang. Kula kepengin nedha menika malih sakmenika. Naminipun nopo dhaharan menika? Print only the answer.</td><td></td><td>5 Endhog foo young</td></tr><tr><td>Javanese (Ngoko)</td><td>Print only the answer with a single answer id (1,2,3,4,5). Wingi aku mangan enak ndek restoran Jepang. Aku pengen mangan neh saiki. Opo jenenge panganan iki? 1. Hangtown fry 2. Zucchini slice 3. Chawanmushi 4. Rolex 5. Endhog foo young Print only the answer with a single answer id (1,2,3,4,5).</td><td>Wingi aku mangan enak ndek restoran Jepang. Aku pengen mangan neh saiki. Opo jenenge panganan iki? Print only the answer.</td><td></td><td>Endhog foo young</td></tr></table>

Table 6: Multilingual prompt example of Task 1 (c) adversarial in 8 language variants (out of 30). The visual image given is an image of Egg foo young, a Chinese cuisine. The “qa\_id” of this example is 1806.

![](images/23af1039cd5991f5cca80ec48e7d46c49c84984e639a23ca1d1a0d1fe417db1b.jpg)  
Figure 7: BERTScore (%) categorized by language (left), language vitality (center), and language family (right). We classify the language vitality by following the classification from Joshi et al. (2020).

![](images/b017ebefec684180f7fe6421542b62ce438a7c6e5bb8298efc75238b9ca484ba.jpg)  
Figure 8: Model performance evaluated with different references on open-ended question.

Javanese One native Javanese speaker with Central Java dialect in the 16–25 age range translates for both registers of the language (Krama and Ngoko).

## A.4.2 Japonic

Japanese Three L2 Japanese speakers with over 10 years of language study contribute to the Japanese translation. Two are in the 26–35 age range, and one is in the 36–45 age range. A native Japanese speaker then proofreads the translated sentences. Additionally, one native Japanese speaker from Western Japan in the 16–25 age range gives input for the casual form.

## A.4.3 Sino-Tibetan

Chinese One native Chinese speaker in the 16– 25 age range is involved as a translator.

Cantonese Two native Cantonese speakers are involved as translators. One is in 36–45 age range, and the other is in the 16–25 age range.

Hokkien Two native Hokkien speakers in the Medan dialect translate for both written and spoken versions of the language. Both are in the 26–35 age range.

## A.4.4 Koreanic

Korean One native Korean speaker in the 16–25 age range translates the formal and casual versions of the language.

## A.4.5 Kra-Dai

Thai One native Thai speaker in the 26–35 age range is involved as a translator.

## A.4.6 Indo-European

English Query and context in English are constructed. All are L2 English speakers with over 20 years of study and have lived in the English speaking countries. Four of the annotators are in the 26–35 age range, and one is in 36–45 age range. Two native English speakers skimmed through the prompt templates.

Spanish One native Spanish speaker in the 26–35 age range translates the Latin-American versions or dialects of the language.

French One native French speaker and one L2 speaker are involved as translators. The native speaker is in the 26–35 age range, and the L2 speaker is in the 36–45 age range.

Russian One native Russian speaker in the 26–35 age range is involved as translators. One L2 speaker in 36–45 proofreads the template for inflection.

Czech One native Czech speaker in the 36–45 age range is involved as a translator.

Italian Two native Italian speakers, both in the 36–45 age range, are involved as translators.

Hindi One native Hindi speaker in the 26–35 age range is involved as a translator.

Bengali One native Bengali speaker in the 26–35 age range is involved as a translator.

Marathi One native Marathi speaker in the 26– 35 age range is involved as a translator.

Sardinian One native Logudorese Sardinian speaker in the 36–45 age range is involved as a translator.

Sinhala One native Sinhala speaker in the 26–35 age range is involved as a translator.

## A.4.7 Afro-Asiatic

Arabic (MSA) One native Arabic speaker in the 26–35 age range is involved in the Modern Standard Arabic (MSA) translation.

## A.4.8 Niger-Congo

Yoruba One native Yoruba speaker in the 16–25 age range is involved as a translator.

## A.4.9 Turkic

Azerbaijani One native Azerbaijani speaker in the 16–25 age range is involved as a translator.

## B Open-Source Collaborative Effort

The WORLDCUISINES data collection and benchmark construction is a fully open-source project. We invite contributions from researchers, practitioners, and grassroots communities, such as local NLP communities, who are interested in participating. Contributions can include data collection, annotation, quality checks, and evaluation. To ensure high-quality data, we engage native speakers of local languages in the annotation process with strict quality control (QC). The contributors who provide substantial contribution are invited to have co-authorship on this paper. We follow the guidelines from ACL for authorship eligibility.<sup>6</sup> Our goal is to develop a resource and benchmark that will have a meaningful impact on future research.

To achieve this, we are dedicated to expanding language coverage and ensuring that contributions are as inclusive and diverse as possible.

## C Detailed Dataset Construction

## C.1 Dataset Compilation

Our dataset, comprising 2,414 dishes and 6,084 images, was meticulously compiled and verified manually. Key metadata includes dish name, alias, coarse- and fine-grained categories, cuisines, regions, descriptions, images, and their licenses. The compilation process followed these steps:

• We listed dish names for annotation.

• Annotators filled metadata fields and selected up to 8 licensed images per dish, guided by instruction documentation with examples for consistent accuracy.

• Post-annotation, annotators formed subgroups to verify specific metadata categories, ensuring detailed and consistent data across fields such as categories, cuisines, regions, descriptions, and images.

## C.2 Negative Sampling

Recall that from our annotations, we have detailed metadata for all 2,414 dishes, including the dish name, coarse-grained categories, fine-grained categories, countries, and text descriptions. The negative answers were sampled using the following procedure:

(1) We used a multilingual model, specifically E5-LARGE Instruct, to compute the text embeddings. Each embedding was generated by concatenating the dish name with its corresponding text description.

(2) To identify negative samples, we computed the cosine similarity between the embeddings of the target dish and those of all other dishes in the dataset. The top-K most similar dishes were selected under three different conditions:

– Same Fine-grained Category: Select top-K dishes from the same fine-grained category as the target dish.

– Same Coarse-grained Category: Select top-K dishes from the same coarsegrained category but potentially different fine-grained categories.

![](images/7d4e080caed1de959e4faf24dd2b9cc4989a19dcf91c30a6af2eca0f96081c7f.jpg)  
Figure 9: Regression Analysis for BERTScore OE vs. Accuracy OE.

– No Category Restriction: Select top-K dishes from the entire dataset without any restriction on categories.

Here, we used K=15, resulting in 45 candidate dishes in total.

(3) Each MCQ consists of five options: one correct answer and four negative answers. The negative answers were chosen as follows:

– Two Difficult Options: The first two negative answers were selected from dishes in the same fine-grained category. These are intended to be more challenging for the model to distinguish.

– One Medium Option: The third negative answer was selected from dishes in the same coarse-grained category.

– One Easy Option: The fourth negative answer was selected from dishes without any category restriction, making it likely to be easier to identify as incorrect.

This approach ensures a balanced difficulty among the negative options, with two difficult, one medium, and one easy negative answer.

(4) (Optional) Specifically for task 2, where the question involves identifying the correct location (country) of a dish, we followed a slightly modified approach:

– From the previously retrieved 4 negative options, we identified the countries associated with each dish.

– We then excluded the countries that are valid locations for the correct dish. The remaining countries were used to create the negative options for the locationbased question.

## D More Results

## D.1 Primary Metric: Accuracy (%)

Table 7 presents the comprehensive results of WC-VQA for both Test Small and Test Large. Additionally, we examine the performance gap between different references used in the evaluation, with the results displayed in Figure 8.

## D.2 Secondary Metric: BERTScore

As a secondary metric, we employ BERTScore using XLM-R Large as the base model. Table 8 presents the comprehensive results of WC-VQA for both Test Small and Test Large. Figure 7 illustrates the model’s performance categorized by language, language vitality, and language family.

Robustness and Error Analysis. Figure 9 illustrates the correlation between BERTScore and accuracy in the open-ended setting through regression analysis. The R-squared value is 0.41, indicating a low correlation between BERTScore and accuracy. Despite this, BERTScore remains a useful metric for assessing whether the model’s predictions have semantic similarity to the gold labels, even if they are not exact matches.

## E Evaluation

## E.1 Prompt Sensitivity

We use the same prompts for all models, with the exception of the Pangea 7B model (Yue et al., 2024). This model is particularly sensitive and lacks robustness in handling diverse prompt instructions, often struggling to follow instructions accurately, especially in multiple-choice questions (MCQs), unless a specific template is applied. In contrast, models like Llama 3.2 Instruct and Qwen2 VL Instruct are more adaptable to varied instructions. After consulting with the authors, we adopted the prompt “Answer with the option letter from the given choices directly.” for MCQ queries when using the Pangea 7B model.

<table><tr><td></td><td colspan="6">Task 1 (Dish Name)</td><td colspan="2">Task 2</td><td colspan="2">Average</td></tr><tr><td>Model (Accuracy %)</td><td colspan="2">(a) no-context</td><td colspan="2">(b) contextualized</td><td colspan="2">(c) adversarial</td><td colspan="2">(Location)</td><td colspan="2">OEQ</td></tr><tr><td></td><td>MCQ</td><td>OEQ</td><td>MCQ</td><td>OEQ</td><td>MCQ</td><td>OEQ</td><td>MCQ</td><td>OEQ</td><td>MCQ</td><td></td></tr><tr><td>Test Small (12k)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Open-Source</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Llava1.6 Vicuna 7B</td><td>33.63</td><td>0.87</td><td>43.13</td><td>2.83</td><td>28.67</td><td>0.60</td><td>27.77</td><td>7.93</td><td>33.30</td><td>3.06</td></tr><tr><td>Llava1.6 Vicuna 13B</td><td>40.87</td><td>1.00</td><td>50.30</td><td>4.17</td><td>38.37</td><td>1.60</td><td>31.07</td><td>8.63</td><td>40.15</td><td>3.85</td></tr><tr><td>Qwen2 VL Instruct 2B</td><td>40.97</td><td>3.33</td><td>44.40</td><td>4.60</td><td>47.07</td><td>3.43</td><td>48.37</td><td>12.50</td><td>45.20</td><td>5.96</td></tr><tr><td>Qwen2 VL Instruct 7B</td><td>63.83 76.13</td><td>4.07</td><td>67.20</td><td>8.57</td><td>57.00</td><td>3.90</td><td>56.80</td><td>21.23</td><td>61.21</td><td>9.44</td></tr><tr><td>Qwen2 VL Instruct 72B</td><td></td><td>10.40 14.37</td><td>81.63 65.57</td><td>17.43</td><td>67.23</td><td>6.27</td><td>56.73</td><td>26.07</td><td>70.43</td><td>15.04</td></tr><tr><td>Llama 3.2 Instruct 11B</td><td>57.93 77.33</td><td></td><td></td><td>19.20</td><td>56.27</td><td>9.50</td><td>46.60</td><td>27.23</td><td>56.59</td><td>17.58</td></tr><tr><td>Llama 3.2 Instruct 90B</td><td>21.87</td><td>14.27 0.00</td><td>83.43</td><td>22.30</td><td>71.23</td><td>9.00</td><td>64.70</td><td>29.73</td><td>74.17</td><td>18.82</td></tr><tr><td>Molmo-E 1B</td><td></td><td></td><td>24.53</td><td>0.13</td><td>20.23</td><td>0.00</td><td>19.60</td><td>1.27</td><td>21.56</td><td>0.35</td></tr><tr><td>Molmo-D 7B</td><td>50.67</td><td>1.00</td><td>57.00</td><td>2.23</td><td>48.67</td><td>1.73</td><td>36.73</td><td>11.70</td><td>48.27</td><td>4.16</td></tr><tr><td>Molmo-O 7B</td><td>46.03</td><td>2.13</td><td>43.27</td><td>4.37</td><td>41.60</td><td>2.10</td><td>26.83</td><td>9.03</td><td>39.43</td><td>4.41</td></tr><tr><td>Pangea 7B</td><td>45.33</td><td>0.43</td><td>59.40</td><td>1.33</td><td>22.17</td><td>0.63</td><td>34.10</td><td>17.90</td><td>40.25</td><td>5.07</td></tr><tr><td>Pangea 7B‡</td><td>54.87</td><td>0.43</td><td>65.77</td><td>1.33</td><td>55.00</td><td>0.63</td><td>48.47</td><td>17.90</td><td>56.03</td><td>5.07</td></tr><tr><td>Aria 25B</td><td>65.77</td><td>2.67</td><td>71.43</td><td>6.47</td><td>57.13</td><td>1.80</td><td>39.60</td><td>15.70</td><td>58.48</td><td>6.66</td></tr><tr><td>Phi-3.5 Vision 4B</td><td>49.27</td><td>1.90</td><td>53.03</td><td>3.03</td><td>42.90</td><td>1.33</td><td>31.23</td><td>8.43</td><td>44.11</td><td>3.67</td></tr><tr><td>Pixtral 12B</td><td>57.57</td><td>0.60</td><td>72.33 78.20</td><td>1.83</td><td>55.40</td><td>0.57</td><td>44.73</td><td>12.83</td><td>57.51</td><td>3.96</td></tr><tr><td>NVLM-D 72B</td><td>75.50</td><td>3.13</td><td></td><td>7.37</td><td>54.67</td><td>1.37</td><td>54.13</td><td>17.40</td><td>65.62</td><td>7.32</td></tr><tr><td>Proprietary</td><td>88.40</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-40</td><td></td><td>16.60</td><td>90.43</td><td>35.47</td><td>82.23</td><td>12.60</td><td>63.60</td><td>35.53</td><td>81.17</td><td>25.05</td></tr><tr><td>GPT-4o Mini</td><td>75.33 78.17</td><td>7.30 16.30</td><td>83.00 82.07</td><td>17.67 23.53</td><td>64.83 71.33</td><td>3.53</td><td>52.87</td><td>26.90</td><td>69.01</td><td>13.85</td></tr><tr><td>Gemini 1.5 Flash</td><td></td><td></td><td></td><td></td><td></td><td>7.33</td><td>66.00</td><td>32.30</td><td>74.39</td><td>19.86</td></tr><tr><td>Test Large (60k)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Open-Source</td><td></td><td>1.59</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Llava1.6 Vicuna 7B Llava1.6 Vicuna 13B</td><td>34.57 40.17</td><td>2.79</td><td>43.48</td><td>4.03</td><td>34.84</td><td>1.41</td><td>32.24 37.79</td><td>9.29</td><td>36.28</td><td>4.08 5.34</td></tr><tr><td>Qwen2 VL Instruct 2B</td><td>41.65</td><td>7.98</td><td>48.17 42.29</td><td>5.85 8.13</td><td>39.05</td><td>2.57</td><td>47.85</td><td>10.16 14.55</td><td>41.30</td><td>9.35</td></tr><tr><td>Qwen2 VL Instruct 7B</td><td>61.48</td><td>6.76</td><td>67.85</td><td></td><td>39.69</td><td>6.74</td><td></td><td></td><td>42.87</td><td>11.07</td></tr><tr><td>Qwen2 VL Instruct 72B</td><td>74.19</td><td>12.67</td><td></td><td>10.36</td><td>53.52</td><td>6.12</td><td>55.90</td><td>21.03</td><td>59.69</td><td></td></tr><tr><td></td><td>59.93</td><td></td><td>80.79</td><td>21.31</td><td>62.43</td><td>8.37</td><td>61.90</td><td>27.27</td><td>69.83</td><td>17.40</td></tr><tr><td>Llama 3.2 Instruct 11B</td><td></td><td>18.75</td><td>64.12</td><td>22.96</td><td>53.17</td><td>13.39</td><td>57.93</td><td>31.58</td><td>58.79</td><td>21.67</td></tr><tr><td>Llama 3.2 Instruct 90B</td><td>77.69</td><td>16.93</td><td>82.92</td><td>23.60</td><td>63.96</td><td>10.87</td><td>67.87</td><td>31.31</td><td>73.11</td><td>20.68</td></tr><tr><td>Molmo-E 1B</td><td>18.81</td><td>0.01</td><td>24.22</td><td>0.23</td><td>19.55</td><td>0.01</td><td>18.97</td><td>1.54</td><td>20.39</td><td>0.45</td></tr><tr><td>Molmo-D 7B</td><td>46.01</td><td>2.89</td><td>55.95</td><td>3.66</td><td>41.61</td><td>2.31</td><td>33.35</td><td>11.45</td><td>44.23</td><td>5.08</td></tr><tr><td>Molmo-O 7B</td><td>39.96</td><td>5.15</td><td>44.93</td><td>6.03 2.73</td><td>38.41 21.77</td><td>3.51 1.57</td><td>29.81 37.15</td><td>10.07 20.15</td><td>38.28 39.56</td><td>6.19 6.49</td></tr><tr><td>Pangea 7B Pangea 7B‡</td><td>41.38 52.35</td><td>1.52 1.52</td><td>57.95 63.07</td></table>

Table 7: Accuracy (%) results of WC-VQA. MCQ and OEQ indicate multiple-choice question and open-ended question, respectively. Best and second-best are bolded and underlined, respectively. ‡We employ an optimized prompt provided by the authors (see Subsection E.1 in the Appendix for further details).

![](images/de70a00cecf6628e5a601e133f901a133ce3cd73a317171346b30d2ea4b1741a.jpg)  
Figure 10: Model performance with different references on open-ended question.

![](images/785c525e75b455f5fdfda66af085ec255af2d011ff014ab372dfa00370411f73.jpg)  
Figure 11: Dish frequency by country showing 189 countries.

<table><tr><td rowspan="2">Model (BERTScore)</td><td colspan="3">Task 1 (Dish Name)</td><td rowspan="2">Task 2 (Location)</td><td rowspan="2">Average</td></tr><tr><td>(a) no-context</td><td>(b) contextualized</td><td>(c) adversarial</td></tr><tr><td>Test Small (12k)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Open-Source</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Llava1.6 Vicuna 7B</td><td>81.49</td><td>82.13</td><td>81.56</td><td>85.45</td><td>82.66</td></tr><tr><td>Llava1.6 Vicuna 13B</td><td>80.50</td><td>80.65</td><td>80.14</td><td>81.77</td><td>80.77</td></tr><tr><td>Qwen2 VL Instruct 2B</td><td>82.48</td><td>82.75</td><td>82.34</td><td>84.29</td><td>82.97</td></tr><tr><td>Qwen2 VL Instruct 7B</td><td>82.65</td><td>83.13</td><td>82.10</td><td>87.22</td><td>83.78</td></tr><tr><td>Qwen2 VL Instruct 72B</td><td>83.78</td><td>84.63</td><td>83.06</td><td>87.10</td><td>84.64</td></tr><tr><td>Llama 3.2 Instruct 11B</td><td>82.45</td><td>82.93</td><td>81.64</td><td>82.59</td><td>82.40</td></tr><tr><td>Llama 3.2 Instruct 90B</td><td>82.82</td><td>83.44</td><td>81.98</td><td>85.70</td><td>83.48</td></tr><tr><td>Molmo-E 1B</td><td>81.17</td><td>81.12</td><td>81.24</td><td>83.58</td><td>81.78</td></tr><tr><td>Molmo-D 7B</td><td>81.26</td><td>81.65</td><td>80.55</td><td>84.87</td><td>82.08</td></tr><tr><td>Molmo-O 7B</td><td>82.14</td><td>82.24</td><td>81.44</td><td>84.38</td><td>82.55</td></tr><tr><td>Pangea 7B</td><td>81.29</td><td>81.78</td><td>80.19</td><td>86.31</td><td>82.39</td></tr><tr><td>Aria 25B</td><td>79.85</td><td>80.26</td><td>79.86</td><td>80.53</td><td>80.12</td></tr><tr><td>Phi-3.5 Vision 4B</td><td>80.82</td><td>79.66</td><td>76.77</td><td>83.25</td><td>80.12</td></tr><tr><td>Pixtral 12B</td><td>78.84</td><td>79.12</td><td>78.90</td><td>86.40</td><td>80.81</td></tr><tr><td>NVLM-D 72B</td><td>81.39</td><td>82.05</td><td>79.98</td><td>85.64</td><td>82.27</td></tr><tr><td>Proprietary</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-40</td><td>84.86</td><td>86.92</td><td>83.89</td><td>88.98</td><td>86.16</td></tr><tr><td>GPT-4o Mini</td><td>83.10</td><td>83.91</td><td>82.16</td><td>87.34</td><td>84.13</td></tr><tr><td>Gemini 1.5 Flash</td><td>84.68</td><td>85.09</td><td>83.11</td><td>89.15</td><td>85.51</td></tr><tr><td>Test Large (60k)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Open-Source</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Llava1.6 Vicuna 7B</td><td>81.63</td><td>82.10</td><td>81.58</td><td>85.81</td><td>82.78</td></tr><tr><td>Llava1.6 Vicuna 13B</td><td>80.65</td><td>80.70</td><td>80.12</td><td>81.86</td><td>80.83</td></tr><tr><td>Qwen2 VL Instruct 2B</td><td>82.95</td><td>83.10</td><td>82.81</td><td>84.51</td><td>83.34</td></tr><tr><td>Qwen2 VL Instruct 7B</td><td>82.92</td><td>83.42</td><td>82.30</td><td>87.39</td><td>84.01</td></tr><tr><td>Qwen2 VL Instruct 72B</td><td>83.72</td><td>85.10</td><td>83.11</td><td>87.42</td><td>84.84</td></tr><tr><td>Llama 3.2 Instruct 11B</td><td>82.54</td><td>82.79</td><td>81.64</td><td>82.88</td><td>82.46</td></tr><tr><td>Llama 3.2 Instruct 90B</td><td>83.05</td><td>83.51</td><td>81.95</td><td>85.85</td><td>83.59</td></tr><tr><td>Molmo-E 1B</td><td>81.17</td><td>81.10</td><td>81.13</td><td>83.87</td><td>81.82</td></tr><tr><td>Molmo-D 7B</td><td>81.39</td><td>81.63</td><td>80.73</td><td>85.10</td><td>82.21</td></tr><tr><td>Molmo-O 7B</td><td>82.27</td><td>82.21</td><td>81.52</td><td>84.63</td><td>82.66</td></tr><tr><td>Pangea 7B</td><td>81.40</td><td>81.91</td><td>80.23</td><td>86.79</td><td>82.58</td></tr><tr><td>Aria 25B</td><td>79.89</td><td>80.20</td><td>79.83</td><td>80.63</td><td>80.14</td></tr><tr><td>Phi-3.5 Vision 4B</td><td>80.98</td><td>79.55</td><td>77.61</td><td>83.31</td><td>80.36</td></tr><tr><td>Pixtral 12B</td><td>79.00</td><td>79.33</td><td>78.98</td><td>86.75</td><td>81.02</td></tr><tr><td>NVLM-D 72B</td><td>81.54</td><td>82.17</td><td>80.05</td><td>85.67</td><td>82.36</td></tr><tr><td>Proprietary</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-40</td><td>85.04</td><td>86.93</td><td>83.92</td><td>89.06</td><td>86.24</td></tr><tr><td>GPT-4o Mini</td><td>83.19</td><td>84.05</td><td>82.38</td><td>87.30</td><td>84.23</td></tr><tr><td>Gemini 1.5 Flash</td><td>84.47</td><td>84.97</td><td>83.14</td><td>89.43</td><td>85.50</td></tr></table>

Table 8: BERTScore results of WC-VQA. Only the results from open-ended (OEQ) are used. Best and second-best are bolded and underlined, respectively.

<table><tr><td>Continents/Regions</td><td># Countries</td><td># Food Entries</td><td>% in Our Data</td></tr><tr><td>Global*</td><td>N/A</td><td>96</td><td>3.98%</td></tr><tr><td>Africa</td><td>52</td><td>190</td><td>7.87%</td></tr><tr><td>Eastern Africa</td><td>18</td><td>40</td><td>1.7%</td></tr><tr><td>Middle Africa</td><td>6</td><td>17</td><td>0.7%</td></tr><tr><td>Northern Africa</td><td>7</td><td>67</td><td>2.8%</td></tr><tr><td>Southern Africa</td><td>5</td><td>33</td><td>1.4%</td></tr><tr><td>Western Africa</td><td>16</td><td>60</td><td>2.5%</td></tr><tr><td>America</td><td>37</td><td>472</td><td>19.55%</td></tr><tr><td>Caribbean</td><td>15</td><td>60</td><td>2.5%</td></tr><tr><td>Central America</td><td>8</td><td>134</td><td>5.6%</td></tr><tr><td>Northern America</td><td>2</td><td>230</td><td>9.5%</td></tr><tr><td>South America</td><td>12</td><td>109</td><td>4.5%</td></tr><tr><td>Europe</td><td>47</td><td>808</td><td>33.47%</td></tr><tr><td>Eastern Europe</td><td>10</td><td>164</td><td>6.8%</td></tr><tr><td>Northern Europe</td><td>15</td><td>237</td><td>9.8%</td></tr><tr><td>Southern Europe</td><td>13</td><td>300</td><td>12.4%</td></tr><tr><td>Western Europe</td><td>9</td><td>233</td><td>9.7%</td></tr><tr><td>Asia</td><td>53</td><td>1,052</td><td>43.58%</td></tr><tr><td>Central Asia</td><td>5</td><td>10</td><td>0.4%</td></tr><tr><td>Eastern Asia</td><td>9</td><td>420</td><td>17.4%</td></tr><tr><td>South Eastern Asia</td><td>12</td><td>362</td><td>15.0%</td></tr><tr><td>Southern Asia</td><td>9</td><td>200</td><td>8.3%</td></tr><tr><td>Western Asia</td><td>18</td><td>155</td><td>6.4%</td></tr><tr><td>Oceania</td><td>3</td><td>37</td><td>1.53%</td></tr><tr><td>Australia &amp; New Zealand</td><td>2</td><td>33</td><td>1.4%</td></tr><tr><td>Melanesia</td><td>1</td><td>4</td><td>0.2%</td></tr><tr><td>Micronesia</td><td>-</td><td>-</td><td></td></tr><tr><td>Polynesia</td><td></td><td></td><td></td></tr></table>

Table 9: Geographical distribution of WC-KB, corresponds to Figure 3. Note that there are food entries linked to multiple regions, with some linked to multiple continents. ∗Global denotes entries with more than five regions.

<table><tr><td rowspan=1 colspan=3>Country               Count   %</td><td rowspan=1 colspan=3>Country             Count  %</td><td rowspan=1 colspan=3>Country                       Count   %</td></tr><tr><td rowspan=1 colspan=3>United States            216  9.47</td><td rowspan=1 colspan=3>Argentina             25   1.10</td><td rowspan=1 colspan=3>Grenada                          8   0.35</td></tr><tr><td rowspan=1 colspan=3>Japan                   182  7.98</td><td rowspan=1 colspan=3>Saudi Arabia           24   1.05</td><td rowspan=1 colspan=3>Cameroon                        8   0.35</td></tr><tr><td rowspan=1 colspan=3>China                   177  7.76</td><td rowspan=1 colspan=1>North Macedonia</td><td rowspan=1 colspan=2>24   1.05</td><td rowspan=1 colspan=3>Somalia                          8   0.35</td></tr><tr><td rowspan=1 colspan=3>Indonesia               143  6.27</td><td rowspan=1 colspan=1>Cuba</td><td rowspan=1 colspan=2>23   1.01</td><td rowspan=1 colspan=3>Antigua and Barbuda              7   0.31</td></tr><tr><td rowspan=1 colspan=3>Philippines              133  5.83</td><td rowspan=1 colspan=1>Libya</td><td rowspan=1 colspan=2>23   1.01</td><td rowspan=1 colspan=3>Maldives                         7   0.31</td></tr><tr><td rowspan=1 colspan=3>Mexico                 132  5.78</td><td rowspan=1 colspan=1>Montenegro</td><td rowspan=1 colspan=2>23   1.01</td><td rowspan=1 colspan=3>Kyrgyzstan                       7   0.31</td></tr><tr><td rowspan=1 colspan=3>India                    129  5.65</td><td rowspan=1 colspan=1>Chile</td><td rowspan=1 colspan=2>23   1.01</td><td rowspan=1 colspan=3>Tajikistan                         7   0.31</td></tr><tr><td rowspan=1 colspan=3>France                  117  5.12</td><td rowspan=1 colspan=3>Ireland                23   1.01</td><td rowspan=1 colspan=3>Togo                             7   0.31</td></tr><tr><td rowspan=1 colspan=3>Italy                     99  4.34</td><td rowspan=1 colspan=1>Peru</td><td rowspan=1 colspan=2>22  0.96</td><td rowspan=1 colspan=3>Uganda                          7   0.31</td></tr><tr><td rowspan=1 colspan=3>Spain                    87  3.81</td><td rowspan=1 colspan=1>Hong Kong</td><td rowspan=1 colspan=2>22  0.96</td><td rowspan=1 colspan=3>Benin                            7   0.31</td></tr><tr><td rowspan=1 colspan=3>United Kingdom          80  3.51</td><td rowspan=1 colspan=1>Denmark</td><td rowspan=1 colspan=2>22  0.96</td><td rowspan=1 colspan=3>Macau                           7   0.31</td></tr><tr><td rowspan=1 colspan=3>Global                   80  3.51</td><td rowspan=1 colspan=1>Colombia</td><td rowspan=1 colspan=1>21</td><td rowspan=1 colspan=1>0.92</td><td rowspan=1 colspan=3>Guyana                          7   0.31</td></tr><tr><td rowspan=1 colspan=2>Germany                 77</td><td rowspan=1 colspan=1>3.37</td><td rowspan=1 colspan=1>Armenia</td><td rowspan=1 colspan=1>21</td><td rowspan=1 colspan=1>0.92</td><td rowspan=1 colspan=3>Saint Kitts and Nevis              7   0.31</td></tr><tr><td rowspan=1 colspan=2>Russia                   76</td><td rowspan=1 colspan=1>3.33</td><td rowspan=1 colspan=1>Lithuania</td><td rowspan=1 colspan=2>21  0.92</td><td rowspan=1 colspan=3>Saint Lucia                       7   0.31</td></tr><tr><td rowspan=1 colspan=3>Turkey                   69  3.02</td><td rowspan=1 colspan=1>Belgium</td><td rowspan=1 colspan=2>20  0.88</td><td rowspan=1 colspan=3>Saint Vincent and the Grenadines    7   0.31</td></tr><tr><td rowspan=1 colspan=3>Korea                    66  2.89</td><td rowspan=1 colspan=1>Brunei Darussalam</td><td rowspan=1 colspan=2>20  0.88</td><td rowspan=1 colspan=3>Fiji                              5   0.22</td></tr><tr><td rowspan=1 colspan=1>Iran</td><td rowspan=1 colspan=2>65  2.85</td><td rowspan=1 colspan=1>Czech Republic</td><td rowspan=1 colspan=2>20  0.88</td><td rowspan=1 colspan=3>Mongolia                         5   0.22</td></tr><tr><td rowspan=1 colspan=1>Thailand</td><td rowspan=1 colspan=2>58   2.54</td><td rowspan=1 colspan=1>New Zealand</td><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>0.88</td><td rowspan=1 colspan=3>Liechtenstein                     5   0.22</td></tr><tr><td rowspan=1 colspan=1>Singapore</td><td rowspan=1 colspan=1>58</td><td rowspan=1 colspan=1>2.54</td><td rowspan=1 colspan=1>Finland</td><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>0.83</td><td rowspan=1 colspan=3>Macedonia                        5   0.22</td></tr><tr><td rowspan=1 colspan=1>Portugal</td><td rowspan=1 colspan=1>57</td><td rowspan=1 colspan=1>2.50</td><td rowspan=1 colspan=1>Dominican Republic</td><td rowspan=1 colspan=1>18</td><td rowspan=1 colspan=1>0.79</td><td rowspan=1 colspan=3>Malta                            5   0.22</td></tr><tr><td rowspan=1 colspan=1>Brazil</td><td rowspan=1 colspan=1>54</td><td rowspan=1 colspan=1>2.37</td><td rowspan=1 colspan=1>Yemen</td><td rowspan=1 colspan=1>18</td><td rowspan=1 colspan=1>0.79</td><td rowspan=1 colspan=3>Mozambique                      5   0.22</td></tr><tr><td rowspan=1 colspan=1>Israel</td><td rowspan=1 colspan=1>48</td><td rowspan=1 colspan=1>2.10</td><td rowspan=1 colspan=1>Azerbaijan</td><td rowspan=1 colspan=2>18  0.79</td><td rowspan=1 colspan=3>Angola                           5   0.22</td></tr><tr><td rowspan=1 colspan=3>Romania                 47  2.06</td><td rowspan=1 colspan=3>Moldova              18  0.79</td><td rowspan=1 colspan=3>Cabo Verde                       5   0.22</td></tr><tr><td rowspan=1 colspan=2>Austria                  46</td><td rowspan=1 colspan=1>2.02</td><td rowspan=1 colspan=2>Bhutan                17</td><td rowspan=1 colspan=1>0.75</td><td rowspan=1 colspan=3>Turkmenistan                     5   0.22</td></tr><tr><td rowspan=1 colspan=1>Poland</td><td rowspan=1 colspan=1>45</td><td rowspan=1 colspan=1>1.97</td><td rowspan=1 colspan=2>Puerto Rico            17</td><td rowspan=1 colspan=1>0.75</td><td rowspan=1 colspan=3>Costa Rica                        5   0.22</td></tr><tr><td rowspan=1 colspan=1>Pakistan</td><td rowspan=1 colspan=1>44</td><td rowspan=1 colspan=1>1.93</td><td rowspan=1 colspan=1>Venezuela</td><td rowspan=1 colspan=1>17</td><td rowspan=1 colspan=1>0.75</td><td rowspan=1 colspan=3>Burkina Faso                      4   0.18</td></tr><tr><td rowspan=1 colspan=1>Vietnam</td><td rowspan=1 colspan=1>44</td><td rowspan=1 colspan=1>1.93</td><td rowspan=1 colspan=1>Uruguay</td><td rowspan=1 colspan=1>17</td><td rowspan=1 colspan=1>0.75</td><td rowspan=1 colspan=1>Luxembourg</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>4   0.18</td></tr><tr><td rowspan=1 colspan=1>Canada</td><td rowspan=1 colspan=1>43</td><td rowspan=1 colspan=1>1.89</td><td rowspan=1 colspan=1>Bolivia</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>0.70</td><td rowspan=1 colspan=1>Djibouti</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>4   0.18</td></tr><tr><td rowspan=1 colspan=1>Greece</td><td rowspan=1 colspan=1>42</td><td rowspan=1 colspan=1>1.84</td><td rowspan=1 colspan=1>Trinidad and Tobago</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>0.70</td><td rowspan=1 colspan=2>Iceland</td><td rowspan=1 colspan=1>4   0.18</td></tr><tr><td rowspan=1 colspan=1>Ukraine</td><td rowspan=1 colspan=1>42</td><td rowspan=1 colspan=1>1.84</td><td rowspan=1 colspan=1>Georgia</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>0.70</td><td rowspan=1 colspan=3>Sierra Leone                      4   0.18</td></tr><tr><td rowspan=1 colspan=1>Bulgaria</td><td rowspan=1 colspan=1>41</td><td rowspan=1 colspan=1>1.80</td><td rowspan=1 colspan=2>Norway               16</td><td rowspan=1 colspan=1>0.70</td><td rowspan=1 colspan=3>Niger                            4   0.18</td></tr><tr><td rowspan=1 colspan=1>Slovenia</td><td rowspan=1 colspan=1>39</td><td rowspan=1 colspan=1>1.71</td><td rowspan=1 colspan=2>Cambodia             15</td><td rowspan=1 colspan=1>0.66</td><td rowspan=1 colspan=3>Mauritius                         3   0.13</td></tr><tr><td rowspan=1 colspan=1>Egypt</td><td rowspan=1 colspan=1>38</td><td rowspan=1 colspan=1>1.67</td><td rowspan=1 colspan=2>Afghanistan            15</td><td rowspan=1 colspan=1>0.66</td><td rowspan=1 colspan=3>Guinea                           3   0.13</td></tr><tr><td rowspan=1 colspan=1>Syria</td><td rowspan=1 colspan=1>38</td><td rowspan=1 colspan=1>1.67</td><td rowspan=1 colspan=1>Slovakia</td><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=1>0.66</td><td rowspan=1 colspan=2>Zimbabwe                        3</td><td rowspan=1 colspan=1>0.13</td></tr><tr><td rowspan=1 colspan=1>Nepal</td><td rowspan=1 colspan=1>37</td><td rowspan=1 colspan=1>1.62</td><td rowspan=1 colspan=1>Ethiopia</td><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=1>0.66</td><td rowspan=1 colspan=2>Namibia                          3</td><td rowspan=1 colspan=1>0.13</td></tr><tr><td rowspan=1 colspan=1>Serbia</td><td rowspan=1 colspan=1>36</td><td rowspan=1 colspan=1>1.58</td><td rowspan=1 colspan=1>Latvia</td><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=1>0.66</td><td rowspan=1 colspan=2>Lesotho</td><td rowspan=1 colspan=1>0.13</td></tr><tr><td rowspan=1 colspan=1>Myanmar</td><td rowspan=1 colspan=1>34</td><td rowspan=1 colspan=1>1.49</td><td rowspan=1 colspan=2>Laos                  14</td><td rowspan=1 colspan=1>0.61</td><td rowspan=1 colspan=2>Zambia</td><td rowspan=1 colspan=1>3   0.13</td></tr><tr><td rowspan=1 colspan=1>Lebanon</td><td rowspan=1 colspan=1>34</td><td rowspan=1 colspan=1>1.49</td><td rowspan=1 colspan=2>Guatemala             14</td><td rowspan=1 colspan=1>0.61</td><td rowspan=1 colspan=3>Congo                           3   0.13</td></tr><tr><td rowspan=1 colspan=1>Tunisia</td><td rowspan=1 colspan=1>34</td><td rowspan=1 colspan=1>1.49</td><td rowspan=1 colspan=2>Ghana                13</td><td rowspan=1 colspan=1>0.57</td><td rowspan=1 colspan=3>Gambia                          3   0.13</td></tr><tr><td rowspan=1 colspan=1>Bangladesh</td><td rowspan=1 colspan=1>33</td><td rowspan=1 colspan=1>1.45</td><td rowspan=1 colspan=2>United Arab Emirates   12</td><td rowspan=1 colspan=1>0.53</td><td rowspan=1 colspan=2>Liberia                           3</td><td rowspan=1 colspan=1>0.13</td></tr><tr><td rowspan=1 colspan=1>Malaysia</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>1.40</td><td rowspan=1 colspan=2>Kuwait                12</td><td rowspan=1 colspan=1>0.53</td><td rowspan=1 colspan=2>Comoros                         3</td><td rowspan=1 colspan=1>0.13</td></tr><tr><td rowspan=1 colspan=1>Sri Lanka</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>1.40</td><td rowspan=1 colspan=1>Paraguay</td><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=1>0.48</td><td rowspan=1 colspan=2>South Korea                      3</td><td rowspan=1 colspan=1>0.13</td></tr><tr><td rowspan=1 colspan=1>Nigeria</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>1.40</td><td rowspan=1 colspan=1>El Salvador</td><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=1>0.48</td><td rowspan=1 colspan=1>Wales</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>0.13</td></tr><tr><td rowspan=1 colspan=1>Jamaica</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>1.40</td><td rowspan=1 colspan=2>Bahrain               11</td><td rowspan=1 colspan=1>0.48</td><td rowspan=1 colspan=1>Honduras</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>0.13</td></tr><tr><td rowspan=1 colspan=1>Netherlands</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>1.40</td><td rowspan=1 colspan=2>Haiti                  11</td><td rowspan=1 colspan=1>0.48</td><td rowspan=1 colspan=2>Anguilla                         1</td><td rowspan=1 colspan=1>0.04</td></tr><tr><td rowspan=1 colspan=1>Albania</td><td rowspan=1 colspan=2>32   1.40</td><td rowspan=1 colspan=2>Uzbekistan            10</td><td rowspan=1 colspan=1>0.44</td><td rowspan=1 colspan=2>Western Sahara                    1</td><td rowspan=1 colspan=1>0.04</td></tr><tr><td rowspan=1 colspan=1>South Africa</td><td rowspan=1 colspan=2>32   1.40</td><td rowspan=1 colspan=2>Kazakhstan            10</td><td rowspan=1 colspan=1>0.44</td><td rowspan=1 colspan=2>Faroe Islands                      1</td><td rowspan=1 colspan=1>0.04</td></tr><tr><td rowspan=1 colspan=1>Australia</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>1.40</td><td rowspan=1 colspan=2>Eritrea                10</td><td rowspan=1 colspan=1>0.44</td><td rowspan=1 colspan=2>Seychelles                        1</td><td rowspan=1 colspan=1>0.04</td></tr><tr><td rowspan=1 colspan=1>Hungary</td><td rowspan=1 colspan=1>31</td><td rowspan=1 colspan=1>1.36</td><td rowspan=1 colspan=2>Oman                 10</td><td rowspan=1 colspan=1>0.44</td><td rowspan=1 colspan=2>Burundi                          1</td><td rowspan=1 colspan=1>0.04</td></tr><tr><td rowspan=1 colspan=1>Palestine</td><td rowspan=1 colspan=2>30   1.32</td><td rowspan=1 colspan=2>Qatar                 10</td><td rowspan=1 colspan=1>0.44</td><td rowspan=1 colspan=2>Rwanda                          1</td><td rowspan=1 colspan=1>0.04</td></tr><tr><td rowspan=1 colspan=1>Iraq</td><td rowspan=1 colspan=2>29   1.27</td><td rowspan=1 colspan=2>Sudan                 10</td><td rowspan=1 colspan=1>0.44</td><td rowspan=1 colspan=2>North Korea                      1</td><td rowspan=1 colspan=1>0.04</td></tr><tr><td rowspan=1 colspan=1>Jordan</td><td rowspan=1 colspan=2>29   1.27</td><td rowspan=1 colspan=2>Suriname              10</td><td rowspan=1 colspan=1>0.44</td><td rowspan=1 colspan=2>Timor-Leste                       1</td><td rowspan=1 colspan=1>0.04</td></tr><tr><td rowspan=1 colspan=1>Bosnia and Herzegovina</td><td rowspan=1 colspan=2>29   1.27</td><td rowspan=1 colspan=2>Mauritania             9</td><td rowspan=1 colspan=1>0.39</td><td rowspan=1 colspan=2>Guernsey                         1</td><td rowspan=1 colspan=1>0.04</td></tr><tr><td rowspan=1 colspan=1>Taiwan</td><td rowspan=1 colspan=2>28   1.23</td><td rowspan=1 colspan=2>Bahamas              9</td><td rowspan=1 colspan=1>0.39</td><td rowspan=1 colspan=2>Madagascar                       1</td><td rowspan=1 colspan=1>0.04</td></tr><tr><td rowspan=1 colspan=1>Algeria</td><td rowspan=1 colspan=1>28</td><td rowspan=1 colspan=1>1.23</td><td rowspan=1 colspan=2>Nicaragua              8</td><td rowspan=1 colspan=1>0.35</td><td rowspan=1 colspan=2>Central African Republic           1</td><td rowspan=1 colspan=1>0.04</td></tr><tr><td rowspan=1 colspan=1>Switzerland</td><td rowspan=1 colspan=1>27</td><td rowspan=1 colspan=1>1.18</td><td rowspan=1 colspan=2>Senegal                8</td><td rowspan=1 colspan=1>0.35</td><td rowspan=1 colspan=2>Monaco                          1</td><td rowspan=1 colspan=1>0.04</td></tr><tr><td rowspan=1 colspan=1>Cyprus</td><td rowspan=1 colspan=1>26</td><td rowspan=1 colspan=1>1.14</td><td rowspan=1 colspan=2>Barbados              8</td><td rowspan=1 colspan=1>0.35</td><td rowspan=2 colspan=3></td></tr><tr><td rowspan=1 colspan=1>Morocco</td><td rowspan=1 colspan=2>25   1.10</td><td rowspan=1 colspan=3>Dominica              8   0.35</td></tr></table>

Table 10: Distribution of food entries by country.