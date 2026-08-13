# Smurfs: Multi-Agent System using Context-Efficient DFSDT for Tool Planning

Junzhi Chen\*, Juhao Liang\*, Benyou Wang<sup>B</sup>   
The Chinese University of Hong Kong, Shenzhen Shenzhen Research Institute of Big Data wangbenyou@cuhk.edu.cn

## Abstract

Teaching large language models (LLMs) to use tools for solving complex problems can grant them human-like reasoning abilities. ReAct and its variants are popular frameworks for tool use in both single-agent and multi-agent systems. To address issues like error propagation and limited exploration in ReAct, the Deep First Search Decision Tree (DFSDT) was proposed, but it faces challenges such as rollback instability, redundant context, and premature termination in single-agent settings. We introduce "Smurfs," a novel multi-agent system (MAS) that enhances DFSDT with a modular, context-efficient, and training-free design. Smurfs surpasses baseline methods in both the open-ended StableToolBench and the closed-ended HotpotQA tasks, reducing token usage by 60.9% compared to DFSDT and enabling Mistral-7b to perform on par with GPT-4- DFSDT. Extensive ablation studies confirm the effectiveness of Smurfs’ core components, offering valuable insights for the construction and interpretation of MAS, and paving the way for future exploration. We release the code at https://github.com/FreedomIntelligence/Smurfs.

## 1 Introduction

The ability to manipulate tools for complex tasks has long been considered a distinctive characteristic of humans (Oakley and Museum, 1972; Ambrose, 2001). Can we extend this ability to today’s large language models (LLMs), enabling them to utilize multiple tools to perform complex tasks beyond their inherent capabilities? If LLMs can use external tools to access knowledge or execute tasks beyond their fixed language modeling capabilities, we can shift the focus of LLM training towards enhancing their reasoning and tool-use skills. This

shift would allow tools to supplement what LLMs should know or execute, thereby improving the parameter efficiency of the LLMs.
<table><tr><td></td><td>Pass Rate ↑ (%)</td><td>Win Rate ↑ (%)</td><td># of Tokens per request ↓</td><td># of Tokens per query ↓</td></tr><tr><td>ReAct</td><td> $4 4 . 4 { \scriptstyle \pm 1 . 1 }$ </td><td>base</td><td>1,424</td><td>6,479</td></tr><tr><td>DFSDT</td><td> $5 5 . 4 \pm 2 . 0$ </td><td>60.4</td><td>1,743</td><td>20,714</td></tr><tr><td>Smurfs (ours)</td><td> $5 7 . 4 { \scriptstyle \pm 1 . 1 }$ </td><td>62.4</td><td>459</td><td>8,096</td></tr></table>

Table 1: Comparison of token cost and performance between tool planning methods over StableToolBench. Existing methods, ReAct and DFSDT, have limitations due to high token costs or poor performance. The results are averaged over the subtasks within StableToolBench.

In previous multi-agent systems for tool planning, methods like Chain-of-Thought (CoT) (Wei et al., 2023), ReAct (Yao et al., 2022), and the more advanced DFSDT (Qin et al., 2024) have been proposed to enhance LLMs’ ability to handle complex multi-step tasks. However, these approaches face notable limitations. ReAct has trouble in handling error propagation and limited exploration. DFSDT incorporates a rollback mechanism and depth-first search approach to address limitations of ReAct, but it suffers from instability when the base model struggles with long-context reasoning. It also introduces inefficiencies due to redundant context handling and risks premature termination when solving multi-step problems. These challenges highlight the need for further innovation to better manage context and reasoning complexity in multi-tool planning systems.

In this paper, we introduce ‘Smurfs’, an innovative multi-agent system (MAS) framework inspired by the collaborative and versatile nature of its namesake cartoon characters. The proposed framework leverages enhanced DFSDT to perform complex tool planning tasks without the need for additional training. The effectiveness of Smurfs is demonstrated through both open-ended and closedended tool planning benchmark experiments (Guo et al., 2024; Yang et al., 2018), where it consistently outperforms baseline methods. An ablation study, followed by a case study, further investigates the reasons behind this effectiveness. These results establish a new state-of-the-art in the field and provide concrete evidence of the advantages of a multi-agent approach in enhancing LLM capabilities.

![](images/a576951090ebd418833cb5b04d7c7902fcba173851b20448ce5ad17610a2798a.jpg)  
Figure 1: Demonstration of the whole process of the Smurfs framework.

The contributions of this paper can be summarized as follows:

1. We introduce a highly modular, contextefficient, and training-free MAS framework that utilizes an enhanced DFSDT to improve the tool planning capabilities of LLMs. Experiments demonstrate the effectiveness of this approach, which also proves to be more costefficient compared to existing methods.

2. Through ablation studies, we uncover the underlying factors contributing to the effectiveness of the MAS framework, offering valuable insights for future research.

## 2 Related Work

To augment LLMs to do multi-tool planning for solving complex problems, previous work has seen numerous attempts. Chain-of-Thought (Wei et al., 2023) is the first to propose the method of thought and answer chain reasoning. ReAct (Yao et al., 2022) further introduce the thought-action-observation format for tool chain reasoning, leading to the development of various multi-tool planning methods (Chen et al., 2023a; Xu et al., 2023; Shinn et al., 2023). The latest work,

Deep First Search Decision Tree (DFSDT) (Qin et al., 2024), is proposed to address the inherent limitations of CoT and ReACT: (1) Error Propagation: Error occurs at early stage of planning will result in wrong answer in the end, but it can only be identified until reaching the end of the planning chain (2) Limited Exploration: Single solution chain can’t explore the planning space completely.

DFSDT is powerful in addressing multi-tool planning problems. Its core concept involves employing a depth-first search (DFS) approach for multi-tool planning and backtracks whenever an LLM think the solving process has entered a wrong state (for more details, see Appendix B). When a tool fails or is deemed inadequate for solving the current problem, DFSDT backtracks to the previous solution state and attempts to resolve the issue using another solution plan. However, three limitations are identified with the mechanism of DFSDT:

Limit I: Instability of the Rollback Mechanism The rollback mechanism in DFSDT is determined by the model. The number of steps to roll back and the selection of new tools after rollback are guided using prompt containing the errors encountered in the previous failed trajectory. When the model is sufficiently robust, this rollback mechanism serves as a highly flexible and efficient planning strategy. However, when the model’s capability is insufficient, it will fail to execute the correct rollback mechanism, i.e. retry the same error tools or roll

Limit II: Redundant Context In the process of planning with DFSDT, each tool plan is generated using the entire conversation history (including all the thoughts, actions, action inputs and tool responses) as context. However, in reality, each step of tool planning only requires a very small portion of the relevant history for effective planning.

The context redundancy not only increases computational overhead but also reduces the accuracy of model inference due to the inclusion of irrelevant historical data. As highlighted by (Liu et al., 2024), redundant context become particularly noticeable in tasks requiring assimilation and processing of large inputs, like verbose tool documents and API responses. The situation worsens when LLMs are supplemented with external information, such as document retrieval or online searching (Petroni et al., 2020; Ram et al., 2023; Mallen et al., 2022). Although numerous language models capable of handling larger contexts are emerging (Dai et al., 2019; Dao et al., 2022), they often face significant performance degradation when the important information is located at some positions (Liu et al., 2024; Shi et al., 2023), which is known as the ‘lostin-the-middle’ problem.

Limit III: Premature Termination The termination mechanism set by DFSDT involves adding a termination tool to the model’s selectable toolkit. When the model selects this termination tool, DFSDT stops and provides an answer. However, in practical applications, this mechanism often prematurely terminates when dealing with complex problems requiring multi-step reasoning. We hypothesize that this issue arises due to the redundant interference of other tool information and history information, which disrupts the model’s ability to judge whether the original problem should be terminated. Instead, the model focuses on whether the current sub-problem requires termination, leading the mechanism to terminate after resolving the sub-problem.

In conclusion, DFSDT relies highly on the base model’s reasoning ability, especially long context reasoning ability to make roll back decision, termination decision and tool choice decision at the same time, which is a very difficult task even with the most powerful LLM like GPT4 (Yuan et al., 2024).

Multi-Agent System To address the limitations inherent in DFSDT and to further enhance LLM’s multi-tool planning capabilities, MAS has emerged as a natural solution. Inspired by human social division of labor and cooperation, MAS aim to enable AI agents to accomplish more complex tasks through the division of labor and collaboration. By decomposing the task of DFSDT to multiple agents and giving them only the information they need, we can enable LLMs to use DFSDT more effectively and more efficiently.

<table><tr><td>Method</td><td>Multi-Agent</td><td>Training</td><td>Generality</td><td>Reflection</td><td>Planning</td></tr><tr><td>REACT (Yao et al., 2022)</td><td></td><td></td><td></td><td></td><td>Iterative</td></tr><tr><td>Reflexion (Shinn et al., 2023)</td><td>X</td><td></td><td></td><td></td><td>Iterative</td></tr><tr><td>Chameleon (Lu et al., 2023)</td><td></td><td></td><td></td><td></td><td>Global</td></tr><tr><td>HuggingGPT (Shen et al., 2023)</td><td></td><td></td><td></td><td>X</td><td>Global</td></tr><tr><td>BOLAA (Liu et al., 2023)</td><td></td><td></td><td></td><td>X</td><td>Iterative</td></tr><tr><td>AgentVerse (Chen et al., 2023b)</td><td></td><td></td><td></td><td></td><td>Iterative</td></tr><tr><td>FIREACT (Chen et al., 2023a)</td><td></td><td></td><td></td><td></td><td>Iterative</td></tr><tr><td>DFSDT (Qin et al., 2024)</td><td></td><td></td><td></td><td></td><td>Iterative</td></tr><tr><td>RESTGPT (Song et al., 2023)</td><td></td><td></td><td></td><td></td><td>Iterative</td></tr><tr><td>Lumos (Yin et al., 2024)</td><td></td><td></td><td></td><td></td><td>Iterative or Global</td></tr><tr><td>AutoAct (Qiao et al., 2024)</td><td></td><td></td><td></td><td></td><td>Iterative</td></tr><tr><td>Smurfs (Ours)</td><td></td><td></td><td></td><td></td><td>Iterative and Global</td></tr></table>

Table 2: Comparison of tool use systems.

Previous work on multi-agent system mainly focus on coding and society simulation area (Hong et al., 2023; Qian et al., 2024; Park et al., 2023; Li et al., 2023; Wu et al., 2023). For tool-use scenario, most multi-agent systems still use the ReAct style reasoning (Qiao et al., 2024; Chen et al., 2023b; Yuan et al., 2024; Liu et al., 2023; Song et al., 2023; Yin et al., 2024; Xu et al., 2023), only using multi-agent discussion and revision to increase reasoning quality, which still inherits the limitation of ReAct. Therefore, this paper aims to construct a novel MAS framework to address the aforementioned limitations. Table 2 shows the difference between different tool-use systems. More detailed discussion can be seen in Appendix A.

Token Compression Token compression refers to compressing tokens fed into LLMs while preserving inference performance, thus reducing computational overhead and mitigating the constraints imposed by long context limits. Previous works (Mu et al., 2024; Fu et al., 2024) have explored token compression techniques, with a focus on compressing token embeddings. In contrast, Smurfs achieves context compression by filtering the input context for each tool planning process. Table 6 shows the detailed difference between different compression techniques.

In the future, these compression methods can be used together in multi-tool planning scenarios to achieve more efficient token compression. Each agent’s system prompt could be compressed using gist tokens (Mu et al., 2024), tool descriptions could be managed with CAMPHOR’s compression approach (Fu et al., 2024), and the context input by each agent could be compressed using Smurfs.

## 3 Smurfs: MAS with Context Efficient DFSDT

The Smurfs, the beloved cartoon characters, symbolize unity and resourcefulness, and are good at using tools to overcome any challenge they encounter.

## 3.1 Framework Overview

Figure 1 illustrates the entire workflow for the Smurfs framework. Initially, the Planning Agent identifies the user’s complex query and breaks it down into manageable sub-tasks. Executor Agents are then tasked with collecting task specific information, utilizing access to external tools. Answer Agent compiles the findings into a cohesive response, which is subsequently verified by the Verifier Agent to ensure accuracy and relevance.

By dividing tasks among different agents, each agent can focus on a specific part of the task, accessing only the necessary history as context during task execution, which effectively addresses the issue of redundant context. The redesign of the rollback mechanism to incorporate memory and tool list rollback mechanisms addresses the instability of the rollback mechanism. Drawing on the concept of least-to-most prompting (Zhou et al., 2023), the original problem is first decomposed into sub-problems for macro-level planning. Subsequently, Smurfs is used to solve each sub-problem at the micro-level, with macro-level planning guiding the micro-level planning, thereby resolving the issue of premature termination.

In the rest of this section, the mechanism of the system and the functions of each agent will be detailed. More details of memory system can be seen at Appendix C.

## 3.2 Agent Components

In this section, we introduce the two core components of the Smurfs system:

Tools The tool documents about the tools that Smurfs can utilize in the completion of a complex task are denoted as $D = \{ n _ { i } , d _ { i } , p _ { i } \} _ { i = 1 } ^ { | d | }$ , where n represents the tool name, d represents tool usage description, p represents parameter description and d represents the amount of the available tools. The available tool list is denoted as $\tau = \{ n _ { i } , d _ { i } \} _ { i = 1 } ^ { | \tau | } .$ . τ<sub>t</sub> denotes the tool list Smurfs can utilize at time t.

Memory The memory of the agent system at time t is the history of the task-solving process before t, denoted as $M = ( m _ { 1 } , m _ { 2 } , . . . , m _ { t - 1 } )$ and $m _ { i } = ( \gamma _ { i } , a _ { i } )$ , where $m _ { i }$ represents memory element at time i and $\gamma _ { i } , a _ { i }$ represents thought and answer generated by the system at time i. There are two types of memory in Smurfs: local memory and global memory. the local memory is used to record the ongoing solution trajectory and to generate the next action in the current trajectory. The global memory, meanwhile, records all trajectories and is used to generate the sub-problem’s answer by combining all trajectory records when the maximum number of retries is exceeded. This local-global combined memory system ensures that the planning of the current solution trajectory is not influenced by the context of erroneous trajectories. It also generates an answer that combines all trajectories when the verifier agent cannot determine task completion within the maximum number of planned steps. This memory system ensures context efficiency during the task-solving process.

## 3.3 Macro Planning

Planning Agent The primary responsibility of the Planning Agent is doing macro-level planning through task decomposition to prevent premature termination. The inference process of the Planning Agent is:

$$
P l a n P : ( p 1 , p 2 , \ldots ) = P A ( q )\tag{1}
$$

Where $p _ { i }$ represents sub-problem of the original query q, PA represents the Planning Agent. After the task decomposition, the agent system will use Executor Agent, Answer Agent an Verifier Agent to solve each sub-problem using DFSDT collaboratively in a sequential order. To utilize the answer of the previous sub-problem when solving subsequent sub-problem, the strategy known as least-to-most prompting (Zhou et al., 2023) is used.

## 3.4 Subtask Solving Process

After introducing the function of plan agent, this section outlines how the agents collaborate to solve sub-tasks, as shown in Figure 2.

Stable Rollback To address the instability of the rollback mechanism in DFSDT, we propose a rollback mechanism based on rules. Whenever an error occurs while using a tool $\tau _ { t , i }$ at time t, the tool list at $\textnormal { t } \tau _ { t }$ will pop $\tau _ { t , i }$ out and reperform tool selection and tool planning (ensuring that the faulty tool is not selected again). If, at time t, the tool list becomes empty, it signifies that after the system choosing tool $\tau _ { t - 1 , j }$ at time t-1, no subsequent trajectory can solve the problem. In this case, the agent system will roll back to time t-1, meaning that the local memory M will pop out the memory element $m _ { t - 1 }$ at time t-1, and the tool list at time $\mathbf { t } { - } 1 ~ \tau _ { t - 1 }$ will pop out tool $\tau _ { t - 1 , j }$ . The agent system will then set the time t=t-1 and continue planning. This rule-based rollback mechanism, compared to the original model-based rollback mechanism of DFSDT, is less flexible and might reduce rollback efficiency. However, it is more stable, ensuring the correctness of deep first search and enabling models with weaker capabilities to utilize DFSDT on tool planning.

![](images/7c1d06668a83552d66e16acac630e573baa345499203416641369b98ce3073cc.jpg)  
Figure 2: Details of the subtask-solving process of the Smurfs framework. The dotted line represents that the agent can see the memory and the full line stands for operation.

Executor Agent The Executor Agent is responsible for choosing and executing the tools to solve the sub-tasks. At each time t, the agent can invoke one tool to tackle the given task:

$$
\gamma = E A . g e n \_ t h o u g h t ( p , M , \tau , h )\tag{2}
$$

$$
\alpha = E A . c h o o s e \_ t o o l ( p , \gamma , \tau )\tag{3}
$$

$$
\beta = E A . g e n \_ a r g u m e n t s ( p , M , D [ \alpha ] )\tag{4}
$$

$$
r = E A . c a l l \_ t o o l ( \alpha , \beta )\tag{5}
$$

Where p is the sub-problem from Planning Agent, h is the hint from the Verifier Agent, τ is the tool list, M is local memory, $D [ \alpha ]$ means the tool document of tool $\alpha .$ The agent, using the ReACT format (Yao et al., 2022) to choose the tool and arguments, then execute the tool. Noticed that each inference process only uses the relevant part from the local memory and tool list to reduce the context redundancy. More detailed information of the Executor Agent can be found in Figure 5.

Answer Agent To mitigate the performance degradation caused by lengthy contexts, we introduce the Answer Agent role, designed to extract crucial content for each step and sub-problem:

$$
A n s w e r : a = A A ( q , r , M )\tag{6}
$$

Where q is sub-problem from the Planning Agent, r is response from the Executor Agent, M is the local memory (or global memory if max retry reaches). As the ‘lost-in-the-middle’ theory described in section 1, retaining all information may not always be beneficial, particularly in cases where the solution path is challenging to discern. Therefore, the primary role of the Answer Agent is to succinctly summarize the generated answers and tool responses to maintain the memory efficiency.

Verifier Agent The Verifier Agent serves as an early-stopping and reflection mechanism, allowing for a balance between effectiveness and efficiency:

$$
h , c = V A ( q , a )\tag{7}
$$

Where $\mathbf { q }$ denotes the sub-problems from the Planning Agent, a denotes the answer from the answer agent, h and c denotes hint and check status respectively. If check status generated is 0, that means the Verifier Agent thinks the sub-problem isn’t completed, the system will add the thought and answer of this time to the local and global memory, set t=t+1 and continue the inference procedure.If check status is 1, the sub-problems is thought to be solved and the system will start to deal with the next sub-problem.

<table><tr><td rowspan="2">Backbone</td><td rowspan="2">Method</td><td colspan="10">StableToolBench</td><td rowspan="2"></td><td colspan="3"></td></tr><tr><td>I1-Inst.</td><td></td><td>I1-Cat.</td><td></td><td>I1-Tool. Pass</td><td>Win</td><td>I2-Cat.</td><td>Win</td><td>I2-Inst.</td><td></td><td>I3-Inst.</td><td></td><td>Average</td><td></td></tr><tr><td></td><td></td><td>Pass</td><td>Win</td><td>Pass</td><td>Win</td><td></td><td></td><td>Pass</td><td></td><td></td><td>Pass</td><td>Win</td><td>Pass</td><td>Win</td><td>Pass</td><td>Win</td></tr><tr><td>GPT-3.5 Turbo</td><td>ReACT</td><td> $4 1 . 6 { \pm } 1 . 2$ </td><td>1</td><td> $4 8 . 4 { \scriptstyle \pm 0 . 5 }$ </td><td>1</td><td> $5 2 . 5 { \scriptstyle \pm 0 . 5 }$ </td><td>/</td><td> $5 2 . 2 { \scriptstyle \pm 1 . 0 }$ </td><td>1</td><td> $3 1 . 6 { \pm } 1 . 2 $ </td><td></td><td>1</td><td> $3 9 . 9 { \scriptstyle \pm 2 . 0 }$ </td><td>/</td><td>44.4±1.1</td><td></td></tr><tr><td>GPT-3.5 Turbo</td><td>DFSDT</td><td> $5 4 . 1 _ { \pm 1 . 0 }$ </td><td>64.4</td><td> $6 0 . 1 { \scriptstyle \pm 0 . 0 }$ </td><td>61.4</td><td> $5 9 . 9 { \scriptstyle \pm 1 . 7 }$ </td><td>53.8</td><td> $6 0 . 9 { \scriptstyle \pm 0 . 9 }$ </td><td>62.9</td><td></td><td> $5 2 . 8 { \scriptstyle \pm 3 . 7 }$ </td><td>66.0</td><td> $4 4 . 3 { \scriptstyle \pm 4 . 8 }$ </td><td>54.1</td><td> $5 5 . 4 \pm 2 . 0$ </td><td>60.4</td></tr><tr><td>GPT-3.5 Turbo</td><td>Smurfs</td><td> $6 0 . 3 _ { \pm 1 . 5 } ^ { - }$ </td><td>65.0</td><td> $6 7 . 0 { \scriptstyle \pm 1 . 0 }$ </td><td>69.9</td><td> $6 0 . 3 { \scriptstyle \pm 1 . 3 }$ </td><td>54.4</td><td> $5 4 . 3 _ { \pm 0 . 4 } ^ { - }$ </td><td>63.7</td><td> $4 2 . 6 { \scriptstyle \pm 1 . 6 }$ </td><td></td><td>64.2</td><td> $6 0 . 1 { \pm } 1 . 0 $ </td><td>57.4</td><td>57.4±1.1</td><td>62.4</td></tr><tr><td>Mistral-7B</td><td>ReACT</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td></td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Mistral-7B</td><td>DFSDT</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td></td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Mistral-7B</td><td>Smurfs</td><td> ${ \bf 7 6 . 3 _ { \pm 0 . 8 } }$ </td><td>63.8</td><td> $\mathbf { 8 6 . 7 \pm 1 . 2 }$ </td><td>62.7</td><td> ${ \bf 8 1 . 0 _ { \pm 1 . 9 } }$ </td><td>58.2 48.1</td><td> ${ \bf 7 0 . 4 _ { \pm 2 . 7 } }$ </td><td>54.0</td><td> ${ \bf 6 3 . 8 { \scriptstyle \pm 2 . 4 } }$ </td><td></td><td>67.0</td><td> ${ \bf 8 5 . 2 _ { \pm 0 . 7 } }$ </td><td>57.4</td><td> ${ \bf 7 7 . 2 _ { \pm 1 . 6 } }$ </td><td>60.5</td></tr><tr><td>GPT-4 Turbo</td><td>ReACT DFSDT</td><td> $4 1 . 1 { \pm } 1 . 5$ </td><td>60.1 69.9</td><td> $5 3 . 2 { \scriptstyle \pm 1 . 3 }$ </td><td>62.1 66.0</td><td> $4 2 . 2 { \scriptstyle \pm 1 . 1 }$   $5 9 . 7 { \scriptstyle \pm 1 . 2 }$ </td><td>58.2</td><td> $5 0 . 0 { \scriptstyle \pm 0 . 7 }$   $5 9 . 3 { \scriptstyle \pm 0 . 7 }$ </td><td>57.3 62.1</td><td> $3 8 . 7 { \scriptstyle \pm 0 . 8 }$   $5 2 . 2 { \scriptstyle \pm 2 . 3 }$ </td><td></td><td>65.1 67.9</td><td> $3 7 . 7 _ { \pm 1 . 3 }$ </td><td>47.5</td><td> $4 3 . 8 { \scriptstyle \pm 1 . 1 }$ </td><td>56.7</td></tr><tr><td>GPT-4 Turbo GPT-4 Turbo</td><td>Smurfs</td><td> $5 2 . 7 _ { \pm 1 . 4 }$ </td><td>71.2</td><td> $5 8 . 2 { \scriptstyle \pm 0 . 9 }$ </td><td>72.5</td><td> $6 7 . 4 { \scriptstyle \pm 0 . 7 }$ </td><td>69.6</td><td> $6 6 . 7 { \scriptstyle \pm 1 . 9 }$ </td><td>73.4</td><td> $5 5 . 5 { \scriptstyle \pm 1 . 4 }$ </td><td></td><td>66.0</td><td> $6 1 . 5 { \scriptstyle \pm 1 . 8 }$   $7 0 . 5 { \scriptstyle \pm 0 . 0 }$ </td><td>65.6 72.1</td><td> $5 7 . 3 { \scriptstyle \pm 1 . 4 }$ </td><td>65.0 70.8</td></tr><tr><td></td><td></td><td> $5 9 . 3 { \scriptstyle \pm 1 . 4 }$ </td><td></td><td> $7 3 . 3 { \scriptstyle \pm 1 . 3 }$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td> $6 5 . 5 { \pm } 1 . 1 $ </td><td></td></tr></table>

Table 3: The open-end tool planning task evaluation on the StableToolBench benchmark (Guo et al., 2024). The most effective approach is highlighted in bold, while the second-best is underlined. Win rate is calculated by comparing each model with ChatGPT-ReACT. A win rate higher than 50% means the model performs better than ChatGPT-ReACT.

## 4 Experiments

To evaluate both the effectiveness and efficiency of the Smurfs framework, in thie section, we carried out two multi-tool planning tasks: (1) an openended task, StableToolBench (Guo et al., 2024), and (2) a closed-ended task, HotpotQA (Yang et al., 2018). In addition to these main experiments designed to assess the entire framework, we conducted an ablation studies followed by a case study to test the capabilities of each component within the multi-agent framework and investigate the underlying reasons for its effectiveness.

## 4.1 Open-ended Task: StableToolBench

StableToolBench is a tool learning benchmark derived from ToolBench (Qin et al., 2024), encompassing multi-step tool usage tasks across over 16,000 APIs. The benchmark employs two metrics for evaluation: (1) Pass Rate measures the percentage of instructions successfully executed within the allocated budget. (2) Win Rate represents the preference selection by a ChatGPT evaluator when presented with two solution paths.

Baselines Following the original paper that introduced the benchmark, we adopt ReACT (CoT) (Wei et al., 2023) and DFSDT (Touvron et al., 2023) as baseline methods for comparison. Additionally, we include the backbones used in the paper: gpt-3.5- turbo-0613 (GPT-3.5 Turbo) (OpenAI) and gpt-4- turbo-preview (GPT-4 Turbo). To explore the adaptability of the tool-planning methods, we also include Mistral-7B-Instruct-v0.2 (Mistral-7B) (Jiang et al., 2023) as one of the selected backbones in our experiments.

Settings To minimize the influence of varying tool APIs on experimental results, we conducted all experiments using the same API cache (Guo et al., 2024). For a fair comparison among the candidate methods and to reduce variability, each model was executed once and evaluated three times, with results averaged. Other settings follow those specified in the original benchmark paper.

Results Table 3 displays the results on Stable-ToolBench. For the untrained LLM, Mistral-7B, existing agent frameworks did not improve its performance in tool planning tasks; Mistral-7B failed these tasks when integrated with the ReACT and DFSDT frameworks \*. However, Smurfs exhibited exceptional performance: when combined with Mistral-7B, Smurfs achieved competitive scores among the baselines. Through its task decomposition mechanism, Smurfs transforms long-context tasks into simpler ones, enabling the untrained model to effectively utilize external tools for managing complex tasks. Regarding closed-source models, specifically GPT4 in these experiments, Smurfs also demonstrated outstanding performance on the benchmark compared to other agent frameworks and achieved state-of-the-art results on the benchmark. Its high success rate suggests that Smurfs is more effective at finding optimal solution paths compared to ChatGPT.

Further Analysis We conducted a detailed analysis of the token costs associated with each tool planning method for the tasks, a critical evaluation aspect for multihop reasoning tasks. As shown in Table 1 (detailed in Appendix G), the average token costs per question and API request are evaluated for ReACT, DFSDT, and Smurfs on StableTool-Bench. The analysis reveals that DFSDT generally requires about 20,000 tokens per question, encompassing both prompt and completion tokens. This is nearly three times the token cost compared to ReACT and twice as much as Smurfs. Despite this higher token cost, DFSDT does not demonstrate commensurate effectiveness improvements over other methods. These findings underscore the cost-efficiency of the proposed MAS framework, Smurfs, which not only reduces token expenditure in solving multihop planning tasks but also delivers outstanding performance in evaluations.

<table><tr><td rowspan="2">Backbone</td><td rowspan="2">Method Multi-Agent</td><td rowspan="2">Single-Agent</td><td colspan="3">HotpotQA</td></tr><tr><td>Easy Medium</td><td>Hard</td><td>All</td></tr><tr><td rowspan="2">GPT-3.5 Turbo</td><td>e CoT</td><td>48.21</td><td>44.52</td><td>34.22</td><td>42.32</td></tr><tr><td>e Zero-Shot Plan</td><td>50.71</td><td>45.17</td><td>38.23</td><td>44.70</td></tr><tr><td rowspan="9">Mistral-7B Instruct-v0.2</td><td>e CoT</td><td></td><td>33.70</td><td>22.38</td><td>22.14 26.07</td></tr><tr><td>e ReAct</td><td>38.09</td><td>27.57</td><td>22.05</td><td>29.24</td></tr><tr><td>e Chameleon</td><td>37.07</td><td>26.67</td><td>19.20</td><td>27.65</td></tr><tr><td>e Reflexion</td><td>40.78</td><td>35.02</td><td>28.36</td><td>34.72</td></tr><tr><td>②BOLAA</td><td>40.86</td><td>32.11</td><td>22.36</td><td>31.78</td></tr><tr><td>②ReWOO</td><td>38.42</td><td>31.89</td><td>25.98</td><td>32.10</td></tr><tr><td>② Smurfs (ours)</td><td>45.94</td><td>40.74</td><td>30.72</td><td>39.13</td></tr><tr><td>FireAct o</td><td>45.52</td><td>32.02</td><td>30.17</td><td>35.90</td></tr><tr><td>  AUTOACT</td><td>48.69</td><td>36.65</td><td>31.37</td><td>38.89</td></tr><tr><td rowspan="10">Llama-2 13B-chat</td><td>e CoT</td><td>37.90</td><td>25.28</td><td>21.64</td><td>28.27</td></tr><tr><td>e ReAct</td><td>28.68</td><td>22.15</td><td>21.69</td><td>24.17</td></tr><tr><td>e Chameleon</td><td>40.01</td><td>25.39</td><td>22.82</td><td>29.41</td></tr><tr><td>e Reflexion</td><td>44.43</td><td>37.50</td><td>28.17</td><td>36.70</td></tr><tr><td>②BOLAA</td><td>33.23</td><td>25.46</td><td>25.23</td><td>27.97</td></tr><tr><td>②ReWOO</td><td>30.09</td><td>24.01</td><td>21.13</td><td>25.08</td></tr><tr><td>  Smurfs (ours)</td><td>42.62</td><td>27.21</td><td>22.92</td><td>30.92</td></tr><tr><td>o FireAct</td><td>45.83</td><td>38.94</td><td>26.06</td><td>36.94</td></tr><tr><td>  AUTOACT</td><td>47.29</td><td>41.27</td><td>32.92</td><td>40.49</td></tr><tr><td>e CoT</td><td>45.37</td><td>36.33</td><td></td><td></td></tr><tr><td rowspan="8">Llama-2 70B-chat</td><td>e ReAct</td><td>39.70</td><td></td><td>32.27</td><td>37.99</td></tr><tr><td>Chameleon</td><td></td><td>37.19</td><td>33.62</td><td>36.83</td></tr><tr><td>e e Reflexion</td><td>46.86</td><td>38.79</td><td>34.43</td><td>40.03</td></tr><tr><td>②BOLAA</td><td>48.01</td><td>46.35</td><td>35.64</td><td>43.33</td></tr><tr><td></td><td>46.44</td><td>37.29</td><td>33.49</td><td>39.07</td></tr><tr><td>②ReWOO</td><td>42.00</td><td>39.58</td><td>35.32</td><td>38.96</td></tr><tr><td> Smurfs (ours)</td><td>52.86</td><td>50.77</td><td>44.87</td><td>49.50</td></tr><tr><td>0 FireAct</td><td>50.82</td><td>41.43</td><td>35.86</td><td>42.70</td></tr><tr><td></td><td> AUTOACT</td><td>56.94</td><td>50.12</td><td>38.35</td><td>48.47</td></tr></table>

Table 4: The closed-end tool planning evaluation on HotpotQA (Yang et al., 2018), with some results derived from (Qiao et al., 2024). The most effective approach for each group is highlighted in bold, while the secondbest is underlined. Methods marked with v require model training.

## 4.2 Closed-ended Task: HotpotQA

Compared to open-ended tasks, closed-ended tasks provide a more stable and robust evaluation. To this end, we evaluate the methods on HotpotQA (Yang et al., 2018) in addition to StableToolBench. HotpotQA is a multi-hop QA task that is challenging due to the requirement for rich background knowledge, with answers typically being short entities or yes/no responses.

Baselines The compared baselines include CoT (Wei et al., 2023), REACT(Yao et al., 2022), Chameleon(Lu et al., 2023), Reflexion (Shinn et al., 2023), BOLAA (Liu et al., 2023), ReWOO (Xu et al., 2023), FIREACT (Chen et al., 2023a), AutoAct(Qiao et al., 2024).

Settings and Metrics Following the settings in (Qiao et al., 2024), we use open-source Llama-2 models (Touvron et al., 2023) and Mistral-7B (Jiang et al., 2023) as the backbones of each agent to evaluate the performance of Smurfs. The evaluation metrics is reward  [0, 1], defined as the F1 score grading between the prediction and ground-truth answer. For more details about the experiment, see Appendix D.

Results Smurfs, as an untrained MAS system, not only comprehensively outperforms untrained agents but also achieves and even surpasses the accuracy of trained agents across most backbone models. This sufficiently demonstrates that the mechanism of smurfs ensures strong generalization capabilities while maintaining high effectiveness.

Observations indicate that the performance of LLama-2-13b-chat on smurfs-related tasks is suboptimal, likely due to its limited capabilities in tool arguments generation. Specifically, the primary issue identified is that, when the Executor agent successfully selects relevant tool, it tends to produce hallucination arguments that can’t be used by the tools. This indicates that LLama-2-13bchat may need further training for usage of tools. The experiment results may substantiate this viewpoint, demonstrating that the untrained methods of llama-2-13b-chat generally exhibit significantly lower accuracy compared to the trained methods. To provide further insight into the model’s performance in the tool planning process, we manually categorized the types of errors made by Smurfs on the hotpotQA hard dataset in appendix E.

## 5 Ablation Study

## 5.1 Importance of each component in MAS

We performed an ablation study to investigate the impact of each agent in our framework. We removed each agent individually, except for the indispensable Executor Agent, and compared the results to the complete framework.

![](images/7a4e3f9698e470498ac0c432649d021772a035d533cb08582deb78bc26704f4e.jpg)  
Figure 3: The illustration of how GPT4-Smurfs and GPT4-DFSDT solve long context problem. The two subquestions and their corresponding answers are marked in two colors.

<table><tr><td rowspan="2"></td><td colspan="2">I3-Inst.</td></tr><tr><td>Pass (%)</td><td>Win (%)</td></tr><tr><td>GPT-3.5 Turbo with Smurfs</td><td>60.1±1.0</td><td>57.4</td></tr><tr><td>w/o Answer Agent</td><td>57.4±2.9</td><td>49.2</td></tr><tr><td>w/o Verifier Agent</td><td>54.1±2.7</td><td>42.6</td></tr><tr><td>w/o Planning Agent</td><td>35.5±3.3</td><td>42.6</td></tr><tr><td>w/o Planning &amp; Verifier Agent</td><td>58.5±2.0</td><td>57.4</td></tr><tr><td>GPT-4 Turbo with Smurfs</td><td>70.5±1.0</td><td>72.1</td></tr><tr><td>w/o Answer Agent</td><td>82.2±2.5</td><td>72.1</td></tr><tr><td>w/o Verifier Agent</td><td>79.2±0.8</td><td>63.9</td></tr><tr><td>w/o Planning Agent</td><td>71.9±2.8</td><td>63.9</td></tr><tr><td>w/o Planning &amp; Verifier Agent</td><td>79.8±2.8</td><td>67.2</td></tr></table>

Table 5: Ablation study on StableToolBench I3-Inst subset to investigate the importance of each component within the framework.

Settings (1) Verifier Agent Removal: Without Verifier Agent, the framework uses a generatl DFSDT in each subtask, i.e. including a finish tool into the tool library and deciding whether to stop at tool choice step. (2) Answer Agent Removal: Without Answer Agent, the framework uses full tool response instead of the summary of tool response in its memory. (3) Planning Agent Removal: Without Planning Agent, the framework uses Verifier Agent to decide whether the task is completed. (4) Planning & Verifier Agent Removal: Without Planning and Verifier Agent, the only difference between Smurfs and DFSDT will be the Answer Agent, i.e. including a finish tool in the tool library, deciding whether to stop at tool choice step and using summary of tool response in its memory.

Results Table 5 shows the experiment result, highlighting several key insights regarding the impact of different components in the multi-agent system.

First, the removal of any component generally reduces the win rate, highlighting each component’s significance. Second, performance degradation trends remain consistent: Eliminating the Answer Agent has minimal impact, whereas removing the Planning Agent causes the greatest decline. However, this does not imply the relative importance of these agents but rather suggests that the Verifier-Planning Agent workflow is more robust than the Verifier-Answer Agent workflow.

Third, the performance impact varies by model capability. For GPT-3.5, the removal of the Answer Agent reduces both the pass and the win rates, while for GPT-4, it preserves the win rate and improves the pass rate, probably due to the superior long-text reasoning ability of GPT-4, which processes additional tool response more effectively. Removal of the Planning Agent significantly reduces the pass rate of GPT-3.5 but has a minimal effect on GPT-4, suggesting that the GPT-4 Verifier Agent is more robust. These findings may indicate that more powerful models can compensate for agent removal, sometimes even benefiting from additional context, while weaker models benefit more from complex workflow and compressed context.

We also noticed an interesting phenomenon: when only the Planning Agent is removed, the system experiences a major decline in performance, while removing the Verifier Agent in addition to the Planning Agent improves performance again. This may indicate that the Verifier Agent’s ability to pause tasks or provide next-step guidance is suboptimal and requires further training.

In conclusion, our findings offer some insight into the relationship between model capabilities and multi-agent system performance. The optimal number of agents and workflow may vary depending on the capacity of the model. We propose the hypothesis that weaker models benefit more from complex multi-agent systems and context segmentation, while stronger models perform better with comprehensive context and simpler agent systems. To validate this hypothesis, more extensive ablation studies are needed under a wider range of models and constraints to explore the influence of different context structures and reasoning workflows. We leave this investigation for future work.

## 5.2 Case Study

As shown in Figure 3, although GPT4-DFSDT and GPT4-Smurfs use the same tool calls to solve the problem, GPT4-DFSDT only answers the first subquestion correctly while GPT4-Smurfs answers both sub-questions accurately. In the process of addressing the second sub-question, it is notable that the tool response only mentions titles of film and television products related to "Star Wars", without addressing OTT platforms. GPT-4-DFSDT erroneously interprets these titles as responses to the question, while GPT-4-Smurfs adeptly identifies this discrepancy and provides a more appropriate response. This case highlights that in situations where tool responses are lengthy and questions are complex, the single agent framework like DFSDT may be susceptible to distractions from irrelevant information, leading to erroneous answers. Conversely, the context-efficient Smurfs framework demonstrates a reduced susceptibility to irrelevant information, thereby generating more accurate answers.

## 6 Conclusion

In this paper, we propose Smurfs, an innovative MAS framework designed to enhance the toolplanning capabilities of LLMs without requiring additional training. Through extensive experiments on both open-ended and closed-ended tool planning benchmarks, Smurfs demonstrate its effectiveness by consistently outperforming baseline methods. Ablation study further provides deeper insights into the impact of each agent in our framework. Based on the ablation study, we propose a hypothesis that can be further tested through more comprehensive experiments in future research: weaker models benefit more from complex multi-agent systems and context segmentation, while stronger models perform better with comprehensive context and simpler agent systems. The findings not only advance the state-of-the-art in multi-tool planning systems but also highlight the potential of modular, training-free frameworks for LLMs in various practical applications.

Looking forward, future research could focus on exploring Smurfs’ use in new domains, such as facilitating the synthesis of high-quality multitool planning data and enhancing the base model’s reasoning and tool-use abilities, further advancing the field of adaptive AI systems.

## Limitations

Model Size Constraints: Due to computational constraints, our experiments did not include larger and more diverse types of LLMs. We believe this would not affect the main observations of this paper.

Agent Component Scale-Up: Although we selected the most common and intuitive agent roles for the proposed MAS, there are many possibilities for researchers to explore. Investigating more well-designed agent roles may help improve the effectiveness of the agent system, and developing automated methods to identify these roles could facilitate effective scaling.

Acknowledging these limitations, future research should aim to address these gaps to provide a more comprehensive understanding of the Smurfs framework’s capabilities and potential areas for improvement.

## Acknowledgements

This work was supported by the Shenzhen Science and Technology Program (JCYJ20220818103001002), Shenzhen Doctoral Startup Funding (RCBS20221008093330065), Tianyuan Fund for Mathematics of National Natural Science Foundation of China (NSFC) (12326608), Shenzhen Key Laboratory of Cross-Modal Cognitive Computing (grant number ZDSYS20230626091302006), and Shenzhen Stability Science Program 2023.

## References

Stanley H Ambrose. 2001. Paleolithic technology and human evolution. Science, 291(5509):1748–1753.

Baian Chen, Chang Shu, Ehsan Shareghi, Nigel Collier, Karthik Narasimhan, and Shunyu Yao. 2023a. Fireact: Toward language agent fine-tuning. Preprint, arXiv:2310.05915.

Weize Chen, Yusheng Su, Jingwei Zuo, Cheng Yang, Chenfei Yuan, Chi-Min Chan, Heyang Yu, Yaxi Lu, Yi-Hsin Hung, Chen Qian, Yujia Qin, Xin Cong, Ruobing Xie, Zhiyuan Liu, Maosong Sun, and Jie Zhou. 2023b. Agentverse: Facilitating multiagent collaboration and exploring emergent behaviors. Preprint, arXiv:2308.10848.

Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V Le, and Ruslan Salakhutdinov. 2019. Transformer-xl: Attentive language models beyond a fixed-length context. arXiv preprint arXiv:1901.02860.

Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022. Flashattention: Fast and memory-efficient exact attention with io-awareness. Advances in Neural Information Processing Systems, 35:16344–16359.

Yicheng Fu, Raviteja Anantha, and Jianpeng Cheng. 2024. Camphor: Collaborative agents for multiinput planning and high-order reasoning on device. Preprint, arXiv:2410.09407.

Zhicheng Guo, Sijie Cheng, Hao Wang, Shihao Liang, Yujia Qin, Peng Li, Zhiyuan Liu, Maosong Sun, and Yang Liu. 2024. Stabletoolbench: Towards stable large-scale benchmarking on tool learning of large language models. Preprint, arXiv:2403.07714.

Sirui Hong, Xiawu Zheng, Jonathan Chen, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, et al. 2023. Metagpt: Meta programming for multi-agent collaborative framework. arXiv preprint arXiv:2308.00352.

Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. 2023. Mistral 7b. arXiv preprint arXiv:2310.06825.

Guohao Li, Hasan Abed Al Kader Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. 2023. Camel: Communicative agents for "mind" exploration of large language model society. Preprint, arXiv:2303.17760.

Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2024. Lost in the middle: How language models use long contexts. Transactions ofthe Association for Computational Linguistics, 12:157–173.

Zhiwei Liu, Weiran Yao, Jianguo Zhang, Le Xue, Shelby Heinecke, Rithesh Murthy, Yihao Feng, Zeyuan Chen, Juan Carlos Niebles, Devansh Arpit, Ran Xu, Phil Mui, Huan Wang, Caiming Xiong, and Silvio Savarese. 2023. Bolaa: Benchmarking and orchestrating llm-augmented autonomous agents. Preprint, arXiv:2308.05960.

Pan Lu, Baolin Peng, Hao Cheng, Michel Galley, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, and Jianfeng Gao. 2023. Chameleon: Plug-and-play compositional reasoning with large language models. Preprint, arXiv:2304.09842.

Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Hajishirzi. 2022. When not to trust language models: Investigating effectiveness of parametric and non-parametric memories. arXiv preprint arXiv:2212.10511.

Jesse Mu, Xiang Lisa Li, and Noah Goodman. 2024. Learning to compress prompts with gist tokens. Preprint, arXiv:2304.08467.

Kenneth Page Oakley and London British Museum. 1972. Man the tool-maker. 538. British Museum (Natural History) London.

OpenAI. ChatGPT. https://openai.com/blog/ chatgpt.

Joon Sung Park, Joseph C. O’Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein. 2023. Generative agents: Interactive simulacra of human behavior. Preprint, arXiv:2304.03442.

Fabio Petroni, Patrick Lewis, Aleksandra Piktus, Tim Rocktäschel, Yuxiang Wu, Alexander H Miller, and Sebastian Riedel. 2020. How context affects language models’ factual predictions. arXiv preprint arXiv:2005.04611.

Chen Qian, Wei Liu, Hongzhang Liu, Nuo Chen, Yufan Dang, Jiahao Li, Cheng Yang, Weize Chen, Yusheng Su, Xin Cong, Juyuan Xu, Dahai Li, Zhiyuan Liu, and Maosong Sun. 2024. Chatdev: Communicative agents for software development. Preprint, arXiv:2307.07924.

Shuofei Qiao, Ningyu Zhang, Runnan Fang, Yujie Luo, Wangchunshu Zhou, Yuchen Eleanor Jiang, Chengfei Lv, and Huajun Chen. 2024. Autoact: Automatic agent learning from scratch for qa via self-planning. Preprint, arXiv:2401.05268.

Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, Sihan Zhao, Lauren Hong, Runchu Tian, Ruobing Xie, Jie Zhou, Mark Gerstein, dahai li, Zhiyuan Liu, and Maosong Sun. 2024. ToolLLM: Facilitating large language models to master 16000+ real-world APIs. In The Twelfth International Conference on Learning Representations.

Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, and Yoav Shoham. 2023. In-context retrieval-augmented language models. Transactions of the Association for Computational Linguistics, 11:1316–1331.

Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. 2023. Hugginggpt: Solving ai tasks with chatgpt and its friends in hugging face. Preprint, arXiv:2303.17580.

Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed Chi, Nathanael Schärli, and Denny Zhou. 2023. Large language models can be easily distracted by irrelevant context. Preprint, arXiv:2302.00093.

Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: Language agents with verbal reinforcement learning. Preprint, arXiv:2303.11366.

Yifan Song, Weimin Xiong, Dawei Zhu, Wenhao Wu, Han Qian, Mingbo Song, Hailiang Huang, Cheng Li, Ke Wang, Rong Yao, Ye Tian, and Sujian Li. 2023. Restgpt: Connecting large language models with realworld restful apis. Preprint, arXiv:2306.06624.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. 2023. Chain-of-thought prompting elicits reasoning in large language models. Preprint, arXiv:2201.11903.

Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, Ahmed Hassan Awadallah, Ryen W White, Doug Burger, and Chi Wang. 2023. Autogen: Enabling next-gen llm applications via multi-agent conversation. Preprint, arXiv:2308.08155.

Binfeng Xu, Zhiyuan Peng, Bowen Lei, Subhabrata Mukherjee, Yuchen Liu, and Dongkuan Xu. 2023.

Rewoo: Decoupling reasoning from observations for efficient augmented language models. Preprint, arXiv:2305.18323.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In Conference on Empirical Methods in Natural Language Processing (EMNLP).

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2022. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629.

Da Yin, Faeze Brahman, Abhilasha Ravichander, Khyathi Chandu, Kai-Wei Chang, Yejin Choi, and Bill Yuchen Lin. 2024. Agent lumos: Unified and modular training for open-source language agents. Preprint, arXiv:2311.05657.

Siyu Yuan, Kaitao Song, Jiangjie Chen, Xu Tan, Yongliang Shen, Ren Kan, Dongsheng Li, and Deqing Yang. 2024. Easytool: Enhancing llm-based agents with concise tool instruction. arXiv preprint arXiv:2401.06201.

Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc Le, and Ed Chi. 2023. Least-to-most prompting enables complex reasoning in large language models. Preprint, arXiv:2205.10625.

## A Comparison between Smurfs and existing MAS

In conclusion, Smurfs stands out compared to existing MAS in followig terms:

Flexibility Smurfs utilize enhanced version of DFSDT, which intergrades global and iterative planning, while existing MAS mainly use ReAct and iterative planning. This make Smurfs more flexible in the planning process.

Adaptability Smurfs realize superior performance on multiple different benchmarks, proving the adaptability of Smurfs. Existing MAS like Rest-GPT, Lumos are tailored for specific downstream task and need additional training to be used in other scenario.

Learning Efficiency Smurfs do not need training, thus have the highest learning efficiency among existing MAS. It only need query and tool documentation from the user.

![](images/4b2f45beca0ff6186c05c8875f54264d5117cf56a01dff164b9c11adc0c70009.jpg)  
Figure 4: Demonstration of the memory of the Smurfs framework.

To further illustrate these points, we conducted a detailed comparison between Smurfs and two wellknown Multi-Agent Systems, highlighting their differences and the adjustments required when learning out-of-box tasks.

CAMEL CAMEL (Li et al., 2023) is a communicative agent framework. It uses role-play technique and inception-prompting technique to achieve autonomous cooperation between agents. CAMEL does not natively support tool use settings. CAMEL is considered to perform poorly when generalizing to new tasks (e.g., on the MATH dataset (Wu et al., 2023)).

Autogen Autogen (Wu et al., 2023) is not designed as an agent framework for any specific task scenario. Instead, it provides a multi-agent conversation framework that allows users to customize agent characteristics and complete tasks through discussions among different agents. Autogen addresses various problems not by employing a uniform workflow, but by allowing users to design customized agents and workflows flexibly based on their tasks.

Smurfs Smurfs is designed as a unified workflow specifically for complex multi-tool planning scenarios. For different tasks, Smurfs only needs to adjust the few-shot examples in the agent prompts and provide documentation for the tools applicable to the task, allowing Smurfs to generalize to other task scenarios. Smurfs was initially designed for the Stable-toolbench, which itself encompasses various types of tasks and has access to over 16,000 plugins. HotpotQA was subsequently introduced to evaluate the performance of Smurfs on closed-end tasks. When migrating from Stabletoolbench to HotpotQA, only the few-shot examples in the planning agent prompts were modified, along with the provision of plugin documentation for HotpotQA. The rest of the system continued using the same unified framework as in Stabletoolbench. Experiment results demonstrate that our untrained out-of-box unified framework achieves and even surpasses performance of agent systems specifically trained on HotpotQA, such as Autoact and Fireact, showcasing Smurfs’ flexibility in generalization.

## B Details of DFSDT

DFSDT (Qin et al., 2024) gives control to the model to stop and rollback the solution trajectory by using Finish tool, thus addressing limitations of ReACT. Finish tool has two parameters give answer: model thinks the task is finished and decide to give answer and give up and restart: model thinks current trajectory can’t lead to correct answer and decide to rollback.

## C Details of the Smurfs

Executor Agent Details As illustrated in Figure 5, given a sub-problem p, Executor Agent first thinks about what to do this time, generates thought γ according to p, local memory M, hint h from the Verifier Agent and tool list at the current step τ. Then it will choose action α to complete the subproblem using p, γ and τ . After that, parameters of α are generated using p, local memory M and tool description of the action D[α]. Tool is then invoked to complete the task.

Memory Details As illustrated in Figure 4, there are four kinds of memory in Smurfs. Local memory stores thought-answer pairs of the current solution trajectory, while global memory stores all history solution trajectory (including those that is backtracked). Tool list only stores available tools’ name and its usage description, while tool doc stores all detailed information about the tools including parameters details. Through using different kinds of memory under different circumstances, Smurfs can use DFSDT in a context efficient way.

<table><tr><td rowspan=1 colspan=1>CompressionMethod</td><td rowspan=1 colspan=1>Applicable Scenario</td><td rowspan=1 colspan=1>Compressed Object</td><td rowspan=1 colspan=1>Implementation</td></tr><tr><td rowspan=1 colspan=1>Gist Tokens (Muet al., 2024)</td><td rowspan=1 colspan=1>General scenarios</td><td rowspan=1 colspan=1>Frequently used systemprompts</td><td rowspan=1 colspan=1>Training LLM to compress sys-tem prompts, reducing token us-age.</td></tr><tr><td rowspan=1 colspan=1>CAMPHOR  (Fuet al., 2024)</td><td rowspan=1 colspan=1>Tool use scenarios</td><td rowspan=1 colspan=1>Tool descriptions</td><td rowspan=1 colspan=1>Adopts a similar approach to gisttokens, compressing each tool de-scription into a single token, thusreducing token cost.</td></tr><tr><td rowspan=1 colspan=1>Smurfs</td><td rowspan=1 colspan=1>Complex multi-tool plan-ning scenarios</td><td rowspan=1 colspan=1>Input context in tool plan-ning</td><td rowspan=1 colspan=1>Operates on multi-tool planningworkflows, compressing the con-text needed for each tool plan-ning process.</td></tr></table>

Table 6: Comparison of Token Compression Methods

Restart Mechanism Every time Smurfs generate an intermediate output, a format checker is used to check whether the output is of the expected format. If not, Smurfs will retry the same step until reach retry limit or generate correct format output. This mechanism is used in addition to the rollback mechanism to handle the situation where the system can generate correct content but fail to follow the output format.

## D Experiment Settings for Hotpot QA

Following settings in (Qiao et al., 2024), which is randomly select 300 dev questions divided into three levels for evaluation, with 100 questions in each level. For tool library that can be used in HotpotQA, see Table 7

## E Error Analysis on HotpotQA

To provide further insight into the model performance in the tool planning process, we manually categorized the types of error made by Smurfs in the hotpotQA hard dataset. Table 8 shows that the most frequent errors committed by mistral-7b Smurfs is tool argument fail, followed by bad planning and answer miss. Smurfs do not make tool choice errors and premature termination errors. This shows that Smurfs actually alleviates the premature termination problem, making the tool choice process more robust. Additionally, we note that a portion of the error samples are false negatives which arise when the generated answers differ in expression from the ground truth but are equivalent in meaning. This highlights potential directions for future improvements in Smurfs.

## F Prompts for multi-agent implementation

Prompts used by each agent and their example outputs are shown in Figure 6 to 12.

## G Token Cost on StableToolBench Evaluation

We analyzed the token cost for the StableTool-Bench experiments. As shown in Table 9, the total token cost for each subtask within the StableToolBench is compared across three candidate tool-planning methods. The data demonstrates that, across all tasks from easy to hard, DFSDT consistently incurs high token costs, while the other two methods maintain relatively low token costs. This verifies the context-efficiency of the proposed method.

![](images/72ff3e29c4a820978b7b4f206681d5bd338dc0801693f25cba01a30f9a9a2196.jpg)  
Figure 5: Details of the executor agent working process

<table><tr><td>Name</td><td>Definition</td><td>Usage</td></tr><tr><td>BingSearch</td><td>BingSearch engine can search for BingSearch[query], rich knowledge on the internet searches the exact detailed query based on keywords, which can compensate for knowledge fal- relevant information to the query. lacy and knowledge outdated.</td><td>which on the Internet and returns the Be specific and precise with your query to increase the chances of getting relevant results. For example, Bingsearch[popular dog breeds in the United States]</td></tr><tr><td>Retrieve</td><td>Retrieve additional background1 knowledge crucial for tackling t cially beneficial for specialized ists. If not, it will return some matics, providing context for the example, Retrieve[Milhouse] task</td><td>Retrieve[entity], which retrieves the exact entity on Wikipedia and complex problems. It is espe- returns the first paragraph if it ex- domains like science and mathe- similar entities to retrieve. For</td></tr><tr><td>Lookup</td><td>ality on the browser.</td><td>A Lookup Tool returns the next Lookup[keyword], which returns sentence containing the target the next sentence containing the string in the page from the search keyword in the last passage tool, simulating Ctrl+F function- successfully found by Retrieve or BingSearch. For example, Lookup[river].</td></tr></table>

Table 7: Tool library for HotpotQA.
<table><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>Bad Planning Answer Miss Tool Wrong Tool Argument Fail False Negative Premature Termination Total Accuracy</td><td></td></tr><tr><td>Mistral-7b-Smurfs</td><td>11</td><td>7</td><td>0</td><td>28</td><td>12</td><td>0</td><td>0.42</td></tr></table>

Table 8: Error analysis for Smurfs on HotpotQA Hard.
<table><tr><td rowspan="2">Backbone</td><td rowspan="2">Method</td><td colspan="10">StableToolBench</td><td rowspan="2"></td><td colspan="4"></td></tr><tr><td>I1-Inst. Total</td><td>Avg.</td><td>Total</td><td>I1-Cat. Avg.</td><td>Total</td><td>I1-Tool. Avg.</td><td>Total</td><td>I2-Cat. Avg.</td><td>Total</td><td>I2-Inst. Avg.</td><td>Total</td><td>I3-Inst. Avg.</td><td>Total</td><td>Average Avg.</td></tr><tr><td>GPT-3.5 Turbo</td><td>ReACT</td><td>1,010,304</td><td>6,198</td><td>824,676</td><td>5,390</td><td>1,010,514</td><td>6,396</td><td>900,855</td><td></td><td>7,265</td><td>824,510</td><td></td><td>461,121</td><td>7,559</td><td>838,663</td><td></td></tr><tr><td>GPT-3.5 Turbo</td><td>DFSDT</td><td>3,303,062</td><td>20,264</td><td>2,745,667</td><td>17,945</td><td>3,152,532</td><td>19,953</td><td>2,560,297</td><td>20,648</td><td></td><td>3,098,365</td><td>7,778 29,230</td><td>1,390,787</td><td>22,800</td><td>2,708,452</td><td>6,764 21,807</td></tr><tr><td>GPT-3.5 Turbo</td><td>Smurfs</td><td>1,090,404</td><td>7,127</td><td>1,917,348</td><td>11,763</td><td>1,464,535</td><td>9,269</td><td>957,088</td><td></td><td>7,638</td><td>1,096,162</td><td>10,341</td><td>632,084</td><td>10,362</td><td>1,191,270</td><td>9,417</td></tr></table>

Table 9: Token costs for various candidate tool-planning methods on the StableToolBench benchmark (Guo et al., 2024). ‘Total’ indicates the total number of tokens used to complete each subtask, including both prompt and completion tokens. ‘Avg.’ represents the average number of tokens used per question within the subtasks. Higher token counts imply greater costs for solving the same task.

![](images/98adf27f5b7d1351fa8272e5fd67396190902ac89df35e30ec8fc90758ff4bcb.jpg)  
Figure 6: An example prompt for task decomposition in the framework.

![](images/745a597a875d5eba3e968a092f3a6c46038c5d47e704ebd90af43b742c704651.jpg)

Figure 7: An example prompt for tool check in the framework.  
![](images/135c61202a7960c0cde7480faf80b44e0e049092624c38d52aee9db9c65396b1.jpg)  
Figure 8: An example prompt for tool check in the framework.

![](images/418af8d9614cd3be8042b14d66b5b4bf98d6f6a702ab75bb35a9469e95053f87.jpg)  
Figure 9: An example prompt for action generation in the framework.

![](images/328694c5477349e3a3a151bc0778c89193841e3827901349291c4782f2546448.jpg)  
Figure 10: An example prompt for action input generation in the framework.

![](images/ac197bfb2afc1224f1649e4114052093d6fc774e2cf96fb938d260a2be54999e.jpg)  
Figure 11: An example prompt for Answer Agent in the framework.

![](images/1b34294be3c6866696108b7cdb3b2c9371e93753b6fb6e2a40796c07a4583144.jpg)  
Figure 12: An example prompt for Verifier Agent in the framework.