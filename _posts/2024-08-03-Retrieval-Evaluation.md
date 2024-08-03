---
layout: post
title: "Synthetic Question Generation for Retrieval Evaluation of RAG"
date: "2024-08-03"
mathjax: true
status: []
categories: [NLP]
---

Synthetic data generation has been frequently used in the generation or expansion of the training set to improve the downstream performance on the test set. It works because of additional synthetic examples, which are small variants in the training instances provide a regularisation effect. There is little harm to using synthetic training data for training, because if the training data is generated poorly, it'll show as little to no performance gains on the evaluation set.

However if we're talking industry, multi-million dollar projects, then a poorly constructed evaluation set will set you in the wrong direction for nearly all aspects of model and algorithm development. This costs your organisation millions of dollars. At the same time, if we're talking industry, then your users' time is expensive and it's near impossible to get labeling support with sufficient coverage over the true distribution of information requirements that users have for your industrial RAG system. 

**The Central Question: Does it make sense to do synthetic data generation of the *evaluation set* itself?**

<br>
#### **Online Wisdom for RAG Evaluation**

Honestly if we're in a prototyping or demo setting, it's really not at all that critical what you do. After all, it's just a small evaluation set for a proof-of-concept. For instance, the [HuggingFace article on RAG evaluation](https://huggingface.co/learn/cookbook/rag_evaluation) describes a simplistic workflow where we generate question and answer from Model A, filter questions with Model A, generate questions with Model B, and evaluate answers with Model C. 

The only ground truth we have, is Model A's truth. However I'll say right now that with 2024 LLMs available to industry (the age of Claude3.5, GPT4, Gemini 1.5) to evaluate the RAG system, **we should focus on evaluating the retrieval not the generation.** Retrieval is the main limiting factor for a RAG system performance where it really matters. 

<br>

#### **Problem Definition**
Let $d_1, \cdots, d_m \in D$ be a set of documents, where each document can be chunked into $[c_1, \cdots, c_{nd}]=d$ chunks. Let $(Q, A)_d$ be the true distribution of Information requirements. 

This is the set of question answer query pairs that the user has for a given document $d$, which has $nd$ chunks. Note that there may be either 0 or multiple relevant $(q, a)$ pairs per chunk. The goal is to generate the test set $(Q, A)_{d'}$ such that it is close to $(Q, A)_d$.

<br>

#### **Question Generation with User's workflow**
Given $c\in [c_1, \cdots, c_{nd}]$ for document $d$, we wish to have a question generator $p(q |c)$ which can approximate the true question distribution. The support of the distribution does not exist for every chunk, i.e., not every chunk has a relevant Question-Answer pair. 

In order to get QA pairs with good coverage, we need to understand how the system will be used in reality, and it follows that we need to understand the tasks and information requirements of the user. The three sources that we mostly have available are

* An existing help system chat logs. 
* Case Studies.
* Expert interviews.

From this, we abstract the user's workflow into a series of steps. We can then prompt the model to do question generation for those categories of questions. For instance, we might be interested in constructing questions related to error codes or understanding default parameters from manufacturing manuals. The QA generation prompt would then look like the following:


>You are given a piece of information. Your task is to generate a pair of Questions and Answers, which is specific and relevant to this information content.
Information: {information}
Please generate a question related to {information_requirement}, where possible. If there is no related question, do not generate it. Instead generate BLANK.


We can then construct the retrieval set, by looping over the information requirements and each information chunk using the prompt above.


<br>
#### **Evaluating the Retrieval Method**

Once we have the questions and chunks, we can evaluate the mean average recall@k for $(q, c)$, for each type of information requirement and quantify which kind of information is the hardest to obtain from the documents. 

<br>

#### **Other Comments**

1. Aside from focusing on retrieval as it is the main bottleneck, trying to evaluate the generation will just confuse everyone involved with free-text similarity metrics and all its problems. In particular the extremely hazy practice of using a more powerful to provide a "score" for a smaller one. (First, this is complete hackery, second, why didnt we just use the large model to do the generation if we had access to it). 

2. A clustering approach can be used to understand the information requirements if using an existing help system or chat logs to understand the user's workflow. Might write about this later. 

3. Synthetic Question-Answer generation consists of a 3 step modeling pipeline consisting of unconditional answer extraction from text, question generation and question filtration. The most critical part is the initial Q, A generation as if this is done well, the latter steps are actually redundant. 

4. This method of question generation has a big weakness in that it does not account for multi-hop questions.
