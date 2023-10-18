---
layout: post
title: "NLP Papers at ICML2022"
date: "2022-10-24"
mathjax: true
status: [Conference Review]
tldr: Several papers were presented at ICML 2022, including topics such as co-training large language models with smaller models, using derivative-free optimization for language models, interpretable text modeling, generative cooperative networks for language generation, language model architecture for zero-shot generalization, coherent entity use in narrative generation, retrieval-augmented language models, and self-conditioning pre-trained language models. Some papers proposed new methods, while others explored existing techniques in various ways.
categories: [Review]
---


Summarising ideas (usually methods) from some of the NLP papers at ICML. 

-> [Single Page View Here]({{ site.baseurl }}{% link multipages/ICML_post/2022-10-24-NLP-at-ICML-all.md %}) 

Can't comment much
about trends as they are mostly one to two papers in each sub NLP category. Although a common
thread is building off your own work. Many times I wondered why a paper used some particular less
well-known model or method and the answer almost always can be found in the list of authors. Apparently happens alot in
ICML in general.



Relying on GPT3 prompt paradigm
* [Co-training improves prompt-based learning for Large LMs]({{ site.baseurl }}{% link multipages/ICML_post/2022-10-24-NLP-at-ICML-3.md %})
* [Blackbox tuning for Language Models as a service]({{ site.baseurl }}{% link multipages/ICML_post/2022-10-24-NLP-at-ICML-2.md %})

Fancy Inference
* [Latent Diffusion Energy-based model for interpretable Text Modeling]({{ site.baseurl }}{% link multipages/ICML_post/2022-10-24-NLP-at-ICML-6.md %})
* [Controlling conditional language models without catastrophic forgetting]({{ site.baseurl }}{% link multipages/ICML_post/2022-10-24-NLP-at-ICML-1.md %})


Probably Novel 

* [Generative Cooperative Networks for Natural Language Generation]({{ site.baseurl }}{% link multipages/ICML_post/2022-10-24-NLP-at-ICML-4.md %})

Empirical ++
* [What Language Model Architecture and Pretraining Objective Works Best for Zero-Shot Generalization?]({{ site.baseurl }}{% link multipages/ICML_post/2022-10-24-NLP-at-ICML-5.md %})

Architecture
* [Towards Coherent and Consistent Use of Entities in Narrative Generation]({{ site.baseurl }}{% link multipages/ICML_post/2022-10-24-NLP-at-ICML-7.md %})
* [Improving Language Models by Retrieving from Trillions of Tokens]({{ site.baseurl }}{% link multipages/ICML_post/2022-10-24-NLP-at-ICML-9.md %})
* Improving Transformers with Probabilistic Attention Keys 

Intepretability
* [Self-conditioning Pre-Trained Language Models]({{ site.baseurl }}{% link multipages/ICML_post/2022-10-24-NLP-at-ICML-8.md %})
<br>


---
<br>

#### High Level non NLP Impressions 

**Neural Architecture** papers appear to be at the level of controllable multi-task mixture of
expert stuff. People are trying to combine meta-learning and multi-task and subspace learning.

**Approximate Inference** I feel like there were few variational inference advances, and also very
few Bayesian NN at the conference. Variational Inference papers seemed to be mostly for
specific architectures, and people seem to be working on sampling methods again.

**NN Theory** Empirical NN papers without theory were more popular at poster sessions than the
two-layer NN theory proof papers. Simply more digestable/convincing/practical?!

**Robustness, fairness, differential privacy** Very popular. Big in the main conference and also in the workshops.


<br>

#### Acknowledgements

Conference attendance was generously supported by my advisor [Kevin Duh](https://www.cs.jhu.edu/~kevinduh/). Also
shout out to my ICML twin [David Mueller](https://damueller.com/#/) who has his more ML focused
coverage [---> here! ](https://damueller.com/#/blog-post/ICML2022)
