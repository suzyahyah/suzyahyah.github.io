---
layout: post
title:  "Noise Contrastive Estimation"
date:   2017-07-01 21:09:09 +0800
mathjax: true
categories: jekyll update
---
### Key Concepts
* A language model is a probability distribution over sequence of words. It is a generative model that can be used to generate words based on its surrounding context (e.g previous words, or window of words)
<br><br>
* Noise Contrastive Estimation is a general parameter estimation technique for locally normalized language models.
<br><br>
* The "Noise" is a distribution that generates samples, for which a probabilistic binary classifier learns to distinguish from the real distribution. 

### Model Preliminaries
