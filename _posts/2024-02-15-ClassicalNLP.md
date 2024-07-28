---
layout: post
title: "A classical NLP researcher and a GPT-era Engineer meet at the coffee machine"
date: "2024-02-15"
mathjax: true
status: [Misc]
categories: [Generative Models]
---

### **Preface**
I sometimes find myself reminiscing on Classical NLP and structured Prediction. This is a conversation (that took place in my head) between my old course notes and <span style='color:blue'>a newcomer to NLP in the GPT-era</span>.



#### **(Definitions)**
<span style='color:blue'>So I've been dying to ask, what does a Language Model mean actually. </span>

In classical terms, "Language Model" meant a statistical model with structure which gave us probability distributions over sequences. For the joint probability of $p(w_1 \cdots w_n)$ we can apply the chain rule to get $p(w_1) p(w_2 \mid w_1) p(w_3\mid w_{1,:2}) \cdots P(w_n\mid w_{1:n-1}) = \prod_{t}^n p(w_t\mid w_{1:t})$.



<span style="color:blue">
I see. I thought "Language Model" means magic black box trained on large-scale data, and llama.cpp that runs inference on your local machine.
</span>
 
<br>

In the old days, we thought a lot harder about what we were trying to model and how we could get accurate estimates
of probabilities over sequences. I contrast this with today’s empirically satisfying, but mathematically unsatisfying approach to research. 

<span style="color:blue">
Do you mean today's research papers which do massive training runs and report scaling laws of
large-scale GPU experiments? Why isn't that understanding? We understand what to expect.</span>

<br>

Yes, but we only understand what to expect by running these large scale experiments. We don't
actually know what to expect based on an inherent understanding of how the models work. It's a more shallow kind of understanding. But maybe for now it's the best we got.


<br>

#### **(Motivation)**

The starting point is the same. We want to have a model human language to do Natural Language tasks. In classical times, we spent more time thinking about the nature of language, and the structure of
the problem. Given a sequence or single word, *how* can we predict what the next word will be? Given an input sequence, *how exactly* do we predict its POS tags? It’s sentiment? It's translation into a different language?

Most of this can be formulated by the conditional probability $p(y\mid X)$. Where $y$ is a variable, a tag, a next word, etc that we would like to classify and $X$ is any observable context (typically words) that accompanies $y$. 



<span style="color:blue">
People used to care about those, really? I thought we want answers to any question that is conceivable by human kind. But $p(y|X)$ doesn't sound that different.  $X$ is the prompt or instructions to the task that we want the model to do.
</span>

<br>


Yes, but $p(y \mid X)$ is so abstract that two completely different approaches can be "similar" at that level. Today, $p(y \mid X)$ is modelled by a massive transformer neural network trained on internet scale data which costs billions of dollars. The main difference being the training scale and access to different types of data. In the old days, data was scarcer and models were trained on academic datasets or industry proprietary datasets, and most of the differences were *modeling* and *algorithmic* differences. 

<span style="color:blue">
Why couldn't people just use one model or architecture?
</span>

<br>


There was a lot of modeling and algorithmic improvements to be made as people were occupied with two key challenges of modeling human language. 

1. The first is that language is observed as a sequence with an *arbitarily* long context. Often, we need to calculate the highest probability assignments over the entire sequence, not just a classification at a single time step. For example, if we predict an Adjective, it is more likely to be followed by a Noun. 


2. The second theme  is ambiguity. The word “fly” could be a verb, adjective, and noun! This is just one example for POS tagging and there are many other instances for other NLP tasks. Because of this ambiguity, we might have taken a number of non-deterministic routes to arrive at the current best prediction over a sequence. So which is the most likely path?’


These two aspects of language, a sequence that is ambiguous, make the potential space of $Y$ exponentially large. If there are $15$ possible POS labels, and the probability of each label is non-zero, then for a sequence of $5$ words, there are $15^5$ possible sequences. Hence many of the classical NLP algorithms make efficient computation a focus. By efficient computation, we mean the calculation and pruning of probabilistic paths.


<span style="color:blue">
I see, in my experience, GPT deals with ambiguity really really well. So it seems that problem is more or less solved. Although there is still work on modeling long contexts, I guess people will never be satisfied until the models can handle infinite context. As for efficiency, it's a problem but not in the same sense of efficient computation over paths, but rather the models are so massive that forward inference takes a long time and self-attention is famously quadratic in the length of the context. A lot of work recently tries to address that. 
</span>

<br>

Yes, and it generalises incredibly well to new situations. The In-context Learning paradigm and
prompt engineering was a field born out of GPT's release which works because the model
generalises incredibly well without additional training. We used to focus a lot on modeling to target generalisation. It turns out that data and massive models were the answer all
along. 


<span style="color:blue">
Still, I think the modeling considerations that people had then were beautiful and might give us a starting point to think about what might theoretically be happening in today's LLMs. Do you mind sharing briefly about what were the previous modeling considerations?
</span>

<br>

The central question is: How can we generalise to test (unseen) data based on the statistics of training (seen) data? How should we design our model, such that it is generalisable?

If we want to be able to model $p(y\mid x)$, why not just model the entire context seen with this
classification i.e. $p(y\mid x_1, x_2, x_3, x_4, … x_n)$? 

<span style="color:blue">
Because we are unlikely to see this exact sequence ever again?
</span>

<br>


Exactly. We can’t possibly capture every context in our model and therefore certain independence
assumptions need to be made for generalisation. We hope that what we condition on actually
affects the outcome. Therefore classical sequence algorithms typically have careful considerations on what we should include in the modeling architecture which fits the task. The starting point was N-gram models, which are based on occurrence counts in the training data. This presents as a weakness in generalisation as it is difficult to generalise to unseen contexts. 

<span style="color:blue">
Yeah it sounds kind of naive.
</span>

<br>

Naive, but principled statistical approach. They were, and are still good for broad tasks like text-categorization/language identification for the entire document. 

After that was Log-linear Models. All sorts of contextual information could be interesting for us to model
  $p(y\mid x)$, not just the raw words. Why not have $x$ be a POS? The POS and a word at position $i$?
Log-linear models give us a way to turn scoring functions $f(x,y)$, into the conditional
probability $p(y\mid x)$, where $x$ is some context seen with the target.

<span style="color:blue">
That sounds a little better. At least we have features, not just word counts.
</span>

<br>

Yes, Log-linear features actually still persist conceptually in deep learning models. We can think of the
last layer of the transformer as a log-linear model over the feature space, which is used to
predict the next token. Just that the work of feature engineering is completely taken over by the
neural network.


<span style="color:blue">
Did we also have models akin to hidden states of neural network?
</span>

<br>

Yes, Hidden Markov Models were our attempt at explicitly modeling hidden states. The key difference from the previous models, is that we modelled ‘hidden’ variables $z$, (also known as latent variables). These latent variables are hidden structures, which we hope help explain our observed data $x$. Perhaps we want to predict the latent variables $z$ directly, or we want to use them in some downstream task to predict y. They may or may not help us to better predict the output label y. But they are often what we believe to be causal unobserved
factors that lead to our observations. For instance, the weather is a latent variable that
caused your professor to eat ice-creams. Or the POS is a latent variable that caused us to observe the
word. 

<span style="color:blue">
These sound like discrete concepts while Neural networks are all fuzzed up inside.
</span>

<br>

That's a good point. Classical NLP had alot of discrete concepts. FSMs are a step-up from
N-grams in making more complex predictions in that it introduces the concepts of having states. The state transition model governs sequences.  In applying FSAs/FSTs to NLP tasks, we are hypothesising that there is an underlying finite state machine that changes state with each input element. The probability of the next item is
conditioned not only on the previous words, but on the state that the machine is in. 

<span style="color:blue">
Transformer models also maintain hidden states conditioned on the previous words. But we can't
really examine or scrutinise what these states are. I think the difference is that we can't
define a functional block or functional unit since we kind of left it to the model to figure
out it's own functional blocks.
</span>
<br>


We trained the model with as much data imaginable, and left it to the model to figure out a lot of things. On an unrelatd note, I heard we're having a lot of challenges evaluating models now? What's that about?

<br>

#### **(Training/Evaluation Metrics)**


<span style="color:blue">
Oh that's actually related to the paradigm of training the models with as much data as possible! Since models were trained on internet data, and the internet has nasty pockets of space, our models can get nasty too. Also, since the output is free-form, it becomes really hard to evaluate whether it's "correct" or not. For these models to be "safe" to use, we do have to be concerned about factuality, biasness, discrimination, plagiarism and other hard to define, sometimes subjective even by humans, and hard to measure concepts.
</span>

<br>




It doesn't sound like we can do anything too useful for training models with such metrics. 

<span style="color:blue">
Why?
</span>

<br>



Metrics are important not only for final evaluation, but are used for training our models and
guiding hyperparameter search. During training, the gradient of the metric (loss) function
tells us the direction to shift model weights so that we can reduce the loss. The metric
performance on the validation set is used a potential signal to stop training, and the metric
for the test set is related to the task that we ultimately care about. Therefore, it is
important to be clear about what each metric is measuring and what our model is optimising for. With such abstract concepts to measure, I don't think we could use it for training.

<span style="color:blue">
What kind of metrics did people use in the past then?
</span>

<br>


In NLP we often talk about cross-entropy, perplexity, error rate, log loss. 

<span style="color:blue">
That sounds familiar. Is that just next token prediction?
</span>

<br>

Next token prediction is a particular task. Loss functions are ways of computing the difference
between predicted and expected value, of which particular forms of loss have implications on the gradient. In general regardless of intrinsic or extrinsic metrics, we would like to have continuous measures of
error so that we can make incremental steps in the right direction. Take for instance, the Log-loss (aka logistic regression loss or cross-entropy loss). The log loss per sample is how suprised our model is to see the correct answer $y_c$. It is defined by: $-\sum_{c}^C I(y_c) \mathrm{log}  p(\hat{y_c}\mid x)$, where $I(y_c)$ is an indicator function for 0 or 1 if the correct class is $c$. So if our model predicted that $p(y_1\mid x)=1$, then our log loss would be 0. Perplexity is an information-theoretic metric for sequences which is the exp(Cross-entropy).


<span style="color:blue">
Yeah I guess it makes sense that we want continuous measures of error. 60% confidence on the wrong classification is better than 80% confidence on the wrong classification. But what do you mean by intrinsic and extrinsic.
</span>

<br>

Intrinsic evaluation measures how good the system is at generating the observed data. That is,
the model includes a probability distribution over the input data, and we evaluate its
“generative ability” by how well it models the distribution. By intrinsic, we mean task
independent, which may or may not be correlated with extrinsic model evaluation (downstream
tasks). Naturally, Language modeling is evaluated using intrinsic metrics because it give us
a probability distribution over a word sequence. Discriminative metrics are constructed with respect to some external task such as document classification, or tagging.
They are called discriminative metrics because they measure how well the model is at
discriminating between classes. But this distinction is mostly lost now because everyone uses
the cross-entropy loss. 


<span style="color:blue">
Ah, self-supervised learning and scale won out in the end.
</span>

<br>

Yes, if I'll be honest, I find it quite unsatisfying.

<span style="color:blue">
But it was in no way easy to get here. It was a convergence of many small things, it didn't
just happen over night, it was an accumulation of hardware advancements, software engineering practices, distributed systems, ML Ops, Deep Learning tool boxes, ironing out the details of stably training
neural networks...
</span>

<br>

I do agree that it took a lot to get here. But we just dont think that much about modeling and generalisation anymore other than let's throw more data at the problem. Although, I concede maybe it's because we have reached quite a mature state of model architecture. I expect there'll be more to improve on, but it seems like the transformer architecture will be the driver of NLP applications for another 5-10 years. 

<span style="color:blue">
What was before Transformers and why was it such a breakthrough?
</span>

<br>

I'm sorry, but do you mind if we continue this another time? I have a meeting in 5 minutes. 

<span style="color:blue">
Yes yes of course, I'm sorry to hold you up!
</span>

<br>


Let's continue later, I'm serious!


##### **Acknowledgements**
Jason Eisner for teaching me everything I know about classical NLP and more. 

**Disclaimer:** This barely scratches the surface of classical NLP, and there are alot more about specific inference and approximation methods, but since we're in the era of deep learning where gradient descent is king, I've chosen to note down more on structure and models rather than learning algorithms. 
