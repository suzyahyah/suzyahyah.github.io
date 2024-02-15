---
layout: post
title: "The Good old Days of Classical NLP (Structured Prediction)"
date: "2024-02-15"
mathjax: true
status: [Misc]
categories: [Misc]
---

### **Preface**
I sometimes find myself reminiscing on Classical NLP and structured Prediction. In classical terms, "Language Model" meant a statistical model with structure which gave us probability distributions over sequences. (The paradigm has certainly shifted now, where "Language Model" means magic black box.) 

In those days, we thought a lot harder about what we were trying to model and how we could get accurate estimates
of probabilities over sequences. I think the modeling considerations that people had then were beautiful and might give us a starting point to think about what might theoretically be happening in today's LLMs. I contrast this with today’s empirically satisfying, but mathematically unsatisfying approach to research. To put it directly, observing massive training runs and scaling laws of large-scale GPU experiments.  

<br>

#### **Classical Motivation**

We want to have a model human language to do Natural Language tasks. Given a sequence or single
word, can we predict what the next word will be? Given an input sequence, can we predict its
POS tags? It’s sentiment? It's translation into a different language?

Most of this can be formulated by the conditional probability $p(y\mid X)$. Where $y$ is a variable, a tag, a next word, etc that we would like to classify and $X$ is any observable context (typically words) that accompanies $y$. 

There are two key aspects of language that motivate the myriad of algorithms and methods that we see
in classical NLP. 

1. The first is that language is observed as a sequence with an *arbitarily* long context. Often, we need to calculate the highest probability assignments over the entire sequence, not just a classification at a single time step. For example, if we predict an Adjective, it is more likely to be followed by a Noun. 

2. The second theme  is ambiguity. The word “fly” could be a verb, adjective, and noun! This is just one example for POS tagging and there are many other instances for other NLP tasks. Because of this ambiguity, we might have taken a number of non-deterministic routes to arrive at the current best prediction over a sequence. So which is the most likely path?’


These two aspects of language, a sequence that is ambiguous, make the potential space of $Y$ exponentially large. If there are $15$ possible POS labels, and the probability of each label is non-zero, then for a sequence of $5$ words, there are $15^5$ possible sequences. Hence many of the classical NLP algorithms make efficient computation a focus. By efficient computation, we mean the calculation and pruning of probabilistic paths.

<br>

#### **Training/Evaluation Metrics**

Metrics are important not only for final evaluation, but are used for training our models and
guiding hyperparameter search. During training, the gradient of the metric (loss) function
tells us the direction to shift model weights so that we can reduce the loss. The metric
performance on the validation set is used a potential signal to stop training, and the metric
for the test set is related to the task that we ultimately care about. 

In NLP we often talk about cross-entropy, perplexity, error rate, log loss. Therefore, it is
important to be clear about what each metric is measuring and what our model is optimising for. 
We distinguish between two broad classes of metrics, intrinsic model evaluation (generative
metrics), and extrinsic model evaluation (discriminative metrics). 


In general regardless of intrinsic or extrinsic, we would like to have continuous measures of
error so that we can make incremental steps in the right direction. 60% confidence on the wrong
classification is better than 80% confidence on the wrong classification. 


<u>Intrinsic model evaluation: (Generative Metrics)</u>

Intrinsic evaluation measures how good the system is at generating the observed data. That is,
the model includes a probability distribution over the input data, and we evaluate its
“generative ability” by how well it models the distribution. By intrinsic, we mean task
independent, which may or may not be correlated with extrinsic model evaluation (downstream
tasks). Naturally, Language modeling is evaluated using intrinsic metrics because it give us
a probability distribution over a word sequence. 

<u>Extrinsic model evaluation: (Discriminative Metrics)</u>

This metric is with respect to some external task such as document classification, or tagging.
They are called discriminative metrics because they measure how well the model is at
discriminating between classes. 

Options for continuous error measurements:

* Expected Error Rate - on average, how likely is the classifier getting it wrong? We can get
continuous measures of error rate as long as our classifier constructs a probability
distribution over the target output. If our $p(y=y_1|x)=0.7$, then on average the classifier will
predict $y_1$ 70% of the time, and the expected number of errors is 70%

* Log-loss (aka logistic regression loss or (binary) cross-entropy loss). The log loss per sample is how suprised our model is to see the correct answer $y_c$. It is defined by: $-\sum_{c}^C I(y_c) \mathrm{log}  p(\hat{y_c}\mid x)$, where $I(y_c)$ is an indicator function for 0 or 1 if the correct class is $c$. So if our model predicted that $p(y_1\mid x)=1$, then our log loss would be 0. Perplexity is an information-theoretic metric for sequences which is the exp(Cross-entropy).

Note the due to the shape of the log-loss function, it penalises wrong answers exponentially more than correct answers. The loss for a single $y$ ranges from [0, $\infty$), so a single very very poorly classified example will penalise our scores much more heavily than average acc or expected error rate.


<br>

#### **The high-level goal of Modeling**

The central question is: How well can we generalise to test (unseen) data based on the statistics of training (seen) data? This problem of generalisation is actually a modeling problem. How should we design our model, such that it is generalisable?

If we want to be able to model $p(y\mid x)$, why not just model the entire context seen with this
classification i.e. $p(y\mid x_1, x_2, x_3, x_4, … x_n)$? Clearly this is not generalisable, because we
are unlikely to see this exact sequence again in the test set. Sparsity of counts of such
occurrences and the estimates based on observed counts from the training set become very
unreliable. Such models have low bias, but high variance and are unable to generalise well. 

We can’t possibly capture every context in our model and therefore certain independence
assumptions need to be made for generalisation. We hope that what we condition on actually
affects the outcome. Therefore sequence algorithms typically assume a ‘Markov assumption’
(informally known as “backoff”). 


*Note: In today's LLMs, we no longer make Markov and assumptions/ are able to take more context into consideration, as we can pack more information into model parameters.*

The following models introduced all make some kind of independence assumption, and were unique
to NLP or sequence learning problems. 

* Ngram Models -  N-gram models are based on occurrence counts in the training data. This
  presents as a weakness in generalisation as it is difficult to generalise to unseen contexts.
The strength of N-gram models are the simplicity in estimating probabilities based on frequency
counts. They are good for broad tasks like text-categorization/language identification for the
entire document. 

* Log-linear Models - All sorts of contextual information could be interesting for us to model
  $p(y\mid x)$, not just the raw words. Why not have $x$ be a POS? The POS and a word at position $i$?
Log-linear models give us a way to turn scoring functions $f(x,y)$, into the conditional
probability $p(y\mid x)$, where $x$ is some context seen with the target.


* HMMs - The key difference from the previous models, is now we model ‘hidden’ variables $z$,
  (also known as latent variables). These latent variables are hidden structures, which we hope
help explain our observed data $x$. Perhaps we want to predict the latent variables $z$ directly,
or we want to use them in some downstream task to predict y. They may or may not help us to
better predict the output label y. But they are often what we believe to be causal unobserved
factors that lead to our observations. For instance, the weather is a latent variable that
caused your professor to eat ice-creams. Or the POS is a latent variable that caused us to observe the
word. 

* FSMs and FSTs  - FSMs are a step-up from N-grams in making more complex predictions in that
  it introduces the concepts of having states. The state transition model governs sequences. 
In applying FSAs/FSTs to NLP tasks, we are hypothesising that there is an underlying finite
state machine that changes state with each input element. The probability of the next item is
conditioned not only on the previous words, but on the state that the machine is in. Finding
the correct path through the Finite State Automata can be viewed as a Search Problem. The
search space is defined by the structure of the machine. 


* CRFs - With HMMs, we calculated $P(Y\mid X)$ by $P(Y\mid X)$ = $P(X, Y)/p(X)$. HMMs are a generative model which models the joint probability distribution of the observed sequences and hidden states. In contrast, CRFs are a discriminative model, which model $P(Y\mid X)$ directly. Since we do not need to model the generative process, this frees us from the Markovian independence assumptions that HMMs have, and hence CRFs can model arbitrary dependencies without assuming a fixed dependence structure.


Essentially, structured prediction and learning problems were addressed by increasingly complicated models. Unlike many areas of machine learning, we have to deal with probability distributions over unboundedly large structured variables such as strings, trees, alignments, and grammars. 


<br>

#### **Syntax**

<u>Grammars</u>

A Grammar is not a statistical model, but more like a structural model of language. It
describes our assumption about how natural language arises. Grammars are a step up from N-grams
and log linear models in trying to model more complex notions of linguistics. Basically they
are important whenever we have a task that follows this modeling philosophy: model
‘constituents’ compositionally, - units of meaning that have been composed from smaller units. 

3 views of grammar rules
1. Generation (Production) i.e, producing a grammatical sentence
2. Parsing (Comprehension) i.e, assigning grammatical structure to the sentence for interpretation.
3. Verification (Acceptance) i.e., checking whether the sentence is valid

It turns out that for each of these different views, we can use the same algorithm, but change
how the paths are combined depending on our objective. For that we use algebraic structures
called semi-rings, for code and algo reusability. Change the semi-ring for a different purpose!
This whole semi-ring business basically tells us that we want three different outcomes with
a single model, instead of using three different models, we can use one model, and three
different operators. 


<u> Parsing </u>

If we're convinced that grammars are useful, how do we learn a model, that given an input
sentence $x$, can predict the grammar rules $y$? This problem, or NLP task is known as parsing.


“Parsing” is a form of structure prediction. The goal of parsing, is to be able to assign
structure to the input sentence of interest. What structure to assign depends on the formalism
we are looking at. For e.g, CFGs are a formalism that we declared, by itself it doesn’t tell us
how the parse tree for sentences should be computed. We can disambiguate structures by finding
the most likely parse with probabilistic CFGs. 

Parsing is essentially a search problem, with the space of all possible trees defined by the
grammar. There are multiple combinations of rules (paths) that assign a structure to our
sentence, which is the highest probability combination (shortest path)? Under the notion of
search, the strategies of ‘top-down’, ‘bottom-up’, ‘brute-force’, pruning, and prioritisation
result in a whole myriad of algorithms for parsing. 

While there are many parsing algorithms with various strategies, the central technique used in
the various parsing algorithms is dynamic programming. It prevents us from re-computing
solutions to that which we have already previously computed. 


<br>

##### **Acknowledgements**
Jason Eisner for teaching me everything I know about classical NLP and more. 

**Disclaimer:** This barely scratches the surface of classical NLP, and there are alot more about specific inference and approximation methods, but since we're in the era of deep learning where gradient descent is king, I've chosen to note down more on structure and models rather than learning algorithms. 
