---
layout: post
title: "A classical NLP researcher and a GPT-era Engineer meet at the canteen"
date: "2024-02-16"
mathjax: true
status: [Misc]
categories: [Misc]
---


<span style="color:blue">
Hi! Is this seat taken? Mind if I join you?
</span>
<br>

No, not at all! Please do!

<span style="color:blue">
So... I've been wondering.. about the Transformers breakthrough?
</span>
<br>


What? Oh oh right. I'd say that the key was the self-attention mechanism. In 2015 (which is not classical times but its an interesting question so let's go into it), Deep Learning was already
popular for NLP. In those days we were using RNN-LSTMs for modeling sequential data which could
encode infinite length sequences and naturally encodes the idea of recency and compositionality. But people
couldn’t really figure out how to do tasks like translation or what we call “sequence to
sequence” tasks better than statistical phrase-based translation methods.  RNNs had a problem with longer sequences, because of the recurrence the model tends to remember the more recent inputs. (LSTMs help to alleviate some of this, but not fully.) 

So the insight that the inventors had was, why dont we let the new hidden state of an RNN in the
decoder additionally be a function of the input sequence hidden states. 

<span style="color:blue">
Wait, you lost me.  A RNN would take in the input string, x1, x2, x3, encode it into a single
hidden state, which we call “encoder vector” here, and then decode that hidden state into y1,
y2 to generate the translation.
</span>
<br>

Right, this effectively means the encoded vector has to condense all the information from the source
sentence! The model has no problem with this 10 or 20 word sentence, but when we need to have
information from a long context, it's a problem.


<span style="color:blue">
I see, the RNN decoder generates the output y which depends only on the hidden state of the decoder. This seems
like an unnatural constraint, because different context words should 
have different importance, when deciding what is the next word to predict. I dont just want to
see a single context vector, I also want to see words from the source sentence. 
</span>
<br>

Also that the same weight matrix is being reused, this makes it very difficult to learn such
a weight matrix that can both be reusable, and condense all that information into one vector.  
The attention mechanism makes this explicit for the RNN. “Attention mechanism” learns to pay
attention to different parts of the input sentence. Such that each time the model generates
a new token, it looks back at the set of source tokens, and takes a weighted average of the
source token representations. 

<span style="color:blue">
So attention was introduced to solve the problem of long range dependencies. But it's weird.
now every decoder state is connected to the input state, yet we have this hidden recurrent
state thing going on at every time step. Seems kind of redundant when everything is already
connected..
</span>
<br>


Additionally, there is another problem with sequential recurrent models, which is that
recurrence prevents parallel computation. Because of this unrolling of the RNN, you always need
to wait for the previous input to be seen before the next token can be fed to the model. This
is a fundamental bottleneck during training, and the recurrence actually prevents us from being
able to scale up training to much larger models and larger dataset sizes. Which is why people
say the transformer architecture was motivated from an engineering perspective. 


<span style="color:blue">
Yeah, in the attention is all you need paper, they got rid of recurrence completely, they
introduced 3 new things - self-attention, positional encoding or positional embedding, and
multi-head attention. Although in the decoding stage, you still decode autoregressively, so the
gains are really in cthe encoder training stage. 
</span>
<br>



**The high-level goal of Modeling**

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


<span style="color:blue">
In today's LLMs, we no longer make Markov and assumptions/ are able to take more context into consideration, as we can pack more information into model parameters.
</span>

<br>


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


<span style="color:blue">
The base model architecture hasn't really changed since 2017. There were some improvements in
positional encodings and mixture of experts but it's really just been about data and scale. 
</span>


<span style="color:blue">
But I think it would be really cool if we could like, use more linguistic information. Maybe
like, use Grammars to predict language better..
</span>

<br>

#### **Syntax**

<u>Grammars</u>

A Grammar is not a statistical model for prediction. It is more like a structural model of language. It
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


<span style="color:blue">
... uhm what? Sorry, you lost me at N-grams.
</span>

<br>

##### **Acknowledgements**
Jason Eisner for teaching me everything I know about classical NLP and more. 

**Disclaimer:** This barely scratches the surface of classical NLP, and there are alot more about specific inference and approximation methods, but since we're in the era of deep learning where gradient descent is king, I've chosen to note down more on structure and models rather than learning algorithms. 
