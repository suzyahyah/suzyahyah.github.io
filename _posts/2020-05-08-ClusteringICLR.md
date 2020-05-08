---
layout: post
title: "Some Clustering Papers at ICLR20"
date: "2020-05-08"
mathjax: true
status: [Under construction]
categories: [Machine Learning]
---


##### [UNSUPERVISED CLUSTERING USING PSEUDO-SEMISUPERVISED LEARNING](https://openreview.net/pdf?id=rJlnxkSYPS)

This paper uses an ensemble of (ladder) networks to do voting on each label, constructs a graph
and applies a graph clustering algo. And then feeds the cluster as training data, progressively
labeling the dataset based on consensus. 

The idea seems reasonable but the motivation for ladder networks is still a little bit shaky to
me. "Ladder networks does not require any domain-dependent augmentation, works for both image
and text datasets, and can be easily jointly trained with supervised and unsupervised losses."
This does not say much about the specific ladder network architecture and I would be
  interested to know if it is the only category of models that fulfil the above criteria. 

<u>Background</u>

To use semi-supervised learning to improve unsupervised clustering, the typical approach is to
generate "pseudo labels" (via K-means for e.g), use these pseudo labels to train a supervised
learning model, take the trained supervised model to predict labels on the datapoint again (new
pseudo labels), and retrain until convergence. The iterative procedure is heavily dependent on
the accuracy of the initial pseudo labels. So the authors invest heavily in getting this
initial pseudo label correct. 

<u>Methodology</u>

They use an ensemble of ladder-networks (trained unsupervised) to independently cluster the
input.

1. First feed unlabelled examples to the ladder network.  (Initialization)

2. If the ensemble assigns two datapoints to the same cluster, then we generate a link between
   the two points. Basically two datapoints are considered 'similar' or 'close' if the models
think they belong to the same cluster.

3. From this we can construct a similarity graph (affinity matrix) and apply a graph clustering
   algorithm.

4. What we then end up with, is datapoints which have high consensus get labelled as one
   cluster, and importantly, those that do not have high consensus do not get labelled. 

5. These pseudo labelled examples, and unlabelled examples get fed to the ensemble again and
   trained under a supervised AND unsupervised loss.

   Repeating steps 2-5 till convergence. 

<u>Problems and Tricks</u>

1. The authors say that ladder networks degenerate to output constant values for all inputs in
   the absence of a supervised loss term. (I dont really know what this means) - To avoid
degeneracy, add an unsupervised loss which maximises diversity in the outputs, using (a)
Information Maximization Loss (i.e, maximise Mutual Information $I(X; Y)$, and (b) Dot Product
Loss. (a) basically means maximising $I(X;Y) = H(Y) - H(Y|X)$, where $Y$ are model outputs and $X$ are model inputs. Recall that max $H(Y)$ is achieved for a uniform distribution over $Y$, and minimizing
$H(Y|X)$ is achieved when the direction $f:X\rightarrow Y$ is as deterministic as possible. Low conditional
entropy of $Y$ given that we know $X$. 


2. Graph clustering algorithms have greater than polynomial time complexity. They do some
   approximate greedy algorithm which didn't look too convincing (see the paper) but seems to
be sufficient. Like one of the reviewers, it feels adhoc to me.

<br>

---
<br>

##### [SELF-LABELLING VIA SIMULTANEOUS CLUSTERING AND REPRESENTATION LEARNING](https://openreview.net/pdf?id=Hyx-jyBFPr)

This paper doesn't do clustering in the "traditional" sense, of saying that elements with
similar features should be grouped together. Instead it considers the 'clustering task' by
assigning a label $q(y_i|x_i)$ , and if $q$ is a neural network, then there are non-linearities
built into the 'clustering' function.    

The part about enforcing equipartition constraints, yet still be "theoretically correct" in
maximising the mutual information (see Discussion) is a neat analysis trick

<u>Background</u>

The goal is to simultaneously cluster, and learn representations at the same time.  By
clustering, we mean assigning data points to labels, and minimizing some cross-entropy loss: $ -\frac{1}{N} \sum_{i=1}^N \log p(y_i\|x_i)$

This is the standard loss for supervised learning. If we have any ground truth labels, this
cross-entropy loss is valid, but if fully unsupervised, then this becomes degenerate, because
everything input instance will be assigned to the same label $y$ , and the model will predict
this $y$ for everything, trivially minimising the cross-entropy loss. 

<u>Methodology</u>

1. Introduce a posterior distribution which is multinomial, such that $\forall y: q(y|x_i) \in
   \{0,1\}$

2. Apply a constraint that all datapoints are split equally amongst all the classes:  

   $\forall y: \sum_i^N q(y\|x_i) = \frac{N}{K}$

3. The objective now can be written as:

    $E(p,q) = -\frac{1}{N}\sum_{i=1}^N \sum_{y}^K q(y\|x_i) \log p(y\|x_i)$, s.t. 

    $\forall y: q(y\|x_i) \in \{0,1\}, \sum_i^N q(y\|x_i) = \frac{N}{K} $
   
   

4. Alternate between training the model wrt to the label assignments $q$  (using regular xent
   loss), and finding $q$ which minimizes $E(p,q)$ 

<u>Problems and Tricks</u>

The question is how to optimise $q$. The insight here is that the objective can be rewritten as
a linear program which can be solved in polynomial time. If we let $P\in \mathbb{R}^{K\times
N}$ be the matrix of probabilities $p(y_i|x_i)$  estimated by the model, and let matrix $Q \in
\mathbb{R}^{K\times N}$ be the matrix of the assigned labels $q(y|x_i)$. The constraints on $q$
can be expressed by the following $Q\cdot \mathbf{1}=\frac{1}{K}\cdot \mathbf{1}$, and $Q^T
\mathbf{1} = \frac{1}{N}\cdot \mathbf{1}$ , where $\mathbf{1}$ is a vector of 1s of the appropriate
dimension. The matrix multiplication of $Q\cdot \mathbf{1}$ would "marginalise" out all the
instances, leaving us with equiprobable classes. 

So we need $min <Q, - \log P>$ , where $<.>$ is the Frobenius dot-product (element wise
multiplication). A fast version of this optimisation is the *Sinkhorn-Knopp algorithm* (which the authors provide an implementation for.)

<u>Additional Discussion</u>

Instead of being conditioned on the features $x_i$, the authors reinterpret $p$ and $q$ as
joint distributions between the label and index, $p(y, i)$ and $q(y, i)$. Then, the form can be
written as 

$E(p,q) + logN = -\sum_{i=1}^N \sum_{y=1}^K q(y,i)\log p(y,i) = H(q,p)$, 

where $H(q,p)$ is the cross-entropy between the joint distributions $q(y,i)$ and $p(y,i)$. To
minimise $E(p,q)$, $p=q$ , and $H(q,q)=H_q =H_q(y,i))$ , 

The entropy of the joint distribution can be rewritten as 

$H_q(y) +H_q(i) - I_q(y,i)$  , since $H_q(y)$ and $H_q(i)$ are constants as set by the
constraints, minimizing $E(p,q)$ becomes maximising the mutual information between the label
$y$ and the index $i$. 


Other Clustering Papers 

[SPECTRAL EMBEDDING OF REGULARIZED BLOCK MODELS](https://openreview.net/pdf?id=H1l_0JBYwS) \\
[LEARNING TO LINK](https://openreview.net/pdf?id=S1eRbANtDB)
