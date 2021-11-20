---
layout: post
title: "Formalising Analogies for A.I"
date: "2021-11-18"
mathjax: true
status: [Review]
categories: [Machine Learning]
---

What is an analogy (and why is it so hard for us to formalise and use in A.I?). This post is
based of a very fun paper I presented at Reading Group, Analogical Proportions: Why They Are
Useful in AI.

---
<br>

If you asked me to define an analogy, I would start with an example of one. One of my
favorite analogies compares electricity to water. “Water is to a reservoir as electricity is to
a [?]”. This lovely example shows that the similarities between “water” and “electricity” are
pretty intangible and only with respect to the “storage” of these quantities can we arrive at
a correct/plausible answer.

If I was pushed for a concrete definition, I might try to trace the mental steps my brain takes
when doing analogical reasoning. It feels like a two step process, one might first recognise
that the new problem instance is similar to old instances in some aspects, and then apply the relationship that was previously used to solve the old problems. 

#### **English Definitions**

I still can't define it, so let's look at the English dictionary.

> “a comparison of two things based on their being alike in some way” - Merriam Webster
<br>
> “the act of comparing two things that are alike in some way” - Merrian Webster
<br>
> “a comparison between things that have similar features, often used to help explain a principle
or idea” - Cambridge Dictionary. 

These definitions of analogy come from the practical perspective of human communication (the Cambridge one being the most obvious). When trying to communicate a new concept rapidly to a listener, we might use analogies to help the receiver understand the “idea” that we are trying to convey. 

However as we can see from the “water”, “resorvoir” example, there’s more profoundness to
analogies than just “similarity” and “alikeness”. Gentner’s (1983) **“Structure Mapping theory”** provides a more nuanced view. She distinguishes analogical reasoning from mere similarity. Previously it
was postulated by Tversky that an analogy is stronger if the two objects have more attributes
in common. Gentner emphasised that it is relationships and not attributes that are prominent in
how humans understand analogy.  Central to the mapping process is the principle of
systematicity: people prefer to map systems of predicates that contain higher-order relations,
rather than to map isolated predicates, which ties in with people’s inherent preference for
coherence. 

<br>
#### **The promise and the problem**
So we know Analogies have to do with "higher-order relations", some notion of "similarity" and
problem-solving in new-domain. Sounds very promising very Transfer Learning, Domain Adaptation
or Creativity in Computer Science. Intuitively we know analogies are useful for reasoning, but
they are actually really hard to define in a way that is computationally useful. Consider at
a very high level the range of computational tools available to us. If you define “being alike”
and “similar” with discrete logic, then analogical reasoning becomes very brittle, it either
is, or it isn’t. If you apply real-valued operations, then we need to first be able to assign
real-values to the discrete items in the world we care about, and also deal with the
computational challenge that there are an infinite number of relations that can connect four
objects.

### **Modeling Analogies**
In order to reason about Analogies computationally, we should define this formally. We deal
with "Analogical Proportions (AP", which are statements of the form "a is to B, as C is to D".
This is denoted with the shorthand: $$a:b :: c:d$$. An example would be "cat is to kitten as
dog is to puppy". 

Given a set of items, $$X$$, an AP is a quaternary relation which obeys the following
relationships $$\forall a, b, c, d \in X$$:

1. $$a:b :: a:b$$ (reflexivity)
2. $$c:d :: a:b$$ (symmmetry)
3. $$a:c :: d:b$$ (central permutation, also called exchange of the means)

If we suscribe to these axioms, then the following corollaries apply:  $$(a:a :: b:b), (b:a ::
d:c), (d:b :: c:a), (d:c :: b:a)$$.


At this point, we should ask, where did these "axioms" come from? And why are they called
proportions? It is believed that this formalism dates back to Aristotle (300 B.C), who observed
**Arithmetic** ($$a-b=c-d$$) and **Geometric** ($$\frac{a}{b}=\frac{c}{d}$$) Proportions. Note
that the relationship between Arithmetic and Geometric proportion is the log function. 

<u> But math is not Language (or common sense)! </u>

If we were modeling common-sense analogies, the validity of the central permutation axiom is 
called into question. Consider the following analogy: "Carlsen is to chess, as Mozart is to
Music", applying the central permuation axiom, we get "Carlsen is to Mozart, as Chess is to
Music". This no longer makes sense because the permutation breaks a mapping across TWO categories. If we had only one category, for e.g, "Cat is to kitten, as Dog is to puppy", then "Cat is to Dog, as Kitten is to Puppy" is still a valid analogy.

Something to keep in mind whenever we consider real-world applications.

#### **Formalisms for Analogical Proportions**
There are four ways Analogical Proportions have been formalised in the literature. Namely,

0. Arithmetic, Geometric
1. Logical 
2. Algebraic 
3. Transformation Complexity 
4. Functional 

**Arithmetic, Geometric View**

If we have vectors of feature values, $$\{a,b,c,d\} \in \mathcal{R}^d$$, then a way to complete the
analogy, $$(a:b :: c:?)$$, is to simply say $$a-b \sim = c-d$$. We use $$\sim =$$ here because we don’t have an exact “match” to a discrete token in vector space. While popularised by word2vec in 2013, this idea had already been suggested by Rumelhart in the 1970s. Note that the parallelogram view of
analogical proportions, is simply a pointwise extension of the arithmetical proportion. 

Given $$a, b, c \in \mathbb{R}^m$$, the goal is to find $$d$$ which completes the analogy.
Naively we can find $$d^* = \mathrm{argmax}_d cos(a-b, c-d)$$, or equivalently in Math but not
in vector space: $$d^* = \mathrm{argmax}_d cos(d, c-a+b)$$. From the Geometric View, Levy
& Goldberg (2014) propose to use $$d^* = \mathrm{argmax}_d \frac{d\cdot b \times d \cdot c}{d
\cdot a}$$

Since inference has been relatively straightforward for the Arithmetic and Geometric view, the challenge has been to find or learn the embedding space, where analogies hold by simple arithmetic. For this reason, the ability to do Analogy completion out of the box (by this we mean the embeddings were not explicitly trained for Analogies but were trained by unsupervised methods like Continuous Bag-of-words), is used as a quality check for learned embeddings.

It's still a curiosity to me why analogies should fall out by a simple linear relationship, and
the paper by Allen & Hospedales (2019), argue that probabilistic paraphrasing gives rise
to linear relationships that factorise PMI. Nice. 

But other than this nice linguistic phenomena that appeared as a by-product of cBOW prediction, which we nowadays take as a measure of word embedding quality, can we explicitly make use of analogies in training models? 

<u>Analogies as Constraints in Object Recognition</u>

Analogy-preserving Semantic Embeddings (Hwang et al., 2013) apply a regularizer $$R$$, to
candidate analogies $$(a, b, c, d)$$, where $$u_a \in \mathbb{R}^d$$ denotes the embedding of
object $$a$$. They consider the shift $$u_b - u_a$$ as the difference between the two objects
$$a$$ and $$b$$ in semantic space. This implies the inequalities: $$u_b-u_a = u_d - u_c$$ and
$$u_c - u_a = u_d - u_b$$. 

\begin{equation}
R = \frac{1}{\sigma_1} \mid\mid (u_b-u_a) - (u_d-u_c) \mid \mid_2^2 + \frac{1}{\sigma_2} \mid \mid (u_c-u_a) - (u_d-u_b) \mid \mid_2^2
\end{equation}

where $$\sigma_1$$ and $$\sigma_2$$ are scaling constants estimated as the mean distances
between data instances from different classes.

<br>
**Logic View**

The formal logic view starts from the expression “a differs from b as c differs from d”, and takes this notion very strictly (applied in a world of 1s and 0s). Formally, an analogical Proportion (AP) is a quarternary propositional logical connective:

$$
\begin{align}
a:b :: c:d &=  a \neq b \Leftrightarrow c \neq d \\
&= ((a \wedge \neg b) \equiv (c \wedge \neg d)) \wedge ((\neg a \wedge b) \equiv
(\neg c \wedge d))
\end{align}  
$$


From this and the postulates of analogy, we can obtain the following 6 rows in the truth table
which evaluate to True for Analogies. (Note that there are $$2^4$$ possible rows in this Truth
Table. Note that this is code independent, 1s and 0s mean the same thing. Hence the straightforward extension for nominal categories (representing membership with $$\square$$ and $$\bigcirc$$) where we would have the three rows as below.

|a|b|c|d|,|a|b|c|d|
|-|-|-|-|-|-|-|-|-|
|0|0|0|0|,|1|1|1|1| -- $$(\square,\square,\square,\square)$$
|0|0|1|1|,|1|1|0|0| -- $$(\square,\square,\bigcirc,\bigcirc)$$
|1|0|1|0|,|0|1|0|1| -- $$(\square,\bigcirc,\square,\bigcirc)$$

<br>
<u> Real-Valued Logic</u>

If we were to consider real-valued logic where $$a,b,c,d \in [0,1]$$, then we can talk about
the *degree* to which the AP is true. To evaluate this in terms of fuzzy logic we apply
$$\mid . \mid$$ for "$$\equiv$$", $$\mathrm{min}$$ in place of "$$\wedge$$", and
$$\mathrm{max}$$ in place of "$$\vee$$". Then the propositional logic formula evaluates to:

$$
\begin{align}
&((a \wedge b) \equiv (b \wedge c)) \wedge ((a \vee d) \equiv (b \vee c)) \\
&= min( \mid min(a,b) - min(b,c)\mid, \mid max(a,d) - max(b,c) \mid )
\end{align}
$$. 

Note that the representation is similar to the modern Arithmetic View with embeddings, it just
applies a different formula (fuzzy logic) to evaluate the degree of AP. I would be curious as
to how this formula compares against the Arithmetic and Geometric versions for real-valued vectors.

<br>

**Algebraic View**

A more generalised view of AP from abstract algebra is given by Yvon and Stroppa (2006), based
on the notion of factorization of items, when the set of items is a *commutative semi-group*.
Recall that for semi-groups, the membership of the set is closed under the operation, i.e, after applying the operation you still get a member of the set, and the operation is associative. Commutative is with respect to the operation $$\oplus$$. A quadruple $$(a, b, c, d)$$ is an AP if 

1) $$(b, c) \in {(a, d), (d, a)}$$ or

2) $$\exists (a, b, c, d)$$ such that $$a = x_1 \oplus x_2, b = x_1 \oplus y_2, c=x_2 \oplus
y_1, d = y_1 \oplus y_2$$.

As a simple example of factorization, consider $$X \in \mathcal{Z}^+$$ and $$\oplus
= "\times"$$. If we have $$x_1=2, x_2=3, y_1=5, y_2=7$$, then we can solve for the following
analogy: $$6:14 :: 15:?$$.

This view is pretty powerful, because using factorization, we can define analogical proportions
between strings, lattices, trees, i.e. higher order linguistic structures. Also I think that this strictly generalises real-valued vectors in the logical view, because we could treat the factorization as a concatenation of each of the dimensions of the real-valued vectors.

<br>

**Complexity View**

An analogy $$(a:b :: c:d)$$ perceives some aspects of the structures of two situations (a and
c) as similar, and applies the strategy used to transform a to b, onto c. However if we depart
from the logical view which is overly brittle, the challenge of analogy completion is finding
what the "correct transformation" should be. The complexity view assumes that similarity and
the "transformation" should be measured in terms of Kolmogorov Complexity. Intuitively, if the
size of the minimal program pr(a, b) able to transform a into b is the same as the program is
pr(c,d) then they are analogous.  Then to solve for an analogy, we find 

\begin{equation}
d^* = \mathrm{argmin}_d K_u(a:b :: c:d)
\end{equation}

What does $$K_u(a:b :: c:d)$$ even mean? Recall that the Kolomogorov complexity of a string 
$$x$$ under a Universal Turing Machine $$u$$, denoted $$K_u(x) = \mathrm{min}_{pr} \{len(pr) | u(pr)=x\}$$ is the "length" of the shortest program executed by the $$u$$ that generates the string $$x$$. E.g, the $$K_u(x)$$, where $$x= "010101010101"$$ has very low complexity, and a string $$x="0011100100100"$$ has high complexity.

For the string $$a:b :: c:d$$, we need to define (1) a program for converting $$a \rightarrow
b$$, (2) a program for converting $$a \rightarrow c$$. Murena (2020) defined
operations/instructions for programs and assigned each of the instructions bit strings. They found that trying to search for the program with brute
force performed very poorly, and they required a heuristic of first finding the shortest
program for $$a$$, then for $$c$$, then for $$a\rightarrow b$$, and then applying that to
$$c$$.


<br>

**Functional View**

Finally we have the functional view, which is the most flexible but also the least developed.
A prototypical AP is $$a:f(a) :: c:f(c)$$, here $$f(a)=b$$ and $$f(c)=d$$. The flexibility of
the functional view is in composing function operations $$a: f(g(a)) :: c: f(g(c))$$, or in
generating a string of analogies from a single initial item, i.e., $$a: g(a) :: f(a):f(g(a))$$. One problem with this is **recognising** and **applying** the function $$f$$. If we had a database of all valid function operations that we cared about, that one could solve the analogy task as in the pioneering COPYCAT model (Hofstadter and Mitchell, 1995). However the beauty or difficulty of analogies is that it is very difficult to define $$f$$ ahead of time in a non-trivial setting (if we went beyond string operations), let alone recognise $$f$$. One thing that would help with discovering $$f$$ is a sequence of analogical proportions so that a model could possibly find recurrent similarity relationships. 


<br>

<u>Other Comments</u>

When I presented this survey at reading group, Professor Jason Eisner pointed out there was an
elephant in the room that none of the functional, complexity or geometric methods addressed,
namely that given the triplet of $$(a, b, c)$$, they would always return a $$d$$ but not a null
for when the analogy is considered invalid. I contended there by formal definition, there was no such thing as an invalid analogy, just an unintuitive one to humans. 


#### References
[Allen & Hospedales, 2019. Analogies Explained: Towards Understanding Word Embeddings](http://proceedings.mlr.press/v97/allen19a.html)\\
[Cornejulos, 1996. Analogy as minimum description length](http://w2.agroparistech.fr/ufr-info/membres/cornuejols/Papers/PUBLIES/1996-analogy-chap.pdf) \\
[Gentner, 1983. Structure Mapping Theory.](https://groups.psych.northwestern.edu/gentner/papers/Gentner83.2b.pdf)\\
[Hofdstadter & Mitchell, 1995. The Copycat project](https://dl.acm.org/doi/10.5555/218753.218767)\\
[Hwang et al., 2013. Analogy preserving semantic embeddings.](https://w.cs.utexas.edu/~grauman/papers/analogies-icml2013.pdf)\\
[Levy & Goldberg, 2014. Linguistic Regularities in Sparse and Explicit Word Representations](http://acl2014.org/acl2014/W14-16/pdf/W14-1618.pdf)
[Miclet & Prade, 2009. Handling Analogical Proportions in Classical Logic and Fuzzy Logics Settings.](https://link.springer.com/content/pdf/10.1007/978-3-642-02906-6_55.pdf)\\
[Murena et al., 2020. Solving analogies on words based on minimal complexity transformation](https://w.ijcai.org/Proceedings/2020/0256.pdf)\\
[Prade & Richard 2001. Analogical Proportions: Why They Are Useful in AI](https://w.ijcai.org/proceedings/2021/0621.pdf)\\
[Stroppa & Yvonn, 2006. Analogical Learning and Formal Proportions](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.103.7953&rep=rep1&type=pdf)\\
