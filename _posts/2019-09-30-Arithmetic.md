---
layout: post
title: "Arithmetic(Book)"
date: "2019-09-30"
mathjax: true
status: [Under construction, Book Review]
categories: [Review]
---

My labmate Mitchell Gordon recommended (and lent) me a book on Arithmetic. If you don't want
spoilers STOP RIGHT NOW. 

Here are a couple of rad things:

1. **The Hindus invented '0' - nothing, so that they could avoid placing gridlines between their
   numbers.** i.e., |1|9|5| |4, is now 19504. So what started out as a symbol for pure laziness
found its way into other areas of mathematics such as the empty set in Set Theory, "False" in
propositional logic, and also appearing in our binary system. Somewhere down the line, '0'
stopped being nothing, any CS students knows that "null" value $\neq 0$ but we'll learn that
some other time. 

Well this kind of also explains why the Chinese character for "zero",  &#x96F6;, seems more like
an afterthought rather than baked into the writing system. Since Chinese used characters like
&#30334;&#21315;&#19975; to distinguish between hundred, thousand, and ten thousands, they didnt use this mysterious zero term for
basic arithmetic.

2. **Our metrics for time dates all the way back to the Babylonians (1800 BC) because there were
   roughly 12 clusters of stars in the Egyptian night sky.**. Considering that we managed to
change the metric systems by standardising stuff to units of ten (money, lengths, weights etc), how is it that we never shook off units of 60 from 1800 BC?! I hope to be enlightened someday.

3. **"The trouble with familiarity is not so much that it breeds contempt, but that it breeds
   loss of perspective."** - probably the most quotable sentence in the book. What's $7 \times
8$ and what's $7 + 8$? We probably can recall the answers based on pure memorisation as we had
been trained in high school. However the spirit of finding the "answer", is to convert this into
the units that modern people work with (units being denominations of 10) so that we can do some
kind of comparison. Why count otherwise? So now $7+8=15$, is actually $3$ units shifting over
to $7$ to make $10$, and having $5$ leftover. And $7 \times 8=56$ is $7$ sets of $8$, you can
imagine a matrix of $7 \times 8$ elements. Being converted to a system based on $5$ sets of
$10$, and $6$ leftover.

Recently Xie Jiamin and I worked through converting decimal values into binary system for Shannon's
coding for data compression, and
[procedure
aside](https://indepth.dev/the-simple-math-behind-decimal-binary-conversion-algorithms/) along
the way we realised that by converting a system based on denominations of 10 to a system based on denominations of $$log_2$$ it is possible to prove that the smallest number of bits required to encode a difference $p$ is $log_2(1/p)$.

#### References ####
[Arithmetic by Paul Lockhart](https://www.amazon.com/Arithmetic-Paul-Lockhart/dp/0674972236)\\
[Shannon Codes](https://en.wikipedia.org/wiki/Shannon_coding)
