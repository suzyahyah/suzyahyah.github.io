---
layout: post
title: "Could Large Language Models be conscious? (David Chalmers @ Neurips 2022)"
date: "2022-11-27"
mathjax: true
status: [Review]
shape: dot
categories: [Misc]
---

What constitutes evidence for or against consciousness? If you foolishly try to enter a debate
about the headline topic with a random Joe you’ll soon get thrown the amateur’s question “well,
how do you define consciousness?” Most of us are not philosophers and can barely define what AI
is, and so the debate quickly fizzles. 

With this Neurips keynote, David Chalmers significantly moves this debate forward (at least for
me). Hopefully this will be the new starting point for meaningful discussion, instead of
getting stuck at the definition of consciousness. Everything which follows is his opinion and
intellectual IP, with some slight effort of reorganising and summarising by me. References not complete.

[Slides](https://nips.cc/media/neurips-2022/Slides/55867.pdf), [Talk](https://nips.cc/virtual/2022/invited-talk/55867) (May be behind conference paywall)

---
<br>
#### **A most welcome definition**

According to Chalmers, consciousness is
*subjective experience.. The feeling of what it is like to be something*[^fn1] A working assumption is that consciousness is real, and is not just an illusion. You could
argue that consciousness doesn’t exist but that’s a separate discussion, we wouldn’t be having
the second order debate of “is X conscious”. 

<br>

#### **Well is it measurable?**

Herein lies the problem, how do you measure or design an evaluation for something that is
*subjective* instead of *objective*? 


Consciousness is not intelligence (sophisticated) or goal directed behavior and consciousness
is not the same as human level intelligence. Many non-human animals are conscious, and there is
also the famous thought experiment of a [philosophical zombie](https://en.wikipedia.org/wiki/Philosophical_zombie), something that acts intelligently based on input and
output, but has no inner experience. In humans verbal reports are used as indication or
evidence of consciousness. However because consciousness is not based on "performance", there’s no one accepted measure of consciousness.
<br>

#### **So then what do we do..**

Chalmers spares us from being the dissatisfaction of the thus far handwavy definition, by
breakdown subjective experience into some dimensions that are more amenable to discussion.
Dimensions of consciousness include
* Sensory (the experience of seeing colors not just seeing a color)
* Affective (feeling pain)
* Cognitive experience (thinking hard)
* Agentive experience (deciding to act)
* Self-consciousness (awareness of oneself). 

<br>

#### **A Framework for arguing for consciousness**

LLMs have X. If a system has X it is likely to be conscious. 

**X = self.report**

There was a famous case of Google’s employees claiming that the Lamda model
   was conscious, and this was primarily based on the [model’s self-reporting](https://theconversation.com/is-googles-lamda-conscious-a-philosophers-view-184987) that it was "sentient". Self-reporting is not a very strong claim for consciousness because firstly models trained on such data can be good at mimicking. Second, even for humans, there is no way to prove to others that we are not in fact a philosophical zombie, what more an entity with no biological brain based on self-reporting? But perhaps the strongest argument against this is that the model can be prompted into the reverse. I.e., If you start with telling the model that it is NOT sentient, it gives a completely different answer arguing that it is not.

**X = conversational abilities/seem sentient**

 This is the most compelling for individuals but
   also the least for philosophers or scientists. Humans always make the mistake of thinking
something is sentient or personifying something when it is not. Even in early days chatbots
like Eliza. Hence this is not a strong argument for consciousness. 

#### **A framework for arguing against consciousness**

LLMs do not have X. If a system does not have X it is likely to be unconscious. 

**X = biology,senses**

With no biological bodies or sensory processing, a virtual entity can’t act in the real world.
They have no symbolic grounding. But a system with no senses and no body could still be
conscious even if it lacks the sensory dimension of consciousness ( a brain in a vat). Does
consciousness require physical biology? It’s highly contentious. Furthermore, agents acting and
sensing in virtual reality blurs the definition of “acting”. Some may argue that virtual
reality is just as real an alternate reality as the one we currently live in. 

**X = world-models and self-models**

LLMs just minimize text prediction error. They don’t have a genuine model of the world. It’s
true that LLM are trained to minimize prediction error, but their processing isn’t just string
matching and can give rise to new processes. An analogy is that maximizing fitness in evolution
leads to very novel processes. It’s plausible that being able to minimize prediction error
would require deep models of the world. The question is has this happened already? Many recent
probing experiments have shown evidence for world models[^fn2], but less so for self-models. 

**X = self.models and unified agency**

LLMs tend to be fickle or take on different personas based
   on the input prompt, but there’s also an argument that the space of parameters is large
enough to maintain several consistent self-models that get interchanged based on the prompt.
Also its highly conceivable to distill or make separate models which are consistent with
a particular persona. We know how to finetune a model for specific tasks, why wouldnt we be
able to finetune a model towards a consistent persona?

**X = recurrent processing and a global workspace**

Current Transformer LLMS “have quasi-memory
   and quasirecurrence by using recirculated outputs and a long window of inputs.” Although
they may lack true recurrent processing and a global workspace, we certainly already know how
to construct architectures that do have these features.[^fn3] They just haven;t been scaled up or
seen as much popularity as GPT, if only Google was as good as marketing as OpenAI. Also
virtually no researcher will tell you that the GPT architecture is the final AGI architecture.

---
<br>

#### **Conclusion**

So really, there are arguments for and against, the arguments for are still highly debatable,
and the arguments against are not eternally unshakeable. According to Chalmer’s concluding
slide, “None of the reasons for denying consciousness in current LLMs are conclusive, but some
are reasonably strong.  These reasons together might yield low credence in current LLM
sentience: <10%”. 

It’s very conceivable that in the next 10-20 years, a model would emerge checking off the
against-arguments, and be compelling wrt to the for-arguments. Without going full bayesian into
the estimates, 50-50 we design models with these prerequisites, and from there 50-50
consciousness might emerge from them. 

 

#### References
[^fn1]: Thomas Nagel, 1974. "What is it like to be a bat?"
[^fn2]: Hobbhahn, Lieberum, Seiler. [Investigating causal understanding in LLMs.](https://openreview.net/pdf?id=st6jtGdW8Ke)
[^fn3]: Juliani, Kanai, Sasai. [The perceiver architecture is a functional global workspace](https://escholarship.org/content/qt2g55b9xx/qt2g55b9xx_noSplash_c60e72d5eabbaae941e9ab65b3459676.pdf)

