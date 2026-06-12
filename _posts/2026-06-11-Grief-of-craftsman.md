---
layout: post
title: "Grief of a Craftsman"
date: 2026-06-11
mathjax: true
status: [Work experience]
categories: [Work Experiences]
---

Claude calls this my Grief. GPT titled this post. 

No other AI perspectives, content, or restructuring were taken into account or used in any of the writing below. 

---
**Intro**

I’m at the cross-roads of this blog now where I'm forced to reevaluate what I write about, because I'm forced to re-evaluate what I think about. 

Before AI, I would spend days reading textbooks, reading the math, working out the implementations and ironing out all the details. I have started several half posts but feel demotivated by the idea that other than working through it myself, the AI could generate all of the content and better, which makes the  hand-rolled content kind of meaningless to put out. 

This is my most [memorable post](https://suzyahyah.github.io/bayesian%20inference/machine%20learning/2019/03/20/CAVI.html), the entire week of thanksgiving, 5 hours a day for 5(7?) days at a cafe in Baltimore in the first year of my PhD and I referenced many materials and lectures to produce this. The motivation was how professor style lectures or PDFs are overly theoretical, sometimes either incomplete or overly verbose, and to actually gather all the pieces, fill in all the missing gaps, and understand something deeply enough, in order to translate the theory to code by hand. 

It seems trivial to do all of this with AI now. It feels like doing such things are only for fun (and it’s not even that fun because the act of information gathering, processing and consolidation is done by AI), and the practice of doing it throws me into a professional existential crisis.

**Craft vs ShipIt**

I’ve always known that I’m more of a craftsman rather than LetsJustShipIt kind of person. I like working on technical problems and I like abstract problems. I never feel proud of hacking a solution quickly or finding shortcut ways to do things, or simply releasing an application into the world. I only feel excited about doing “difficult” or “principled” work, or making very thoughtful and calculated trade-offs for product delivery. 

But the value of being a craftsman in software engineering is diminishing. While it is very painful for people like myself who invested more than 10 years (others 20) into the craft [because we love it so much](https://x.com/suzyahyah/status/1911271829249073387), 6 months since Claude4.6 is enough to make me sit down and write this. All of the industry is heading towards the working model where the evaluations, verifications, customer satisfaction justifies the means. 



**On code maintainability and technical debt**

There are a few old guards who still care about code quality and maintainability and readability. But if humans will not read code anymore, then the standards that were set up for humans to collaborate on a piece of software no longer hold. Its a movement that no one can stop as it's accelerated by industry competition to ship.

One of the things many SWEs who haven't been in startups dont realise, is that the business would want us to ship worse quality code, if it ships faster and better than the competitors. The cost of market share / market capture sometimes outweighs technical debt even before AI. 

The business just don't want things breaking in production, but quality of code is not and has never been the metric. Technical debt is only a problem because it relates to the speed (cost) of fixing bugs and adapting software to new features. But After AI, the cost of making changes dramatically reduces, and the balance shifts even more in favor of shipping "poor quality code" (by human readability and old software engineering maintainability definitions), and technical debt may soon be an outdated concept. If the AI bungles on the new features, tear it all down and build it again with the new requirements. 

If AI reaches the point where it doesn't bungle itself up, and every company is willing to forgo traditional software engineering principles in the interest of “speed of execution”. There will come a point where engineers (except maybe the originator) can no longer read the code and future generations regress to no longer have any meaningful technical steering direction.

**Code Efficiency/Performance**

Aside from maintainability, engineers also care about Efficiency/Performance. These code problems can't be caught by failing a test because the output is correct. It only shows up at scale, and tests are usualy small scale checking correct I/O of the function. For instance (trivial example), the following (Claude code implemented) is $O(n^2)$, where $n$ is the length of the `series_ids`.

{% highlight python %}
splits_by_id = {
    sid: list(rolling_origin(df[df["unique_id"] == sid], horizon, n_folds))
    for sid in series_ids
}
{% endhighlight %}

The correct $O(n)$ implementation should be using `groupby` which internally builds a hash table based on one-pass through the dataset.

{% highlight python %}

splits_by_id = {
    sid: list(rolling_origin(group, horizon, n_folds))
    for sid, group in df.groupby("unique_id", sort=False)
}
{% endhighlight %}


It is however, possible that function profiling and then directing advanced versions of AI to address targeted areas, will resolve software performance issues.




**What's left for Craftsmen**

A typical response to the above is “you should transition to Architect”, but can’t (future) AI be the Architect? Give it the 20x load tests, the operational constraints, the cost budget, acceptance criteria, compliance, evaluations and even historical incidents. Would it not generate the architecture with a more business-satisfying speed to correctness trade-off for the software to ship? 

Another typical response is "you should decide what problems to solve". Perhaps, but product people, and engineering managers closer to business lines are better positioned to do that. Business acumen and product sense is a rarer commodity than the technical skills when AI is increasingly advanced, because software can be a closed loop optimisation problem on any given metrics. 

I think what's left for us is working in the space of undefined / unclear business requirements, where certain context picked up in human interactions with other humans, or in the physical space does not (yet) get fed into AI. The other major area is scaffolding the glue between other large scale systems. For instance scaffolding production to evaluation pipelines and providing strong priors on non-standard ways of deploying, or wiring up software within a company's proprietary tech stack. If we fully embraced the idea of closed loop optimisation, most of our time would be spent in gathering optimisable / verifiable metrics. 


**Maybe the Skills.md**

The highest probability generated sequence from AI is not the best sequence for our use-case.

Unless our use-case is a textbook example of an application development where perhaps best practices exist on the internet, it's likely to be a lower probability sequence.

The usual recommendation is to put our human judgement and steering into a `skills.md`, which we can load up with different agents for different tasks. Any disatisfaction we have with the default agent behavior can be steered. For instance mine contains stuff like:

> Always prefer tried and tested library implementations rather than reimplementing methods yourself.


But after we've put all this down, what is there left to do? 


I think that as humans we should treat the skills.md file differently from the prompt file, in that it as an extension of our own understanding. Every day, every week, something new should be added that is based on our own initiative, and not merely by constructing self-learning loops. We should get better at the meta-level thinking of how to do something cleaner, better, and more efficient.

**A non-conclusion**

It’s June 2026 and if it sounds overly pessimistic right now, consider that we are nowhere near the ceiling of AI software engineering capabilities.

Maybe the true costs of AI Generated software or research will be transferred to consumers and this trend might reverse or settle on a different equilibrium. Maybe a security incident costing human lives will set this trend back completely. Although companies recover from commercial incidents, political votes don't. 

In any case, I wrote this to try and come to terms with "my grief" (the loss of the craft) and to accept a potentially different reality, contrary to my professional taste and identity.


As always, this blog is 100% human generated unless otherwise stated.
