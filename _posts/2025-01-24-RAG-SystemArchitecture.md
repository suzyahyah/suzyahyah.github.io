--- 
layout: post
title: "RAG System Architecture"
date: "2025-01-24"
mathjax: true
status: [Architecture]
categories: [Projects]
---

How I think about RAG these days. 

The most exciting thing to me is the self-improving RAG System with strong LLMs to do Conversational Analysis giving labels to conversations which achieved high user satisfaction. This means other than designing for user feedback explicitly, we have an automatic post-hoc analysis module that provides a reward signal to train the models (with RLHF). 

However if naively implemented by LLM chaining, even if the performance is one day good and consistent enough, the latency would be completely unacceptable. Hence we would still need to invest in optimisation of the pipeline by some form of model compression.

<div id="image-container">
    <a href="{{ site.baseurl }}/assets/RAG.png" target="_blank" id="zoomable-link">
        <img src="{{ site.baseurl }}/assets/RAG.png" alt="Zoomable Image">
    </a>
</div>


<br>
