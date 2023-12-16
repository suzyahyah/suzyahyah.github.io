--- 
layout: post
title: "A modern LLM research landscape"
date: "2023-12-05"
mathjax: true
status: [NLP]
categories: []
---

The papers from this year's major NLP/ML conferences (ACL, EMNLP, Neurips, ICML) indicate that community has pretty much accepted that Large Language Models (LLM) are here to stay. The paradigm has clearly shifted from one that is task or methods centric, to one that is model centric. One might wonder what the "NLP" research areas in this new paradigm are. The chart below is my current interpretation of the landscape in 2023. 

<u>Training and Inference Pipelines</u>
 
Since we have shifted to a pre-trained model paradigm, we can think of LLM work along two main pipelines, 
* **Improving Inference** with the existing pretrained models.
* **Further training**, typically lightweight approaches. 

I also consider a separate category of things as "add-ons" which are related to both training and inference, but are mostly about making LLMs play-well with all sorts of data (tables), existing knowledge (graphs), and of course generation (fairness and other things). 

<u>Components of the standard Pipeline</u>

Within the training and inference pipelines, we often see work around Data, Context, Model, Controllable Generation, Prediction Tasks and Evaluation. These are the columns in the chart below.

<div id="image-container">
    <a href="{{ site.baseurl }}/assets/LLMResearchLandscape.png" target="_blank" id="zoomable-link">
        <img src="{{ site.baseurl }}/assets/LLMResearchLandscape.png" alt="Zoomable Image">
    </a>
</div>

<br>

The above picture is free to [Download here](/assets/LLMResearchLandscape.pptx) as a pptx that you can modify as you please. I would be grateful if you would be willing to attribute it to this page when using, or send me a note if you have any thoughts while modifying. 





