---
layout: default
title: Categories
permalink: /categories
---

<!--<head>
<link rel="stylesheet" type="text/css" href="{{ site.styles }}boot.css"/>
</head> 
 -->

<body>

<div id="archives">
{% for category in site.categories %}
  <div class="archive-group">
    {% capture category_name %}{{ category | first }}{% endcapture %}
    <div id="#{{ category_name }}"></div>
    <p></p>
    
    <h3 class="category-head" id="{{ category_name | slugify }}">{{ category_name }} <span class="category-count">{{ site.categories[category_name].size }}</span></h3>
    {% for post in site.categories[category_name] %}
      {% assign date_format = site.minima.date_format | default: "%b, %Y" %}
      <a class="category-post-link" href="{{ site.baseurl }}{{ post.url }}"><span class="category-post-year">{{ post.date | date: "%Y" }}</span>{{ post.title }}</a>
    {% endfor %}
  </div>
{% endfor %}
</div>
</body>
