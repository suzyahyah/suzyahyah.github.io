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
    
    <h3 class="category-head">{{ category_name }}</h3>
    <a name="{{ category_name | slugize }}"></a>
    {% for post in site.categories[category_name] %}
      {% assign date_format = site.minima.date_format | default: "%b, %Y" %}
      <a class="category-post-link" padding-left="20px" href="{{ site.baseurl }}{{ post.url }}">{{post.date | date: "%Y"}} &nbsp; {{post.title}}</a>
    {% endfor %}
  </div>
{% endfor %}
</div>
</body>
