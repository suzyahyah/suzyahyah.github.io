---
layout: post
title: "Migrating from python 2.7 to python 3 (and maintaining compatibility)"
date: 2018-05-10
mathjax: true
status: Personal experience
categories: [Projects, Work Experiences]
---

We wish to maintain a single-source codebase that is Python 2/3 compatible. Modules such as [futurize](http://python-future.org/automatic_conversion.html) and [modernize](https://python-modernize.readthedocs.org/en/latest/) are helpful but moving forward, we want to write futurized and compatible code.

**Basic functionalities**
<br>
* Include this in the python header or template file so that it is automatically included everytime a .py file is created. 
{% highlight python %}
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals
{% endhighlight %}
* Write print functions as `print("Something")`
* Format print functions with `print("Something {}".format('!'))`
* Import custom modules as `from . import mymodule`

**Dealing with exceptions**
* Raising exceptions is `raise "Your useful error message"` in python2 but should be written as `raise Exception("Your useful error message")` 

**Text versus binary**
* Handle binary and text with edge code (reading in and writing out) that decodes and encodes binary data. `str` in python2 was a valid type for both text and binary data, but python3 makes the distinction clear with `str` for text and `bytes` for binary.

{% highlight python %}
with open('somefile.txt', 'r') as f:
  text = f.readlines()
with open('somefile', 'rb') as f:
  binary = f.read()

text = binary.decode('utf-8')
binary = text.encode('utf-8')
{% endhighlight %}

* Do not use `unicode` type, instead just use `str`. `unicode` in python2 was a valid type for unicode but does not exist in python3. 
<br><br>

**Inbuilt dictionary**
* In python2 we could examine the value of the dictionary with `dict[dict.keys()[0]]`. However python3 `dict.keys()` returns a non-indexable iterable.
* Get arbitary key of dictionary with `list(dict.keys())[0]`
* Get arbitary item of dictionary with `list(dict.values)[0]`
<br>
*Indexing at 0 is arbitary because python dictionaries are unordered.*

**Library dependencies**
* `cPickle` in python2 is now `_pickle` in python3
* `importlib` in python2 is now `importlib2` in python3
{% highlight python %}
try:
  import cPickle as pickle
except:
  import _pickle as pickle
{% endhighlight %}

**Bonus Stuff**
* Use *generator* comprehensions instead of *list* comprehensions if performing operations on the sequence. 
`sum(x for x in xrange(100000))`
* This is both more time ans space efficient especially on huge lists. Both list and generator are compatible in both python2 and 3.

#### References 
90% of this is written from own experience. For a better understanding on testing, packaging code and other software-engineering oriented practices, best to consult the [official docs](https://python-modernize.readthedocs.org/en/latest/).

