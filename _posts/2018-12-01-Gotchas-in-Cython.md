---
layout: post
title: "Gotchas in Cython"
date: 2018-12-01
mathjax: true
status: [Under construction]
categories: [Cython]
---

Some brief notes and gotchas from converting python scripts into cython. 

#### **Compilation**
The cython tutorial gives a basic way of how to do this, however as one who likes to folderize my scripts a fair bit, I have found this [magic script](https://raw.githubusercontent.com/justou/cython_package_demo/master/setup.py) to be pretty fool proof. 

<br><br>
#### **Importing modules**
* **Importing `my_module.py` and `my_module....so`** will cause an import conflict and `my_module.py` gets imported instead. Best to use different names.
<br>

* **Mysterious `cimport numpy as np` and `import numpy as np` convention.** `cimport` imports C functions from the Numpy C API: see `__init__.pxd` from the Cython project [here](https://github.com/cython/cython/tree/master/Cython/Includes/numpy). For reasons of perhaps convenience, the convention is to import both as `np`. I assume internally Cython checks the C API for availability of the class or method, and only if it is not present uses the normal python API. 

<br><br>
#### **Numpy initialisations** 

* **When to use `np.float64_t` vs `np.float64`, `np.int32_t` vs `np.int32`**. Thanks to the above naming convention which causes ambiguity in which `np` we are using, errors like `float64_t is not a constant, variable or function identifier` may be encountered. Essentially, we use `np.float64_t` to declare the C object type, and use `np.float64` to create the object.

{% highlight python %}
def init():

  cdef np.ndarray[np.float64_t, ndim=1] arr1
  array = np.zeros(10, dtype=np.float64)

{% endhighlight %}
<br>
* **When not to use `np.ndarray[np.float64_t, ndim=1]`i.** Our intuitive `np.ndarray` initialisation will fail when used as an attribute of a class. The following will throw an error: `Buffer types only allowed as function local variables`. 

{% highlight python %}
cdef class FailCase():
  cdef np.ndarray[np.float64_t, ndim=1] arr1

  def __init__(self):
    self.arr1 = np.zeros(10, dtype=np.float64)

{% endhighlight %}

The reason for this is that attributes of our `cdef class` are members of `struct` and hence we can only expose simple C datatypes. If we are taking in arbitary python objects, then the way to do this is with `cdef object`. Although for numpy arrays we have something better: memory views.

{% highlight python %}
cdef class BetterCase():

  cdef np.float64_t[:] arr1
  cdef object myObject

  def __init__(self, my_object):
    self.arr1 = np.zeros(10, dtype=np.float64)
    self.my_object = my_object

{% endhighlight %}
<br>

* **Numpy operations on memory views.** Numpy vector and matrix operations would require us to convert the memory views back to `np.ndarrays` as computations cannot be done in the memory views directly. The good news is numpy arrays can be written directly into memory views after they have been manipulated.

{% highlight python %}

cdef class arrayOps():
  
  cdef np.float64_t[:, :] arrays

  cdef __init__(self):
    self.arrays = np.zeros((10, 10), dtype=np.float64)

  cdef add(self, np.ndarray[dtype=np.float64 , ndim=2] x):
    # need to convert to array before vector multiplication
    arr1 = np.asarray(self.arrays[k], dtype=np.float64)
    arr1 += x
    self.arrays[k] = arr1

{% endhighlight %}
<br><br>

#### **Achieving speedups as advertised**

* **Memory views.** Memory views allow efficient access to memory buffers underlying the numpy arrays, allowing us amazing speedups in lookups and writes. Definitely don't tldr [this](https://cython.readthedocs.io/en/latest/src/userguide/memoryviews.html)

<br>
* **Enumerating over python arrays.**
A common pattern in python is `for i, val in enumerate(values):`, however there is no equivalent in C so we should simply index the value instead: `for i in range(len(values)): val = values[i]` 

<br><br>
#### **Mysterious Segmentation faults**
This can occur when calling c-code from Python and in my case there was no indication which line caused the fault. Thanks to my lab mate Mitchell Gordon for pointing out that I should use a debugger to step through line by line until the `segmentation fault` occurs instead of stupidly printing line by line. (It was when a line in a try-except statement was doing some illegal indexing.)

<br><br>
#### **Throwing Exceptions**
The type of exception must match the return type of the python function. When our return type is void, the only option is to throw `*`. Always throw the exception, or a bug in the code will have the program complaining from the start to the end.

{% highlight cython %}

cdef str getName(x) except "No name found":
...

cdef int getVal(x) except -999:
...

cdef void doSomething(x) except *:
...

{% endhighlight %}
