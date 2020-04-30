---
layout: post
title: ""
date: "2020-01-17"
mathjax: true
status: [Instructional]
categories: [Machine Learning]
---

I'm killing myself slowly. Wasting my life away. With all the bugs I introduce. The more fluent I get the more bugs I create, cos its no longer syntax errors and this while loop seems to run with no stop condition other than my death:

{% highlight python%}

while suz.isalive():
  everything.compiles()
  everything.runs()

  if lucky:
    time.sleep(86400) # one day

  else:
    time.sleep(86400*7) # one week

  print("An obvious bug: you didn't check the output dummy")
  code.fix()
  
{% endhighlight %}

The problem is I'm too damn lazy to (1) Set up and check the output of a log file, (2) drop `pdb.set_trace()` and `print` around my code even if nothing crashes. Hey why fix what isn't broken (Ans: See while loop above)

The bigger problem is when we have a for-loops and we want to inspect just one run of the for
loop, I get even lazier to write: `if i==0: print(a, b, c, d)`. Worse still, I accidentally
print/log a huge text file/dictionary/list/obj/numpy array and have to bang furiously on the
ctrl+c or just give up and go for the 10th tea break.

The answer is, a one script fully contained debugger for people like me who don't use IDEs.
check the value of ANY variable from ANY FILE and any function, without fear of spamming your
screen again. For lazy people to do manual checks without writing manual code, esp since its hard to write sensible unit tests for ml projects.

I have just decided to call it the screen-life-saver.  Never spam your screen again!!

### Benefits:

1. ONLY PRINT/TRACE ONCE! Even if you're inside for loops! 
2. Dont need to instantiate complicated class to debug in Ipython
3. Turn on and off print statements really super easily.
4. Check small functions - no excuses to be lazy anymore!
5. Print all variables in the function - ultimate lazy!
6. Automatically fixes your significant figures printing!

### Why you Should NOT use this.

1. If you want more functionality and actually want to record everything (check out python logging module)
2. If you can actually write unit tests for your code (check out python unittest)
3. If you want to step through your code (check out pdb)


### Usage:

1. cp this to a utilities folder in your home dir, i.e. ~/global_utils
2. Add this to your .bashrc, export PYTHONPATH="${PYTHONPATH}:~/global_utils"
3. source ~/.bashrc
3. Then in any .py file, (I add it to my .py templates)

from debugger import Debugger
DB = Debugger()
DB.debug_mode=True  # Set to False to kill all printing

class ComplicatedClass:
    def complicated_fn_to_debug(self):
        var1 = ..
        var2 = ..
        DB.dp({'var1_name': var1, 'var2_name':var2}) # before

        var1 = dosomething(var1)
        var2 = dosomething(var2)
        DB.dp({'var1_name':var1, 'var2_name':var2}) # after

        DB.dp() 
        # if no arguments given, print all variables in this function
""" 


