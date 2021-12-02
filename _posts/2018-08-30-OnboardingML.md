---
layout: post
title:  "Onboarding for Practical Machine Learning Research"
date:  2018-08-30
mathjax: true
status: [Code samples]
categories: [Machine Learning]
---

Contributors: Suzanna Sia, Pan Xinghao, Chieu Hai Leong

*If you're from DSO, I also recommend: [advice about the workplace]({{ site.baseurl }}{% link _posts/2015-01-13-Advice.md %})*

### Introduction

The intended audience is people who are just starting out in a ML research outfit. Although examples are written using a Linux, NLP, Python slant, the points here are meant to be OS, Language and Application agnostic. 
<br><br>

### Preamble

Congratulations! You impressed during the interviews, and have just started your fancy new research job. You learnt about cross-validation, deep neural xyzs and all that jazz and have a plenty of practice doing tutorials on sklearn, medium, etc. and have your preferred text editor or jupyter blah all setup. 

Your supervisor has handed you a paper to go and ‘try out’. Bring it on you say! 

<br><br>

### Getting Started

#### Coming up with initial ideas

In research, we write code to test ideas. 

Ideas reflect our understanding of the problem, and represent the hypothesis that we have about the data. Reading recent papers about how other people have tackled similar problems often provides a starting point which is closer to state-of-art. It also provides inspiration for what could be improved, especially if the author’s have made certain assumptions when coming up with their solution which you think can be better addressed.

<br>

#### Setting up Baselines

One of the first things to do is to set up a baseline, which is what you will be 'trying to beat' with each new idea. This is usually the results from the paper that you are trying to implement. 

* If their code is available, get it running. The easiest thing to do is to search for the author's implementation online.  

* If code is not available, look for implementations online or check with your supervisor if they really want you to implement this from scratch. 

* If you are dealing with a non-academic dataset where there is no baseline, then do the simplest textbook baseline first.

A word about baselines: it is your duty as a scientist to ensure that you build the strongest baseline possible. That is, you should choose a model / algorithm / approach that has a good chance of success, and to tune the hyperparameters to achieve the best performance for the baseline approach. Do not be afraid of having strong baselines to beat --- only by beating strong alternatives can you be sure that your idea(s) will stand up to the test of time in actual use.

The sooner you set up your pipeline from processing to evaluation, the sooner you can implement more complex ideas and see if they are really working.

<br>

#### Setting up Pipelines

In data analytics or NLP, we often have to process data in one format and send them through a pipeline. E.g, tokenization, part-of-speech, sentiment analysis etc. 

Very often, parts of the pipelines are open source code written in various languages. E.g, [Stanford NLP which is written in Java](https://stanfordnlp.github.io/CoreNLP/) is often used as a baseline for tokenization, part-of-speech etc, while the [script for bleu score evaluation is written in perl](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl) 

We also reuse scripts at various parts of the pipeline that were written in previous projects.  Hence, it is worth modularise research code into the various pipeline steps rather than have a huge program that does everything all at once.

Instead, you can try something like

`Step1.sh`: run Stanford NLP, takes in raw text files, outputs into a directory of tokenized files and a directory of parse trees

`Step2.sh`: run Analysis Algo, takes in tokenized files, outputs Algo scores of each sentence for each input file.
..
etc.

Note: You should avoid editing raw data manually even if it is easier to do so than it is to write a script. Your code should move the raw data through a pipeline to the final analysis, ensuring your results can be replicated. 

<br>

#### Checkpoints (saving and loading binaries to file)

Checkpointing is a useful strategy for saving time when debugging and testing minor code changes. Certain steps in your pipeline may take a long time to run, or result in objects which cannot easily be written to disk in conventional formats (such as trained SKlearn models). You can consider 'checkpointing' the progress by saving and loading the intermediate objects from binaries. 

The defacto way to do this in Python is via [pickling](https://docs.python.org/3/library/pickle.html)

Note: Pickles are super useful but be careful about overdoing things. Intermediate outputs like dictionaries should be written to json file, and matrices should be saved as matrices. Also, it is python and version specific.

<br><br>

### Testing Ideas

#### Baseline and Experimental Conditions

You finally got the baseline running, cool! 

At this point you have n different ideas to try, and are keen to implement a whole bunch. However when conducting research we need to have a control condition and experimental condition that only changes one thing, while keeping everything else constant. This is a fundamental tenet of empricism. Changing one thing allows us to know whether each of the ideas is working and by how much. 

Occassionally there are reasons to prefer running experiments via bash than within the main code itself depending on the level of abstraction for your program. A common pattern I like to use in combination with python's argparse is:

{% highlight bash %}
EXP_CONDITIONS=(a b c d)
for ((i=1, i< ${#EXP_CONDITIONS[@]}, i++)); do
    echo Running exp $i
    python src/main.py --exp ${EXP_CONDITIONS[$i]}
done
{% endhighlight %}

<br>

#### Getting Results

The results are in for the experimental condition, and because we took the effort to set up our baseline we can actually tell how good we're doing :) There are a few scenarios of interest:

**1. Too good to be true.**

If your results are too good to be true they most likely are. Check if you have done any of the following: 

* Test predictor variables used in training
* Train outcome variable used in testing
* Test distribution used in testing (this happened to me recently in an MCMC setting)

**2. Marginally better than baseline.**

Every little tweak that you do might result in better or worse scores. Question is whether it improving the results in a significant or important way.

**3. Seemingly random or worse.**

Check if your pipeline is set up correctly. One sanity check is to do a train and test which includes the dependent variable (y) itself as a feature, if your pipeline is set up correctly you should be getting near 100% accuracy..

**4. Getting results that are different on every run even though the code is the same.**

If you are getting different results at every run even though nothing has changed in your code, check if your algorithm is using a randomised process. You will nearly always be able to set the random seed. This leads us to the next point.

<br>

#### Repeatability, Reproducibility

Whatever experiments you run should be repeatable (giving same results when you run it multiple times) and reproducible (giving similar results when *others* run your code or implement your approach). Without either means that no one can be verify your results, and your excellent work cannot be transited to solving real-world problems. This requires that you do the following:

**1. Always, always control your randomness.**
   
Set random seeds whenever possible (i.e. always), **and** conduct multiple runs with different random seeds to ensure that results do not vary too wildly (otherwise your results are brittle, depending on a magical random seed choice).

**2. Document every single hyperparameter, every knob that can be turned.**
   
Two years after writing your code, you must be able to come back, set the knobs at exactly the right positions, and get exactly the same results as you had two years back. Of course, this implies knowing exactly where to set all the hyperparameters, *including* your random seed choices. One way of doing this is to organise everything in config files (see below discussion).

<br>

#### Error Analysis and Prioritising next steps

You’ve probably heard of the confusion matrix. If you’ve seen it, you’ll know its not terribly useful beyond telling you which class the classifier prefers. To guide our efforts in research, we usually perform error analysis to understand where the algorithm is lacking and how we can make a better model.

You may find it useful to write a set of scripts just to manipulate and sort errors. In NLP especially, errors can stem from many many areas, it is important to be able to automatically quantify or get a quick qualitative overview of the types of errors that you are getting. This allows you to prioritise what to address, that results in the highest yield for the performance metric.

<br><br>

### Monitoring Resources
`htop` - this is a DSO cfl favorite. Also, `ps aux`. Lots of information about this on web. A combination I use most frequently is `htop` and <F5> to see parent and child processes together with printing the process ID of the script so that it is easier to monitor:


{% highlight bash %}
import os; print(os.getpid())
{% endhighlight %}

<br>

#### Estimating time/program completion
Estimate how long a program will run for using printouts. It’s not enough to just do 
    
{% highlight python %}
import time
def big_process():
    start = time.time()
    for i in range(n_inputs):
        ...
    finish = time.time()-start. 
{% endhighlight %}


What if your code doesn’t run finish over the weekend and you dont know how much longer it will take?

For small functions, a common pattern which prints every 10% completion can look like this  

{% highlight python %}
start = time.time()
if i%(n_inputs/10)==0:
    print("{} complete - time elapsed:{}".format(i/n_inputs, start-time.time()))
{% endhighlight %}

Alternatively use something like `tqdm` (but dont get overly reliant on this as it slows down your program and is a third party library which you dont always get access to when you have to code in an underground bunker).

<br>

#### Disk Space

This is usually not something we worry about, until it becomes something to worry about. You might get triggered to do so after someone (could be anyone) encounters memory errors, leading to `du -sh*` on intermediate data directories, and balking at the amount of space used. The other time this might happen is when your research code is being translating into industrial grade, and someone does a calculation at how much space is required but your data folder far exceeds that. Much of your code needs to be rewritten to use better data structures. Assuming you haven't taken a data structures course and just need to get by without causing too much of an uproar, one simple way to think about how to save space, is what is being repeated in the data that was saved to disk. 
<br>

#### Runtime

When your job seems like its taking a long time to run, running parallel jobs is one of the most tempting things to do. As a result, it gets misused ALOT. Because resources in the lab are shared, running parallel jobs deprives someone else of compute power. Instead, consider if you can write your code in a more optimal fashion instead of jumping into getting a cheap win with parallelisation. Most of the time it can be, especially if you were writing it the first time, and have never worked through an Algo & DS course.

<br><br>

### Organising Code

#### Directory Separation

This may not seem necessary in small projects, but when you feel sufficiently frustrated at various files lumped within the same directory, you can consider organising your project directories to be more friendly. 

Separate Data and Configs from Code. Seperate Logs from Data. There should be a clear conceptual separation of code and everything else. Ideally your code should run data from anywhere given the data file path. Seperate Input Data from Working Data, and Output Data. There are many discussions about this and there is no hard and fast rule, but I gravitate towards something like the following:

{% highlight bash %}
/home/staff1
    Data
        raw
    Projects
        Project1
        Project2
            README
            requirements.txt
            configs
            bin
                process1.sh
                process2.sh
                run_all.sh
            src
                __init__.py
                ...
            data 
                raw (symlink)
                working
            logs
            packages
{% endhighlight %}

<br>

#### Version Control

Don’t stop learning git at git clone. Version control (with well-written commit messages) is really useful to freeze projects and so that you dont have lots of copies of the same codebase lying around. It also allows your to revert to an old version of the code that ran previously. 

If you move in and out of projects, you may find it useful to log the commit version, time of the experiment and experimental parameters within the log file itself. 

{% highlight python %}
import subprocess; subprocess.check_output([‘git’, ‘describe’, ‘--always’])
import datetime; datetime.datetime.now().strftime(“%Y-%m-%d %H:%M”)
{% endhighlight %}

<br>

#### Config Files

In software engineering projects, config files are your keys database URLs etc. in research projects, config files are your experimental hyperparameters. Configs in Machine Learning projects are usually nested, hence you may find it useful to move beyond txt files and use something more appropriate like JSON, HOCON, or YML formats.

People also like [argparse](https://docs.python.org/3/library/argparse.html) as entry to running scripts.

<br><br> 

### Presenting Your Work

#### Document, present!

You've set up your pipeline, ran your experiments, and now time to fight the second half of the battle: documenting and presenting your work. As technical people, we often get excited with getting results, and forget that our job does not end with a `print` statement. Work that is not properly documented and disseminated might as well be work that was left undone. Imagine if Einstein had just stopped at drawing his conclusions from his thought experiments, and failed to share his work with the world!

Do not see documentation as a chore. Recognise that your report and presentations are the external face of your excellent work. The greater your work, the more effort should be placed into writing it up.

One key point: Cater your report and presentation to *your audiences*. A common mistake is writing reports as if we would be the only people reading them, which often manifests as a verbal diarrhoea of too many details. Instead, convey the high-level ideas first, explaining the problem you are tackling, and why you chose a particular approach. If the reader only finishes reading your concise introduction, she should already know (and be impressed by!) the work you have done.

<br>

#### Preparing for Demos 

Depending on your luck, you may be activated for a demo for the work that you did in some distant past deliverable.

You have two options, run the demo data on your algo and pass the processed file back and hope that this is a one-off, or give the Program/SWL access to running your algo. 

Because demos go through much back and forth between management, the program manager and the technical team, you should anticipate requests like “Can you run your algo on this additional input? Boss want to know what it looks like...” 

Hence I recommend setting up a minimalist web server and restful API to allow the front-end to make dynamic calls to your engine. Define the input and output format with the SWL or Programme counterpart and they can test all they want. Personally I use flask or expressjs which are microframeworks for running very lightweight web apps.

Any demo must give the appearance of “explainability”. When the “explainability” of the system is conveyed through text output explanation, make sure these are easily modifiable/accessible to everyone in text files. There will definitely be requests to change the ‘explanation’ given to the audience.
<br><br>

### Productivity with Linux (bonus)

Needless to say, productivity with tools is an integral part of research productivity. In CFL, Linux OS is the defacto option and you should get used to working with it asap. 

#### Read, Filter, Display

The following are mostly tips to read and search for information via the command line. Sure you can always write code to read in files, search+filter and print, but that's much less efficient than the following:

* jq for reading json files, e.g., `jq -C '.' file.json | less -R`

* use alias for shortcuts keys.

* diff (or vimdiff): to compare old and new outputs

* cut, sort, uniq: to compute data statistics

* perl regular expressions, e.g.,
{% highlight bash %}
      cat file | egrep "\w Mary \w" | perl -pe "s/^.*?(\w+? Mary \w+).*$/\1/g"  | sort | uniq -c | sort -rg
{% endhighlight %}

the above should count all the trigrams with Mary in the middle. 

<br>

#### Multiple Screens

A common desired situation is to open multiple 'screens' in the linux server to run different experiments or code and run stuff at the same time. There are several options available and it's just a matter of personal preference. People seem to be using screen, byobu or tmux. Lots of resources on the internet, here's an example of how screen can be used:


`$:screen -S myscreen`
(starts a new "screen")

`$:./run.sh any_job`
(run your jobs)

Type Ctrl-A Ctlrl-D
(detach from screen) 

Then you can logout and your jobs will be safely running in the server. The next time you login, you can type

 `$:screen -r myscreen`

(returns to your previous "screen")


#### Keeping a Research Log 

"Keep a research diary. Have a single log for every day you work on the project and section it
into focus/meeting notes/bug fixes/ bug watch/changes etc. and preliminary results just copying
and pasting from logs. It helps keep track of things you've tried and how you fixed errors."  -- Alexandra Delucia


---

