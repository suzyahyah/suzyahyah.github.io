---
layout: post
title: "Lean OmegaConf Argparse System"
date: "2023-10-01" 
mathjax: true 
status: [Code samples]  
tldr: Discusses a YAML-based hierarchical configuration system called OmegaConf, which is useful for managing configurations across multiple sources. It explains the challenges of using argparse for nested configurations and introduces a custom file naming system for easy organization and retrieval of project-specific files. The post concludes by highlighting the benefits of this approach in terms of reducing project switch lag, improving file naming, and streamlining configuration management for various projects.
categories: [Code]
---



<br>
## Preliminaries 

### OmegaConf

OmegaConf is a YAML based hierarchical configuration system, with support for merging
configurations from multiple sources (files, CLI argument, environment variables). Yaml is
my preferred way of shipping configs around due to its hierarchy, reusability (of file names
within the same file). 

It's got a lot of neat features, but I use [Variable Interpolation](https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#variable-interpolation) most frequently.

### Argparse

Omegaconf has its own CLI, but argparse enjoys high proliferation in python code (especially in
research). Argparse is very intuitive for most people, but it doesn't allow hierarchical structure for nesting
arguments. 

Let's say you wanted to change the learning rate $\alpha$, and also the weighting
factor your decoding $\alpha$, now we could call this something else but when you're dealing
with data, model, prompt, decoding, training configs, suddenly your "CLI calls are a mile
long" - D.Mueller.  

<br>


## A Lean and Just-in-time File Naming System

My filenaming system is managed by a `configs/file_paths/{project_name}.yaml`

{% highlight python %}
HOME: /home/suzyahyah/projects/project_name
fn: gen-{generator.name}/data-{data.testdataset}/train-model-{model.name}-{model.hidden_size}/format-{format.caps}

gen_fn: ${HOME}/generated/${fn}.csv.hyp
res_fn: ${HOME}/results/${fn}.json

# Analysis filenames
analysis_fn: ${HOME}/analysis/${fn}.csv

# Other Artifacts generated
...
{% endhighlight %}


Note that `fn`, `gen_fn` and `res_fn` are dynamically generated. At the appropriate save function in the code base, we'll have the following which constructs filenames based on the current configs.

{% highlight python %} 
gen_fn = config_files['gen_fn'].format(**args)
res_fn = config_files['res_fn'].format(**args)
{% endhighlight %}

This syntax takes advantage of [string format](https://docs.python.org/3/tutorial/inputoutput.html) in python3. Note that `gen_fn` is generated **on-the-fly** with whatever the config is. For e.g., if `generator.name` is `baseline`, or `generator.name` is `beamsearch-5`, the file name is saved as such. `generator.name` can also be generated **on-the-fly** by [variable interpolation](https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#variable-interpolation)

<br>
### Reading Configs from YAML Files
This follows the familiar argparse format, except we are going to pass in the file paths to yaml config files, instead of having a long train of arguments.

{% highlight python %}
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--seed', default=0, type=int)
    argparser.add_argument('--do_analysis', default="")

    argparser.add_argument('--training_cf', default='configs/training/default.yaml')
    argparser.add_argument('--format_cf', default='configs/format/default.yaml')
    argparser.add_argument('--data_cf', default='configs/data/default.yaml')
    argparser.add_argument('--generator_cf', default='configs/generator/default.yaml')
    argparser.add_argument('--model_cf', default='configs/model/default.yaml')

   # ... etc ...

    argparser.add_argument('--file_paths_cfg', default="")
{% endhighlight %}

Every module has its own config file with a "default", and a project specific config file if required. For instance, if I was experimenting with the prompt format, I would have different config files in `configs/format/{}.yaml` reflecting different experimental conditions. See [here](https://github.com/suzyahyah/icl_coherence_mt/tree/master/configs) for an example.

Next, we need to 

1. Sort out our "known arguments", whatever has been included by `argparser.add_argument` and "unknown arguments"; CLI arguments that are used to overwrite the default (useful for looping through experimental conditions).
2. Load our file paths configs (this contains all our file paths).
3. Merge the known args and unknown args.

{% highlight python %}
args, uk_args = argparser.parse_known_args()
cfp = OmegaConf.load(args.file_paths_cfg)
args = io_utils.merge_omegaconf_w_argparse(args, uk_args)
{% endhighlight %}

<br>
### Emabling Argparse Style CLI with OmegaConf 

It would be super to send nested configs `--data.direction en-fr --model.hidden_size 100` in an `argparse` style syntax. However `argparse` does not support nested configs, and `omegaconf` does not allow merging with argparse. (see [issue](https://github.com/omry/omegaconf/issues/569))

So we'll just do it ourselves!

<details>
<summary> $\rightarrow$ Click to show gist </summary>
<script src="https://gist.github.com/suzyahyah/ab3eead087e6eaaa4f19bd8397a5260d.js"></script>
</details>

<br>

### Conclusion

It took a while to develop but utimately it paid off immensely as it

* greatly reduce the project switch lag.
* resolved all of my adhoc file naming problems across projects.
* resolved confusions over what argument changed for each run. 
* enabled reliable retrieving of results by changing just the config file or changing a nested config.
* greatly reduced clutter in my bash scripts and main run file.
* kept things modular by organising configs to match modules. 
* enabled me to run all 5 projects in my dissertation using the same codebase.

One of the biggest secrets of how I finished the PhD in 5 years with a baby. The second biggest secret is my build system, which relates to the point about "kept things modular by organising configs to match modules". I might write about this later.

P.S When I started my PhD I was running bash scripts [like this](https://github.com/suzyahyah/adaptive_mixture_topic_model/blob/master/bin/runGaussian_py.sh).
