---
layout: post
title: "Agile OmegaConf Argparse System"
date: "2023-10-01" 
mathjax: true 
status: [Code samples]  
categories: [Code]
---

Probably one of the biggest secrets of how I completed my PhD in 5 years while having a baby is developing a highly agile config and file naming system. 


<br>
## Preliminaries 

### OmegaConf

OmegaConf is a YAML based hierarchical configuration system, with support for merging
configurations from multiple sources (files, CLI argument, environment variables). Yaml is
my preferred way of shipping configs around due to its hierarchy, reusability (of file names
within the same file). It's got a lot of neat features, but I use [Variable Interpolation](https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#variable-interpolation) most frequently.

### Argparse

Omegaconf has its own CLI, but argparse enjoys high proliferation in python code (especially in
research). Argparse is very intuitive for most people, but it doesn't allow hierarchical structure for nesting
arguments. Let's say you wanted to change the learning rate $\alpha$, and also the weighting
factor your decoding $\alpha$, now we could call this something else but when you're dealing
with data, model, prompt, decoding, training configs, suddenly your "CLI calls are a mile
long" - D.Mueller.  

<br>


## My Filenaming system

My filenaming system is managed by a `configs/file_paths/{project_name}.yaml`

{% highlight python %}
HOME: $HOME/projects/project_name
fn: gen-{generator.name}/data-{data.testdataset}/train-model-{model.name}-{model.hidden_size}/format-{format.caps}

gen_fn: ${HOME}/generated/${fn}.csv.hyp
res_fn: ${HOME}/results/${fn}.json

# Analysis filenames
analysis_fn: ${HOME}/analysis/${fn}.csv

# Other Artifacts generated
...
{% endhighlight %}


At the appropriate save_fn in the code we'll have the following which constructs filenames based on the current configs.

{% highlight python %} 
gen_fn = config_files['gen_fn'].format(**args)
res_fn = config_files['res_fn'].format(**args)
{% endhighlight %}

This syntax is just a realisation of the [string format](https://docs.python.org/3/tutorial/inputoutput.html) in python3. Note that `gen_fn` is generated **on-the-fly** with whatever the config is. For e.g., if `generator.name` is `baseline`, or `generator.name` is `beamsearch-5`, the file name is saved as such. `generator.name` can also be generated **on-the-fly** by [variable interpolation](https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#variable-interpolation)

### Reading configs
This follows the familiar argparse format, except we are going to pass in the file paths to yaml config files, instead of single args.

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

Every module has its own config file with a "default", and a project specific config file if required. For instance, if I was experimenting with the prompt format, I would have different config files in `configs/format/{}.yaml` reflecting different experimental conditions. See [here](https://github.com/suzyahyah/icl_coherence_mt/tree/master/configs) for an example of how I organise config files.

Next, we need to 

1. Sort out our "known arguments", whatever has been included by `argparser.add_argument` and "unknown arguments"; everything else. 
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

<script src="https://gist.github.com/suzyahyah/ab3eead087e6eaaa4f19bd8397a5260d.js"></script>

<br>

### Conclusion

It took a while to develop but utimately it paid off immensely as I managed to 

* reduce the project switch lag.
* resolved all of my file naming problems across projects.
* resolved confusions over what argument changed for each run. 
* greatly reduced clutter in my bash scripts and main run file.
* keep things modular by organising configs to match modules. 
* reliably retrieve results by changing just the config file or changing a nested config.
* ran all 5 projects in my dissertation using the same codebase.

The second biggest secret is my build system, which I might write about later.

P.S When I started my PhD I was running bash scripts [like this](https://github.com/suzyahyah/adaptive_mixture_topic_model/blob/master/bin/runGaussian_py.sh).
