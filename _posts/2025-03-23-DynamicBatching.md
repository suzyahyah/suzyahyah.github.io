---
layout: post
title: "Dynamic Batching for Training Large Sequence Models (LLMs)"
date: "2025-03-25"
mathjax: true
status: [Code samples]
categories: [Code, PyTorch]
---


#### **Preliminaries**

To maximise GPU memory when training large models, we want to pack tokens such that sequence padding is minimised and GPU memory is maximised. 

1. `torch.utils.data.dataloader` is an python iterable over a PyTorch dataset
2. `torch.utils.data.dataset` implements `__getitem()__`, which maps keys to data samples.
3. `torch.utils.data.sampler` specifies the sequences of keys used in data loading.

By default, the `DataLoader` will collate individual fetched samples into batches using the arguments `batch_size`, `drop_last`, `batch_sampler`, and `collate_fn`. An alternatively, if `batch_size` is None, we can construct a `BatchSampler` which yields a list of keys at a time.


#### **Default Approach**

We have several options, starting with the default.

The most default thing to do is to pad every sequence to the maximimum context window, and return a fixed batch size. However, this is incredibly wasteful. Imagine a batch size of 2, where we have a sequence X1 of length 10 and sequence X2 of length 1000 in the same batch. Sequence X1 will be padded for 990 token positions, which is nearly 50% wasted GPU memory.

<br>

#### **Concatenate and Slice**

The second approach, is to concatenate all data together, and slice the long sequence into smaller sequences, each with the length equal to the maximum context window. This is more space efficient, but may create the wrong long-range dependencies, if we had concatenated many non-related small sequences together.

<br>

### **Dynamic Batching Method**

The idea behind dynamic batching, is that we're going to maximise the GPU memory to avoid the wastefulness of the default approach, yet not introduce spurious relationships by concatenating unrelated sequences. Intuitively, we want short sentences to be grouped together and have a larger batch size, and long sequences to be grouped together with a smaller batch size.

1. Create 4 bins, with ranges $(0, L/4)$, $(L/4, L/2)$, $(L/2, L*(3/4))$, $(L * (3/4), L)$, where $L$ is the maximum sequence length or maximum context window. 

{% highlight python %}
def construct_bins(self):
  print("Constructing bins", len(self.seq_lengths))

  for i, val in tqdm(enumerate(self.seq_lengths)):
    if val < (self.max_seq_len / 4):
      self.bins['small'].append(i)
    elif val < (self.max_seq_len / 2):
      self.bins['med'].append(i)
    elif val < ((self.max_seq_len / 4) * 3):
      self.bins['large'].append(i)
    else:
      self.bins['xl'].append(i)
{% endhighlight %}

<br>

2. Sample from these bins proportional to how much data is in these bins. To ensure we have seen all the data, we keep track of the current index of the bin, and when the current index exceeds the bin size, reset it to 0, and shuffle the bin. Note that the BatchSampler must yield a list of values, unlike Sampler (non-batch) which yields *from* a iterable.

{% highlight python %}

def __iter__(self):
  while True:
    choices = [('small', self.med_batchsize*2), ('med', self.med_batchsize), ('large', (self.med_batchsize/2)*3), ('xl', self.med_batchsize//2)]
    weights=[len(self.bins['small']), len(self.bins['med']), len(self.bins['large']) len(self.bins['xl'])]

    bin_type, size = random.choices(choices, weights, k=1)[0]

    print("SAMPLER ITER CALLED", bin_type, size)
    cur_index = self.current_index[bin_type]

    self.current_index[bin_type] += size
    if self.current_index[bin_type] > len(self.bins[bin_type]):
      self.current_index[bin_type] = 0
      random.shuffle(self.bins[bin_type])

    yield self.bins[bin_type][cur_index:cur_index + size]

{% endhighlight %}

We want to interface with HuggingFace trainer and Pytorch Dataloader, and therefore encapsulate our methods inside a Custom class that inherits from [BatchSampler](https://github.com/pytorch/pytorch/blob/main/torch/utils/data/sampler.py)

<br>

The complete class looks like this


{% highlight python %}
class DynamicBatchSampler(BatchSampler):
    def __init__(self, seq_lengths, max_seq_len=1024, med_batchsize=64):
        self.med_batchsize = med_batchsize
        self.seq_lengths = seq_lengths
        self.bins = {"small": [], "med": [], "large": [], 'xl':[]}
        self.max_seq_len = max_seq_len
        self.construct_bins()
        self.current_index = {"small": 0, "med": 0, "large": 0, 'xl':0}

    def __iter__(self):
      ...
    def construct_bins(self):
      ...
    def __len__(self):
        # Return an estimate of the number of batches
        return sum([
            len(self.bins['small']) // (self.med_batchsize*2 + 1),
            len(self.bins['med']) // (self.med_batchsize + 1),
            len(self.bins['large']) // (self.med_batchsize//2 * 3 + 1),
            len(self.bins['xl']) // (self.med_batchsize//2 + 1)
        ]) // 4 

{% endhighlight %}


<br>

#### **HuggingFace Trainer Compatibility**

To make this compatible with HuggingFace trainer, we subclass the `_get_train_sampler()` method to return our newly constructed DynamicBatchSampler.


{% highlight python %}
class MyTrainer(Trainer):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def _get_train_sampler(self):
    dynamic_sampler = DynamicBatchSampler((trainer.train_dataset['lengths']), max_seq_len=1024, med_batchsize=64)
    return dynamic_sampler
  
  def get_train_dataloader(self):
  # override this method
    ...
    dataloader_params = {"collate_fn": data_collator,
                         "num_workers": self.args.dataloader_num_workers,
                         "pin_memory": self.args.dataloader_pin_memory,
                         "shuffle": False,
                         "sampler": None,
                         "batch_sampler": self._get_train_sampler(),
                         "drop_last": False,
                         "persistent_workers": self.args.dataloader_persistent_workers}
    dataloader = DataLoader(self.train_dataset, **dataloader_params)
    return self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))

{% endhighlight %}

Then we can easily do `trainer = MyTrainer(..); trainer.train()`. 

*Implementation Note*: 

It's necessary to override the `get_train_dataloader()` method to have full control over the dataloader_params. For instance, because we constructed a custom BatchSampler, the `batch_size` argument given to trainer should be empty or there will be an error thrown regarding a conflict in `batch_size` number. This defaults to 1 in Pytorch but defaults to 8 in HuggingFace Trainer.

<br>
#### **References**

[PyTorch Data Utils Reference](https://pytorch.org/docs/stable/data.html)
