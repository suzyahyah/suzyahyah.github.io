---
layout: post
title: "GPU Out Of Memory when Training. Tricks and Monitoring Usage"
date: "2024-01-25"
mathjax: true
status: [Code samples]
categories: [code, PyTorch]
---

### Tricks

To fit training into memory, there are several tricks we can employ. One of the best writing on this might be [HuggingFace: Efficient Training Techniques](https://huggingface.co/docs/transformers/perf_train_gpu_one) which covers all the things you can do to tackle OOM training with their library. There is also a lot of advice on the internet and since I've tried a bunch of them, here are the caveats.

1. "Decrease the batch size". It may be tempting to just decrease the batch size until whatever you're trying to do fits into memory - all you need to change is a single number. However an overly low batch size can result in too much oscillation of the loss function and poor convergence. 

2. "Use Gradient Accumulation". Another config which seemingly solves your problems is to run multiple passes and accumulate the gradients, thereby saving memory. However each gradient accumulation step is an entire forward pass, meaning if gradient accumulation steps=2, the time it takes to go through one epoch is doubled.

3. "Mixed Precision Training". Mixed Precision Training is great when it works out of the box, but is not always compatible with various models, architectures and training libraries. It might be the default in the future to cater for this though.

4. "Do Gradient checkpointing". Instead of storing all the forward activations, gradient checkpointing only stores some of the intermediate activations and recomputes the gradients during the backward pass. It introduces overhead of recomputing some of the activations to get the full gradient signal, which can be quite large if the network is large. 

5. "Use Multiple GPU training". This is the method of choice in large scale model training. If the infrastructure supports it, go right ahead. It is quite straightforward to launch a job with $M GPUS 

`qsub -N gpu=$M .. bin/submit_train.sh $args `

and in bin/submit_train.sh:

`torchrun --nproc_per_node $M code/train.py $args`

### Monitoring Usage

Next, after we've changed some setting and fit into memory, we do want to see if we are maximising utilisation, i.e., maximising training throughput. The following assumes Nvidia GPUs.

1. "Monitor nvidia-smi". `nvidia-smi` might be one of the first things a command-line ML person learns because of its universality and convenience. However it is quite klunky to print everything and even worse, monitor/eyeball it continuously as we are training. What if you have 8 GPUs and only 1 of them (which one indeed) is being used by your script while the rest are used by others? 

Recently I discovered the Python bindings for the Nvidia Management Library: `pynvml` which is a library for monitoring and managing various states within NVIDIA GPUs. I was thrilled to know it the underlying library for the `nvidia-smi` tool! With this library, we can construct a simple gpu utilisation function, `print_gpu_utilisation()`, and insert it together with training code. I would recommend running 1 epoch just to check no further OOM, and then checking gpu utilisation  (to check for underutilisation). 

{% highlight python %}
import os
import pynvml
import torch
import numpy as np

def print_gpu_utilisation():
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        torch_gpu_id = torch.cuda.current_device()
        pynvml.nvmlInit() 
        devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        nvml_gpu_id = int(devices[torch_gpu_id]) 
        handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_gpu_id)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        info_used = info.used//1024**2 
        info_total = info.total//1024**2

        print(f"GPU {nvml_gpu_id} memory occupied: {info_used}/{info_total} MB\
        = {np.round((info_used*100)/info_total,2)}%.")

        pynvml.nvmlShutdown()
{% endhighlight %}

Ultimately we need to trade off between training stability, memory requirements, resource requirements and time. The main thing to pin down first is training stability i.e., batch size and learning rate. It is far better to take longer to train the models, than to launch many jobs and later realise that convergence was poor and have to restart the experiments anyway.

#### References
[HuggingFace:Efficient Training Techniques](https://huggingface.co/docs/transformers/perf_train_gpu_one)\\
[Nvidia Management Library API](https://docs.nvidia.com/deploy/nvml-api/nvml-api-reference.html#nvml-api-reference)
