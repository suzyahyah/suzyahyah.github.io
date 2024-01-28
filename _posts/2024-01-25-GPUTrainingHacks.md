---
layout: post
title: "Python Decorators for Monitoring GPU Usage"
date: "2024-01-25"
mathjax: true
status: [Code samples]
categories: [Code, PyTorch]
---


One typically needs to monitor GPU usage for various reasons, such as checking if we are maximising utilisation, i.e., maximising training throughput, or if we are over-utilising GPU memory. The following assumes Nvidia GPUs.

#### **Monitor `nvidia-smi`**

`nvidia-smi` might be one of the first things a command-line ML person learns because of its universality and convenience. However it is quite klunky to print everything and even worse, monitor/eyeball the numbers based on the refresh rate as we are training. Imagine you have 8 GPUs and only 3 of them (which one indeed) is being used by your script while the rest are used by others. Or if you need to check the performance of multiple functions.

<br>

#### **`pynvml`: Python bindings for Nvidia Management Library**

Recently I discovered the Python bindings for the Nvidia Management Library: `pynvml` which is a library for monitoring and managing various states within NVIDIA GPUs. I was thrilled to know it the underlying library for the `nvidia-smi` tool! With this library, we can construct a simple gpu utilisation function, `print_gpu_utilisation()`, and insert it together with training code. Before training, I would recommend running 1 epoch just to check no further OOM, and then checking gpu utilisation  (to check for underutilisation). 

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

<br>

#### **Monitoring GPU utilisation with Python decorator**

We can also turn this into a [Decorator](https://book.pythontips.com/en/latest/decorators.html), where we can do something like the following:

{% highlight python %}
def gpu_util(func):
  def fn_decorator(*arg, **kwargs):
    result = func(*arg, **kwargs)
    output = print_gpu_utilization()
    print(f"Function {func.__name__} --> {output}")
    return result
  return fn_decorator

@gpu_util
def get_model(...):
  ...
  return model

@gpu_util
def generate(...):
  ...
  return generated_output
{% endhighlight %}


Ultimately we need to trade off between training stability, memory requirements, resource requirements and time. The main thing to pin down first is training stability i.e., batch size and learning rate. It is far better to take longer to train the models, than to launch many jobs and later realise that convergence was poor and have to restart the experiments anyway.

<br>
#### References
[HuggingFace:Efficient Training Techniques](https://huggingface.co/docs/transformers/perf_train_gpu_one)\\
[Nvidia Management Library API](https://docs.nvidia.com/deploy/nvml-api/nvml-api-reference.html#nvml-api-reference)
