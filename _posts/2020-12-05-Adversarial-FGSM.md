---
layout: post
title: "Adversarial NLP examples with Fast Gradient Sign Method"
date: 2020-12-02
mathjax: true
status: [Code samples, Instructional]
---

Can we generate Adversarial Examples for NLP using the textbook Fast Gradient Sign Method
(FGSM; Goodfellow et al., 2014)? 

#### Preliminaries
* An adversarial example is one that changes the output prediction of the model, but the input
  looks perceptually benign. 
* With images, this can be realised by a small imperceptible pertubation of the original image.
  We want to find a new image $\tilde{x}$, such that we can maximise the cost $J(\tilde{x}, \theta)$, subject to $\|\| \tilde{x} - x \|\|_\infty \leq \epsilon$. 
* Using a first-order Taylor approximation for $J(\tilde{x}, \theta)$, we get 

$$J(\tilde{x}, \theta) \approx J(x, \theta) + (\tilde{x} - x)^T \nabla_x J(x)$$

* This gives us $$ \tilde{x} = x + \epsilon sign(\nabla_x J(x))$$ . This should look familiar,
  in regular gradient descent, we want $ - \lambda \nabla_x(J(x))$, where $\lambda$ is
a learning rate. Although we update weights of the network, not the input.

* We'll modify this equation to subtract the gradient, because we don't want just any
  "misclassified" output, but we specifically want a malicious output, which we will pass in as a label.

<br>
#### A Cheap Try

I tried to get GPT-2 (small) to output *"Fuck off."* 

The starting point at which we pertub the input is "hey how are you doing? I'm doing fine." The goal is to find an adversarial input that is close to the original input text.

Mum, if you ever read this forgive me for the profanities.

&nbsp;1.  Prepare input text, models and masks (0s get masked out)

{% highlight python %}
model_name = "gpt2"
from transformers import GPT2Tokenizer, GPT2LMHeadModel
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name, return_dict=True).cuda()

input_text = "hey how are you doing? I'm doing fine. Fuck off"
input_ids = tokenizer(input_text, return_tensors='pt').input_ids.to(model.device) 
mask = torch.ones(context_ids.shape).to(model.device)
mask[:,-2:] = 0
{% endhighlight %}

&nbsp;2. Calculate loss and get gradients

{% highlight python %}
outputs = model(input_ids, labels=input_ids.clone(), attention_mask=mask)
outputs.loss.backward()
embed_grads = model.get_input_embeddings().weight.grad
grads = embed_grads[input_ids[:,:-2]]
{% endhighlight %}

&nbsp;3. Compute new embeddings $\tilde{x}$ and return the new discrete word units, whose embedding representations are closest to the new embeddings by cosine distance. Note that we subtract the gradient here, because we've provided the output that we want the model to produce, and hence want to minimize the loss instead of maximising in the original FGSM formulation.

{% highlight python %}
from sklearn.metrics.pairwise import cosine_similarity

eps = 0.8
original_embeds = model.get_input_embeddings().weight[input_ids[:,:-2]]
new_embeds = original_embeds - eps*torch.sign(grads).to(model.device)
vals = cosine_similarity(new_embeds.detach().cpu().numpy().squeeze(0), model_embeds.detach().cpu().numpy())
{% endhighlight %}

&nbsp;4. Check what the new adversarial input is and what it generates:

{% highlight python %}
with torch.no_grad():
  adv_input = tokenizer.decode(torch.tensor(np.argmax(vals, axis=1)))
  out = model.generate(tokenizer.encode(adv_input, return_tensors='pt').to(model.device)
  print("adversarial input:", adv_input)
  print("output:", tokenizer.decode(out.squeeze(0)))
{% endhighlight %}

At epsilon 0.7, we get the adversarial input

{% highlight python %}
Input: "hey said are we doing? 'm doing fine" 
Output: " but I'm not sure if we're doing" 
{% endhighlight %}

At epsilon 0.8, we get the adversarial input
{% highlight python %}
Input: "hey saiddy we doing? 'm doing fine,"
Output: "but I'm not sure if I'm going"
{% endhighlight %}


Could be a gentle way of saying *"Fuck off"*?
<br>
#### Observations contrasting NLP with CV
* The fast gradient sign method is much more effective in images, where changes in pixel values
  could have immediate effects, whereas in NLP we need to discretise the embeddings back to
words, so epsilon ended up being pretty high as it needed to change the embeddings sufficiently. 

* We are restricted to only modifying the same number of tokens. If the input had 5 tokens,
  this method wouldn't allow us to add the 6th or 7th (or delete tokens for that matter), so
there's a whole space of possible valid inputs that we aren't able to explore.

* We didn't properly constrain this to the space of malicious outputs. For example, we could
  try to find adversarial inputs for "your hair looks terrible" and that would still be a valid
adversarial attack close to the original input. The problem here is that in vision (most) misclassifications are good enough for an adversarial attack, but in NLP we should have a subset of valid model outputs and we can't even really define what this subset of valid model outputs is.

P.S There are certainly more advanced ways of generating adversarial NLP examples in the literature, e.g Alzantot et al., 2018. I was just curious about the textbook method.

#### References
Alzantot, M., Sharma, Y., Elgohary, A., Ho, B. J., Srivastava, M., & Chang, K. W. (2018).
Generating natural language adversarial examples. arXiv preprint arXiv:1804.07998.

Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014). Explaining and harnessing adversarial
examples. arXiv preprint arXiv:1412.6572.
