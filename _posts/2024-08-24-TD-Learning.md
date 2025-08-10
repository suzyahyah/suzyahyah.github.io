---
layout: post
title: "Temporal Difference Learning: Taking advantage of Incomplete Trajectories"
date: "2024-08-24"
mathjax: true
status: []
categories: [Reinforcement Learning]
---

### Summary

In a RL problem, the goal is to learn the best action to take in a given state. Value-based
methods, as opposed to policy gradient methods or model-based approaches, try to
learn the value of states $V(S)$ of the Value of State-action pairs $Q(S, a)$. 

**Temporal Difference (TD)** learning, refers to a method which learns the state-value
function, based on the difference between estimates at different time steps. Temporal
difference was a major breakthrough in the 1980s, because it allowed learning from incomplete
reward trjaectories, therefore lowering sample complexity. 


Today we take TD for granted as the precursor to the more famous Q-learning and Deep Q-learning. I want to briefly reconstruct the logical process that led to the various forms of TD.

<br>

#### Preliminaries

**General Policy Iteration** is the framework for approaching the solution for Value-based RL. It
consists of two steps:

1. **Policy evaluation**; estimating better state-values or state-action values.
2. **Policy Improvement**; choosing better actions in the given state with our current
   estimate. 

MC, TD, TD(0), TD($\lambda$), SARSA, Q-learning are approaches related to *policy evaluation*.

<br>


### **Deriving TD from Monte-Carlo**

If we do not have a model of the environment (i.e. no knowlege about state-transition distribution), the most naive way of learning State-action Values is by using **Monte-Carlo** Learning. This involves collecting rewards after the episode has terminated, and then averaging the rewards by the
number of times the agent has visited that state. For each episode's reward $G_t$ and learning
rate $\alpha$, the update of $V(S_t)$ is done once each time an episode terminates:

\begin{align}
V(S_t) = V(S_t) + \alpha [G_t - V(S_t)]
\end{align}

The problem is that episodes may take a long time to terminate or never terminate. 

The insight of TD Learning, is that the future value is
>  "not confirmed or disconfirmed all at once, but rather a step at a time" (Sutton, 1988). 

Representing this intuition mathematically, the estimation error, $G_t - V(S_t)$, where $G_t$ are accumulated rewards starting from time step $t$ until termination following the policy, can be represented as a sum of differences on adjacent time steps from $t$ until $T$ termination.

\begin{align}
V(S_t) &= V(S_t) + \alpha[\sum_{k=t}^T (R_{k+1} + \gamma V(S_{k+1}) - V(S_k))] 
\end{align}


The equivalence of Eq (2) to Eq (1) Monte-Carlo Learning, can be seen by expanding and
cancelling out all the terms $V(S)$ terms, and accumulating the rewards into $G_t$. The update
rule is, in effect, updating the current state by using the difference in the successive prediction
values, hence the name Temporal-Difference learning. 

<br>

### **Flavours of TD**

In $TD(n)$, where $0 < n < T$, $n$ is how many time steps we consider for this update.

\begin{align}
V(S_t) = V(S_t) + \alpha [\sum_{k=t}^{T=k+1+n} (R_{t+1} + \gamma V(S_{k+1}) - V(S_k))]
\end{align}

$TD(0)$ update is performed based on a single time step

\begin{align}
V(S_t) = V(S_t) + \alpha(R_{t+1} + \gamma V(S_{t+1}) - V(S_t))
\end{align}

In $TD(\lambda)$, where $0 < \lambda < 1$, $\lambda$ is a weighted mixture between $TD(0)$ and
full monte-carlo learning, i.e. $TD(T)$.

\begin{align}
V(S_t) = V(S_t) + \alpha (G_t ^{\lambda} - V(S_t))
\end{align}

where $G_t^{\lambda}$ is a geometric weighting function on the reward:

\begin{align}
G_t^{\lambda}= (1-\lambda) \sum_{k=1}^{T-(t+1)} \lambda^{k-1} G_{t:t+k} + (1-\lambda)
\lambda^{T-(t+1)}G_t
\end{align}

This equation looks complicated but it's just a decay on the reward at future time steps.


<br>

**Analysis:** Is TD "better" than MC?

1. TD is a mathematical relaxation of Monte-carlo learning. Both will converge asymptotically to the correct predictions. I dont know if there is a mathematical proof that one is better than another, although TD typically should converge faster than MC methods on stochastic tasks, especially for episodes which take longer to terminate. 

2. TD can introduce bias because they rely on value estimates, if the value estimates are poor,
   inaccuracies in the value function are propagated through the updates. MC have unbiased
estimates of the value function. 

3. TD has lower variance than MC because the updates are more frequent and based on smaller
   steps. Instead, MC updates can vary significantly, and can also be influenced by random
events that happen throughout the entire episode.

<br>

### **Evolution of TD towards SARSA, Q-learning, Deep Q-learning**

Other practical and theoretical considerations of value-based RL methods subsequently shaped TD
towards SARSA, and Q-Learning. 

**SARSA** is essentially $TD(0)$ (Equation 4), but with state-action values. Instead of $V(S)$, we are now trying to update $Q(S, A)$. Since we are trying to choose the best action, it is more direct to just learn state-action values instead of performing an additional step to find the desired action.

\begin{align}
Q(S_t, A_t) = Q(S_t, A_t) + \alpha(R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t))
\end{align}


Learning state-action values is almost the same as learning state values. Computationally it is
a more direct approach for policy control, to learn state-action values as the state-action
value already includes the expected reward of taking that action. 

If we only had $V(s)$, we still need a model of the environment to figure out which action would lead to the next state with high values.

However in terms of memory,
the agent must store a much larger number of State $\times$ Action, vs just storing States. 


<br>

**Q-learning** is essentially **SARSA** but with an *offline policy*. It was invented because
sometimes the agent follows an exploratory policy (for e.g., $\epsilon$ greedy), and therefore does not always take the best action. Therefore, the *offline policy* update says regardless of what action was taken in the behavior policy, anyway, compute the temporal difference error using the *best action from the target policy*, i.e., $\max_{A_{t+1}}Q(S_{t+1}, A_{t+1})$. 

\begin{align}
Q(S_t, A_t) = Q(S_t, A_t) + \alpha (R_{t+1} + \gamma \max_{A_{t+1}} Q(S_{t+1}, A_{t+1}) - Q(S_t,
A_t))
\end{align}


The key is $\max_{A+1}$, which indicates that the update will be performed based on the best Action, rather than the actual action taken. 



<br>

**Deep Q-learning** is essentially **Q-learning**, but using *function approximation* and
gradient descent instead of tabular methods to learn the Q-values. 

Tabular methods are implemented by finding the values for a grid of $S \times A$ cells. The only
way an agent can get a good state-value function, is to explore every action in every state,
otherwise the state can never be updated. This limitation can be addressed with function
approximation using Deep-Q networks, which allow the value function or policy to generalise
across states, rather than updating each state individually.

<br>

*Author Note:* RL is a very vast and deep subject. This post was written in silent protest against online tutorials which just say here's the formula, here's a jupyter notebook. Let's train a Deep Q-network.

<br>

#### References
[RL University of Alberta](https://w.coursera.org/specializations/reinforcement-learning/) \\
[RL Textbook](http://incompleteideas.net/book/the-book.html) \\
[Old Stanford Course 
notes](https://web.stanford.edu/group/pdplab/pdphandbook/handbookch10.html#:~:text=We%20use%20the%20terms%20'supervised,the%20function%20making%20the%20prediction.)
