---
layout:     post
title:      "Deep Reinforcement Learning III"
subtitle:   "Model-Based Reinforcement Learning"
date:       2022-10-24 15:00:00
author:     "Yubin"
header-img: "img/Headers/grossgasteiger.jpg"
mathjax: true
catalog: true
tags:
    - Deep Learning
    - Reinforcement Learning
    - Optimal Control
    - Model-Based Reinforcement Learning
---

# Reinforcement Learning III

These are the methods that assume access to a known model of the system and use that model to make decisions. These models are looked very different from model free reinforcement learning algorithms. But we will find that some planning algorithms can be used in conjunction with learned models to make decisions more optimally.

## Recap and Introduction

Previously, we learned about algorithms that optimize reinforcement learning objective. The objective is to maximize the expected value:

$$
\underbrace{p_\theta\left(\mathbf{s}_1, \mathbf{a}_1, \ldots, \mathbf{s}_T, \mathbf{a}_T\right)}_{\pi_\theta(\tau)}=p\left(\mathbf{s}_1\right) \prod_{t=1}^T \pi_\theta\left(\mathbf{a}_t \mid \mathbf{s}_t\right) p\left(\mathbf{s}_{t+1} \mid \mathbf{s}_t, \mathbf{a}_t\right) \\

\theta^{\star}=\arg \max _\theta E_{\tau \sim p_\theta(\tau)}\left[\sum_t r\left(\mathbf{s}_t, \mathbf{a}_t\right)\right]
$$

In the algorithms we have learned so far, we assumed a model-free formulation, which meaning that we assume that $p\left(\mathbf{s}_{t+1} \mid \mathbf{s}_t, \mathbf{a}_t\right)$ is unknown, and we don't even attempt to learn it. So these were all algorithms that managed to get away with only sampling from that along full trajectories.

But what if we knew the transition dynamics? Actually often we do know the dynamics, for example, in Games (e.g., Atari games, chess, Go), easily modeled systems (e.g., navigating a car) or simulated environments (e.g., simulated robots, video games). 

In some other cases, although we don't know the dynamics, they might be fairly easy to learn. A very large domain in robotics, system identification, deals with fitting unknown parameters of a known model. You can also – fit a general-purpose model to observed transition data.

Model-based reinforcement learning refers to a way of approaching reinforcement learning problems where we first learn the transition dynamics, then figure out how to choose actions by using this transition dynamics. We will first talk about how can we choose the actions under perfect knowledge of the system dynamics, which is basically called optimal control or trajectory optimization. Then we go to the method to derive the unknown dynamics. Finally, we will talk about how can we then also learn policies? (e.g. by imitating optimal control)


## Optimal Control and Planning

During the deterministic case, the environment tells the robot what the state your robot is in,  and then the agent perform an optimization given their state $\mathbf{s}_1$, can they imagine a sequence of actions that will minimize the total cost:

![Open Loop Control](/img/Notes/2022-10/1.png)

$$
\mathbf{a}_1, \ldots, \mathbf{a}_T=\arg \max _{\mathbf{a}_1, \ldots, \mathbf{a}_T} \sum_{t=1}^T r\left(\mathbf{s}_t, \mathbf{a}_t\right) \text { s.t. } \mathbf{a}_{t+1}=f\left(\mathbf{s}_t, \mathbf{a}_t\right)
$$

