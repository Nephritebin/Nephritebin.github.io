---
layout:     post
title:      "Learning-based quadruped robot visual locomotion"
subtitle:   " \"Citius, Altius, Fortius – Communiter\""
date:       2023-12-23 12:00:00
author:     "Yubin"
header-img: "img/Headers/yoru.jpg"
mathjax: true
catalog: true
scholar:
    bibliography: rinko.bib
tags:
    - Reinforcement Learning
    - Robotics
    - Deep Learning
---

This is a summary of my presentation in my first Rinko at the University of Tokyo. Rinko (輪講) is a course for each graduate student to join mandatorily until their graduation in the EEIS Department. In this course, students need to prepare a presentation each semester to show their survey of research topics, research progress, and results. In my first Rinko, I chose the topic of **learning-based quadruped robot visual locomotion**.

# Abstract

Locomotion presents a fundamental and ubiquitous challenge in the field of quadruped robot research. Recent advancements in robot hardware and reinforcement learning technologies have enabled these robots to exhibit agile locomotion within complex environments compared with traditional optimal control-based methods. This integration is achieved through sophisticated sim-to-real methods such as hierarchical formulation or distillation techniques. This survey aims to introduce the fundamental two-stage training framework for quadruped robot locomotion control via reinforcement learning, along with several notable achievements derived from this approach. Additionally, it will evaluate recent progress in this area and explore potential directions for future research.

# Introduction

Legged robots show great potential to navigate in complex and challenging environments. Compared with wheeled robots, legged robots have the ability to traverse complex terrain, such as stairs, slopes, and uneven ground. With a sufficiently diminutive size and also a considerable stature that enables the quadruped robot to carry the requisite payload, three prominent and highly recognized robots, [Spot](https://www.bostondynamics.com/products/spot), [Unitree series](https://shop.unitree.com/), and [Anymal](https://www.anybotics.com/robotics/anymal/), show great success in numerous applications. 

The empirical demonstration of quadruped robots' maneuverability across diverse terrains establishes their commendable performance. Various methodologies are utilized to enable quadruped robots to navigate diverse terrains, with a notable focus in recent years being on learning-based approaches {% cite lee2020learning %}{% cite kumar2021rma %}{% cite agarwal2023legged %}{% cite smith2023learning %}{% cite feng2023genloco %}{% cite cheng2023extreme %}.  The following figure exemplifies the captured action sequences using a single neural network operating directly on depth from a single, front-facing camera for governing locomotion in a quadruped robot{% cite cheng2023extreme %}. With such methods, these robots are capable of performing walking, trotting, climbing, and jumping in various indoor and outdoor environments.

<figure>
    <img width="100%" align="middle" src="/img/Notes/2023-12/parkour.png" style="margin-top: 0px; margin-bottom: 5px"/>
    <div style="font-size: 12px; text-align: center; margin-top: 0px;">
    A Unitree A1 robot does parkour with a single neural network directly on depth images. The robot is able to long jump across gaps $2\times$ of its own length, high jump over obstacles $2\times$ its own height, and run over tilted ramps.{% cite cheng2023extreme %}
    </div>
</figure>

To enable quadruped robots to perform agile maneuvers akin to their biological counterparts, like dogs or cats, it's crucial to explore and understand the robot's action space thoroughly. Inspired by curriculum learning, an incremental hardness training strategy is used to adapt the difficulty level for each terrain type individually {% cite rudin2022learning %}{% cite zhuang2023robot %}. Beyond merely enabling robots to walk or move on uneven terrain, some researchers are transforming quadruped robots into mobile platforms by attaching robotic arms, thus enabling them to perform various mobile manipulation tasks {% cite fu2023deep %}. Furthermore, the robot itself can also function as a manipulator, interacting with objects in various ways, such as pressing buttons, opening doors {% cite cheng2023legs}, or manipulating large objects {% cite jeon2023learning %}. 

Based on the above description, the robot can execute multiple action sequences with visual image inputs. Therefore, it is natural to consider linking these capabilities together to achieve fully agile navigation. Recently, some researchers decomposed the quadruped robot navigation problem into three parts and solved the navigation problem using a hierarchical network {% cite hoeller2023anymal %}. Others distillate the specialist skills to train a transformer-based generalist locomotion policy to finish a self-designed obstacle course benchmark {% cite caluwaerts2023barkour %}.

In this survey, we would like to introduce and discuss the following topics:
 - Some necessary background of reinforcement learning (RL) and imitation learning (IL) algorithms.
 - basic two-stage training framework of quadruped robot locomotion control using reinforcement learning.
 - Some variations and their comparison between visual input network structures.
 - More motions beyond just walking: after improving training environment configurations, we can let the robot perform more naturalistic and dexterous actions.
 - The utilization of existing locomotion techniques to achieve agile navigation.

The subsequent four chapters will detail these topics, each addressing a specific aspect. The final chapter will present the conclusion and outline avenues for future research.

# Background

## Reinforcement learning

Reinforcement Learning (RL) is a domain of machine learning where an agent learns to make decisions by interacting with an environment. Deep Learning (DL), on the other hand, has shown remarkable success in handling high-dimensional data, like images and natural language. It leverages neural networks with multiple layers to extract features and learn representations from large amounts of data. The integration of RL with DL, known as Deep Reinforcement Learning (DRL), combines the decision-making prowess of RL with the perception abilities of DL, allowing agents to learn from raw, high-dimensional data and make complex decisions.

Reinforcement learning always models the interaction of the robots and the world as a Partially Observed Markov Decision Process (POMDP). The POMDP is defined as a tuple $(\mathcal{S}, \mathcal{A}, \mathcal{O}, T, P, r, \gamma, \rho_0)$, where $\mathcal{S}$ is the state space, $\mathcal{A}$ is the action space, and $\mathcal{O}$ is the observation space. 
$T$ denotes the transition probability distribution which describes the model of the environment as $T(\textbf{s}_{t+1}|\textbf{s}_t, \textbf{a}_t)$. 
$P$ is the observation probability of the form $P(\textbf{o}_t|\textbf{s}_t)$ denotes the sensor model. $r: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$ denotes the reward function, $\gamma \in [0, 1]$ is a discount factor and $\rho_0$ is the initial state distribution. 

The agent interacts with the environment by taking actions 
$\textbf a_t\in \mathcal{A}$ at time step $t$, 
and receiving rewards and observations $(\textbf r_{t}, \textbf o_{t+1})$. 
This may repeat for several steps, and the agent will receive a trajectory: $\tau = \{(\textbf o_t, \textbf a_t, \textbf r_t, \textbf s_{t+1})|t=1,2...n \}$, where $T$ is the time horizon. Then the agents update their policy $\pi_\theta$ using the trajectories. 
The goal of reinforcement learning is to learn a distribution over actions conditions on observations $\textbf o_t$ or observation histories $\textbf o_{t:t-h}$ with the form $\pi_{\theta}(\textbf a_t|\textbf o_{t:t-h})$, which maximize the sum of the discounted rewards, which is formulated by:

$$
\pi_{\theta}(\tau) = p_{\theta}(\textbf{s}_1, \textbf{a}_1, ..., \textbf{s}_n, \textbf{a}_n) = p(\textbf{s}_1) \prod_{t=1}^n 
\pi_{\theta}(\textbf{a}_t|\textbf{o}_{t:t-h})
T(\textbf{s}_{t+1}|\textbf{s}_t, \textbf{a}_t)
$$

$$
    \theta^* = \arg\max_\theta \mathbb{E}_{\tau \sim p_{\theta}(\tau)} \left[ \sum_{t=1}^n \gamma^t r(\textbf{s}_t, \textbf{a}_t) \right]
$$

Here $\theta$ is the parameter of the policy $\pi_{\theta}(\textbf a_{t}|\textbf{o}_{t:t-h})$. 
In deep reinforcement learning, the policy is usually parameterized by a neural network.

## Imitation learning

Imitation learning, also known as behavioral cloning, is used to train a policy to imitate an expert's behavior, which may originally come from the automated driving area {% cite bojarski2016end %}. Given a dataset with observations and demonstrations, the policy is trained to minimize the difference between the output of the policy and the demonstration to imitate the expert.

However, one of the problems is that the policy may drift away from the expert's behavior due to the compounding error. For example, the agent may make a small mistake and be in a slightly different state from what it has trained before, thus diverging from the learned trajectory. To solve this problem, researchers propose a method called Dataset Aggregation (DAgger) {% cite ross2011reduction %}. In DAgger, the policy is trained on the dataset collected by the previous policy, which can be treated as a form of iterative imitation learning.

<figure>
    <img width="80%" align="middle" src="/img/Notes/2023-12/dagger.png" style="margin-top: 0px; margin-bottom: 5px"/>
    <div style="font-size: 12px; text-align: center; margin-top: 0px;">
    Dataset Aggregation (DAgger) algorithm.
    </div>
</figure>

# Two-stage training framework
# Conclusion

<!-- ---------------------------------------------------------------------------------------- -->
# References
{% bibliography --cited %}
