---
layout:     post
title:      "Reinforcement learning"
subtitle:   "Implementations and comparisons"
date:       2024-01-01 12:00:00
author:     "Yubin"
header-img: "img/Headers/mountains.jpg"
mathjax: true
catalog: true
published: false
tags:
    - Reinforcement Learning
    - Deep Learning
---

This is my report for the homeworks of the [*CS285: Deep Reinforcement Learning course*](http://rail.eecs.berkeley.edu/deeprlcourse/) in UC Berkeley. The homeworks are implemented in PyTorch. You can find the original starter-code in [here](https://github.com/berkeleydeeprlcourse/homework_fall2023) and my implementation at [here](https://github.com/Nephritebin/Course-UCB-CS285-DeepRL). If you have any questions or any copyright issues, please contact me via email yubinliu925@gmail.com.

# Deep reinforcement learning

## 1. Imitation Learning

The first homework is about imitation learning: behavior cloning and DAgger. Since the task is quite simple, I would also like to introduce the implementation of the whole reinforcement learning framework provided by the course staff.

### 1.1. Code structure

Basically, in reinforcement learning (Here is the computer simulation, not the real environment reinforcement learning), we may need to check three parts of the code: **environment simulation**, **the agent policy**, **the training algorithm** and then the whole process. Now I would like to introduce these four parts of the homeworks.

**Environment simulation**

This is the most interesting part during reinforcement learning, we may use several kinds of physical simulation softwares like isaac-gym to do the interaction between the simulator and the agent to imitate the real world and get feedbacks. No matter what kind of simulator we use, we need to can abstract them to a python class named `env`. In this class, we can do steps and get observations and rewards. In this homework, we use the `gym` as the simulator, and it has already provided the `env` class and some toy environments for us.  

We can make the environment by:
```python
import gym
# Make and reset the gym environment
env = gym.make(params['env_name'], render_mode=None)
env.reset(seed=seed)
```
And after we get an instance of the environment, we can check some of its properties:
```python
# Continuous or discrete action space
assert isinstance(env.action_space, gym.spaces.Box), "Environment must be continuous"
# Observation and action sizes
ob_dim = env.observation_space.shape[0]
ac_dim = env.action_space.shape[0]
```
During the learning process, we need the environment take actions and return the observations and rewards. We can do this by:
```python
# Step the env and store the transition
action = agent_policy.get_action(observation)
next_observation, reward, is_done, _ = env.step(action)
```
Everything related with the environment is in the `env` class, and in the homeworks, actually we don't care about the implementation of the environment. We just need to know how to use it.

**Agent policy**

Usually the agent policy is a neural network because of "deep" reinforcement learning. In this homework, we use the `torch.nn` and the neural network class architecture is a simply multi layer perceptron in `cs285/policies` named `BasePolicy`. We can instant the agent policy `MLPPolicySL` inheriting from `BasePolicy` by:
```python
actor = MLPPolicySL(ac_dim, ob_dim, params['n_layers'], params['size'],
        learning_rate=params['learning_rate'])
```
Similarly, the expert policy used in DAgger is also a neural network inheriting from `BasePolicy`. Something important is that the networks don't output the action directly, but the mean and the log standard deviation of the action distribution which we assume it as a normal distribution.
```python
self.logstd = nn.Parameter(torch.zeros(self.ac_dim, dtype=torch.float32))
def forward(self, observation: torch.FloatTensor) -> Any:
    # Defines the forward pass of the network
    mean = self.mean_net(observation)
    result = distributions.Normal(mean, torch.exp(self.logstd))
    return result
```
Notice that here the `log_std` is from `nn.Parameter` class, which means it is a learnable parameter. We can then update the network with the output distribution and designed loss or rewards. When we want to get the action, we can sample from the distribution:
```python
def get_action(self, observation):
    observation = ptu.from_numpy(observation)
    distribution = self.forward(observation)
    return ptu.to_numpy(distribution.sample()).reshape(1, -1)
```

**Training algorithm**

The training algorithm is how to update your policy network and it is the most important part of the reinforcement learning. In this homework, since it is **imitation learning**, we use the behavior cloning. We define the loss to update the network by the probablity of the expert action in the output distribution. The loss is defined in `cs285/policies/MLP_policy.py`:
```python
distribution = self.forward(observations)
loss = - distribution.log_prob(actions).sum()
```
And then update the network following traditional deep learning process.

**Whole process**

The whole process of behavior cloning and DAgger is summarized as follows:
```python
# Initialize environment, policy and logger
env = gym.make(params['env_name'], render_mode=None)
env.reset(seed=seed)
agent_policy, expert_policy = init_policy()
logger.init()

# Training loop
for itr in range(n_iter):
    if behavior_cloning:
        paths = load_trajectories(expert_data)
    else:
        # DAgger
        paths = sample_trajectories(env, agent_policy, params['batch_size'], params['ep_len'])
    # add collected data to replay buffer
    replay_buffer.add_rollouts(paths)
    # update the policy
    for step in range(train_steps):
        ob_batch, ac_batch = replay_buffer.random_sample()
        agent_policy.update(ob_batch, ac_batch)
        logger.perform_log()
```

### 1.2. Behavior cloning

Here is the table for the results of behavior cloning. We keep all the hyper parameters as default, except for the number of gradient steps for training policy, We set it as 10000 to make the agent learning a better policy. The results are shown in the table below:

|             | Evaluation Average Return | Evaluation Std | Training Average Return | Training Std | Return Percentage |
|:-----------:|:-------------------------:|:--------------:|:-----------------------:|:------------:|:-----------------:|
|     Ant     |          4519.165         |     971.522    |         4681.892        |    30.709    |       96.52%      |
| HalfCheetah |          4026.416         |     114.703    |         4034.800        |    32.868    |       99.79%      |
|    Hopper   |          2463.768         |     950.311    |         3717.513        |     0.353    |       66.27%      |
|   Walker2d  |          4494.833         |     1226.292    |         5383.310        |    54.152    |       83.50%      |

Then we do more test for the number of gradient steps for training policy based on the Walker2d environment. The results are shown in the figure below. You may find that the evaluation return is increasing with the number of gradient steps, but finally decreasing. I think this is because the agent is over-fitting to the expert data. The best number of gradient steps is 10000.

<figure>
    <img width="50%" align="middle" src="/img/Notes/2024-01/average_bc.png" style="margin-top: 0px; margin-bottom: 5px"/>
    <img width="50%" align="middle" src="/img/Notes/2024-01/std_bc.png" style="margin-top: 0px; margin-bottom: 5px"/>
    <div style="font-size: 12px; text-align: center; margin-top: 0px;">
    The average and std of evaluation return using behavior cloning with different number of gradient steps for training policy.
    </div>
</figure>

While average performance seems to be quite good, the standard deviation over the course of training is a bit more telling, as is the min/max returns. The agent continues to have trials where it makes a mistake and is unable to recover, resulting in a terrible rollout and a large standard deviation. If the agent was really learning to perform well in the environment we would see the standard deviation fall as it begins to consistently do well. This perfectly illustrates the weaknesses of behavioral cloning.

### 1.3. DAgger

<figure>
    <img width="50%" align="middle" src="/img/Notes/2024-01/dagger_average.png" style="margin-top: 0px; margin-bottom: 5px"/>
    <img width="50%" align="middle" src="/img/Notes/2024-01/dagger_std.png" style="margin-top: 0px; margin-bottom: 5px"/>
    <div style="font-size: 12px; text-align: center; margin-top: 0px;">
    The average and std of evaluation return using DAgger in different iterations.
    </div>
</figure>

