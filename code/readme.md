## Vulnerability-Aware Poisoning Mechanism for Online RL with Unknown Dynamics

**Abstract**: Poisoning attacks on Reinforcement Learning (RL) systems could take advantage of RL algorithm’s vulnerabilities and cause failure of the learning. However, prior works on poisoning RL usually either unrealistically assume the attacker knows the underlying Markov Decision Process (MDP), or directly apply the poisoning methods in supervised learning to RL. In this work, we build a generic poisoning framework for online RL via a comprehensive investigation of heterogeneous poisoning models in RL. Without any prior knowledge of the MDP, we propose a strategic poisoning algorithm called Vulnerability-Aware Adversarial Critic Poison (VA2C-P), which works for most policy-based deep RL agents, closing the gap that no poisoning method exists for policy-based RL agents. VA2C-P uses a novel metric, stability radius in RL, that measures the vulnerability of RL algorithms. Experiments on multiple deep RL agents and multiple environments show that our poisoning algorithm successfully prevents agents from learning a good policy or teaches the agents to converge to a target policy, with a limited attacking budget.



## Code Information

We implement our proposed poisoning algorithm VA2C-P on VPG, PPO, A2C and ACKTR. Implementation for VPG and PPO is in folder "vpg_ppo", while implementation for A2C and ACKTR is in folder "a2c_acktr".