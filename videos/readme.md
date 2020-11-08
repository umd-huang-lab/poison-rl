## Vulnerability-Aware Poisoning Mechanism for Online RL with Unknown Dynamics



There are 3 .mp4 files in this folder, showing the comparison of our proposed poisoning algorithm and the random poisoning method.

We train 3 A2C agents with the same hyper-parameters on Hopper-v2, under different poisoning methods with the same attack budget and power. The videos show how the trained agents perform in the test phase.

*Note: the goal of the agent in Hopper is to hop forward as fast as possible.*



*File1:a2c_hopper_no_poison.mp4* is a baseline showing the original not poisoned agent.

*File2:a2c_hopper_random_poison_epsilon=0.1_frac=0.3.mp4* shows the agent under random poisoning attack with $\epsilon=0.1$, $C/K=0.3$.

*File2:a2c_hopper_random_poison_epsilon=0.1_frac=0.3.mp4* shows the agent under our proposed VA2C-P attack with $\epsilon=0.1$, $C/K=0.3$.



**Conclusion**: the random poisoning does not make the policy worse (according to the numerical results, the policy is even better. See paper Section 6 for details and analysis.) In constrast, our proposed poisoning algorithm successfully prevent the agent from hopper forward by poisoning only 30% iterations. 