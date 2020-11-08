# VA2CP: Poisoning Reinforcement Learning

This is a implementation of VA2C-P on VPG and PPO. 

Requirements: 

1. create the spinningup environment https://spinningup.openai.com/en/latest/user/installation.html
2. (optional) install mujoco-py https://github.com/openai/mujoco-py

## Run Sample Experiments:

We provide an example script to simply test the attacking algorithm on CartPole-v0 environment. Please run

> bash run_example.sh



## Pre-trained results

See *sample\_results* for sample results (of per-episode reward).

Name rule:

\{ENVIRONMNET\}\\_\{LEARNER\}\_n\{EPISODES\}\_\{POISON ALGORITHM\}\_s0.05\_m10\_r\{POWER\}\_f\{FRACTION\}\_run0.txt

where 

- POISON ALGORITHM=wb means VA2C-P,
- POISON ALGORITHM=semirand means AC-P
- POISON ALGORITHM=rand means Random Poisoning



In addition, files under folder *log* record the poisoned and unpoisoned data at each iteration.

## Customize Experiments

Using command 

>  python main.py 

to preform attacking, which has options

> --env: name of gym environment
>
> --no-attack: if don't want to attack
>
> --compute: if want to compute policy discrepancies
>
> --type(=wb,rand,semirand,bb,targ): wb for white-box VA2C-P / rand for random poisoning / semirand for AC-P / bb for black-box VA2C-P / targ for targeted attack (the targeted attack is currently only implemented for discreate actions)
>
> --aim(=reward,obs,action): the poison aim
>
> --radius: attack power ($\epsilon$ in paper)
>
> --radius-s (only used for aim=hybrid): attack power for state
>
> --radius-a (only used for aim=hybrid): attack power for action
>
> --radius-r (only used for aim=hybrid): attack power for reward
>
> --frac: attack for what fraction ($C/K$ in paper)
>
> --episodes: maximum episodes (our results use 1000 for CartPole and Walker2d, 3000 for Hopper, 2000 for HalfCheetah)
>
> --steps: maximum steps per episodes (our results use 300 for CartPole, Walker2d and HalfCheetah, 1000 for Hopper)
>
> --device: device to use (cpu or cuda:0)
>
> --algo(=vpg, ppo): the learner's algorithm, default is vpg