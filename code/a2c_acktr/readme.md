# VA2CP: Poisoning Reinforcement Learning

This is a implementation of VA2C-P on A2C and ACKTR. 

The base learning algorithms A2C and ACKTR are from an open-souce implementation <https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail>

```bash
# Install requirements
# PyTorch
conda install pytorch torchvision -c soumith

# Baselines for Atari preprocessing
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .

# Other requirements
pip install -r requirements.txt
```

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



## Customize Experiments

Using command 

>  python attack_main.py 

to preform attacking, which has options

> --env-name: name of gym environment
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
> --num-env-steps: total number of steps (default: 80000)
>
> --cuda-id: the id of gpu (use --no-cuda if using cpu)
>
> --algo(=a2c, acktr): the learner's algorithm, default is a2c