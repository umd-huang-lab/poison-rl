import torch
import argparse
import gym
#import mujoco_py
import numpy as np
from gym.spaces import Box, Discrete
import setup
import math
from poison_rl.memory import Memory
from poison_rl.agents.vpg import VPG
from poison_rl.agents.ppo import PPO
from poison_rl.attackers.wb_attacker import WbAttacker
from poison_rl.attackers.fgsm_attacker import FGSMAttacker
from poison_rl.attackers.targ_attacker import TargAttacker
from poison_rl.attackers.bb_attacker import BbAttacker
from poison_rl.attackers.rand_attacker import RandAttacker
from torch.distributions import Categorical, MultivariateNormal

import logging
from datetime import datetime

now = datetime.now()
current_time = now.strftime("%m-%d %H:%M:%S")

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default="cpu")
parser.add_argument('--run', type=int, default=-1)
# env settings
parser.add_argument('--env', type=str, default="CartPole-v0")
parser.add_argument('--episodes', type=int, default=1000)
parser.add_argument('--steps', type=int, default=300)

# learner settings
parser.add_argument('--learner', type=str, default="vpg", help="vpg, ppo, sac")
parser.add_argument('--lr', type=float, default=1e-3)

# attack settings
parser.add_argument('--norm', type=str, default="l2")
parser.add_argument('--stepsize', type=float, default=0.05)
parser.add_argument('--maxiter', type=int, default=10)
parser.add_argument('--radius', type=float, default=0.5)
parser.add_argument('--radius-s', type=float, default=0.5)
parser.add_argument('--radius-a', type=float, default=0.05)
parser.add_argument('--radius-r', type=float, default=0.5)
parser.add_argument('--frac', type=float, default=1.0)
parser.add_argument('--type', type=str, default="wb", help="rand, wb, semirand")
parser.add_argument('--aim', type=str, default="reward", help="reward, obs, action")

parser.add_argument('--attack', dest='attack', action='store_true')
parser.add_argument('--no-attack', dest='attack', action='store_false')
parser.set_defaults(attack=True)

parser.add_argument('--compute', dest='compute', action='store_true')
parser.add_argument('--no-compute', dest='compute', action='store_false')
parser.set_defaults(compute=False)

# file settings
parser.add_argument('--logdir', type=str, default="logs/")
parser.add_argument('--resdir', type=str, default="results/")
parser.add_argument('--moddir', type=str, default="models/")
parser.add_argument('--loadfile', type=str, default="")

args = parser.parse_args()

def get_log(file_name):
    logger = logging.getLogger('train') 
    logger.setLevel(logging.INFO) 

    fh = logging.FileHandler(file_name, mode='a') 
    fh.setLevel(logging.INFO) 
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)  
    return logger


if __name__ == '__main__':
    ############## Hyperparameters ##############
    env_name = args.env #"LunarLander-v2"
    
    max_episodes = args.episodes        # max training episodes
    max_steps = args.steps         # max timesteps in one episode
    attack = args.attack
    compute = args.compute
    attack_type = args.type
    learner = args.learner
    aim = args.aim
    
    stepsize = args.stepsize
    maxiter = args.maxiter
    radius = args.radius
    frac = args.frac
    lr = args.lr
    device = args.device
    ############ For All #########################
    gamma = 0.99                # discount factor
    random_seed = 0 
    render = False
    update_every = 300
    save_every = 100
    
    ########## creating environment
    env = gym.make(env_name)
    
    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)
    
    
    ########## file related 
    filename = env_name + "_" + learner + "_n" + str(max_episodes)
    if attack:
        filename += "_" + attack_type + "_" + aim
        filename += "_s" + str(stepsize) + "_m" + str(maxiter) + "_r" + str(radius) + "_f" + str(frac)
    if args.run >=0:
        filename += "_run" + str(args.run)
        
        
    logger = get_log(args.logdir + filename + "_" +current_time)
    logger.info(args)
    
    rew_file = open(args.resdir + filename + ".txt", "w")
    if attack_type == "targ" or attack_type == "fgsm":
        targ_file = open(args.resdir + filename + "_targ.txt", "w")
    if compute:
        radius_file = open(args.resdir + filename + "_radius" + "_s" + str(stepsize) + "_m" + str(maxiter) + "_r" + str(radius) + ".txt", "w")
    
    
    ########## create learner
    if learner == "vpg":
        policy_net = VPG(env.observation_space, env.action_space, gamma=gamma, device=device, learning_rate=lr)
    elif learner == "ppo":
        policy_net = PPO(env.observation_space, env.action_space, gamma=gamma, device=device, learning_rate=lr)
    
    
    ########## create attacker
    if attack_type == "wb":
        attack_net = WbAttacker(env.observation_space, env.action_space, policy_net, maxat=int(frac*max_episodes), maxeps=max_episodes,
                                gamma=gamma, learning_rate=lr, maxiter=maxiter, radius=radius, stepsize=stepsize, device=device)
    elif attack_type == "bb":
        attack_net = BbAttacker(env.observation_space, env.action_space, policy_net, maxat=int(frac*max_episodes), maxeps=max_episodes,
                                gamma=gamma, learning_rate=lr, maxiter=maxiter, radius=radius, stepsize=stepsize, device=device)
    elif attack_type == "rand":
        attack_net = RandAttacker(env.action_space, radius=radius, frac=frac, maxat=int(frac*max_episodes), device=device)
    elif attack_type == "semirand":
        attack_net = WbAttacker(env.observation_space, env.action_space, policy_net, maxat=int(frac*max_episodes), maxeps=max_episodes,
                                gamma=gamma, learning_rate=lr, maxiter=maxiter, radius=radius, stepsize=stepsize, rand_select=True)
    elif attack_type == "targ":
        if isinstance(env.action_space, Discrete):
            action_dim = env.action_space.n
            target_policy = action_dim - 1
        elif isinstance(env.action_space, Box):
            action_dim = env.action_space.shape[0]
            target_policy = torch.zeros(action_dim)
#            target_policy[-1] = 1
        print("target policy is", target_policy)
        
        attack_net = TargAttacker(env.observation_space, env.action_space, policy_net, maxat=int(frac*max_episodes), maxeps=max_episodes,
                                targ_policy=target_policy, gamma=gamma, learning_rate=lr, maxiter=maxiter, radius=radius, stepsize=stepsize, device=device)
    elif attack_type == "fgsm":
        if isinstance(env.action_space, Discrete):
            action_dim = env.action_space.n
            target_policy = action_dim - 1
        elif isinstance(env.action_space, Box):
            action_dim = env.action_space.shape[0]
            target_policy = torch.zeros(action_dim)
        def targ_policy(obs):
            return target_policy
        attack_net = FGSMAttacker(policy_net, env.action_space, targ_policy, radius=radius, frac=frac, maxat=int(frac*max_episodes), device=device)
    
    if aim == "obs" or aim == "hybrid":
        attack_net.set_obs_range(env.observation_space.low, env.observation_space.high)
    
    start_episode = 0
    # load learner from checkpoint
    if args.loadfile != "":
        checkpoint = torch.load(args.moddir + args.loadfile)
        print("load from ", args.moddir + args.loadfile)
        policy_net.set_state_dict(checkpoint['model_state_dict'], checkpoint['optimizer_state_dict'])
        start_episode = checkpoint['episode']
    
    memory = Memory()
    
    all_rewards = []
    timestep = 0
    update_num = 0
    
    ######### training
    for episode in range(start_episode, max_episodes):
        state = env.reset()
        rewards = []
        total_targ_actions = 0
        for steps in range(max_steps):
            timestep += 1
            
            if render:
                env.render()
                
            state_tensor, action_tensor, log_prob_tensor = policy_net.act(state)
            
            if isinstance(env.action_space, Discrete):
                action = action_tensor.item()
                if attack_type == "targ" or attack_type == "fgsm":
                    if action == target_policy:
                        total_targ_actions += 1
            else:
                action = action_tensor.cpu().data.numpy().flatten()
                if attack_type == "targ" or attack_type == "fgsm":
                    total_targ_actions += np.linalg.norm(action - target_policy.numpy()) ** 2
#            print(action, target_policy, total_targ_actions)
                
            new_state, reward, done, _ = env.step(action)
            
            if attack_type == "fgsm":
#                before_attack = new_state.copy()
#                print("before attack", new_state)
                new_state = attack_net.attack(new_state)
#                print("after attack", new_state)
#                print("attack norm", np.linalg.norm(before_attack - new_state))
            
            rewards.append(reward)
            
            memory.add(state_tensor, action_tensor, log_prob_tensor, reward, done)
            
            if done or steps == max_steps-1: #timestep % update_every == 0:
                if attack and attack_type != "fgsm":
                    if aim == "reward":
                        attack_r = attack_net.attack_r_general(memory)
                        logger.info(memory.rewards)
                        memory.rewards = attack_r.copy()
                        logger.info(memory.rewards)
                    elif aim == "obs":
                        attack_s = attack_net.attack_s_general(memory)
                        logger.info(torch.stack(memory.states).to(device).detach().cpu().numpy().tolist())
                        memory.states = attack_s.copy()
                        logger.info(torch.stack(memory.states).to(device).detach().cpu().numpy().tolist())
                    elif aim == "action":
                        attack_a = attack_net.attack_a_general(memory)
                        logger.info(torch.stack(memory.actions).to(device).detach().cpu().numpy().tolist())
                        memory.actions = attack_a.copy()
                        logger.info(torch.stack(memory.actions).to(device).detach().cpu().numpy().tolist())
                    elif aim == "hybrid":
                        res_aim, attack = attack_net.attack_hybrid(memory, args.radius_s, args.radius_a, args.radius_r)
                        print("attack ", res_aim)
                        if res_aim == "obs":
                            logger.info(memory.states)
                            memory.states = attack.copy()
                            logger.info(memory.states)
                        elif res_aim == "action":
                            logger.info(memory.actions)
                            memory.actions = attack.copy()
                            logger.info(memory.actions)
                        elif res_aim == "reward":
                            logger.info(memory.rewards)
                            memory.rewards = attack.copy()
                            logger.info(memory.rewards)
                    if attack_type == "bb": # and attack_net.buffer.size() > 128:
                        attack_net.learning(memory)
#                    print("attacker")
#                    attack_net.print_paras(attack_net.im_policy)
                if compute:
                    stable_radius = attack_net.compute_radius(memory)
                    print("stable radius:", stable_radius)
                    radius_file.write("episode: {}, radius: {}\n".format(episode, np.round(stable_radius, decimals = 3)))
                if attack_type == "targ" or attack_type == "fgsm":
                    if isinstance(env.action_space, Discrete):
                        targ_file.write(str(float(total_targ_actions) / steps) + "\n")
                        print("percent of target", float(total_targ_actions) / steps)
                    else:
                        targ_file.write(str(math.sqrt(total_targ_actions / steps)) + "\n")
                        print("average distance to target", math.sqrt(total_targ_actions / steps))
                policy_net.update_policy(memory)
#                print("learner")
#                attack_net.print_paras(policy_net.policy)
                memory.clear_memory()
                timestep = 0
                update_num += 1
                
            state = new_state
            
            if done or steps == max_steps-1:
                all_rewards.append(np.sum(rewards))
                logger.info("episode: {}, total reward: {}\n".format(episode, np.round(np.sum(rewards), decimals = 3)))
                rew_file.write("episode: {}, total reward: {}\n".format(episode, np.round(np.sum(rewards), decimals = 3)))
                break
        
        
        
        if (episode+1) % save_every == 0 and attack_type != "rand" and attack_type != "fgsm":
            path = args.moddir + filename
            torch.save({
                'episode': episode,
                'model_state_dict': policy_net.policy.state_dict(),
                'optimizer_state_dict': policy_net.optimizer.state_dict(),
                'attack_critic': attack_net.critic.state_dict(),
                'attack_critic_optim': attack_net.critic_optim.state_dict()
                }, path)
    if attack:
        logger.info("total attacks: {}\n".format(attack_net.attack_num))
        print("total attacks: {}\n".format(attack_net.attack_num))
        print("update number:", update_num)
        
    rew_file.close()
    if compute:
        radius_file.close()
    env.close()
            
