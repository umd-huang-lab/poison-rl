import copy
import glob
import os
import time
from collections import deque
import logging
from datetime import datetime

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.attackers.rand_attacker import RandAttacker
from a2c_ppo_acktr.attackers.wb_attacker import WbAttacker
from a2c_ppo_acktr.attackers.bb_attacker import BbAttacker
from a2c_ppo_acktr.attackers.targ_attacker import TargAttacker
from a2c_ppo_acktr.attackers.fgsm_attacker import FGSMAttacker
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate
from torch.distributions import Categorical, MultivariateNormal
from gym.spaces import Box, Discrete

now = datetime.now()
current_time = now.strftime("%m-%d %H:%M:%S")

def get_log(file_name):
    logger = logging.getLogger('train') 
    logger.setLevel(logging.INFO) 

    fh = logging.FileHandler(file_name, mode='a') 
    fh.setLevel(logging.INFO) 
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)  
    return logger

def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:"+str(args.cuda_id) if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                args.env_name.split('-')[0].lower()))
        
        expert_dataset = gail.ExpertDataset(
            file_name, num_trajectories=4, subsample_frequency=20)
        drop_last = len(expert_dataset) > args.gail_batch_size
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)


    ########## file related 
    filename = args.env_name + "_" + args.algo + "_n" + str(args.max_episodes)
    if args.attack:
        filename += "_" + args.type + "_" + args.aim
        filename += "_s" + str(args.stepsize) + "_m" + str(args.maxiter) + "_r" + str(args.radius) + "_f" + str(args.frac)
    if args.run >=0:
        filename += "_run" + str(args.run)
        
    logger = get_log(args.logdir + filename + "_" +current_time)
    logger.info(args)
    
    rew_file = open(args.resdir + filename + ".txt", "w")
    
    if args.compute:
        radius_file = open(args.resdir + filename + "_radius" + "_s" + str(args.stepsize) 
        + "_m" + str(args.maxiter) + "_th" + str(args.dist_thres) + ".txt", "w")
    if args.type == "targ" or args.type == "fgsm":
        targ_file = open(args.resdir + filename + "_targ.txt", "w")
    
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
            
    if args.type == "wb":
        attack_net = WbAttacker(agent, envs, int(args.frac*num_updates), num_updates, args, device=device)
    if args.type == "bb":
        attack_net = BbAttacker(agent, envs, int(args.frac*num_updates), num_updates, args, device=device)
    elif args.type == "rand":
        attack_net = RandAttacker(envs, radius=args.radius, frac=args.frac, maxat=int(args.frac*num_updates), device=device)
    elif args.type == "semirand":
        attack_net = WbAttacker(agent, envs, int(args.frac*num_updates), num_updates, args, device, rand_select=True)
    elif args.type == "targ":
        if isinstance(envs.action_space, Discrete):
            action_dim = envs.action_space.n
            target_policy = action_dim - 1
        elif isinstance(envs.action_space, Box):
            action_dim = envs.action_space.shape[0]
            target_policy = torch.zeros(action_dim)
#            target_policy[-1] = 1
        print("target policy is", target_policy)
        attack_net = TargAttacker(agent, envs, int(args.frac*num_updates), num_updates, target_policy, args, device=device)
    elif args.type == "fgsm":
        if isinstance(envs.action_space, Discrete):
            action_dim = envs.action_space.n
            target_policy = action_dim - 1
        elif isinstance(envs.action_space, Box):
            action_dim = envs.action_space.shape[0]
            target_policy = torch.zeros(action_dim)
        def targ_policy(obs):
            return target_policy
        attack_net = FGSMAttacker(envs, agent, targ_policy, radius=args.radius, frac=args.frac, maxat=int(args.frac*num_updates), device=device)
#    if args.aim == "obs" or aim == "hybrid":
#        obs_space = gym.make(args.env_name).observation_space
#        attack_net.set_obs_range(obs_space.low, obs_space.high)
    
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    episode = 0

    start = time.time()
    
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            if args.type == "fgsm":
#                print("before", rollouts.obs[step])
                rollouts.obs[step] = attack_net.attack(rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step]).clone()
#                print("after", rollouts.obs[step])
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])
            if args.type == "targ" or args.type == "fgsm":
                if isinstance(envs.action_space, Discrete):
                    num_target = (action == target_policy).nonzero()[:,0].size()[0]
                    targ_file.write(str(num_target / args.num_processes) + "\n")
                    print("percentage of target:", num_target / args.num_processes)
                elif isinstance(envs.action_space, Box):
                    target_action = target_policy.repeat(action.size()[0], 1)
                    targ_file.write(str(torch.norm(action-target_action).item() / args.num_processes) + "\n")
                    print("percentage of target:", torch.sum(action).item() / args.num_processes)
            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action.cpu())
            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
#                    rew_file.write("episode: {}, total reward: {}\n".format(episode, info['episode']['r']))
                    episode += 1

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        if args.gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step])

        if args.attack and args.type != "fgsm":
            if args.aim == "reward":
                logger.info(rollouts.rewards.flatten())
                rollouts.rewards = attack_net.attack_r_general(rollouts, next_value).clone().detach()
                logger.info("after attack")
                logger.info(rollouts.rewards.flatten())
            elif args.aim == "obs":
                origin = rollouts.obs.clone()
                rollouts.obs = attack_net.attack_s_general(rollouts, next_value).clone().detach()
                logger.info(origin)
                logger.info("after")
                logger.info(rollouts.obs)
            elif args.aim == "action":
                origin = torch.flatten(rollouts.actions).clone()
                rollouts.actions = attack_net.attack_a_general(rollouts, next_value).clone().detach()
                logger.info("attack value")
                logger.info(torch.flatten(rollouts.actions) - origin)
            elif args.aim == "hybrid":
                res_aim, attack = attack_net.attack_hybrid(rollouts, next_value, args.radius_s, args.radius_a, args.radius_r)
                print("attack ", res_aim)
                if res_aim == "obs":
                    origin = rollouts.obs.clone()
                    rollouts.obs = attack.clone().detach()
                    logger.info(origin)
                    logger.info("attack obs")
                    logger.info(rollouts.obs)
                elif res_aim == "action":
                    origin = torch.flatten(rollouts.actions).clone()
                    rollouts.actions = attack.clone().detach()
                    logger.info("attack action")
                    logger.info(torch.flatten(rollouts.actions) - origin)
                elif res_aim == "reward":
                    logger.info(rollouts.rewards.flatten())
                    rollouts.rewards = attack.clone().detach()
                    logger.info("attack reward")
                    logger.info(rollouts.rewards.flatten())
        if args.compute:
            stable_radius = attack_net.compute_radius(rollouts, next_value)
            print("stable radius:", stable_radius)
            radius_file.write("update: {}, radius: {}\n".format(j, np.round(stable_radius, decimals = 3)))
        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        
        if args.attack and args.type == "bb":
            attack_net.learning(rollouts)
        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) >= 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))
            rew_file.write("updates: {}, mean reward: {}\n".format(j, np.mean(episode_rewards)))

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)
        
#        if episode > args.max_episodes:
#            print("reach episodes limit")
#            break
    
    if args.attack:
        logger.info("total attacks: {}\n".format(attack_net.attack_num))
        print("total attacks: {}\n".format(attack_net.attack_num))
        
    rew_file.close()
    if args.compute:
        radius_file.close()
    if args.type == "targ" or args.type == "fgsm":
        targ_file.close()

if __name__ == "__main__":
    main()
