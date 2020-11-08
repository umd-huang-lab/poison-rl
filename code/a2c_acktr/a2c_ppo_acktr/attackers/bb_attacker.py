import sys
import torch  
import numpy as np  
import random
import math
import copy
import torch.optim as optim
import torch.nn as nn
from gym.spaces import Box, Discrete
from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.attackers.attack_model import Value

class BbAttacker:
    def __init__(self, learner, envs, maxat, maxupd, args, device="cpu", hidden_sizes=(64,64), 
                 activation=nn.Tanh, rand_select=False):
        super(BbAttacker, self).__init__()
        
        self.args = args
        
        self.learner = learner
        self.gamma = args.gamma
        self.device = device
        
        self.radius = args.radius
        self.frac = args.frac
        self.stepsize = args.stepsize
        self.maxiter = args.maxiter
        self.maxat = maxat
        self.maxupd = maxupd
        self.delta = args.delta
        self.dist_thres = args.dist_thres
        self.rand_select = rand_select
        self.disc_action = isinstance(envs.action_space, Discrete)
        if self.disc_action:
            self.action_dim = envs.action_space.n
        
        attack_policy = Policy(
            envs.observation_space.shape,
            envs.action_space,
            base_kwargs={'recurrent': args.recurrent_policy})
        attack_policy.to(device)
        
        my_policy = Policy(
            envs.observation_space.shape,
            envs.action_space,
            base_kwargs={'recurrent': args.recurrent_policy})
        my_policy.to(device)
        
        if isinstance(learner, algo.A2C_ACKTR):
            self.my_learner = algo.A2C_ACKTR(
                my_policy,
                args.value_loss_coef,
                args.entropy_coef,
                lr=args.lr,
                eps=args.eps,
                alpha=args.alpha,
                max_grad_norm=args.max_grad_norm,
                acktr=learner.acktr)
            self.im_learner = algo.A2C_ACKTR(
                attack_policy,
                args.value_loss_coef,
                args.entropy_coef,
                lr=args.lr,
                eps=args.eps,
                alpha=args.alpha,
                max_grad_norm=args.max_grad_norm,
                acktr=learner.acktr)
        
        elif isinstance(learner, algo.PPO):
            self.my_learner = algo.PPO(
                my_policy,
                args.clip_param,
                args.ppo_epoch,
                args.num_mini_batch,
                args.value_loss_coef,
                args.entropy_coef,
                lr=args.lr,
                eps=args.eps,
                max_grad_norm=args.max_grad_norm)
            self.im_learner = algo.PPO(
                attack_policy,
                args.clip_param,
                args.ppo_epoch,
                args.num_mini_batch,
                args.value_loss_coef,
                args.entropy_coef,
                lr=args.lr,
                eps=args.eps,
                max_grad_norm=args.max_grad_norm)
            
#        self.my_learner.actor_critic.load_state_dict(self.learner.actor_critic.state_dict())
#        self.my_learner.optimizer.load_state_dict(self.learner.optimizer.state_dict())
        
        self.critic = Value(envs.observation_space.shape[0], hidden_sizes, activation).to(device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=args.lr)
        
        self.dist_list = np.array([])
        self.attack_num = 0
        self.update_num = 0
        
        self.state_buffer = None
        self.state_buffer_limit = 100
    
    def store_states(self, states):
#        print("origin buffer", self.state_buffer)
#        print("states", states)
        if self.state_buffer is None:
            self.state_buffer = states.clone().detach()
        else:
            self.state_buffer = torch.cat([self.state_buffer, states.clone().detach()])
#        print("new buffer", self.state_buffer)  
        if self.state_buffer.size()[0] > self.state_buffer_limit:
            self.state_buffer = self.state_buffer[-self.state_buffer_limit:]
#            print("cut buffer", self.state_buffer) 
    
    def get_dist_general(self):
        masks = torch.ones((self.state_buffer.size()[0], self.state_buffer.size()[1], 1))
        recurrent = torch.zeros((self.state_buffer.size()[0], self.state_buffer.size()[1], 1))
        if self.disc_action and self.args.algo != "acktr":
            return self.im_learner.actor_critic.get_dist(self.state_buffer, recurrent, masks)
        else:
            return self.get_dist(self.state_buffer, recurrent, masks)
    
    def cp_net(self):
        self.im_learner.actor_critic.load_state_dict(self.my_learner.actor_critic.state_dict())
        self.im_learner.optimizer.load_state_dict(self.my_learner.optimizer.state_dict())
    
    def get_dist(self, obs, recurrent, masks):
        num_steps = obs.size()[0]
        dists = []
        for i in range(num_steps):
            dist = self.im_learner.actor_critic.get_dist(obs[i], recurrent[i], masks[i])
            dists.append(dist)
        return dists
#         else:
#             return self.im_learner.actor_critic.get_dist(rollouts.obs, rollouts.recurrent_hidden_states, rollouts.masks)
    
    def dist_distance(self, dist1, dist2, method="mean"):
#         if self.args.algo == "acktr":
        dts = []
        num_steps = len(dist1)
        if method == "mean":
            for i in range(num_steps):
                dt = torch.distributions.kl.kl_divergence(dist1[i], dist2[i]).mean().item()
                dts.append(dt)
            return np.array(dts).mean()
        elif method == "max":
            for i in range(num_steps):
                dt = torch.distributions.kl.kl_divergence(dist1[i], dist2[i]).max().item()
                dts.append(dt)
            return np.array(dts).max()
#         else:
#             if method == "mean":
#                 return torch.distributions.kl.kl_divergence(dist1, dist2).mean()
#             elif method == "max":
#                 return torch.distributions.kl.kl_divergence(dist1, dist2).max()
    
    def attack_hybrid(self, rollouts, next_value, radius_s, radius_a, radius_r):
        if self.attack_num >= self.maxat:
            print("exceeds budget")
            return "noat", rollouts
        
        max_distance = 0
        aim = ""
        attack = None
        
        attack_s, s_distance = self.attack_s_general(rollouts, next_value, hybrid=True, radius=radius_s)
        if s_distance >= max_distance:
            aim = "obs"
            attack = attack_s
            max_distance = s_distance
            
        attack_a, a_distance = self.attack_a_general(rollouts, next_value, hybrid=True, radius=radius_a)
        if a_distance >= max_distance:
            aim = "action"
            attack = attack_a
            max_distance = a_distance
        
        attack_r, r_distance = self.attack_r_general(rollouts, next_value, hybrid=True, radius=radius_r)
        if r_distance >= max_distance:
            aim = "reward"
            attack = attack_r
            max_distance = r_distance
        
        self.dist_list = np.append(self.dist_list, np.array([max_distance]))
            
        frac = min((self.maxat - self.attack_num) / (self.maxupd - self.update_num),1)
        self.update_num += 1
        
        if max_distance >= np.quantile(self.dist_list, 1-frac):
            print("attack with frac", frac)
            self.attack_num += 1
            return aim, attack
        else:
            return "noat", rollouts
        
    
    
    def attack_r_general(self, rollouts, next_value, hybrid=False, radius=None):
        '''Attack with the current rollouts'''
        if self.attack_num >= self.maxat:
            print("exceeds budget")
            return rollouts.rewards
        if radius is None:
            radius = self.radius
            
        self.store_states(rollouts.obs)
        cur_r = rollouts.rewards.clone().detach()
        
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()
        
        # imitate the rollouts
        im_rollouts = copy.deepcopy(rollouts)
        
        im_rollouts.compute_returns(next_value, self.args.use_gae, self.args.gamma,
                                 self.args.gae_lambda, self.args.use_proper_time_limits)
        
        # update attacker's own value function
        advantages = self.update_value(im_rollouts)
#        for name, param in self.my_learner.actor_critic.named_parameters():
#            if name == "base.actor.2.weight":
#                print("my learner", name, param)
#        for name, param in self.learner.actor_critic.named_parameters():
#            if name == "base.actor.2.weight":
#                print("learner", name, param)
        # imitate the learner's behavior
        self.cp_net()
        _, old_log_probs, _ = self.evaluate_policy(im_rollouts, obs_shape, action_shape)
        self.im_learner.update(im_rollouts)
        _, new_log_probs, _ = self.evaluate_policy(im_rollouts, obs_shape, action_shape)

        ratios = torch.exp(new_log_probs.detach() - old_log_probs.detach()).view(num_steps, num_processes, 1)
        true_loss = - (ratios * advantages).mean()
#        print("true loss", true_loss)
        true_action_dists = self.get_dist_general() #self.get_dist(rollouts, num_steps)
#        print(true_action_dists.probs)
        # compute the gradients of rewards
        grads = torch.zeros(cur_r.size()).to(self.device)
        for step in range(num_steps):
            for proc in range(num_processes):
                cur_r[step][proc][0] += self.delta
                im_rollouts.rewards.copy_(cur_r)
                im_rollouts.compute_returns(next_value, self.args.use_gae, self.args.gamma,
                                 self.args.gae_lambda, self.args.use_proper_time_limits)
                
                # update using poisoned rewards
                self.cp_net()
                self.im_learner.update(im_rollouts)
                
                _, poison_log_probs, _ = self.evaluate_policy(im_rollouts, obs_shape, action_shape)
                ratios = torch.exp(poison_log_probs.detach() - old_log_probs.detach()).view(num_steps, num_processes, 1)
                poison_loss = - (ratios * advantages).mean()
#                print("poison_loss", poison_loss)
                grads[step][proc][0] = (poison_loss - true_loss) / self.delta # want it to be large
                cur_r = rollouts.rewards.clone().detach()
                
        if torch.norm(grads).item() > 0:
            cur_r = cur_r + self.radius * math.sqrt(num_steps * num_processes) * grads / torch.norm(grads).item()
            
#        print(grads)
        # update using poisoned rewards
        self.cp_net()
        im_rollouts.rewards.copy_(cur_r)
        im_rollouts.compute_returns(next_value, self.args.use_gae, self.args.gamma,
                         self.args.gae_lambda, self.args.use_proper_time_limits)
        self.im_learner.update(im_rollouts)
        
        poison_action_dists = self.get_dist_general() #self.get_dist(rollouts, num_steps)
#        print(poison_action_dists.probs)
        if self.disc_action and self.args.algo != "acktr":
            dist_distance = torch.distributions.kl.kl_divergence(true_action_dists, poison_action_dists).mean() 
        else:
            dist_distance = self.dist_distance(true_action_dists, poison_action_dists)
        print("distribution distance:", dist_distance)
        
        if hybrid: 
            return cur_r, dist_distance
        else:
            self.dist_list = np.append(self.dist_list, np.array([dist_distance]))
            
            frac = min((self.maxat - self.attack_num) / (self.maxupd - self.update_num),1)
            self.update_num += 1
            
            if not self.rand_select:
                if dist_distance >= np.quantile(self.dist_list, 1-frac):
                    print("attack with frac", frac)
                    self.attack_num += 1
                    return cur_r
                else:
                    print("not attack with frac", frac)
                    return rollouts.rewards
            else:
                if random.random() < frac:
                    print("random attack with frac", frac)
                    self.attack_num += 1
                    return cur_r
                else:
                    print("not random attack with frac", frac)
                    return rollouts.rewards
    
    def set_obs_range(self, low, high):
        self.obs_low = torch.tensor(low).float().to(self.device)
        self.obs_high = torch.tensor(high).float().to(self.device)
        print("low:", self.obs_low)
        print("high:", self.obs_high)
    
    def clip_obs(self, obs):
        return torch.max(torch.min(obs, self.obs_high), self.obs_low)
    
    def attack_s_general(self, rollouts, next_value, hybrid=False, radius=None):
        '''Attack with the current rollouts'''
        if self.attack_num >= self.maxat:
            print("exceeds budget")
            return rollouts.obs
        if radius is None:
            radius = self.radius
        self.store_states(rollouts.obs)
        cur_s = rollouts.obs.clone().detach()
#        print(cur_s)
#         print(rollouts.obs.size())
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()
        
        # imitate the rollouts
        im_rollouts = copy.deepcopy(rollouts)
        
        im_rollouts.compute_returns(next_value, self.args.use_gae, self.args.gamma,
                                 self.args.gae_lambda, self.args.use_proper_time_limits)
        
        # update attacker's own value function
        advantages = self.update_value(im_rollouts)
        
        # imitate the learner's behavior
        self.cp_net()
        _, old_log_probs, _ = self.evaluate_policy(im_rollouts, obs_shape, action_shape)
        self.im_learner.update(im_rollouts)
        _, new_log_probs, _ = self.evaluate_policy(im_rollouts, obs_shape, action_shape)
        ratios = torch.exp(new_log_probs.detach() - old_log_probs.detach()).view(num_steps, num_processes, 1)
        true_loss = - (ratios * advantages).mean()
        
        true_action_dists = self.get_dist_general() #self.get_dist(rollouts, num_steps)
        
        # compute the gradients of states
        grads = torch.zeros(cur_s.size()).to(self.device)
        for step in range(num_steps):
            for proc in range(num_processes):
                for s in range(obs_shape[0]):
                    cur_s[step][proc][s] += self.delta
                    im_rollouts.obs.copy_(cur_s)
                    im_rollouts.compute_returns(next_value, self.args.use_gae, self.args.gamma,
                                     self.args.gae_lambda, self.args.use_proper_time_limits)
                    
                    # update using poisoned rewards
                    self.cp_net()
                    self.im_learner.update(im_rollouts)
                    
                    _, poison_log_probs, _ = self.evaluate_policy(im_rollouts, obs_shape, action_shape)
                    ratios = torch.exp(poison_log_probs.detach() - old_log_probs.detach()).view(num_steps, num_processes, 1)
                    poison_loss = - (ratios * advantages).mean()
                    
                    grads[step][proc][s] = (poison_loss - true_loss) / self.delta # want it to be large
                    cur_s = rollouts.obs.clone().detach()
        
#        grads = torch.clamp(grads, min=-1, max=1)
#        print("norm", torch.norm(grads).item())
#        for step in range(num_steps):
#            for proc in range(num_processes):
#                if torch.norm(grads[step][proc]).item() > 0:
#                    cur_s[step][proc] = cur_s[step][proc] + self.radius * grads[step][proc] / torch.norm(grads[step][proc]).item()
#                    print("grad:", grads[step][proc])
#                print("after:", cur_s[step][proc])
#                    cur_s[step][proc] = self.clip_obs(cur_s[step][proc])
        if torch.norm(grads).item() > 0:
            cur_s = (cur_s + self.radius * math.sqrt(num_steps * num_processes) * grads / torch.norm(grads).item()).detach()
        cur_s = cur_s.detach()
#        print(cur_s)
        # update using poisoned rewards
        self.cp_net()
        im_rollouts.obs.copy_(cur_s)
        im_rollouts.compute_returns(next_value, self.args.use_gae, self.args.gamma,
                         self.args.gae_lambda, self.args.use_proper_time_limits)
        self.im_learner.update(im_rollouts)
        
        poison_action_dists = self.get_dist_general() #self.get_dist(rollouts, num_steps)
        
        if self.disc_action and self.args.algo != "acktr":
            dist_distance = torch.distributions.kl.kl_divergence(true_action_dists, poison_action_dists).mean() 
        else:
            dist_distance = self.dist_distance(true_action_dists, poison_action_dists)
        print("distribution distance:", dist_distance)
        
        if hybrid: 
            return cur_s, dist_distance
        else:
            self.dist_list = np.append(self.dist_list, np.array([dist_distance]))
            
            frac = min((self.maxat - self.attack_num) / (self.maxupd - self.update_num),1)
            self.update_num += 1
            
            if not self.rand_select:
                if dist_distance >= np.quantile(self.dist_list, 1-frac):
                    print("attack with frac", frac)
                    self.attack_num += 1
                    return cur_s
                else:
                    print("not attack with frac", frac)
                    return rollouts.obs
            else:
                if random.random() < frac:
                    print("random attack with frac", frac)
                    self.attack_num += 1
                    return cur_s
                else:
                    print("not random attack with frac", frac)
                    return rollouts.obs

    
    def attack_a_general(self, rollouts, next_value, hybrid=False, radius=None):
        '''Attack with the current rollouts'''
        if self.attack_num >= self.maxat:
            print("exceeds budget")
            return rollouts.actions
        if radius is None:
            radius = self.radius
        self.store_states(rollouts.obs)
        cur_a = rollouts.actions.clone().detach()
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()
        # imitate the rollouts
        im_rollouts = copy.deepcopy(rollouts)
        
        im_rollouts.compute_returns(next_value, self.args.use_gae, self.args.gamma,
                                 self.args.gae_lambda, self.args.use_proper_time_limits)
        
        # update attacker's own value function
        advantages = self.update_value(im_rollouts)
        
        # imitate the learner's behavior
        self.cp_net()
        _, old_log_probs, _ = self.evaluate_policy(im_rollouts, obs_shape, action_shape)
        self.im_learner.update(im_rollouts)
        _, new_log_probs, _ = self.evaluate_policy(im_rollouts, obs_shape, action_shape)
        ratios = torch.exp(new_log_probs.detach() - old_log_probs.detach()).view(num_steps, num_processes, 1)
        true_loss = - (ratios * advantages).mean()
        
        true_action_dists = self.get_dist_general() #self.get_dist(rollouts, num_steps)
        
        if self.disc_action:
            # compute the gradients of states
            grads = torch.zeros(cur_a.size()).to(self.device)
            for step in range(num_steps):
                for proc in range(num_processes):
                    cur_a[step][proc][0] = (cur_a[step][proc][0]+1) % self.action_dim
                    im_rollouts.actions.copy_(cur_a)
                    im_rollouts.compute_returns(next_value, self.args.use_gae, self.args.gamma,
                                     self.args.gae_lambda, self.args.use_proper_time_limits)
                    
                    # update using poisoned rewards
                    self.cp_net()
                    self.im_learner.update(im_rollouts)
                    
                    _, poison_log_probs, _ = self.evaluate_policy(im_rollouts, obs_shape, action_shape)
                    ratios = torch.exp(poison_log_probs.detach() - old_log_probs.detach()).view(num_steps, num_processes, 1)
                    poison_loss = - (ratios * advantages).mean()
#                    print("poison prob", torch.flatten(poison_log_probs.detach()))
                    grads[step][proc][0] = (poison_loss - true_loss)  
                    cur_a = rollouts.actions.clone().detach()
            thres = np.quantile(torch.flatten(grads).detach().cpu().numpy(), self.radius)
            for step in range(num_steps):
                for proc in range(num_processes):
                    if grads[step][proc][0] < 0 and grads[step][proc][0] < thres:
                        cur_a[step][proc][0] = (cur_a[step][proc][0] + 1) % self.action_dim
        else:
            grads = torch.zeros(cur_a.size()).to(self.device)
            for step in range(num_steps):
                for proc in range(num_processes):
                    for a in range(action_shape):
                        cur_a[step][proc][a] += self.delta
                        im_rollouts.actions.copy_(cur_a)
                        im_rollouts.compute_returns(next_value, self.args.use_gae, self.args.gamma,
                                         self.args.gae_lambda, self.args.use_proper_time_limits)
                        
                        # update using poisoned rewards
                        self.cp_net()
                        self.im_learner.update(im_rollouts)
                        
                        _, poison_log_probs, _ = self.evaluate_policy(im_rollouts, obs_shape, action_shape)
                        ratios = torch.exp(poison_log_probs.detach() - old_log_probs.detach()).view(num_steps, num_processes, 1)
                        poison_loss = - (ratios * advantages).mean()
                        
                        grads[step][proc][a] = (poison_loss - true_loss) / self.delta  
                        cur_a = rollouts.actions.clone().detach()
#            print("grads", grads)
#            grads = torch.clamp(grads, min=-1, max=1)
#            print("norm", torch.norm(grads).item())
#            for step in range(num_steps):
#                for proc in range(num_processes):
#                    if torch.norm(grads[step][proc]).item() > 0:
#                        cur_a[step][proc] = cur_a[step][proc] + self.radius * grads[step][proc] / torch.norm(grads[step][proc]).item()
            grads = torch.sign(grads)
            if torch.norm(grads).item() > 0:
                cur_a = (cur_a + self.radius * math.sqrt(num_steps * num_processes) * grads / torch.norm(grads).item()).detach()
        
#        print("diff", cur_a - rollouts.actions)
        # update using poisoned actions
        cur_a = cur_a.detach()
        self.cp_net()
        im_rollouts.actions.copy_(cur_a)
        im_rollouts.compute_returns(next_value, self.args.use_gae, self.args.gamma,
                         self.args.gae_lambda, self.args.use_proper_time_limits)
        self.im_learner.update(im_rollouts)
        
        poison_action_dists = self.get_dist_general() #self.get_dist(rollouts, num_steps)
        
        if self.disc_action and self.args.algo != "acktr":
            dist_distance = torch.distributions.kl.kl_divergence(true_action_dists, poison_action_dists).mean() 
        else:
            dist_distance = self.dist_distance(true_action_dists, poison_action_dists)
        print("distribution distance:", dist_distance)
        
        if hybrid: 
            return cur_a, dist_distance
        else:
            self.dist_list = np.append(self.dist_list, np.array([dist_distance]))
            
            frac = min((self.maxat - self.attack_num) / (self.maxupd - self.update_num),1)
            self.update_num += 1
            
            if not self.rand_select:
                if dist_distance >= np.quantile(self.dist_list, 1-frac):
                    print("attack with frac", frac)
                    self.attack_num += 1
                    return cur_a
                else:
                    print("not attack with frac", frac)
                    return rollouts.actions
            else:
                if random.random() < frac:
                    print("random attack with frac", frac)
                    self.attack_num += 1
                    return cur_a
                else:
                    print("not random attack with frac", frac)
                    return rollouts.actions
    
    def compute_disc(self, rollouts, next_value):
        
        cur_r = rollouts.rewards.clone().detach()
        
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()
        
        # imitate the rollouts
        im_rollouts = copy.deepcopy(rollouts)
        
        im_rollouts.compute_returns(next_value, self.args.use_gae, self.args.gamma,
                                 self.args.gae_lambda, self.args.use_proper_time_limits)
        
        # update attacker's own value function
        advantages = self.update_value(im_rollouts)
        
        # imitate the learner's behavior
        self.cp_net()
        _, old_log_probs, _ = self.evaluate_policy(im_rollouts, obs_shape, action_shape)
        self.im_learner.update(im_rollouts)
        _, new_log_probs, _ = self.evaluate_policy(im_rollouts, obs_shape, action_shape)
        ratios = torch.exp(new_log_probs.detach() - old_log_probs.detach()).view(num_steps, num_processes, 1)
        true_loss = - (ratios * advantages).mean()
        
        true_action_dists = self.get_dist(rollouts, num_steps)
        
        # compute the gradients of rewards
        grads = torch.empty_like(cur_r)
        for step in range(num_steps):
            for proc in range(num_processes):
                cur_r[step][proc][0] += self.delta
                im_rollouts.rewards.copy_(cur_r)
                im_rollouts.compute_returns(next_value, self.args.use_gae, self.args.gamma,
                                 self.args.gae_lambda, self.args.use_proper_time_limits)
                
                # update using poisoned rewards
                self.cp_net()
                self.im_learner.update(im_rollouts)
                
                _, poison_log_probs, _ = self.evaluate_policy(im_rollouts, obs_shape, action_shape)
                ratios = torch.exp(poison_log_probs.detach() - old_log_probs.detach()).view(num_steps, num_processes, 1)
                poison_loss = - (ratios * advantages).mean()
                
                grads[step][proc][0] = (poison_loss - true_loss) / self.delta # want it to be large
                cur_r = rollouts.rewards.clone().detach()
                
        if torch.norm(grads).item() > 0:
            cur_r = cur_r + self.radius * grads / torch.norm(grads).item()
            
            
        # update using poisoned rewards
        self.cp_net()
        im_rollouts.rewards.copy_(cur_r)
        im_rollouts.compute_returns(next_value, self.args.use_gae, self.args.gamma,
                         self.args.gae_lambda, self.args.use_proper_time_limits)
        self.im_learner.update(im_rollouts)
        
        poison_action_dists = self.get_dist(rollouts, num_steps)
        
        dist_distance = self.dist_distance(true_action_dists, poison_action_dists, num_steps)
        print("distribution distance:", dist_distance)
        
        return dist_distance.item()
    
    
    def compute_radius(self, rollouts, next_value):
        '''Compute upper bound of the stability radius'''
        
        cur_r = rollouts.rewards.clone().detach()
        
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()
        
        # imitate the rollouts
        im_rollouts = copy.deepcopy(rollouts)
        
        im_rollouts.compute_returns(next_value, self.args.use_gae, self.args.gamma,
                                 self.args.gae_lambda, self.args.use_proper_time_limits)
        
        # update attacker's own value function
        advantages = self.update_value(im_rollouts)
        
        # imitate the learner's behavior
        self.cp_net()
        _, old_log_probs, _ = self.evaluate_policy(im_rollouts, obs_shape, action_shape)
        self.im_learner.update(im_rollouts)
        _, new_log_probs, _ = self.evaluate_policy(im_rollouts, obs_shape, action_shape)
        ratios = torch.exp(new_log_probs.detach() - old_log_probs.detach()).view(num_steps, num_processes, 1)
        true_loss = - (ratios * advantages).mean()
        
        true_action_dists = self.get_dist(rollouts, num_steps)
#        print("t", true_action_dists[0])
        it = 0
        dist_distance = 0
        last_r = cur_r.clone().detach()
        
        while dist_distance < self.dist_thres:
            it += 1
            
            # compute the gradients of rewards
            grads = torch.empty_like(cur_r)
            for step in range(num_steps):
                for proc in range(num_processes):
                    cur_r[step][proc][0] += self.delta
                    im_rollouts.rewards.copy_(cur_r)
                    im_rollouts.compute_returns(next_value, self.args.use_gae, self.args.gamma,
                                     self.args.gae_lambda, self.args.use_proper_time_limits)
                    
                    # update using poisoned rewards
                    self.cp_net()
                    self.im_learner.update(im_rollouts)
                    
                    _, poison_log_probs, _ = self.evaluate_policy(im_rollouts, obs_shape, action_shape)
                    ratios = torch.exp(poison_log_probs.detach() - old_log_probs.detach()).view(num_steps, num_processes, 1)
                    poison_loss = - (ratios * advantages).mean()
                    
                    grads[step][proc][0] = (poison_loss - true_loss) / self.delta # want it to be large
                    cur_r = last_r.clone().detach()

            if torch.norm(grads).item() > 0:
                cur_r = (last_r + self.stepsize * grads / torch.norm(grads).item()).detach()

            # update using poisoned rewards
            im_rollouts.rewards.copy_(cur_r)
            im_rollouts.compute_returns(next_value, self.args.use_gae, self.args.gamma,
                             self.args.gae_lambda, self.args.use_proper_time_limits)
            self.cp_net()
            self.im_learner.update(im_rollouts)
            poison_action_dists = self.get_dist(rollouts, num_steps)
            
            dist_distance = self.dist_distance(true_action_dists, poison_action_dists, num_steps, method="max")
#            print("p", poison_action_dists[0])
#            print(true_action_dists.sample()[0])
#            print(poison_action_dists.sample()[0])
            print(torch.norm(cur_r - rollouts.rewards).item(), "distribution distance:", dist_distance)
        
            if it >= self.maxiter:
                return np.inf
            last_r = cur_r.clone().detach()
        
        return torch.norm(cur_r - rollouts.rewards).item()
    
    
    
    def evaluate_policy(self, rollouts, obs_shape, action_shape):
        value, old_log_probs, dist_entropy, _ = self.im_learner.actor_critic.evaluate_actions(
                    rollouts.obs[:-1].view(-1, *obs_shape),
                    rollouts.recurrent_hidden_states[0].view(
                        -1, self.im_learner.actor_critic.recurrent_hidden_state_size),
                    rollouts.masks[:-1].view(-1, 1),
                    rollouts.actions.view(-1, action_shape))
        return value, old_log_probs, dist_entropy
    
    def update_value(self, rollouts):
        obs_shape = rollouts.obs.size()[2:]
        num_steps, num_processes, _ = rollouts.rewards.size()

        values = self.critic(rollouts.obs[:-1].view(-1, *obs_shape))

        values = values.view(num_steps, num_processes, 1)

        MseLoss = nn.MSELoss()
        loss = MseLoss(values, rollouts.returns[:-1])
        self.critic_optim.zero_grad()
        loss.mean().backward()
        self.critic_optim.step()
        
        new_values = self.critic(rollouts.obs[:-1].view(-1, *obs_shape)).view(num_steps, num_processes, 1)
        
        return rollouts.returns[:-1] - new_values
    
    def compute_returns(self, returns, next_value, rewards, masks):
        
        returns[-1] = next_value
        for step in reversed(range(rewards.size(0))):
            returns[step] = returns[step + 1] * \
                self.gamma * masks[step + 1] + rewards[step]
        return returns
    
            
    def proj(self, old_r, new_r, radius):
        
        norm = torch.norm(new_r-old_r)
        print("dist of r:", norm)
        proj_r = (old_r + (new_r - old_r) * radius / norm)
        return proj_r
    
    
    def learning(self, rollouts):
        self.my_learner.update(rollouts)