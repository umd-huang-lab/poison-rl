import sys
import torch  
import gym
import numpy as np  
import math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import random
import copy
from gym.spaces import Box, Discrete
from poison_rl.agents.model import Actor, ContActor, ActorCritic, ContActorCritic, Value
from poison_rl.agents.updates import vpg_update, ppo_update
from poison_rl.agents.vpg import VPG
from poison_rl.agents.ppo import PPO
from torch.distributions import Categorical, MultivariateNormal

class WbAttacker:
    def __init__(self, state_space, action_space, learner, maxat, maxeps, hidden_sizes=(64,64), 
                 activation=nn.Tanh, learning_rate=3e-4, gamma=0.9, device="cpu", 
                 stepsize=0.1, maxiter=1, radius=0.5, delta=10, dist_thres=0.1, rand_select=False,
                 obs_low=None, obs_high=None):
        super(WbAttacker, self).__init__()

        self.learner = learner
        self.gamma = gamma
        self.stepsize = stepsize
        self.max_iter = maxiter
        self.radius = radius
        self.maxat = maxat
        self.maxeps = maxeps
        self.device = device
        self.delta = delta
        self.dist_thres = dist_thres
        self.rand_select = rand_select
        self.obs_low = obs_low
        self.obs_high = obs_high
        self.disc_action = isinstance(action_space, Discrete)
        
        if isinstance(self.learner, VPG):
            self.init_imitator_vpg(state_space, action_space, hidden_sizes, activation, learning_rate)
            self.alg = "vpg"
        if isinstance(self.learner, PPO):
            self.init_imitator_ppo(state_space, action_space, hidden_sizes, activation, learning_rate)
            self.alg = "ppo"
            self.delta = 1
        
        self.critic = Value(state_space.shape[0], hidden_sizes, activation).to(self.device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        self.dist_list = np.array([])
        self.attack_num = 0
        self.eps_num = 0
        
        self.state_buffer = []
        self.state_buffer_limit = 1000

    
    def init_imitator_vpg(self, state_space, action_space, hidden_sizes, activation, learning_rate):
        '''Initialize attacker's policy and optimizer to imitate the learner's behaviors'''
        
        state_dim = state_space.shape[0]
        action_std = 0.5
        if isinstance(action_space, Discrete):
            self.action_dim = action_space.n
            self.im_policy = Actor(state_dim, self.action_dim, hidden_sizes, activation).to(self.device)
        elif isinstance(action_space, Box):
            self.action_dim = action_space.shape[0]
            self.im_policy = ContActor(state_dim, self.action_dim, hidden_sizes, activation, action_std, self.device).to(self.device)
        
        self.im_optimizer = optim.Adam(self.im_policy.parameters(), lr=learning_rate)
        
    
    def init_imitator_ppo(self, state_space, action_space, hidden_sizes, activation, learning_rate):
        '''Initialize attacker's policy and optimizer to imitate the learner's behaviors'''
        
        state_dim = state_space.shape[0]
        action_std = 0.5
        if isinstance(action_space, Discrete):
            self.action_dim = action_space.n
            self.im_policy = ActorCritic(state_dim, self.action_dim, hidden_sizes, activation).to(self.device)
        elif isinstance(action_space, Box):
            self.action_dim = action_space.shape[0]
            self.im_policy = ContActorCritic(state_dim, self.action_dim, hidden_sizes, activation, action_std, self.device).to(self.device)
        
        self.im_optimizer = torch.optim.Adam(self.im_policy.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        self.eps_clip = self.learner.eps_clip
        self.K_epochs = self.learner.K_epochs
    
    def set_obs_range(self, low, high):
        self.obs_low = torch.tensor([low]).float().to(self.device)
        self.obs_high = torch.tensor([high]).float().to(self.device)
        print("low:", self.obs_low)
        print("high:", self.obs_high)
    
    def clip_obs(self, obs):
        return torch.max(torch.min(obs, self.obs_high), self.obs_low)
    
    def store_states(self, states):
        if len(states) + len(self.state_buffer) > self.state_buffer_limit:
            self.state_buffer = self.state_buffer[len(states):]
        self.state_buffer += states
    
    def get_dist_general(self, policy):
        buf_states = torch.stack(self.state_buffer).to(self.device).detach()
        return self.im_policy.get_dist(buf_states, self.device)
    
    def attack_hybrid(self, memory, radius_s, radius_a, radius_r):
        if self.attack_num >= self.maxat:
            print("exceeds budget")
            return "noat", memory
        
        max_distance = 0
        aim = ""
        attack = None
        
        attack_s, s_distance = self.attack_s_general(memory, hybrid=True, radius=radius_s)
        if s_distance >= max_distance:
            aim = "obs"
            attack = attack_s
            max_distance = s_distance
            
        attack_a, a_distance = self.attack_a_general(memory, hybrid=True, radius=radius_a)
        if a_distance >= max_distance:
            aim = "action"
            attack = attack_a
            max_distance = a_distance
        
        attack_r, r_distance = self.attack_r_general(memory, hybrid=True, radius=radius_r)
        if r_distance >= max_distance:
            aim = "reward"
            attack = attack_r
            max_distance = r_distance
        
        
        self.dist_list = np.append(self.dist_list, np.array([max_distance]))
        
        frac = min((self.maxat - self.attack_num) / (self.maxeps - self.eps_num),1)
        self.eps_num += 1
        
        if max_distance >= np.quantile(self.dist_list, 1-frac):
            print("attack with frac", frac)
            self.attack_num += 1
            return aim, attack
        else:
            return "noat", memory
        
    
    def attack_a_general(self, memory, hybrid=False, radius=None):
#        actions = []
#        
#        for a in memory.actions:
#            if random.random() > 0.5:
#                actions.append(torch.tensor(1) - a)
#            else:
#                actions.append(a)
#
#        return actions
        
        if self.attack_num >= self.maxat:
            print("exceeds budget")
            return memory.actions 
        
        self.store_states(memory.states)
        if radius is None:
            radius = self.radius
        # convert list to tensor
        cur_a = torch.stack(memory.actions).to(self.device).detach()
        old_a = torch.stack(memory.actions).to(self.device).detach()
        old_states = torch.stack(memory.states).to(self.device).detach()
        old_discounted_rewards = self.get_discount_rewards(memory, memory.rewards).detach()
#        print("cur_a:", cur_a)
        
        if self.disc_action:
            T = cur_a.size()[0]
            
            advantages = self.update_value(old_states, old_discounted_rewards)
                    
            # copy weights from the learner
            old_logprobs = self.cp_net(old_states, old_a)
            
            # a normal update without poisoning
            self.update_policy(old_logprobs, memory.rewards, memory)
            
    #        self.print_paras(self.im_policy)
            true_action_dists = self.get_dist_general(self.im_policy) #self.im_policy.get_dist(old_states, self.device)
            
            new_logprobs = self.im_policy.act_prob(old_states, old_a, self.device)
            ratios = torch.exp(new_logprobs.detach() - old_logprobs.detach())
            true_obj = (ratios * advantages).mean()
            
            grads = np.zeros(T)
            
            for t in range(T):
                cur_a[t] = (cur_a[t] + 1) % self.action_dim
                # a imitating update with one poisoned reward
                old_logprobs = self.cp_net(old_states, cur_a)
                self.update_policy(old_logprobs, memory.rewards, memory)
    #            self.print_paras(self.im_policy)
                new_logprobs = self.im_policy.act_prob(old_states, cur_a, self.device)
                ratios = torch.exp(new_logprobs.detach() - old_logprobs.detach())
                poison_obj = (ratios * advantages).mean()
                grads[t] = poison_obj - true_obj
                cur_a = copy.deepcopy(old_a)
#            print("grads", grads)
            thres = np.quantile(np.array(grads), radius)
#            print("thres", thres)
            for t in range(T):
                if grads[t] < 0 and grads[t] < thres:
                    cur_a[t] = (cur_a[t] + 1) % self.action_dim
#            print("new a", cur_a)
            
        else:
            T = cur_a.size()[0]
            A = cur_a.size()[1]
            
            # update attacker's own value function
            advantages = self.update_value(old_states, old_discounted_rewards)
                    
            # copy weights from the learner
            old_logprobs = self.cp_net(old_states, old_a)
            
            # a normal update without poisoning
            self.update_policy(old_logprobs, memory.rewards, memory)
            
    #        self.print_paras(self.im_policy)
            true_action_dists = self.get_dist_general(self.im_policy) #self.im_policy.get_dist(old_states, self.device)
            
            new_logprobs = self.im_policy.act_prob(old_states, old_a, self.device)
            ratios = torch.exp(new_logprobs.detach() - old_logprobs.detach())
            true_obj = (ratios * advantages).mean()
            
            grads = np.zeros((T, A))
            
            for t in range(T):
                for s in range(A):
                    cur_a[t][s] += 0.1
                    # a imitating update with one poisoned reward
                    old_logprobs = self.cp_net(old_states, cur_a)
                    self.update_policy(old_logprobs, memory.rewards, memory)
        #            self.print_paras(self.im_policy)
                    new_logprobs = self.im_policy.act_prob(old_states, cur_a, self.device)
                    ratios = torch.exp(new_logprobs.detach() - old_logprobs.detach())
                    poison_obj = (ratios * advantages).mean()
                    grads[t][s] = poison_obj - true_obj
                    cur_a = copy.deepcopy(cur_a)
        
#            for t in range(T):
#                if np.linalg.norm(grads[t]) > 0:
#                    cur_a[t] = old_a[t] - torch.Tensor(radius * grads[t] / np.linalg.norm(grads[t]))
            if np.linalg.norm(grads) > 0:
                cur_a = old_a - torch.Tensor(radius * math.sqrt(T) * grads / np.linalg.norm(grads)).to(self.device)
#            print("new a", cur_a)
        
        # update use the new rewards
        old_logprobs = self.cp_net(old_states, cur_a)
        self.update_policy(old_logprobs, memory.rewards, memory)
        
        poison_action_dists = self.get_dist_general(self.im_policy) #self.im_policy.get_dist(old_states, self.device)
        
        dist_distance = torch.distributions.kl.kl_divergence(true_action_dists, poison_action_dists).mean()
        print("distribution distance:", dist_distance)
        
        if hybrid: 
            return [torch.tensor(a) for a in cur_a], dist_distance
        else:
            self.dist_list = np.append(self.dist_list, np.array([dist_distance]))
            
            frac = min((self.maxat - self.attack_num) / (self.maxeps - self.eps_num),1)
            self.eps_num += 1
            
            if not self.rand_select:
                if dist_distance >= np.quantile(self.dist_list, 1-frac):
                    print("attack with frac", frac)
                    self.attack_num += 1
                    return [torch.tensor(a) for a in cur_a]
                else:
                    print("not attack with frac", frac)
                    return memory.actions
            else:
                if random.random() < frac:
                    print("random attack with frac", frac)
                    self.attack_num += 1
                    return [torch.tensor(a) for a in cur_a]
                else:
                    print("not random attack with frac", frac)
                    return memory.actions
    
    def attack_s_general(self, memory, hybrid=False, radius=None):
        
        if self.attack_num >= self.maxat:
            print("exceeds budget")
            return memory.states
        
        self.store_states(memory.states)
        
        if radius is None:
            radius = self.radius
        # convert list to tensor
        cur_s = torch.stack(memory.states).to(self.device).detach()
        old_s = torch.stack(memory.states).to(self.device).detach()
        old_actions = torch.stack(memory.actions).to(self.device).detach()
        old_discounted_rewards = self.get_discount_rewards(memory, memory.rewards).detach()
#        print("cur_s:", cur_s)
        
        T = cur_s.size()[0]
        S = cur_s.size()[1]
        
        # update attacker's own value function
        advantages = self.update_value(old_s, old_discounted_rewards)
                
        # copy weights from the learner
        old_logprobs = self.cp_net(old_s, old_actions)
        
        # a normal update without poisoning
        self.update_policy(old_logprobs, memory.rewards, memory)
        
#        self.print_paras(self.im_policy)
        true_action_dists = self.get_dist_general(self.im_policy) #self.im_policy.get_dist(old_s, self.device)
        
        new_logprobs = self.im_policy.act_prob(old_s, old_actions, self.device)
        ratios = torch.exp(new_logprobs.detach() - old_logprobs.detach())
        true_obj = (ratios * advantages).mean()
        
        grads = np.zeros((T, S))
        
        for t in range(T):
            for s in range(S):
                cur_s[t][s] += 0.1
                # a imitating update with one poisoned reward
                old_logprobs = self.cp_net(cur_s, old_actions)
                self.update_policy(old_logprobs, memory.rewards, memory)
    #            self.print_paras(self.im_policy)
                new_logprobs = self.im_policy.act_prob(old_s, old_actions, self.device)
                ratios = torch.exp(new_logprobs.detach() - old_logprobs.detach())
                poison_obj = (ratios * advantages).mean()
                grads[t][s] = poison_obj - true_obj
                cur_s = copy.deepcopy(old_s)
        
#        print("grad of s:", grads)
#        print("norm:", np.linalg.norm(grads))
        
        if np.linalg.norm(grads) > 0:
            cur_s = old_s - torch.Tensor(radius * math.sqrt(T) * grads / np.linalg.norm(grads)).to(self.device)
#            cur_s = self.clip_obs(cur_s)
#        for t in range(T):
#            if np.linalg.norm(grads[t]) > 0:
#                cur_s[t] = old_s[t] - torch.Tensor(radius * grads[t] / np.linalg.norm(grads[t])).to(self.device)
#                cur_s[t] = self.clip_obs(cur_s[t])
#        print("new s", cur_s)
        
        # update use the new rewards
        old_logprobs = self.cp_net(cur_s, old_actions)
        self.update_policy(old_logprobs, memory.rewards, memory)
        
        poison_action_dists = self.get_dist_general(self.im_policy) #self.im_policy.get_dist(old_s, self.device)
        
        dist_distance = torch.distributions.kl.kl_divergence(true_action_dists, poison_action_dists).mean()
        print("distribution distance:", dist_distance)
        
        if hybrid: 
            return [s.clone().detach() for s in cur_s], dist_distance
        else:
            self.dist_list = np.append(self.dist_list, np.array([dist_distance]))
            
            frac = min((self.maxat - self.attack_num) / (self.maxeps - self.eps_num),1)
            self.eps_num += 1
            
            if not self.rand_select:
                if dist_distance >= np.quantile(self.dist_list, 1-frac):
                    print("attack with frac", frac)
                    self.attack_num += 1
                    return [s.clone().detach() for s in cur_s]
                else:
                    print("not attack with frac", frac)
                    return memory.states
            else:
                if random.random() < frac:
                    print("random attack with frac", frac)
                    self.attack_num += 1
                    return [s.clone().detach() for s in cur_s]
                else:
                    print("not random attack with frac", frac)
                    return memory.states
    
    def proj_tensor(self, old_tensor, new_tensor, radius):
        norm = torch.norm(new_tensor - old_tensor)
#        print("dist:", norm)
        proj = (old_tensor + (new_tensor - old_tensor) * radius / norm)
        return proj
        
    def attack_r_general(self, memory, hybrid=False, radius=None):
        
        if self.attack_num >= self.maxat:
            print("exceeds budget")
            return memory.rewards
        
        self.store_states(memory.states)
        
        if radius is None:
            radius = self.radius
            
        cur_r = memory.rewards.copy()
        T = len(cur_r)
        cur_r = np.array(cur_r)
        old_r = np.copy(cur_r)
        
        # convert list to tensor
        old_states = torch.stack(memory.states).to(self.device).detach()
        old_actions = torch.stack(memory.actions).to(self.device).detach()
        old_discounted_rewards = self.get_discount_rewards(memory, memory.rewards).detach()
        
        # update attacker's own value function
        advantages = self.update_value(old_states, old_discounted_rewards)
                
        # copy weights from the learner
        old_logprobs = self.cp_net(old_states, old_actions)
        
        # a normal update without poisoning
        self.update_policy(old_logprobs, memory.rewards, memory)
        
#        self.print_paras(self.im_policy)
        true_action_dists = self.get_dist_general(self.im_policy) #self.im_policy.get_dist(old_states, self.device)
#        true_action_dists = self.im_policy.get_dist(old_states, self.device)
        
        new_logprobs = self.im_policy.act_prob(old_states, old_actions, self.device)
        ratios = torch.exp(new_logprobs.detach() - old_logprobs.detach())
        true_obj = (ratios * advantages).mean()
        
        grads = np.zeros(T)
        
        for t in range(T):
            # change r[t] a little
            cur_r[t] += self.delta
            # a imitating update with one poisoned reward
            old_logprobs = self.cp_net(old_states, old_actions)
            self.update_policy(old_logprobs, cur_r, memory)
#            self.print_paras(self.im_policy)
            new_logprobs = self.im_policy.act_prob(old_states, old_actions, self.device)
            ratios = torch.exp(new_logprobs.detach() - old_logprobs.detach())
            poison_obj = (ratios * advantages).mean()
            grads[t] = poison_obj - true_obj
            cur_r = old_r.copy()
        
#        print("grad of r:", grads)
#        cur_r = old_r - self.radius * np.sign(grads)
        if np.linalg.norm(grads) > 0:
            cur_r = old_r - radius * math.sqrt(T) * grads / np.linalg.norm(grads)
#        print("cur_r", cur_r)
        
        # update use the new rewards
        old_logprobs = self.cp_net(old_states, old_actions)
        self.update_policy(old_logprobs, cur_r, memory)
        
        poison_action_dists = self.get_dist_general(self.im_policy) # self.im_policy.get_dist(old_states, self.device)
#        poison_action_dists = self.im_policy.get_dist(old_states, self.device)
        
        dist_distance = torch.distributions.kl.kl_divergence(true_action_dists, poison_action_dists).mean()
        print("distribution distance:", dist_distance)
        
        if hybrid:
            return cur_r.tolist(), dist_distance
        else:
            self.dist_list = np.append(self.dist_list, np.array([dist_distance]))
            
            frac = min((self.maxat - self.attack_num) / (self.maxeps - self.eps_num),1)
            self.eps_num += 1
            
            if not self.rand_select:
                if dist_distance >= np.quantile(self.dist_list, 1-frac):
#                if dist_distance <= np.quantile(self.dist_list, frac):
                    print("attack with frac", frac)
                    self.attack_num += 1
                    return cur_r.tolist()
                else:
                    print("not attack with frac", frac)
                    return old_r.tolist()
            else:
                if random.random() < frac:
                    print("random attack with frac", frac)
                    self.attack_num += 1
                    return cur_r.tolist()
                else:
                    print("not random attack with frac", frac)
                    return old_r.tolist()
    
    def compute_radius(self, memory):
        '''compute the upper bound of stability radius'''
        
        cur_r = memory.rewards.copy()
        T = len(cur_r)
        cur_r = np.array(cur_r)
        old_r = np.copy(cur_r)
        
        # convert list to tensor
        old_states = torch.stack(memory.states).to(self.device).detach()
        old_actions = torch.stack(memory.actions).to(self.device).detach()
        old_discounted_rewards = self.get_discount_rewards(memory, memory.rewards).detach()
                
        # copy weights from the learner
        old_logprobs = self.cp_net(old_states, old_actions)
        
        # a normal update without poisoning
        self.update_policy(old_logprobs, memory.rewards, memory)
#        self.print_paras(self.im_policy)
        true_action_dists = self.im_policy.get_dist(old_states, self.device)
        
        new_logprobs = self.im_policy.act_prob(old_states, old_actions, self.device)
        ratios = torch.exp(new_logprobs.detach() - old_logprobs.detach())
        true_obj = (ratios * old_discounted_rewards).mean()
        
        
        it = 0
        dist_distance = 0
        last_r = cur_r.copy()
        while dist_distance < self.dist_thres:
            it += 1
            
            grads = np.zeros(T)

            for t in range(T):
                # copy weights from the learner
                old_logprobs = self.cp_net(old_states, old_actions)
                # change r[t] a little
                cur_r[t] += self.delta
                # a imitating update with one poisoned reward
                self.update_policy(old_logprobs, cur_r, memory)
    #            self.print_paras(self.im_policy)
                new_logprobs = self.im_policy.act_prob(old_states, old_actions, self.device)
                ratios = torch.exp(new_logprobs.detach() - old_logprobs.detach())
                poison_obj = (ratios * old_discounted_rewards).mean()
                grads[t] = poison_obj - true_obj
                cur_r = last_r.copy()

    #        print("grad of r:", grads)
            if np.linalg.norm(grads) > 0:
                cur_r = last_r - self.stepsize * grads / np.linalg.norm(grads)

            # update with the new rewards
            old_logprobs = self.cp_net(old_states, old_actions)
            self.update_policy(old_logprobs, cur_r, memory)
            
            last_r = cur_r.copy()
            poison_action_dists = self.im_policy.get_dist(old_states, self.device)

            dist_distance = torch.distributions.kl.kl_divergence(true_action_dists, poison_action_dists).max()
            print("distribution distance:", dist_distance)
            if it > self.max_iter:
                return np.inf
        
        return np.linalg.norm(cur_r - old_r)
    
    def update_value(self, states, discounted_rewards):
        state_value = self.critic(states)
        MseLoss = nn.MSELoss()
        loss = MseLoss(state_value, discounted_rewards)
        self.critic_optim.zero_grad()
        loss.mean().backward()
        self.critic_optim.step()
        
        new_state_value = self.critic(states)
        return discounted_rewards - new_state_value
    
    def compute_disc(self, memory, aim):
        self.store_states(memory.states)

        if aim == "reward":
            cur_r = memory.rewards.copy()
            T = len(cur_r)
            cur_r = np.array(cur_r)
            old_r = np.copy(cur_r)
            
            # convert list to tensor
            old_states = torch.stack(memory.states).to(self.device).detach()
            old_actions = torch.stack(memory.actions).to(self.device).detach()
            old_discounted_rewards = self.get_discount_rewards(memory, memory.rewards).detach()
            
            # update attacker's own value function
            advantages = self.update_value(old_states, old_discounted_rewards)
                    
            # copy weights from the learner
            old_logprobs = self.cp_net(old_states, old_actions)
            
            # a normal update without poisoning
            self.update_policy(old_logprobs, memory.rewards, memory)
            
    #        self.print_paras(self.im_policy)
            true_action_dists = self.get_dist_general(self.im_policy) #self.im_policy.get_dist(old_states, self.device)
            
            new_logprobs = self.im_policy.act_prob(old_states, old_actions, self.device)
            ratios = torch.exp(new_logprobs.detach() - old_logprobs.detach())
            true_obj = (ratios * advantages).mean()
            
            grads = np.zeros(T)
            
            for t in range(T):
                # change r[t] a little
                cur_r[t] += self.delta
                # a imitating update with one poisoned reward
                old_logprobs = self.cp_net(old_states, old_actions)
                self.update_policy(old_logprobs, cur_r, memory)
    #            self.print_paras(self.im_policy)
                new_logprobs = self.im_policy.act_prob(old_states, old_actions, self.device)
                ratios = torch.exp(new_logprobs.detach() - old_logprobs.detach())
                poison_obj = (ratios * advantages).mean()
                grads[t] = poison_obj - true_obj
                cur_r = old_r.copy()
            
    #        print("grad of r:", grads)
            if np.linalg.norm(grads) > 0:
                cur_r = old_r - self.radius * math.sqrt(T) * grads / np.linalg.norm(grads)
    #        print("cur_r", cur_r)
            
            # update use the new rewards
            old_logprobs = self.cp_net(old_states, old_actions)
            self.update_policy(old_logprobs, cur_r, memory)
            
            poison_action_dists = self.get_dist_general(self.im_policy) #self.im_policy.get_dist(old_states, self.device)
            
            dist_distance = torch.distributions.kl.kl_divergence(true_action_dists, poison_action_dists).mean()
            print("distribution distance:", dist_distance)
            return dist_distance.item()
        
        elif aim == "action":
            cur_a = torch.stack(memory.actions).to(self.device).detach()
            old_a = torch.stack(memory.actions).to(self.device).detach()
            old_states = torch.stack(memory.states).to(self.device).detach()
            old_discounted_rewards = self.get_discount_rewards(memory, memory.rewards).detach()
    #        print("cur_a:", cur_a)
            
            if self.disc_action:
                T = cur_a.size()[0]
                
                advantages = self.update_value(old_states, old_discounted_rewards)
                        
                # copy weights from the learner
                old_logprobs = self.cp_net(old_states, old_a)
                
                # a normal update without poisoning
                self.update_policy(old_logprobs, memory.rewards, memory)
                
        #        self.print_paras(self.im_policy)
                true_action_dists = self.get_dist_general(self.im_policy) #self.im_policy.get_dist(old_states, self.device)
                
                new_logprobs = self.im_policy.act_prob(old_states, old_a, self.device)
                ratios = torch.exp(new_logprobs.detach() - old_logprobs.detach())
                true_obj = (ratios * advantages).mean()
                
                grads = np.zeros(T)
                
                for t in range(T):
                    cur_a[t] = (cur_a[t] + 1) % self.action_dim
                    # a imitating update with one poisoned reward
                    old_logprobs = self.cp_net(old_states, cur_a)
                    self.update_policy(old_logprobs, memory.rewards, memory)
        #            self.print_paras(self.im_policy)
                    new_logprobs = self.im_policy.act_prob(old_states, cur_a, self.device)
                    ratios = torch.exp(new_logprobs.detach() - old_logprobs.detach())
                    poison_obj = (ratios * advantages).mean()
                    grads[t] = poison_obj - true_obj
                    cur_a = copy.deepcopy(old_a)
    #            print("grads", grads)
                thres = np.quantile(np.array(grads), self.radius)
    #            print("thres", thres)
                for t in range(T):
                    if grads[t] < 0 and grads[t] < thres:
                        cur_a[t] = (cur_a[t] + 1) % self.action_dim
    #            print("new a", cur_a)
                
            else:
                T = cur_a.size()[0]
                A = cur_a.size()[1]
                
                # update attacker's own value function
                advantages = self.update_value(old_states, old_discounted_rewards)
                        
                # copy weights from the learner
                old_logprobs = self.cp_net(old_states, old_a)
                
                # a normal update without poisoning
                self.update_policy(old_logprobs, memory.rewards, memory)
                
        #        self.print_paras(self.im_policy)
                true_action_dists = self.get_dist_general(self.im_policy) #self.im_policy.get_dist(old_states, self.device)
                
                new_logprobs = self.im_policy.act_prob(old_states, old_a, self.device)
                ratios = torch.exp(new_logprobs.detach() - old_logprobs.detach())
                true_obj = (ratios * advantages).mean()
                
                grads = np.zeros((T, A))
                
                for t in range(T):
                    for s in range(A):
                        cur_a[t][s] += 0.1
                        # a imitating update with one poisoned reward
                        old_logprobs = self.cp_net(old_states, cur_a)
                        self.update_policy(old_logprobs, memory.rewards, memory)
            #            self.print_paras(self.im_policy)
                        new_logprobs = self.im_policy.act_prob(old_states, cur_a, self.device)
                        ratios = torch.exp(new_logprobs.detach() - old_logprobs.detach())
                        poison_obj = (ratios * advantages).mean()
                        grads[t][s] = poison_obj - true_obj
                        cur_a = copy.deepcopy(cur_a)
            
    #            for t in range(T):
    #                if np.linalg.norm(grads[t]) > 0:
    #                    cur_a[t] = old_a[t] - torch.Tensor(radius * grads[t] / np.linalg.norm(grads[t]))
                if np.linalg.norm(grads) > 0:
                    cur_a = old_a - torch.Tensor(self.radius * math.sqrt(T) * grads / np.linalg.norm(grads)).to(self.device)
    #            print("new a", cur_a)
            
            # update use the new rewards
            old_logprobs = self.cp_net(old_states, cur_a)
            self.update_policy(old_logprobs, memory.rewards, memory)
            
            poison_action_dists = self.get_dist_general(self.im_policy) #self.im_policy.get_dist(old_states, self.device)
            
            dist_distance = torch.distributions.kl.kl_divergence(true_action_dists, poison_action_dists).mean()
            print("distribution distance:", dist_distance)
            return dist_distance.item()
        elif aim == "obs":
            cur_s = torch.stack(memory.states).to(self.device).detach()
            old_s = torch.stack(memory.states).to(self.device).detach()
            old_actions = torch.stack(memory.actions).to(self.device).detach()
            old_discounted_rewards = self.get_discount_rewards(memory, memory.rewards).detach()
    #        print("cur_s:", cur_s)
            
            T = cur_s.size()[0]
            S = cur_s.size()[1]
            
            # update attacker's own value function
            advantages = self.update_value(old_s, old_discounted_rewards)
                    
            # copy weights from the learner
            old_logprobs = self.cp_net(old_s, old_actions)
            
            # a normal update without poisoning
            self.update_policy(old_logprobs, memory.rewards, memory)
            
    #        self.print_paras(self.im_policy)
            true_action_dists = self.get_dist_general(self.im_policy) #self.im_policy.get_dist(old_s, self.device)
            
            new_logprobs = self.im_policy.act_prob(old_s, old_actions, self.device)
            ratios = torch.exp(new_logprobs.detach() - old_logprobs.detach())
            true_obj = (ratios * advantages).mean()
            
            grads = np.zeros((T, S))
            
            for t in range(T):
                for s in range(S):
                    cur_s[t][s] += 0.1
                    # a imitating update with one poisoned reward
                    old_logprobs = self.cp_net(cur_s, old_actions)
                    self.update_policy(old_logprobs, memory.rewards, memory)
        #            self.print_paras(self.im_policy)
                    new_logprobs = self.im_policy.act_prob(old_s, old_actions, self.device)
                    ratios = torch.exp(new_logprobs.detach() - old_logprobs.detach())
                    poison_obj = (ratios * advantages).mean()
                    grads[t][s] = poison_obj - true_obj
                    cur_s = copy.deepcopy(old_s)
            
    #        print("grad of s:", grads)
    #        print("norm:", np.linalg.norm(grads))
            
            if np.linalg.norm(grads) > 0:
                cur_s = old_s - torch.Tensor(self.radius * math.sqrt(T) * grads / np.linalg.norm(grads)).to(self.device)
    #            cur_s = self.clip_obs(cur_s)
    #        for t in range(T):
    #            if np.linalg.norm(grads[t]) > 0:
    #                cur_s[t] = old_s[t] - torch.Tensor(radius * grads[t] / np.linalg.norm(grads[t])).to(self.device)
    #                cur_s[t] = self.clip_obs(cur_s[t])
    #        print("new s", cur_s)
            
            # update use the new rewards
            old_logprobs = self.cp_net(cur_s, old_actions)
            self.update_policy(old_logprobs, memory.rewards, memory)
            
            poison_action_dists = self.get_dist_general(self.im_policy) #self.im_policy.get_dist(old_s, self.device)
            
            dist_distance = torch.distributions.kl.kl_divergence(true_action_dists, poison_action_dists).mean()
            print("distribution distance:", dist_distance)
            return dist_distance.item()
    
    def compute_radius1(self, memory):
        '''compute the upper bound of stability radius'''
        
        cur_r = memory.rewards.copy()
        T = len(cur_r)
        cur_r = np.array(cur_r)
        old_r = np.copy(cur_r)
        
        # convert list to tensor
        old_states = torch.stack(memory.states).to(self.device).detach()
        old_actions = torch.stack(memory.actions).to(self.device).detach()
        old_discounted_rewards = self.get_discount_rewards(memory, memory.rewards).detach()
                
        # copy weights from the learner
        old_logprobs = self.cp_net(old_states, old_actions)
        
        # a normal update without poisoning
        self.update_policy(old_logprobs, memory.rewards, memory)
#        self.print_paras(self.im_policy)
        true_action_dists = self.im_policy.get_dist(old_states, self.device)
        
        new_logprobs = self.im_policy.act_prob(old_states, old_actions, self.device)
        ratios = torch.exp(new_logprobs.detach() - old_logprobs.detach())
        true_obj = (ratios * old_discounted_rewards).mean()
        
        grads = np.zeros(T)
        
        for t in range(T):
            # copy weights from the learner
            old_logprobs = self.cp_net(old_states, old_actions)
            # change r[t] a little
            cur_r[t] += self.delta
            # a imitating update with one poisoned reward
            self.update_policy(old_logprobs, cur_r, memory)
#            self.print_paras(self.im_policy)
            new_logprobs = self.im_policy.act_prob(old_states, old_actions, self.device)
            ratios = torch.exp(new_logprobs.detach() - old_logprobs.detach())
            poison_obj = (ratios * old_discounted_rewards).mean()
            grads[t] = poison_obj - true_obj
            cur_r = old_r.copy()
        
#        print("grad of r:", grads)
        if np.linalg.norm(grads) > 0:
            cur_r = old_r - self.radius * grads / np.linalg.norm(grads)
        # steepest direction
        
        # update with the new rewards
        old_logprobs = self.cp_net(old_states, old_actions)
        self.update_policy(old_logprobs, cur_r, memory)
        
        poison_action_dists = self.im_policy.get_dist(old_states, self.device)
        
        dist_distance = torch.distributions.kl.kl_divergence(true_action_dists, poison_action_dists).max()
        
#         dist_distance = torch.norm(true_action_dists - poison_action_dists, p=1, dim=1).max().item()
        
        print("distribution distance:", dist_distance)
        
        it = 0
        while dist_distance < self.dist_thres:
            it += 1
            
            cur_r -= self.stepsize * grads / np.linalg.norm(grads)
            # recompute
            old_logprobs = self.cp_net(old_states, old_actions)
            self.update_policy(old_logprobs, cur_r, memory)
            
            poison_action_dists = self.im_policy.get_dist(old_states, self.device)
            
            dist_distance = torch.distributions.kl.kl_divergence(true_action_dists, poison_action_dists).max()
            print(np.linalg.norm(cur_r - old_r), dist_distance)
            if it > self.max_iter:
                return np.inf
            
        return np.linalg.norm(cur_r - old_r)
    
    def cp_net(self, states, actions):
        self.im_policy.load_state_dict(self.learner.policy.state_dict())
        self.im_optimizer.load_state_dict(self.learner.optimizer.state_dict())
        logprobs = self.im_policy.act_prob(states, actions, self.device)
        
        return logprobs
    
    
    def get_discount_rewards(self, memory, rewards):
        T = len(rewards)
        discounted_rewards = torch.empty(T, requires_grad=False).to(self.device)

        Gt = 0
        t = T-1
        for reward, is_terminal in zip(reversed(rewards), reversed(memory.is_terminals)):
            if is_terminal:
                Gt = 0
            Gt = reward + (self.gamma * Gt)
            discounted_rewards[t] = Gt
            t -= 1
#        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)
        return discounted_rewards
    
    
    def attack_r_vpg(self, memory):
        '''mathematically compute the gradient'''
        cur_r = memory.rewards.copy()
        
        cur_r = np.array(cur_r)
        old_r = np.copy(cur_r)
        # convert list to tensor
        old_states = torch.stack(memory.states).to(self.device).detach()
        old_actions = torch.stack(memory.actions).to(self.device).detach()
        old_logprobs = self.im_policy.act_prob(old_states, old_actions, self.device)
        
        old_discounted_rewards = self.get_discount_rewards(memory, memory.rewards).detach()
        
        # copy weights from the learner
        self.im_policy.load_state_dict(self.learner.policy.state_dict())
        self.im_optimizer.load_state_dict(self.learner.optimizer.state_dict())
        old_action_dists = self.im_policy.action_layer(old_states).detach()
        
        policy_r_grad = self.r_gradient(old_logprobs, memory.is_terminals, self.im_policy)
        
        for it in range(self.max_iter):
            # use the poisoned reward to generate new policys
            
            # copy weights from the learner
            self.im_policy.load_state_dict(self.learner.policy.state_dict())
            self.im_optimizer.load_state_dict(self.learner.optimizer.state_dict())
        
            self.update_policy(old_logprobs, cur_r, memory)
            new_logprobs = self.im_policy.act_prob(old_states, old_actions, self.device)
            ratios = torch.exp(new_logprobs - old_logprobs.detach())
            attack_obj = ratios * old_discounted_rewards
            
            self.im_policy.zero_grad()
            attack_obj.mean().backward()
            policy_grad = torch.cat([param.grad.view(-1) for param in self.im_policy.parameters()])
            
            final_r_grad = torch.mv(policy_r_grad, policy_grad)
            
            cur_r -= self.stepsize * final_r_grad.cpu().numpy()
            cur_r = self.proj(old_r, cur_r, self.radius)
            norm = np.linalg.norm(cur_r - old_r)
            print("dist of r:", norm)
            if self.radius - norm < 1e-6:
                break
        
        new_action_dists = self.im_policy.action_layer(old_states).detach()
        
        
        print(old_action_dists.size(), new_action_dists.size())
        dist_distance = torch.norm(old_action_dists - new_action_dists, p = 1)
        print("distribution distance:", dist_distance)
        
        self.dist_list = np.append(self.dist_list, np.array([dist_distance]))
        
        if dist_distance > np.quantile(self.dist_list, 1-self.frac):
            print("attack")
            self.attack_num += 1
            return cur_r.tolist()
        else:
            print("not attack")
            return old_r.tolist()
            
    
    def r_gradient(self, logprobs, is_terminals, policy):
        '''compute gradient of r
            probs: the probabilities of the learner policy choosing the original actions 
        '''
        grad = []
        
        partial_sum = 0
        for i, is_terminal in enumerate(is_terminals):
            if is_terminal: 
                partial_sum = 0
            
            policy.zero_grad()
            logprobs[i].backward(retain_graph=True)
            grad_paras = torch.cat([param.grad.view(-1) for param in policy.parameters()])
            grad.append(partial_sum * self.gamma + grad_paras)
        
        grad_matrix = torch.stack(grad).to(self.device).detach()
        
        return grad_matrix
    
    def update_policy(self, log_probs, cur_r, memory):
        '''Imitate the poicy update of the learner'''
        if self.alg == "vpg":
            vpg_update(self.im_optimizer, log_probs, cur_r, memory.is_terminals, self.gamma)
        elif self.alg == "ppo":
            ppo_update(self.im_policy, self.im_optimizer, log_probs, cur_r, memory, 
                       self.gamma, self.K_epochs, self.eps_clip, self.loss_fn, self.device)
        
        
    def print_paras(self, model):
        for param in model.parameters():
            print(param.data)
            
    def proj(self, old_r_array, new_r_array, radius):
        norm = np.linalg.norm(new_r_array-old_r_array)
        print("dist of r:", norm)
        if norm > radius:
            proj_r = (old_r_array + (new_r_array - old_r_array) * radius / norm)
            return proj_r
        else:
            return new_r_array