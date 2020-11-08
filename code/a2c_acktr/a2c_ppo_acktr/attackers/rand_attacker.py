import sys
import torch  
import numpy as np  
import random
import math
from gym.spaces import Box, Discrete

class RandAttacker:
    def __init__(self, envs, radius=0.5, frac=1.0, maxat=1000, device="cpu"):
        super(RandAttacker, self).__init__()

        self.radius = radius
        self.frac = frac
        self.maxat = maxat
        self.attack_num = 0
        self.device = device
        self.disc_action = isinstance(envs.action_space, Discrete)
        if self.disc_action:
            self.action_dim = envs.action_space.n
        else:
            self.action_dim = envs.action_space.shape[0]
        
    def attack_r_general(self, memory, next_value):
        '''Attack with the current memory'''
        if self.attack_num >= self.maxat:
            print("exceeds budget")
            return memory.rewards
        
        randr = torch.rand(memory.rewards.size()).to(self.device)
        num_steps, num_processes, _ = memory.rewards.size()
        
        attack_r = self.proj(memory.rewards, randr, self.radius * math.sqrt(num_steps * num_processes))
        
        if random.random() < self.frac:
            self.attack_num += 1
            return attack_r
        else:
            return memory.rewards
        
    def set_obs_range(self, low, high):
        self.obs_low = torch.tensor(low).float().to(self.device)
        self.obs_high = torch.tensor(high).float().to(self.device)
        print("low:", self.obs_low)
        print("high:", self.obs_high)
    
    def clip_obs(self, obs):
        return torch.max(torch.min(obs, self.obs_high), self.obs_low)
    
    def attack_s_general(self, memory, next_value):
        '''Attack with the current memory'''
        if self.attack_num >= self.maxat:
            print("exceeds budget")
            return memory.obs
        
        num_steps, num_processes, _ = memory.rewards.size() 
        new_obs = memory.obs.clone().detach()
        
        for step in range(num_steps):
            for proc in range(num_processes):
                rands = torch.randn(memory.obs[step][proc].size()).to(self.device)
                attack = self.proj_tensor(memory.obs[step][proc], rands, self.radius)
                new_obs[step][proc] = attack #self.clip_obs(attack)
        
        if random.random() < self.frac:
            self.attack_num += 1
            return new_obs
        else:
            return memory.obs
    
    def attack_a_general(self, memory, next_value):
        '''Attack with the current memory'''
        if self.attack_num >= self.maxat:
            print("exceeds budget")
            return memory.actions
        
        num_steps, num_processes, _ = memory.rewards.size()
        
        
        if self.disc_action:
            new_action = torch.zeros(memory.actions.size()).to(self.device)
            for step in range(num_steps):
                for proc in range(num_processes):
                    if random.random() < self.radius:
                        attack = (memory.actions[step][proc][0] + 1) % self.action_dim
                        new_action[step][proc][0] = attack
                    else:
                        new_action[step][proc] = memory.actions[step][proc]
        else:
#            print("old", memory.actions)
            new_action = torch.zeros(memory.actions.size()).to(self.device)
            for step in range(num_steps):
                for proc in range(num_processes):
                    randa = torch.rand(memory.actions[step][proc].size()).to(self.device)
                    attack = self.proj_tensor(memory.actions[step][proc], randa, self.radius)
                    new_action[step][proc] = attack
#            print("new", new_action)
        
        if random.random() < self.frac:
            self.attack_num += 1
            return new_action
        else:
            return memory.actions
    
    
            
    def proj(self, old_r, new_r, radius):
        
        norm = torch.norm(new_r-old_r)
        
        proj_r = (old_r + (new_r - old_r) * radius / norm)
        return proj_r
    
    def proj_tensor(self, old_tensor, new_tensor, radius):
        norm = torch.norm(new_tensor - old_tensor)
#        print("dist:", norm)
        proj = (old_tensor + (new_tensor - old_tensor) * radius / norm)
        return proj