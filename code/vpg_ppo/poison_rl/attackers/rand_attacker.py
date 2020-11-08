import sys
import torch  
import numpy as np  
import math
import random
from gym.spaces import Box, Discrete

class RandAttacker:
    def __init__(self, action_space, radius=0.5, frac=1.0, maxat=1000, device="cpu"):
        super(RandAttacker, self).__init__()

        self.radius = radius
        self.frac = frac
        self.maxat = maxat
        self.attack_num = 0
        self.device = device
        self.disc_action = isinstance(action_space, Discrete)
        if self.disc_action:
            self.action_dim = action_space.n
        else:
            self.action_dim = action_space.shape[0]
    
    def set_obs_range(self, low, high):
        self.obs_low = torch.tensor([low]).float().to(self.device)
        self.obs_high = torch.tensor([high]).float().to(self.device)
        print("low:", self.obs_low)
        print("high:", self.obs_high)
    
    def clip_obs(self, obs):
        return torch.max(torch.min(obs, self.obs_high), self.obs_low)
    
    def attack_r_general(self, memory):
        '''Attack with the current memory'''
        if self.attack_num >= self.maxat:
            print("exceeds budget")
            return memory.rewards
        randr = np.random.randn(len(memory.rewards))
        attack_r = self.proj(np.array(memory.rewards), randr, self.radius*math.sqrt(len(memory.rewards))).tolist()
        
        if random.random() < self.frac:
            self.attack_num += 1
            return attack_r
        else:
            return memory.rewards
        
    def attack_s_general(self, memory):
        '''Attack with the current memory'''
        if self.attack_num >= self.maxat:
            print("exceeds budget")
            return memory.states
        
        T = len(memory.states)
        l = memory.states[0].numel()
        
        new_s = torch.zeros((T,l)).to(self.device)
        for t in range(T):
#            print("old:", memory.states[t])
            rands = torch.randn(l).to(self.device)
#            print("rands:", rands)
            attack = self.proj_tensor(memory.states[t], rands, self.radius)
#            print("attack:", attack)
            new_s[t] = self.clip_obs(attack)
#            print("new:", attack_s[t])
        
        if random.random() < self.frac:
            self.attack_num += 1
            return [s.clone().detach() for s in new_s]
        else:
            return memory.states
        
    def attack_a_general(self, memory):
        '''Attack with the current memory'''
        if self.attack_num >= self.maxat:
            print("exceeds budget")
            return memory.actions
        
        T = len(memory.actions)
        if self.disc_action:
            attack_a = []
#            arr = np.arange(T)
#            np.random.shuffle(arr)
#            rand_inds = arr[:int(self.radius)]
#            for t in range(T):
#                if t in rand_inds:
#                    attack = (memory.actions[t] + 1) % self.action_dim
#                    attack_a.append(attack)
#                else:
#                    attack_a.append(memory.actions[t])
            for t in range(T):
                if random.random() < self.radius:
                    attack = (memory.actions[t] + 1) % self.action_dim
                    attack_a.append(attack)
                else:
                    attack_a.append(memory.actions[t])
        else:
            attack_a = []
            for t in range(T):
                randa = torch.rand(self.action_dim).to(self.device)
                attack = self.proj_tensor(memory.actions[t], randa, self.radius)
                attack_a.append(attack)
            
        if random.random() < self.frac:
            self.attack_num += 1
            return attack_a
        else:
            return memory.actions
    
    
            
    def proj(self, old_r_array, new_r_array, radius):
        
        norm = np.linalg.norm(new_r_array-old_r_array)
#        print("dist of r:", norm)
        proj_r = (old_r_array + (new_r_array - old_r_array) * radius / norm)
        return proj_r
    
    def proj_tensor(self, old_tensor, new_tensor, radius):
        norm = torch.norm(new_tensor - old_tensor)
#        print("dist:", norm)
        proj = (old_tensor + (new_tensor - old_tensor) * radius / norm)
        return proj
    