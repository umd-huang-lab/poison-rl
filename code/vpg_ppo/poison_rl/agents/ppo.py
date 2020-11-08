import sys
import torch  
import gym
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from gym.spaces import Box, Discrete
from .model import ActorCritic, ContActorCritic
from .updates import ppo_update
from torch.distributions import Categorical

class PPO(nn.Module):
    def __init__(self, state_space, action_space, K_epochs=4, eps_clip=0.2, hidden_sizes=(64,64), 
                 activation=nn.Tanh, learning_rate=3e-4, gamma=0.9, device="cpu", action_std=0.5):
        super(PPO, self).__init__()
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device
        
        # deal with 1d state input
        state_dim = state_space.shape[0]
        
        if isinstance(action_space, Discrete):
            self.action_dim = action_space.n
            self.policy = ActorCritic(state_dim, self.action_dim, hidden_sizes, activation).to(self.device)
            
        elif isinstance(action_space, Box):
            self.action_dim = action_space.shape[0]
            self.policy = ContActorCritic(state_dim, self.action_dim, hidden_sizes, activation, action_std, self.device).to(self.device)
            
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
    
    def act(self, state):
        return self.policy.act(state, self.device)
        
    
    def update_policy(self, memory):
        
        old_logprobs = torch.stack(memory.logprobs).to(self.device).detach()
        
        ppo_update(self.policy, self.optimizer, old_logprobs, memory.rewards, 
                   memory, self.gamma, self.K_epochs, self.eps_clip, self.loss_fn, self.device)
        
    
    def get_state_dict(self):
        return self.policy.state_dict(), self.optimizer.state_dict()