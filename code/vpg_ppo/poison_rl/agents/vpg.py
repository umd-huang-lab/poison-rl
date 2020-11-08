import sys
import torch  
import gym
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from gym.spaces import Box, Discrete
from .model import Actor, ContActor
from .updates import vpg_update
from torch.distributions import Categorical

class VPG(nn.Module):
    def __init__(self, state_space, action_space, hidden_sizes=(64,64), 
                 activation=nn.Tanh, learning_rate=3e-4, gamma=0.9, device="cpu", action_std=0.5):
        super(VPG, self).__init__()
        
        # deal with 1d state input
        state_dim = state_space.shape[0]
        
        self.gamma = gamma
        self.device = device
        
        if isinstance(action_space, Discrete):
            self.action_dim = action_space.n
            self.policy = Actor(state_dim, self.action_dim, hidden_sizes, activation).to(self.device)
        elif isinstance(action_space, Box):
            self.action_dim = action_space.shape[0]
            self.policy = ContActor(state_dim, self.action_dim, hidden_sizes, activation, action_std, self.device).to(self.device)
            
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

    
    def act(self, state):
        return self.policy.act(state, self.device)
        
    
    def update_policy(self, memory):
        
        old_states = torch.stack(memory.states).to(self.device).detach()
        old_actions = torch.stack(memory.actions).to(self.device).detach()
        
        logprobs = self.policy.act_prob(old_states, old_actions, self.device)
        
        vpg_update(self.optimizer, logprobs, memory.rewards, memory.is_terminals, self.gamma)
        
#        rewards = []
#        discounted_reward = 0
#        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
#            if is_terminal:
#                discounted_reward = 0
#            discounted_reward = reward + (self.gamma * discounted_reward)
#            rewards.insert(0, discounted_reward)
#        
#        # Normalizing the rewards:
##        rewards = torch.tensor(rewards).to(self.device)
##        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
#        
#        policy_gradient = []
#        for log_prob, Gt in zip(memory.logprobs, rewards):
#            policy_gradient.append(-log_prob * Gt)
#        
#        self.optimizer.zero_grad()
#        policy_gradient = torch.stack(policy_gradient).sum()
#        policy_gradient.backward()
#        self.optimizer.step()
    
    def get_state_dict(self):
        return self.policy.state_dict(), self.optimizer.state_dict()
    
    
    def set_state_dict(self, state_dict, optim):
        self.policy.load_state_dict(state_dict)
        self.optimizer.load_state_dict(optim)
        
        