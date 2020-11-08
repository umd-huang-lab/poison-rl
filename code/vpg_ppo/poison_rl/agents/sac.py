import sys
import torch  
import gym
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from gym.spaces import Box, Discrete
from .model_sac import GaussianPolicy, QNetwork
from .model import ContActor
from .updates import soft_update, hard_update, sac_update
from torch.distributions import Categorical

class SAC(nn.Module):
    def __init__(self, state_space, action_space, tau=0.005, alpha=0.2, hidden_size=256, 
                 activation=nn.Tanh, learning_rate=3e-4, gamma=0.9, device="cpu", action_std=0.5,
                 update_interval=1):
        super(SAC, self).__init__()
        '''Implementation of SAC (only works for continuous action space)'''
        state_dim = state_space.shape[0]
        
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.update_interval = update_interval
        self.device = device
        
        self.critic = QNetwork(state_dim, action_space.shape[0], hidden_size).to(device=self.device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.critic_target = QNetwork(state_dim, action_space.shape[0], hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        self.policy = GaussianPolicy(state_dim, action_space.shape[0], hidden_size, action_space).to(self.device)
#         self.policy = ContActor(state_dim, action_space.shape[0], hidden_size, activation, action_std, self.device).to(self.device)
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.updates = 0
    
    def act(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
#         _, action, _ = self.policy.act(state, self.device)
#         return action.cpu().data.numpy().flatten()
        action, _, _ = self.policy.sample(state)
        
        return action.detach().cpu().numpy()[0]
        
    
    def update_policy(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch):
        
        sac_update(self.policy, self.policy_optim, self.critic, self.critic_optim, self.critic_target,  
               state_batch, action_batch, reward_batch, next_state_batch, mask_batch, 
               self.alpha, self.gamma, self.device)
        
        if self.updates % self.update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        
        