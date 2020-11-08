import sys
import torch  
import gym
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import random
from PPO import Memory, ActorCritic

class PPOWbAttacker(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip,
                 stepsize=0.1, max_iter=2, radius=0.5, device="cpu"):
        super(PPOWbAttacker, self).__init__()

        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        self.stepsize = stepsize
        self.max_iter = max_iter
        self.radius = radius
        self.device = device


    
    def attack_ep_r(self, learner, memory):
        '''One-episode attack: change the rewards of one episode into poisoned rewards'''
        
        cur_r = memory.rewards.copy()
        T = len(cur_r)
        cur_r = np.array(cur_r, dtype=np.float64)
        old_r = np.copy(cur_r)
        
        for it in range(self.max_iter):
            # copy weights from the learner
            self.policy.load_state_dict(learner.policy_old.state_dict())
            
            # old statistics
            old_states = torch.stack(memory.states).to(self.device).detach()
            old_actions = torch.stack(memory.actions).to(self.device).detach()
            old_logprobs = torch.stack(memory.logprobs).to(self.device).detach()
            _, state_values, _ = self.policy.evaluate(old_states, old_actions)
            
            # get policy improved with the old policy and the poisoned rewards
            self.update_policy(memory, cur_r)
            
            # make rewards better poisoned so that the policy performs badly
            r_trainable = Variable(torch.FloatTensor(cur_r).to(self.device), requires_grad=True)
#            print(r_trainable)
            discounted_rewards = torch.empty(T, requires_grad=False).to(self.device)

            Gt = 0
            t = T-1
            for reward, is_terminal in zip(reversed(r_trainable), reversed(memory.is_terminals)):
                if is_terminal:
                    Gt = 0
                Gt = reward + (self.gamma * Gt)
                discounted_rewards[t] = Gt
                t -= 1
                
            # get policy gradient of the new policy wrt the new rewards
            policy_gradient = []
            new_logprobs, _, _ = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(new_logprobs.detach() - old_logprobs.detach()).detach()
                
            advantages = discounted_rewards - state_values.detach()

            policy_gradient = ratios * advantages
            loss = policy_gradient.mean()
            print("loss:", loss.data.item())
            
            loss.backward()
#            print("r grad:", r_trainable.grad)
            
            cur_r -= self.stepsize * r_trainable.grad.numpy()
            r_trainable.grad.data.zero_()
            cur_r = self.proj(old_r, cur_r, self.radius)
#             print("new_r:", cur_r)
            
        memory.rewards = cur_r.tolist()

    
#    def update_policy(self, log_probs, rewards):
#        '''Imitate the poicy update of the learner'''
#        discounted_rewards = []
#        for t in range(len(rewards)):
#            Gt = 0 
#            pw = 0
#            for r in rewards[t:]:
#                Gt = Gt + self.gamma**pw * r
#                pw = pw + 1
#            discounted_rewards.append(Gt)
#            
#        discounted_rewards = torch.tensor(discounted_rewards)
#        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9) # normalize discounted rewards
#    
#        # get policy gradient of the new policy wrt the new rewards
#            
#        policy_gradient = []
#        for log_prob, Gt in zip(log_probs, discounted_rewards):
#            policy_gradient.append(-log_prob * Gt)
#        
#        self.optimizer.zero_grad()
#        policy_gradient = torch.stack(policy_gradient).sum()
#        policy_gradient.backward(retain_graph=True)
#        self.optimizer.step()
        
    def update_policy(self, memory, cur_r):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(cur_r), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.stack(memory.states).to(self.device).detach()
        old_actions = torch.stack(memory.actions).to(self.device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(self.device).detach()
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())
                
            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        
        
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
        
        