import torch
import torch.nn as nn
import math
import torch.nn.functional as F

def vpg_update(optimizer, logprobs, rewards, is_terminals, gamma):
    discounted_reward = []
    Gt = 0
    for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
        if is_terminal:
            Gt = 0
        Gt = reward + (gamma * Gt)
        discounted_reward.insert(0, Gt)
    
#    discounted_reward = torch.tensor(discounted_reward)
#    discounted_reward = (discounted_reward - discounted_reward.mean()) / (discounted_reward.std() + 1e-5)
    # Normalizing the rewards:
    #        rewards = torch.tensor(rewards).to(self.device)
    #        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
    
    policy_gradient = []
    for log_prob, Gt in zip(logprobs, discounted_reward):
        policy_gradient.append(-log_prob * Gt)
    
    optimizer.zero_grad()
    policy_gradient = torch.stack(policy_gradient).sum()
    policy_gradient.backward()
    optimizer.step()


def ppo_update(policy, optimizer, logprobs, rewards, memory, 
               gamma, K_epochs, eps_clip, loss_fn, device):
    
    old_states = torch.stack(memory.states).to(device).detach()
    old_actions = torch.stack(memory.actions).to(device).detach()
    
    discounted_reward = []
    Gt = 0
    for reward, is_terminal in zip(reversed(rewards), reversed(memory.is_terminals)):
        if is_terminal:
            Gt = 0
        Gt = reward + (gamma * Gt)
        discounted_reward.insert(0, Gt)
        
    discounted_reward = torch.tensor(discounted_reward).to(device)
    
    # Optimize policy for K epochs:
    for _ in range(K_epochs):
        # Evaluating old actions and values :
        new_logprobs, state_values, dist_entropy = policy.evaluate(old_states, old_actions)
        
        # Finding the ratio (pi_theta / pi_theta__old):
        ratios = torch.exp(new_logprobs - logprobs.detach())
            
        # Finding Surrogate Loss:
        advantages = discounted_reward - state_values.detach()
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1-eps_clip, 1+eps_clip) * advantages
        
        loss = -torch.min(surr1, surr2) + 0.5*loss_fn(state_values, discounted_reward) - 0.01*dist_entropy
        
        # take gradient step
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

def sac_update(policy, policy_optim, critic, critic_optim, critic_target,  # nets and optimizers
               state_batch, action_batch, reward_batch, next_state_batch, mask_batch, # batch data
               alpha, gamma, device): # other parameters
    
    state_batch = torch.FloatTensor(state_batch).to(device)
    next_state_batch = torch.FloatTensor(next_state_batch).to(device)
    action_batch = torch.FloatTensor(action_batch).to(device)
    reward_batch = torch.FloatTensor(reward_batch).to(device).unsqueeze(1)
    mask_batch = torch.FloatTensor(mask_batch).to(device).unsqueeze(1)
#     print(state_batch, action_batch)
    with torch.no_grad():
        next_state_action, next_state_log_pi, _ = policy.sample(next_state_batch)
#         _, next_state_action, next_state_log_pi = policy.act(next_state_batch, device)
        qf1_next_target, qf2_next_target = critic_target(next_state_batch, next_state_action)
        min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
        
        next_q_value = reward_batch + mask_batch * gamma * (min_qf_next_target)
    
    qf1, qf2 = critic(state_batch, action_batch) 
    
    qf1_loss = F.mse_loss(qf1, next_q_value)  
    qf2_loss = F.mse_loss(qf2, next_q_value)  

    pi, log_pi, _ = policy.sample(state_batch)
#     _, pi, log_pi = policy.act(state_batch, device)

    qf1_pi, qf2_pi = critic(state_batch, pi)
    min_qf_pi = torch.min(qf1_pi, qf2_pi)

    policy_loss = ((alpha * log_pi) - min_qf_pi).mean() 

    critic_optim.zero_grad()
    qf1_loss.backward()
    critic_optim.step()

    critic_optim.zero_grad()
    qf2_loss.backward()
    critic_optim.step()

    policy_optim.zero_grad()
    policy_loss.backward()
    policy_optim.step()
    

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


