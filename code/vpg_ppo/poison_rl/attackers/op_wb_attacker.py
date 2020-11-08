import sys
import torch  
import gym
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import random
from gym.spaces import Box, Discrete
from poison_rl.agents.model import Actor, ContActor, ActorCritic, ContActorCritic
from poison_rl.agents.model_sac import GaussianPolicy, QNetwork
from poison_rl.agents.updates import vpg_update, ppo_update, sac_update, soft_update, hard_update
from poison_rl.agents.vpg import VPG
from poison_rl.agents.ppo import PPO
from poison_rl.agents.sac import SAC
from torch.distributions import Categorical, MultivariateNormal
import higher

class OpWbAttacker:
    def __init__(self, state_space, action_space, learner, maxat, maxeps, hidden_size=256, tau=0.005,
                 alpha=0.2, learning_rate=3e-4, gamma=0.9, device="cpu", update_interval=1,
                 stepsize=0.1, maxiter=1, radius=0.5, delta=10, dist_thres=0.1, rand_select=False):
        super(OpWbAttacker, self).__init__()

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
        self.update_interval = update_interval
        
        if isinstance(self.learner, SAC):
            self.init_imitator_sac(state_space, action_space, learning_rate, hidden_size, tau, alpha)
            self.alg = "sac"
            self.delta = 1
        
        self.good_critic = QNetwork(state_space.shape[0], action_space.shape[0], hidden_size).to(device=self.device)
        self.good_critic_optim = optim.Adam(self.good_critic.parameters(), lr=learning_rate)

        self.good_critic_target = QNetwork(state_space.shape[0], action_space.shape[0], hidden_size).to(self.device)
        hard_update(self.good_critic_target, self.good_critic)
        
        self.dist_list = np.array([])
        self.attack_num = 0
        self.eps_num = 0

    def init_imitator_sac(self, state_space, action_space, learning_rate, hidden_size, tau, alpha):
        '''Initialize attacker's policy and optimizer to imitate the learner's behaviors'''
        
        state_dim = state_space.shape[0]
        
        self.tau = tau
        self.alpha = alpha
        
        self.im_critic = QNetwork(state_dim, action_space.shape[0], hidden_size).to(device=self.device)
        self.im_critic_optim = optim.Adam(self.im_critic.parameters(), lr=learning_rate)

        self.im_critic_target = QNetwork(state_dim, action_space.shape[0], hidden_size).to(self.device)
        hard_update(self.im_critic_target, self.im_critic)

        self.im_policy = GaussianPolicy(state_dim, action_space.shape[0], hidden_size, action_space).to(self.device)
        self.im_policy_optim = optim.Adam(self.im_policy.parameters(), lr=learning_rate)
        
    def attack_r_fast(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch):
        '''Attack with the current memory'''
        if self.attack_num >= self.maxat:
            print("exceeds budget")
            return memory.rewards
        
        # learner a good q value function
        self.update_critic(state_batch, action_batch, reward_batch, next_state_batch, mask_batch)
        
        cur_r = reward_batch.copy()
        T = len(cur_r)
        cur_r = np.array(cur_r)
        old_r = np.copy(cur_r)
        
        # convert list to tensor
        old_states = torch.FloatTensor(state_batch).to(self.device)
        old_actions = torch.FloatTensor(action_batch).to(self.device)
        old_rewards = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        old_masks = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        old_next_states = torch.FloatTensor(next_state_batch).to(self.device)
        old_discounted_rewards = self.get_discount_rewards(old_rewards, old_masks).detach()
        
        # copy weights from the learner
        self.cp_net()
        
        # a normal update without poisoning
        sac_update(self.im_policy, self.im_policy_optim, self.im_critic, self.im_critic_optim, 
                   self.im_critic_target, state_batch, action_batch, reward_batch, next_state_batch, 
                   mask_batch, self.alpha, self.gamma, self.device)
        
#        self.print_paras(self.im_policy)
        true_action_dists = self.im_policy.get_dist(old_states)
    
        old_loss = np.inf
        # make rewards better poisoned so that the policy performs badly
        for it in range(self.max_iter):
            
#            print("iteration", it)
            r_trainable = Variable(torch.FloatTensor(cur_r).to(self.device).unsqueeze(1), requires_grad=True)
            self.cp_net()
        
            with higher.innerloop_ctx(self.im_critic, self.im_critic_optim) as (critic, critic_optim):
                
                with torch.no_grad():
                    next_state_action, next_state_log_pi, _ = self.im_policy.sample(old_next_states)
                    qf1_next_target, qf2_next_target = self.im_critic_target(old_next_states, next_state_action)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi

                next_q_value = r_trainable + old_masks * self.gamma * (min_qf_next_target)
                qf1, qf2 = critic(old_states, old_actions) 
                qf1_loss = F.mse_loss(qf1, next_q_value)  
                qf2_loss = F.mse_loss(qf2, next_q_value) 
            
                critic_optim.step(qf1_loss)

                critic_optim.step(qf2_loss)

                pi, log_pi, _ = self.im_policy.sample(old_states)

                qf1_pi, qf2_pi = critic(old_states, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                
                good_qf1_pi, good_qf2_pi = self.good_critic(old_states, pi)
                good_min_qf_pi = torch.min(good_qf1_pi, good_qf2_pi)
                
                # want the learned q values be as bad as possible
                attack_loss = - torch.norm(min_qf_pi - good_min_qf_pi)
                print("loss:", attack_loss.item())
                attack_loss.backward()
                grad = r_trainable.grad.numpy().flatten()
                
                norm_grad = np.linalg.norm(grad)
                if norm_grad > 0:
                    cur_r -= grad * self.stepsize / norm_grad
                
                norm = np.linalg.norm(cur_r-old_r)
                print("change of r:", norm, end="  ")
                if norm > self.radius:
                    cur_r = (old_r + (cur_r - old_r) * self.radius / norm)
                    break
                if old_loss - attack_loss.item() < 1e-8:
                    break
                old_loss = attack_loss.item()
            
        return cur_r.tolist()
    
    def attack_r_general(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch):
        '''attack with a sampled mini-batch'''
        if self.attack_num >= self.maxat:
            print("exceeds budget")
            return memory.rewards
        
        # learner a good q value function
        self.update_critic(state_batch, action_batch, reward_batch, next_state_batch, mask_batch)
        
        cur_r = reward_batch.copy()
        T = len(cur_r)
        cur_r = np.array(cur_r)
        old_r = np.copy(cur_r)
        
        # convert list to tensor
        old_states = torch.FloatTensor(state_batch).to(self.device)
        old_actions = torch.FloatTensor(action_batch).to(self.device)
        old_rewards = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        old_masks = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        
        old_discounted_rewards = self.get_discount_rewards(old_rewards, old_masks).detach()
        
        # copy weights from the learner
        self.cp_net()
        
        # a normal update without poisoning
        sac_update(self.im_policy, self.im_policy_optim, self.im_critic, self.im_critic_optim, 
                   self.im_critic_target, state_batch, action_batch, reward_batch, next_state_batch, 
                   mask_batch, self.alpha, self.gamma, self.device)
        
#        self.print_paras(self.im_policy)
        true_action_dists = self.im_policy.get_dist(old_states)
        
        # compute policy loss
        true_obj = self.im_policy_obj(old_states)
        print("obj:", true_obj)
        grads = np.zeros(T)
        
        for t in range(T):
            print(t,end=" ")
            # change r[t] a little
            cur_r[t] += self.delta
            # a imitating update with one poisoned reward
            self.cp_net()
            sac_update(self.im_policy, self.im_policy_optim, self.im_critic, self.im_critic_optim, 
                       self.im_critic_target, state_batch, action_batch, cur_r, next_state_batch, 
                       mask_batch, self.alpha, self.gamma, self.device)

            poison_obj = self.im_policy_obj(old_states)
            grads[t] = poison_obj - true_obj
            cur_r = old_r.copy()
        
#        print("grad of r:", grads)
        if np.linalg.norm(grads) > 0:
            cur_r = old_r - self.radius * grads / np.linalg.norm(grads)
#        print("cur_r", cur_r)
        
        # update use the new rewards
        self.cp_net()
        sac_update(self.im_policy, self.im_policy_optim, self.im_critic, self.im_critic_optim, 
                       self.im_critic_target, state_batch, action_batch, cur_r, 
                   next_state_batch, mask_batch, self.alpha, self.gamma, self.device)
        poison_obj = self.im_policy_obj(old_states)
        print("poisoned obj:", poison_obj)
        
        poison_action_dists = self.im_policy.get_dist(old_states)
        
        dist_distance = torch.distributions.kl.kl_divergence(true_action_dists, poison_action_dists).mean()
        print("distribution distance:", dist_distance)
        
        self.dist_list = np.append(self.dist_list, np.array([dist_distance]))
        
        frac = min((self.maxat - self.attack_num) / (self.maxeps - self.eps_num),1)
        self.eps_num += 1
        
        if not self.rand_select:
            if dist_distance >= np.quantile(self.dist_list, 1-frac):
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
    
    def compute_radius(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch):
        '''compute the upper bound of the stability radius'''
        
        # learner a good q value function
        self.update_critic(state_batch, action_batch, reward_batch, next_state_batch, mask_batch)
        
        cur_r = reward_batch.copy()
        T = len(cur_r)
        cur_r = np.array(cur_r)
        old_r = np.copy(cur_r)
        
        # convert list to tensor
        old_states = torch.FloatTensor(state_batch).to(self.device)
        old_actions = torch.FloatTensor(action_batch).to(self.device)
        old_rewards = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        old_masks = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        
        old_discounted_rewards = self.get_discount_rewards(old_rewards, old_masks).detach()
        
        # copy weights from the learner
        self.cp_net()
        
        # a normal update without poisoning
        sac_update(self.im_policy, self.im_policy_optim, self.im_critic, self.im_critic_optim, 
                   self.im_critic_target, state_batch, action_batch, reward_batch, next_state_batch, 
                   mask_batch, self.alpha, self.gamma, self.device)
        
#        self.print_paras(self.im_policy)
        true_action_dists = self.im_policy.get_dist(old_states)
        true_obj = self.im_policy_obj(old_states)
        
        it = 0
        dist_distance = 0
        last_r = cur_r.copy()
        while dist_distance < self.dist_thres:
            it += 1
            
            grads = np.zeros(T)

            for t in range(T):
                # copy weights from the learner
                self.cp_net()
                # change r[t] a little
                cur_r[t] += self.delta
                # a imitating update with one poisoned reward
                sac_update(self.im_policy, self.im_policy_optim, self.im_critic, self.im_critic_optim, 
                   self.im_critic_target, state_batch, action_batch, cur_r, next_state_batch, 
                   mask_batch, self.alpha, self.gamma, self.device)
                
                poison_obj = self.im_policy_obj(old_states)
                grads[t] = poison_obj - true_obj
                cur_r = last_r.copy()

    #        print("grad of r:", grads)
            if np.linalg.norm(grads) > 0:
                cur_r = last_r - self.stepsize * grads / np.linalg.norm(grads)

            # update with the new rewards
            self.cp_net()
            sac_update(self.im_policy, self.im_policy_optim, self.im_critic, self.im_critic_optim, 
                   self.im_critic_target, state_batch, action_batch, cur_r, next_state_batch, 
                   mask_batch, self.alpha, self.gamma, self.device)
            
            last_r = cur_r.copy()
            poison_action_dists = self.im_policy.get_dist(old_states)

            dist_distance = torch.distributions.kl.kl_divergence(true_action_dists, poison_action_dists).max()
            print("distribution distance:", dist_distance)
            if it > self.max_iter:
                return np.inf
        
        return np.linalg.norm(cur_r - old_r)
    
    def cp_net(self):
        self.im_critic.load_state_dict(self.learner.critic.state_dict())
        self.im_critic_optim.load_state_dict(self.learner.critic_optim.state_dict())

        self.im_critic_target.load_state_dict(self.learner.critic_target.state_dict())

        self.im_policy.load_state_dict(self.learner.policy.state_dict())
        self.im_policy_optim.load_state_dict(self.learner.policy_optim.state_dict())
        
    
    def update_critic(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch): 
    
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.im_policy.sample(next_state_batch)
            
            qf1_next_target, qf2_next_target = self.good_critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi

            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        qf1, qf2 = self.good_critic(state_batch, action_batch) 

        qf1_loss = F.mse_loss(qf1, next_q_value)  
        qf2_loss = F.mse_loss(qf2, next_q_value)  

        self.good_critic_optim.zero_grad()
        qf1_loss.backward()
        self.good_critic_optim.step()

        self.good_critic_optim.zero_grad()
        qf2_loss.backward()
        self.good_critic_optim.step()
    
    def im_policy_obj(self, states):
        pi, log_pi, _ = self.im_policy.sample(states)

        qf1_pi, qf2_pi = self.good_critic(states, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = (min_qf_pi - (self.alpha * log_pi)).mean()
        
        return policy_loss
    
    def get_discount_rewards(self, rewards, is_terminals):
        T = len(rewards)
        discounted_rewards = torch.empty(T, requires_grad=False).to(self.device)

        Gt = 0
        t = T-1
        for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
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