import sys
import torch  
import gym
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from PPO import Memory
import random

class QNet(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size):
        super(QNet, self).__init__()
        # the action value network
        self.nn = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions)
        )

    def forward(self, x):
        return self.nn(x)


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def add(self, s0, a, r, s1, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((s0[None, :], a, r, s1[None, :], done))
        
    def sample(self, batch_size):
        s0, a, r, s1, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(s0), a, r, np.concatenate(s1), done

    def size(self):
        return len(self.buffer)
    

class BbAttacker:
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4, gamma=0.9,
                 max_iter=1, step_size=0.05, radius=0.5, update_gap=10):
        super(BbAttacker, self).__init__()
        # the action value network
        
        self.qnet = QNet(num_inputs, num_actions, hidden_size)
#        self.tnet = QNet(num_inputs, num_actions, hidden_size)
#        self.tnet.load_state_dict(self.qnet.state_dict())
#        self.tnet.eval()
        
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=learning_rate)
        
        self.buffer = ReplayBuffer(1000)
        
        self.num_actions = num_actions
        self.gamma = gamma
        self.max_iter = max_iter
        self.step_size = step_size
        self.radius = radius
#        self.update_gap = update_gap
#        self.update_ct = 0
    
#    def learn(self, obs_s, obs_a, obs_r):
#        
#        state_batch = torch.cat([torch.from_numpy(s).float().unsqueeze(0) for s in obs_s])
#        action_batch = torch.LongTensor(obs_a).view(len(obs_a),1)
##        print(action_batch)
#        reward_batch = torch.FloatTensor(obs_r)
#        next_state_batch = torch.cat([torch.from_numpy(s).float().unsqueeze(0) for s in obs_s[1:]])
#        
#        state_action_values = self.qnet(state_batch).gather(1, action_batch)
##        print(state_action_values)
#        
#        next_state_values = torch.zeros(len(obs_r))
#        next_state_values[:-1] = self.tnet(next_state_batch).max(1)[0].detach()
##        print(next_state_action_values)
#        
#        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
#        
#        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
##        print(loss)
#        # Optimize the model
#        self.optimizer.zero_grad()
#        loss.backward()
#        self.optimizer.step()
#        
#        self.update_ct += 1
#        if self.update_ct == self.update_gap:
#            self.tnet.load_state_dict(self.qnet.state_dict())
#            self.update_ct = 0
        
    def learning(self):
        s0, a, r, s1, done = self.buffer.sample(128)

        s0 = torch.tensor(s0, dtype=torch.float)
        s1 = torch.tensor(s1, dtype=torch.float)
        a = torch.tensor(a, dtype=torch.long)
        r = torch.tensor(r, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.float)


        q_values = self.qnet(s0)
        next_q_values = self.qnet(s1)
        next_q_value = next_q_values.max(1)[0]

        q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
        expected_q_value = r + self.gamma * next_q_value * (1 - done)
        # Notice that detach the expected_q_value
        loss = (q_value - expected_q_value.detach()).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

#        print("loss", loss.item())
        
    
    def act(self, state, epsilon=0.1):
        if random.random() > epsilon:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            q_value = self.qnet.forward(state)
            action = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.num_actions)
        return action
    
    def worst_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        q_value = self.qnet.forward(state)
        return q_value.min(1)[1].item()
    
    def attack_ep_r(self, obs_s, obs_a, obs_r):
        '''One-episode attack: change the rewards of one episode into poisoned rewards'''
        
        cur_r = obs_r.copy()
        T = len(cur_r)
        cur_r = np.array(cur_r)
        old_r = np.copy(cur_r)
        
        for it in range(self.max_iter):
            r_trainable = Variable(torch.FloatTensor(cur_r), requires_grad=True)
            discounted_rewards = torch.empty(T, requires_grad=False)
            for t in range(T):
                Gt = 0 
                pw = 0
                for r in r_trainable[t:]:
                    Gt = Gt + self.gamma**pw * r
                    pw = pw + 1
                discounted_rewards[t] = Gt
            mask = torch.ones(T)
            for t in range(T):
                wa = self.worst_action(obs_s[t])
                if wa == obs_a[t]: 
                    mask[t] = -1
            
            inv_reward = torch.dot(mask, discounted_rewards)
            inv_reward.backward()
#            print("grad:", r_trainable.grad)
            with torch.no_grad():
                cur_r -= self.step_size * r_trainable.grad.numpy()
                
            cur_r = self.proj(old_r, cur_r, self.radius)
            
        return cur_r.tolist()
            
    
    def r_gradient(self, learner, rewards, probs):
        '''compute gradient of r
            probs: the probabilities of the learner policy choosing the original actions 
        '''
        T = len(rewards)
        grad = torch.zeros(T)
        
        # new probabilities / log probabilities
        new_probs, new_log_probs = self.get_probs(learner.states, learner.actions)
        print("new probs", len(new_probs), len(new_log_probs))
        
        for t in range(T):
            print("t:", t)
            ratio = new_probs[t] / probs[t]
#            print("ratio:", ratio)
            
    def get_probs(self, states, actions):
        probs = []
        log_probs = []
        for s, a in zip(states, actions):
            prob, log_prob = self.get_action_prob(s,a)
            probs.append(prob)
            log_probs.append(log_prob)
        return probs, log_probs
    
    def update_policy(self, log_probs, rewards):
        '''Imitate the poicy update of the learner'''
        discounted_rewards = []
        for t in range(len(rewards)):
            Gt = 0 
            pw = 0
            for r in rewards[t:]:
                Gt = Gt + self.gamma**pw * r
                pw = pw + 1
            discounted_rewards.append(Gt)
            
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9) # normalize discounted rewards
    
        # get policy gradient of the new policy wrt the new rewards
            
        policy_gradient = []
        for log_prob, Gt in zip(log_probs, discounted_rewards):
            policy_gradient.append(-log_prob * Gt)
        
        self.optimizer.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum()
        policy_gradient.backward(retain_graph=True)
        self.optimizer.step()
        
        
    def print_paras(self, model):
        for param in model.parameters():
            print(param.data)
        
    def proj(self, old_r_array, new_r_array, radius):
        
        norm = np.linalg.norm(new_r_array-old_r_array)
#        print("dist of r:", norm)
        if norm > radius:
            proj_r = (old_r_array + (new_r_array - old_r_array) * radius / norm)
            return proj_r
        else:
            return new_r_array
        
        