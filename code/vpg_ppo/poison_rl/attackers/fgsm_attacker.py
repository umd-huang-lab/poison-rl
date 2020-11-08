import sys
import torch  
import numpy as np  
import math
import random
from gym.spaces import Box, Discrete

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad

    return perturbed_image


class FGSMAttacker:
    def __init__(self, learner, action_space, target_policy, radius=0.1, frac=1.0, maxat=1000, device="cpu"):
        super(FGSMAttacker, self).__init__()

        self.learner = learner
        self.target_policy = target_policy
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
    
    def attack(self, obs):
        obs = torch.tensor(obs, dtype = torch.float).to(self.device)

        action_adv = self.target_policy(obs)
        obs.requires_grad = True
#        print("obs", obs)
#        print("len", N)
#        print("action adv", action_adv)
        # want it larger
        dist = self.learner.policy.get_dist(obs, self.device)
#        print("dist", dist.probs)
        loss = dist.probs[action_adv]
#        print("loss", loss)
        
        self.learner.optimizer.zero_grad()
        # Calculate gradients of model in backward pass
        loss.backward()
        # Collect datagrad
        state_grad = obs.grad.data
#        print("grad", state_grad)
        # Call FGSM Attack
        perturbed_state = fgsm_attack(obs, self.radius/2, state_grad)
        dist = self.learner.policy.get_dist(perturbed_state, self.device)
#        print("new dist", dist.probs)
        return perturbed_state.cpu().detach().numpy()
    
    
    
    