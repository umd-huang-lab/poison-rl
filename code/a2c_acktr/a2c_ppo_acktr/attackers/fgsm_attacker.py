import sys
import torch  
import numpy as np  
import random
from gym.spaces import Box, Discrete

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad

    return perturbed_image

def policyList(state):
    return 1

class FGSMAttacker:
    def __init__(self, envs, learner, target_policy, radius=0.5, frac=1.0, maxat=1000, device="cpu"):
        super(FGSMAttacker, self).__init__()

        self.learner = learner
        self.target_policy = target_policy
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
        
    def attack(self, obs, recurrent, mask):

        action_adv = self.target_policy(obs)
        obs.requires_grad = True
#        print("obs", obs)
#        print("len", N)
#        print("action adv", action_adv)
        # want it larger
        dist = self.learner.actor_critic.get_dist(obs, recurrent, mask)
#        print("dist", dist.probs)
        loss = dist.probs[:,action_adv]
#        print("loss", loss)
        
        self.learner.optimizer.zero_grad()
        # Calculate gradients of model in backward pass
        loss.mean().backward()
        # Collect datagrad
        state_grad = obs.grad.data
#        print("grad", state_grad)
        # Call FGSM Attack
        perturbed_state = fgsm_attack(obs, self.radius/2, state_grad)
#        print("attack", perturbed_state)
#        dist = self.learner.actor_critic.get_dist(perturbed_state, recurrent, mask)
#        print("dist after poison", dist.probs)
        return perturbed_state.detach()