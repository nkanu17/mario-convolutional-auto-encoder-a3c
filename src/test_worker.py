import torch
import torch.nn.functional as F
import torchvision
from torchvision import models
import torchvision.transforms as transforms
from torchvision import utils

import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms

from torch.distributions import Categorical
from collections import deque
from tensorboardX import SummaryWriter
import timeit

import time
import matplotlib
import matplotlib.pyplot as plt

from src.environment import build_environment
from src.optimizer import SharedAdam
from src.model import ActorCritic
from src.model import Convolutional_AutoEncoder

#from convolutional_autoencoder import Convolutional_AutoEncoder


from skimage import img_as_float


def test_a3c(index, args, A3C_shared_model, CAE_shared_model):
    
    #load CAE models weights into this project
    CAE_shared_model.load_state_dict(torch.load(args.pretrained_model_weights_path), strict = False)
    CAE_shared_model.eval()
    
    torch.manual_seed(123 + index)
    #instantiate the environment
    env, num_states, num_actions = build_environment(args.world, args.stage)
    #initialize CAE worker modell
    CAE_local_model = Convolutional_AutoEncoder()
    #load weights 
    CAE_local_model.load_state_dict(torch.load(args.pretrained_model_weights_path), strict = False)
    CAE_local_model.eval()
    #initialize A3C part of the model
    a3c_local_model = ActorCritic(num_states, num_actions)
    a3c_local_model.eval()
    state = torch.from_numpy(env.reset())
    done = True
    curr_step = 0
    actions = deque(maxlen=args.max_actions)
    # this loop runs until number of steps expire
    while True:
        curr_step += 1
        
        if done:
            a3c_local_model.load_state_dict(A3C_shared_model.state_dict())
        with torch.no_grad():
            if done:
                hx = torch.zeros((1, 512), dtype=torch.float)
                cx = torch.zeros((1, 512), dtype=torch.float)
            else:
                hx = hx.detach()
                cx = cx.detach()
        #send state output as input into CAE
        outputs_cae = CAE_local_model(state)
        #CAE output into A3C model as input 
        #in return receive the policy, value, and memories and then choose action
        logits, value, hx, cx = a3c_local_model(outputs_cae, hx, cx)
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        #use the chosen action and make step
        state, reward, done, _ = env.step(action)
        env.render()
        actions.append(action)
        # if finished or defeated, finish the test
        if curr_step > args.max_steps or actions.count(actions[0]) == actions.maxlen:
            done = True
        if done:
            curr_step = 0
            actions.clear()
            state = env.reset()
        state = torch.from_numpy(state)
