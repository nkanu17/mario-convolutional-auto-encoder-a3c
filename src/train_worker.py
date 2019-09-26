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


from torchsummary import summary

def train_a3c(index, args, A3C_optimizer,A3C_shared_model, CAE_shared_model, CAE_optimizer ,save=True,new_stage=False):
    
    # load the weights of pretrained model. In pytorch if the model is saved with GPU
    #the data is mapped differently 
    if (args.use_cuda):
        CAE_shared_model.load_state_dict(torch.load(args.pretrained_model_weights_path), strict = False)
    else:
        CAE_shared_model.load_state_dict(torch.load(args.pretrained_model_weights_path), strict = False, map_location=device)
    CAE_shared_model.eval()
    torch.manual_seed(123 + index)
    
     
    if save:
        start_time = timeit.default_timer()
    #tensorboard
    writer = SummaryWriter(args.sum_path)
    #initialize the environment
    # in return, environment, states, action (12)
    env, num_states, num_actions = build_environment(args.world, args.stage)
    print('Num of states: {}'.format(num_states))
    #CAE model worker
    CAE_local_model = Convolutional_AutoEncoder()
    #a3c model Worker
    a3c_local_model = ActorCritic(num_states, num_actions)
    #use cuda
    if args.use_cuda:
        a3c_local_model.cuda()
        CAE_local_model.cuda()
    #train the a3c but not the cae as we need itto have frozen weights   
    a3c_local_model.train()
    state = torch.from_numpy(env.reset())
    
    if args.use_cuda:
        state = state.cuda()
    done = True
    curr_step = 0
    curr_episode = 0
    #Runs for each episode until the max episode limit is reached
    while True:
        #save model every 500 episodes
        if save:
            if curr_episode % args.save_interval == 0 and curr_episode > 0:
                
                torch.save(CAE_shared_model.state_dict(),"{}/CAE_super_mario_bros_{}_{}_enc2".format(args.trained_models_path, args.world, args.stage))
                                                                                                     
                torch.save(A3C_shared_model.state_dict(),
                           "{}/a3c_super_mario_bros_{}_{}_enc2".format(args.trained_models_path, args.world, args.stage))
            print("Process {}. Episode {}".format(index, curr_episode))
        curr_episode += 1
        a3c_local_model.load_state_dict(A3C_shared_model.state_dict())
        
        # strict is false because we are only loading the weights for the encoder layers
        # gpu check
        if(args.use_cuda):
            
            CAE_local_model.load_state_dict(torch.load("{}/CAE_super_mario_bros_1_1_enc2".format(args.trained_models_path)))
        else:

            CAE_local_model.load_state_dict(torch.load("{}/CAE_super_mario_bros_1_1_enc2".format(args.trained_models_path),map_location='cpu'))
        
        CAE_local_model.eval()
        #empty tensors that will the first time per episode to the LSTM
        if done:
            hx = torch.zeros((1, 512), dtype=torch.float)
            cx = torch.zeros((1, 512), dtype=torch.float)
        else:
            hx = hx.detach()
            cx = cx.detach()
        if args.use_cuda:
            hx = hx.cuda()
            cx = cx.cuda()
        
        
        log_policies = []
        values = []

        rewards = []
        entropies = []
        # start training for each step in an episode
        for _ in range(args.num_steps):
            curr_step += 1
            #Freezing CAE local models weights so it does not get updates
            for param in CAE_local_model.parameters():
                param.requires_grad = False
            #the state is ent as input to CAE Local model which returns an output 
            #from the last layer of the encoder
            output_cae = CAE_local_model(state)
            #then the output of cae is sent to the a3c part of the model, 
            #which returns the values, policy, and memories
            logits, value, hx, cx = a3c_local_model
            #best policy and log policy is determined through softmax
            #action is chosen through the probability 
            #distributions of the policy and entropy
            policy = F.softmax(logits, dim=1)
            log_policy = F.log_softmax(logits, dim=1)
            entropy = -(policy * log_policy).sum(1, keepdim=True)
            m = Categorical(policy)
            action = m.sample().item()
            #using the chosen action to take a step in the environment
            #at return reward for the action and next state 
            state, reward, done, _ = env.step(action)
            state = torch.from_numpy(state)
            #check if max episodes reached
            if args.use_cuda:
                state = state.cuda()
            if (curr_step > args.max_steps) or (curr_episode > int(args.max_episodes)):
                done = True
            #check if level is done
            if done:
                curr_step = 0
                state = torch.from_numpy(env.reset())
                if args.use_cuda:
                    state = state.cuda()
            #append all relavant data to use it in the next iteration
            values.append(value)
            log_policies.append(log_policy[0, action])
            rewards.append(reward)
            entropies.append(entropy)

            if done:
                break

        R = torch.zeros((1, 1), dtype=torch.float)
        if args.use_cuda:
            R = R.cuda()
        if not done:
            output_ = CAE_local_model(state)
            _, R, _, _ = a3c_local_model(output_, hx, cx)

        gae = torch.zeros((1, 1), dtype=torch.float)
        if args.use_cuda:
            gae = gae.cuda()
        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
        next_value = R
        # calculate loss
        for value, log_policy, reward, entropy in list(zip(values, log_policies, rewards, entropies))[::-1]:
            gae = gae * args.gamma * args.tau
            gae = gae + reward + args.gamma * next_value.detach() - value.detach()
            next_value = value
            actor_loss = actor_loss + log_policy * gae
            R = R * args.gamma + reward
            critic_loss = critic_loss + (R - value) ** 2 / 2
            entropy_loss = entropy_loss + entropy

        total_loss = -actor_loss + critic_loss - args.beta * entropy_loss
        writer.add_scalar("Train_{}/Loss".format(index), total_loss, curr_episode)
        A3C_optimizer.zero_grad()
        total_loss.backward()
        #update model
        for local_param, global_param in zip(a3c_local_model.parameters(), A3C_shared_model.parameters()):
            if global_param.grad is not None:
                break
            global_param._grad = local_param.grad

        A3C_optimizer.step()

        if curr_episode == int(args.max_episodes):
            print("Training process {} terminated".format(index))
            if save:
                end_time = timeit.default_timer()
                print('The code runs for %.2f s ' % (end_time - start_time))
            return


