import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
from environment import create_train_env
from actor_critic_model import ActorCritic
from actor_critic_model import A3C_original

from convolutional_autoencoder import Convolutional_AutoEncoder
import torch.nn.functional as F

def run_test(args):
    torch.manual_seed(123)
   
    env, num_states, num_actions = create_train_env(args.world, args.stage)#, args.action_type,"{}/video_{}_{}.mp4".format(args.output_path, args.world, args.stage))
    CAE_model = Convolutional_AutoEncoder()
    
    if(args.a3c_only):
        a3c_model = A3C_original(num_states, num_actions)
    else: 
        a3c_model = ActorCritic(num_states, num_actions)

    if torch.cuda.is_available():
        if(args.a3c_only):
            a3c_model.load_state_dict(torch.load("{}/a3c_super_mario_bros_{}_{}".format(args.A3C_path, args.file_num, args.file_num)))
        else:
            a3c_model.load_state_dict(torch.load("{}/a3c_super_mario_bros_{}_{}_enc1".format(args.A3C_path, args.file_num,args.file_num)))
        a3c_model.cuda()
        CAE_model.cuda()
    else:
        if(args.a3c_only):
            a3c_model.load_state_dict(torch.load("{}/a3c_super_mario_bros_{}_{}".format(args.A3C_path, args.world, args.stage),
                                         map_location=lambda storage, loc: storage))
        else:
            a3c_model.load_state_dict(torch.load("{}/a3c_super_mario_bros_{}_{}_enc1".format(args.A3C_path, args.world, args.stage),
                                         map_location=lambda storage, loc: storage))
            
    CAE_model.load_state_dict(torch.load(args.CAE_path2), strict = False)

    a3c_model.eval()
    CAE_model.eval()
    state = torch.from_numpy(env.reset())
    done = True
    while True:
        if done:
            h_0 = torch.zeros((1, 512), dtype=torch.float)
            c_0 = torch.zeros((1, 512), dtype=torch.float)
            env.reset()
        else:
            h_0 = h_0.detach()
            c_0 = c_0.detach()
        if torch.cuda.is_available():
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()
            state = state.cuda()
        if(args.a3c_only):
            logits, value, h_0, c_0 = a3c_model(state, h_0, c_0)
        else:
            output_cae = CAE_model(state)
            logits, value, h_0, c_0 = a3c_model(output_cae, h_0, c_0)
            
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        action = int(action)
        state, reward, done, info = env.step(action)
        state = torch.from_numpy(state)
        env.render()
        if info["flag_get"]:
            print("World {} stage {} completed".format(args.world, args.stage))
            break

