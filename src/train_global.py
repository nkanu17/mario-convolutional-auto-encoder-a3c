import os

import gym
import src.environment
from src.environment import build_environment
from src.model import ActorCritic
from src.model import Convolutional_AutoEncoder
from src.train_worker import train_a3c
from src.test_worker import test_a3c
from src.model import SharedAdam

import numpy as np
import torch
import torch.cuda
import torch.multiprocessing as _mp

import shutil

def shared_learn(args):
   
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.manual_seed(123)
    # create path for logs
    if os.path.isdir(args.sum_path):
        shutil.rmtree(args.sum_path)
    os.makedirs(args.sum_path)
    if not os.path.isdir(args.trained_models_path):
        os.makedirs(args.trained_models_path)
    mp = _mp.get_context('spawn')
   
    # create initial mario environment
    env, num_states, num_actions = build_environment(args.world, args.stage)
   
    print('Num of states: {}'.format(num_states)) #4
    print('environment: {}'.format(env))
    print('Num of actions: {}'.format(num_actions)) #12
   
    # check if cuda is available else cpu
    device = torch.device('cuda' if (args.use_cuda and torch.cuda.is_available()) else 'cpu')    
   
    CAE_shared_model = Convolutional_AutoEncoder()#.to(device)
    A3C_shared_model = ActorCritic(num_states, num_actions)#.to(device)
    # if a new stage, then it picks up previous saved model 
    if args.new_stage:
        A3C_shared_model.load_state_dict(torch.load('{}/a3c_super_mario_bros_{}_{}_enc2'.format(args.world,args.stage,args.trained_models_path)))
        A3C_shared_model.eval()
    # GPU check
    if(args.use_cuda and torch.cuda.is_available()):
        A3C_shared_model.cuda()
        CAE_shared_model.cuda()
    # shares memory with worker instances   
    CAE_shared_model.share_memory()
   
    A3C_shared_model.share_memory()
   
    print('A3C')
    print(A3C_shared_model)
    # intialize optimizer
    optimizer_cae = CAE_shared_model.createLossAndOptimizer(CAE_shared_model, 0.001)
    optimizer_a3c = SharedAdam(A3C_shared_model.parameters(), lr = args.lr)
    #optimizer.share_memory()
   
    # processes
    workers = []

    # start train process (run for the set number of workers)
    for rank in range(args.num_processes):
        if rank == 0:
            worker = mp.Process(target=train_a3c, args=(rank, args, optimizer_a3c, A3C_shared_model, CAE_shared_model, optimizer_cae,  True))
        else:
            worker = mp.Process(target=train_a3c, args=(rank, args, optimizer_a3c, A3C_shared_model, CAE_shared_model, optimizer_cae,True))
        worker.start()
        worker.append(worker)
   
    # test worker
    worker = mp.Process(target=test_a3c, args=(rank, args, A3C_shared_model, CAE_shared_model))
    worker.start()
    workers.append(worker)
   
    # join all processes
    for worker in workers:
        worker.join()
