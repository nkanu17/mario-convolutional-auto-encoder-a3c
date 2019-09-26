# the following class was inspired from the following github:
# https://github.com/Ameyapores/Mario

import gym_super_mario_bros
from gym.spaces import Box
from gym import Wrapper
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import cv2
import numpy as np
import subprocess as sp


def process_frame(frame):
    if frame is not None:
        #print('Frame shape before convertion: {}'.format(frame.shape))

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        #print('Frame shape between convertion: {}'.format(frame.shape))
        # changing the size to match auto encoder and reshaping from (128,128,3) to (3, 128, 128)
        frame = cv2.resize(frame, (84, 84))[None, :, :] / 255.
        frame = frame.reshape(1,84,84)
        #print('Frame shape after convertion: {}'.format(frame.shape))

        return frame
    else:
        return np.zeros((1, 84, 84))


class GetReward(Wrapper):
    def __init__(self, env=None):
        super(GetReward, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.curr_score = 0
        

    def step(self, action):
        state, reward, done, info = self.env.step(action)
     
        #print('Frame stateshape before convertion: {}'.format(state.shape))
        state = process_frame(state)
        #print('Frame')
        
        reward += (info["score"] - self.curr_score) / 40.
        self.curr_score = info["score"]
        if done:
            if info["flag_get"]:
                reward += 50
            else:
                reward -= 50
        return state, reward / 10., done, info

    def reset(self):
        self.curr_score = 0
        #print('Frame env reset: {}'.format(env.reset.shape))
        return process_frame(self.env.reset())


class GetFrame(Wrapper):
    def __init__(self, env, skip=4):
        super(GetFrame, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(4, 84, 84)) #4*3,84,84
        self.skip = skip

    def step(self, action):
        total_reward = 0
        states = []
        state, reward, done, info = self.env.step(action)
        for i in range(self.skip):
            if not done:
                state, reward, done, info = self.env.step(action)
                total_reward += reward
                states.append(state)
            else:
                states.append(state)
        states = np.concatenate(states, 0)[None, :, :, :]
        return states.astype(np.float32), reward, done, info

    def reset(self):
        state = self.env.reset()
        states = np.concatenate([state for _ in range(self.skip)], 0)[None, :, :, :]
        return states.astype(np.float32)


def build_environment(world, stage):

    env_name = "SuperMarioBros-{}-{}-v0".format(world, stage)
    env = gym_super_mario_bros.make(env_name)

    actions = COMPLEX_MOVEMENT
    env = JoypadSpace(env, actions)
    env = GetReward(env)
    env = getFrame(env)

    return env, env.observation_space.shape[0], len(actions)
