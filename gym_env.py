import copy
import sys

import gym
import numpy as np
from gym import spaces
from matplotlib import pyplot as plt

from harl.envs.gym.auction_env import P2P

import scipy.io as sio
import torch
import torch.nn as nn


class GYMEnv:
    def __init__(self, args,algo):
        self.args = copy.deepcopy(args)
        self.env = P2P()
        # self.safe = safe()
        self.algo = algo
        self.n_agents = 99
        self.ti = 0
        """self.share_observation_space = [self.env.observation_space]
        self.observation_space = [self.env.observation_space]
        self.action_space = [self.env.action_space]"""


        self.action_space = [
            spaces.Box(low=-1, high=+1, shape=(4,), dtype=np.float32)
            for _ in range(self.n_agents)
        ]

        self.observation_space = [
            spaces.Box(low=-1, high=+1, shape=(6,), dtype=np.float32)
            for _ in range(self.n_agents)
        ]
        self.share_observation_space =[
            spaces.Box(low=-1, high=+1, shape=(6*self.n_agents,), dtype=np.float32)
            for _ in range(self.n_agents)
        ]

        self.discrete = False


        # self.hpri_record = []
        # self.epri_record = []
        self.V_bus = []
        self.cost = []
        self.buyer_list = []
        self.seller_list = []
        self.P_da = []
        self.pri = []
        self.c_da = []
        self.B_bid = []
        self.S_bid = []
        self.prilist8 = []
        self.rewardlist = []
        self.reward=[]
        self.cost1=[]
        self.pri1 = []
        self.pri2 = []
        self.pri3 = []
        self.pri4 = []
        self.pri5 = []
        self.pri6 = []
        self.pri7 = []
        self.pri8 = []
        self.emission=[]
        self.variable=[]
        self.res = []
        self.line=[]

    def step(self, actions):
        """
        return local_obs, global_state, rewards, dones, infos, available_actions

        """
        def clip_actions(actions):
            return np.clip(actions, -1, 1)

        actions2 = clip_actions(actions)  ## 为了方便 on policy算法使用这个函数，提前设置一个clip，修正一下动作 （类似TRPO和 PPO）
        # #

        num_agents=99
        obs, s_obs,rew, done, info, cost, buyer, seller, P_da, c_da= self.env.step(*[actions2[i, 0:4] for i in range(num_agents)])
        # #累加log_info和reward

        self.buyer_list.append(buyer)
        self.seller_list.append(seller)
        self.cost.append(cost)
        self.c_da.append(c_da)
        self.P_da.append(P_da)
        self.rewardlist.append(rew[0])
        if done[0]:
            if (
                    "TimeLimit.truncated" in info.keys()
                    and info["TimeLimit.truncated"] == True
            ):
                info["bad_transition"] = True
            # # # 训练的时候注释掉，测试专用画图。
            # plt.figure()
            # plt.plot(self.V_bus,label="V")
            # plt.legend()
            # # plt.figure()
            # # # plt.plot(self.DR, label="dr")
            # # # plt.legend()
            # # # plt.figure()
            # plt.show()
            self.ti += 1  ## 这里新学的写法，list是24小时的，但是array里面是数组

            self.env.reset()
            if self.ti % 50 == 0 and self.ti > 0:
                sio.savemat('sactrading' + '.mat',
                            {'sacreward': self.reward, 'saccost': self.cost, 'sacbuyer':self.buyer_list,'sacseller':self.seller_list, 'quanity':self.P_da, 'c_dao2':self.c_da})
                # plt.figure()
                # plt.plot(self.reward, label="reward")
                # plt.legend()
                # plt.show()
            self.reward.append(float(sum(self.rewardlist)))
            # self.cost1.append(np.sum(self.cost1list, axis=0))

            self.rewardlist = []
            self.cost1list = []
            self.buyer_list = []
            self.seller_list= []
            self.P_da = []
            self.pri = []
            self.c_da = []
            self.B_bid = []
            self.S_bid = []
            self.prilist8 = []
            self.variable = []
            self.cost = []
            self.line = []


        #缩放obs的范围，防止过大或过小
        for i in range(len(obs)):
            for j in range(len(obs[i])):
                if obs[i][j]>5000:
                    obs[i][j]=5000
                if obs[i][j]<-5000:
                    obs[i][j]=-5000

        return obs, s_obs, rew, done, [info], self.get_avail_actions()

    def reset(self):
        """Returns initial observations and states"""
        obs = self.env.reset()
        num_agents = 99
        single_array = np.ones((1,6*num_agents))
        return obs,single_array, self.get_avail_actions()

    def get_avail_actions(self):
        if self.discrete:
            avail_actions = [[1] * self.action_space[0].n]
            return avail_actions
        else:
            return None

    def render(self):
        # self.env.render()
        pass

    def close(self):
        # self.env.close()
        pass

    def seed(self, seed):
        # self.env.seed(seed)
        pass
