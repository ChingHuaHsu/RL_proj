import gym
from gym import spaces
import numpy as np
import random
import time
import itertools

class TwoAOneBEnv(gym.Env):
    def __init__(self):
        # 定义动作空间（Action Space）和观察空间（Observation Space）
        # self.action_space = gym.spaces.Discrete(48)  # 4位数字可以有8种交换位置的组合，每位数字可以有10种改变数值的动作
        self.action_space = gym.spaces.Discrete(len(list(itertools.permutations(range(10), 4))))
        self.observation_space = gym.spaces.Box(low=0, high=10, shape=(6,), dtype=np.int32)

        # 初始化游戏状态和目标状态
        self.game_state = np.array([0, 0, 0, 0 ,0 ,0], dtype=np.int32)  # 初始游戏状态：[A1, A2, B1, B2, # of A, # of B]
        self.target = np.array([1, 2, 3, 4], dtype=np.int32)  # 目标状态，例如[1, 2, 3, 4]
        self.all_combinations = list(itertools.permutations(range(10), 4))

    def step(self, action):
        # print(f"action= {action}")
        # 执行动作并计算奖励
        # if action < 40:  # 修改数值的动作
        #     position = action // 10  # 要修改的位置
        #     value = action % 10  # 要修改的数值
        #     if value != self.game_state[position]:  # 确保至少有一位数字变化
        #         self.game_state[position] = value
        # else:  # 交换位置的动作
        #     position1 = (action - 40) // 4
        #     position2 = (action - 40) % 4
        #     self.game_state[position1], self.game_state[position2] = self.game_state[position2], self.game_state[position1]
        
        self.all_combinations = list(itertools.permutations(range(10), 4))
        self.game_state[:4] = self.all_combinations[action[0]]

        done = np.array_equal(self.game_state[:4], self.target)  # 判断是否达到目标状态
        reward = self._calculate_reward()  # 根据A和B的数量计算奖励return np.array(self.game_state, dtype=np.int32), reward, done, {"A": self._calculate_AB()[0], "B": self._calculate_AB()[1], "episode": {"r": reward}}
        # print("guess:", self.game_state[:4] , "A:" , self.game_state[4] , "B" , self.game_state[5])
        return np.array(self.game_state, dtype=np.int32), reward, done, {"A": self._calculate_AB()[0], "B": self._calculate_AB()[1], "episode": {"r": reward}}

    def reset(self):
        # 重置游戏状态
        if np.array_equal(self.game_state[:4], self.target):
            print("guess:", self.game_state[:4] , "A:" , self.game_state[4] , "B" , self.game_state[5])
            self.target = np.array(random.sample(range(10), 4), dtype=np.int32)
            print(self.target)
            time.sleep(5)
        print("game reset")
        self.game_state = np.array([0, 0, 0, 0 ,0 ,0], dtype=np.int32)
        return np.array(self.game_state, dtype=np.int32)

    def render(self, mode='human'):
        # 绘制游戏状态
        print(f"Current game state: {self.game_state}")

    def close(self):
        # 清理环境
        pass

    def _calculate_AB(self):
        # A = sum([1 for i in range(4) if self.game_state[i] == self.target[i]])
        # B = 0
        # for i in range(4):
        #     if self.game_state[i] != self.target[i] and self.game_state[i] in self.target[i]:
        #         B += 1
        # B = sum([1 for i in range(4) if self.game_state[i] != self.target[i] and self.game_state[i] in self.target])
        A=0
        B=0
        target_count = {}
        for num in self.target:
            if num in target_count:
                target_count[num] += 1
            else:
                target_count[num] = 1
        for i in range(4):
            if self.game_state[i] == self.target[i]:
                A += 1
                target_count[self.game_state[i]] -= 1
        for i in range(4):
            if self.game_state[i] != self.target[i] and self.game_state[i] in target_count and target_count[self.game_state[i]] > 0:
                B += 1
                target_count[self.game_state[i]] -= 1
    
        self.game_state[4] = A
        self.game_state[5] = B
        return A, B

    def _calculate_reward(self):
        A, B = self._calculate_AB()
        # 根据A、B的值计算奖励
        if A == 0 and B == 0:
            r = -0.5
            return r
        else:
            r = A*1 + B*0.5
            # if r >= 3:
            #     print("guess:", self.game_state[:4] , "A:" , self.game_state[4] , "B" , self.game_state[5])
            #     time.sleep(1)
            return r
