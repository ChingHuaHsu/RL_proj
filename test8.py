import gym
from gym import spaces
import numpy as np
import random

class TwoAOneBEnv(gym.Env):
    def __init__(self):
        # 定义动作空间（Action Space）和观察空间（Observation Space）
        self.action_space = gym.spaces.Discrete(48)  # 4位数字可以有8种交换位置的组合，每位数字可以有10种改变数值的动作
        self.observation_space = gym.spaces.Box(low=0, high=10, shape=(4,), dtype=np.int32)

        # 初始化游戏状态和目标状态
        self.game_state = np.array([0, 0, 0, 0], dtype=np.int32)  # 初始游戏状态：[A1, A2, B1, B2]
        self.target = np.array([1, 2, 3, 4], dtype=np.int32)  # 目标状态，例如[1, 2, 3, 4]

    def step(self, action):
        # 执行动作并计算奖励
        if action < 40:  # 修改数值的动作
            position = action // 10  # 要修改的位置
            value = action % 10  # 要修改的数值
            if value != self.game_state[position]:  # 确保至少有一位数字变化
                self.game_state[position] = value
        else:  # 交换位置的动作
            position1 = (action - 40) // 4
            position2 = (action - 40) % 4
            self.game_state[position1], self.game_state[position2] = self.game_state[position2], self.game_state[position1]

        done = np.array_equal(self.game_state, self.target)  # 判断是否达到目标状态
        reward = self._calculate_reward()  # 根据A和B的数量计算奖励

        return np.array(self.game_state, dtype=np.int32), reward, done, {"A": self.A, "B": self.B}

    def reset(self):
        # 重置游戏状态
        self.game_state = np.array([0, 0, 0, 0], dtype=np.int32)
        return np.array(self.game_state, dtype=np.int32)

    def render(self, mode='human'):
        # 绘制游戏状态
        print(f"Current game state: {self.game_state}")

    def close(self):
        # 清理环境
        pass

    def _calculate_AB(self):
        A = sum([1 for i in range(4) if self.game_state[i] == self.target[i]])
        B = 0
        for i in range(4):
            if self.game_state[i] != self.target[i] and self.game_state[i] in self.target:
                B += 1
        return A, B

    def _calculate_reward(self):
        A, B = self._calculate_AB()
        # 根据A、B的值计算奖励
        if A == 4:  # 全部正确
            return 1.0
        elif A == 3 and B == 1:  # 三A一B
            return 0.8
        elif A == 2 and B == 2:  # 两A两B
            return 0.6
        elif A == 1 and B == 3:  # 一A三B
            return 0.4
        elif A == 0 and B == 4:  # 四B
            return 0.2
        else:
            return -0.1


# 创建环境
env = TwoAOneBEnv()

# 运行示例代码
for episode in range(10):
    state = env.reset()
    done = False
    print(f"Episode {episode + 1} starting...")
    while not done:
        action = env.action_space.sample()  # 随机选择动作
        next_state, reward, done, info = env.step(action)
        env.render()
        print(f"A: {info['A']}, B: {info['B']}, Action: {action}, reward: {reward}")
    print(f"Episode {episode + 1} ended.\n")