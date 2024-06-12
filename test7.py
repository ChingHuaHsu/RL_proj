import gym
import numpy as np
import random

class _1A2BEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(48)  # 4位數字可以有8種交換位置的組合，每位數字可以有10種改變數值的動作
        self.observation_space = gym.spaces.Box(low=0, high=10, shape=(4,), dtype=np.int32)
        self.game_state = np.array([0, 0, 0, 0], dtype=np.int32)  # 初始遊戲狀態：[A1, A2, B1, B2]
        self.target = np.array([1, 2, 3, 4], dtype=np.int32)  # 目標狀態，例如[1, 2, 3, 4]

    def step(self, action):
        if action < 40:  # 修改數值的動作
            position = action // 10  # 要修改的位置
            value = action % 10  # 要修改的數值
            if value != self.game_state[position]:  # 确保至少有一位数字变化
                self.game_state[position] = value
        else:  # 交換位置的動作
            position1 = (action - 40) // 4
            position2 = (action - 40) % 4
            self.game_state[position1], self.game_state[position2] = self.game_state[position2], self.game_state[position1]

        # done = self.game_state == self.target
        done = np.array_equal(self.game_state, self.target)  # 判斷是否達到目標狀態
        reward = 1 if done else -0.1

        A, B = self._calculate_AB(self.game_state, self.target)

        return np.array(self.game_state), reward, done, {"A": A, "B": B}

    def reset(self):
        self.game_state = np.array([0, 0, 0, 0], dtype=np.int32)
        return np.array(self.game_state, dtype=np.int32)

    def render(self, mode='human'):
        print(f"Current game state: {self.game_state}")

    def close(self):
        pass

    def _calculate_AB(self, game_state, target):
        A = sum([1 for i in range(4) if game_state[i] == target[i]])
        B = 0
        for i in range(4):
            if game_state[i] != target[i] and game_state[i] in target:
                B += 1
        return A, B

# 创建环境
env = _1A2BEnv()

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
