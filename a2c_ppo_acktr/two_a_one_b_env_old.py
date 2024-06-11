import gym
from gym import spaces
import numpy as np

class TwoAOneBEnv(gym.Env):
    def __init__(self):
        # 定義動作空間（Action Space）和觀察空間（Observation Space）
        self.action_space = spaces.Discrete(4)  # 4個動作：A1, A2, B1, B2
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)  # 4個數字，表示遊戲狀態
        print("TwoAOneBEnv initialized")

        # 初始化遊戲狀態和數據庫
        self.game_state = [0, 0, 0, 0]  # 初始遊戲狀態：[A1, A2, B1, B2]
        self.target = [1, 2, 3, 4]  # 目標狀態，例如[1, 2, 3, 4]

    def step(self, action):
        # 執行動作並計算獎勵
        
        if action == 0:  # A1
            self.game_state[0] += 1
            if self.game_state[0] == 10:
                self.game_state[0] = 0
        elif action == 1:  # A2
            self.game_state[1] += 1
            if self.game_state[1] == 10:
                self.game_state[1] = 0
        elif action == 2:  # B1
            self.game_state[2] += 1
            if self.game_state[2] == 10:
                self.game_state[2] = 0
        elif action == 3:  # B2
            self.game_state[3] += 1
            if self.game_state[3] == 10:
                self.game_state[3] = 0

        print(f"Action taken: {action}")

        done = self.game_state == self.target  # 判斷是否達到目標狀態
        reward = 1 if done else 0  # 達到目標狀態時給予獎勵

        return np.array(self.game_state), reward, done, {}

    def reset(self):
        # 重置遊戲狀態
        self.game_state = [0, 0, 0, 0]
        print("Environment reset")
        return np.array(self.game_state)

    def render(self, mode='human'):
        # 繪製遊戲狀態
        print(f"Current game state: {self.game_state}")

    def close(self):
        # 清理環境
        pass
