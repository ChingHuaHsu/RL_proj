import gym
from gym import spaces
import numpy as np
import random
from itertools import combinations
import time
from pathlib import Path
from datetime import datetime
import os

# swap_combinations generates all unique pairs (i, j) for i, j in range(4) and i < j
swap_combinations = list(combinations(range(4), 2))
str_result = ''
starttime = ''
penalty_A = -4
penalty_B = -0.2
penalty_all = -5

class OneATwoBEnv(gym.Env):
    def __init__(self):
        # 動作空間包含768種動作
        self.action_space = gym.spaces.Discrete(768)
        self.observation_space = gym.spaces.Box(low=0, high=9, shape=(6,), dtype=np.int32)
        
        self.game_state = np.array([0, 0, 0, 0, 0, 0], dtype=np.int32)
        self.target = np.array([1, 2, 3, 4], dtype=np.int32)
        
        # 建立 log
        # # Path("../log/").mkdir(parents=True, exist_ok=True)
        # date = f'{datetime.now().year}{datetime.now().month}{datetime.now().day}'
        # now = datetime.now().strftime("%H%M%S")
        # global starttime
        # starttime = f'{date}{now}'
        
        # Get the current date and time
        global starttime
        starttime = datetime.now().strftime("%Y%m%d%H%M%S")

    def step(self, action):
        prev_state = self.game_state.copy()
        
        # print(f"action= {action}")
        if action < 24:
            # 單純位置交換
            self._swap_positions(action)
        elif action < 144:
            # 單純數字修改
            self._modify_numbers(action - 24)
        else:
            # 位置交換且修改數字
            self._swap_and_modify(action - 144)
        
        while np.array_equal(self.game_state[:4], prev_state[:4]):
            if action < 24:
                self._swap_positions(random.randint(0, 23))
            elif action < 144:
                self._modify_numbers(random.randint(24, 143) - 24)
            else:
                self._swap_and_modify(random.randint(144, 767) - 144)
        
        done = np.array_equal(self.game_state[:4], self.target)
        reward = self._calculate_reward()
        
        tmp = f"reward= {reward} self.game_state= {self.game_state} A= {self._calculate_AB()[0]} B= {self._calculate_AB()[1]}\n"
        global str_result
        str_result += tmp
        # print(f"reward= {reward} self.game_state= {self.game_state} A= {self._calculate_AB()[0]} B= {self._calculate_AB()[1]}")
        return np.array(self.game_state, dtype=np.int32), reward, done, {"A": self._calculate_AB()[0], "B": self._calculate_AB()[1], "episode": {"r": reward}}

    def _swap_positions(self, action):

        # Calculate the number of swaps (1 to 4)
        num_swaps = int(((action // len(swap_combinations)) % 4) + 1)

        # Calculate the specific combination index (0 to len(swap_combinations)-1)
        swap_indices = int(action % len(swap_combinations))

        # Determine the positions to swap
        indices_to_swap = swap_combinations[swap_indices]

        # Swap the positions num_swaps times
        for _ in range(num_swaps):
            i, j = indices_to_swap
            self.game_state[i], self.game_state[j] = self.game_state[j], self.game_state[i]

    def _modify_numbers(self, action):
        num_modifications = int((action // 30) + 1)
        modification_index = action % 30
        
        positions = list(range(4))
        values = list(range(10))
        
        for _ in range(num_modifications):
            pos = modification_index % 4
            val = modification_index % 10
            if self.game_state[pos] != val:
                self.game_state[pos] = val
                modification_index //= 10

    def _swap_and_modify(self, action):
        # Calculate the number of swaps (1 to 4)
        num_swaps = int(((action // len(swap_combinations)) % 4) + 1)
        
        # Calculate the specific combination index (0 to len(swap_combinations)-1)
        swap_indices = int((action // 10) % len(swap_combinations))
        indices_to_swap = swap_combinations[swap_indices]
        
        # Swap the positions num_swaps times
        for _ in range(num_swaps):
            i, j = indices_to_swap
            self.game_state[i], self.game_state[j] = self.game_state[j], self.game_state[i]

        # Calculate the number of modifications (1 to 4)
        num_modifications = int((action % 4) + 1)

        # Modify num_modifications positions
        modified_positions = random.sample(range(4), num_modifications)
        for position in modified_positions:
            new_value = random.randint(0, 9)
            while new_value == self.game_state[position]:  # Ensure the new value is different
                new_value = random.randint(0, 9)
            self.game_state[position] = new_value

    def reset(self):
        global str_result
        # print(str_result)
        
        # Get the current date and time
        date = f'{datetime.now().year}{datetime.now().month:02}{datetime.now().day:02}'
        now = datetime.now().strftime("%H:%M:%S")
        starttime = datetime.now().strftime("%Y%m%d%H%M%S")

        global penalty_A, penalty_B, penalty_all
        folder_path = f'log/{date}_{penalty_A}_{penalty_B}_{penalty_all}/'
        print(f'time: {now}\nsave at {folder_path}')
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, f'{starttime}.txt')  # Construct the file path
        print(file_path)
        # if not folder_path.exists():
        #     folder_path.mkdir()
        if (str_result != ''):
            with open(file_path, 'w') as text_file:
                text_file.write(str_result)
                text_file.write('\n\n\n')

        # 重置游戏状态
        if np.array_equal(self.game_state[:4], self.target):
            out = "guess:", self.game_state[:4] , "A:" , self.game_state[4] , "B" , self.game_state[5]
            print(out)
            f'{out}'
            self.target = np.array([random.randint(0, 9) for _ in range(4)], dtype=np.int32)  # 随机生成新的目标状态
            print(self.target)
            f'{self.target}'
            # time.sleep(5)
        print("game reset")
        
        str_result = f'{date} {now}\n'
        self.game_state = np.array([0, 0, 0, 0 ,0 ,0], dtype=np.int32)
        self.done = False
        return np.array(self.game_state, dtype=np.int32)

    def render(self, mode='human'):
        print(f"Current game state: {self.game_state}")

    def close(self):
        pass

    def _calculate_AB(self):
        
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
        global penalty_A, penalty_B, penalty_all
        if A == 0 and B == 0:
            return penalty_all
            # return -0.5
        else:
            return penalty_A / (2 ** A) + penalty_B / (2 ** B)
            # return A * 1 + B * 0.5
