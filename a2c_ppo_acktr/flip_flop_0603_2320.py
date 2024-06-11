import gym
from gym import spaces
import numpy as np
import random
import time
import itertools
from .final_project import mbffg
from .final_project.mbffg import MBFFG as MBFFG
from pprint import pprint


class Flip_Flop(gym.Env):
    def __init__(self):
        # 定義晶片資訊
        self.input_path = "./a2c_ppo_acktr/final_project/new_c1.txt"
        self.mbffg = MBFFG(self.input_path)
        self.ori_score = self.mbffg.scoring()
        self.mb_info = self.mbffg.get_ffs()
        self.mb_lib = self.mbffg.get_library()
        # 定義 Action Space
        self.ff1_count = len([inst for inst in self.mb_info if inst.lib_name == 'ff1'])
        self.ff2_count = len([inst for inst in self.mb_info if inst.lib_name == 'ff2'])
        self.choices = 2
        self.ff1_actions = list(itertools.combinations(range(self.ff1_count), 2))
        self.ff2_actions = list(itertools.combinations(range(self.ff2_count), 2))
        print(f'ff1= {self.ff1_actions}, ff2= {self.ff2_actions}\n # of ff1= {len(list((itertools.combinations(range(self.ff1_count), 2))))}, # of ff2= {len(list((itertools.combinations(range(self.ff2_count), 2))))}')
        # time.sleep(10)
        self.actions = list(itertools.product(self.ff1_actions)) + list(itertools.product(self.ff2_actions))
        self.action_space = spaces.Discrete(len(self.actions)*2)
        # 定義最大 flip-flop 數量
        self.max_flip_flops = 100
        # 每個 flip-flop 的屬性數量
        self.num_attributes = 6
        # 定義 Observation Space
        self.observation_space = gym.spaces.Box(
            low=0,
            high=np.inf,
            # shape=(self.max_flip_flops * self.num_attributes,),
            shape=(1200,),
            dtype=np.float32
        )
        # 定義晶片資訊表格
        self.ff1_chart = self.convert_to_dict(self.mb_info,filter_lib_name="ff1")
        self.ff2_chart = self.convert_to_dict(self.mb_info,filter_lib_name="ff2")
        print("----------------------------------------------------------------")
        print("ff1_chart:")
        pprint(self.ff1_chart)
        print("----------------------------------------------------------------")
        print("ff2_chart:")
        pprint(self.ff2_chart)
        print("----------------------------------------------------------------")
        # 初始化
        self.observation = self.state_to_observation()
        self.done = False
        # 已使用flip-flop初始化
        self.taken_flip_flop = [] 
        
        print("ENV finish initialize!")

    def step(self, action):
        combo=''
        reward=''
        
        if action < len(self.ff1_actions)*2 :
            act = action//len(self.ff1_actions)
            if act == 0:
                #抓flip_flop1, flip_flop2
                combo = self.ff1_actions[int(action)]
                flip_flop1, flip_flop2 = combo
                # print(f'combo= {combo}, flip_flop1= {flip_flop1}, flip_flop2= {flip_flop2}')
                #判斷有沒有抓過了
                # if flip_flop1['name'] in self.taken_flip_flop or flip_flop2['name']in self.taken_flip_flop:
                # if self.ff1_chart[flip_flop1]['name'] in self.taken_flip_flop or self.ff1_chart[flip_flop2]['name']in self.taken_flip_flop:
                #     reward = -1
                # else:
                if flip_flop1 < len(self.ff1_chart) and flip_flop2 < len(self.ff1_chart):
                    if self.ff1_chart[flip_flop1]['name'] and self.ff1_chart[flip_flop2]['name']:
                        #更新taken_flip_flop
                        self.taken_flip_flop.append(self.ff1_chart[flip_flop1]['name'])
                        self.taken_flip_flop.append(self.ff1_chart[flip_flop2]['name'])
                        # self.taken_flip_flop.append(flip_flop1['name'])
                        # self.taken_flip_flop.append(flip_flop2['name'])
                        print("抓ff2,合")
                    else:
                        reward = -300
                        print("抓到一樣的")
                else:
                    reward=-300
                    print("抓到一樣的")
            elif act==1:
                #action值要更改
                action = action - len(self.ff1_actions)
                #抓flip_flop1, flip_flop2
                combo = self.ff1_actions[int(action)]
                flip_flop1, flip_flop2 = combo
                # if self.ff1_chart[flip_flop1]["name"] in self.taken_flip_flop or self.ff1_chart[flip_flop2]["name"]in self.taken_flip_flop:
                #     reward = -1
                # else:
                if flip_flop1 < len(self.ff1_chart) and flip_flop2 < len(self.ff1_chart):
                    if self.ff1_chart[flip_flop1]['name'] and self.ff1_chart[flip_flop2]['name']:
                        #更新taken_flip_flop
                        self.taken_flip_flop.append(self.ff1_chart[flip_flop1]["name"])
                        self.taken_flip_flop.append(self.ff1_chart[flip_flop2]["name"])
                        #合flip_flop1, flip_flop2
                        # print(f'self.ff1_chart[flip_flop1]["name"]= {self.ff1_chart[flip_flop1]["name"]}, self.ff1_chart[flip_flop2]["name"]= {self.ff1_chart[flip_flop2]["name"]}')
                        self.mbffg.merge_ff(f'{self.ff1_chart[flip_flop1]["name"]},{self.ff1_chart[flip_flop2]["name"]}', 'ff2')
                        #更新資料 - observation、mb_info、ff1_chart、ff2_chart
                        self.mb_info = self.mbffg.get_ffs()
                        self.ff1_chart = self.convert_to_dict(self.mb_info,filter_lib_name="ff1")
                        self.ff2_chart = self.convert_to_dict(self.mb_info,filter_lib_name="ff2")
                        self.observation = self.state_to_observation()
                        print("抓ff1,合")
                    else:
                        reward = -300
                        print("抓到一樣的")
                else:
                    reward = -300
                    print("抓到一樣的")
        else:
            act = (action - len(self.ff1_actions)*2)//len(self.ff2_actions)
            # act = (action - len(self.ff1_actions)*2)//len(self.ff1_actions)
            if act == 0:
                #action值要更改
                action = action - len(self.ff1_actions)*2
                #抓flip_flop1, flip_flop2
                combo = self.ff2_actions[int(action)]
                flip_flop1, flip_flop2 = combo
                #判斷有沒有抓過了
                # if self.ff2_chart[flip_flop1]["name"] in self.taken_flip_flop or self.ff2_chart[flip_flop1]["name"]in self.taken_flip_flop:
                # # if flip_flop1["name"] in self.taken_flip_flop or flip_flop2["name"]in self.taken_flip_flop:
                #     reward = -1
                # else:
                if flip_flop1 < len(self.ff2_chart) and flip_flop2 < len(self.ff2_chart):
                # if flip_flop1 < len(self.ff1_chart) and flip_flop2 < len(self.ff1_chart):
                    if self.ff2_chart[flip_flop1]['name'] and self.ff2_chart[flip_flop2]['name']:
                        #更新taken_flip_flop
                        self.taken_flip_flop.append(self.ff2_chart[flip_flop1]["name"])
                        self.taken_flip_flop.append(self.ff2_chart[flip_flop2]["name"])
                        print("抓ff2,不合")
                    else:
                        reward = -300
                        print("抓到一樣的")
                else:
                    reward = -300
                    print("抓到一樣的")
            elif act==1:
                #action值要更改
                action = action - len(self.ff1_actions)*2 - len(self.ff2_actions)
                #抓flip_flop1, flip_flop2
                combo = self.ff2_actions[int(action)]
                flip_flop1, flip_flop2 = combo
                if flip_flop1 < len(self.ff2_chart) and flip_flop2 < len(self.ff2_chart):
                # if flip_flop1 < len(self.ff1_chart) and flip_flop2 < len(self.ff1_chart):
                    if self.ff2_chart[flip_flop1]['name'] and self.ff2_chart[flip_flop2]['name']:
                        #更新taken_flip_flop
                        self.taken_flip_flop.append(self.ff2_chart[flip_flop1]["name"])
                        self.taken_flip_flop.append(self.ff2_chart[flip_flop2]["name"])
                        #合flip_flop1, flip_flop2
                        self.mbffg.merge_ff(f'{self.ff2_chart[flip_flop1]["name"]},{self.ff2_chart[flip_flop2]["name"]}', 'ff4')
                        print(f'{self.ff2_chart[flip_flop1]["name"]}, {self.ff2_chart[flip_flop2]["name"]}', 'ff4')
                        #更新資料 - observation、mb_info、ff1_chart、ff2_chart
                        self.mb_info = self.mbffg.get_ffs()
                        self.ff1_chart = self.convert_to_dict(self.mb_info,filter_lib_name="ff1")
                        self.ff2_chart = self.convert_to_dict(self.mb_info,filter_lib_name="ff2")
                        self.observation = self.state_to_observation()
                        print("抓ff2,合")
                    else:
                        reward = -300
                        print("抓到一樣的")
                else:
                    reward = -300
                    print("抓到一樣的")
        #reward
        self.mbffg.legalization()
        final_score = self.mbffg.scoring()
        reward = final_score - self.ori_score
        # self.ori_score = final_score
        
        print(f'combo= {combo}, action= {action}, ori_score= {self.ori_score}, final_score= {final_score}, reward= {reward}, cnt of ff1= {self.ff1_count}, cnt of ff2= {self.ff2_count}, len of taken_ff= {len(self.taken_flip_flop)}')
        self.ori_score = final_score
        #done
        if len(self.taken_flip_flop) == self.ff1_count + self.ff2_count:
            self.done = True 
        return self.observation, reward, self.done, {"episode": {"r": reward}}
    
    def reset(self):
        self.input_path = "./a2c_ppo_acktr/final_project/new_c1.txt"
        self.mbffg = MBFFG(self.input_path)
        self.ori_score = self.mbffg.scoring()
        self.mb_info = self.mbffg.get_ffs()
        self.max_flip_flops = 100
        self.num_attributes = 6
        self.ff1_chart = self.convert_to_dict(self.mb_info,filter_lib_name="ff1")
        self.ff2_chart = self.convert_to_dict(self.mb_info,filter_lib_name="ff2")
        self.observation = self.state_to_observation()
        self.taken_flip_flop = [] 
        self.done = False
        print("finish reset")
        return self.observation
        
    def render(self, mode='human'):
        pass

    def close(self):
        pass
    
    def convert_to_dict(self, inst_list, filter_lib_name=None):
        filtered_list = [inst for inst in inst_list if filter_lib_name in inst.lib_name] if filter_lib_name else inst_list
        return [{'name': inst.name, 'index': idx, 'x': inst.x, 'y': inst.y, 'width': self.mb_lib[inst.lib_name].width, 'height': self.mb_lib[inst.lib_name].height} for idx, inst in enumerate(filtered_list)]
    
    def state_to_observation(self):
    # Convert chart to arrays
        def convert_chart_to_array(chart, max_length):
            array = np.zeros((max_length, self.num_attributes), dtype=np.float32)
            for i, entry in enumerate(chart):
                if i >= max_length:
                    break
                array[i, 0] = entry['index']
                array[i, 1] = entry['x']
                array[i, 2] = entry['y']
                array[i, 3] = entry['width']
                array[i, 4] = entry['height']
            return array

        ff1_array = convert_chart_to_array(self.ff1_chart, self.max_flip_flops)
        ff2_array = convert_chart_to_array(self.ff2_chart, self.max_flip_flops)

        # Concatenate ff1 and ff2 arrays into a single state array
        state = np.concatenate((ff1_array.flatten(), ff2_array.flatten()))

        return state

