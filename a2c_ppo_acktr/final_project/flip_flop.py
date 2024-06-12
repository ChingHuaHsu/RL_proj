import gym
from gym import spaces
import gymnasium as gym
import numpy as np
import random
import time
import itertools
from .mbffg import MBFFG as MBFFG
from pprint import pprint
import matplotlib.pyplot as plt


class Flip_Flop(gym.Env):
    def __init__(self):
        # 定義晶片資訊
        self.input_path = "./a2c_ppo_acktr/final_project/new_c1.txt"
        self.mbffg = MBFFG(self.input_path)
        self.ori_score = self.mbffg.scoring()
        self.mb_info = self.mbffg.get_ffs()
        self.mb_lib = self.mbffg.get_library()
        self.mbffg.transfer_graph_to_setting(extension="svg")
        # 定義 Action Space
        self.ff1_count = len([inst for inst in self.mb_info if inst.lib_name == 'ff1'])
        self.ff2_count = len([inst for inst in self.mb_info if inst.lib_name == 'ff2'])
        self.choices = 2
        self.ff1_actions = list(itertools.combinations(range(self.ff1_count), 2))
        self.ff2_actions = list(itertools.combinations(range(self.ff2_count), 2))
        self.actions = list(itertools.product(self.ff1_actions)) + list(itertools.product(self.ff2_actions))
        self.action_space = spaces.Discrete(len(self.actions)*self.choices)
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
        self.ff1_chart_org = self.ff1_chart
        self.ff2_chart_org = self.ff2_chart
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
        global used_actions
        used_actions = []
        self.record = []
        
        print("ENV finish initialize!")

    def step(self, action):
        combo=''
        reward=''
        info = {"episode": {}}


        if action < len(self.ff1_actions)*2 : # 是取 ff1 還是 ff2 
            act = action//len(self.ff1_actions)
            if act == 0: # 不合
                #抓flip_flop1, flip_flop2
                combo = self.ff1_actions[int(action)]
                flip_flop1, flip_flop2 = combo
                # print(f'combo= {combo}, flip_flop1= {flip_flop1}, flip_flop2= {flip_flop2}')
                #判斷有沒有抓過了
                if self.ff1_chart_org[flip_flop1]['name'] in self.taken_flip_flop or self.ff1_chart_org[flip_flop2]['name']in self.taken_flip_flop:
                    reward = 0
                    print("抓到一樣的")
                    print(f'combo= {combo}, action= {action}, reward= {reward}, len of taken_ff= {len(self.taken_flip_flop)}')
                    return self.observation, reward, self.done, {"episode": {"r": reward}}
                else:
                        #更新taken_flip_flop
                        self.taken_flip_flop.append(self.ff1_chart_org[flip_flop1]['name'])
                        self.taken_flip_flop.append(self.ff1_chart_org[flip_flop2]['name'])
                        self.used_action_update(flip_flop1,flip_flop2,0)
                        print("抓ff1,不合")
                        print(self.ff1_chart_org[flip_flop1]['name'])
                        print(self.ff1_chart_org[flip_flop2]['name'])

            elif act==1: # 合
                #action值要更改
                action = action - len(self.ff1_actions)
                #抓flip_flop1, flip_flop2
                combo = self.ff1_actions[int(action)]
                flip_flop1, flip_flop2 = combo
                if self.ff1_chart_org[flip_flop1]["name"] in self.taken_flip_flop or self.ff1_chart_org[flip_flop2]["name"]in self.taken_flip_flop:
                    reward = 0
                    print("抓到一樣的")
                    print(f'combo= {combo}, action= {action}, reward= {reward}, len of taken_ff= {len(self.taken_flip_flop)}')
                    return self.observation, reward, self.done, {"episode": {"r": reward}}
                else:
                        #更新taken_flip_flop
                        self.taken_flip_flop.append(self.ff1_chart_org[flip_flop1]["name"])
                        self.taken_flip_flop.append(self.ff1_chart_org[flip_flop2]["name"])
                        #合flip_flop1, flip_flop2
                        self.mbffg.merge_ff(f'{self.ff1_chart_org[flip_flop1]["name"]},{self.ff1_chart_org[flip_flop2]["name"]}', 'ff2')
                        #更新資料 - observation、mb_info、ff1_chart、ff2_chart
                        self.used_action_update(flip_flop1,flip_flop2,0)
                        self.mb_info = self.mbffg.get_ffs()
                        self.ff1_chart = self.convert_to_dict(self.mb_info,filter_lib_name="ff1")
                        self.ff2_chart = self.convert_to_dict(self.mb_info,filter_lib_name="ff2")
                        self.observation = self.state_to_observation()
                        print("抓ff1,合")
                        print(self.ff1_chart_org[flip_flop1]['name'])
                        print(self.ff1_chart_org[flip_flop2]['name'])

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
                if self.ff2_chart_org[flip_flop1]["name"] in self.taken_flip_flop or self.ff2_chart_org[flip_flop2]["name"]in self.taken_flip_flop:
                    reward = 0
                    print("抓到一樣的")
                    print(f'combo= {combo}, action= {action}, reward= {reward}, len of taken_ff= {len(self.taken_flip_flop)}')
                else:
                        #更新taken_flip_flop
                        self.taken_flip_flop.append(self.ff2_chart_org[flip_flop1]["name"])
                        self.taken_flip_flop.append(self.ff2_chart_org[flip_flop2]["name"])
                        self.used_action_update(flip_flop1,flip_flop2,1)
                        print("抓ff2,不合")
                        print(self.ff2_chart_org[flip_flop1]['name'])
                        print(self.ff2_chart_org[flip_flop2]['name'])
            elif act==1:
                #action值要更改
                action = action - len(self.ff1_actions)*2 - len(self.ff2_actions)
                #抓flip_flop1, flip_flop2
                combo = self.ff2_actions[int(action)]
                flip_flop1, flip_flop2 = combo
                if self.ff2_chart_org[flip_flop1]["name"] in self.taken_flip_flop or self.ff2_chart_org[flip_flop2]["name"]in self.taken_flip_flop:
                    reward = 0
                    print("抓到一樣的")
                    print(f'combo= {combo}, action= {action}, reward= {reward}, len of taken_ff= {len(self.taken_flip_flop)}')
                    return self.observation, reward, self.done, {"episode": {"r": reward}}
                else:
                        #更新taken_flip_flop
                        self.taken_flip_flop.append(self.ff2_chart_org[flip_flop1]["name"])
                        self.taken_flip_flop.append(self.ff2_chart_org[flip_flop2]["name"])
                        #合flip_flop1, flip_flop2
                        self.mbffg.merge_ff(f'{self.ff2_chart_org[flip_flop1]["name"]},{self.ff2_chart_org[flip_flop2]["name"]}', 'ff4')
                        #更新資料 - observation、mb_info、ff1_chart、ff2_chart
                        self.used_action_update(flip_flop1,flip_flop2,1)
                        self.mb_info = self.mbffg.get_ffs()
                        self.ff1_chart = self.convert_to_dict(self.mb_info,filter_lib_name="ff1")
                        self.ff2_chart = self.convert_to_dict(self.mb_info,filter_lib_name="ff2")
                        self.observation = self.state_to_observation()
                        print("抓ff2,合")
                        print(self.ff2_chart_org[flip_flop1]['name'])
                        print(self.ff2_chart_org[flip_flop2]['name'])
        #reward
        self.mbffg.legalization()
        final_score = self.mbffg.scoring()
        reward = self.ori_score - final_score
        self.record.append(final_score)
        # self.ori_score = final_score
        print(f'combo= {combo}, action= {action}, ori_score= {self.ori_score}, final_score= {final_score}, reward= {reward}, len of taken_ff= {len(self.taken_flip_flop)}')
        self.ori_score = final_score
        #
        #done
        if len(self.taken_flip_flop) == self.ff1_count + self.ff2_count:
            self.done = True 

        print(f"Info before assignment: {info}")
        info["episode"]["r"] = reward  # Make sure reward is defined before this line
        print(f"Info after assignment: {info}")
        
        return self.observation, reward, self.done, info, None
        
    def reset(self, seed=None):
        if self.done:
            plt.clf()
            plt.plot(self.record)
            plt.xlabel('Time')
            plt.ylabel('Final score')
            plt.title('Score trend')
            plt.grid(True)
            plt.savefig('final_score.png')
            self.mbffg.transfer_graph_to_setting(extension="svg")
        self.input_path = "./a2c_ppo_acktr/final_project/new_c1.txt"
        self.mbffg = MBFFG(self.input_path)
        self.ori_score = self.mbffg.scoring()
        self.mb_info = self.mbffg.get_ffs()
        self.max_flip_flops = 100
        self.num_attributes = 6
        self.ff1_chart = self.convert_to_dict(self.mb_info, filter_lib_name="ff1")
        self.ff2_chart = self.convert_to_dict(self.mb_info, filter_lib_name="ff2")
        self.observation = self.state_to_observation()
        self.taken_flip_flop = []
        self.record = []
        global used_actions
        used_actions = []
        self.done = False
        print("finish reset")
        time.sleep(3)
        return self.observation, {}  # 返回一个包含两个元素的元组
            
    def render(self, mode='human'):
        pass

    def close(self):
        pass
    
    def seed(self, seed=None):
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
    
    def used_action_update(self, flip_flop1, flip_flop2, switch):
        if switch == 0:
            for i in range (len(self.ff1_actions)):
                if flip_flop1 in self.ff1_actions[i] or flip_flop2 in self.ff1_actions[i]:
                    x = i + len(self.ff1_actions)
                    if i not in used_actions:
                        used_actions.append(i)
                    if x not in used_actions:
                        used_actions.append(x)
        elif switch == 1:
            for i in range (len(self.ff2_actions)):
                if flip_flop1 in self.ff2_actions[i] or flip_flop2 in self.ff2_actions[i]:
                    x = i + (len(self.ff1_actions)*2)
                    y = x + len(self.ff2_actions)
                    if x not in used_actions:
                        used_actions.append(x)
                    if y not in used_actions:
                        used_actions.append(y)
        return
    def return_action():
        return used_actions

