import torch
import numpy as np
from a2c_ppo_acktr.model import Policy  # 導入模型類別
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
# from evaluation import evaluate

def evaluate_loaded_model(model_file, env_name, seed, num_processes, eval_log_dir, device):
    # 加載訓練好的模型
    actor_critic, obs_rms = torch.load(model_file, map_location=device)
    actor_critic.to(device)
    
    # 創建評估環境
    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                              None, eval_log_dir, device, True)

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.obs_rms = obs_rms

    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    while len(eval_episode_rewards) < 10:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)

        # Observe reward and next obs
        obs, _, done, infos = eval_envs.step(action)

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    eval_envs.close()

    print("Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))

if __name__ == "__main__":
    model_file = '/home/mchiou/miniconda3/a_proj/pytorch-a2c-ppo-acktr-gail/trained_models/ppo/Flip_Flop.pt' # 替換為你的模型檔案路徑
    env_name = "Flip_Flop"  # 替換為你的環境名稱
    seed = 1  # 隨機種子
    num_processes = 1  # 處理數量
    eval_log_dir = 'logs/'  # 評估日誌目錄
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自動選擇設備

    evaluate_loaded_model(model_file, env_name, seed, num_processes, eval_log_dir, device)
