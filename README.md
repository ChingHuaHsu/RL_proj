以下檔案需要修改
在虛擬環境內
miniconda3/envs/p37/lib/python3.7/site-packages/stable_baselines3/common/vec_env



1. dummy_vec_env.py

  step_wait() 最後新增：

    truncated = [False] * self.num_envs  # 或者根據你的環境邏輯設置
        
    return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), truncated, deepcopy(self.buf_infos))



2. vec_normalize.py

  step_wait() 開頭與最後：

    #開頭
    obs, rewards, dones, truncated, infos = self.venv.step_wait() 

    #最後
    return obs, rewards, dones, truncated, infos
   
