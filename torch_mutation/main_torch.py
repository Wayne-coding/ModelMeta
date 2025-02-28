import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch_mutation.generate_models.run_random_torch import run_random_torch
from torch_mutation.generate_models.run_mcmc_torch import run_mcmc_torch
from torch_mutation.generate_models.run_q_torch import run_q_torch
from torch_mutation.generate_models.run_log_torch import run_log_torch
import time

if __name__ == '__main__':
    seed_model ="vgg16" # 选择模型
    log_path ="./torch_mutation/results/vgg11/2024_10_03_15_47_24/TORCH_LOG_DICT_cpu.json" #变异日志
    mutate_times = 20 # 变异次数
    num_samples = 1  # 随机选择几个数据
    data_x_path='' #输入的x
    data_y_path='' #输入的y，这两个用于缺陷检测：detect_bugs.py
    path_flag=False #用于缺陷检测：detect_bugs.py numpy数据是否用选定的

    run_option = 2 # 0代表随机选择，1代表通过MCMC选择，2代表Q网络方法，3代表根据日志对模型detect_bugs
    if run_option == 0: # 随机选择
        run_random_torch(seed_model, mutate_times,num_samples)
    elif run_option == 1: # MCMC选择
        run_mcmc_torch(seed_model, mutate_times,num_samples)
    elif run_option == 2: # Q网络方法
        run_q_torch(seed_model, mutate_times,num_samples)
    elif run_option == 3:  # 根据日志对模型detect_bugs
        run_log_torch(seed_model, mutate_times,log_path, num_samples,data_x_path,data_y_path,path_flag)

