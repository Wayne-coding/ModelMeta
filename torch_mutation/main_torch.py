import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch_mutation.generate_models.run_random_torch import run_random_torch
from torch_mutation.generate_models.run_mcmc_torch import run_mcmc_torch
from torch_mutation.generate_models.run_q_torch import run_q_torch
from torch_mutation.generate_models.run_log_torch import run_log_torch
import time

if __name__ == '__main__':
    seed_model ="vgg16" 
    log_path ="./torch_mutation/results/vgg11/2024_10_03_15_47_24/TORCH_LOG_DICT_cpu.json" 
    mutate_times = 20 
    num_samples = 1  
    data_x_path='' 
    data_y_path='' 
    path_flag=False 

    run_option = 2 
    if run_option == 0: 
        run_random_torch(seed_model, mutate_times,num_samples)
    elif run_option == 1: 
        run_mcmc_torch(seed_model, mutate_times,num_samples)
    elif run_option == 2: 
        run_q_torch(seed_model, mutate_times,num_samples)
    elif run_option == 3:  
        run_log_torch(seed_model, mutate_times,log_path, num_samples,data_x_path,data_y_path,path_flag)

