import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mindspore_mutation.generate_models.run_random_ms import run_random_ms
from mindspore_mutation.generate_models.run_mcmc_ms import run_mcmc_ms
from mindspore_mutation.generate_models.run_log_ms import run_log_ms
from mindspore_mutation.generate_models.run_q_ms import run_q_ms

import time
import argparse

import sys
from mindspore_mutation.cargo import net_cargo
import mindspore
mindspore.set_context(mode=mindspore.PYNATIVE_MODE, device_target="GPU")

class Logger(object):
    def __init__(self, filename="print_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)  
        self.log.write(message)       

    def flush(self):
        self.terminal.flush()
        self.log.flush()

if __name__ == '__main__':

    current_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    seed_model = "openpose" 
    log_txt = f"/data1/czx/SemTest_master/mindspore_mutation/results/{seed_model}/{current_time}/log.txt"
    os.makedirs(os.path.dirname(log_txt), exist_ok=True)
    sys.stdout = Logger(log_txt)
    log_path = "/data1/czx/SemTest_master/mindspore_mutation/results/openpose/2024_10_24_17_03_16/TORCH_LOG_DICT_GPU.json"
    mutate_times = 100
    
    num_samples = 1 
    data_x_path=''
    data_y_path=''
    path_flag=False

    sstart=time.time()
    run_option = 2 
    if run_option == 0:
        run_random_ms(seed_model, mutate_times,num_samples)
        
    elif run_option == 1:
        run_mcmc_ms(seed_model, mutate_times,num_samples)
        
    elif run_option == 2:
        run_q_ms(seed_model, mutate_times,num_samples,1)
        
    elif run_option == 3: 
        run_log_ms(seed_model, mutate_times, num_samples, 1, log_path)

    eend = time.time()
    print('~~~~~~~~~~~~~~~~~~~~~~~~')
    print(eend-sstart)
