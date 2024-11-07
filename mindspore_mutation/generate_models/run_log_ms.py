import pandas as pd
import mindspore
import mindspore as ms
import collections
import uuid
import copy
import datetime
import json
import os
import platform
import random
import sys
import time
from mindspore import JitConfig
import mindspore.context as context
from mindspore import export, load_checkpoint, load_param_into_net
from mindspore.rewrite import ScopedValue, NodeType
from mindspore.rewrite.node import Node, NodeManager
from numpy import ndarray
from openpyxl import Workbook
import mindspore.numpy as mnp
from mindspore import Tensor
from infoplus.MindSporeInfoPlus import mindsporeinfoplus
import torch
import torch.optim as optim
from mindspore import Tensor
from mindspore.rewrite import SymbolTree

import pickle
from mindspore_mutation.cargo import *
import torch.distributions as dist
import copy
import time
import json
import torch.fx as fx
from mindspore_mutation.MR_structure import *
from mindspore_mutation.cargo import match_rule,reflect_name,MCMC,compute_gpu_cpu
from mindspore_mutation.api_mutation import api_mutation
from mindspore_mutation.calculate_coverage import model2cov,find_layer_type
from mindspore_mutation.cargo import select_places,max_seed_model_api_times
import psutil
import sys
from openpyxl import Workbook
import os
import numpy as np
from mindspore import Tensor
from mindspore import dtype as mstype
import mindspore.context as context
from mindspore import save_checkpoint
import torch_mutation.config as pt_config
import mindspore_mutation.config as ms_config

from mindspore_mutation.handel_shape import handle_format
from mindspore_mutation import metrics
import gc


ms_device=ms_config.device
pt_device=pt_config.device
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="GPU")

MR_structure_name_list = ['UOC', 'PIOC', 'ABSOC_A', 'ABSOC_B']
MR_structures_map = {"UOC": UOC, "PIOC": PIOC, "ABSOC_A": ABSOC_A, "ABSOC_B": ABSOC_B} 
nlp_cargo = ["LSTM","FastText", "TextCNN", "SentimentNet", "GPT"]



deadcode_name_list=['Dense', 'SELayer', 'DenseLayer', 'Inception_A', 'PWDWPW_ResidualBlock', 'ResidualBlock', 'DropPath']



def run_log_ms(seed_model, mutate_times, num_samples, mr_index, log_path):
    
    localtime = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    log_dict = {}
    dtypes = [mindspore.float32] if seed_model not in nlp_cargo else [mindspore.int32]

    with open(log_path, "r") as json_file:
        log_dict = json.load(json_file)

    
    if isinstance(datasets_path_cargo[seed_model],list):
        data_0 = np.load(datasets_path_cargo[seed_model][0])
        data_1 = np.load(datasets_path_cargo[seed_model][1])
        samples_0 = np.random.choice(data_0.shape[0], num_samples, replace=False)
        samples_data_0 = data_0[samples_0] 

        samples_1 = np.random.choice(data_1.shape[0], num_samples, replace=False)
        samples_data_1 = data_1[samples_1] 
        data_selected_0 = Tensor(samples_data_0, dtype=mstype.float32 if seed_model in nlp_cargo else mstype.int32)
        data_selected_1 = Tensor(samples_data_1, dtype=mstype.float32 if seed_model in nlp_cargo else mstype.int32)
        data_selected = (data_selected_0, data_selected_1)
        data_npy = [data_selected_0.asnumpy(), data_selected_1.asnumpy()]

        npy_path = os.path.join("results", seed_model, str(localtime), 'data0_npy.npy')
        os.makedirs(os.path.dirname(npy_path), exist_ok=True)
        np.save(npy_path, data_npy[0])
        npy_path = os.path.join("results", seed_model, str(localtime), 'data1_npy.npy')
        os.makedirs(os.path.dirname(npy_path), exist_ok=True)
        np.save(npy_path, data_npy[1])
    else:
        data = np.load(datasets_path_cargo[seed_model])
        samples = np.random.choice(data.shape[0], num_samples, replace=False)
        samples_data = data[samples] 
        data_selected = Tensor(samples_data, dtype=mstype.int32 if seed_model in nlp_cargo else mstype.float32)
        data_npy = data_selected.asnumpy()


        npy_path = os.path.join("results", seed_model, str(localtime), 'data0_npy.npy')
        os.makedirs(os.path.dirname(npy_path), exist_ok=True)
        np.save(npy_path, data_npy)

    seed_model_net = get_model(seed_model)
    new_net = copy.deepcopy(seed_model_net)
    stree = SymbolTree.create(new_net)
    metrics_dict = dict()
    option_layers = []
    for name, child in new_net.cells_and_names():
        if not has_child_node(new_net, name) and not name == '' and not 'deadcode' in str(type(child)):
            if name.split("_")[0] not in MR_structure_name_list:
                option_layers.append((name, child, name, type(child)))
    original_outputs = handle_format(seed_model_net(data_selected))
    new_outputs = original_outputs
    select_d_name = seed_model
    D = {seed_model: stree} 
    O = {seed_model: original_outputs} 
    N = {seed_model: seed_model_net} 
    R = {0:[0.0001, seed_model]} 
    MR_structure_selected_nums = {k: 0 for k in MR_structure_name_list}  
    seed_model_api_times=0 

    option_index = []
    
    with open('/data1/czx/SemTest_master/mindspore_mutation/results/TextCNN/example.txt', 'a', encoding='utf-8') as file:
        file.write('text_to_append' + "\n")
    file.close()

    for n in range(mutate_times):
        print('-----------------------total_Mutate_time:%d start!-----------------------' % n)
        start_time=time.time()
        
        if "Success" in log_dict[str(n)]['state']: 
            log_dict[n] = {}
            log_dict[n]['d_name'] = select_d_name
            old_d_name=select_d_name 

            
            selected_deadcode_name =log_dict[str(n)]['select_deadcode']

            
            selected_MR_structure_name = log_dict[str(n)]['selected_MR_structure']

            
            d_new_name = log_dict[str(n)]['d_new_name']

            
            api_mutation_type = log_dict[str(n)]['api_mutation_type(seed_model or deadcode)']
            

            
            nodedict = collections.OrderedDict()  
            hash_table = collections.defaultdict(int)  
            flag, nodedict = scan_node(stree, hash_table, nodedict)
            node_list=list(nodedict.values())
            node_list = []
            for k, v in nodedict.items():
                node_list.append(k)
            length=len(nodedict)
            print('length', length)
            print("mutate_type:", selected_MR_structure_name, ";  op_type:", selected_deadcode_name, ';  api_mutation_type:', api_mutation_type, flush=True)

            
            subs_place, dep_places = log_dict[str(n)]['subs_place'], log_dict[str(n)]['dep_places']
            a = node_list[dep_places[-1]]
            b = node_list[dep_places[-2]]
            c = node_list[dep_places[-3]]
            d = node_list[dep_places[-4]]
            aa = mindspore.rewrite.api.node.Node(a)
            bb = mindspore.rewrite.api.node.Node(b)
            cc = mindspore.rewrite.api.node.Node(c)
            dd = mindspore.rewrite.api.node.Node(d)
            add_module = MR_structures_map[selected_MR_structure_name](selected_deadcode_name, api_mutation_type, log_dict, n, LOG_FLAG=True)

            seat = 0
            if selected_MR_structure_name == "PIOC" and selected_deadcode_name in ["Dense", "Conv", "SELayer", "DenseLayer", "Inception_A",
                                                "PWDWPW_ResidualBlock", "ResidualBlock", "DropPath"]:

                tree = cc.get_symbol_tree()  

                position = tree.after(cc)  
                next_node = cc.get_users()[0]  
                if len(next_node.get_args()) > 1:
                    for idx, arg in enumerate(next_node.get_args()):
                        if arg == cc.get_targets()[0]:
                            seat = idx  
                            break
                        
                new_node = mindspore.rewrite.api.node.Node.create_call_cell(add_module,
                                                                            
                                                                            targets=[stree.unique_name("x")],
                                                                            name="{}_{}".format(selected_MR_structure_name,MR_structure_selected_nums[selected_MR_structure_name]),  
                                                                            args=ScopedValue.create_name_values(["aa", "bb", "cc"]))
                new_node.set_arg_by_node(0, aa) 
                new_node.set_arg_by_node(1, bb)
                new_node.set_arg_by_node(2, cc)
            else:  
                tree = dd.get_symbol_tree()
                position = tree.after(dd)
                next_node = dd.get_users()[0]
                if len(next_node.get_args()) > 1:
                    for idx, arg in enumerate(next_node.get_args()):
                        if arg == dd.get_targets()[0]:
                            seat = idx
                            break
                new_node = mindspore.rewrite.api.node.Node.create_call_cell(add_module,
                                                                            
                                                                            targets=[stree.unique_name("x")],
                                                                            name="{}_{}".format(selected_MR_structure_name,MR_structure_selected_nums[selected_MR_structure_name]),
                                                                            args=ScopedValue.create_name_values(["aa", "bb", "cc","dd"]))
                new_node.set_arg_by_node(0, aa)
                new_node.set_arg_by_node(1, bb)
                if selected_MR_structure_name == "UOC":
                    new_node.set_arg_by_node(2, cc)
                    
                    new_node.set_arg_by_node(3, dd)
                    
                else:
                    new_node.set_arg_by_node(2, dd)
                    new_node.set_arg_by_node(3, cc)
            tree.insert(position, new_node)
            next_node.set_arg_by_node(seat, new_node)
            new_net = stree.get_network()
            new_outputs = new_net(data_selected) 
            D[d_new_name] = stree  
            
            
            
            
            new_output = handle_format(new_outputs)
            N[d_new_name] = new_net  
            
            new_output = handle_format(new_outputs)
            O[d_new_name] = copy.deepcopy(new_output)

            print('ChebyshevDistance:',metrics.ChebyshevDistance(original_outputs,new_output),';  MAEDistance:',metrics.MAEDistance(original_outputs,new_output))
            dist_chess = metrics.ChebyshevDistance(original_outputs,new_output)
            gpu_memory2, cpu_memory2 = compute_gpu_cpu()
            metrics_dict[d_new_name] = [dist_chess,gpu_memory2,cpu_memory2]
            if new_output.shape!=original_outputs.shape:
                print('new_output.shape!=original_outputs.shape!')

        df = pd.DataFrame([(index, v[0], v[1], v[2]) for index, (k, v) in enumerate(metrics_dict.items())],
            columns=['name', 'Distance', 'Gpu_Memory_Used', 'Cpu_Memory_Used'])
        save_path = os.path.join("results", seed_model, str(localtime),"METRICS_RESULTS_" + str(ms_device).replace(':', '_') + ".xlsx")
        df.to_excel(save_path, index=False)

        
        dict_save_path = os.path.join("results", seed_model, str(localtime),"TORCH_LOG_DICT_" + str(ms_device).replace(':', '_') + ".json")
        os.makedirs(os.path.dirname(dict_save_path), exist_ok=True)
        with open(dict_save_path, 'w', encoding='utf-8') as file:
            json.dump(log_dict, file, ensure_ascii=False, indent=4)

    for index, (name, new_net) in enumerate(N.items()):
        print(index)
        if index == 0:
            continue
        try:
            new_net, stree, log_dict,option_index = api_mutation(new_net, option_layers, option_index, log_dict, n, LOG_FLAG=False) 
            print(f"Success during api_mutation")
        except Exception as e:
            print(f"Error during api_mutation: {e}")
            log_dict[n]['state'] = f"Failed: api_mutation failed: {str(e)}"
            
        option_layers = []
        
        for name, child in new_net.cells_and_names():
            if not has_child_node(new_net, name) and not name == '' and not 'deadcode' in str(type(child)):
                if name.split("_")[0] not in MR_structure_name_list:
                    option_layers.append((name, child, name, type(child)))


        

        

