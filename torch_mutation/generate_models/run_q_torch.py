import numpy as np
import pandas as pd
from torch_mutation.cargo import *
import torch
import torch.optim as optim
import torch.distributions as dist
import copy
import time
import json
import torch.fx as fx
from torch_mutation.MR_structure import *
from torch_mutation.cargo import match_rule,reflect_name,MCMC,compute_gpu_cpu
from torch_mutation.api_mutation import api_mutation
from torch_mutation.cargo import select_places,max_seed_model_api_times
import psutil
import sys
import torch_mutation.config
from torch_mutation.calculate_coverage import model2cov
from torch_mutation.handel_shape import handle_format
import gc
from memory_profiler import profile

MR_structure_name_list = ['UOC', 'PIOC', 'ABSOC_A', 'ABSOC_B']
MR_structures_map = {"UOC": UOC, "PIOC": PIOC, "ABSOC_A": ABSOC_A, "ABSOC_B": ABSOC_B}
deadcode_name_list=['Dense', 'SELayer', 'DenseLayer', 'Inception_A', 'PWDWPW_ResidualBlock', 'ResidualBlock', 'DropPath']

device = config.device

def run_q_torch(seed_model, mutate_times,num_samples):
    localtime = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    log_dict = {}

    
    data = np.load(datasets_path_cargo[seed_model])
    samples = np.random.choice(data.shape[0], num_samples, replace=False)
    samples_data = data[samples] 
    data_selected = torch.tensor(samples_data,dtype=torch.int32 if seed_model in ["LSTM", "textcnn", "FastText"] else torch.float32).to(device)
    
    npy_path = os.path.join("results", seed_model, str(localtime), 'data0_npy.npy')
    os.makedirs(os.path.dirname(npy_path), exist_ok=True)
    np.save(npy_path, [data_selected.cpu().numpy()])

    seed_model_net=get_model(seed_model, device).to(device)
    d=fx.symbolic_trace(seed_model_net)
    d.eval()
    with torch.no_grad():
        original_outputs = handle_format(d(data_selected))[0]

    
    n_actions = 4  
    state_dim = 1 
    for i in original_outputs.shape:
        state_dim *= i
    gamma = 0.99  
    target_update = 5  
    epsilon = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    
    Quantum_Q = QRDQN(state_dim, n_actions).to(device)
    Target_Q = copy.deepcopy(Quantum_Q).to(device)
    Target_Q.eval()  
    
    optimizer = optim.Adam(Quantum_Q.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    metrics_dict = dict()
    select_d_name = seed_model
    D = {seed_model: d} 
    R = {0:[0.0001, seed_model]} 
    MR_structure_selected_nums = {k: 0 for k in MR_structure_name_list}  
    seed_model_api_times=0 
    d = copy.deepcopy(d)

    
    for n in range(mutate_times):
        print('-----------------------total_Mutate_time:%d start!-----------------------' % n)
        start_time=time.time()
        try:
            log_dict[n] = {}
            log_dict[n]['d_name'] = select_d_name
            old_d_name=select_d_name 

            
            selected_deadcode_name=random.choice(deadcode_name_list)

            
            if random.uniform(0, 1) <= epsilon:
                selected_MR_structure_idx = random.choice([0, 1, 2, 3])
            else:
                with torch.no_grad():
                    selected_MR_structure_idx = Quantum_Q(handle_format(d(data_selected))[0]).argmax().item()
            selected_MR_structure_name = MR_structure_name_list[selected_MR_structure_idx]
            MR_structure_selected_nums[selected_MR_structure_name] += 1

            
            d_new_name = "{}-{}{}".format(seed_model, selected_MR_structure_name,MR_structure_selected_nums[selected_MR_structure_name])
            log_dict[n]['d_new_name'] = d_new_name

            
            if selected_deadcode_name in ('DropPath', 'Dense') and seed_model_api_times<max_seed_model_api_times(seed_model):  
                api_mutation_type = 'seed_model'
                seed_model_api_times+=1
            elif selected_deadcode_name not in ('DropPath', 'Dense') and seed_model_api_times<max_seed_model_api_times(seed_model): 
                api_mutation_type = random.choice(['seed_model', 'deadcode'])  
                if api_mutation_type=='seed_model':
                    seed_model_api_times += 1
            elif selected_deadcode_name not in ('DropPath', 'Dense') and seed_model_api_times >= max_seed_model_api_times(seed_model):  
                api_mutation_type = 'deadcode'
            else: 
                api_mutation_type = 'None'
                log_dict[n]['state'] = "Success:But no APIs available for mutation, so no API-level mutation was performed."
            

            with torch.no_grad():
                graph = d.graph
                nodelist = []
                for node in graph.nodes:
                    if node.op in ['call_module', 'root'] or \
                            (node.op == "call_function" and  any(substring in node.name for substring in ['uoc', 'pioc', 'absoc_a', 'absoc_b'])):
                        nodelist.append(node)

                
                subs_place, dep_places = select_places(range(0, len(nodelist)), 5)

                if subs_place is None or dep_places is None: 
                    log_dict[n]['state'] = f"Failed:Cannot find suitable places！"

                print("~~~~~~~~~~~~~~~~~选择对%s中的算子进行api变异！~~~~~~~~~~~~~~~" % api_mutation_type)
                add_module = MR_structures_map[selected_MR_structure_name](selected_deadcode_name, api_mutation_type, log_dict, n, LOG_FLAG=False)

                dep_places.sort(reverse=True)
                aa = nodelist[dep_places[-1]]
                bb = nodelist[dep_places[-2]]
                cc = nodelist[dep_places[-3]]
                dd = nodelist[dep_places[-4]]

                if selected_MR_structure_name == "PIOC":
                    if len(aa.users) == 0 or len(bb.users) == 0 or len(cc.users) == 0:
                        log_dict[n]['state'] = "Failed:选择插入的节点位置不正确！"
                    with cc.graph.inserting_after(cc):
                        new_hybrid_node = cc.graph.call_function(add_module, args=(cc, cc, cc))
                        cc.replace_all_uses_with(new_hybrid_node)
                        new_hybrid_node.update_arg(0, aa)
                        new_hybrid_node.update_arg(1, bb)
                        new_hybrid_node.update_arg(2, cc)
                else:  
                    if len(aa.users) == 0 or len(bb.users) == 0 or len(cc.users) == 0 or len(dd.users) == 0:
                        log_dict[n]['state'] = "Failed:选择插入的节点位置不正确！"
                    with dd.graph.inserting_after(dd):
                        new_hybrid_node = dd.graph.call_function(add_module, args=(dd, dd, dd, dd))
                        dd.replace_all_uses_with(new_hybrid_node)
                        new_hybrid_node.update_arg(0, aa)
                        new_hybrid_node.update_arg(1, bb)
                        new_hybrid_node.update_arg(2, cc)
                        new_hybrid_node.update_arg(3, dd)

                graph.lint() 
                d.recompile()
                D[d_new_name] = d  

                if api_mutation_type == 'seed_model':
                    api_mutation(d, log_dict, n,LOG_FLAG=False) 

                
                d = d.to(device)
                new_outputs = handle_format(d(data_selected))[0]
                if new_outputs.shape!=original_outputs.shape:
                    sys.exit('new_outputs.shape!=original_outputs.shape!')

                
                
                

        except Exception as e:
            log_dict[n]['state'] = f"Failed: Error during mutation: {str(e)}"

        
        with torch.no_grad():
            if ('state' in log_dict[n]) and ("Success" not in log_dict[n]['state']):  
                reward,done=-1,True
                
                d_probs = torch.distributions.Beta(torch.tensor([value[0] for value in R.values()]), torch.ones(len(R))).sample()
                select_d_name = R[torch.argmax(d_probs).item()][1]
                d = copy.deepcopy(D[select_d_name])
                metrics_dict[d_new_name]=["None"]*4
            else:
                done = False
                json_file_path=os.path.join("results", seed_model, str(localtime),"model_json" , d_new_name+ ".json")
                api_config_pool_path=r"./torch_api_config_pool.json"
                
                input_cov, config_cov, api_cov = model2cov(d, data_selected, [torch.float32], json_file_path,api_config_pool_path)
                reward= (input_cov+config_cov+api_cov)/3
                metrics_dict[d_new_name]=[input_cov,config_cov,api_cov,reward]

                R[len(R)]=[reward,d_new_name] 
                d = copy.deepcopy(d)  
                select_d_name=d_new_name
        

        
        
        formatted_data = handle_format(D[old_d_name](data_selected))[0].unsqueeze(0)
        quantum_q_value = Quantum_Q(formatted_data)
        Quantum_Q_value = quantum_q_value[0, selected_MR_structure_idx].unsqueeze(0)
        with torch.no_grad():
            next_q_value = Target_Q(handle_format(d(data_selected))[0].unsqueeze(0)).max(1)[0]
        Target_Q_value = reward + gamma * next_q_value * (1 - done)

        
        
        

        loss = criterion(Target_Q_value, Quantum_Q_value) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if n % target_update == 0:  
            Target_Q.load_state_dict(Quantum_Q.state_dict())

        epsilon = max(epsilon_end, epsilon_decay * epsilon)

        end_time = time.time()
        elapsed_time = end_time - start_time  
        metrics_dict[d_new_name].append(elapsed_time)
        del formatted_data,next_q_value
        gc.collect()
        torch.cuda.empty_cache()


        
        if ('state' in log_dict[n]) and ("Success" not in log_dict[n]['state']):  
            log_dict[n]['select_d_name'] = select_d_name
        else:
            log_dict[n]['state']='Success!'
            log_dict[n]['select_deadcode'] = selected_deadcode_name
            log_dict[n]['selected_MR_structure'] = selected_MR_structure_name
            log_dict[n]['subs_place'], log_dict[n]['dep_places'] = subs_place, dep_places
            log_dict[n]['api_mutation_type(seed_model or deadcode)'] = api_mutation_type
            log_dict[n]['select_d_name'] = select_d_name
        
        dict_save_path = os.path.join("results", seed_model, str(localtime),"TORCH_LOG_DICT_" + str(device).replace(':', '_') + ".json")
        os.makedirs(os.path.dirname(dict_save_path), exist_ok=True)
        with open(dict_save_path, 'w', encoding='utf-8') as file:
            json.dump(log_dict, file, ensure_ascii=False, indent=4)

        
        df = pd.DataFrame([(k, v[0], v[1],v[2],v[3],v[4]) for k, v in metrics_dict.items()],
                          columns=['New_Model_Name', 'Input_cov','Config_cov','Api_cov','Avg_cov','Elapsed_time'])
        save_path = os.path.join("results", seed_model, str(localtime),"METRICS_RESULTS_" + str(device).replace(':', '_') + ".xlsx")
        df.to_excel(save_path, index=False)

        print('-----------------------total_Mutate_time:%d ended!-----------------------' % n)
