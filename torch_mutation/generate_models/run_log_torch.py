import numpy as np
import pandas as pd
from torch_mutation.cargo import *
import torch
import copy
import time
import json
import torch.fx as fx
from torch_mutation.MR_structure import *
from torch_mutation.cargo import compute_gpu_cpu
from torch_mutation.api_mutation import api_mutation
from torch_mutation.metrics import MAEDistance
from torch_mutation.cargo import get_loss
import sys
import torch_mutation.config as config
from torch_mutation.handel_shape import handle_format

MR_structure_name_list = ['UOC', 'PIOC', 'ABSOC_A', 'ABSOC_B']
MR_structures_map = {"UOC": UOC, "PIOC": PIOC, "ABSOC_A": ABSOC_A, "ABSOC_B": ABSOC_B}

device = config.device

def run_log_torch(seed_model, mutate_times,log_path, num_samples,data_x_path,data_y_path,path_flag):
    localtime = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    with open(log_path, "r") as json_file:
        log_dict = json.load(json_file)

    if path_flag is False: # 随机选择数据并保存
        # 数值数据
        data = np.load(datasets_path_cargo[seed_model])
        # print(data.shape)
        samples = np.random.choice(data.shape[0], num_samples, replace=False)
        samples_data = data[samples] # 随机选择num_samples个数据 (num_samples, 3, 32, 32)
        data_selected = torch.tensor(samples_data,dtype=torch.int32 if seed_model in ["LSTM", "textcnn", "FastText"] else torch.float32).to(device)
        # 保存选中的数据
        npy_path = os.path.join("results", seed_model, str(localtime), 'data0_npy.npy')
        os.makedirs(os.path.dirname(npy_path), exist_ok=True)
        np.save(npy_path, [data_selected.cpu().numpy()])

        # y
        y_outputs =np.load('/root/myz/data_npy/cifar10_y_one_hot_encoded.npy')
        y_outputs=y_outputs[samples]
        y_outputs=torch.tensor(y_outputs,dtype=torch.int32 if seed_model in ["LSTM", "textcnn", "FastText"] else torch.float32).to(device)[0]
        # 保存

    else: # 根据地址选择数据
        samples_data=np.load(data_x_path)
        data_selected = torch.tensor(samples_data, dtype=torch.int32 if seed_model in ["LSTM", "textcnn","FastText"] else torch.float32).to(device)
        y_outputs=np.load(data_x_path)
        y_outputs = torch.tensor(y_outputs, dtype=torch.int32 if seed_model in ["LSTM", "textcnn","FastText"] else torch.float32).to(device)

        # 保存选中的数据,以验证选择的数据是否相同
        npy_path = os.path.join("results", seed_model, str(localtime), 'data0_npy.npy')
        os.makedirs(os.path.dirname(npy_path), exist_ok=True)
        np.save(npy_path, [data_selected.cpu().numpy()])


    seed_model_net=get_model(seed_model, device).to(device)
    d=fx.symbolic_trace(seed_model_net)
    d.eval()
    with torch.no_grad():
        original_outputs = handle_format(d(data_selected))[0] 

    metrics_dict = dict()
    D = {seed_model: d} # 所有模型
    d = copy.deepcopy(d)

    # 循环变异：
    for n in range(mutate_times):
        print('-----------------------total_Mutate_time:%d start!-----------------------' % n)
        with torch.no_grad():
            if "Success" in log_dict[str(n)]['state']: # 对于变异成功的模型：正常生成变异模型，正常检测
                start_time = time.time()

                # 选择死代码
                selected_deadcode_name =log_dict[str(n)]['select_deadcode']

                # 选择 MR_structure
                selected_MR_structure_name = log_dict[str(n)]['selected_MR_structure']

                # 命名新模型：种子模型、MR_structure、第几次用这个MR_structure
                d_new_name = log_dict[str(n)]['d_new_name']

                # 算子变异对死代码还是原模型？
                api_mutation_type = log_dict[str(n)]['api_mutation_type(seed_model or deadcode)']

                graph = d.graph
                nodelist = []
                for node in graph.nodes:
                    if node.op in ['call_module', 'root'] or (node.op == "call_function" and any(substring in node.name for substring in ['uoc', 'pioc', 'absoc_a', 'absoc_b'])):
                        nodelist.append(node)

                # 选择插入位置
                subs_place, dep_places = log_dict[str(n)]['subs_place'], log_dict[str(n)]['dep_places']
                if subs_place is None or dep_places is None: # 没找到合适的插入位置
                    sys.exit("mutate failed for Cannot find suitable places！")

                print("~~~~~~~~~~~~~~~~~选择对%s中的算子进行api变异！~~~~~~~~~~~~~~~" % api_mutation_type)

                try:
                    add_module = MR_structures_map[selected_MR_structure_name](selected_deadcode_name, api_mutation_type, log_dict,n, LOG_FLAG=True)
                except Exception as e:
                    exit(e)

                dep_places.sort(reverse=True)
                aa = nodelist[dep_places[-1]]
                bb = nodelist[dep_places[-2]]
                cc = nodelist[dep_places[-3]]
                dd = nodelist[dep_places[-4]]

                if selected_MR_structure_name == "PIOC":
                    if len(aa.users) == 0 or len(bb.users) == 0 or len(cc.users) == 0:
                        sys.exit("选择插入的节点位置不正确！")
                    with cc.graph.inserting_after(cc):
                        new_hybrid_node = cc.graph.call_function(add_module, args=(cc, cc, cc))
                        cc.replace_all_uses_with(new_hybrid_node)
                        new_hybrid_node.update_arg(0, aa)
                        new_hybrid_node.update_arg(1, bb)
                        new_hybrid_node.update_arg(2, cc)
                else:  # selected_MR_structure_name != "PIOC"
                    if len(aa.users) == 0 or len(bb.users) == 0 or len(cc.users) == 0 or len(dd.users) == 0:
                        sys.exit("选择插入的节点位置不正确！")
                    with dd.graph.inserting_after(dd):
                        new_hybrid_node = dd.graph.call_function(add_module, args=(dd, dd, dd, dd))
                        dd.replace_all_uses_with(new_hybrid_node)
                        new_hybrid_node.update_arg(0, aa)
                        new_hybrid_node.update_arg(1, bb)
                        new_hybrid_node.update_arg(2, cc)
                        new_hybrid_node.update_arg(3, dd)
                graph.lint()  # 检查是否有图错误并重新编译图
                d.recompile()
                D[d_new_name] = d  # 加入到种子模型池

                if api_mutation_type == 'seed_model':
                    try:
                        api_mutation(d, log_dict, n, LOG_FLAG=True)  # 进行API变异
                    except Exception as e:
                        print(f"Error during api_mutation: {e}")
                end_time = time.time()
                elapsed_time = end_time - start_time  # 生成1个模型的时间

                # 推理
                d = d.to(device)
                new_outputs = handle_format(d(data_selected))[0]  # torch.Size([5, 10])  torch.Size([10])
                if isinstance(new_outputs, torch.Tensor):
                    distance = MAEDistance(original_outputs, new_outputs)  # np.max(np.abs(x - y))
                    # 损失函数
                    loss_fun_ms, loss_fun_torch = get_loss('CrossEntropy')
                    loss_fun_ms, loss_fun_torch = loss_fun_ms(), loss_fun_torch().to(device)
                    loss_torch = loss_fun_torch(new_outputs, y_outputs)

                gpu_memory1, cpu_memory1 = compute_gpu_cpu()
                d = copy.deepcopy(D[log_dict[str(n)]['select_d_name']]) # 下一个种子模型
                gpu_memory2, cpu_memory2 = compute_gpu_cpu()

                gpu_memory_used = gpu_memory2 - gpu_memory1 # 本次蜕变生成的模型占用的资源
                cpu_memory_used = cpu_memory1 - cpu_memory2
                metrics_dict[d_new_name] = [distance,elapsed_time,gpu_memory_used,cpu_memory_used,loss_torch]

            else: # 对于crash的新模型，都取None
                d_new_name=log_dict[str(n)]['d_new_name']
                d = copy.deepcopy(D[log_dict[str(n)]['select_d_name']])  # 下一个种子模型
                metrics_dict[d_new_name] = ['None']*5


        # 保存指标
        df = pd.DataFrame([(k, v[0], v[1], v[2], v[3],v[4]) for k, v in metrics_dict.items()],
            columns=['New_Model_Name', 'MAE_Distance', 'Time', 'Gpu_Memory_Used', 'Cpu_Memory_Used','Loss'])
        save_path = os.path.join("results", seed_model, str(localtime),"METRICS_RESULTS_" + str(device).replace(':', '_') + ".xlsx")
        df.to_excel(save_path, index=False)

    # 保存json:
    dict_save_path = os.path.join("results", seed_model, str(localtime),
                                  "TORCH_LOG_DICT_" + str(device).replace(':', '_') + ".json")
    os.makedirs(os.path.dirname(dict_save_path), exist_ok=True)
    with open(dict_save_path, 'w', encoding='utf-8') as file:
        json.dump(log_dict, file, ensure_ascii=False, indent=4)







