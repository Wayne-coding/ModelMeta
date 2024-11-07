import torch
from infoplus.TorchInfoPlus import torchinfoplus
import gc
import json
from union import union_json
from Coverage import CoverageCalculatornew
import os
from memory_profiler import profile



def get_module_config(module):
    config_params = {}
    for attr_name, attr_value in module.__dict__.items():
        if not attr_name.startswith('_'):
            if isinstance(attr_value, torch.dtype):
                attr_value = str(attr_value)
            config_params[attr_name] = attr_value
    return config_params


def traverse_network(net, layer_config):

    for name, sub_module in net.named_children():

        layer_type_name = type(sub_module).__name__
        if layer_type_name not in layer_config.keys():
            layer_config[layer_type_name] = []

        config_params = get_module_config(sub_module)
        layer_config[layer_type_name].append(config_params)

        if isinstance(sub_module, torch.nn.Module):
            traverse_network(sub_module, layer_config)
    
    for key, value in layer_config.items():
        layer_config[key] = [dict(t) for t in {tuple(d.items()) for d in layer_config[key]}]

    return layer_config


def torch_model2json(model, input_tensor, input_dtypes):
    with torch.no_grad():
        result, global_layer_info, summary_list = torchinfoplus.summary( 
            model=model,
            input_data=input_tensor,
            dtypes=input_dtypes,
            col_names=['input_size', 'output_size', 'name'], depth=8,
            verbose=1)

        
        
        
        

        
        
        
        
        
        
        
        

        model_json = {}
        edge_list_list = []
        cur_edge_num = 0
        for index in range(len(summary_list) - 1):
            if int(summary_list[index].depth) >= 1:
                edge_list = []
                input_type = summary_list[index].class_name
                output_type = summary_list[index + 1].class_name
                
                edge_list.append(input_type)
                edge_list.append(output_type)
                cur_edge_num += 1
                if edge_list not in edge_list_list:
                    edge_list_list.append(edge_list)
        model_json["edges"] = edge_list_list
        layer_config = {}
        layer_config = traverse_network(model, layer_config)
        model_json["layer_config"] = layer_config

        layer_input_info = {}
        for layer_info in summary_list:
            layer_input_info_dist = {}
            if int(layer_info.depth) >= 1 and layer_info.input_size != []:
                input_name = layer_info.class_name
                
                output_name = layer_info.name
                
                input_size = layer_info.input_size[0]
                output_size = layer_info.output_size
                len_input_size = len(input_size)
                len_output_size = len(output_size)
                if input_name not in layer_input_info.keys():
                    layer_input_info_dist["input_dims"] = [len_input_size]
                    layer_input_info_dist["dtype"] = [str(input_dtypes)]
                    layer_input_info_dist["shape"] = [str(input_size)]
                    layer_input_info[input_name] = layer_input_info_dist
                else:
                    if len_input_size not in layer_input_info[input_name]["input_dims"]:
                        layer_input_info[input_name]["input_dims"].append(len_input_size)
                    if str(input_dtypes) not in layer_input_info[input_name]["dtype"]:
                        layer_input_info[input_name]["dtype"].append(str(input_dtypes))
                    if str(input_size) not in layer_input_info[input_name]["shape"]:
                        layer_input_info[input_name]["shape"].append(str(input_size))
        model_json["layer_input_info"] = layer_input_info

        current_layer_num = 0
        layer_type_list = []
        for ooo in summary_list:
            if int(ooo.depth) >= 1:
                input_name = ooo.class_name
                if input_name not in layer_type_list:
                    layer_type_list.append(input_name)
                current_layer_num += 1
        model_json["layer_num"] = current_layer_num
        model_json["layer_type"] = layer_type_list

        layer_dims = {}
        for layer_info in summary_list:
            if int(layer_info.depth) >= 1 and layer_info.input_size != []:
                input_name = layer_info.class_name
                
                input_size = layer_info.input_size[0]
                output_size = layer_info.output_size
                len_input_size = len(input_size)
                len_output_size = len(output_size)
                if input_name not in layer_dims:
                    layer_dims[input_name] = {
                        "input_dims": [len_input_size],
                        "output_dims": [len_output_size]
                    }
                else:
                    if len_input_size not in layer_dims[input_name]["input_dims"]:
                        layer_dims[input_name]["input_dims"].append(len_input_size)
                    if len_output_size not in layer_dims[input_name]["output_dims"]:
                        layer_dims[input_name]["output_dims"].append(len_output_size)
        
        model_json["cur_edge_num"] = cur_edge_num
        model_json["layer_dims"] = layer_dims

        
        return model_json

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()  
        elif isinstance(obj, torch.nn.Module):
            
            return "Instance of {}".format(obj.__class__.__name__)
        
        try:
            return super().default(obj)  
        except TypeError:
            return str(obj)  

@profile
def model2cov(model,input,dtype,d_file_path,api_config_pool_path):
    model_json_1 = torch_model2json(model, input, dtype) 
    os.makedirs(os.path.dirname(d_file_path), exist_ok=True)
    with open(d_file_path, 'w', encoding='utf-8') as file:
        json.dump(model_json_1, file,cls=CustomEncoder, ensure_ascii=False, indent=4)
        

    
    cal_cov = CoverageCalculatornew(api_config_pool_path)
    cal_cov.load_json(d_file_path)
    input_cov,config_cov,api_cov= cal_cov.cal_coverage()

    
    
    del model_json_1,model, input, dtype,cal_cov
    
    gc.collect()
    return input_cov,config_cov,api_cov





import torch.nn as nn
import torch.fx as fx
class vgg11(nn.Module):
    def __init__(self):
        super(vgg11, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1) 
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1) 
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1) 
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1) 
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv6 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv7 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv8 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.averagepool = nn.AvgPool2d(kernel_size=1, stride=1)
        self.fc1 = nn.Linear(512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = torch.nn.Flatten()

    def forward(self, inputs):
        x = self.relu1(self.conv1(inputs))
        x = self.maxpool(x)
        x = self.relu1(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu1(self.conv3(x))
        x = self.relu1(self.conv4(x))
        x = self.maxpool(x)
        x = self.relu1(self.conv5(x))
        x = self.relu1(self.conv6(x))
        x = self.maxpool(x)
        x = self.relu1(self.conv7(x))
        x = self.relu1(self.conv8(x))
        x = self.maxpool(x)
        x = self.averagepool(x)
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu1(self.fc2(x))
        x = self.fc3(x)
        return x

selected_MR_structure_name ="UOC"
from torch_mutation.MR_structure import *
MR_structures_map = {"UOC": UOC, "PIOC": PIOC, "ABSOC_A": ABSOC_A, "ABSOC_B": ABSOC_B}

if __name__ == '__main__':
    with torch.no_grad():
        net = vgg11()
        a = torch.randn(5, 3, 32, 32)
        d=fx.symbolic_trace(net)
        

        print('1111111111111111111111')
        model_json_1 = torch_model2json(d, a, [torch.float32])  
        print('1111111111111111111111')
        with torch.no_grad():
            graph = d.graph
            nodelist = []
            for node in graph.nodes:
                if node.op in ['call_module', 'root'] or \
                        (node.op == "call_function" and any(
                            substring in node.name for substring in ['uoc', 'pioc', 'absoc_a', 'absoc_b'])):
                    nodelist.append(node)
            

            try:
                add_module = MR_structures_map[selected_MR_structure_name]('conv')
            except Exception as e:
                exit(e)

            aa = nodelist[6]
            bb = nodelist[13]
            cc = nodelist[3]
            dd = nodelist[22]
            print(aa,bb,cc,dd) 


            if selected_MR_structure_name == "PIOC":
                with cc.graph.inserting_after(cc):
                    new_hybrid_node = cc.graph.call_function(add_module, args=(cc, cc, cc))
                    cc.replace_all_uses_with(new_hybrid_node)
                    new_hybrid_node.update_arg(0, aa)
                    new_hybrid_node.update_arg(1, bb)
                    new_hybrid_node.update_arg(2, cc)
            else:  
                with dd.graph.inserting_after(dd):
                    new_hybrid_node = dd.graph.call_function(add_module, args=(dd, dd, dd, dd))
                    dd.replace_all_uses_with(new_hybrid_node)
                    new_hybrid_node.update_arg(0, aa)
                    new_hybrid_node.update_arg(1, bb)
                    new_hybrid_node.update_arg(2, cc)
                    new_hybrid_node.update_arg(3, dd)
            graph.lint()  
            d.recompile()

        print('2222222222222222')
        print(d)
        model_json_1 = torch_model2json(d, a, [torch.float32])  
        print('2222222222222222')
        
        

