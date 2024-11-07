import os

import mindspore
import numpy as np
from mindspore import Tensor
from mindspore.rewrite import SymbolTree

import json
from Coverage import CoverageCalculatornew
from infoplus.MindSporeInfoPlus import mindsporeinfoplus

from union import union_json

import networkx as nx
from mindspore.rewrite import NodeType
from mindspore.nn import SequentialCell
import mindspore.nn as nn
from mindspore import ops
from mindspore_mutation.msmodel2json import *

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

def model2cov(model, input, dtype, file_path_1, all_json_path, api_config_pool_path, folder_path):
    model_json_1, inside, output_datas = ms_model2json(model, input, dtype)
    with open(file_path_1, 'w', encoding='utf-8') as file:
        json.dump(model_json_1, file, ensure_ascii=False, indent=4)
    file.close()

    cal_cov = CoverageCalculatornew(all_json_path, api_config_pool_path)
    cal_cov.load_json(file_path_1)
    
    input_cov, config_cov, api_cov, op_type_cov, op_num_cov, edge_cov = cal_cov.cal_coverage()
    return input_cov, config_cov, api_cov, op_type_cov, op_num_cov, edge_cov, inside, output_datas


def find_layer_type(new_net):
    layer_type_list = []
    layers = new_net.cells_and_names()
    for i, j in layers:
        if i != '':
            if not has_child_node(new_net, i):
                if j.__class__.__name__ not in layer_type_list:
                    layer_type_list.append(j.__class__.__name__)
    return layer_type_list

    


















#     return input_cov,config_cov,api_cov