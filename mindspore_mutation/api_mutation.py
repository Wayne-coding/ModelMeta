import numpy as np
import mindspore.nn as nn
from mindspore.rewrite import ScopedValue, NodeType
from mindspore.rewrite.node import Node, NodeManager
from mindspore.rewrite import SymbolTree, Node, NodeType
from mindspore_mutation.cargo import *
import copy
import time
import json
import mindspore_mutation.config as ms_config
import collections
import mindspore
mindspore.set_context(mode=mindspore.PYNATIVE_MODE, device_target="GPU")

ms_device=ms_config.device
rules_dict = ms_config.rules_dict 

def has_child_node(net, node_name):
    layers = net.cells_and_names()
    parent_node = None
    for name, _ in layers:
        if name == node_name:
            parent_node = name
            continue
        if parent_node is not None and name.startswith(parent_node + '.'):
            return True
    return False


MR_structure_name_list = ['UOC', 'PIOC', 'ABSOC_A', 'ABSOC_B']


def api_mutation(net, option_layers, option_index, log_dict, n, LOG_FLAG):
    if LOG_FLAG == False:

        
        
        available_indices = [i for i in range(len(option_layers)) if i not in option_index]

        
        random_index = random.choice(available_indices)
        
 
        option_name, option_instance, option_instance_type, option_node_type = option_layers[random_index]
        
        
        
        option_rule = random.choice(rules_dict[type(option_instance)])  
        
        new_instance = rule_reflect_class(option_rule,option_instance)(option_instance)  


        log_dict[n]['seedmodel_api_name'] = option_name
        if option_rule.__name__[-2:] in ('10', '11', '12', '13', '14', '15', '16', '17', '18'):
            log_dict[n]['seedmodel_api_rule'] = option_rule.__name__[-6:]
        else:
            log_dict[n]['seedmodel_api_rule'] = option_rule.__name__[-5:]

    else:  
        option_name = log_dict[str(n)]['seedmodel_api_name']
        option_rule_name = log_dict[str(n)]['seedmodel_api_rule']
        option_instance = layer._cells.get(option_name, None)

        if option_instance is not None:
            option_rule = match_rule(option_rule_name)
            new_instance = rule_reflect_class(option_rule, option_instance)(option_instance)  
            net._cells[option_name] = new_instance
            net.update_parameters_name()
        return net, SymbolTree.create(net), log_dict, option_index

    
    new_name = reflect_name(option_name, option_rule)
    net._cells[option_name] = new_instance
    net.update_parameters_name()
    i = 0
    for name, child in net.cells_and_names():
        if not has_child_node(net, name) and not name == '' and not 'deadcode' in str(type(child)):
            i += 1
            if name == option_name and i not in option_index:
                option_index.append(i)
                break

    
    return net, SymbolTree.create(net), log_dict, option_index

