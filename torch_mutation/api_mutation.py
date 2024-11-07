import numpy as np
from torch_mutation.cargo import *
import torch
import copy
import time
import json
import torch.fx as fx

from cargo import match_rule,reflect_name,MCMC,rule_reflect_class
import config

device=config.device
rules_dict = config.rules_dict 

def api_mutation(d,log_dict,n,LOG_FLAG):
    graph=d.graph
    if LOG_FLAG == False:
        option_layers = list()  
        for node in graph.nodes:
            if node.op == 'call_module' and '_mutate' not in node.name:  
                module_name = node.target
                module_instance = d.get_submodule(module_name)  
                option_layers.append((node, module_instance, node.name))
        
        

        if len(option_layers) != 0:
            option_node, option_instance, option_name = random.choice(
                option_layers)  
            option_rule = random.choice(rules_dict[type(option_instance)])  
            new_instance = rule_reflect_class(option_rule,option_instance)(option_instance)  

            print('选择对其变异的node：', option_node)
            print('选择对其变异的node的名字：', option_name)
            print('选择对其变异的instance：', option_instance)
            print('选择的变异规则：', option_rule)
            print('变异后新的层：', new_instance)

            log_dict[n]['seedmodel_api_name'] = option_name
            if option_rule.__name__[-2:] in ('10', '11', '12', '13', '14', '15', '16', '17', '18'):
                log_dict[n]['seedmodel_api_rule'] = option_rule.__name__[-6:]
            else:
                log_dict[n]['seedmodel_api_rule'] = option_rule.__name__[-5:]

    else:  
        option_name = log_dict[str(n)]['seedmodel_api_name']
        option_rule_name = log_dict[str(n)]['seedmodel_api_rule']

        for node in graph.nodes:
            if node.name == option_name:
                module_name = node.target
                module_instance = d.get_submodule(module_name)  
                option_node, option_instance, option_name = node, module_instance, node.name
                break

        option_rule = match_rule(option_rule_name)
        new_instance = rule_reflect_class(option_rule,option_instance)(option_instance)  

        print('选择对其变异的node：', option_node)
        print('选择对其变异的node的名字：', option_name)
        print('选择对其变异的instance：', option_instance)
        print('选择的变异规则：', option_rule)
        print('变异后新的层：', new_instance)

    
    

    
    new_name = reflect_name(option_name, option_rule)

    
    d.add_module(new_name, new_instance)
    
    with option_node.graph.inserting_after(option_node):
        new_node = option_node.graph.call_module(new_name, args=option_node.args)
        option_node.replace_all_uses_with(new_node)
        d.graph.erase_node(option_node)
    
    graph.lint()
    d.recompile()



