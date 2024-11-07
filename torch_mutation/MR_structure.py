import numpy as np
import torch
from torch import nn, fx
from torch_mutation.deadcode import SELayer, Inception_A, DenseLayer, DropPath, Dense, ResidualBlock, PWDWPW_ResidualBlock
from torch_mutation.handel_shape import handle_shape_final, handle_shape_strict, make_unsqueeze, make_reduce
from cargo import match_rule,reflect_name,rule_reflect_class
import random
import config

device=config.device
rules_dict = config.rules_dict 






















class UOC(nn.Module):
    def __init__(self, op_type,operator_mutation_type,log_dict,i,LOG_FLAG):
        super().__init__()
        self.__name__ = "UOC"
        self.op_type = op_type
        if self.op_type == "SELayer":
            layer = SELayer()
        if self.op_type == "DenseLayer":
            layer = DenseLayer()
        if self.op_type == "Inception_A":
            layer = Inception_A()
        if self.op_type == "PWDWPW_ResidualBlock":
            layer = PWDWPW_ResidualBlock()
        if self.op_type == "ResidualBlock":
            layer = ResidualBlock()
        if self.op_type == "DropPath":
            layer = DropPath()
        if self.op_type == "Dense":
            layer = Dense()

        if operator_mutation_type == 'deadcode':
            self.op_layer = self.modify_layer_with_fx(layer, log_dict, i, LOG_FLAG)  
        else:
            self.op_layer = layer

    def modify_layer_with_fx(self, layer,log_dict,i,LOG_FLAG):
        new_module=torch.fx.symbolic_trace(layer)
        graph=new_module.graph

        if LOG_FLAG == False:
            option_layers = list()  
            for node in graph.nodes:
                if node.op == 'call_module' and '_mutate' not in node.name:  
                    module_name = node.target
                    module_instance = new_module.get_submodule(module_name)  
                    option_layers.append((node, module_instance, node.name))
            

            if len(option_layers) != 0:
                option_node, option_instance, option_name = random.choice(option_layers)  
                option_rule = random.choice(rules_dict[type(option_instance)])  
                
                new_instance = rule_reflect_class(option_rule,option_instance)(option_instance)  

                print('uoc：选择对其变异的node：', option_node)
                print('uoc：选择对其变异的node的名字：', option_name)
                print('uoc：选择对其变异的instance：', option_instance)
                print('uoc：选择的变异规则：', option_rule)
                print('uoc：变异后新的层：', new_instance)

                log_dict[i]['deadcode_api_name'] = option_name
                if option_rule.__name__[-2:] in ('10', '11', '12', '13', '14', '15', '16', '17', '18'):
                    log_dict[i]['deadcode_api_rule'] = option_rule.__name__[-6:]
                else:
                    log_dict[i]['deadcode_api_rule'] = option_rule.__name__[-5:]

        else:  
            option_name = log_dict[str(i)]['deadcode_api_name']
            option_rule_name = log_dict[str(i)]['deadcode_api_rule']

            for node in graph.nodes:
                if node.name == option_name:
                    module_name = node.target
                    module_instance = new_module.get_submodule(module_name)  
                    option_node, option_instance, option_name = node, module_instance, node.name
                    break

            option_rule = match_rule(option_rule_name)
            new_instance = rule_reflect_class(option_rule,option_instance)(option_instance)  

            print('uoc：选择对其变异的node：', option_node)
            print('uoc：选择对其变异的node的名字：', option_name)
            print('uoc：选择对其变异的instance：', option_instance)
            print('uoc：选择的变异规则：', option_rule)
            print('uoc：变异后新的层：', new_instance)

        
        

        
        new_name = reflect_name(option_name, option_rule)

        
        new_module.add_module(new_name, new_instance)
        
        with option_node.graph.inserting_after(option_node):
            new_node = option_node.graph.call_module(new_name, args=option_node.args)
            option_node.replace_all_uses_with(new_node)
            new_module.graph.erase_node(option_node)
        
        graph.lint()
        new_module.recompile()
        return new_module

    def forward(self, a, b, deada, deadb):
        if isinstance(a, tuple) or isinstance(b, tuple) or isinstance(deada, tuple) or isinstance(deadb, tuple):
            print("mutate failed for input tuple")
            print("type a", type(a))
            print("type b", type(b))
            print("type deada", type(deada))
            print("type deadb", type(deadb))
            return a, b, deada, deadb

        a, b = handle_shape_strict(a, b)  

        a2 = torch.mul(a, a)
        b2 = torch.mul(b, b)
        ab = torch.mul(a, b)
        ab2 = torch.mul(ab, -2)
        uoc = torch.add(a2, b2)
        uoc = torch.add(uoc, ab2)
        uoc = torch.add(uoc, 1e-10)
        uoc = torch.neg(uoc)
        uoc = torch.sub(uoc, 1e-5)
        
        uoc = random.choice([nn.ReLU(), nn.ReLU6(), nn.Hardtanh()])(uoc) 

        dead = make_unsqueeze(deada)  
        add_edge = self.op_layer(dead)
        uoc, add_edge = handle_shape_strict(uoc, add_edge)  
        out0 = torch.mul(uoc, add_edge)
        dtype = deadb.dtype
        out, deadbb = handle_shape_final(out0, deadb)
        out = torch.add(out, deadbb)
        out = out.to(dtype)

        return out


class ABSOC_A(nn.Module):
    def __init__(self, op_type,operator_mutation_type,log_dict,i,LOG_FLAG):
        super().__init__()
        self.__name__ = "ABSOC_A"
        self.op_type = op_type
        if self.op_type == "SELayer":
            layer = SELayer()
        if self.op_type == "DenseLayer":
            layer = DenseLayer()
        if self.op_type == "Inception_A":
            layer = Inception_A()
        if self.op_type == "PWDWPW_ResidualBlock":
            layer = PWDWPW_ResidualBlock()
        if self.op_type == "ResidualBlock":
            layer = ResidualBlock()
        if self.op_type == "DropPath":
            layer = DropPath()
        if self.op_type == "Dense":
            layer = Dense()

        if operator_mutation_type == 'deadcode':
            self.op_layer = self.modify_layer_with_fx(layer,log_dict,i,LOG_FLAG) 
        else:
            self.op_layer = layer

    def modify_layer_with_fx(self, layer,log_dict,i,LOG_FLAG):
        new_module=torch.fx.symbolic_trace(layer)
        graph=new_module.graph

        if LOG_FLAG == False:
            option_layers = list()  
            for node in graph.nodes:
                if node.op == 'call_module' and '_mutate' not in node.name:  
                    module_name = node.target
                    module_instance = new_module.get_submodule(module_name)  
                    option_layers.append((node, module_instance, node.name))
            

            if len(option_layers) != 0:
                option_node, option_instance, option_name = random.choice(option_layers)  
                option_rule = random.choice(rules_dict[type(option_instance)])  
                new_instance = rule_reflect_class(option_rule,option_instance)(option_instance)  

                print('ABSOC_A：选择对其变异的node：', option_node)
                print('ABSOC_A：选择对其变异的node的名字：', option_name)
                print('ABSOC_A：选择对其变异的instance：', option_instance)
                print('ABSOC_A：选择的变异规则：', option_rule)
                print('ABSOC_A：变异后新的层：', new_instance)

                log_dict[i]['deadcode_api_name'] = option_name
                if option_rule.__name__[-2:] in ('10', '11', '12', '13', '14', '15', '16', '17', '18'):
                    log_dict[i]['deadcode_api_rule'] = option_rule.__name__[-6:]
                else:
                    log_dict[i]['deadcode_api_rule'] = option_rule.__name__[-5:]

        else:  
            option_name = log_dict[str(i)]['deadcode_api_name']
            option_rule_name = log_dict[str(i)]['deadcode_api_rule']

            for node in graph.nodes:
                if node.name == option_name:
                    module_name = node.target
                    module_instance = new_module.get_submodule(module_name)  
                    option_node, option_instance, option_name = node, module_instance, node.name
                    break

            option_rule = match_rule(option_rule_name)
            new_instance = rule_reflect_class(option_rule,option_instance)(option_instance)  

            print('ABSOC_A：选择对其变异的node：', option_node)
            print('ABSOC_A：选择对其变异的node的名字：', option_name)
            print('ABSOC_A：选择对其变异的instance：', option_instance)
            print('ABSOC_A：选择的变异规则：', option_rule)
            print('ABSOC_A：变异后新的层：', new_instance)

        
        

        
        new_name = reflect_name(option_name, option_rule)

        
        new_module.add_module(new_name, new_instance)
        
        with option_node.graph.inserting_after(option_node):
            new_node = option_node.graph.call_module(new_name, args=option_node.args)
            option_node.replace_all_uses_with(new_node)
            new_module.graph.erase_node(option_node)
        
        graph.lint()
        new_module.recompile()
        

        return new_module

    def forward(self, a, b, deada, deadb):
        if isinstance(a, tuple) or isinstance(b, tuple) or isinstance(deada, tuple) or isinstance(deadb, tuple):
            print("mutate failed for input tuple")
            print("type a", type(a))
            print("type b", type(b))
            print("type deada", type(deada))
            print("type deadb", type(deadb))
            return a, b, deada, deadb
        a, b = handle_shape_strict(a, b)
        a1 = torch.abs(a)
        b1 = torch.abs(b)
        a1b1 = torch.add(a1, b1)  
        ab = torch.abs(torch.add(a, b))  
        absoc_a = torch.sub(a1b1, ab)  
        absoc_a = torch.add(absoc_a, 1e-10)  
        absoc_a = torch.neg(absoc_a)  
        
        absoc_a = random.choice([nn.ReLU(), nn.ReLU6(), nn.Hardtanh()])(absoc_a)  
        dead = make_unsqueeze(deada)
        add_edge = self.op_layer(dead)
        absoc_a, add_edge = handle_shape_strict(absoc_a, add_edge)
        out = torch.mul(absoc_a, add_edge)
        dtype = deadb.dtype
        out, deadb = handle_shape_final(out, deadb)
        out = torch.add(out, deadb)
        out = out.to(dtype)
        return out


class ABSOC_B(nn.Module):
    def __init__(self, op_type,operator_mutation_type,log_dict,i,LOG_FLAG):
        super().__init__()
        self.__name__ = "ABSOC_B"
        self.op_type = op_type
        if self.op_type == "SELayer":
            layer = SELayer()
        if self.op_type == "DenseLayer":
            layer = DenseLayer()
        if self.op_type == "Inception_A":
            layer = Inception_A()
        if self.op_type == "PWDWPW_ResidualBlock":
            layer = PWDWPW_ResidualBlock()
        if self.op_type == "ResidualBlock":
            layer = ResidualBlock()
        if self.op_type == "DropPath":
            layer = DropPath()
        if self.op_type == "Dense":
            layer = Dense()

        if operator_mutation_type == 'deadcode':
            self.op_layer = self.modify_layer_with_fx(layer, log_dict, i, LOG_FLAG)  
        else:
            self.op_layer = layer

    def modify_layer_with_fx(self, layer,log_dict,i,LOG_FLAG):
        new_module=torch.fx.symbolic_trace(layer)
        graph=new_module.graph

        if LOG_FLAG == False:
            option_layers = list()  
            for node in graph.nodes:
                if node.op == 'call_module' and '_mutate' not in node.name:  
                    module_name = node.target
                    module_instance = new_module.get_submodule(module_name)  
                    option_layers.append((node, module_instance, node.name))
            

            if len(option_layers) != 0:
                option_node, option_instance, option_name = random.choice(option_layers)  
                option_rule = random.choice(rules_dict[type(option_instance)])  
                new_instance = rule_reflect_class(option_rule,option_instance)(option_instance)  

                print('ABSOC_B：选择对其变异的node：', option_node)
                print('ABSOC_B：选择对其变异的node的名字：', option_name)
                print('ABSOC_B：选择对其变异的instance：', option_instance)
                print('ABSOC_B：选择的变异规则：', option_rule)
                print('ABSOC_B：变异后新的层：', new_instance)

                log_dict[i]['deadcode_api_name'] = option_name
                if option_rule.__name__[-2:] in ('10', '11', '12', '13', '14', '15', '16', '17', '18'):
                    log_dict[i]['deadcode_api_rule'] = option_rule.__name__[-6:]
                else:
                    log_dict[i]['deadcode_api_rule'] = option_rule.__name__[-5:]

        else:  
            option_name = log_dict[str(i)]['deadcode_api_name']
            option_rule_name = log_dict[str(i)]['deadcode_api_rule']

            for node in graph.nodes:
                if node.name == option_name:
                    module_name = node.target
                    module_instance = new_module.get_submodule(module_name)  
                    option_node, option_instance, option_name = node, module_instance, node.name
                    break

            option_rule = match_rule(option_rule_name)
            new_instance = rule_reflect_class(option_rule,option_instance)(option_instance)  

            print('ABSOC_B：选择对其变异的node：', option_node)
            print('ABSOC_B：选择对其变异的node的名字：', option_name)
            print('ABSOC_B：选择对其变异的instance：', option_instance)
            print('ABSOC_B：选择的变异规则：', option_rule)
            print('ABSOC_B：变异后新的层：', new_instance)

        
        

        
        new_name = reflect_name(option_name, option_rule)

        
        new_module.add_module(new_name, new_instance)
        
        with option_node.graph.inserting_after(option_node):
            new_node = option_node.graph.call_module(new_name, args=option_node.args)
            option_node.replace_all_uses_with(new_node)
            new_module.graph.erase_node(option_node)
        
        graph.lint()
        new_module.recompile()
        

        return new_module

    def forward(self, a, b, deada, deadb):
        if isinstance(a, tuple) or isinstance(b, tuple) or isinstance(deada, tuple) or isinstance(deadb, tuple):
            print("mutate failed for input tuple")
            print("type a", type(a))
            print("type b", type(b))
            print("type deada", type(deada))
            print("type deadb", type(deadb))
            return a, b, deada, deadb
        
        
        
        
        
        
        a, b = handle_shape_strict(a, b)
        
        
        a1 = torch.abs(a)
        b1 = torch.abs(b)
        a1b1 = torch.sub(a1, b1)  
        a1b1 = torch.abs(a1b1)  
        ab = torch.abs(torch.add(a, b))  
        absoc_b = torch.sub(a1b1, ab)  
        absoc_b = torch.sub(absoc_b, 1e-5)  
        
        absoc_b = random.choice([nn.ReLU(), nn.ReLU6(), nn.Hardtanh()])(absoc_b)  
        dead = make_unsqueeze(deada)
        add_edge = self.op_layer(dead)
        absoc_b, add_edge = handle_shape_strict(absoc_b, add_edge)
        
        out = torch.mul(absoc_b, add_edge)
        dtype = deadb.dtype
        out, deadb = handle_shape_final(out, deadb)
        out = torch.add(out, deadb)
        
        out = out.to(dtype)
        
        return out


class PIOC(nn.Module):
    def __init__(self, op_type, operator_mutation_type, log_dict, i, LOG_FLAG):
        super().__init__()
        self.__name__ = "PIOC"
        self.op_type = op_type
        if self.op_type == "SELayer":
            layer = SELayer()
        if self.op_type == "DenseLayer":
            layer = DenseLayer()
        if self.op_type == "Inception_A":
            layer = Inception_A()
        if self.op_type == "PWDWPW_ResidualBlock":
            layer = PWDWPW_ResidualBlock()
        if self.op_type == "ResidualBlock":
            layer = ResidualBlock()
        if self.op_type == "DropPath":
            layer = DropPath()
        if self.op_type == "Dense":
            layer = Dense()

        if operator_mutation_type == 'deadcode':
            self.op_layer = self.modify_layer_with_fx(layer, log_dict, i, LOG_FLAG)  
        else:
            self.op_layer = layer

    def modify_layer_with_fx(self, layer, log_dict, i, LOG_FLAG):
        new_module = torch.fx.symbolic_trace(layer)
        graph = new_module.graph

        if LOG_FLAG == False:
            option_layers = list()  
            for node in graph.nodes:
                if node.op == 'call_module' and '_mutate' not in node.name:  
                    module_name = node.target
                    module_instance = new_module.get_submodule(module_name)  
                    option_layers.append((node, module_instance, node.name))
            

            if len(option_layers) != 0:
                option_node, option_instance, option_name = random.choice(option_layers)  
                option_rule = random.choice(rules_dict[type(option_instance)])  
                new_instance = rule_reflect_class(option_rule,option_instance)(option_instance)  

                print('PIOC：选择对其变异的node：', option_node)
                print('PIOC：选择对其变异的node的名字：', option_name)
                print('PIOC：选择对其变异的instance：', option_instance)
                print('PIOC：选择的变异规则：', option_rule)
                print('PIOC：变异后新的层：', new_instance)

                log_dict[i]['deadcode_api_name'] = option_name
                if option_rule.__name__[-2:] in ('10', '11', '12', '13', '14', '15', '16', '17', '18'):
                    log_dict[i]['deadcode_api_rule'] = option_rule.__name__[-6:]
                else:
                    log_dict[i]['deadcode_api_rule'] = option_rule.__name__[-5:]

        else:  
            option_name = log_dict[str(i)]['deadcode_api_name']
            option_rule_name = log_dict[str(i)]['deadcode_api_rule']

            for node in graph.nodes:
                if node.name == option_name:
                    module_name = node.target
                    module_instance = new_module.get_submodule(module_name)  
                    option_node, option_instance, option_name = node, module_instance, node.name
                    break

            option_rule = match_rule(option_rule_name)
            new_instance = rule_reflect_class(option_rule,option_instance)(option_instance)  

            print('PIOC：选择对其变异的node：', option_node)
            print('PIOC：选择对其变异的node的名字：', option_name)
            print('PIOC：选择对其变异的instance：', option_instance)
            print('PIOC：选择的变异规则：', option_rule)
            print('PIOC：变异后新的层：', new_instance)

        
        

        
        new_name = reflect_name(option_name, option_rule)

        
        new_module.add_module(new_name, new_instance)
        
        with option_node.graph.inserting_after(option_node):
            new_node = option_node.graph.call_module(new_name, args=option_node.args)
            option_node.replace_all_uses_with(new_node)
            new_module.graph.erase_node(option_node)
        
        graph.lint()
        new_module.recompile()
        

        return new_module

    def forward(self, input, deada, final_dead, deadb=None):
        if isinstance(input, tuple) or isinstance(deada, tuple) or isinstance(final_dead, tuple):
            print("mutate failed for input tuple")
            print("type a", type(input))
            print("type b", type(deada))
            print("type c", type(final_dead))
            return input, deada, final_dead
        reduce_edge = make_reduce(input)
        dtype = reduce_edge.dtype
        
        if torch.get_device(reduce_edge) != "cpu":
            const_np = reduce_edge.detach().cpu().numpy()
        else:
            const_np = reduce_edge.detach().numpy()
        const_edge = torch.tensor(const_np, dtype=dtype).to(device)
        sub_edge = torch.sub(reduce_edge, const_edge)

        deada, _ = handle_shape_strict(deada, deada)
        deada = make_unsqueeze(deada)
        add_edge = self.op_layer(deada)
        sub_edge, add_edge = handle_shape_strict(sub_edge, add_edge)
        mul_edge = torch.mul(add_edge, sub_edge)

        mul_edge_1, final_dead_1 = handle_shape_final(mul_edge, final_dead)
        
        
        out = torch.add(mul_edge_1, final_dead_1)
        pioc_equal = np.allclose(out.detach().cpu().numpy(), final_dead_1.detach().cpu().numpy())
        if not pioc_equal:
            print("pioc不相等！")
        assert pioc_equal
        
        
        return out
