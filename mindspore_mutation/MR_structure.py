import mindspore
import mindspore.ops as ops
import numpy as np
from mindspore import nn
import random
from mindspore_mutation.deadcode import SELayer, Inception_A, DenseLayer, DropPath, Dense, ResidualBlock, PWDWPW_ResidualBlock
from mindspore_mutation.handel_shape import handle_shape_final, handle_shape_strict, make_unsqueeze, make_reduce
from cargo import match_rule,reflect_name,rule_reflect_class
import config
import collections
from mindspore.rewrite.node import NodeManager
from mindspore.rewrite import ScopedValue, NodeType
from mindspore.rewrite import SymbolTree

banned_ops = [mindspore.ops.operations.array_ops.Shape,
              mindspore.ops.operations.array_ops.Concat,
              mindspore.ops.operations.array_ops.TupleToArray,
              mindspore.ops.operations.array_ops.Reshape,
              mindspore.ops.operations.array_ops.Tile,
              type(None)
              ]
banned_cell = [mindspore.nn.layer.CentralCrop, ]
banned_trees = [mindspore.ops.ResizeBilinear,
                mindspore.ops.operations.Shape,
                type(None)
                ]

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

rules_dict = config.rules_dict 
class UOC(nn.Cell):
    def __init__(self, op_type,operator_mutation_type,log_dict,i,LOG_FLAG):
        super().__init__()
        self.__name__ = "UOC"
        self.op_type = op_type
        if self.op_type == "SELayer":
            self.op_layer = SELayer()
        if self.op_type == "DenseLayer":
            self.op_layer = DenseLayer()
        if self.op_type == "Inception_A":
            self.op_layer = Inception_A()
        if self.op_type == "PWDWPW_ResidualBlock":
            self.op_layer = PWDWPW_ResidualBlock()
        if self.op_type == "ResidualBlock":
            self.op_layer = ResidualBlock()
        if self.op_type == "DropPath":
            self.op_layer = DropPath()
        if self.op_type == "Dense":
            self.op_layer = Dense()

        if operator_mutation_type == 'deadcode':
            self.op_layer = self.modify_layer_with_fx(self.op_layer, log_dict, i, LOG_FLAG)  
        else:
            self.op_layer = self.op_layer

    def modify_layer_with_fx(self, layer, log_dict, i, LOG_FLAG):
        
        hash_table = collections.defaultdict(int)
        nodedict = collections.OrderedDict()
        if LOG_FLAG == False:
            option_layers = list()  

            
            for name, child in layer.cells_and_names():
                if not has_child_node(layer,name) and not name == '' and not 'deadcode' in str(type(child)):
                    
                    option_layers.append((name, child, name))

            if len(option_layers) != 0:
                option_name, option_instance, option_node_name = random.choice(option_layers)  
                option_rule = random.choice(rules_dict[type(option_instance)])  
                new_instance = rule_reflect_class(option_rule, option_instance)(option_instance)  

                print('UOC:node', option_node_name)
                print('UOC:node name', option_name)
                print('UOC:instance', option_instance)
                print('UOC:rule', option_rule)
                print('UOC:new layer', new_instance)

                log_dict[i]['deadcode_api_name'] = option_name
                if option_rule.__name__[-2:] in ('10', '11', '12', '13', '14', '15', '16', '17', '18'):
                    log_dict[i]['deadcode_api_rule'] = option_rule.__name__[-6:]
                else:
                    log_dict[i]['deadcode_api_rule'] = option_rule.__name__[-5:]

                
                layer._cells[option_name] = new_instance

        else:  
            try:
                option_name = log_dict[str(i)]['deadcode_api_name']
                option_rule_name = log_dict[str(i)]['deadcode_api_rule']

                option_instance = layer._cells.get(option_name, None)
                if option_instance is not None:
                    option_rule = match_rule(option_rule_name)
                    new_instance = rule_reflect_class(option_rule, option_instance)(option_instance)  

                    print('UOC:option_name', option_name)
                    print('UOC:instance', option_instance)
                    print('UOC:rule', option_rule)
                    print('UOC:new layer', new_instance)

                    
                    layer._cells[option_name] = new_instance
            except:
                print("")

        
        layer.update_parameters_name()
        return layer


    def construct(self, a, b, deada, deadb):
        
        if isinstance(a, tuple) or isinstance(b, tuple) or isinstance(deada, tuple) or isinstance(deadb, tuple):
            print("mutate failed for input tuple")
            return a, b, deada, deadb
        
        
        a, b = handle_shape_strict(a, b)
        
        
        a2 = ops.mul(a, a)
        
        b2 = ops.mul(b, b)
        
        ab = ops.mul(a, b)
        
        ab2 = ops.mul(ab, -2)
        
        uoc = ops.add(a2, b2)
        
        uoc = ops.add(uoc, ab2)
            
        uoc = ops.add(uoc, 1e-10)
        
        uoc = ops.neg(uoc)
        
        uoc = ops.ReLU()(uoc)
        
        dead = make_unsqueeze(deada)
        add_edge = self.op_layer(dead)
        uoc, add_edge = handle_shape_strict(uoc, add_edge)
        
        out = ops.mul(uoc, add_edge)
        dtype = deadb.dtype1
        out, deadb = handle_shape_final(out, deadb)
        out = ops.add(out, deadb)
        out = out.to(dtype)
        return out


class PIOC(nn.Cell):
    def __init__(self, op_type,operator_mutation_type,log_dict,i,LOG_FLAG):
        super().__init__()
        self.__name__ = "PIOC"
        self.op_type = op_type
        if self.op_type == "SELayer":
            self.op_layer = SELayer()
        if self.op_type == "DenseLayer":
            self.op_layer = DenseLayer()
        if self.op_type == "Inception_A":
            self.op_layer = Inception_A()
        if self.op_type == "PWDWPW_ResidualBlock":
            self.op_layer = PWDWPW_ResidualBlock()
        if self.op_type == "ResidualBlock":
            self.op_layer = ResidualBlock()
        if self.op_type == "DropPath":
            self.op_layer = DropPath()
        if self.op_type == "Dense":
            self.op_layer = Dense()

        if operator_mutation_type == 'deadcode':
            self.op_layer = self.modify_layer_with_fx(self.op_layer, log_dict, i, LOG_FLAG)  
        else:
            self.op_layer = self.op_layer

    def modify_layer_with_fx(self, layer, log_dict, i, LOG_FLAG):
        
        hash_table = collections.defaultdict(int)
        nodedict = collections.OrderedDict()
        if LOG_FLAG == False:
            option_layers = list()  

            
            for name, child in layer.cells_and_names():
                if not has_child_node(layer,name) and not name == '' and not 'deadcode' in str(type(child)):
                    option_layers.append((name, child, name))

            if len(option_layers) != 0:
                option_name, option_instance, option_node_name = random.choice(option_layers)  
                option_rule = random.choice(rules_dict[type(option_instance)])  
                new_instance = rule_reflect_class(option_rule, option_instance)(option_instance)  

                print('PIOC:node', option_node_name)
                print('PIOC:node name', option_name)
                print('PIOC:instance', option_instance)
                print('PIOC:rule', option_rule)
                print('PIOC:new layer', new_instance)

                log_dict[i]['deadcode_api_name'] = option_name
                if option_rule.__name__[-2:] in ('10', '11', '12', '13', '14', '15', '16', '17', '18'):
                    log_dict[i]['deadcode_api_rule'] = option_rule.__name__[-6:]
                else:
                    log_dict[i]['deadcode_api_rule'] = option_rule.__name__[-5:]

                
                layer._cells[option_name] = new_instance

        else:  
            try:
                option_name = log_dict[str(i)]['deadcode_api_name']
                option_rule_name = log_dict[str(i)]['deadcode_api_rule']

                option_instance = layer._cells.get(option_name, None)
                if option_instance is not None:
                    option_rule = match_rule(option_rule_name)
                    new_instance = rule_reflect_class(option_rule, option_instance)(option_instance)  

                    print('uoc:option_name', option_name)
                    print('uoc:instance', option_instance)
                    print('uoc:rule', option_rule)
                    print('uoc:new layer', new_instance)

                    
                    layer._cells[option_name] = new_instance
            except:
                print("")

        
        layer.update_parameters_name()
        return layer

    def construct(self, input, deada, final_dead, deadb=None):
        reduce_edge = make_reduce(input)
        dtype = reduce_edge.dtype
        
        const_np = reduce_edge.asnumpy()
        
        const_edge = mindspore.Tensor(const_np, dtype)
        sub_edge = ops.Sub()(reduce_edge, const_edge)
        if self.op_type == "Add":
            deada, deadb = handle_shape_strict(deada, deadb)
            add_edge = ops.Add()(deada, deadb)
            
            
            add_edge, sub_edge = handle_shape_strict(add_edge, sub_edge)
            
            
            mul_edge = ops.Mul()(add_edge, sub_edge)
            
        elif self.op_type in ["DenseLayer", "SELayer", "Inception_A", "PWDWPW_ResidualBlock",
                              "ResidualBlock", "DropPath", "Dense"]:
            deada, _ = handle_shape_strict(deada, deada)
            deada = make_unsqueeze(deada)
            add_edge = self.op_layer(deada)
            sub_edge, add_edge = handle_shape_strict(sub_edge, add_edge)
            mul_edge = ops.Mul()(add_edge, sub_edge)
        else:
            raise NotImplementedError("optype Not Implemented for optype: {}".format(self.op_type))

        mul_edge_1, final_dead_1 = handle_shape_final(mul_edge, final_dead             
        out = ops.Add()(mul_edge_1, final_dead_1)        
        return out

class ABSOC_A(nn.Cell):
    def __init__(self, op_type,operator_mutation_type,log_dict,i,LOG_FLAG):
        super().__init__()
        self.__name__ = "uoc"
        self.op_type = op_type
        if self.op_type == "Add":
            self.op_layer = SELayer()
        if self.op_type == "SELayer":
            self.op_layer = SELayer()
        if self.op_type == "DenseLayer":
            self.op_layer = DenseLayer()
        if self.op_type == "Inception_A":
            self.op_layer = Inception_A()
        if self.op_type == "PWDWPW_ResidualBlock":
            self.op_layer = PWDWPW_ResidualBlock()
        if self.op_type == "ResidualBlock":
            self.op_layer = ResidualBlock()
        if self.op_type == "DropPath":
            self.op_layer = DropPath()
        if self.op_type == "Dense":
            self.op_layer = Dense()
        if operator_mutation_type == 'deadcode':
            self.op_layer = self.modify_layer_with_fx(self.op_layer, log_dict, i, LOG_FLAG)  
        else:
            self.op_layer = self.op_layer

    def modify_layer_with_fx(self, layer, log_dict, i, LOG_FLAG):
        
        hash_table = collections.defaultdict(int)
        nodedict = collections.OrderedDict()
        if LOG_FLAG == False:
            option_layers = list()  

            
            for name, child in layer.cells_and_names():
                if not has_child_node(layer,name) and not name == '' and not 'deadcode' in str(type(child)):
                    option_layers.append((name, child, name))

            if len(option_layers) != 0:
                option_name, option_instance, option_node_name = random.choice(option_layers)  
                option_rule = random.choice(rules_dict[type(option_instance)])  
                new_instance = rule_reflect_class(option_rule, option_instance)(option_instance)  

                print('ABSOC_A:node', option_node_name)
                print('ABSOC_A:node name', option_name)
                print('ABSOC_A:instance', option_instance)
                print('ABSOC_A:rule', option_rule)
                print('ABSOC_A:new layer', new_instance)

                log_dict[i]['deadcode_api_name'] = option_name
                if option_rule.__name__[-2:] in ('10', '11', '12', '13', '14', '15', '16', '17', '18'):
                    log_dict[i]['deadcode_api_rule'] = option_rule.__name__[-6:]
                else:
                    log_dict[i]['deadcode_api_rule'] = option_rule.__name__[-5:]

                
                layer._cells[option_name] = new_instance

        else:  
            try:
                option_name = log_dict[str(i)]['deadcode_api_name']
                option_rule_name = log_dict[str(i)]['deadcode_api_rule']

                option_instance = layer._cells.get(option_name, None)
                if option_instance is not None:
                    option_rule = match_rule(option_rule_name)
                    new_instance = rule_reflect_class(option_rule, option_instance)(option_instance)  

                    print('ABSOC_A:option_name', option_name)
                    print('ABSOC_A:instance', option_instance)
                    print('ABSOC_A:rule', option_rule)
                    print('ABSOC_A:new layer', new_instance)

                    
                    layer._cells[option_name] = new_instance
            except:
                print("")
        
        layer.update_parameters_name()
        return layer

    def construct(self, a, b, deada, deadb):
        
        if isinstance(a, tuple) or isinstance(b, tuple) or isinstance(deada, tuple) or isinstance(deadb, tuple):
            print("mutate failed for input tuple")
            print("type a", type(a))
            print("type b", type(b))
            print("type deada", type(deada))
            print("type deadb", type(deadb))
            return a, b, deada, deadb
 
        a, b = handle_shape_strict(a, b)

        a1 = mindspore.ops.abs(a)
        b1 = mindspore.ops.abs(b)
        a1b1 = mindspore.ops.add(a1, b1)  
        ab = mindspore.ops.abs(mindspore.ops.add(a, b))  
        absoc_a = mindspore.ops.sub(a1b1, ab)  
        absoc_a = mindspore.ops.add(absoc_a, 1e-10)  
        absoc_a = mindspore.ops.neg(absoc_a)  
        absoc_a = mindspore.nn.ReLU()(absoc_a)  
        
        dead = make_unsqueeze(deada)
        add_edge = self.op_layer(dead)
        absoc_a, add_edge = handle_shape_strict(absoc_a, add_edge)
        
        out = mindspore.ops.mul(absoc_a, add_edge)
        dtype = deada.dtype
        out, deada = handle_shape_final(out, deada)
        out = mindspore.ops.add(out, deada)
        
        out = out.to(dtype)
        
        return out

class ABSOC_B(nn.Cell):
    def __init__(self, op_type,operator_mutation_type,log_dict,i,LOG_FLAG):
        super().__init__()
        self.__name__ = "uoc"
        self.op_type = op_type
        if self.op_type == "Add":
            self.op_layer = SELayer()
        if self.op_type == "SELayer":
            self.op_layer = SELayer()
        if self.op_type == "DenseLayer":
            self.op_layer = DenseLayer()
        if self.op_type == "Inception_A":
            self.op_layer = Inception_A()
        if self.op_type == "PWDWPW_ResidualBlock":
            self.op_layer = PWDWPW_ResidualBlock()
        if self.op_type == "ResidualBlock":
            self.op_layer = ResidualBlock()
        if self.op_type == "DropPath":
            self.op_layer = DropPath()
        if self.op_type == "Dense":
            self.op_layer = Dense()


        if operator_mutation_type == 'deadcode':
            self.op_layer = self.modify_layer_with_fx(self.op_layer, log_dict, i, LOG_FLAG)  
        else:
            self.op_layer = self.op_layer

    def modify_layer_with_fx(self, layer, log_dict, i, LOG_FLAG):
        
        hash_table = collections.defaultdict(int)
        nodedict = collections.OrderedDict()
        if LOG_FLAG == False:
            option_layers = list()  

            
            for name, child in layer.cells_and_names():
                if not has_child_node(layer,name) and not name == '' and not 'deadcode' in str(type(child)):
                    option_layers.append((name, child, name))

            if len(option_layers) != 0:
                option_name, option_instance, option_node_name = random.choice(option_layers)  
                option_rule = random.choice(rules_dict[type(option_instance)])  
                new_instance = rule_reflect_class(option_rule, option_instance)(option_instance)  

                print('ABSOC_B:node', option_node_name)
                print('ABSOC_B:node name', option_name)
                print('ABSOC_B:instance', option_instance)
                print('ABSOC_B:rule', option_rule)
                print('ABSOC_B:new layer', new_instance)

                log_dict[i]['deadcode_api_name'] = option_name
                if option_rule.__name__[-2:] in ('10', '11', '12', '13', '14', '15', '16', '17', '18'):
                    log_dict[i]['deadcode_api_rule'] = option_rule.__name__[-6:]
                else:
                    log_dict[i]['deadcode_api_rule'] = option_rule.__name__[-5:]

                
                layer._cells[option_name] = new_instance

        else:  
            try:
                option_name = log_dict[str(i)]['deadcode_api_name']
                option_rule_name = log_dict[str(i)]['deadcode_api_rule']

                option_instance = layer._cells.get(option_name, None)
                if option_instance is not None:
                    option_rule = match_rule(option_rule_name)
                    new_instance = rule_reflect_class(option_rule, option_instance)(option_instance)  

                    print('ABSOC_B:option_name', option_name)
                    print('ABSOC_B:instance', option_instance)
                    print('ABSOC_B:rule', option_rule)
                    print('ABSOC_B:new layer', new_instance)

                    
                    layer._cells[option_name] = new_instance
            except:
                print("")
        
        layer.update_parameters_name()
        return layer


    def construct(self, a, b, deada, deadb):
        
        if isinstance(a, tuple) or isinstance(b, tuple) or isinstance(deada, tuple) or isinstance(deadb, tuple):
            print("mutate failed for input tuple")
            print("type a", type(a))
            print("type b", type(b))
            print("type deada", type(deada))
            print("type deadb", type(deadb))
            return a, b, deada, deadb

        a, b = handle_shape_strict(a, b)

        a1 = mindspore.ops.abs(a)
        b1 = mindspore.ops.abs(b)
        a1b1 = mindspore.ops.subtract(a1, b1)  
        a1b1 = mindspore.ops.abs(a1b1)  
        ab = mindspore.ops.abs(mindspore.ops.add(a, b))  
        absoc_b = mindspore.ops.subtract(a1b1, ab)  
        absoc_b = mindspore.ops.subtract(absoc_b, 1e-5)  
        absoc_b = mindspore.nn.ReLU()(absoc_b)  
        
        dead = make_unsqueeze(deada)
        add_edge = self.op_layer(dead)
        absoc_b, add_edge = handle_shape_strict(absoc_b, add_edge)
        
        out = mindspore.ops.mul(absoc_b, add_edge)
        dtype = deada.dtype
        out, deada = handle_shape_final(out, deada)
        out = mindspore.ops.add(out, deada)
        
        out = out.to(dtype)
        
        return out
