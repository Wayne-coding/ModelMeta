import copy
import os
import numpy as np
from mindspore_mutation.rules_ms import rule1,rule2,rule3,rule4,rule5,rule6,rule7,rule8,rule9,\
                                        rule10,rule11,rule12,rule13,rule14,rule15,rule16,rule17,rule18
import torch
import random
import pennylane as qml
import torch.nn as nn
import scipy.io as scio
import torch.fx as fx
import psutil
import mindspore
from mindspore.rewrite import ScopedValue, NodeType
from mindspore.rewrite.node import Node, NodeManager
import mindspore as ms

from models.vgg16.vgg16_torch import vgg16 

from models.UNet.Unet import UNetMedical, create_Unet_dataset

from models.resnet50.resnet50 import resnet50, create_cifar10_dataset, update_params

from models.ssimae.src.network import AutoEncoder as AutoEncoder_ms
from models.ssimae.model_utils.config import config as ssimae_cfg


from models.textcnn.dataset import MovieReview
from models.textcnn.textcnn import TextCNN



ms.set_context(mode=ms.PYNATIVE_MODE, device_target='GPU')


datasets_path_cargo = {
    "vgg16": "/data1/czx/semtest/data/cifar10_x.npy", 
 
    "resnet": "/data1/czx/semtest/data/ResNet_data0.npy",  
    
    "SSDresnet50fpn":"/data1/czx/semtest/data/ssdresnet50_data.npy",
    
    "UNetMedical": "/data1/czx/semtest/data/UNetMedical_data0.npy", 
    
    "TextCNN": "/data1/czx/semtest/data/TextCNN_data0.npy",  

    "ssimae": "/data1/czx/semtest/data/ssimae_data0.npy", 
    
    "ssimae":"/data1/czx/semtest/data/ssimae.npy",
    
}


net_cargo = {
    "resnet": resnet50,
    "UNetMedical": UNetMedical,
    "TextCNN": TextCNN,
    
    
}


def get_model(model_name):
    net_cargo = {
        "resnet": resnet50,
        "UNetMedical": UNetMedical,
        "TextCNN": TextCNN,
        "ssimae":AutoEncoder_ms,
    }

    
    if model_name == "resnet":
        model = net_cargo[model_name]()
        return model
    elif model_name == "UNetMedical":
        model = net_cargo[model_name](n_channels=1, n_classes=2)
        return model
    
    elif model_name == "TextCNN":
        model = net_cargo[model_name](vocab_len=20305, word_len=51, num_classes=2, vec_length=40)
        return model
    
    elif model_name == "ssimae":
        model = net_cargo[model_name](ssimae_cfg)
        return model
    

if __name__ == '__main__':
    
    model_name = "CRNN"
    model=get_model(model_name,"cpu")
    print(model)
    d=fx.symbolic_trace(model)
    print("-*"*10)
    print(d)

    option_layers = list()  
    nn_types=set()
    for node in d.graph.nodes:
        if node.op == 'call_module' and '_mutate' not in node.name:  
            module_name = node.target
            module_instance = d.get_submodule(module_name)  
            option_layers.append((node, module_instance, node.name))
            nn_types.add(type(module_instance))
    print(option_layers)
    print(len(option_layers))
    print(nn_types)


def loss_unet_ms():
    from models.UNet.Unet import CrossEntropyWithLogits
    return CrossEntropyWithLogits()


def loss_unet_torch():
    from models.UNet.main_torch import CrossEntropyWithLogits
    return CrossEntropyWithLogits()

def loss_textcnn_ms():
    from models.textcnn.run_textcnn import loss_com_ms
    return loss_com_ms


def loss_textcnn_torch():
    from models.textcnn.run_textcnn_torch import loss_com
    return loss_com()


def loss_ssimae_ms():
    from models.ssimae.src.network import SSIMLoss
    return SSIMLoss()


def loss_ssimae_torch():
    from models.ssimae.src.network_torch import SSIMLoss as SSIMLoss_torch
    return SSIMLoss_torch()


def get_loss(loss_name):
    loss = {}
    loss['CrossEntropy'] = [mindspore.nn.CrossEntropyLoss, torch.nn.CrossEntropyLoss]
    loss['unetloss'] = [loss_unet_ms, loss_unet_torch]
    loss['textcnnloss'] = [loss_textcnn_ms, loss_textcnn_torch]
    loss['ssimae'] = [loss_ssimae_ms, loss_ssimae_torch]
    return loss[loss_name]


def max_seed_model_api_times(model_name):
    if model_name == "vgg16": 
        return 39
    elif model_name == "resnet": 
        return 158
    elif model_name == "UNetMedical":
        return 49
    elif model_name == "TextCNN": 
        return 12
    elif model_name == "ssimae":
        return 39


def reflect_name(option_name,option_rule):
    if option_rule is rule1:
        new_name = option_name + "_mutated_rule1"
    elif option_rule is rule2:
        new_name = option_name + "_mutated_rule2"
    elif option_rule is rule3:
        new_name = option_name + "_mutated_rule3"
    elif option_rule is rule4:
        new_name = option_name + "_mutated_rule4"
    elif option_rule is rule5:
        new_name = option_name + "_mutated_rule5"
    elif option_rule is rule6:
        new_name = option_name + "_mutated_rule6"
    elif option_rule is rule7:
        new_name = option_name + "_mutated_rule7"
    elif option_rule is rule8:
        new_name = option_name + "_mutated_rule8"
    elif option_rule is rule9:
        new_name = option_name + "_mutated_rule9"
    elif option_rule is rule10:
        new_name = option_name + "_mutated_rule10"
    elif option_rule is rule11:
        new_name = option_name + "_mutated_rule11"
    elif option_rule is rule12:
        new_name = option_name + "_mutated_rule12"
    elif option_rule is rule13:
        new_name = option_name + "_mutated_rule13"
    elif option_rule is rule14:
        new_name = option_name + "_mutated_rule14"
    elif option_rule is rule15:
        new_name = option_name + "_mutated_rule15"
    elif option_rule is rule16:
        new_name = option_name + "_mutated_rule16"
    elif option_rule is rule17:
        new_name = option_name + "_mutated_rule17"
    elif option_rule is rule18:
        new_name = option_name + "_mutated_rule18"
    return new_name

def match_rule(option_rule_name): 
    match_rule_dict = {
        'rule1': rule1,
        'rule2': rule2,
        'rule3': rule3,
        'rule4': rule4,
        'rule5': rule5,
        'rule6': rule6,
        'rule7': rule7,
        'rule8': rule8,
        'rule9': rule9,
        'rule10': rule10,
        'rule11': rule11,
        'rule12': rule12,
        'rule13': rule13,
        'rule14': rule14,
        'rule15': rule15,
        'rule16': rule16,
        'rule17': rule17,
        'rule18': rule18
    }
    return match_rule_dict.get(option_rule_name, None)


def rule_reflect_class(option_rule, option_instance):
    if option_rule is rule1:
        if isinstance(option_instance, ms.nn.Conv2d):
            return rule1.TransLayerRule1Conv2d
        elif isinstance(option_instance, ms.nn.AvgPool2d):
            return rule1.TransLayerRule1AvgPool2d
        elif isinstance(option_instance, ms.nn.MaxPool2d):
            return rule1.TransLayerRule1MaxPool2d
        elif isinstance(option_instance, ms.nn.ReLU):
            return rule1.TransLayerRule1ReLU
        elif isinstance(option_instance, ms.nn.ReLU6):
            return rule1.TransLayerRule1ReLU6
        elif isinstance(option_instance, ms.nn.BatchNorm2d):
            return rule1.TransLayerRule1BatchNorm2d
        
        
        elif isinstance(option_instance, ms.nn.Flatten):
            return rule1.TransLayerRule1Flatten
        elif isinstance(option_instance, ms.nn.HSigmoid):
            return rule1.TransLayerRule1Hardsigmoid
        elif isinstance(option_instance, ms.nn.Sigmoid):
            return rule1.TransLayerRule1Sigmoid
        elif isinstance(option_instance, ms.nn.Softmax):
            return rule1.TransLayerRule1Softmax
        elif isinstance(option_instance, ms.nn.Tanh):
            return rule1.TransLayerRule1Tanh
        elif isinstance(option_instance, ms.nn.Conv2dTranspose):
            return rule1.TransLayerRule1ConvTranspose2d
        elif isinstance(option_instance, ms.nn.LeakyReLU):
            return rule1.TransLayerRule1LeakyReLU
        elif isinstance(option_instance, ms.nn.AdaptiveAvgPool2d):
            return rule1.TransLayerRule1AdaptiveAvgPool2d
        elif isinstance(option_instance, ms.nn.Dropout):
            return rule1.TransLayerRule1Dropout
        elif isinstance(option_instance, ms.nn.Embedding):
            return rule1.TransLayerRule1Embedding
        elif isinstance(option_instance, ms.nn.LSTM):
            return rule1.TransLayerRule1LSTM
    elif option_rule is rule2:
        return rule2.TransLayerRule2
    elif option_rule is rule3:
        if isinstance(option_instance, ms.nn.Conv2d):
            return rule3.Conv2dToConv3d
        elif isinstance(option_instance, ms.nn.AvgPool2d):
            return rule3.MaxPool2dToMaxPool3d
        elif isinstance(option_instance, ms.nn.MaxPool2d):
            return rule3.AvgPool2dToAvgPool3d
    elif option_rule is rule4:
        return rule4.TransLayerRule4
    elif option_rule is rule5:
        return rule5.TransLayerRule5
    elif option_rule is rule6:
        return rule6.TransLayerRule6
    elif option_rule is rule7:
        return rule7.TransLayerRule7
    elif option_rule is rule8:
        return rule8.TransLayerRule8
    elif option_rule is rule9:
        return rule9.TransLayerRule9
    elif option_rule is rule10:
        return rule10.TransLayerRule10
    elif option_rule is rule11:
        return rule11.TransLayer_rule11
    elif option_rule is rule12:
        if isinstance(option_instance, ms.nn.AdaptiveAvgPool2d):
            return rule12.TransLayerRule12AdaptiveAvgPool2d
        elif isinstance(option_instance, ms.nn.AvgPool2d):
            return rule12.TransLayerRule12AvgPool2d
        elif isinstance(option_instance, ms.nn.MaxPool2d):
            return rule12.TransLayerRule12MaxPool2d
    elif option_rule is rule13:
        if isinstance(option_instance, ms.nn.AdaptiveAvgPool2d):
            return rule13.TransLayer_rule13_AdaptiveAvgPool2d
        elif isinstance(option_instance, ms.nn.AvgPool2d):
            return rule13.TransLayer_rule13_AvgPool2d
        elif isinstance(option_instance, ms.nn.MaxPool2d):
            return rule13.TransLayer_rule13_MaxPool2d
    elif option_rule is rule14:
        if isinstance(option_instance, ms.nn.AdaptiveAvgPool2d):
            return rule14.TransLayer_rule14_AdaptiveAvgPool2d
        elif isinstance(option_instance, ms.nn.AvgPool2d):
            return rule14.TransLayer_rule14_AvgPool2d
        elif isinstance(option_instance, ms.nn.MaxPool2d):
            return rule14.TransLayer_rule14_MaxPool2d
    elif option_rule is rule15:
        if isinstance(option_instance, ms.nn.ReLU):
            return rule15.TransLayerRule15ReLU
        elif isinstance(option_instance, ms.nn.LeakyReLU):
            return rule15.TransLayerRule15LeakyReLU
    elif option_rule is rule16:
        return rule16.TransLayer_rule16


def select_places(sequence, k):
    for i in range(5):
        try:
            chosen = random.choices(sequence, k=k)
        except Exception as e:
            
            
            return None, None
        subs_place = max(chosen)
        chosen.remove(subs_place)
        if max(chosen) != subs_place:
            
            
            return subs_place, chosen
    
    return None, None


np.random.seed(20200501)
class MCMC:
    class Mutator:
        def __init__(self, name, total=0, delta_bigger_than_zero=0, epsilon=1e-7):
            self.name = name
            self.total = total
            self.delta_bigger_than_zero = delta_bigger_than_zero
            self.epsilon = epsilon

        @property
        def score(self, epsilon=1e-7):
            rate = self.delta_bigger_than_zero / (self.total + epsilon)
            return rate

    def __init__(self, mutate_ops=['UOC', 'PIOC', 'ABSOC_A', 'ABSOC_B']):
        self.p = 1 / len(mutate_ops)
        self._mutators = [self.Mutator(name=op) for op in mutate_ops]

    @property
    def mutators(self):
        mus = {}
        for mu in self._mutators:
            mus[mu.name] = mu
        return mus

    def choose_mutator(self, mu1=None):
        if mu1 is None: 
            return self._mutators[np.random.randint(0, len(self._mutators))].name
        else:
            self.sort_mutators() 
            k1 = self.index(mu1) 
            k2 = -1
            prob = 0
            while np.random.rand() >= prob:
                k2 = np.random.randint(0, len(self._mutators))
                prob = (1 - self.p) ** (k2 - k1)
            mu2 = self._mutators[k2]
            return mu2.name

    def sort_mutators(self):
        random.shuffle(self._mutators)
        self._mutators.sort(key=lambda mutator: mutator.score, reverse=True)

    def index(self, mutator_name):
        for i, mu in enumerate(self._mutators):
            if mu.name == mutator_name:
                return i
        return -1



n_qubits = 4  
n_layers = 2  
dev = qml.device("default.qubit", wires=n_qubits)


def quantum_circuit(params, inputs):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.BasicEntanglerLayers(params, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


weight_shapes = {"params": (n_layers, n_qubits)}
quantum_layer = qml.qnn.TorchLayer(qml.QNode(quantum_circuit, dev), weight_shapes)


class QRDQN(nn.Module):
    def __init__(self, state_dim, n_actions):
        super(QRDQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, n_qubits)  
        self.quantum_layer = quantum_layer
        self.fc3 = nn.Linear(n_qubits, n_actions)

    def forward(self, state):
        x = torch.relu(self.fc1(state.view(1, -1)))
        x = torch.relu(self.fc2(x))
        x = self.quantum_layer(x)
        q_values = self.fc3(x)
        return q_values
import pynvml
pynvml.nvmlInit()
def compute_gpu_cpu(): 
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    
    mem = psutil.virtual_memory()  
    gpu_memory = float(memory_info.used) / (1024 ** 2)
    cpu_memory = float(mem.available) / 1024 / 1024
    return gpu_memory,cpu_memory


banned_ops = [mindspore.ops.operations.array_ops.Shape,
              mindspore.ops.operations.array_ops.Concat,
              type(None)
              ]
banned_cell = [mindspore.nn.layer.CentralCrop, ]
banned_trees = [mindspore.ops.ResizeBilinear, 
                mindspore.ops.operations.Shape,
                type(None)
                ]


def scan_node(stree, hash_table, nodedict=None, depth=0):
    
    
    if type(stree) == mindspore.rewrite.api.symbol_tree.SymbolTree:
        stree = stree._symbol_tree
    for node in stree.all_nodes():
        if isinstance(node, NodeManager):
            for sub_node in node.get_tree_nodes():
                subtree = sub_node.symbol_tree
                scan_node(subtree, hash_table, nodedict=nodedict, depth=depth + 1)
        if (node.get_node_type() == NodeType.CallCell and node.get_instance_type() not in banned_cell) or (
                node.get_node_type() == NodeType.CallPrimitive and node.get_instance_type() not in banned_ops) \
                or (node.get_node_type() == NodeType.Tree and node.get_instance_type() not in banned_trees) \
                or node.get_node_type() == NodeType.CellContainer:
            if hash_table[mindspore.rewrite.api.node.Node(node).get_handler()] == 1:
                continue
            hash_table[mindspore.rewrite.api.node.Node(node).get_handler()] += 1
            if node.get_node_type() not in [NodeType.CellContainer, NodeType.Tree]:
                nodedict[mindspore.rewrite.api.node.Node(node).get_handler()] = node.get_belong_symbol_tree()
    return True,nodedict

def check_node(node):
    if len(node.get_users()) == 0 or len(node.get_targets()) != 1:
        
        return False
    return True