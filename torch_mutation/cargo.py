import os

import numpy as np
from torch_mutation.rules_torch import rule1,rule2,rule3,rule4,rule5,rule6,rule7,rule8,rule9,\
                                        rule10,rule11,rule12,rule13,rule14,rule15,rule16,rule17,rule18
import torch
import random
import pennylane as qml
import torch.nn as nn
import scipy.io as scio
import torch.fx as fx
import psutil

from models.vgg16.vgg16_torch import vgg16 
from models.resnet50.resnet50_torch import resnet50
from models.openpose.main_parallel_torch import OpenPoseNet as OpenPoseNet_torch 
from models.openpose.src.model_utils.config import config as openpose_config 
from models.SSD.backbone_resnet50_fpn_pytorch import ssd_resnet50fpn_torch 
from models.SSD.backbone_mobilenetv1_pytorch import SSDWithMobileNetV1 
from models.UNet.main_torch import UNetMedical_torch 
from models.textcnn.textcnn_torch import TextCNN 
from models.PatchCore.model_torch import wide_resnet50_2 as PatchCore 
from models.ssimae.src.network_torch import AutoEncoder as AutoEncoder_torch 
from models.ssimae.model_utils.config import config as ssimae_cfg  


datasets_path_cargo = {
    "vgg16": "/data1/czx/semtest/data/cifar10_x.npy", 
    
    "resnet": "/data1/czx/semtest/data/ResNet_data0.npy", 

    "UNetMedical": "/data1/czx/semtest/data/UNetMedical_data0.npy", 

    "textcnn": "/data1/czx/semtest/data/TextCNN_data0.npy",  

    "ssimae": "/data1/czx/semtest/data/ssimae_data0.npy", 
}

def get_model(model_name, device):
    net_cargo = {
        

        "vgg16": vgg16, 
        "resnet": resnet50, 
        "UNetMedical": UNetMedical_torch,
        "textcnn": TextCNN,  
        "ssimae": AutoEncoder_torch,  
    }
    if model_name == "vgg16": 
        model = net_cargo[model_name](10)
        return model
    elif model_name == "resnet": 
        model = net_cargo[model_name]()
        return model
    elif model_name == "UNetMedical":
        model = net_cargo[model_name](1, 2)
        return model
    elif model_name == "textcnn": 
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

    elif model_name == "textcnn": 
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


def rule_reflect_class(option_rule,option_instance):
    if option_rule is rule1:
        if isinstance(option_instance, torch.nn.Conv2d):
            return rule1.TransLayer_rule1_Conv2d
        elif isinstance(option_instance, torch.nn.AvgPool2d):
            return rule1.TransLayer_rule1_AvgPool2d
        elif isinstance(option_instance, torch.nn.MaxPool2d):
            return rule1.TransLayer_rule1_MaxPool2d
        elif isinstance(option_instance, torch.nn.ReLU):
            return rule1.TransLayer_rule1_ReLU
        elif isinstance(option_instance, torch.nn.ReLU6):
            return rule1.TransLayer_rule1_ReLU6
        elif isinstance(option_instance, torch.nn.BatchNorm2d):
            return rule1.TransLayer_rule1_BatchNorm2d
        elif isinstance(option_instance, torch.nn.Linear):
            return rule1.TransLayer_rule1_Linear
        elif isinstance(option_instance, torch.nn.Flatten):
            return rule1.TransLayer_rule1_Flatten
        elif isinstance(option_instance, torch.nn.Hardsigmoid):
            return rule1.TransLayer_rule1_Hardsigmoid
        elif isinstance(option_instance, torch.nn.Sigmoid):
            return rule1.TransLayer_rule1_Sigmoid
        elif isinstance(option_instance, torch.nn.Softmax):
            return rule1.TransLayer_rule1_Softmax
        elif isinstance(option_instance, torch.nn.Tanh):
            return rule1.TransLayer_rule1_Tanh
        elif isinstance(option_instance, torch.nn.ConvTranspose2d):
            return rule1.TransLayer_rule1_ConvTranspose2d
        elif isinstance(option_instance, torch.nn.LeakyReLU):
            return rule1.TransLayer_rule1_LeakyReLU
        elif isinstance(option_instance, torch.nn.AdaptiveAvgPool2d):
            return rule1.TransLayer_rule1_AdaptiveAvgPool2d
        elif isinstance(option_instance, torch.nn.Dropout):
            return rule1.TransLayer_rule1_Dropout
        elif isinstance(option_instance, torch.nn.Embedding):
            return rule1.TransLayer_rule1_Embedding
        elif isinstance(option_instance, torch.nn.LSTM):
            return rule1.TransLayer_rule1_LSTM
    elif option_rule is rule2:
        return rule2.TransLayer_rule2
    elif option_rule is rule3:
        if isinstance(option_instance, torch.nn.Conv2d):
            return rule3.TransLayer_rule3_Conv2d
        elif isinstance(option_instance, torch.nn.AvgPool2d):
            return rule3.TransLayer_rule3_AvgPool2d
        elif isinstance(option_instance, torch.nn.MaxPool2d):
            return rule3.TransLayer_rule3_MaxPool2d
    elif option_rule is rule4:
        return rule4.TransLayer_rule4
    elif option_rule is rule5:
        return rule5.TransLayer_rule5
    elif option_rule is rule6:
        return rule6.TransLayer_rule6
    elif option_rule is rule7:
        return rule7.TransLayer_rule7
    elif option_rule is rule8:
        return rule8.TransLayer_rule8
    elif option_rule is rule9:
        return rule9.TransLayer_rule9
    elif option_rule is rule10:
        return rule10.TransLayer_rule10
    elif option_rule is rule11:
        return rule11.TransLayer_rule11
    elif option_rule is rule12:
        if isinstance(option_instance, torch.nn.AdaptiveAvgPool2d):
            return rule12.TransLayer_rule12_AdaptiveAvgPool2d
        elif isinstance(option_instance, torch.nn.AvgPool2d):
            return rule12.TransLayer_rule12_AvgPool2d
        elif isinstance(option_instance, torch.nn.MaxPool2d):
            return rule12.TransLayer_rule12_MaxPool2d
    elif option_rule is rule13:
        if isinstance(option_instance, torch.nn.AdaptiveAvgPool2d):
            return rule13.TransLayer_rule13_AdaptiveAvgPool2d
        elif isinstance(option_instance, torch.nn.AvgPool2d):
            return rule13.TransLayer_rule13_AvgPool2d
        elif isinstance(option_instance, torch.nn.MaxPool2d):
            return rule13.TransLayer_rule13_MaxPool2d
    elif option_rule is rule14:
        if isinstance(option_instance, torch.nn.AdaptiveAvgPool2d):
            return rule14.TransLayer_rule14_AdaptiveAvgPool2d
        elif isinstance(option_instance, torch.nn.AvgPool2d):
            return rule14.TransLayer_rule14_AvgPool2d
        elif isinstance(option_instance, torch.nn.MaxPool2d):
            return rule14.TransLayer_rule14_MaxPool2d
    elif option_rule is rule15:
        if isinstance(option_instance, torch.nn.ReLU):
            return rule15.TransLayer_rule15_ReLU
        elif isinstance(option_instance, torch.nn.LeakyReLU):
            return rule15.TransLayer_rule15_LeakyReLU
    elif option_rule is rule16:
        return rule16.TransLayer_rule16


def select_places(sequence, k): 
    for i in range(5):
        try:
            chosen = random.choices(sequence, k=k)
        except Exception as e:
            print("sequence is", sequence)
            return None, None
        subs_place = max(chosen)
        chosen.remove(subs_place)
        if max(chosen) != subs_place:
            return subs_place, chosen
    print("Cannot find suitable places")
    return None, None

def select_places(sequence, k): 
    for i in range(5):
        try:
            chosen = random.choices(sequence, k=k)
        except Exception as e:
            print("sequence is", sequence)
            return None, None
        subs_place = max(chosen)
        chosen.remove(subs_place)
        if max(chosen) != subs_place:
            return subs_place, chosen
    print("Cannot find suitable places")
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


def compute_gpu_cpu(): 
    gpu_memory = torch.cuda.memory_allocated() / 1024 ** 2  
    mem = psutil.virtual_memory()  
    cpu_memory = float(mem.available) / 1024 / 1024
    return gpu_memory,cpu_memory

