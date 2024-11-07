import device_op
import mindspore
if device_op.device_option=='cpu':
    device='CPU'
elif device_op.device_option=='gpu':
    device='GPU'
mindspore.set_context(mode=mindspore.PYNATIVE_MODE, device_target="GPU")
import mindspore.nn as nn
from mindspore_mutation.rules_ms import rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, \
                                        rule10, rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18

rules_dict = {
    
    nn.Conv2d: [rule1, rule3, rule5, rule6, rule7, rule8],  
    nn.AvgPool2d: [rule1, rule3, rule12, rule13, rule14],
    
    nn.MaxPool2d: [rule1, rule3, rule12, rule13, rule14],
    nn.ReLU: [rule1, rule15],
    nn.ReLU6: [rule1],
    nn.BatchNorm2d: [rule1, rule4, rule9, rule10, rule11],
    nn.Dense: [rule1],
    nn.Flatten: [rule1],
    nn.HSigmoid: [rule1],
    nn.Sigmoid: [rule16, rule1],
    nn.Softmax: [rule17, rule1],
    nn.Tanh: [rule18, rule1],
    
    nn.Conv2dTranspose: [rule1],
    nn.LeakyReLU: [rule1, rule15],
    nn.AdaptiveAvgPool2d: [rule1, rule12, rule13, rule14],
    nn.Dropout: [rule1],
    nn.Embedding: [rule1],
    nn.LSTM: [rule1]
}