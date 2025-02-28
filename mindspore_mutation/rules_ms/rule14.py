"""
变异Pool算子，input.transpose【3个pool都可以】
已验证，成功
"""

import copy
import mindspore
import mindspore.nn as nn
import numpy as np
from mindspore import Tensor, context
from mindspore.ops import Transpose

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

# 常量定义
DTYPE = "float32"  # 数据类型
DELTA = 10         # 随机生成张量的范围，可自定义

class TransLayer_rule14_AvgPool2d(nn.Cell):
    def __init__(self, layer_pool):
        super(TransLayer_rule14_AvgPool2d, self).__init__()
        
        self.layer_pool = layer_pool
        self.delta = Tensor(np.random.uniform(-DELTA, DELTA, 1)[0], mindspore.float32)

    def construct(self, x):
        mut_x = Transpose()(x, (0, 1, 3, 2))
        return Transpose()(self.layer_pool(mut_x), (0, 1, 3, 2))


class TransLayer_rule14_MaxPool2d(nn.Cell):
    def __init__(self, layer_pool):
        super(TransLayer_rule14_MaxPool2d, self).__init__()

        self.layer_pool = layer_pool
        self.delta = Tensor(np.random.uniform(-DELTA, DELTA, 1)[0], mindspore.float32)

    def construct(self, x):
        mut_x = Transpose()(x, (0, 1, 3, 2))
        return Transpose()(self.layer_pool(mut_x), (0, 1, 3, 2))


class TransLayer_rule14_AdaptiveAvgPool2d(nn.Cell):
    def __init__(self, layer_pool):
        super(TransLayer_rule14_AdaptiveAvgPool2d, self).__init__()

        self.layer_pool = layer_pool
        self.delta = Tensor(np.random.uniform(-DELTA, DELTA, 1)[0], mindspore.float32)

    def construct(self, x):
        mut_x = Transpose()(x, (0, 1, 3, 2))
        return Transpose()(self.layer_pool(mut_x), (0, 1, 3, 2))


if __name__ == "__main__" and False:
    mindspore.set_seed(0)
    np.random.seed(0)

    # 超参数和设备
    DELTA = 10

    # 定义原始池化层
    original_avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
    original_max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
    original_adaptive_avg_pool = nn.AdaptiveAvgPool2d(output_size=(2, 2))

    trans_avg_pool = TransLayer_rule14_AvgPool2d(original_avg_pool)
    trans_max_pool = TransLayer_rule14_MaxPool2d(original_max_pool)
    trans_adaptive_avg_pool = TransLayer_rule14_AdaptiveAvgPool2d(original_adaptive_avg_pool)

    # 创建输入数据
    input_tensor = Tensor(np.random.randn(1, 1, 4, 4), mindspore.float32)

    # 计算原始池化层的输出
    original_avg_output = original_avg_pool(input_tensor)
    original_max_output = original_max_pool(input_tensor)
    original_adaptive_avg_output = original_adaptive_avg_pool(input_tensor)

    # 计算变异池化层的输出
    trans_avg_output = trans_avg_pool(input_tensor)
    trans_max_output = trans_max_pool(input_tensor)
    trans_adaptive_avg_output = trans_adaptive_avg_pool(input_tensor)

    # 打印结果以比较
    print("原始AvgPool2d池化层输出:")
    print(original_avg_output)

    print("\n变异AvgPool2d池化层输出:")
    print(trans_avg_output)

    # 比较输出的差异
    diff_avg = (original_avg_output - trans_avg_output).abs().max()
    print("\nAvgPool2d最大输出差异:", diff_avg)

    print("\n原始MaxPool2d池化层输出:")
    print(original_max_output)

    print("\n变异MaxPool2d池化层输出:")
    print(trans_max_output)

    # 比较输出的差异
    diff_max = (original_max_output - trans_max_output).abs().max()
    print("\nMaxPool2d最大输出差异:", diff_max)

    print("\n原始AdaptiveAvgPool2d池化层输出:")
    print(original_adaptive_avg_output)

    print("\n变异AdaptiveAvgPool2d池化层输出:")
    print(trans_adaptive_avg_output)

    # 比较输出的差异
    diff_adaptive_avg = (original_adaptive_avg_output - trans_adaptive_avg_output).abs().max()
    print("\nAdaptiveAvgPool2d最大输出差异:", diff_adaptive_avg)