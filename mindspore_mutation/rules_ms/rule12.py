import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from mindspore import Tensor, context

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

# 超参数和设备

DTYPE = "float32"  # 数据类型
DELTA = 10         # 随机生成张量的范围，可自定义

class TransLayerRule12AvgPool2d(nn.Cell):
    def __init__(self, layer_pool):
        super(TransLayerRule12AvgPool2d, self).__init__()
        self.layer_pool = layer_pool
        self.delta = Tensor(np.random.uniform(-DELTA, DELTA, 1)[0], mindspore.float32)

    def construct(self, x):
        mut_x = x * self.delta
        return self.layer_pool(mut_x) / self.delta

class TransLayerRule12MaxPool2d(nn.Cell):
    def __init__(self, layer_pool):
        super(TransLayerRule12MaxPool2d, self).__init__()
        self.layer_pool = layer_pool
        self.delta = Tensor(np.random.uniform(-DELTA, DELTA, 1)[0], mindspore.float32)

    def construct(self, x):
        mut_x = x * self.delta
        return self.layer_pool(mut_x) / self.delta

class TransLayerRule12AdaptiveAvgPool2d(nn.Cell):
    def __init__(self, layer_pool):
        super(TransLayerRule12AdaptiveAvgPool2d, self).__init__()
        self.layer_pool = layer_pool
        self.delta = Tensor(np.random.uniform(-DELTA, DELTA, 1)[0], mindspore.float32)

    def construct(self, x):
        mut_x = x * self.delta
        return self.layer_pool(mut_x) / self.delta

if __name__ == "__main__" and False:
    mindspore.set_seed(0)
    np.random.seed(0)

    # 定义原始池化层并创建输入数据
    input_tensor = Tensor(np.random.randn(1, 1, 4, 4), mindspore.float32)

    # 测试 AvgPool2d
    original_pool_avg = nn.AvgPool2d(kernel_size=2, stride=2)
    trans_pool_avg = TransLayerRule12AvgPool2d(original_pool_avg)
    original_output_avg = original_pool_avg(input_tensor)
    trans_output_avg = trans_pool_avg(input_tensor)
    print("AvgPool2d 原始池化层输出:")
    print(original_output_avg)
    print("\nAvgPool2d 变异池化层输出:")
    print(trans_output_avg)
    diff_avg = ops.Abs()(original_output_avg - trans_output_avg).max()
    print("\nAvgPool2d 最大输出差异:", diff_avg.asnumpy())

    # 测试 MaxPool2d
    original_pool_max = nn.MaxPool2d(kernel_size=2, stride=2)
    trans_pool_max = TransLayerRule12MaxPool2d(original_pool_max)
    original_output_max = original_pool_max(input_tensor)
    trans_output_max = trans_pool_max(input_tensor)
    print("\nMaxPool2d 原始池化层输出:")
    print(original_output_max)
    print("\nMaxPool2d 变异池化层输出:")
    print(trans_output_max)
    diff_max = ops.Abs()(original_output_max - trans_output_max).max()
    print("\nMaxPool2d 最大输出差异:", diff_max.asnumpy())

    # 测试 AdaptiveAvgPool2d
    original_pool_adaptive = nn.AdaptiveAvgPool2d(output_size=(2, 2))
    trans_pool_adaptive = TransLayerRule12AdaptiveAvgPool2d(original_pool_adaptive)
    original_output_adaptive = original_pool_adaptive(input_tensor)
    trans_output_adaptive = trans_pool_adaptive(input_tensor)
    print("\nAdaptiveAvgPool2d 原始池化层输出:")
    print(original_output_adaptive)
    print("\nAdaptiveAvgPool2d 变异池化层输出:")
    print(trans_output_adaptive)
    diff_adaptive = ops.Abs()(original_output_adaptive - trans_output_adaptive).max()
    print("\nAdaptiveAvgPool2d 最大输出差异:", diff_adaptive.asnumpy())