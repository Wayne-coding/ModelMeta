import copy
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from mindspore import Tensor, context
from mindspore.common.initializer import Uniform
from mindspore import dtype as mstype



DTYPE = "float32"  # 数据类型
DELTA = 10         # 随机生成张量的范围，可自定义

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")  # 设置MindSpore上下文


class TransLayer_rule13_AvgPool2d(nn.Cell):
    def __init__(self, layer_pool):
        super(TransLayer_rule13_AvgPool2d, self).__init__()

        self.layer_pool = layer_pool
        self.delta = Tensor(np.random.uniform(-DELTA, DELTA, 1)[0], dtype=mstype.float32)

    def construct(self, x):
        mut_x = x + self.delta
        return self.layer_pool(mut_x) - self.delta


class TransLayer_rule13_MaxPool2d(nn.Cell):
    def __init__(self, layer_pool):
        super(TransLayer_rule13_MaxPool2d, self).__init__()

        self.layer_pool = layer_pool
        self.delta = Tensor(np.random.uniform(-DELTA, DELTA, 1)[0], dtype=mstype.float32)

    def construct(self, x):
        mut_x = x + self.delta
        return self.layer_pool(mut_x) - self.delta


class TransLayer_rule13_AdaptiveAvgPool2d(nn.Cell):
    def __init__(self, layer_pool):
        super(TransLayer_rule13_AdaptiveAvgPool2d, self).__init__()

        self.layer_pool = layer_pool
        self.delta = Tensor(np.random.uniform(-DELTA, DELTA, 1)[0], dtype=mstype.float32)

    def construct(self, x):
        mut_x = x + self.delta
        return self.layer_pool(mut_x) - self.delta

if __name__ == "__main__" and False:
    # 创建输入数据
    input_tensor = Tensor(np.random.randn(1, 1, 4, 4), dtype=mstype.float32)

    # 测试 TransLayer_rule13_AvgPool2d
    layer_pool_avg = nn.AvgPool2d(kernel_size=2, stride=2)
    trans_layer_avg = TransLayer_rule13_AvgPool2d(layer_pool_avg)
    original_output_avg = layer_pool_avg(input_tensor)
    trans_output_avg = trans_layer_avg(input_tensor)
    print("原始 AvgPool2d 输出:")
    print(original_output_avg)
    print("\n变异 AvgPool2d 输出:")
    print(trans_output_avg)
    diff_avg = ops.Abs()(original_output_avg - trans_output_avg).max()
    print("\nAvgPool2d 最大输出差异:", diff_avg.asnumpy())
    assert diff_avg.asnumpy() <= 1e-5, f"AvgPool2d 最大输出差异超出允许范围: {diff_avg.asnumpy()}"

    # 测试 TransLayer_rule13_MaxPool2d
    layer_pool_max = nn.MaxPool2d(kernel_size=2, stride=2)
    trans_layer_max = TransLayer_rule13_MaxPool2d(layer_pool_max)
    original_output_max = layer_pool_max(input_tensor)
    trans_output_max = trans_layer_max(input_tensor)
    print("\n原始 MaxPool2d 输出:")
    print(original_output_max)
    print("\n变异 MaxPool2d 输出:")
    print(trans_output_max)
    diff_max = ops.Abs()(original_output_max - trans_output_max).max()
    print("\nMaxPool2d 最大输出差异:", diff_max.asnumpy())
    assert diff_max.asnumpy() <= 1e-5, f"MaxPool2d 最大输出差异超出允许范围: {diff_max.asnumpy()}"

    # 测试 TransLayer_rule13_AdaptiveAvgPool2d
    layer_pool_adaptive = nn.AdaptiveAvgPool2d(output_size=(2, 2))
    trans_layer_adaptive = TransLayer_rule13_AdaptiveAvgPool2d(layer_pool_adaptive)
    original_output_adaptive = layer_pool_adaptive(input_tensor)
    trans_output_adaptive = trans_layer_adaptive(input_tensor)
    print("\n原始 AdaptiveAvgPool2d 输出:")
    print(original_output_adaptive)
    print("\n变异 AdaptiveAvgPool2d 输出:")
    print(trans_output_adaptive)
    diff_adaptive = ops.Abs()(original_output_adaptive - trans_output_adaptive).max()
    print("\nAdaptiveAvgPool2d 最大输出差异:", diff_adaptive.asnumpy())
    assert diff_adaptive.asnumpy() <= 1e-5, f"AdaptiveAvgPool2d 最大输出差异超出允许范围: {diff_adaptive.asnumpy()}"