""
"""
变异BatchNorm算子，bias+=delta
已验证，成功
"""


import mindspore.nn as nn
import mindspore.ops as ops
import mindspore
from mindspore import Tensor, Parameter
import numpy as np
from mindspore.common.initializer import initializer
from mindspore import context
import copy

# 常量定义
DTYPE = "float32"  # 数据类型
DELTA = 10         # 随机生成张量的范围，可自定义

# 设置上下文
# context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

class TransLayerRule9(nn.Cell):
    def __init__(self, layer_bn):
        super(TransLayerRule9, self).__init__()

        self.layer_bn = copy.deepcopy(layer_bn)

        # 直接操作 beta 参数，因为没有 use_bias 属性
        delta = np.random.uniform(-DELTA, DELTA, 1).astype(DTYPE)
        self.delta = Parameter(Tensor(delta, mindspore.float32), name="delta")
        self.layer_bn.beta += self.delta

        if self.layer_bn.use_batch_statistics:
            self.layer_bn.moving_mean = Parameter(self.layer_bn.moving_mean.clone(), name="moving_mean")
            self.layer_bn.moving_variance = Parameter(self.layer_bn.moving_variance.clone(), name="moving_variance")

    def construct(self, x):
        return self.layer_bn(x) - self.delta

if __name__ == "__main__" and False:
    # 创建一个标准的 BatchNorm 层
    batch_norm = nn.BatchNorm2d(10, use_batch_statistics=True)

    # 测试数据
    x = Tensor(np.random.randn(5, 10, 32, 32), mindspore.float32)

    # 通过 BatchNorm 层
    bn_output = batch_norm(x)
    # print(batch_norm.beta)

    # 创建 TransLayer 层，初始化为与 batch_norm 层相同的参数
    trans_layer = TransLayerRule9(batch_norm)

    # 通过 TransLayer 层
    trans_output = trans_layer(x)

    # print(trans_layer.layer_bn.beta)
    # print(trans_layer.delta)

    # 打印输出结果
    # print("\nBatchNorm output:")
    # print(bn_output)

    # print("\nTransLayer output:")
    # print(trans_output)

    # 计算和打印输出差异
    dis = ops.ReduceSum()(ops.Abs()(bn_output - trans_output))
    print("\nMaximum difference between BatchNorm and TransLayer outputs:")
    print(dis)