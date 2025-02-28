'''
变异softmax算子，input+=delta
已验证，成功
'''

import copy
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from mindspore import Tensor, context
from mindspore.common.initializer import initializer
from mindspore import dtype as mstype

# 设置设备为 GPU 模式，如果使用其他设备请更改 'GPU' 为相应设备类型
# context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

class TransLayer_rule17(nn.Cell):
    def __init__(self, layer_softmax):
        super(TransLayer_rule17, self).__init__()

        self.layer_softmax = layer_softmax
        self.delta = Tensor(10, mstype.float32)

    def construct(self, x):
        mut_x = x + self.delta
        return self.layer_softmax(mut_x)

if __name__ == "__main__" and False:
    # 创建 Softmax 层
    softmax = nn.Softmax(axis=1)

    # 创建变异层实例
    trans_layer = TransLayer_rule17(softmax)

    # 生成随机数据
    x = initializer('normal', [5, 10], mstype.float32)  # 5个样本，每个样本10维

    # 计算正常 softmax 的输出
    original_output = softmax(x)
    print("Original Softmax Output:")
    print(original_output)

    # 计算变异 softmax 的输出
    mutated_output = trans_layer(x)
    print("\nMutated Softmax Output:")
    print(mutated_output)

    # 比较两个输出是否一致
    print("Are the outputs equal?", np.allclose(original_output.asnumpy(), mutated_output.asnumpy(), atol=1e-5))

    # 打印变异系数
    print("\nDelta used for mutation:")
    print(trans_layer.delta)