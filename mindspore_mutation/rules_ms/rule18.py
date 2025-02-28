"""
变异tanh算子，input*(-1)
已验证，成功
"""

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from mindspore import Tensor, context

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

class TransLayerRule18(nn.Cell):
    def __init__(self, layer_tanh):
        super(TransLayerRule18, self).__init__()
        self.layer_tanh = layer_tanh

    def construct(self, x):
        return -self.layer_tanh(-x)

if __name__ == "__main__" and False:
    # 创建 Tanh 层
    tanh = nn.Tanh()

    # 创建变异层实例
    trans_layer = TransLayerRule18(tanh)

    # 生成随机数据
    x = Tensor(np.random.randn(5, 1), mindspore.float32)

    # 计算正常 softmax 的输出
    original_output = tanh(x)
    print(original_output)

    # 计算变异 softmax 的输出
    mutated_output = trans_layer(x)
    print("\nMutated Softmax Output:")
    print(mutated_output)

    # 比较两个输出是否一致
    print("Are the outputs equal?", np.allclose(original_output.asnumpy(), mutated_output.asnumpy(), atol=1e-5))