
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransLayer_rule2(nn.Module):
    def __init__(self, layer_2d):
        super(TransLayer_rule2, self).__init__()
        if not isinstance(layer_2d, nn.Conv2d):
            raise ValueError("This wrapper only supports Conv2d layers")

        
        self.layer_2d = nn.Conv2d(
            in_channels=layer_2d.in_channels,
            out_channels=layer_2d.out_channels,
            kernel_size=layer_2d.kernel_size,
            stride=layer_2d.stride,
            padding=0,  
            dilation=layer_2d.dilation,
            bias=(layer_2d.bias is not None)
        )

        
        with torch.no_grad():
            self.layer_2d.weight.copy_(layer_2d.weight)
            if layer_2d.bias is not None:
                self.layer_2d.bias.copy_(layer_2d.bias)

    def forward(self, x):
        
        kernel_size = self.layer_2d.kernel_size
        stride = self.layer_2d.stride
        dilation = self.layer_2d.dilation

        
        padding_h = self._calculate_padding(x.shape[2], stride[0], kernel_size[0], dilation[0])
        padding_w = self._calculate_padding(x.shape[3], stride[1], kernel_size[1], dilation[1])

        
        x = F.pad(x, (padding_w // 2, padding_w - padding_w // 2, padding_h // 2, padding_h - padding_h // 2))

        
        x = self.layer_2d(x)
        return x

    def _calculate_padding(self, input_size, stride, kernel_size, dilation):
        output_size = (input_size + stride - 1) // stride  
        total_padding = max((output_size - 1) * stride + (kernel_size - 1) * dilation + 1 - input_size, 0)
        return total_padding

if __name__ == "__main__" and False:
    
    conv2d = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding='same')  
    same_padding_conv_wrapper = TransLayer_rule2(conv2d)


    input_2d = torch.randn(1, 3, 224, 224)  

    
    ans=conv2d(input_2d)
    
    print(ans.shape)

    
    
    ans2=same_padding_conv_wrapper(input_2d)
    
    print(ans.shape)

    
    inequality_mask = ans != ans2
    
    inequality_count = torch.sum(inequality_mask).item()
    print(inequality_count) ## 已验证，完全相等！！！！！！！！！！