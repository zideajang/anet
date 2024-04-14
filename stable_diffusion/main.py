import io
import struct
from utils import fake_torch_load_zipped,get_child
import torch
import torch.nn as nn

FILENAME = "sd-v1-4.ckpt"

dat = fake_torch_load_zipped(open(FILENAME, "rb"), load_weights=False)

for k,v in dat['state_dict'].items():
    print(f"{str(v.shape):30s}",k)
exit(0)

class Normalize(nn.Module):
    def __init__(self,in_channels,num_groups=32) -> None:
        self.weight = torch.Tensor.uniform_(in_channels)
        self.bias = torch.Tensor.uniform_(in_channels)

    def forward(self,x):
        # groupnorm???
        pass 



class ResnetBlock:
    def __init__(self,in_channels,out_channels=None) -> None:
        self.norm1 = Normalize(in_channels)



class Encoder:
    def __init__(self,decode=False) -> None:
        sz = [(128,128),(128,256),(256,512),(512,512)]
        # self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True
        self.conv_in = nn.Conv2d(3,128,3)

        arr = []
        for i, s in enumerate(sz):
            arr.append({"block:"})



# 实现 Autoencoder 自动编码器
class AutoencoderKL:
    def __init__(self):
        self.encoder = Encoder()
        # conv 128, 3, 3, 3
        # implementation encoder
        # implementation decoder
        # quant_conv
        # post_quant_conv #(4, 4, 1, 1)

class StableDiffusion:
    def __init__(self) -> None:
        self.first_stage_model = AutoencoderKL()

for k,v in dat['state_dict'].items():
    if "first_stage_model" in k:
        print(k,v.shape)

