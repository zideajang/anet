import unittest
import numpy as np
import torch

import unittest
from anet.tensor import Tensor, Conv2D

# 卷积测试用例
class TestConvolution(unittest.TestCase):
    def setUp(self) -> None:
        self.im = np.random.randn(5,3,5,5)
        self.w = np.random.randn(2,3,3,3)
        return super().setUp()
    
    
    def test_conv2d(self):
        out = torch.nn.functional.conv2d(torch.tensor(self.im),torch.tensor(self.w))
        ret = Conv2D.apply(Conv2D,Tensor(self.im),Tensor(self.w))

        np.testing.assert_allclose(ret.data,out.numpy(),atol=1e-5)



if __name__ == "__main__":
    unittest.main()
    # im = torch.randn((5,2,5,5))
    # w = torch.randn((4,2,3,3))

    # out = torch.nn.functional.conv2d(im,w)
    # print(out.shape)#torch.Size([5, 4, 3, 3])