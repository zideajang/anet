import numpy as np
import torch
from tensor import Tensor

"""
out = x@w
out_relu = out.
"""

# def forward_visilaztion():


np.random.seed(42)

x = np.random.randn(1,3).astype(np.float32)
w = np.random.randn(3,3).astype(np.float32)
m = np.random.randn(1,3).astype(np.float32)


# tensor 实现
def test_anet():
    x_tensor = Tensor(x)
    w_tensor = Tensor(w)
    m_tensor = Tensor(m)

    out = x_tensor.dot(w_tensor)
    out_relu = out.relu()
    out_logsoftmax = out_relu.logsoftmax()
    out_m = out_logsoftmax.mul(m_tensor)
    out_sum = out_m.sum()
    out_sum.backward()
    print(f"grad of x_tensor:{x_tensor.grad}")
    print(f"grad of w_tensor:{w_tensor.grad}")
    print(f"grad of m_tensor:{m_tensor.grad}")

    print(f"grad of out_relu:{out_relu.grad}")
    print(f"grad of out_logsoftmax:{out_logsoftmax.grad}")
    print(f"grad of out_m:{out_m.grad}")
    print(f"grad of out_sum:{out_sum.grad}")#[1.]
    return out_sum.data, x_tensor.grad, w_tensor.grad

# pytorch 实现
def test_pytorch():
    x_tensor = torch.tensor(x,requires_grad=True)
    w_tensor = torch.tensor(w,requires_grad=True)
    m_tensor = torch.tensor(m)

    out = x_tensor.matmul(w_tensor)
    out_relu = out.relu()
    out_logsoftmax = torch.nn.functional.log_softmax(out_relu,dim=1)
    out_m = out_logsoftmax.mul(m_tensor)
    # out_m 是 tensor
    out_sum = out_m.sum()
    
    out_sum.backward()
  
    
    return out_sum.detach().numpy(), x_tensor.grad, w_tensor.grad



for x,y in zip(test_anet(),test_pytorch()):

    # print(x)
    # print(y)
    np.testing.assert_allclose(x,y,atol=1e-6)