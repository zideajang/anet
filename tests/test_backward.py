from importlib.metadata import requires
import numpy as np
import torch

import unittest
from anet.tensor import Tensor

class TestBackwardAndForward(unittest.TestCase):

  def setUp(self) -> None:
    self.x = np.random.randn(1,3).astype(np.float32)
    self.w = np.random.randn(3,3).astype(np.float32)
    self.m = np.random.randn(1,3).astype(np.float32)
    return super().setUp()

  def test_backward(self):
    def test_anet():
      x_tensor = Tensor(self.x)
      w_tensor = Tensor(self.w)
      m_tensor = Tensor(self.m)

      out = x_tensor.dot(w_tensor).relu()
      out = out.logsoftmax()
      out = out.mul(m_tensor).add(m_tensor).sum()
      out.backward()
      return out.data, x_tensor.grad, w_tensor.grad

    def test_pytorch():
      x_tensor = torch.tensor(self.x,requires_grad=True)
      w_tensor = torch.tensor(self.w,requires_grad=True)
      m_tensor = torch.tensor(self.m)

      out = x_tensor.matmul(w_tensor).relu()
      out = torch.nn.functional.log_softmax(out,dim=1)
      out = out.mul(m_tensor).add(m_tensor).sum()
      out.backward()
      return out.detach().numpy(), x_tensor.grad, w_tensor.grad

    for x,y in zip(test_anet(),test_pytorch()):
      np.testing.assert_allclose(x,y,atol=1e-5)

  def tearDown(self) -> None:
    return super().tearDown()


if __name__ == "__main__":
  unittest.main()