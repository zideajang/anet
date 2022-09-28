from unittest import result
import numpy as np

from anet.tensor import Tensor
import unittest

# 测试前向传播

# Sum 方法的前向传播测试用例
class TestSumForward(unittest.TestCase):
    def test_sum_forward(self):
        t1 = Tensor(np.array([1,2,3]))
        result = t1.sum()
        self.assertEqual(result.data,np.array(6))


class TestAddForward(unittest.TestCase):
    def test_add_forward(self):
        t1 = np.array([1,2,3])
        t2 = np.array([2,1,1])

        t1_tensor = Tensor(t1)
        t2_tensor = Tensor(t2)

        result = t1_tensor.add(t2_tensor)

        self.assertTrue((result.data == (t1+t2)).all())

class TestLogSoftMax(unittest.TestCase):
    def test_log_softmax(self):
        t1 = np.array([[1,1,1]])
        t1_tensor = Tensor(t1)

        result = t1_tensor.logsoftmax()
        # log[] exp ()
        result = np.exp(result.data)
        result = result.sum()
        self.assertTrue(np.array(1.0) == np.ceil(result))