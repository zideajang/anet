# 优化器

from distutils.command.config import config
from typing import List
import numpy as np
from anet.tensor import Tensor

class Optimizer:
    def __init__(self,params:List[Tensor]):
        self.name = self.__class__.__name__
        self.config = {"name":self.name}
        self.params = params

class SGD(Optimizer):

    def __init__(self, params,lr=0.001):
        super(SGD,self).__init__(params)
        self.config["learning_rate"] = lr

    def step(self):
        for t in self.params:
            t.data -=  self.config["learning_rate"] * t.grad

class Adam(Optimizer):
    def __init__(self, params: List[Tensor],lr=0.001,b1=0.9,b2=0.999, eps=1e-8):
        super(Adam,self).__init__(params)

        self.config['lr'] = lr
        self.config['b1'] = b1
        self.config['b2'] = b2
        self.config['eps'] = eps
        self.t = 0


        self.m = [np.zeros_like(t.data) for t in self.params ] 
        self.v = [np.zeros_like(t.data) for t in self.params ] 

    def step(self):
        for i,t in enumerate(self.params):
            self.t += 1
            self.m[i] = self.config['b1'] * self.m[i] + (1 - self.config['b1']) * t.grad
            self.v[i] = self.config['b2'] * self.m[i] + (1 - self.config['b2']) * np.square(t.grad)
            
            m_hat = self.m[i] /(1. - self.config['b1']**self.t)
            v_hat = self.v[i] / (1. - self.config['b2']** self.t)

            t.data -= self.config['lr']*m_hat / (np.sqrt(v_hat) + self.config['eps'])

