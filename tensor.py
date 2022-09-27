##############################
# 学着实现一个深度学习框架 anet
#############################

from functools import partialmethod
import numpy as np

# 反向传播环境类
class Context:
    def __init__(self,arg,*tensors):
        """
        arg:Function 运算符、例如 Sum、Dot 等继承了 Function 的
        parents:Tensor : 参与运算的元素
        parents:ndarray
        """
        self.arg = arg
        self.parents = tensors
        self.saved_tensors=[]

    def save_for_backward(self,*x):
        self.saved_tensors.extend(x)



class Tensor:
    def __init__(self,data,_children=()):
        if isinstance(data,(list,tuple)):
            data = np.array(data)

        self.data = data
        self.grad = None

        self._ctx = None

    def __str__(self):
        return f"Tensor of shape {self.data.shape} with grad {self.grad}"
    
    def backward(self,allow_fill=True):
        if self._ctx is None:
            return

        if self.grad is None and allow_fill:
            assert self.data.size == 1
            self.grad = np.ones_like(self.data)

        assert(self.grad is not None)

        # 
        grads = self._ctx.arg.backward(self._ctx,self.grad)
        print(grads)
        if len(self._ctx.parents) == 1:
            grads = [grads]
        
        for t,g in zip(self._ctx.parents,grads):
            print("111")
            if g.shape != t.data.shape:
                print("grad shape must match tensor shape in %r, %r != %r" %(self._ctx.arg, g.shape, t.data.shape))
                #assert(False) "print something"
            t.grad = g
            t.backward(False)
    def mean(self):
        div = Tensor(np.array([1/self.data.size]))
        return self.sum().mul(div)

"""
运算符的基类
"""
class Function:
    """
    arg:Function

    """
    def apply(self:Tensor,arg,*x):
        # initial Context
        # self is instance of Tensori
        # arg - sum, self- tensor
        # y = wx dot dot x.dot(w) self- x *x- w 
        ctx = Context(arg,self,*x)
        # dot.forward
        ret = Tensor(arg.forward(ctx, self.data,*[t.data for t in x]))
        ret._ctx = ctx
        return ret

def register(name,fxn):

    # set method to Tensor cls
    # sum.apply (fxn) (*x
    setattr(Tensor,name,partialmethod(fxn.apply,fxn))

        
"""

"""
class ReLU(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return np.maximum(input,0)

    @staticmethod
    def backward(ctx,grad_output):
        
        input,= ctx.saved_tensors
        grad_input = grad_output.copy()
        grad_input[input < 0] = 0

        return grad_input

register("relu",ReLU)

class Dot(Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input,weight)
        return input.dot(weight)

    @staticmethod
    def backward(ctx,grad_output):
        # grad_output * local grad
        
        input,weight = ctx.saved_tensors

        grad_input = grad_output.dot(weight.T)
        grad_weight = grad_output.T.dot(input).T

        return grad_input,grad_weight

register("dot",Dot)

"""
求和
"""
class Sum(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return np.array([input.sum()])

    @staticmethod
    def backward(ctx,grad_output):
        # grad_output * local grad
        input, = ctx.saved_tensors
        return grad_output * np.ones_like(input)

register("sum",Sum)
"""
logSoftmax
"""
class LogSoftmax(Function):
    @staticmethod
    def forward(ctx, input):
        def logsumexp(x):
            c = x.max(axis=1)
            return c + np.log(np.exp(x-c.reshape((-1,1))).sum(axis=1))
        output = input - logsumexp(input).reshape((-1,1))
        ctx.save_for_backward(output)
    
        return output

    @staticmethod
    def backward(ctx,grad_output):
        # grad_output * local grad
        output, = ctx.saved_tensors
        return grad_output - np.exp(output)*grad_output.sum(axis=1).reshape((-1,1))

register("logsoftmax",LogSoftmax)


class Mul(Function):
    @staticmethod
    def forward(ctx, x,y):
        ctx.save_for_backward(x,y)
    
        return x*y

    @staticmethod
    def backward(ctx,grad_output):
        x,y  = ctx.saved_tensors
        return y*grad_output,x*grad_output

register("mul",Mul)

if __name__ == "__main__":


   import torch
   import random


   torch.manual_seed(42)
    
   random.seed(42)
   np.random.seed(42)
   

   x = torch.randn((1,3),requires_grad=True)
   w = torch.randn((3,3),requires_grad=True)

   print(x)
   print(w)
   z = torch.matmul(x,w)

   y = z.sum()
   y.backward()
   print(f"x:{x.grad}")
   
   
   input_tensor = Tensor(x.detach().numpy())
   weight_tensor = Tensor(w.detach().numpy())

   z_tensor = input_tensor.dot(weight_tensor)

   y_tensor = z_tensor.sum()
   y_tensor.backward()
   print(f"input:{input_tensor.grad}")


   #validate logsoftmax works
   from torch.nn.functional import log_softmax
   x1 = torch.tensor([[1.,2.,1.]]).float()
   print('-'*50)
   print(log_softmax(x1,dim=1))


   x_tensor = Tensor(x1.numpy())

   print(x_tensor.logsoftmax().data)


   #############################
   # test ReLU
   ############################

   x2 = torch.randn(2).unsqueeze(0)
   from torch.nn import functional as F
   output = F.relu(x2)
   print(output)
   x2_tensor = Tensor(x2.numpy())
   output = x2_tensor.relu()
   print(output.data)

    


    
""" 
    np_arr = np.random.randn(1,3)

    t1 = Tensor(np.array([1,2,3]))
    res = t1.sum()
    print(res.data)

    t1_data = np.random.randn(1,3)
    t2_data = np.random.randn(3,3)

    t1 = Tensor(t1_data)
    t2 = Tensor(t2_data)
    res = t1.dot(t2)
    print(res.data)
    print(np_arr)

    x1 = np.array([1,2,3])
    x2 = Sum.forward(None,x1)
    print(x2)
    def f(*x):
        res = []
        res.extend(x)
        return res
    b = f([1,2,3],[2,3,5])
    print(b)


    a = [1,2,3] + list([2,3,5])
    print(a)

"""