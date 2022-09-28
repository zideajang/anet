##############################
# 学着实现一个深度学习框架 anet
#############################

from functools import partialmethod
import numpy as np

# basic classes

"""
Tensor
"""
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

        print(self.grad)
        print(self._ctx)
        print(self.data)
        
        
        if self.grad is None and allow_fill:
            assert self.data.size == 1
            self.grad = np.ones_like(self.data)

        assert(self.grad is not None)
    
        grads = self._ctx.arg.backward(self._ctx,self.grad)
        if len(self._ctx.parents) == 1:
            grads = [grads]
        
        print(grads)
        print("="*50)
        for t,g in zip(self._ctx.parents,grads):
            if g.shape != t.data.shape:
                print("grad shape must match tensor shape in %r, %r != %r" %(self._ctx.arg, g.shape, t.data.shape))
                #assert(False) "print something"
            t.grad = g
            t.backward(False)

        
    def mean(self):
        div = Tensor(np.array([1/self.data.size]))
        return self.sum().mul(div) 

"""
反向传播上下类
"""     
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

    def __str__(self) -> str:
        res =  f"arg: {self.arg}, parents: {[x.data for x in self.parents]}, saved_tensors:{self.saved_tensors}"

        return res

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
Sum 实现
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

class Add(Function):
    @staticmethod
    def forward(ctx, x,y):
        return x + y

    @staticmethod
    def backward(ctx,grad_output):
        return grad_output,grad_output

register("add",Add)

# 实现卷积算子
class Conv2D(Function):
    @staticmethod
    def forward(ctx, x,w):
        cout,cin,H,W = w.shape
        ret = np.zeros((x.shape[0]))

