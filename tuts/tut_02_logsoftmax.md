## Softmax 求导

### Context

```python
class Context:
  def __init__(self, arg, *tensors):
    self.arg = arg
    self.parents = tensors
    self.saved_tensors = []

  def save_for_backward(self, *x):
    self.saved_tensors.extend(x)
```

```python
class Function:
  def apply(self, arg, *x):
    ctx = Context(arg, self, *x)
    ret = Tensor(arg.forward(ctx, self.data, *[t.data for t in x]))
    ret._ctx = ctx
    return ret
```
- 
每一个运算符(Sum, Add)都持有 Context 在 Context 中会保存


```python
class LogSoftmax(Function):
  @staticmethod
  def forward(ctx, input):
    def logsumexp(x):
      c = x.max(axis=1)
      return c + np.log(np.exp(x-c.reshape((-1, 1))).sum(axis=1))
    output = input - logsumexp(input)
    ctx.save_for_backward(output)
    return output

  @staticmethod
  def backward(ctx, grad_output):
    output, = ctx.saved_tensors
    return grad_output - np.exp(output)*grad_output.sum(axis=1).reshape((-1, 1))
register('logsoftmax', LogSoftmax)
```

```python
def register(name, fxn):
  setattr(Tensor, name, partialmethod(fxn.apply, fxn))
```

$$
softmax: \mathbb{R}^n \rightarrow \mathbb{R}^n
$$

$$
s = softmax(z)
$$

这里 $s_i $ 是输出概率分布向量一个分量，那么

$$
s_i = \frac{e^{z_i}}{\sum_i e^{z_i}}
$$

$$
grad - e^{grad} \times \sum_{i=1} grad
$$

```python
grad_output - np.exp(output)*grad_output.sum(axis=1).reshape
```
grad 是 n 向量，也就是 $grad \in \mathbb{R}^n $，那么输入 z 和输出 s 也都是 n 维度，也就是 ，输入 $z$ 表示向量，那么 $z_i$ 表示一个向量的元素，softmax 看作函数，输出是一个内维向量 n 

$$
s_i = \frac{\exp(z_i)}{\sum_{i=1}^n \exp(z_l)}
$$

$$
J_{softmax} =  \begin{bmatrix} 
\frac{\partial s_1}{\partial z_1} & \frac{\partial s_1}{\partial z_2} & \cdots  \frac{\partial s_1}{\partial z_n} \\
\frac{\partial s_2}{\partial z_1} & \frac{\partial s_2}{\partial z_2} & \cdots  \frac{\partial s_2}{\partial z_n} \\
\vdots & \vdots & \vdots & \vdots &\\
\frac{\partial s_n}{\partial z_1} & \frac{\partial s_n}{\partial z_2} & \cdots  \frac{\partial s_n}{\partial z_n} \\
\end{bmatrix}
$$

$$
\frac{\partial s_i}{\partial z_j}
$$

$$
\frac{\partial }{\partial z_j}\log(s_i) = \frac{1}{s_i} \frac{\partial s_i}{\partial z_j}
$$

$$
\frac{\partial s_i}{\partial z_j} = s_i \frac{\partial }{\partial z_j}\log(s_i) 
$$

$$
\log(s_i) = \log \left( \frac{\exp(z_i)}{\sum_{i=1}^n \exp(z_l)} \right)\\
\log(s_i) = z_i - \log \left(\sum_{i=1}^n \exp(z_l) \right)
$$

$$
\frac{\partial}{\partial z_j}(\log s_i) = \frac{\partial z_i}{\partial z_j} - \frac{\partial }{\partial z_j} \log(\sum_{l=1}^n e^{z_l})
$$ 

当 $i=j$  时候
$$
\frac{\partial z_i}{\partial z_j} = 1
$$

当 $otherwise$  时候

$$
\frac{\partial z_i}{\partial z_j} = 0
$$


$$
\frac{\partial}{\partial z_j}(\log s_i) = 1\{i=j\} - \frac{\partial }{\partial z_j} \log(\sum_{l=1}^n e^{z_l})
$$

$$
\frac{\partial}{\partial z_j}(\log s_i) = 1\{i=j\} -  \frac{1}{\sum_{l=1}^n e^{z_l}} \left(\frac{\partial }{\partial z_j} \sum_{l=1}^n e^{z_l}\right)
$$


$$
\frac{d}{dx} \log(x) = \frac{1}{x}
$$

$$
\frac{\partial }{\partial z_j} \sum_{l=1}^n e^{z_l} = \frac{\partial }{\partial z_j} [e^{z_1}+ e^{z_2}+ \cdots + e^{z_j}+ \cdots + e^{z_n}] =  e^{z_j}
$$
### log softmax 求导


$$
\frac{\partial}{\partial z_j}(\log s_i) = 1\{i=j\} - \frac{e^{z_j}}{ \sum_{l=1}^n e^{z_l}} = 1\{i=j\} - s_j
$$

$$
\frac{\partial L}{\partial s_i}\frac{\partial}{\partial z_j}(\log s_i) = \frac{\partial L}{\partial s_i}\left( 1\{i=j\} - \frac{e^{z_j}}{ \sum_{l=1}^n e^{z_l}} \right) = 1\{i=j\} - s_j
$$

```python
grad_output - np.exp(output)*grad_output.sum(axis=1).reshape((-1,1))
```
这里 grad_output n 维 


$[]$

- 定义情况 $i=j$

- 其他



$$
(bs,output)
$$
梯度
$$
(bs.output)
$$

$$
grad * (1 - )
$$

$$
\frac{\partial s_i}{\partial z_j} = s_i \frac{\partial}{\partial z_j} \log(s_i) = s_i(1\{i=j\} -s_j)
$$

$$
J_{softmax} =  \begin{bmatrix} 
s_1 (1-s_1) & -s_1s_2 & -s_1s_3  & -s_1s_4 \\
-s_2s_1 & s_2(1-s_2) & -s_2s_3  & -s_2s_4 \\
-s_3s_1 & -s_3s_2 & s_3(1-s_3)  & -s_3s_4 \\
-s_4s_1 & -s_4s_2 & -s_4s_3  & s_4(1-s_4) \\
\end{bmatrix}
$$



$$
\frac{L}{\partial z} = s - y
$$






#### 代码实现

```python
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
```