import numpy as np

from tqdm import trange
from anet_python.tensor import Tensor
from anet_python.utils import fetch

# 加载数据
X_train = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
Y_train = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]

# 参数初始化
def layer_init(m, h):
  ret = np.random.uniform(-1., 1., size=(m,h))/np.sqrt(m*h)
  return ret.astype(np.float32)

class ANet:
  def __init__(self):
    # 定义神经网络
    # 输入 (batch_size * (28*28))
    self.l1 = Tensor(layer_init(784, 128))
    self.l2 = Tensor(layer_init(128, 10))

  def forward(self,x):
    x = x.dot(self.l1)
    x = x.relu()
    x = x.dot(self.l2)
    out = x.logsoftmax()
    return out 

model = ANet()


# 定义超参数
lr = 0.01
BS = 128
losses, accuracies = [], []
for i in (t := trange(1000)):
  samp = np.random.randint(0, X_train.shape[0], size=(BS))
  
  x = Tensor(X_train[samp].reshape((-1, 28*28)))
  Y = Y_train[samp]
  y = np.zeros((len(samp),10), np.float32)
  y[range(y.shape[0]),Y] = -1.0
  y = Tensor(y)
  
  outs = model.forward(x)

  #NLL loss function
  loss = outs.mul(y).mean()
  loss.backward()
  
  cat = np.argmax(outs.data, axis=1)
  accuracy = (cat == Y).mean()
  
  # SGD
  model.l1.data = model.l1.data - lr*model.l1.grad
  model.l2.data = model.l2.data - lr*model.l2.grad
  
  loss = loss.data
  losses.append(loss)
  accuracies.append(accuracy)
  t.set_description("loss %.2f accuracy %.2f" % (loss, accuracy))

def numpy_eval():
  Y_test_preds_out = model.forward(X_test.reshape((-1, 28*28)))
  Y_test_preds = np.argmax(Y_test_preds_out, axis=1)
  return (Y_test == Y_test_preds).mean()

accuracy = numpy_eval()
print(f"test accuracy: {accuracy}")
