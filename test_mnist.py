import numpy as np

from tqdm import trange
from tensor import Tensor
from datasets import fetch

# 加载数据
X_train = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
Y_train = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]

# 参数初始化
def layer_init(m, h):
  ret = np.random.uniform(-1., 1., size=(m,h))/np.sqrt(m*h)
  return ret.astype(np.float32)

# 定义神经网络
l1 = Tensor(layer_init(784, 128))
l2 = Tensor(layer_init(128, 10))

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
  
  x = x.dot(l1)
  x = x.relu()
  x = x_l2 = x.dot(l2)
  x = x.logsoftmax()
  x = x.mul(y)
  x = x.mean()
  x.backward()
  
  loss = x.data
  cat = np.argmax(x_l2.data, axis=1)
  accuracy = (cat == Y).mean()
  
  # SGD
  l1.data = l1.data - lr*l1.grad
  l2.data = l2.data - lr*l2.grad
  
  losses.append(loss)
  accuracies.append(accuracy)
  t.set_description("loss %.2f accuracy %.2f" % (loss, accuracy))

def forward(x):
  x = x.dot(l1.data)
  x = np.maximum(x, 0)
  x = x.dot(l2.data)
  return x

def numpy_eval():
  Y_test_preds_out = forward(X_test.reshape((-1, 28*28)))
  Y_test_preds = np.argmax(Y_test_preds_out, axis=1)
  return (Y_test == Y_test_preds).mean()

print("test set accuracy is %f" % numpy_eval())

