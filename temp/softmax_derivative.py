import numpy as np



def logsumexp(x):
    c = x.max(axis=1)
    return c + np.log(np.exp(x-c.reshape((-1,1))).sum(axis=1))


def logsoftmax(x):
    return x - logsumexp(x).reshape((-1,1))

def logsoftmax_backward(out,grad):
    return grad * ( 1 -np.exp(out))

if __name__ == "__main__":
    z = np.array([[1,2,1],[2,3,1]])
    s = softmax(z)
    # print(s.shape)
    # print(np.exp(s).sum(axis=1))