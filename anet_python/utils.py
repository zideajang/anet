import numpy as np
from anet_python.tensor import Tensor

import requests, gzip, os, hashlib, numpy

# 加载数据
def fetch(url):
  fp = os.path.join("./tmp", hashlib.md5(url.encode('utf-8')).hexdigest())
  if not os.path.isfile(fp):
    with open(fp, "rb") as f:
      dat = f.read()
  else:
    with open(fp, "wb") as f:
      dat = requests.get(url).content
      f.write(dat)
  return numpy.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()


# train a model
