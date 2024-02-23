import numpy as np
import time


N = 512

A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)

gflop = 2 * N**3/1e9

start_time = time.monotonic()
C = np.dot(A,B)

end_time = time.monotonic()
s = end_time - start_time
print(f"GFLOPS: {gflop/s}")

with open("./tmp/matmul","wb") as f:
    f.write(A.data)
    f.write(B.data)
    f.write(C.data)