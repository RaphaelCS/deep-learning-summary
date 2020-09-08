import numpy as np
import torch  # This is far more slow

from raphtools.timer import Timer


n = 100000000
a = np.ones(n)
b = np.ones(n)

# c = np.zeros(n)
timer = Timer()
# for i in range(n):
#     c[i] = a[i] + b[i]
# print(f"{timer.stop():.5f} sec")

timer.start()
d = a + b
print(f"{timer.stop():.5f} sec")


a = torch.ones(n)
b = torch.ones(n)
timer.start()
d = a + b
print(f"{timer.stop():.5f} sec")
