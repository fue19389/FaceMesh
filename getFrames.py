import matplotlib.pyplot as plt
import numpy as np

x = np.array([0,0,-1,0,0])
y = np.array([1,2,3,0,0])
x = x.astype(int)
y = x.astype(int)

i = 0


# while True:
#
#     val = int(input('No. de 0 a 5'))
#     if val == 1:
#         y[i] = y[i - 1]
#         x[i] = x[i - 1]
#
#     if val == 4:
#         y[i] = y[i - 1] + 1
#         x[i] = x[i - 1]
#
#     if val == 0:
#         y[i] = y[i - 1]
#         x[i] = x[i - 1] - 1
#
#     if val == 2:
#         y[i] = y[i - 1]
#         x[i] = x[i - 1] + 1
#
#     if val == 3:
#         y[i] = y[i - 1] + 1
#         x[i] = x[i - 1] - 1
#
#     if val == 5:
#         y[i] = y[i - 1] + 1
#         x[i] = [i - 1] + 1
#
#     i = i + 1
#     print(x)
#     print(y)
#     print(i)
plt.plot(x, y)
plt.show()

