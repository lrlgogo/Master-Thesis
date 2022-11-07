import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


conv3d = Conv3D(
    1, 9, padding='same', use_bias=False, input_shape=(5, 5, 5, 1)
)
conv3d.build(input_shape=(5, 5, 5, 1))

input = np.zeros((5, 5, 5, 1), np.float32)
for i in range(5):
    for j in range(5):
        for k in range(5):
            if i + j + k > 3:
                input[i, j, k, 0] = 1

u, v, w = [[], [], []]
for i in range(5):
    for j in range(5):
        for k in range(5):
            if input[i, j, k, 0]:
                u.append(i)
                v.append(j)
                w.append(k)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(u, v, w, marker='o')
#plt.show()

kernel = np.zeros((9, 9, 9), np.float32)
kernel[: 5, : 5, : 5] = np.ones((5, 5, 5), np.float32)
conv3d.kernel = np.reshape(kernel, kernel.shape + (1, 1))

x = np.expand_dims(input, 0)
y = conv3d(x)
y = np.where(y > 1.1, 0, 1)
y = 0.5 * (x + y)
y = np.where(y > 0.8, 1, 0)
u, v, w = [[], [], []]
for i in range(5):
    for j in range(5):
        for k in range(5):
            if y[0, i, j, k, 0] > 0.5:
                u.append(i)
                v.append(j)
                w.append(k)

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
ax.scatter(u, v, w, marker='x')
#plt.show()

conv3d_1 = Conv3D(
    1, 3, padding='same', input_shape=(5, 5, 5, 1)
)
conv3d_1.build(input_shape=(5, 5, 5, 1))
kernel_3 = np.zeros((3, 3, 3), np.float32)
kernel_3[: 2, : 2, : 2] = np.ones((2, 2, 2), np.float32)
conv3d_1.kernel = np.reshape(kernel_3, kernel_3.shape + (1, 1))

y = np.copy(x)
for num in range(4):
    y = conv3d_1(y)
y = np.where(y > 1.1, 0, 1)
y = 0.5 * (x + y)
y = np.where(y > 0.8, 1, 0)
u, v, w = [[], [], []]
for i in range(5):
    for j in range(5):
        for k in range(5):
            if y[0, i, j, k, 0] > 0.5:
                u.append(i)
                v.append(j)
                w.append(k)

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
ax.scatter(u, v, w, marker='o', c='r', s=50)
plt.show()
