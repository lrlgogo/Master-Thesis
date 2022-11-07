""
""
""
"==================    IMPORT        ===================================="
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import initializers

import numpy as np

import matplotlib.pyplot as plt
import pickle
import os
"==================    FUNC DEF      ===================================="
def kernel_initial(u=2):

    kernel = np.zeros(2 * (2*u - 1,), dtype='float32')
    kernel[u-1:, :u] = np.ones(2 * (u,), dtype='float32')

    return kernel

def activation(x, input_mat):
    
    a = np.where(x > 1.1, 0, 1)
    a = 0.5 * (a + input_mat)
    
    return np.where(a > 0.8, 1, 0)

def funConv(A, K):

    x = A.reshape((1,) + A.shape + (1,))

    conv = Conv2D(1, K.shape[0], padding='same',
                  kernel_initializer=initializers.zeros,
                  use_bias=False)
    conv.build(x.shape)
    conv.kernel = K
    y = np.reshape(conv(x), A.shape)
    y = np.where(y > 1., 2, y)

    return y

def funPFRec(input_mat, u):

    if input_mat.dtype != 'float32':
        input_mat = np.asarray(input_mat, dtype='float32')
    k = input_mat.shape

    kernel = kernel_initial(u)
    kernel = kernel.reshape(kernel.shape + (1, 1))

    w = 0
    A = np.copy(input_mat)
    while w * (u - 1) < (k[0] - 1):
        C = funConv(A, kernel)
        A = C
        w += 1

    y = activation(A, input_mat)

    return y

"==================    TEST          ===================================="
file_dir = r'D:\lrl\workspace\MOP\ParetoPattern\test_pic\test'
file_name = 'test_2.dat'  # 'input.dat'
def test(u=7, file_name=file_name, file_dir=file_dir):
    fp = open(os.path.join(file_dir, file_name), 'rb')

    A = pickle.load(fp)
    fp.close()

    y1 = funPFRec(A, u)
    # print(y1)

    plt.subplot(121)
    plt.imshow(A, cmap='Greys')
    plt.subplot(122)
    plt.imshow(y1, cmap='Greys')
    plt.show()

    return 0
