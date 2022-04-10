""
""
""
"==================    IMPORT        ===================================="
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import initializers

import numpy as np

import matplotlib.pyplot as plt
import pickle
"==================    FUNC DEF      ===================================="
def activation(x, input_mat):
    
    a = np.where(x > 1., 0, 1)
    a = 0.5 * (a + input_mat)
    
    return np.where(a > 0.5, 1, 0)

def funPFRecAll(input_mat):

    if input_mat.dtype != 'float32':
        input_mat = np.asarray(input_mat, dtype='float32')
    kernel_shape = 2 * np.array(input_mat.shape) - 1
    kernel = np.zeros(kernel_shape, dtype='float32')
    """
    for i in range(input_mat.shape[0] - 1, kernel.shape[0]):
        for j in range(input_mat.shape[1]):
            kernel[i, j] = 1.
    """
    k = input_mat.shape
    kernel[k[0] - 1:, : k[1]] = np.ones((k[0], k[1]), dtype='float32')
    kernel = kernel.reshape(kernel.shape + (1, 1))
    
    x = input_mat.reshape((1,) + input_mat.shape + (1,))
    
    conv = Conv2D(1, kernel.shape[0], padding='same', kernel_initializer=initializers.zeros,
              use_bias=False)
    
    conv.build(x.shape)
    conv.kernel = kernel

    y = np.reshape(conv(x), input_mat.shape)
    y = activation(y, input_mat)

    return y
"==================    MAIN          ===================================="
file_dir = r'D:\workspace\DLMOP\ParetoPattern\test_pic\test'
file_name = 'test_1.dat'
def test(file_name, file_dir=file_dir):
    fp = open(os.path.join(file_dir, file_name), 'rb')

    A = pickle.load(fp)

    y1 = funPFRecAll(A)
    print(y1)

    plt.subplot(121)
    plt.imshow(A, cmap='Greys')
    plt.subplot(122)
    plt.imshow(y1, cmap='Greys')
    plt.show()

    return 0
