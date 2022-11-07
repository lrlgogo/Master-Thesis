""
""
""
"==================    IMPORT        ===================================="
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import initializers

import numpy as np

import matplotlib.pyplot as plt
"==================    CONST DEC     ===================================="
B_PF = np.array([
    [0, 0, 0],
    [1, 1, 0],
    [1, 1, 0]], dtype='float32')

PF_kernel = B_PF.reshape(B_PF.shape + (1, 1))
"==================    FUNC DEF      ===================================="
def activation(x, input_mat):
    
    a = np.where(x > 1., 0, 1)
    a = 0.5 * (a + input_mat)
    return np.where(a > 0.5, 1, 0)

def funPFRec3(input_mat, kernel=PF_kernel):

    if input_mat.dtype != 'float32':
        input_mat = np.asarray(input_mat, dtype='float32')
    x = input_mat.reshape((1,) + input_mat.shape + (1,))
    
    conv = Conv2D(1, 3, padding='same', kernel_initializer=initializers.zeros,
              use_bias=False)
    
    conv.build(x.shape)
    conv.kernel = kernel

    y = np.reshape(conv(x), input_mat.shape)
    y = activation(y, input_mat)

    return y

def funPFRecAll(input_mat):

    if input_mat.dtype != 'float32':
        input_mat = np.asarray(input_mat, dtype='float32')
    kernel_shape = 2 * np.array(input_mat.shape) - 1
    kernel = np.zeros(kernel_shape, dtype='float32')
    for i in range(input_mat.shape[0] - 1, kernel.shape[0]):
        for j in range(input_mat.shape[1]):
            kernel[i, j] = 1.
    kernel = kernel.reshape(kernel.shape + (1, 1))
    
    x = input_mat.reshape((1,) + input_mat.shape + (1,))
    
    conv = Conv2D(1, kernel.shape[0], padding='same', kernel_initializer=initializers.zeros,
              use_bias=False)
    
    conv.build(x.shape)
    conv.kernel = kernel

    y = np.reshape(conv(x), input_mat.shape)
    y = activation(y, input_mat)

    return y, conv

"==================    MAIN          ===================================="
A = np.array([
    [1, 1, 1],
    [0, 1, 1],
    [0, 1, 0]], dtype='float32')

y = funPFRec3(A)
print(y)
y1, conv = funPFRecAll(A)
print(y1)
