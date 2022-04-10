import tensorflow as tf
import numpy as np
import time


for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
del gpu

n = 64
m = 32
iteration = 10000

w = 1.1
c1, c2 = 2, 2
r1, r2 = np.random.uniform(0, 1, (2,))

v_max = 0.01
x_range = (-10, 10)

start = time.time()

X = tf.random.uniform((n, m), *x_range)
V = tf.random.uniform((n, m), -v_max, v_max)

F = tf.reduce_sum(X ** 2, 1)
p_best = tf.Variable(X)
index_best = tf.argmin(F)
global_min = tf.reduce_min(F)
g_best = tf.Variable(X[index_best])

for index in range(iteration):

    V = w * V + c1 * r1 * (p_best - X) + c2 * r2 * (g_best - X)
    V = tf.clip_by_value(V, -v_max, v_max)
    X += V
    X = tf.clip_by_value(X, *x_range)

    F_pre = tf.Variable(F)
    F = tf.reduce_sum(X ** 2, 1)
    temp = tf.where(F > F_pre, 0, 1)

    # temp_0 = np.zeros(p_best)
    for i in range(n):
        if temp[i] == 1:
            p_best[i].assign(X[i])

    temp = tf.reduce_min(F)
    if temp < global_min:
        global_min = temp
        index_best = tf.argmin(F)
        g_best = tf.Variable(X[index_best])

    # print('{}--{}'.format(index, global_min))
end = time.time()
print('time: ', end - start)
