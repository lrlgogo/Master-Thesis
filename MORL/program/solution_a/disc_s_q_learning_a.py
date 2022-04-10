import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque

max_time = 1000000
max_explore_time = 500
batch_size = 256
learning_rate = 1e-3
alpha = 0.8
gamma = 0.9
initial_epsilon = 1.
final_epsilon = 0.01

state_range = [-1000000., 1000000.]


class QNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=20, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=20, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(units=2)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

    def predict(self, inputs):
        q_values = self(inputs)
        return tf.argmax(q_values, axis=-1)


def fun_SCH(x):
    return x[0] ** 2, (x[0] - 2) ** 2


def fun_non_dominant_procedure(add_one, list_input):
    remove_list = []
    for index, sample in enumerate(list_input):
        if (add_one == sample).all():
            return False
        flag = sum(np.where(add_one - sample >= 0, 1, 0))
        if flag == len(add_one):
            return False
        elif flag == 0:
            remove_list.append(index)
    if remove_list:
        flag = 0
        for index in remove_list:
            del list_input[index - flag]
            flag += 1
            print('DONE!DONE!YET!YET!YET!YET!YET!YET!YET!YET!')
    list_input.append(add_one)
    return True


def fun_reward(val_obj, list_input):
    if fun_non_dominant_procedure(val_obj, list_input):
        return 1
    else:
        return 0


def fun_next_state(x, act, step=0.01):
    return x + (2 * act - 1) * abs(step)


if __name__ == '__main__':
    elite_list = []
    avg_reward_list = []
    elite_count_list = []

    model = QNetwork()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    replay_buffer = deque(maxlen=1000)
    epsilon = initial_epsilon

    state = np.array(
        [random.uniform(state_range[0], state_range[1])],
        dtype='float'
    )
    state = np.array([0.5*state_range[-1]], dtype='float')
    state0 = state
    avg_reward = 0

    for iter_time in range(max_time):
        alpha = 1 / (iter_time + 1)
        val_fun = np.array(fun_SCH(state), dtype='float')
        epsilon = max(
            initial_epsilon * (max_explore_time - (iter_time - batch_size)
                               ) / max_explore_time,
            final_epsilon
        )
        if random.random() < epsilon:
            action = random.randint(0, 1)
            # q_action = np.random.random((2,))
        else:
            # q_action = model(np.expand_dims(state, axis=0)).numpy().squeeze()
            action = model.predict(np.expand_dims(state, axis=0)).numpy()
            action = action[0]

        next_state = np.clip(
            fun_next_state(state, action),
            state_range[0], state_range[1],
        )
        reward = fun_reward(val_fun, elite_list)
        avg_reward += alpha * (reward - avg_reward)
        replay_buffer.append((state, action, reward, next_state))
        state = next_state

        elite_count_list.append(len(elite_list))
        avg_reward_list.append(avg_reward)

        print(iter_time, avg_reward, epsilon, state, action, len(elite_list))

        if len(replay_buffer) >= batch_size:
            batch_state, batch_action, batch_reward, batch_next_state = zip(
                    *random.sample(replay_buffer, batch_size)
                )
            batch_state, batch_reward, batch_next_state = [
                np.array(a, dtype=np.float32) for a in [
                    batch_state, batch_reward, batch_next_state
                ]
            ]
            batch_action = np.array(batch_action, dtype=np.int32)

            q_value = model(batch_next_state)
            y = batch_reward + gamma * tf.reduce_max(q_value, axis=1)
            with tf.GradientTape() as tape:
                loss = tf.keras.losses.mean_squared_error(
                    y_true=y,
                    y_pred=tf.reduce_sum(model(batch_state) * tf.one_hot(
                        batch_action, depth=2), axis=1)
                )
            grads = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads,
                                                         model.variables))

plt.scatter(*np.array(elite_list).T, s=1)
plt.show()
plt.plot(range(len(elite_count_list)), elite_count_list)
plt.show()
plt.plot(range(len(avg_reward_list)), avg_reward_list)
plt.show()
