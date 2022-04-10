import collections
import gym
import numpy as np
import tensorflow as tf
import tqdm

from matplotlib import pyplot as plt
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple

# set gpu memory dynamic growth
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
del gpu

# environment
env = gym.make("CartPole-v0")

# set seed for experiment reproducibility
seed = 43
env.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

# small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()


# model
class ActorCritic(tf.keras.Model):
    def __init__(
            self,
            num_actions: int,
            num_hidden_units: int):
        super().__init__()

        self.common_0 = layers.Dense(num_hidden_units, activation='relu')
        self.common_1 = layers.Dense(num_hidden_units, activation='relu')
        self.actor = layers.Dense(num_actions)
        self.critic = layers.Dense(1)

    def call(self, inputs: tf.Tensor, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.common_0(inputs)
        x = self.common_1(x)
        return self.actor(x), self.critic(x)


num_actions = env.action_space.n
num_hidden_units = 64

model = ActorCritic(num_actions, num_hidden_units)


# training
# 1 collecting training data
def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """return state, reward, done flag"""
    state, reward, done, _ = env.step(action)
    return (state.astype(np.float32),
            np.array(reward, np.int32),
            np.array(done, np.int32))


def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
    return tf.numpy_function(
        env_step,
        [action],
        [tf.float32, tf.int32, tf.int32]
    )


def run_episode(
        initial_state: tf.Tensor,
        model: tf.keras.Model,
        max_steps: int
) -> List[tf.Tensor]:
    """runs a single episode to collect training data."""

    action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

    initial_state_shape = initial_state.shape
    state = initial_state

    for t in tf.range(max_steps):
        # convert state into a batched tensor
        state = tf.expand_dims(state, 0)

        # run the model and to get action probabilities and critic value
        action_logits_t, value = model(state)

        # sample next action from the action probability distribution
        action = tf.random.categorical(action_logits_t, 1)[0, 0]
        action_probs_t = tf.nn.softmax(action_logits_t)

        # store critic values
        values = values.write(t, tf.squeeze(value))

        # store log probability of the action chosen
        action_probs = action_probs.write(t, action_probs_t[0, action])

        # apply action to the environment to get next state and reward
        state, reward, done = tf_env_step(action)
        state.set_shape(initial_state_shape)

        # store reward
        rewards = rewards.write(t, reward)

        if tf.cast(done, tf.bool):
            break

    action_probs = action_probs.stack()
    values = values.stack()
    rewards = rewards.stack()

    return [action_probs, values, rewards]


def get_expected_return(
        rewards: tf.Tensor,
        gamma: float,
        standardize: bool = True
) -> tf.Tensor:
    """compute expected returns per timestep"""

    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)

    # start from the end of 'rewards' and accumulate reward sums
    # into the 'returns' array
    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    discounted_sum = tf.constant(0.0)
    discounted_sum_shape = discounted_sum.shape
    for i in tf.range(n):
        reward = rewards[i]
        discounted_sum = reward + gamma * discounted_sum
        discounted_sum.set_shape(discounted_sum_shape)
        returns = returns.write(i, discounted_sum)
    returns = returns.stack()[::-1]

    if standardize:
        returns = ((returns - tf.math.reduce_mean(returns)) /
                   (tf.math.reduce_std(returns) + eps))

    return returns


huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)


def compute_loss(
        action_probs: tf.Tensor,
        values: tf.Tensor,
        returns: tf.Tensor
) -> tf.Tensor:
    """computes the combined actor-critic loss"""

    advantage = returns - values

    action_log_probs = tf.math.log(action_probs)
    actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

    critic_loss = huber_loss(values, returns)

    return actor_loss + critic_loss


optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


@tf.function
def train_step(
        initial_state: tf.Tensor,
        model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        gamma: float,
        max_steps_per_episode: int
) -> tf.Tensor:
    """runs a model training step"""

    with tf.GradientTape() as tape:

        # run the model for one episode to collect training data
        action_probs, values, rewards = run_episode(
            initial_state, model, max_steps_per_episode
        )

        # calculate expected returns
        returns = get_expected_return(rewards, gamma)

        # convert training data to appropriate TF tensor shapes

        action_probs, values, returns = [
            tf.expand_dims(x, 1) for x in [action_probs, values, returns]
        ]

        # calculating loss values to update our network
        loss = compute_loss(action_probs, values, returns)

    # compute the gradients from the loss
    grads = tape.gradient(loss, model.trainable_variables)

    # apply the gradients to the model's parameters
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    episode_reward = tf.math.reduce_sum(rewards)

    return episode_reward


max_episodes = 10000
max_steps_per_episode = 1000

reward_threshold = 195
running_reward = 0

gamma = 0.99

with tqdm.trange(max_episodes) as t:
    for i in t:
        initial_state = tf.constant(env.reset(), dtype=tf.float32)
        episode_reward = int(
            train_step(
                initial_state, model, optimizer, gamma, max_steps_per_episode
            ).numpy()
        )

        running_reward = episode_reward * 0.01 + running_reward * 0.99

        t.set_description(f'Episode {i}')
        t.set_postfix(
            episode_reward=episode_reward, running_reward=running_reward
        )

        # show average episode reward every 10 episodes
        if i % 10 == 0:
            pass

        if running_reward > reward_threshold:
            break

    print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')
