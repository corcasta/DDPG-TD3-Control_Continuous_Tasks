import numpy as np
import tensorflow as tf
import random
from collections import deque


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
                self.x_prev
                + self.theta * (self.mean - self.x_prev) * self.dt
                + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def get_buffer(self):
        return self.buffer

    def push(self, state, action, reward, next_state, done, info, achieved_goal=None):
        if achieved_goal is None:
            self.buffer.append((state, action, np.array([reward]), next_state, done, info))
        else:
            self.buffer.append((state, action, np.array([reward]), next_state, done, info, achieved_goal))

    def clear(self):
        return self.buffer.clear()

    def sample(self, batch_size, random_=True):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []
        info_batch = []
        achieved_goal_batch = []

        if random_ is True:
            batch = random.sample(self.buffer, batch_size)
        else:
            batch = self.buffer

        counter = 1  # This counter is just necessary when sampling in order,
        for experience in batch:
            state_batch.append(experience[0])
            action_batch.append(experience[1])
            reward_batch.append(experience[2])
            next_state_batch.append(experience[3])
            done_batch.append(experience[4])
            info_batch.append(experience[5])
            achieved_goal_batch.append(experience[6])
            if random_ is False and counter == batch_size:
                break
            counter += 1

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch, info_batch, achieved_goal_batch

    def __len__(self):
        return len(self.buffer)
