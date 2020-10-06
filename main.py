"""
import gym
# from gym.envs.robotics.fetch_env import FetchEnv
# from gym.envs.robotics.fetch.reach import FetchReachEnv

import os
from gym import utils
from gym.envs.robotics import fetch_env

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'reach.xml')


class FetchReachEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.4049,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=False, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)


#env = gym.make("Pendulum-v0")
env = FetchReachEnv()
num_states = env.observation_space['observation'].shape[0]
print("Size of State Space ->  {}".format(num_states))
num_actions = env.action_space
print("Size of Action Space ->  {}".format(num_actions))

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))

for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(observation)
        print(reward)
        print(info)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
env.close()
"""

"""
import tensorflow as tf
import numpy as np
from models import *
actor = Actor(hidden_size_l1=300, hidden_size_l2=600, output_size=4)
actor.build(input_shape=(None, 11))

actor_target = Actor(hidden_size_l1=300, hidden_size_l2=600, output_size=4)
actor_target.build(input_shape=(None, 11))

#actor_target.set_weights(actor.get_weights())
print(actor.get_weights())
print("Bienvenido")
print(actor_target.get_weights())
print("Welcome")
tau = 0.01
actor_target.set_weights(np.array(tau)*actor.get_weights()
                         + (np.array(1-tau))*actor_target.get_weights())
print(actor_target.get_weights())
"""





"""
import tensorflow as tf
a = tf.Variable(5)
b = a
print("a:", a.numpy())
print("b:", b.numpy())

b.assign(10)
print("a:", a.numpy())
print("b:", b.numpy())
"""



"""
import tensorflow as tf
x = tf.keras.optimizers.get(identifier={"class_name": 'Adam',
                                        "config": {"learning_rate": 0.001}})
y = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')
print(x)
print(x._decayed_lr('float32').numpy())
print(y)
print(y._decayed_lr('float32').numpy())
"""




"""
from utils import *


memoria = ReplayBuffer(size=5)
state = [1,2,3,4,5]
memoria.push(state, 10, 20, 30, 1)
memoria.push(state, 20, 20, 30, 1)
memoria.push(state, 30, 20, 30, 1)
memoria.push(state, 40, 20, 30, 1)
memoria.push(state, 50, 20, 30, 1)
print(memoria.buffer)
memoria.push(state, 60, 20, 30, 1)
print(memoria.buffer)
state, action, reward, next_state, done = memoria.sample(batch_size=2)
print(state)
print(action)
print(reward)
print(next_state)
print(done)
"""







"""
from models import Critic, Actor
import gym

env = gym.make("Copy-v0")
states = env.observation_space
print(type(states))
if isinstance(states, gym.spaces.discrete.Discrete):
    print(states)
states = env.observation_space.shape
print(states)

env = gym.make("AirRaid-ram-v0")
states = env.observation_space
print(type(states))
if isinstance(states, gym.spaces.discrete.Discrete):
    print(states)
states = env.observation_space.shape
print(states)

env = gym.make("BipedalWalker-v3")
states = env.observation_space
print(type(states))
print(states)
states = env.observation_space.shape
print(states)

env = gym.make("Acrobot-v1")
states = env.observation_space
print(type(states))
print(states)
states = env.observation_space.shape
print(states)

env = gym.make("Ant-v2")
states = env.observation_space
print(type(states))
print(states)
states = env.observation_space.shape
print(states)

env = gym.make("FetchPickAndPlace-v1")
states = env.observation_space
print(type(states))
print(states)
print(states["observation"])
print(states["observation"].shape)

actions = env.action_space
print(actions)
print(actions.shape[0])
x = actions.shape[0]
print(x)


env = gym.make("Pendulum-v0")
states = env.observation_space
print(type(states))
print(states)
actions = env.action_space
print(actions)
print(actions.shape[0])
x = actions.shape[0]
print(x)
"""


"""
critic_model = Critic(name='critic_model')
print(type(critic_model))
critic_model.build((None, 10))
critic_model.summary()
#tf.keras.utils.plot_model(critic_model, to_file='/home/corcasta/Desktop/critic_model_test.png', show_shapes=True, show_layer_names=False)
critic_model.build_graph(input_shape=(10,), to_file='/home/corcasta/Desktop/critic_model_test.png')
critic_model.print_summary(input_shape=(10,))
#*********************************************************************************
"""


"""
import gym
import tensorflow as tf
import numpy as np
from collections import deque

env = gym.make("Pendulum-v0")
observation = env.reset()
print(observation)
observation = tf.convert_to_tensor(observation)
print(observation)
observation = tf.expand_dims(observation, 0)
print("expand_dims: ",observation)
print("SALTO")

buffer = deque(maxlen=5)
state = env.reset()
print(state)
action = env.action_space.sample()
print(action)
buffer.append((state, action))
print(buffer)
state = env.reset()
print(state)
action = env.action_space.sample()
print(action)
buffer.append((state, action))
print(buffer)

state_batch = []
for memory in buffer:
    state_batch.append(memory[0])
state_batch = tf.convert_to_tensor(state_batch)
print(state_batch)

test = [[1,2,3,4,5]]
test = tf.convert_to_tensor(test)
print(test)
test = tf.squeeze(test)
print(test)
# tf.expand_dims  es lo opuesto a tf.squeeze
"""

"""
import gym
env = gym.make("FetchReach-v1")
lower_bound = env.action_space.low[0]
print(lower_bound)
upper_bound = env.action_space.high[0]
print(upper_bound)
"""