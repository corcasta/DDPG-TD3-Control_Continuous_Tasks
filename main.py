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
import sys
import gym
import matplotlib.pyplot as plt
from ddpg import DDPGAgent
from td3 import TD3Agent
from utils import *
from her import HER
import csv
# *****************************************************************************************
import os
from gym import utils
from gym.envs.robotics import fetch_env

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'reach.xml')


class FetchReachEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.10,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.5,
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=False, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)


# *****************************************************************************************
"""
class FetchWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

     def observation(self, observation):
         observation =
"""


class FetchWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        state_ = obs['observation']
        desired_goal = obs['desired_goal']
        achieved_goal = obs['achieved_goal']
        return state_, desired_goal, achieved_goal

    def step(self, act):
        observation, reward, done, info = self.env.step(act)
        state, desired_goal, achieved_goal = self.observation(observation)
        return state, desired_goal, achieved_goal, reward, done, info

    def observation(self, obs):
        state = obs['observation']
        desired_goal = obs['desired_goal']
        achieved_goal = obs['achieved_goal']
        return state, desired_goal, achieved_goal


# env = gym.make('FetchReach-v1')
# env = FetchReachEnv()
# env = gym.make('Pendulum-v0')
env = FetchWrapper(gym.make('FetchPush-v1'))
# env = FetchWrapper(gym.make('FetchPush-v1'))

# epochs = 200
# cycles = 50
episodes = 30000
steps = 50  # 1000
opt_steps = 50 #optimizer_step
batch_size = 128  # 128  # 64
# ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(0.2) * np.ones(1))
agent = DDPGAgent(env, long_memory_size=1000000, short_memory_size=1000) # DDPG
#agent = TD3Agent(env, long_memory_size=1000000, short_memory_size=1000) # TD3
rewards = []
avg_rewards = []

k = 4
# episode_memory = ReplayBuffer(size=steps)
"""
for epoch in range(epochs):
    success_counter = 0
    for cycle in range(cycles):
"""
for episode in range(episodes):
    state, desired_goal, achieved_goal = env.reset()
    """ 
    print('Episode: ', episode)
    print('state: ', state)
    print('desired_goal: ', desired_goal)
    print('achieved_goal: ', achieved_goal)
    #episode_memory = ReplayBuffer(size=steps)
    """
    # For all environments outside robotics section OpenAI GYM
    # state = env.reset()
    # episode_memory = ReplayBuffer(size=steps)
    episode_reward = 0
    #ou_noise.reset()
    agent.short_memory.clear()

    for step in range(steps):
        """ 
        # For all environments outside robotics section OpenAI GYM
        action = agent.get_action(state, ou_noise)
        new_state, reward, done, info = env.step(action)
        # env.render()
        """


        #action = agent.get_action(np.concatenate((state, desired_goal)), ou_noise) # DDPG
        action = agent.get_action(np.concatenate((state, desired_goal)))  # TD3
        action = np.squeeze(action)
        new_state, desired_goal, achieved_goal, reward, done, info = env.step(action)
        """ 
        print("Step: ", step)
        print("Action: ", action)
        print("New State: ", new_state)
        print("Desired Goal: ", desired_goal)
        print("Achieved Goal: ", achieved_goal)
        print("Rewards: ", reward)
        print("Done: ", done)
        print("Info: ", info)
        env.render()
        """
        env.render()
        
        """ 
        # For all environments outside robotics section OpenAI GYM
        agent.memory.push(state, action,
                          reward, new_state,
                          done, info)

        if len(agent.memory) > batch_size:
            agent.train(batch_size)
        """

        agent.long_memory.push(np.concatenate((state, desired_goal)), action, reward,
                               np.concatenate((new_state, desired_goal)), done, info,
                               achieved_goal)

        agent.short_memory.push(np.concatenate((state, desired_goal)), action, reward,
                                np.concatenate((new_state, desired_goal)), done, info,
                                achieved_goal)

        state = new_state
        episode_reward += reward

        if done:
            if episode == 0:
                sys.stdout.write(
                    "episode: {}, reward: {}, average _reward: {} \n".format(episode,
                                                                             np.round(episode_reward,
                                                                                      decimals=2),
                                                                             "nan"))
            else:
                sys.stdout.write(
                    "episode: {}, long_mem_len: {}, shor_mem_len: {}, reward: {}, average _reward: {} \n".format(
                        episode,
                        len(
                            agent.long_memory),
                        len(
                            agent.short_memory),
                        np.round(episode_reward,
                                 decimals=2),
                        np.mean(rewards[-10:])))
            break
    """
    if episode % 10 == 0:
       agent.actor.save_weights('/home/corcasta/Documents/DDPG-TD3-Control_Continuous_Tasks/Weights/Fetch_Push/DDPG/norm/Test_1/actor_weights')
       agent.critic.save_weights('/home/corcasta/Documents/DDPG-TD3-Control_Continuous_Tasks/Weights/Fetch_Push/DDPG/norm/Test_1/critic_weights')
       agent.target_actor.save_weights('/home/corcasta/Documents/DDPG-TD3-Control_Continuous_Tasks/Weights/Fetch_Push/DDPG/norm/Test_1/target_actor_weights')
       agent.target_critic.save_weights('/home/corcasta/Documents/DDPG-TD3-Control_Continuous_Tasks/Weights/Fetch_Push/DDPG/norm/Test_1/target_critic_weights')

    rewards.append(episode_reward)
    avg_rewards.append(np.mean(rewards[-10:]))
    agent.generate_artificial_transitions(k=4, strategy='future')
    #print('Short Memory len: ', len(agent.short_memory))
    #print('Short Memory len: ', len(agent.long_memory))
    for _ in range(opt_steps):
        if len(agent.long_memory) > batch_size:
            agent.train(batch_size)
            # print('Epoch: {}, Cycle: {}, Episode: {}, Average Reward: {}, Success_rate: {}'.format(epoch, cycle, episode, avg_rewards[-1], success_rate[-1]))
    # success_rate.append(success_counter/(cycles*episodes))

data = zip(rewards, avg_rewards)
filepath = 'ddpg_her_rewards_FetchPush_BatchNorm_1.csv'
with open(filepath, "w") as f:
    writer = csv.writer(f)
    writer.writerow(('rewards', 'avg_rewards'))
    for row in data:
        writer.writerow(row)

plt.plot(rewards)
plt.plot(avg_rewards)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Success Rate')
plt.show()
"""











"""
    state, action, _, next_state, done, info, achieved_goal = episode_memory.sample(batch_size=steps, random_=False)
    # print('len of episode_memory: ', len(episode_memory))
    for step in range(len(episode_memory)):
        for i in range(k):
            # state, action, np.array([reward]), next_state, done, info, achieved_goal
            if step == len(episode_memory):
                future = len(episode_memory) - 1
            else:
                future = np.random.randint(step, len(episode_memory))
            # print('future =', future)
            recycle_state = state[step][0:-3]
            recycle_action = action[step]
            recycle_next_state = next_state[step][0:-3]
            recycle_done = done[step]
            recycle_info = info[step]
            recycle_achieved_goal = achieved_goal[step]
            recycle_goal = achieved_goal[future]
            recycle_reward = env.compute_reward(recycle_achieved_goal, recycle_goal, recycle_info)

            agent.long_memory.push(np.concatenate((recycle_state, recycle_goal)), recycle_action, recycle_reward,
                                   np.concatenate((recycle_next_state, recycle_goal)), recycle_done, recycle_info,
                                   recycle_achieved_goal)

            # if recycle_reward == 0:
            #    print('Succes tuple =', recycle_state, recycle_goal, recycle_action, recycle_reward,
            #          recycle_next_state, recycle_goal, recycle_done, recycle_info,
            #          recycle_achieved_goal)

            # print('Reward = ', reward)
        if len(agent.long_memory) > batch_size:
            agent.train(batch_size)

        # if episode % 10 == 0:
        #    agent.actor.save_weights('/home/corcasta/Thesis/DDPG-TD3-Control_Continuous_Tasks/weights/actor_weights')
        #    agent.critic.save_weights('/home/corcasta/Thesis/DDPG-TD3-Control_Continuous_Tasks/weights/critic_weights')
        #    agent.target_actor.save_weights('/home/corcasta/Thesis/DDPG-TD3-Control_Continuous_Tasks/weights/target_actor_weights')
        #    agent.target_critic.save_weights('/home/corcasta/Thesis/DDPG-TD3-Control_Continuous_Tasks/weights/target_critic_weights')
    #rewards.append(episode_reward)
    #avg_rewards.append(np.mean(rewards[-10:]))
plt.plot(rewards)
plt.plot(avg_rewards)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()
"""
