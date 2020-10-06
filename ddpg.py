import gym
import numpy as np
import tensorflow as tf
from models import *
from utils import *


class DDPGAgent:
    def __init__(self, env, hidden_size_l1=300, hidden_size_l2=600, actor_optimizer='Adam', critic_optimizer='Adam',
                 actor_learning_rate=0.0001, critic_learning_rate=0.001, gamma=0.99, tau=0.01,
                 replay_buffer_size=50000):
        # Class basic attributes
        self.__state_space = env.observation_space["observation"].shape[0]
        self.__action_space = env.action_space.shape[0]
        self.__upper_bound = env.action_space.high[0]
        self.__lower_bound = env.action_space.low[0]
        self.__tau = tau
        self.__gamma = gamma

        # Initializing Actor-Critic & target networks
        self.actor = Actor(hidden_size_l1=hidden_size_l1, hidden_size_l2=hidden_size_l2,
                           output_size=self.__action_space)
        self.critic = Critic(hidden_size_l1=hidden_size_l1, hidden_size_l2=hidden_size_l2)
        self.target_actor = Actor(hidden_size_l1=hidden_size_l1, hidden_size_l2=hidden_size_l2,
                                  output_size=self.__action_space)
        self.target_critic = Critic(hidden_size_l1=hidden_size_l1, hidden_size_l2=hidden_size_l2)

        # Assigning target networks same weights as Actor & Critic networks
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.target_critic.get_weights())

        # Setting up Actor & Critic optimizers
        self.actor_optimizer = tf.keras.optimizers.get(identifier={"class_name": actor_optimizer,
                                                                   "config": {"learning_rate": actor_learning_rate}})
        self.critic_optimizer = tf.keras.optimizers.get(identifier={"class_name": critic_optimizer,
                                                                    "config": {"learning_rate": critic_learning_rate}})
        self.memory = ReplayBuffer(size=replay_buffer_size)

    def get_action(self, state):
        state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        sample_action = tf.squeeze(self.actor(state))
        legal_action = np.clip(sample_action, self.__lower_bound, self.__upper_bound)
        return [np.squeeze(legal_action)]

    @tf.function
    def update(self, batch_size):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(batch_size=batch_size)

        # Critic loss
        with tf.GradientTape() as tape:
            y = reward_batch + self.__gamma * self.target_critic([next_state_batch, self.target_actor(next_state_batch)])
            critic_value = self.critic([state_batch, action_batch])
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
        # Calculate gradients for each trainable variable (weights) of model
        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        # Update trainable variables from model with new values through defined optimizer
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

        # Actor/Policy loss
        with tf.GradientTape() as tape:
            critic_value = self.critic([state_batch, self.actor(state_batch)])
            actor_loss = -tf.math.reduce_mean(critic_value)  # "-" because it's going to be applied gradient "ascent"
        # Calculate gradients for each trainable variable (weights) of model
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        # Update trainable variables from model with new values through defined optimizer
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

        self._update_target_network(self.actor, self.target_actor)
        self._update_target_network(self.critic, self.target_critic)

    @tf.function
    def _update_target_network(self, base_network, target_network):
        target_network.set_weights(np.array(self.__tau)*base_network.get_weights() +
                                   (np.array(1-self.__tau))*target_network.get_weights())





