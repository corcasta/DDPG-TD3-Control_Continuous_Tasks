from models import *
from utils import *


class DDPGAgent:
    def __init__(self, env, hidden_size_l1=256, hidden_size_l2=256, actor_optimizer='Adam', critic_optimizer='Adam',
                 actor_learning_rate=0.0001, critic_learning_rate=0.001, gamma=0.99, tau=0.01,
                 replay_buffer_size=50000):
        # Class basic attributes
        #self.__state_space = env.observation_space["observation"].shape[0]
        self.__state_space = env.observation_space.shape[0]
        self.__action_space = env.action_space.shape[0]
        self.__upper_bound = env.action_space.high[0]
        self.__lower_bound = env.action_space.low[0]
        self.__tau = tau
        self.__gamma = gamma

        # Initializing Actor-Critic & target networks
        self.actor = Actor(hidden_size_l1=hidden_size_l1, hidden_size_l2=hidden_size_l2, output_size=self.__action_space)
        self.critic = Critic(hidden_size_l1=hidden_size_l1, hidden_size_l2=hidden_size_l2)
        self.target_actor = Actor(hidden_size_l1=hidden_size_l1, hidden_size_l2=hidden_size_l2, output_size=self.__action_space)
        self.target_critic = Critic(hidden_size_l1=hidden_size_l1, hidden_size_l2=hidden_size_l2)

        # Assigning target networks same weights as Actor & Critic networks
        self._update_target_networks()

        # Setting up Actor & Critic optimizers
        self.actor_optimizer = tf.keras.optimizers.get(identifier={"class_name": actor_optimizer,
                                                                   "config": {"learning_rate": actor_learning_rate}})
        self.critic_optimizer = tf.keras.optimizers.get(identifier={"class_name": critic_optimizer,
                                                                    "config": {"learning_rate": critic_learning_rate}})
        self.memory = ReplayBuffer(size=replay_buffer_size)

    def get_action(self, state, noise_object):
        state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        noise = noise_object()
        sample_action = tf.squeeze(self.actor(state))
        sample_action = sample_action.numpy() + noise
        legal_action = np.clip(sample_action, self.__lower_bound, self.__upper_bound)
        #return np.squeeze(legal_action)
        return [np.squeeze(legal_action)]

    def train(self, batch_size):
        state_batch, action_batch, reward_batch, next_state_batch, _ = self.memory.sample(batch_size=batch_size)

        state_batch = tf.convert_to_tensor(state_batch, dtype=tf.float64)
        action_batch = tf.convert_to_tensor(action_batch, dtype=tf.float64)
        reward_batch = tf.convert_to_tensor(reward_batch, dtype=tf.float64)
        next_state_batch = tf.convert_to_tensor(next_state_batch, dtype=tf.float64)

        # Actor/Policy loss
        with tf.GradientTape() as tape:
            critic_value = self.critic([state_batch, self.actor(state_batch)])
            actor_loss = -tf.math.reduce_mean(critic_value)  # "-" because it's going to be applied gradient "ascent"
        # Calculate gradients for each trainable variable (weights) of model
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        # Update trainable variables from model with new values through defined optimizer
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

        # Critic loss
        with tf.GradientTape() as tape:
            target_critic_value = reward_batch + self.__gamma * self.target_critic([next_state_batch, self.target_actor(next_state_batch)])
            critic_value = self.critic([state_batch, action_batch])
            critic_loss = tf.keras.losses.MSE(target_critic_value, critic_value)
        # Calculate gradients for each trainable variable (weights) of model
        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        # Update trainable variables from model with new values through defined optimizer
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

        self._update_target_networks(tau=self.__tau)

    def _update_target_networks(self, tau=None):
        if tau is None:
            tau = 1

        self.target_actor.set_weights(np.array(tau) * self.actor.get_weights() +
                                      (np.array(1 - tau)) * self.target_actor.get_weights())

        self.target_critic.set_weights(np.array(tau) * self.critic.get_weights() +
                                       (np.array(1 - tau)) * self.target_critic.get_weights())

    """
    def update_target(self, target_weights, weights):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * self.__tau + a * (1 - self.__tau))

    def update_target_v2(self):
        weights1 = []
        targets1 = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights1.append(weight * self.__tau + targets1[i] * (1 - self.__tau))
        self.target_actor.set_weights(weights1)

        weights2 = []
        targets2 = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights2.append(weight * self.__tau + targets2[i] * (1 - self.__tau))
        self.target_critic.set_weights(weights2)

    def update_target_v3(self):
        weights = []
        target_actor_weights = self.target_actor.get_weights()
        for i, weight in enumerate(self.actor.get_weights()):
            weights.append(self.__tau * weight + (1 - self.__tau) * target_actor_weights[i])
        self.target_actor.set_weights(weights)

        weights = []
        target_critic_weights = self.target_critic.get_weights()
        for i, weight in enumerate(self.critic.get_weights()):
            weights.append(self.__tau * weight + (1 - self.__tau) * target_critic_weights[i])
        self.target_critic.set_weights(weights)
    """
