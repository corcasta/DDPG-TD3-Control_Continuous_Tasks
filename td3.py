from models import *
from utils import *


class TD3Agent:
    """ Build up agent based on DDPG algorithm

    Args:
        env (:obj: env): instance of the environment
        hidden_size_l1 (int): number of nodes of first hidden layer for both networks
        hidden_size_l2 (int): number of nodes of second hidden layer for both networks
        actor_optimizer (str): tensorflow optimizer alias for actor network
        critic_optimizer (str): tensorflow optimizer alias for critic network
        actor_learning_rate (float): rate at which actor network learns
        critic_learning_rate (float): rate at which critic network learns
        gamma (int): discount factor that determines how much the agent cares about future and immediate rewards
        tau (float): rate at which target networks gets updated
        long_memory_size (int): max capacity of experiences that main buffer stores.
        short_memory_size (int): max capacity of experiences that secondary buffer stores it is intended to be used with HER.
        iterations (int): Every d iterations, the policy is updated with respect to Qθ1
    """

    def __init__(self, env, hidden_size_l1=256, hidden_size_l2=256, actor_optimizer='Adam', critic_optimizer='Adam',
                 actor_learning_rate=0.0001, critic_learning_rate=0.001, iterations=2, gamma=0.99, tau=0.01,
                 long_memory_size=100000, short_memory_size=1000):
        # Class basic attributes
        # print('observation_space: ', env.observation_space["observation"].shape[0])
        # self.__state_space = env.observation_space["observation"].shape[0]
        # self.__state_space = env.observation_space.shape[0]
        self.env = env
        self.tau = tau
        self.gamma = gamma
        self.time_step = 1
        self.iterations = iterations
        self.action_space = env.action_space.shape[0]
        self.upper_bound = env.action_space.high[0]
        self.lower_bound = env.action_space.low[0]

        # Initializing Actor-Critic & target networks
        self.actor = Actor(hidden_size_l1=hidden_size_l1, hidden_size_l2=hidden_size_l2, output_size=self.action_space)
        self.critic = Critic(hidden_size_l1=hidden_size_l1, hidden_size_l2=hidden_size_l2)
        self.critic_2 = Critic(hidden_size_l1=hidden_size_l1, hidden_size_l2=hidden_size_l2)
        self.target_actor = Actor(hidden_size_l1=hidden_size_l1, hidden_size_l2=hidden_size_l2,
                                  output_size=self.action_space)
        self.target_critic = Critic(hidden_size_l1=hidden_size_l1, hidden_size_l2=hidden_size_l2)
        self.target_critic_2 = Critic(hidden_size_l1=hidden_size_l1, hidden_size_l2=hidden_size_l2)

        #self.actor.load_weights('/home/corcasta/Documents/DDPG-TD3-Control_Continuous_Tasks/Weights/Fetch_Reach/TD3/norm/Test_1/actor_weights')
        #self.critic.load_weights('/home/corcasta/Documents/DDPG-TD3-Control_Continuous_Tasks/Weights/Fetch_Reach/TD3/norm/Test_1/critic_weights')
        #self.critic_2.load_weights('/home/corcasta/Documents/DDPG-TD3-Control_Continuous_Tasks/Weights/Fetch_Reach/TD3/norm/Test_1/critic_weights')

        # Assigning target networks same weights as Actor & Critic networks
        self._update_target_networks(tau=1)

        # Setting up Actor & Critic optimizers
        self.actor_optimizer = tf.keras.optimizers.get(identifier={"class_name": actor_optimizer,
                                                                   "config": {"learning_rate": actor_learning_rate}})
        self.critic_optimizer = tf.keras.optimizers.get(identifier={"class_name": critic_optimizer,
                                                                    "config": {"learning_rate": critic_learning_rate}})
        self.long_memory = ReplayBuffer(size=long_memory_size)
        self.short_memory = ReplayBuffer(size=short_memory_size)

    def get_action(self, state):
        """ Retrieve agents action

        :param state:
        :param noise_object:
        :return:
        """
        state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        noise = np.random.normal(0, 0.1)
        sample_action = tf.squeeze(self.actor(state))
        sample_action = sample_action.numpy() + noise
        legal_action = np.clip(sample_action, self.lower_bound, self.upper_bound)
        # return np.squeeze(legal_action)
        # print('squeeze:', [np.squeeze(legal_action)])
        return [np.squeeze(legal_action)]

    def train(self, batch_size):
        """ Initiate agent training

        :param batch_size:
        :return:
        """
        state_batch, action_batch, reward_batch, next_state_batch, _, _, _ = self.long_memory.sample(
            batch_size=batch_size)

        state_batch = tf.convert_to_tensor(state_batch, dtype=tf.float64)
        action_batch = tf.convert_to_tensor(action_batch, dtype=tf.float64)
        reward_batch = tf.convert_to_tensor(reward_batch, dtype=tf.float64)
        next_state_batch = tf.convert_to_tensor(next_state_batch, dtype=tf.float64)

        # Critic loss
        with tf.GradientTape(persistent=True) as tape:
            noise = np.clip(np.random.normal(0, 0.2), -0.5, 0.5)
            target_critic_value = self.target_critic([next_state_batch, self.target_actor(next_state_batch) + noise])
            target_critic_value_2 = self.target_critic_2([next_state_batch, self.target_actor(next_state_batch) + noise])
            #print(target_critic_value)
            min_target_critic_value = np.minimum(target_critic_value, target_critic_value_2)
            target_critic_value = reward_batch + self.gamma * min_target_critic_value
            critic_value = self.critic([state_batch, action_batch])
            critic_value_2 = self.critic_2([state_batch, action_batch])
            critic_loss = tf.keras.losses.MSE(target_critic_value, critic_value)
            critic_loss_2 = tf.keras.losses.MSE(target_critic_value, critic_value_2)
        # Calculate gradients for each trainable variable (weights) of model
        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        critic_gradients_2 = tape.gradient(critic_loss_2, self.critic_2.trainable_variables)
        # Update trainable variables from model with new values through defined optimizer
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_gradients_2, self.critic_2.trainable_variables))

        # Actor/Policy loss
        if self.time_step % self.iterations == 0:
            with tf.GradientTape() as tape:
                critic_value = self.critic([state_batch, self.actor(state_batch)])
                actor_loss = -tf.math.reduce_mean(
                    critic_value)  # "-" because it's going to be applied gradient "ascent"
            # Calculate gradients for each trainable variable (weights) of model
            actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
            # Update trainable variables from model with new values through defined optimizer
            self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
            # Update target networks
            self._update_target_networks()
        self.time_step += 1

    def generate_artificial_transitions(self, k, strategy='future'):
        if strategy == 'future':
            self._future_strategy(k)

        elif strategy == 'episode':
            self._episode_strategy(k)

        elif strategy == 'random':
            self._random_strategy(k)

    def _update_target_networks(self, tau=None):
        if tau is None:
            tau = self.tau

        self.target_actor.set_weights(np.array(tau) * self.actor.get_weights() +
                                      (np.array(1 - tau)) * self.target_actor.get_weights())

        self.target_critic.set_weights(np.array(tau, dtype=float) * self.critic.get_weights() +
                                       (np.array((1 - tau), dtype=float) * self.target_critic.get_weights()))

        self.target_critic_2.set_weights(np.array(tau, dtype=float) * self.critic_2.get_weights() +
                                         (np.array((1 - tau), dtype=float) * self.target_critic_2.get_weights()))

    def _future_strategy(self, k):
        state, action, _, next_state, done, info, achieved_goal = self.short_memory.sample(
            batch_size=len(self.short_memory),
            random_=False)
        # print('len of episode_memory: ', len(episode_memory))
        for step in range(len(self.short_memory)):
            for i in range(k):
                # state, action, np.array([reward]), next_state, done, info, achieved_goal
                if step == len(self.short_memory):
                    future = len(self.short_memory) - 1
                else:
                    future = np.random.randint(step, len(self.short_memory))
                # print('future =', future)
                recycle_state = state[step][0:-3]
                recycle_action = action[step]
                recycle_next_state = next_state[step][0:-3]
                recycle_done = done[step]
                recycle_info = info[step]
                recycle_achieved_goal = achieved_goal[step]
                recycle_goal = achieved_goal[future]
                new_reward = self.env.compute_reward(recycle_achieved_goal, recycle_goal, recycle_info)
                """ 
                print("********************************")
                print("Recycle State: ", recycle_state)
                print("Recycle Action: ", recycle_action)
                print("Recycle Next State: ", recycle_next_state)
                print("Recycle Done: ", recycle_done)
                print("Recycle Info: ", recycle_info)
                print("Recycle Achieved Goal: ", recycle_achieved_goal)
                print("Recycle Goal: ", recycle_goal)
                print("New Reward: ", new_reward)
                print("")
                """
                self.long_memory.push(np.concatenate((recycle_state, recycle_goal)), recycle_action,
                                      new_reward,
                                      np.concatenate((recycle_next_state, recycle_goal)), recycle_done,
                                      recycle_info,
                                      recycle_achieved_goal)

    def _episode_strategy(self, k):
        pass

    def _random_strategy(self, k):
        pass

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
