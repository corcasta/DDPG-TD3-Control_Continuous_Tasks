import tensorflow as tf


class Critic(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Critic, self).__init__(**kwargs)
        self.hidden_l1 = tf.keras.layers.Dense(300, activation=tf.nn.relu)
        self.hidden_l2 = tf.keras.layers.Dense(600, activation=tf.nn.relu)
        self.output_l = tf.keras.layers.Dense(1, activation=tf.nn.tanh)

    def call(self, inputs, **kwargs):
        x = self.hidden_l1(inputs)
        x = self.hidden_l2(x)
        return self.output_l(x)


class Actor(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Actor, self).__init__(**kwargs)
        self.hidden_l1 = tf.keras.layers.Dense(300, activation=tf.nn.relu)
        self.hidden_l2 = tf.keras.layers.Dense(600, activation=tf.nn.relu)
        self.output_l = tf.keras.layers.Dense(1, activation=tf.nn.tanh)

    def call(self, inputs, **kwargs):
        x = self.hidden_l1(inputs)
        x = self.hidden_l2(x)
        return self.output_l(x)
