import tensorflow as tf


class Critic(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Critic, self).__init__(**kwargs)
        self.hidden_l1 = tf.keras.layers.Dense(300, activation=tf.nn.relu)
        self.hidden_l2 = tf.keras.layers.Dense(600, activation=tf.nn.relu)
        self.output_l = tf.keras.layers.Dense(1, activation=tf.nn.tanh)

    def call(self, input_tensor, **kwargs):
        x = self.hidden_l1(input_tensor)
        x = self.hidden_l2(x)
        return self.output_l(x)

    def build_graph(self, input_shape, to_file):
        x = tf.keras.Input(shape=input_shape)
        model = tf.keras.Model(inputs=x, outputs=self.call(x))
        return tf.keras.utils.plot_model(model, to_file=to_file, show_shapes=True)

    def print_summary(self, input_shape):
        x = tf.keras.Input(shape=input_shape)
        model = tf.keras.Model(inputs=x, outputs=self.call(x))
        return model.summary()

class Actor(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Actor, self).__init__(**kwargs)
        self.hidden_l1 = tf.keras.layers.Dense(300, activation=tf.nn.relu)
        self.hidden_l2 = tf.keras.layers.Dense(600, activation=tf.nn.relu)
        self.output_l = tf.keras.layers.Dense(1, activation=tf.nn.tanh)

    def call(self, input_tensor, **kwargs):
        x = self.hidden_l1(input_tensor)
        x = self.hidden_l2(x)
        return self.output_l(x)

    def print_graph(self, input_shape, to_file):
        x = tf.keras.Input(shape=input_shape)
        model = tf.keras.Model(inputs=x, outputs=self.call(x))
        return tf.keras.utils.plot_model(model, to_file=to_file, show_shapes=True)