import tensorflow as tf

tf.keras.backend.set_floatx('float64')


class Actor(tf.keras.Model):
    def __init__(self, hidden_size_l1, hidden_size_l2, output_size, **kwargs):
        super(Actor, self).__init__(**kwargs)
        self.input_bnorm_l = tf.keras.layers.BatchNormalization()
        self.hidden_bnorm_l = tf.keras.layers.BatchNormalization()
        self.hidden_l1 = tf.keras.layers.Dense(hidden_size_l1, activation=tf.nn.relu)
        self.hidden_l2 = tf.keras.layers.Dense(hidden_size_l2, activation=tf.nn.relu)
        self.output_l = tf.keras.layers.Dense(output_size, activation=tf.nn.tanh)

    @tf.function
    def call(self, state_input, **kwargs):
        x = self.input_bnorm_l(state_input)
        x = self.hidden_l1(x)
        x = self.hidden_bnorm_l(x)
        x = self.hidden_l2(x)
        x = self.hidden_bnorm_l(x)
        return self.output_l(x)

    def print_graph(self, input_shape, to_file):
        x = tf.keras.Input(shape=input_shape)
        model = tf.keras.Model(inputs=x, outputs=self.call(x))
        return tf.keras.utils.plot_model(model, to_file=to_file, show_shapes=True)

    def print_summary(self, input_shape):
        x = tf.keras.Input(shape=input_shape)
        model = tf.keras.Model(inputs=x, outputs=self.call(x))
        return model.summary()


class Critic(tf.keras.Model):
    def __init__(self, hidden_size_l1, hidden_size_l2, **kwargs):
        super(Critic, self).__init__(**kwargs)
        self.input_bnorm_l = tf.keras.layers.BatchNormalization()
        self.hidden_bnorm_l = tf.keras.layers.BatchNormalization()
        self.hidden_l1 = tf.keras.layers.Dense(hidden_size_l1, activation=tf.nn.relu)
        self.hidden_l2 = tf.keras.layers.Dense(hidden_size_l2, activation=tf.nn.relu)
        self.output_l = tf.keras.layers.Dense(1, activation=None)

    @tf.function
    def call(self, state_action_input, **kwargs):
        x = tf.concat(state_action_input, 1)
        x = self.input_bnorm_l(x)
        x = self.hidden_l1(x)
        x = self.hidden_bnorm_l(x)
        x = self.hidden_l2(x)
        x = self.hidden_bnorm_l(x)
        return self.output_l(x)

    def build_graph(self, input_shape, to_file):
        x = tf.keras.Input(shape=input_shape)
        model = tf.keras.Model(inputs=x, outputs=self.call(x))
        return tf.keras.utils.plot_model(model, to_file=to_file, show_shapes=True)

    def print_summary(self, input_shape):
        x = tf.keras.Input(shape=input_shape)
        model = tf.keras.Model(inputs=x, outputs=self.call(x))
        return model.summary()
