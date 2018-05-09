import tensorflow as tf
import numpy as np

# Predefined function to build a feedforward neural network
def build_mlp(input_placeholder, 
              output_size,
              scope, 
              n_layers=2, # number of hidden layers
              size=500, 
              activation=tf.tanh,
              output_activation=None
              ):
    out = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out

class NNDynamicsModel(): # (f_theta : normalized state/action -> normalized delta)
    def __init__(self, 
                 env, 
                 n_layers,
                 size, 
                 activation, 
                 output_activation, 
                 normalization,
                 batch_size,
                 iterations,
                 learning_rate,
                 sess
                 ):
        """ YOUR CODE HERE """
        """ Note: Be careful about normalization """
        self.sess = sess
        self.normalization = normalization
        self.iterations = iterations
        self.batch_size = batch_size
        input_size = env.obs_dim + env.action_space.shape[0]
        output_size = env.obs_dim

        # prediction
        self.input_placeholder = tf.placeholder(tf.float32, shape=(None, input_size))
        self.model = build_mlp(self.input_placeholder, output_size, 'NNDynamicsModel', n_layers, size, activation, output_activation) # (N, obs_dim)

        # optimization
        self.target_placeholder = tf.placeholder(tf.float32, shape=(None, output_size))
        self.loss = tf.reduce_mean(tf.nn.l2_loss(self.model - self.target_placeholder))
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def fit(self, data):
        """
        Write a function to take in a dataset of (unnormalized)states, (unnormalized)actions, (unnormalized)next_states and fit the dynamics model going from normalized states, normalized actions to normalized state differences (s_t+1 - s_t)
        """
        """YOUR CODE HERE """
        epsilon = 1e-10 # prevent divide by zero
        normalized_states = np.vstack([elem['observations'] for elem in data]) # (N, obs_dim)
        normalized_states = (normalized_states - self.normalization[0])/(self.normalization[1] + epsilon)
        normalized_actions = np.vstack([elem['actions'] for elem in data]) # (N, action_dim)
        normalized_actions = (normalized_actions - self.normalization[4])/(self.normalization[5] + epsilon)
        training_input = np.hstack((normalized_states, normalized_actions)) # (N, sum_dim)
        training_target = np.vstack([elem['next_observations'] - elem['observations'] for elem in data])
        training_target = (training_target - self.normalization[2])/(self.normalization[3] + epsilon) # (N, obs_dim)

        for i in range(1, self.iterations + 1):
            random_idx = np.random.choice(training_input.shape[0], self.batch_size, replace=False)
            X_batch, Y_batch = training_input[random_idx], training_target[random_idx]
            feed_dict = {self.input_placeholder : X_batch, self.target_placeholder : Y_batch}
            loss, _ = self.sess.run([self.loss, self.optimizer], feed_dict = feed_dict)
            if (i%5 == 0): # for debugging purposes
                print("Minibatch loss at step %d: %f" %(i, loss))

    def predict(self, states, actions):
        """ Write a function to take in a batch of (unnormalized) states and (unnormalized) actions and return the (unnormalized) next states as predicted by using the model """
        """ YOUR CODE HERE """
        epsilon = 1e-10 # prevent divide by zero
        normalized_states = (states - self.normalization[0])/(self.normalization[1] + epsilon) # (N, obs_dim) or (obs_dim, )
        normalized_states = np.reshape(normalized_states, (-1, normalized_states.shape[-1])) # (N, obs_dim)
        normalized_actions = (actions - self.normalization[4])/(self.normalization[5] + epsilon) # (N, action_dim) or (obs_dim, )
        normalized_actions = np.reshape(normalized_actions, (-1, normalized_actions.shape[-1])) # (N, obs_dim)
        normalized_inputs = np.hstack((normalized_states, normalized_actions)) # (N, sum_dim)
        feed_dict = {self.input_placeholder : normalized_inputs}
        normalized_deltas = self.sess.run(self.model, feed_dict = feed_dict) # (N, obs_dim)
        deltas = self.normalization[2] + self.normalization[3]*normalized_deltas

        return states + deltas # (N, obs_dim)
