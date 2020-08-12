'''
## Network ##
# Defines the DQN network - architecture, inference and training step
@author: Mark Sinton (msinto93@gmail.com) 
'''

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from utils.ops import conv2d, dense, flatten

class DeepQNetwork:
    def __init__(self, num_actions, state, action=None, target=None, learning_rate=None, scope='DQN'):
        # State - Input state to pass through the network
        # Action - Action for which the Q value should be predicted (only required for training)
        # Target - Target Q value (only required for training)
        self.input = state
        self.action = action
        self.target = target
        self.num_actions = num_actions
        self.scope = scope
        if learning_rate is not None:
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=0.95, epsilon=0.01)

        with tf.variable_scope(self.scope):

            with tf.variable_scope('input_layers'):
                self.input_float =  tf.to_float(self.input)
                self.input_norm = tf.divide(self.input_float, 255.0)

            self.conv1 = conv2d(self.input_norm, 8, 16, 4, tf.nn.relu, scope='conv1')  
            self.conv2 = conv2d(self.conv1, 4, 32, 2, tf.nn.relu, scope='conv2')
            self.flatten = flatten(self.conv2, scope='flatten')
            self.dense1 = dense(self.flatten, 256, tf.nn.relu, scope='dense1')
            self.dense2 = dense(self.dense1, 256, tf.nn.relu, scope='dense2')
            self.dense3 = dense(self.dense2, 256, tf.nn.relu, scope='dense3')
            self.output = dense(self.dense3, self.num_actions, scope='output')

        self.network_params = tf.trainable_variables(scope=self.scope)

    def predict(self):
        with tf.variable_scope(self.scope):
            with tf.variable_scope('predict'):
                Q_action = tf.argmax(self.output, axis=1)
                return Q_action

    def train_step(self):
        with tf.variable_scope(self.scope):
            with tf.variable_scope('train'):
                Q_vals = self.output
                action_one_hot = tf.one_hot(self.action, self.num_actions)
                Q_val_action = tf.reduce_sum(Q_vals * action_one_hot, reduction_indices=1)     # We only want the Q value for the action that was taken

                self.loss = tf.losses.huber_loss(self.target, Q_val_action, reduction=tf.losses.Reduction.MEAN)
                train_step = self.optimizer.minimize(self.loss, var_list=self.network_params)

                return train_step
