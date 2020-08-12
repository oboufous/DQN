'''
## Ops ##
# Common ops for the network
@author: Mark Sinton (msinto93@gmail.com) 
'''


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow as tf2

def conv2d(inputs, kernel_size, filters, stride, activation=None, use_bias=True, scope='conv'):
    with tf.variable_scope(scope):
        if use_bias:
            return tf.layers.conv2d(inputs, filters, kernel_size, stride, 'valid', activation=activation, 
                                    use_bias=use_bias, kernel_initializer=tf2.initializers.GlorotUniform())
        else:
            return tf.layers.conv2d(inputs, filters, kernel_size, stride, 'valid', activation=activation, 
                                    use_bias=use_bias, kernel_initializer=tf2.initializers.GlorotUniform(),
                                    bias_initializer=None)
            
def batchnorm(inputs, is_training, scope='batch_norm'):
    with tf.variable_scope(scope):
        return tf.layers.batch_normalization(inputs, momentum=0.9, gamma_initializer=tf.random_normal_initializer(1.0, 0.02),
                                         training=is_training, fused=True)

def dense(inputs, output_size, activation=None, scope='dense'):
    with tf.variable_scope(scope):
        return tf.layers.dense(inputs, output_size, activation=activation, kernel_initializer=tf2.initializers.GlorotUniform())

def flatten(inputs, scope='flatten'):
    with tf.variable_scope(scope):
        return tf.layers.flatten(inputs)