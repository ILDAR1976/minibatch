import sys
#Here you need to specify the path to the folder where the project folder is located
sys.path.insert(0, 'F:\\projects\\python\\DeepLearning\\DeepLearningIha')

from minibatch.datatools import input_data
mnist = input_data.read_data_sets("data/", one_hot=True)

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import time, shutil, os

def get_minibatch():
    minibatch_x, minibatch_y = mnist.train.next_batch(batch_size)
    return minibatch_x

if __name__ == '__main__':
    batch_size = 100
    x = tf.placeholder(tf.float32,shape=[None,784],name="x")
    W = tf.Variable(tf.random_uniform([784,10],-1,1),name = "W")
    b = tf.Variable(tf.zeros([10]), name = "biases")
    output = tf.matmul(x, W) + b
    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)
    feed_dict_source = {x: get_minibatch()}
    res = sess.run(output,feed_dict=feed_dict_source)
    print(res)
