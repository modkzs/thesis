# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

p1 = tf.placeholder(tf.float32)
p2 = tf.placeholder(tf.float32)

x = p1 + p2

p = tf.train.GradientDescentOptimizer(0.1)
t = [x[0] for x in p.compute_gradients(x, var_list=[p2, p1])]

with tf.Session() as sess:
    print(sess.run(t, {p1: np.random.rand(1, 2), p2: np.random.rand(1, 2)}))
