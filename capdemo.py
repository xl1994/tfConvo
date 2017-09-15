#  -*- Coding: UTF-8 -*-
#!/usr/bin/env python

import tensorflow as tf
import TFRec as tfr
import numpy as np
# import tensorflow.examples.tutorials.mnist.input_data as input_data
# from libs.utils import *
# import matplotlib.pyplot as plt

def weight_variable(shape):
   initial = tf.random_normal(shape,mean=0.0,stddev=0.01)
   return tf.Variable(initial)

def bias_variable(shape):
   initial = tf.random_normal(shape,mean=0.0,stddev=0.01)
   return tf.Variable(initial)

if '__main__' == __name__:
   # mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)
   # adjust acoording to size of the picture
   # define parameters
   img_width = 40
   img_height = 60
   img_pixels = img_width*img_height
   num_class = 62
   channels = 3
   x = tf.placeholder(tf.float32,[None, img_pixels, channels])
   # number of output class
   y = tf.placeholder(tf.float32,[None, num_class])
   
   # construct the conv network
   # conv layer 1
   x_tensor = tf.reshape(x,[-1, img_width, img_height, channels])
   filter_size = 3
   filter_num1 = 16
   w_conv1 = weight_variable([filter_size,filter_size,channels,filter_num1])
   b_conv1 = bias_variable([filter_num1])
   conv1 = tf.nn.conv2d(input=x_tensor,filter=w_conv1,strides=[1,2,2,1],padding='SAME')+b_conv1
   h_conv1 = tf.nn.relu(conv1)
   # conv layer 2
   filter_num2 = 16
   w_conv2 = weight_variable([filter_size,filter_size,filter_num1,filter_num2])
   b_conv2 = bias_variable([filter_num2])
   conv2 = tf.nn.conv2d(input=h_conv1,filter=w_conv2,strides=[1,2,2,1],padding='SAME')+b_conv2
   h_conv2 = tf.nn.relu(conv2)
   
   # add full connection hiden layer, how to modify this
   h_conv2_flat = tf.reshape(h_conv2,[-1,10*15*filter_num2])
   n_fc = 1024
   w_fc1 = weight_variable([10*15*filter_num2,n_fc])
   b_fc1 = bias_variable([n_fc])
   h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat,w_fc1)+b_fc1)
   # add dropout to prevent overfitting
   keep_prob = tf.placeholder(tf.float32)
   h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
   # softmax layer
   w_fc2 = weight_variable([n_fc, num_class])
   b_fc2 = bias_variable([num_class])
   y_pred = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2)+b_fc2)
   # target rule
   cross_entropy = -tf.reduce_sum(y*tf.log(y_pred))
   optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)
   # calc accurate rate
   correct_prediction = tf.equal(tf.argmax(y_pred,1),tf.argmax(y,1))
   accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))
   # start to train with mini-batch
   sess = tf.Session()
   batch_size = 100
   n_epochs = 5
   # imgs, labels = sess.run([imgs, labels])
   # records
   imgs, labels = tfr.read_and_decode('train.tfrecords',batch_size)
   with tf.Session() as sess:
       sess.run(tf.global_variables_initializer())
       threads = tf.train.start_queue_runners(sess=sess)
       for epoch_i in range(n_epochs):
           print 'start train epoch', epoch_i
           for i in range(5):
               batch_xs, labels = sess.run([imgs, labels])
               print batch_xs.shape, labels
               ys = np.zeros([batch_size, num_class])
               for t in range(batch_size):
                   ys[t, labels[t]] = 1
               batch_ys = tf.convert_to_tensor(ys)
               print batch_ys
               sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob:0.5})
               print 'tarin finished batch:',i
