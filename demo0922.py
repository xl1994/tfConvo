#  -*- Coding: UTF-8 -*-
#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import tfr
# import tensorflow.examples.tutorials.mnist.input_data as input_data
# from libs.utils import *
# import matplotlib.pyplot as plt

def weight_variable(shape):
   initial = tf.random_normal(shape,mean=0.0,stddev=0.01)
   return tf.Variable(initial)

def bias_variable(shape):
   initial = tf.random_normal(shape,mean=0.0,stddev=0.01)
   return tf.Variable(initial)

def add_conv_layer(filter_size, filter_num, channel, stride, input_tensor):
   # w is the kernel, channel must equal to channel of the image
   # filter_num is the number of kernel
   w = weight_variable([filter_size, filter_size, channel, filter_num])
   b = bias_variable([filter_num])
   conv = tf.nn.conv2d(input=input_tensor, filter=w, strides=stride, padding='SAME') + b
   return tf.nn.relu(conv)

def add_fcon_layer(input_flat, input_len, num_fcon):
   w = weight_variable([input_len, num_fcon])
   b = bias_variable([num_fcon])
   return tf.nn.relu(tf.matmul(input_flat, w) + b)

if '__main__' == __name__:
   # mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)
   # adjust acoording to size of the picture
   # width = 40; height = 60
   # here the size of picture is 40*60, so there is 2400
   img_pixels = 2400
   channel = 3
   x = tf.placeholder(tf.float32,[None, img_pixels, channel])
   # number of output class  62
   num_class =62
   y = tf.placeholder(tf.float32,[None, num_class])
   
   # construct the conv network
   # conv layer 1
   x_tensor = tf.reshape(x,[-1,40,60,3])
   filter_size = 3
   filter_num1 = 16
   # h_conv1  ---   20*30*16 ? h_conv1 has a channel of 16
   h_conv1 = add_conv_layer(filter_size, filter_num1, channel, [1,2,2,1], x_tensor)
   # conv layer 2
   filter_num2 = 16
   # h_conv2   ---   10*15*16?
   # input h_conv1 has channel 16, kernel must has the same channel as input
   # therefor the 3rd parameter is filter_num1 rather than 1
   h_conv2 = add_conv_layer(filter_size, filter_num2, filter_num1, [1,2,2,1], h_conv1)
   
   # add full connection hiden layer, how to modify this
   h_conv2_flat = tf.reshape(h_conv2,[-1,10*15*filter_num2])
   n_fc = 1024
   h_fc1 = add_fcon_layer(h_conv2_flat, 10*15*filter_num2, n_fc)
   
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
   batch_size = 5000
   n_epochs = 5
   
   # start trainning
   # how to train
   imgs, labels = tfr.read_and_decode('train.tfrecords',batch_size)
   with tf.Session() as sess:
       sess.run(tf.global_variables_initializer())
       coord = tf.train.Coordinator()
       threads = tf.train.start_queue_runners(coord=coord)
       for i in range(1):
           batch_xs, batch_ys = sess.run([imgs, labels])
           print 'batch_xs:', batch_xs.shape
           batch_ys = tf.one_hot(batch_ys, tfr.num_class, 1, 0).eval()
           sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob:0.5})
           print 'Training finished: ', i
           # p_x, p_y = sess.run([imgs, labels])
           # print 'p_y:', p_y
           # p_y = tf.one_hot(p_y, tfr.num_class, 1, 0).eval()
           # print batch_ys
           # print sess.run(correct_prediction, feed_dict={x: batch_xs, y: batch_ys, keep_prob:1.0})
           print sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob:1.0})
