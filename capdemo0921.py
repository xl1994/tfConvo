#  -*- Coding: UTF-8 -*-
#!/usr/bin/env python

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
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
   mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)
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
   # w_conv1 = weight_variable([filter_size,filter_size,1,filter_num1])
   # b_conv1 = bias_variable([filter_num1])
   # conv1 = tf.nn.conv2d(input=x_tensor,filter=w_conv1,strides=[1,2,2,1],padding='SAME')+b_conv1
   # h_conv1 = tf.nn.relu(conv1)
   # h_conv1  ---   20*30*16 ? h_conv1 has a channel of 16
   h_conv1 = add_conv_layer(filter_size, filter_num1, channel, [1,2,2,1], x_tensor)
   # conv layer 2
   filter_num2 = 16
   # w_conv2 = weight_variable([filter_size,filter_size,filter_num1,filter_num2])
   # b_conv2 = bias_variable([filter_num2])
   # conv2 = tf.nn.conv2d(input=h_conv1,filter=w_conv2,strides=[1,2,2,1],padding='SAME')+b_conv2
   # h_conv2 = tf.nn.relu(conv2)
   # h_conv2   ---   10*15*16?
   # input h_conv1 has channel 16, kernel must has the same channel as input
   # therefor the 3rd parameter is filter_num1 rather than 1
   h_conv2 = add_conv_layer(filter_size, filter_num2, filter_num1, [1,2,2,1], h_conv1)
   
   # add full connection hiden layer, how to modify this
   h_conv2_flat = tf.reshape(h_conv2,[-1,7*7*filter_num2])
   n_fc = 1024
   # w_fc1 = weight_variable([7*7*filter_num2,n_fc])
   # b_fc1 = bias_variable([n_fc])
   # h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat,w_fc1)+b_fc1)
   h_fc1 = add_fcon_layer(h_conv2_flat, 7*7*filter_num2, n_fc)
   
   # add dropout to prevent overfitting
   keep_prob = tf.placeholder(tf.float32)
   h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
   
   # softmax layer
   w_fc2 = weight_variable([n_fc,10])
   b_fc2 = bias_variable([10])
   y_pred = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2)+b_fc2)
   
   # target rule
   cross_entropy = -tf.reduce_sum(y*tf.log(y_pred))
   optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)
   
   # calc accurate rate
   correct_prediction = tf.equal(tf.argmax(y_pred,1),tf.argmax(y,1))
   accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))
   
   # start to train with mini-batch
   sess = tf.Session()
   sess.run(tf.initialize_all_variables())
   batch_size = 100
   n_epochs = 5
   # records
   h = []
   for epoch_i in range(n_epochs):
       for batch_i in range(mnist.train.num_examples/batch_size):
           batch_xs, batch_ys = mnist.train.next_batch(batch_size)
           sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob:0.5})
           h.append(sess.run(accuracy, feed_dict={x: mnist.validation.images,y:mnist.validation.labels,keep_prob:1.0}))
           print sess.run(accuracy, feed_dict={x: mnist.validation.images,y:mnist.validation.labels,keep_prob:1.0})
