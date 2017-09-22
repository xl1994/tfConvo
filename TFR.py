#  -*- Coding: UTF-8 -*-
'''
THIS IS THE PYTHON MODULE THAT IMPLEMENT BASIC FUNCTION OF CONVERTING PICTURE FILE TO TENDORFLOW TFRECORDS FILE AND
READING AND DECODING TFRECORDS FILE
THE PROGRAM WAS INITIALLY RUN ON A NONGRAPHIC CENTOS MACHINE
'''
import os 
import tensorflow as tf 
from PIL import Image 
# import matplotlib.pyplot as plt 
import numpy as np

# turn off warning
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

number =['0','1','2','3','4','5','6','7','8','9']
alphabet_lower = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
alphabet_upper = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
classes = number + alphabet_lower + alphabet_upper
num_train_examples = 400*62
num_valid_examples = 80*62
img_width = 40
img_height = 60
img_pixels = img_width * img_height
num_class = 62

# In order to use gentfr() function, your picture should be stored in the directory under the current working directory
def gentfr(cwd, classes, filename):
    # cwd -> directory that stores your pictures
    # classes -> all classes needed in the classification program
    # filename -> file name of tfrecords file
    # rewrite labels on 2017/09/22
    # label should be column vector
    label = np.zeros([62,1])
    writer= tf.python_io.TFRecordWriter(filename)
    for index,name in enumerate(classes):
         class_path=cwd+'/'+name+'/'
         for img_name in os.listdir(class_path): 
              img_path=class_path+img_name
              img=Image.open(img_path)
              label[index] = 1
              # img= img.resize((128,128))
              img_raw=img.tobytes()
              # modify the following line change int64_list=tf.train.Int64List(value=[index])) to int64_list=tf.train.Int64List(value=[label]))
              example = tf.train.Example(features=tf.train.Features(feature={'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))}))
              writer.write(example.SerializeToString())
    writer.close()
    return filename

def read_and_decode(filename, b_size):
    global img_height
    global img_width
    global img_pixels
    global class_num
    # BATCH_SIZE = 100
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,features={'label': tf.FixedLenFeature([], tf.int64),'img_raw' : tf.FixedLenFeature([], tf.string)})
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [img_pixels, 3])
    # img.set_shape([img_pixels, 1, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    img_batch, label_batch = tf.train.shuffle_batch([img, label], batch_size=b_size, capacity=2000, min_after_dequeue=1000)
    # label_batch = np.zeros([class_num,b_size])
    # for i in range(b_size):
    #     label_batch[label, i] = 1
    return img_batch, label_batch

def test_records(filename):
    b_size = 400
    image_batch, label_batch = read_and_decode(filename, b_size)
    with tf.Session() as sess:
         sess.run(tf.global_variables_initializer())
         threads = tf.train.start_queue_runners(sess = sess)
         for i in range(5):
              val, label = sess.run([image_batch, label_batch])
              print val.shape, label.shape

def fetchData(filename, b_size, mode=1):
    # generate one-hot lobels
    # mode = 1, parse label as column vectors
    # mode = 0, parse lable as row vectors
    image_batch, label_batch = read_and_decode(filename, b_size)
    with tf.Session() as sess:
         sess.run(tf.global_variables_initializer())
         threads = tf.train.start_queue_runners(sess = sess)
         img, label = sess.run([image_batch, label_batch])
    if made == 1:
         label_array = np.zeros([num_class,b_size])
    else:
         label_array = np.zeros([b_size,num_class])
    for i in range(b_size):
         if mode == 1:
              label_array[label[i],i] = 1
         else:
              label_array[i,label[i]] = 1
    return img, label_array
  
def one_hot_labels(labels):
 
 
if '__main__' == __name__:
    # if os.path.exists('train.tfrecords'):
    #     os.remove('train.tfrecords')
    # if os.path.exists('valid.tfrecords'):
    #     os.remove('valid.tfrecords')
    # path = os.getcwd()+'/train'
    # train_file = gentfr(path,classes,'train.tfrecords')
    # valid_file = gentfr(path,classes,'valid.tfrecords')

    img, label = fetchData('train.tfrecords',100)
    print img.shape
    print label.shape
    for i in range(100):
         print label[:,i]
         print '\n'
