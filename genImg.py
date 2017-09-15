#  -*- Coding: UTF-8 -*-
import os
from captcha.image import ImageCaptcha
import numpy as np
from PIL import Image
import tensorflow as tf
import random
# import time

NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ALPHABET_LOWER = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',\
 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
ALPHABET_UPPER = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',\
 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

def gen_captcha_text(char_set=NUMBER+ALPHABET_LOWER+ALPHABET_UPPER, captcha_size=4):
    '''
    NOT WELL DEFINED YET
    '''
    CAPTCHA_TEXT = []
    for i in range(captcha_size):
        C = random.choice(char_set)
        CAPTCHA_TEXT.append(C)
    return CAPTCHA_TEXT
    
def gen_captcha_data(captcha_text):
    '''
    NOT WELL DEFINED YET
    '''
    img = ImageCaptcha()
    captcha_text = ' '.join(captcha_text)
    captcha_data = img.generate(captcha_text)
    captcha_data = Image.open(captcha_data)
    captcha_data = np.array(captcha_data)
    return captcha_text, captcha_data

# IMAGE DATE TO TFRECORDS
def img_to_tfrecords(output_filename, input_directory, classes, width=128, height=128):
    '''
    CLASS OF IMAGE
    '''
    writer = tf.python_io.TFRecordWriter(output_filename)
    for index, name in enumerate(classes):
        class_path = input_directory + '/' + name
        for img_name in os.listdir(class_path):
            img_path = class_path + '/' + img_name
            img = Image.open(img_path)
            # img = img.resize(width, height)
            img_raw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={\
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),\
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))}))
            writer.write(example.SerializeToString())
    writer.close()
    return output_filename

def imgGen(num, charset, dstr):
    flag = True
    class_set = set()
    try:
         for item in charset:
              classes_path = os.getcwd() + '/' + dstr + '/' + item
              if not item in class_set:
                   class_set.add(item)
              else:
                   continue
              if not os.path.exists(classes_path):
                   os.makedirs(dstr+ '/' + item)
              for i in range(num):
                   FILE_NAME = classes_path + '/label_' + str(i) + '.jpg'
                   ImageCaptcha().write(item, FILE_NAME)
                   img = Image.open(FILE_NAME)
                   region = (0,0,img.size[0]/4,img.size[1])
                   img = img.crop(region)
                   img.save(FILE_NAME)
    except Exception as e:
         print str(e)
         flag = False
    return flag

def imgTrain(num, charset):
    return imgGen(num, charset, 'train')

def imgValidation(num, charset):
    return imgGen(num, charset, 'valid')

if __name__ == '__main__':
    # number of sample each character
    num_train = 400
    num_valid = 80
    charset = NUMBER + ALPHABET_LOWER + ALPHABET_UPPER
    if imgTrain(num_train, charset):
         print 'Train: each charatcter',num_train,'images generated!'
    if imgValidation(num_valid, charset):
         print 'Validation: each charatcter',num_valid,'images generated!'

