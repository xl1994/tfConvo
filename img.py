# -*- Coding: UTF-8 -*-
from PIL import Image
import numpy as np
import os
from PIL import ImageEnhance
def genData(num=500):
   data = list()
   # number of imgs per character
   path = os.getcwd()
   number =['0','1','2','3','4','5','6','7','8','9']
   alphabet_lower = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
   alphabet_upper = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
   SET = number + alphabet_lower + alphabet_upper
   labels = np.zeros((num*62,62))
   for index,item in enumerate(SET):
       for i in range(num):
           filename = path + '/captcha_img/' +item+'/label_'+str(i)+'.jpg'
           im = Image.open(filename).convert('L')
           enhancer = ImageEnhance.Contrast(im)
           im = enhancer.enhance(6)
           im = np.array(im.convert('1'),'f')
           im = im.reshape(im.shape[0]*im.shape[1])
           data.append(im)
           labels[index*num+i,index] = 1
   data = np.array(data)
   return data, labels

def next_batch(b_size):
   
