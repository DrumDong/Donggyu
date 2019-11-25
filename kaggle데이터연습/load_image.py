# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 16:31:46 2019

@author: ehdrb
"""

import tensorflow as tf
from PIL import Image
import glob
import numpy as np

"""이미지 로드후 넘파이 배열로 변환"""
def load_train_image():
    image_width = 64
    image_height = 64
    image_label = []
    All_Img = []

    image_file_list = glob.glob('C:/Users/ehdrb/Desktop/데이터/catdog/train/*.jpg')

    for img in image_file_list:
        if 'cat' in img:
            image_label.append(0)
        elif 'dog' in img:
            image_label.append(1)
        image = Image.open(img)
        image = image.resize((image_width,image_height))
        All_Img.append(np.float32(image))
    
    #image_label[12500:]=1
    label = np.array(image_label)
    All_Img = np.array(All_Img)
                
    return All_Img,label

def load_test_image():
    image_width = 64
    image_height = 64
    image_label = []
    All_Img = []

    image_file_list = glob.glob('C:/Users/ehdrb/Desktop/데이터/catdog/test/*.jpg')

    for img in image_file_list:
        if 'cat' in img:
            image_label.append(0)
        elif 'dog' in img:
            image_label.append(1)
        image = Image.open(img)
        image = image.resize((image_width,image_height))
        All_Img.append(np.float32(image))
    
    #image_label[12500:]=1
    label = np.array(image_label)
    All_Img = np.array(All_Img)
                
    return All_Img,label
            



