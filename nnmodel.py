#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 23:45:58 2017

@author: manish
"""
import pandas as pd
import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm

TRAIN_DIR = '/home/manish/Manish/Machine_Learning/Kaggle/Cifar-10/train'
TEST_DIR = '/home/manish/Manish/Machine_Learning/Kaggle/Cifar-10/test'
IMG_SIZE = 32
LR = 0.001

MODEL_NAME = 'cifar_10-{}-{}.model'.format(LR, '6-conv-basic')

labels_df = pd.read_csv('/home/manish/Manish/Machine_Learning/Kaggle/Cifar-10/trainLabels.csv')
labels_arr = []
for i in range(0, len(labels_df)):
    labels_arr += [[labels_df.iloc[i,0], labels_df.iloc[i,1]]]

def label_image(img):
    label_no = int(img.split('.')[0])
    if  labels_arr[label_no-1][1] == 'airplane':
        return [1,0,0,0,0,0,0,0,0,0]
    elif labels_arr[label_no-1][1] == 'automobile':
        return [0,1,0,0,0,0,0,0,0,0]
    elif labels_arr[label_no-1][1] == 'bird':
        return [0,0,1,0,0,0,0,0,0,0]
    elif labels_arr[label_no-1][1] == 'cat':
        return [0,0,0,1,0,0,0,0,0,0]
    elif labels_arr[label_no-1][1] == 'deer':
        return [0,0,0,0,1,0,0,0,0,0]
    elif labels_arr[label_no-1][1] == 'dog':
        return [0,0,0,0,0,1,0,0,0,0]
    elif labels_arr[label_no-1][1] == 'frog':
        return [0,0,0,0,0,0,1,0,0,0]
    elif labels_arr[label_no-1][1] == 'horse':
        return [0,0,0,0,0,0,0,1,0,0]
    elif labels_arr[label_no-1][1] == 'ship':
        return [0,0,0,0,0,0,0,0,1,0]
    elif labels_arr[label_no-1][1] == 'truck':
        return [0,0,0,0,0,0,0,0,0,1]
    
def create_train_set():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_image(img)
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

def create_test_set():
    test_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        test_data.append([np.array(img), img_num])
    shuffle(test_data)
    np.save('test_data.npy', test_data)
    return test_data

#train_data = create_train_set()   #Saved the data

train_data = np.load('train_data.npy')


import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5 , activation = 'relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation = 'relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet,0.8)

convnet = fully_connected(convnet, 10, activation='softmax')
convnet = regression(convnet, optimizer= 'adam', learning_rate = LR, loss= 'categorical_crossentropy',name='targets')

model = tflearn.DNN(convnet, tensorboard_dir = 'log')

if os.path.exists('/home/manish/Manish/Machine_Learning/Kaggle/DogvsCats/{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('Model Loaded')
    
    
train = train_data[:-1200]
test = train_data[-1200:]

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE, IMG_SIZE, 1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_y = [i[1] for i in test]

model.fit({'input':X}, {'targets' : Y}, n_epoch = 3, validation_set= ({'input':test_x},{'targets':test_y}), snapshot_step =500, show_metric = True, run_id= MODEL_NAME)

model.save(MODEL_NAME)

#test_data = create_test_set()
test_data = np.load('test_data.npy')

labels = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6:'frog', 7:'horse', 8:'ship', 9: 'truck'}


import matplotlib.pyplot as plt
fig = plt.figure()
for num, data in enumerate(test_data[:12]):
    img_num = data[1]
    img_data = data[0]
    
    y = fig.add_subplot(3,4,num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
    
    model_out = model.predict([data])[0]
    
    str_label = labels[np.argmax(model_out)]
        
    y.imshow(orig)
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
    
plt.show()


with open('submission_file.csv', 'w') as f:
    f.write('id,label\n')
    
with open('submission_file.csv', 'a') as f:
    for data in tqdm(test_data):
        img_num = data[1]
        img_data = data[0]
        orig = img_data
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
        model_out = model.predict([data])[0]
        str_label = labels[np.argmax(model_out)]
        f.write("{},{}\n".format(img_num, str_label))
        