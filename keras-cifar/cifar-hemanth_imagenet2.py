from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
import glob
import argparse
import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten ,  GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D

from keras import __version__
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

from keras.optimizers import SGD
import pickle
import wandb
from wandb.keras import WandbCallback

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']
num_classes = len(class_names)

num1 = 1000 #X_train.shape[0]
X_train_resized = np.zeros((num1,299,299,3))
for i in range(num1):
    X_train_resized[i] =cv2.resize(X_train[i], (299,299), interpolation = cv2.INTER_CUBIC)
    X_train_resized[i] = X_train_resized[i].astype('float32') / 255.   #if float needs to be between 0 and 1, if int between 0 and 255

num2 = 1000
X_test_resized = np.zeros((num2,299,299,3))
for i in range(num2):
    X_test_resized[i] =cv2.resize(X_test[i], (299,299), interpolation = cv2.INTER_CUBIC)
    X_test_resized[i] = X_test_resized[i].astype('float32') / 255.   #if float needs to be between 0 and 1, if int between 0 and 255



f = open('X_train_resized.pckl', 'wb')
pickle.dump(X_train_resized, f)
f.close()
f = open('X_test_resized.pckl', 'wb')
pickle.dump(X_test_resized, f)
f.close()


f = open('X_train_resized.pckl', 'rb')
X_train_resized=pickle.load(f)
f.close()
f = open('X_test_resized.pckl', 'rb')
X_test_resized=pickle.load(f)
f.close()

'''
# just copy 32x32 to 299x299 image, rest all zeros
X_train_resized = np.zeros((X_train.shape[0],299,299,3))
for i in range(X_train.shape[0]):
    X_train_resized[i,0:32,0:32,:] = X_train[i]

X_test_resized = np.zeros((X_test.shape[0],299,299,3))
for i in range(X_test.shape[0]):
    X_test_resized[i,0:32,0:32,:] = X_test[i]
'''

print('Predicted InceptionV3')
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
model1 = InceptionV3(weights='imagenet')


for i in range(100):
    x = image.img_to_array(X_test_resized[i])
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds1 = model1.predict(x)
    #preds1 = model1.predict(X_test_resized[i])
   #plt.subplot(10,1,i+1)
   #plt.imshow(X_test_resized[i], cmap='gray', interpolation='none')
   #plt.title(class_names[int(y_test[i])] ,'Predicted InceptionV3:', decode_predictions(preds1, top=2)[0])
    print('Actual:',class_names[int(y_test[i])], 'Predicted InceptionV3:', decode_predictions(preds1, top=2)[0])

