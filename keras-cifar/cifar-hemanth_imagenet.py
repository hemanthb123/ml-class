from __future__ import print_function
import os
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

import wandb
from wandb.keras import WandbCallback

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']
num_classes = len(class_names)


X_train_resized = np.zeros((X_train.shape[0],299,299,3))
for i in range(X_train.shape[0]):
    X_train_resized[i] =cv2.resize(X_train[i], (299,299), interpolation = cv2.INTER_CUBIC)
    X_train_resized[i] = X_train_resized[i].astype('float32') / 255.   #if float needs to be between 0 and 1, if int between 0 and 255

X_test_resized = np.zeros((X_test.shape[0],299,299,3))
for i in range(X_test.shape[0]):
    X_test_resized[i] =cv2.resize(X_test[i], (299,299), interpolation = cv2.INTER_CUBIC)
    X_test_resized[i] = X_test_resized[i].astype('float32') / 255.   #if float needs to be between 0 and 1, if int between 0 and 255
  

print('Predicted InceptionV3')
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
model1 = InceptionV3(weights='imagenet')

for i in range(10):
    x = image.img_to_array(X_test_resized)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds1 = model1.predict(x)
    
    plt.subplot(10,1,i+1)
    plt.imshow(X_test_resized[i], cmap='gray', interpolation='none')
    plt.title(class_names[int(y_test[i+offset])])
    
    print('Predicted InceptionV3:', decode_predictions(preds1, top=3)[0])




print('Predicted InceptionV3')
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
model1 = InceptionV3(weights='imagenet')
img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
#model.summary()
preds1 = model1.predict(x)
print('Predicted InceptionV3:', decode_predictions(preds1, top=3)[0])

print('Predicted ResNet50')
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
model2 = ResNet50(weights='imagenet')
img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
#model.summary()
preds2 = model2.predict(x)
print('Predicted ResNet50:', decode_predictions(preds2, top=3)[0])