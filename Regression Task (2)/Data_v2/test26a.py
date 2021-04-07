"""

This is for morphological classification of galaxies by CNN,
New regresssion for B/D ratio
By Kenji Bekki, on 2018/3/30

"""


import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils import np_utils
from keras.models import model_from_json
### Added 2018/3/30
from keras.applications import imagenet_utils
from keras.models import load_model
###
import keras.callbacks
import numpy as np
#import keras.backend.tensorflow_backend as KTF
#import tensorflow as tf
import os.path

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)
### Total model number = (100*1) * nmodel

#iset=int(input('Input the total number of sets of models '))
#nmodel0=int(input('Input the total number of images per model '))
#nmodel=nmodel0*iset
#epochs=int(input('Input the number of epochs'))
#iset=5
#nmodel0=1000
epochs=3000
#nmodel=nmodel0*iset
nmodel=1000
print('nmodel',nmodel)

### Original values
#batch_size = 128
#num_classes = 10
#epochs = 12
batch_size = 1
num_classes = 2
#epochs = 500
nb_epoch=epochs
n_mesh=50
#n_mesh=20
#nmodel=4000

img_rows, img_cols = n_mesh, n_mesh
n_mesh2=n_mesh*n_mesh-1
n_mesh3=n_mesh*n_mesh


print(img_rows, img_cols, n_mesh2)
#stop



#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

input_shape = (img_rows, img_cols, 1)

#x_train = x_train.astype('float32')
#x_test = x_test.astype('float32')
#x_train /= 255
#x_test /= 255
#print('x_train shape:', x_train.shape)
#print(x_train.shape[0], 'train samples')
#print(x_test.shape[0], 'test samples')
#print(y_test.shape[0], 'y.test samples')
#print(str(y_test[0]))
#print(str(y_test[1]))
#print(str(y_test[2]))

#y_train = y_train.astype('int32')
#y_test = y_test.astype('int32')
#y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
#y_test =  keras.utils.np_utils.to_categorical(y_test, num_classes)

# This is for simlation data sets

with open('2dft.dat') as f:
  lines=f.readlines()
with open('2dftn1.dat') as f:
  lines1=f.readlines()
with open('2dftn2.dat') as f:
  lines2=f.readlines()


x_train=np.zeros((nmodel,n_mesh3))
x_test=np.zeros((nmodel,n_mesh3))
#y_train=np.zeros(nmodel,dtype=np.int)
#y_test=np.zeros(nmodel,dtype=np.int)
y_train=np.zeros((nmodel,2))
y_test=np.zeros((nmodel,2))
#y_test=np.zeros(nmodel)
#print(y_train)

# For 2D density map data
ibin=0
jbin=-1
for num,j in enumerate(lines):
  jbin=jbin+1
  tm=j.strip().split()
  x_train[ibin,jbin]=float(tm[0])
  x_test[ibin,jbin]=float(tm[0])
#  print('ibin,jbin',ibin,jbin)
  if jbin == n_mesh2:
    ibin+=1
    jbin=-1

# For morphological map (theta)
ibin=0
for num,j in enumerate(lines1):
  tm=j.strip().split()
  y_train[ibin,0]=float(tm[0])
  y_test[ibin,0]=float(tm[0])
#  y_train[ibin]=int(tm[0])-1
#  y_test[ibin]=int(tm[0])-1
#  print('ibin, (Morpholigcl type)',ibin,y_train[ibin])
  ibin+=1

# For morphological map (phi)
ibin=0
for num,j in enumerate(lines2):
  tm=j.strip().split()
  y_train[ibin,1]=float(tm[0])
  y_test[ibin,1]=float(tm[0])
#  y_train[ibin]=int(tm[0])-1
#  y_test[ibin]=int(tm[0])-1
#  print('ibin, (Morpholigcl type)',ibin,y_train[ibin])
  ibin+=1



x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
# For laelling
#y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
#y_test =  keras.utils.np_utils.to_categorical(y_test, num_classes)

print('Galaxy type',y_train[:5])

#stop

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

### For labelling of morphological types

#model.add(Dense(num_classes, activation='softmax'))
#model.compile(loss=keras.losses.categorical_crossentropy,
#              optimizer=keras.optimizers.Adadelta(),
#              metrics=['accuracy'])

### For regression of B/D

model.add(Dense(2, activation='linear'))
#model.add(Dense(1))
#model.add(Activation=('linear'))
#model.add(activation=('linear'))
model.compile(loss='mean_squared_error',
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])


print('save the architecture of a model')

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")




