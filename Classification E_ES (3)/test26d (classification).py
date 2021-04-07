"""

This is for morphological classification of galaxies by CNN,
By Kenji Bekki, on 2020/2/14 for Nair & Abraham 2010

"""


import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils import np_utils
from keras.models import model_from_json
import keras.callbacks
import numpy as np
#import keras.backend.tensorflow_backend as KTF
#import tensorflow as tf
import os.path


### Total model number = (nmodle0) * nmodel

#iset=int(input('Input the total number of sets of models '))
#nmodel0=int(input('Input the total number of images per model'))
#nmodel=nmodel0*iset
nmodel=2000
print('nmodel',nmodel)

### Original values
#batch_size = 128
#num_classes = 10
#epochs = 12
batch_size = 100
#num_classes = 5
num_classes = 2
epochs = 1000
nb_epoch=epochs
n_mesh=50
#nmodel=1000
print('nmodel',nmodel)
print('num_classes',num_classes)

img_rows, img_cols = n_mesh, n_mesh
n_mesh2=n_mesh*n_mesh-1
n_mesh3=n_mesh*n_mesh


print(img_rows, img_cols, n_mesh2)
#stop



#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

input_shape = (img_rows, img_cols, 1)
print(input_shape)
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
GD_W_DISC_PATH = "galaxy_data_including_disc/m1.dir/"
GD_WO_DISC_PATH = "galaxy_data_without_disc/m1.dir/"

GD_FN = "2dft.dat"
DATA_PART_NMODEL = int(nmodel/2)


def load_data(path, filename):
    return np.genfromtxt(os.path.join(path, filename), autostrip=True, max_rows=DATA_PART_NMODEL*n_mesh3)

x_dataset_wd = load_data(GD_W_DISC_PATH, GD_FN).reshape(DATA_PART_NMODEL, img_rows, img_cols, 1)
y_dataset_wd = np.ones((DATA_PART_NMODEL, 1))

x_dataset_wod = load_data(GD_WO_DISC_PATH, GD_FN).reshape(DATA_PART_NMODEL, img_rows, img_cols, 1)
y_dataset_wod = np.zeros((DATA_PART_NMODEL, 1))

x_dataset = np.append(x_dataset_wd, x_dataset_wod, axis=0)
y_dataset = np.append(y_dataset_wd, y_dataset_wod, axis=0)

y_dataset = keras.utils.np_utils.to_categorical(y_dataset, num_classes)

print(x_dataset.shape)
print(y_dataset.shape)

from sklearn import model_selection
x_train, x_test, y_train, y_test = model_selection.train_test_split(x_dataset, y_dataset, test_size=0.2)

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
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
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
