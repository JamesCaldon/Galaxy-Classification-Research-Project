from math import ceil
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.metrics import SparseCategoricalAccuracy
import random
import numpy as np

############################################
##########HELPER FUNCTIONS##################


def check_input_shape_validity(input_shape):

    try:
        if len(input_shape) != 3:
            raise Exception(
                "input_shape has to be a tuple (typically of 3 dimensions) or any of the available keywords")
    except BaseException:
        raise Exception(
            "input_shape has to be a tuple (typically of 3 dimensions) or any of the available keywords")

    return input_shape


def check_imgs_validity(img_array):

    if isinstance(img_array, np.ndarray):
        if len(img_array.shape) != 4:
            raise Exception(
                "The Image Array needs to have 4 dimensions. (num,x,y,bands)")
    else:
        raise Exception(
            "The Image Array Needs to be a 4 Dimensional Numpy Array. (num,x,y,bands)")


def check_labels_validity(labels):
    pass # we will modify it to use sparse_categorical
    # if isinstance(labels, np.ndarray):
    #     if labels.shape[1] != 3:
    #         raise Exception(
    #             "The Labels Array needs to have 2 dimensions. (num,(target_1,target_2,target_3))")
    # else:
    #     raise Exception(
    #         "The Lables Array Needs to be a 2 Dimensional Numpy Array. (num,(target_1,target_2,target_3))")


def check_bools_validity(bools):

    if (bools == 'train_bools_SDSS'):
        bools = [True] * 8
    elif (bools == 'load_bools_SDSS'):
        bools = [True, True, True, True, True, False, False, False]

    try:
        for element in bools:
            if not isinstance(element, bool):
                raise Exception(
                    "The Supplied Array of Bools doesn't look okay")

        if len(bools) != 8:
            raise Exception("The Supplied Array of Bools doesn't look okay")

    except BaseException:
        raise Exception("The Supplied Array of Bools doesn't look okay")

    return bools


###########################################
###########################################


############################################
##########KERAS FUNCTIONS##################

# Implementing LRN in Keras. Code from "Deep Learning with Python" by F.
# Chollet
class LocalResponseNormalization(Layer):

    def __init__(self, n=5, alpha=0.0001, beta=0.75, k=1.0, **kwargs):
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.k = k
        super(LocalResponseNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.shape = input_shape
        super(LocalResponseNormalization, self).build(input_shape)

    def call(self, x, mask=None):
        if K.image_data_format == "channels_first":
            _, f, r, c = self.shape
        else:
            _, r, c, f = self.shape
        squared = K.square(x)
        pooled = K.pool2d(squared, (self.n, self.n), strides=(1, 1),
                          padding="same", pool_mode="avg")
        if K.image_data_format == "channels_first":
            summed = K.sum(pooled, axis=1, keepdims=True)
            averaged = self.alpha * K.repeat_elements(summed, f, axis=1)
        else:
            summed = K.sum(pooled, axis=3, keepdims=True)
            averaged = self.alpha * K.repeat_elements(summed, f, axis=3)
        denom = K.pow(self.k + averaged, self.beta)
        return x / denom

    def get_output_shape_for(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = super(LocalResponseNormalization, self).get_config()
        config.update({"n": self.n,
        "alpha": self.alpha,
        "beta": self.beta,
        "k": self.k})
        return config


# def gamornet_build_model_keras(input_shape):

#     input_shape = check_input_shape_validity(input_shape)

#     # uniform scaling initializer
#     uniform_scaling = VarianceScaling(
#         scale=1.0, mode='fan_in', distribution='uniform', seed=None)

#     # Building GaMorNet
#     model = Sequential()

#     model.add(Conv2D(96, 11, strides=4, activation='relu', input_shape=input_shape,
#                      padding='same', kernel_initializer=uniform_scaling))
#     model.add(MaxPooling2D(pool_size=3, strides=2, padding='same'))
#     model.add(LocalResponseNormalization())

#     model.add(Conv2D(256, 5, activation='relu', padding='same',
#                      kernel_initializer=uniform_scaling))
#     model.add(MaxPooling2D(pool_size=3, strides=2, padding='same'))
#     model.add(LocalResponseNormalization())

#     model.add(Conv2D(384, 3, activation='relu', padding='same',
#                      kernel_initializer=uniform_scaling))
#     model.add(Conv2D(384, 3, activation='relu', padding='same',
#                      kernel_initializer=uniform_scaling))
#     model.add(Conv2D(256, 3, activation='relu', padding='same',
#                      kernel_initializer=uniform_scaling))
#     model.add(MaxPooling2D(pool_size=3, strides=2, padding='same'))
#     model.add(LocalResponseNormalization())

#     model.add(Flatten())
#     model.add(Dense(4096, activation='tanh',
#                     kernel_initializer='TruncatedNormal'))
#     model.add(Dropout(0.5))
#     model.add(Dense(4096, activation='tanh',
#                     kernel_initializer='TruncatedNormal'))
#     model.add(Dropout(0.5))
#     model.add(Dense(2, activation='softmax',
#                     kernel_initializer='TruncatedNormal')) # Only two classes

#     return model
from tensorflow.keras import regularizers
def gamornet_build_model_keras(input_shape):

    input_shape = check_input_shape_validity(input_shape)

    # uniform scaling initializer
    uniform_scaling = VarianceScaling(
        scale=1.0, mode='fan_in', distribution='uniform', seed=None)

    # Building GaMorNet
    model = Sequential()

    model.add(Conv2D(96, 11, strides=4, activation='relu', input_shape=input_shape,
                     padding='same', kernel_initializer=uniform_scaling))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LocalResponseNormalization())

    model.add(Conv2D(256, 5, activation='relu', padding='same',
                     kernel_initializer=uniform_scaling))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LocalResponseNormalization())

    model.add(Conv2D(384, 3, activation='relu', padding='same',
                     kernel_initializer=uniform_scaling))
    model.add(BatchNormalization())
    model.add(Conv2D(384, 3, activation='relu', padding='same',
                     kernel_initializer=uniform_scaling))
    model.add(BatchNormalization())
    model.add(Conv2D(256, 3, activation='relu', padding='same',
                     kernel_initializer=uniform_scaling))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.25))
    model.add(LocalResponseNormalization())

    model.add(Flatten())
    model.add(Dense(4096, activation='tanh',
                    kernel_initializer='TruncatedNormal', kernel_regularizer=regularizers.l1(0.01), activity_regularizer=regularizers.l1(0.01)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Dense(4096, activation='tanh',
                    kernel_initializer='TruncatedNormal', kernel_regularizer=regularizers.l1(0.01), activity_regularizer=regularizers.l1(0.01)))
    model.add(Dropout(0.65))
    model.add(BatchNormalization())

    model.add(Dense(2, activation='softmax',
                    kernel_initializer='TruncatedNormal')) # Only two classes

    return model


def gamornet_train_keras(training_imgs, training_labels, validation_imgs, validation_labels, input_shape, files_save_path="./",
                         epochs=100, checkpoint_freq=0, batch_size=64, lr=0.0001, momentum=0.9, decay=0.0, nesterov=False,
                         loss='sparse_categorical_crossentropy', load_model=False, model_load_path="./", save_model=False, verbose=1,
                        callbacks = []):
    """
    Trains and returns a GaMorNet model using Keras.

    Parameters
    -----------

    training_imgs: Numpy ndarray [nsamples,x,y,ndim]
        The array of images which are to be used for the training process. We insist on numpy arrays
        as many of the underlying deep learning frameworks work better with numpy arrays compared to
        other array-like elements.

    training_labels: Numpy ndarray [nsamples,label_arrays]
        The truth labels for each of the training images. 

    validation_imgs: Numpy ndarray [nsamples,x,y,ndim]
        The array of images which are to be used for the validation process. We insist on numpy arrays
        as many of the underlying deep learning frameworks work better with numpy arrays compared to
        other array-like elements.

    validation_labels: Numpy ndarray [nsamples,label_arrays]
        The truth labels for each of the validation images. 

    input_shape: tuple of ints (x, y, ndim) or allowed str
        The shape of the images being used in the form of a tuple. 

        This parameter can also take the following special values:-

        * ``SDSS`` - Sets the input shape to be (167,167,1) as was used for the SDSS g-band images in Ghosh et. al. (2020)

    files_save_path: str
        The full path to the location where files generated during the training process are to be saved. This
        includes the ``metrics.csv`` file as well as the trained model.

        Set this to ``/dev/null`` on a unix system if you don't want to save the output.

    epochs: int
        The number of epochs for which you want to train the model.

    checkpoint_freq: int
        The frequency (in terms of epochs) at which you want to save models. For eg. setting this
        to 25, would save the model at its present state every 25 epochs.

    batch_size: int
        This variable specifies how many images will be processed in a single batch. This is a
        hyperparameter. The default value is a good starting point

    lr: float or schedule
        This is the learning rate to be used during the training process. This is a
        hyperparameter that should be tuned during the training process. The default value is a good
        starting point.

        Instead of setting it at a single value, you can also set a schedule using
        ``keras.optimizers.schedules.LearningRateSchedule``

    momentum: float
        The value of the momentum to be used in the gradient descent optimizer that is used to train GaMorNet.
        This must always be :math:`\geq 0`. This accelerates the gradient descent process. This is a
        hyperparameter. The default value is a good starting point.

    decay: float
        The amount of learning rate decay to be applied over each update.

    nesterov: bool
        Whether to apply Nesterov momentum or not.

    loss: allowed str or function
        The loss function to be used. If using the string option, you need to specify the name of
        the loss function. This can be set to be any loss available in ``keras.losses``

    load_model: bool
        Whether you want to start the training from a previously saved model.

        We strongly recommend using the ``gamornet_tl_keras`` function for more
        control over the process when starting the training from a previously
        saved model.

    model_load_path: str
        Required iff ``load_model == True``. The path to the saved model.

    save_model: bool
        Whether you want to save the model in its final trained state.

        Note that this parameter does not affect the models saved by the
        ``checkpoint_freq`` parameter

    verbose: {0, 1, 2}
        The level of verbosity you want from Keras during the training process.
        0 = silent, 1 = progress bar, 2 = one line per epoch.


    Returns
    --------

    Trained Keras Model: Keras ``Model`` class


    """

    check_imgs_validity(training_imgs)
    check_imgs_validity(validation_imgs)
    check_labels_validity(training_labels)
    check_labels_validity(validation_labels)

    model = gamornet_build_model_keras(input_shape=input_shape)

    sgd = optimizers.SGD(learning_rate=lr, momentum=momentum,
                         decay=decay, nesterov=nesterov)
    model.compile(loss=loss, optimizer=sgd, metrics=[SparseCategoricalAccuracy(), 'accuracy'])

    callbacks_list = []
    callbacks_list.extend(callbacks) # Custom Modification
    if checkpoint_freq != 0:
        checkpoint = ModelCheckpoint(files_save_path + 'model_{epoch:02d}.hdf5', monitor='val_loss',
                                     verbose=verbose, save_best_only=False, save_weights_only=False, mode='auto', period=checkpoint_freq)
        callbacks_list.append(checkpoint)

    csv_logger = CSVLogger(files_save_path + "metrics.csv",
                           separator=',', append=False)
    callbacks_list.append(csv_logger)

    if load_model is True:
        model = model.load_weights(model_load_path) # Custom modification

    train_history = model.fit(training_imgs, training_labels, batch_size=batch_size, epochs=epochs, verbose=verbose,
              validation_data=(validation_imgs, validation_labels), shuffle=True, callbacks=callbacks_list)

    if save_model is True:
        model.save(files_save_path + "trained_model.hdf5")

    return model, train_history


def gamornet_train_keras_with_datagen(training_datagen, training_imgs, training_labels, validation_datagen, validation_imgs, validation_labels, input_shape, files_save_path="./",
                         epochs=100, checkpoint_freq=0, batch_size=64, lr=0.0001, momentum=0.9, decay=0.0, nesterov=False,
                         loss='sparse_categorical_crossentropy', load_model=False, model_load_path="./", save_model=False, verbose=1,
                         callbacks = []):
    """
    Trains and returns a GaMorNet model using Keras.

    Parameters
    -----------

    training_imgs: Numpy ndarray [nsamples,x,y,ndim]
        The array of images which are to be used for the training process. We insist on numpy arrays
        as many of the underlying deep learning frameworks work better with numpy arrays compared to
        other array-like elements.

    training_labels: Numpy ndarray [nsamples,label_arrays]
        The truth labels for each of the training images. 

    validation_imgs: Numpy ndarray [nsamples,x,y,ndim]
        The array of images which are to be used for the validation process. We insist on numpy arrays
        as many of the underlying deep learning frameworks work better with numpy arrays compared to
        other array-like elements.

    validation_labels: Numpy ndarray [nsamples,label_arrays]
        The truth labels for each of the validation images. 

    input_shape: tuple of ints (x, y, ndim) or allowed str
        The shape of the images being used in the form of a tuple. 

        This parameter can also take the following special values:-

        * ``SDSS`` - Sets the input shape to be (167,167,1) as was used for the SDSS g-band images in Ghosh et. al. (2020)

    files_save_path: str
        The full path to the location where files generated during the training process are to be saved. This
        includes the ``metrics.csv`` file as well as the trained model.

        Set this to ``/dev/null`` on a unix system if you don't want to save the output.

    epochs: int
        The number of epochs for which you want to train the model.

    checkpoint_freq: int
        The frequency (in terms of epochs) at which you want to save models. For eg. setting this
        to 25, would save the model at its present state every 25 epochs.

    batch_size: int
        This variable specifies how many images will be processed in a single batch. This is a
        hyperparameter. The default value is a good starting point

    lr: float or schedule
        This is the learning rate to be used during the training process. This is a
        hyperparameter that should be tuned during the training process. The default value is a good
        starting point.

        Instead of setting it at a single value, you can also set a schedule using
        ``keras.optimizers.schedules.LearningRateSchedule``

    momentum: float
        The value of the momentum to be used in the gradient descent optimizer that is used to train GaMorNet.
        This must always be :math:`\geq 0`. This accelerates the gradient descent process. This is a
        hyperparameter. The default value is a good starting point.

    decay: float
        The amount of learning rate decay to be applied over each update.

    nesterov: bool
        Whether to apply Nesterov momentum or not.

    loss: allowed str or function
        The loss function to be used. If using the string option, you need to specify the name of
        the loss function. This can be set to be any loss available in ``keras.losses``

    load_model: bool
        Whether you want to start the training from a previously saved model.

        We strongly recommend using the ``gamornet_tl_keras`` function for more
        control over the process when starting the training from a previously
        saved model.

    model_load_path: str
        Required iff ``load_model == True``. The path to the saved model.

    save_model: bool
        Whether you want to save the model in its final trained state.

        Note that this parameter does not affect the models saved by the
        ``checkpoint_freq`` parameter

    verbose: {0, 1, 2}
        The level of verbosity you want from Keras during the training process.
        0 = silent, 1 = progress bar, 2 = one line per epoch.


    Returns
    --------

    Trained Keras Model: Keras ``Model`` class


    """

    check_imgs_validity(training_imgs)
    check_imgs_validity(validation_imgs)
    check_labels_validity(training_labels)
    check_labels_validity(validation_labels)

    model = gamornet_build_model_keras(input_shape=input_shape)

    sgd = optimizers.SGD(learning_rate=lr, momentum=momentum,
                         decay=decay, nesterov=nesterov)
    #adam = optimizers.Adam(learning_rate=lr)

    model.compile(loss=loss, optimizer=sgd, metrics=[SparseCategoricalAccuracy(), 'accuracy'])

    callbacks_list = []
    callbacks_list.extend(callbacks) # Custom Modification

    if checkpoint_freq != 0:
        checkpoint = ModelCheckpoint(files_save_path + 'model_{epoch:02d}.hdf5', monitor='val_loss',
                                     verbose=verbose, save_best_only=False, save_weights_only=False, mode='auto', period=checkpoint_freq)
        callbacks_list.append(checkpoint)

    csv_logger = CSVLogger(files_save_path + "metrics.csv",
                           separator=',', append=False)
    callbacks_list.append(csv_logger)

    if load_model is True:
        model = model.load_weights(model_load_path) # Custom modification

    #train_history = model.fit(training_imgs, training_labels, batch_size=batch_size, epochs=epochs, verbose=verbose,
    #        validation_data=(validation_imgs, validation_labels), shuffle=True, callbacks=callbacks_list)
    train_history = model.fit(training_datagen.flow(training_imgs, training_labels, batch_size=batch_size), batch_size=None, epochs=epochs, verbose=verbose,
            validation_data=(validation_imgs, validation_labels), shuffle=True, callbacks=callbacks_list)


    if save_model is True:
        model.save(files_save_path + "trained_model.hdf5")

    return model, train_history


def gamornet_tl_keras(train_datagen, train_imgs, train_labels, val_datagen, val_imgs, val_labels, input_shape, load_layers_bools=[True] * 8,
                      trainable_bools=[True] * 8, model_load_path="./", files_save_path="./", epochs=100, checkpoint_freq=0, batch_size=64,
                      lr=0.0001, momentum=0.9, decay=0.0, nesterov=False, loss='sparse_categorical_crossentropy', save_model=False, verbose=1,
                      callbacks = []):
    """
    Performs Transfer Learning (TL) using a previously trained GaMorNet model.

    Parameters
    -----------

    img_datagen: fitted instance of tensorflow.keras.preprocessing.image.ImageDataGenerator
        ImageDataGenerator should contain training and validation subsets

    input_shape: tuple of ints (x, y, ndim) or allowed str
        The shape of the images being used in the form of a tuple. 

        This parameter can also take the following special values:-

        * ``SDSS`` - Sets the input shape to be (167,167,1) as was used for the SDSS g-band images in Ghosh et. al. (2020)

    load_layers_bools: array of bools
        This variable is used to identify which of the 5 convolutional and 3 fully-connected layers of GaMorNet will be
        loaded during the transfer learning process from the supplied starting model. The rest of the layers will be
        initialized from scratch.

        The order of the bools correspond to the following layer numbers [2, 5, 8, 9, 10, 13, 15, 17] in GaMorNet. Please see
        Figure 4 and Table 2 of Ghosh et. al. (2020) to get more details. The first five layers are the convolutional
        layers and the last three are the fully connected layers.

        This parameter can also take the following special values which are handy when you are using our models to
        perform predictions:-

        * ``load_bools_SDSS`` - Sets the bools according to what was done for the SDSS data in Ghosh et. al. (2020)

    trainable_bools: array of bools
        This variable is used to identify which of the 5 convolutional and 3 fully-connected layers of GaMorNet will be
        trainable during the transfer learning process. The rest are frozen at the values loaded from the previous
        model.

        The order of the bools correspond to the following layer numbers [2, 5, 8, 9, 10, 13, 15, 17] in GaMorNet. Please see
        Figure 4 and Table 2 of Ghosh et. al. (2020) to get more details. The first five layers are the convolutional
        layers and the last three are the fully connected layers.

        This parameter can also take the following special values which are handy when you are using our models to
        perform predictions:-

        * ``train_bools_SDSS`` - Sets the bools according to what was done for the SDSS data in Ghosh et. al. (2020)

    model_load_path: str
        Full path to the saved Keras model, which will serve as the starting point for transfer learning.

        Additionally, this parameter can take the following special values

        * ``SDSS_sim`` -- Downloads and uses the GaMorNet model trained on SDSS g-band simulations at z~0 from Ghosh et. al. (2020)
        * ``SDSS_tl`` -- Downloads and uses the GaMorNet model trained on SDSS g-band simulations and real data at z~0 from Ghosh et. al. (2020)

    files_save_path: str
        The full path to the location where files generated during the training process are to be saved. This
        includes the ``metrics.csv`` file as well as the trained model.

        Set this to ``/dev/null`` on a unix system if you don't want to save the output.

    epochs: int
        The number of epochs for which you want to train the model.

    checkpoint_freq: int
        The frequency (in terms of epochs) at which you want to save models. For eg. setting this
        to 25, would save the model at its present state every 25 epochs.

    batch_size: int
        This variable specifies how many images will be processed in a single batch. This is a
        hyperparameter. The default value is a good starting point

    lr: float or schedule
        This is the learning rate to be used during the training process. This is a
        hyperparameter that should be tuned during the training process. The default value is a good
        starting point.

        Instead of setting it at a single value, you can also set a schedule using
        ``keras.optimizers.schedules.LearningRateSchedule``

    momentum: float
        The value of the momentum to be used in the gradient descent optimizer that is used to train GaMorNet.
        This must always be :math:`\geq 0`. This accelerates the gradient descent process. This is a
        hyperparameter. The default value is a good starting point.

    decay: float
        The amount of learning rate decay to be applied over each update.

    nesterov: bool
        Whether to apply Nesterov momentum or not.

    loss: allowed str
        The loss function to be used. If using the string option, you need to specify the name of
        the loss function. This can be set to be any loss available in ``keras.losses``

    save_model: bool
        Whether you want to save the model in its final trained state.

        Note that this parameter does not affect the models saved by the
        ``checkpoint_freq`` parameter

    verbose: {0, 1, 2}
        The level of verbosity you want from Keras during the training process.
        0 = silent, 1 = progress bar, 2 = one line per epoch.


    Returns
    --------

    Trained Keras Model: Keras ``Model`` class


    """

    print(load_layers_bools)
    load_layers_bools = check_bools_validity(load_layers_bools)
    trainable_bools = check_bools_validity(trainable_bools)

    model = gamornet_build_model_keras(input_shape=input_shape)
    model_new = clone_model(model)
    model.load_weights(model_load_path) # Custom modification


    # Reversing the Order of the Bools because I will call .pop() on these
    # later
    load_layers_bools.reverse()
    trainable_bools.reverse()

    for i in range(len(model_new.layers)):

        if model_new.layers[i].count_params() != 0 and not isinstance(model_new.layers[i], BatchNormalization):
            model_new.layers[i].trainable = trainable_bools.pop()
            if load_layers_bools.pop() is True:

                model_new.layers[i].set_weights(model.layers[i].get_weights())
                print("Loading Layer" + str(i) + " from previous model.")
            else:
                print("Initializing Layer" + str(i) + " from scratch")

        else:
            model_new.layers[i].set_weights(model.layers[i].get_weights())

    sgd = optimizers.SGD(learning_rate=lr, momentum=momentum,
                         decay=decay, nesterov=nesterov)
    model_new.compile(loss=loss, optimizer=sgd, metrics=[SparseCategoricalAccuracy(), 'accuracy'])

    callbacks_list = []
    callbacks_list.extend(callbacks) # Custom Modification

    if checkpoint_freq != 0:
        checkpoint = ModelCheckpoint(files_save_path + 'model_{epoch:02d}.hdf5', monitor='val_loss',
                                     verbose=verbose, save_best_only=False, save_weights_only=False, mode='auto', period=checkpoint_freq)
        callbacks_list.append(checkpoint)

    csv_logger = CSVLogger(files_save_path + "metrics.csv",
                           separator=',', append=False)
    callbacks_list.append(csv_logger)
    transfer_history = model_new.fit(train_datagen.flow(train_imgs, train_labels, batch_size=batch_size), batch_size=None, epochs=epochs, verbose=verbose,
                  validation_data=(val_imgs, val_labels), shuffle=True, callbacks=callbacks_list)
                  #validation_data=val_datagen.flow(val_imgs, val_labels, batch_size=batch_size), shuffle=True, callbacks=callbacks_list)

    if save_model is True:
        model_new.save(files_save_path + "trained_model.hdf5")

    return model_new, transfer_history


###########################################
###########################################
