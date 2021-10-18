import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten, Dense, Add, ReLU, AveragePooling2D, Input, GaussianNoise
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras import regularizers
def train_model(x_train, y_train, x_val, y_val, datagen = None, lr=0.001, epochs = 100, batch_size=128, callbacks=[], metrics=['binary_accuracy']):
    loss_function = binary_crossentropy

    used_metrics = [loss_function]
    used_metrics.extend(metrics)

    model = build_model(input_shape=x_train[0].shape)
    model.compile(loss=loss_function,
                    optimizer=Adam(learning_rate=lr),
                    metrics=used_metrics)

    train_history = None
    if datagen is not None and isinstance(datagen, ImageDataGenerator):
        train_history = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), batch_size=None, epochs=epochs,
            verbose=1, validation_data=(x_val, y_val), callbacks=callbacks)
    else:
        train_history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,
            verbose=1, validation_data=(x_val, y_val), callbacks=callbacks)
        

    return model, train_history

def transfer_model(base_model, x_train, y_train, x_val, y_val, datagen = None, lr=0.00001, epochs = 1000, batch_size=128, callbacks=[], metrics=['binary_accuracy'], tuning = False):
    loss_function = binary_crossentropy
    used_metrics = [loss_function]
    used_metrics.extend(metrics)

    transfer_model = clone_model(base_model)
    
    
    train_layer = tuning
    for base_layer, transfer_layer in zip(base_model.layers, transfer_model.layers):
        if isinstance(base_layer, Flatten): # Pivot point in this architecture to start training layers
            train_layer = True
        if (transfer_layer.count_params() != 0):
            transfer_layer.trainable = train_layer
            if not train_layer:
                transfer_layer.set_weights(base_layer.get_weights())


    #transfer_model = insert_intermediate_layer_in_keras(transfer_model, 1, GaussianNoise(0.01))

    transfer_model.compile(loss=loss_function,
                optimizer=Adam(learning_rate=lr),
                metrics=used_metrics)

    train_history = None
    if datagen is not None and isinstance(datagen, ImageDataGenerator):
        train_history = transfer_model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), batch_size=None, epochs=epochs,
            verbose=1, validation_data=(x_val, y_val), callbacks=callbacks)
    else:
        train_history = transfer_model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,
            verbose=1, validation_data=(x_val, y_val), callbacks=callbacks)
        

    return transfer_model, train_history


def build_model(input_shape=(100, 100, 1)):
    model = Sequential([
        Conv2D(32, kernel_size=(7, 7),
                        activation='elu',
                        input_shape=input_shape,
                        kernel_initializer = GlorotUniform(seed=2021), kernel_regularizer=regularizers.l1(0.001), activity_regularizer=regularizers.l2(0.001)),
        MaxPooling2D(pool_size=(2, 2)),
        #Dropout(0.5),
        BatchNormalization(),
        Conv2D(16, (3, 3), activation='elu', kernel_initializer = GlorotUniform(seed=2021), kernel_regularizer=regularizers.l1(0.001), activity_regularizer=regularizers.l2(0.001)),
        MaxPooling2D(pool_size=(2, 2)),
        #Dropout(0.5),
        BatchNormalization(),
        Flatten(),
        Dense(64, activation='elu', kernel_initializer = GlorotUniform(seed=2021), bias_initializer = GlorotUniform(seed=2021), kernel_regularizer=regularizers.l1(0.001), activity_regularizer=regularizers.l2(0.001)),
        #Dropout(0.5),
        BatchNormalization(),
        Dense(1, activation='sigmoid') # Maybe just raw outputs?
    ])
    return model

def insert_intermediate_layer_in_keras(model, layer_id, new_layer):
    layers = [l for l in model.layers]

    x = layers[0].output
    for i in range(1, len(layers)):
        if i == layer_id:
            x = new_layer(x)
        x = layers[i](x)

    new_model = Model(input=layers[0].input, output=x)
    return new_model

# def build_model(input_shape=(100, 100, 1)):
#     model = Sequential([
#         Conv2D(32, kernel_size=(7, 7),
#                         activation='elu',
#                         input_shape=input_shape,
#                         kernel_initializer = GlorotUniform(seed=2021)),
#         MaxPooling2D(pool_size=(4, 4)),
#         Dropout(0.5),
#         BatchNormalization(),
#         Conv2D(16, (3, 3), activation='elu', kernel_initializer = GlorotUniform(seed=2021)),
#         MaxPooling2D(pool_size=(2, 2)),
#         Dropout(0.75),
#         BatchNormalization(),
#         Flatten(),
#         Dense(128, activation='elu', kernel_initializer = GlorotUniform(seed=2021), bias_initializer = GlorotUniform(seed=2021)),
#         Dropout(0.5),
#         BatchNormalization(),
#         Dense(1, activation='sigmoid') # Maybe just raw outputs?
#     ])
#     return model

# def build_model(input_shape=(100, 100, 1)):
#     model = Sequential([
#         Conv2D(1, kernel_size = (5, 5),
#                         activation = 'elu',
#                         input_shape = input_shape,
#                         kernel_initializer = GlorotUniform(seed=2021)),
#         MaxPooling2D(pool_size = (8, 8)),
#         #Dropout(0.5),
#         # BatchNormalization(),
#         # Conv2D(1, kernel_size = (3, 3),
#         #                 activation = 'elu',
#         #                 kernel_initializer = GlorotUniform(seed=2021)),
#         # MaxPooling2D(pool_size = (4, 4)),
#         Dropout(0.5),
#         BatchNormalization(),
#         Flatten(),
#         Dense(64, activation = 'elu', kernel_initializer = GlorotUniform(seed=2021)),
#         Dropout(0.5),
#         BatchNormalization(),
#         Dense(1, activation ='sigmoid') # Maybe just raw outputs?
#     ])
#     return model

def relu_bn(inputs):
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn

def res_block(x, downsample, filters, kernel_size = 3):
    y = Conv2D(kernel_size=kernel_size,
               strides= (1 if not downsample else 2),
               filters=filters,
               padding="same")(x)
    y = relu_bn(y)
    y = Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)

    if downsample:
        x = Conv2D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same")(x)
    out = Add()([x, y])
    out = relu_bn(out)
    return out

def build_model_res(input_shape=(100, 100, 1)):
    inputs = Input(shape=input_shape)
    num_filters = 64
    
    t = BatchNormalization()(inputs)
    t = Conv2D(kernel_size=3,
               strides=1,
               filters=num_filters,
               padding="same")(t)
    t = relu_bn(t)
    
    num_blocks_list = [2, 5, 5, 2]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = res_block(t, downsample=(j==0 and i!=0), filters=num_filters)
        num_filters *= 2
    
    t = AveragePooling2D(4)(t)
    t = Flatten()(t)
    outputs = Dense(2, activation='sigmoid')(t)
    
    model = Model(inputs, outputs)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

from tensorflow.keras.layers import UpSampling2D

def autoencode():

    input_img = Input(shape=(100, 100, 1))

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder