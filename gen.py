from utility.initialize_tf import *
import tensorflow as tf
from tensorflow.keras import Sequential, Model, layers, activations, optimizers, Input, initializers
import numpy as np
from IPython.display import clear_output
import time
from matplotlib import pyplot as plt
import os
import sys
import tensorflow.keras.backend as K


class GAN:
    def __init__(self, generator = None, discriminator = None, batch_size = 128, epochs = 30, noise_dim = 100, num_examples_to_generate = 16, checkpoint_dir = './training_checkpoints', save_interval = 100):
        self.batch_size = batch_size
        self.epochs = epochs
        self.save_interval = save_interval
        self.generator = generator
        self.discriminator = discriminator
        if (self.generator is None):
            self.generator = self.create_generator()
        if (self.discriminator is None):
            self.discriminator = self.create_discriminator()
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5) #0.0002
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
        # This method returns a helper function to compute cross entropy loss
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.noise_dim = noise_dim
        self.gen_losses = []
        self.disc_losses = []
        self.step = tf.Variable(0)
        
        # You will reuse this seed overtime (so it's easier)
        # to visualize progress in the animated GIF)
        self.seed = tf.random.normal([num_examples_to_generate, noise_dim])

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                        discriminator_optimizer=self.discriminator_optimizer,
                                        generator=self.generator,
                                        discriminator=self.discriminator,
                                        step=self.step)
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

    def create_generator(self):
        model = Sequential()
        model.add(layers.Dense(3*3*256, use_bias=False, input_shape=(100,), activation=activations.LeakyReLU))
        #model.add(layers.BatchNormalization())
        #print(model.output_shape)
        model.add(layers.Reshape((3, 3, 256)))

        model.add(layers.Conv2DTranspose(256, (7, 7), strides = (2, 2), padding = 'same', activation=activations.LeakyReLU))
        model.add(layers.BatchNormalization())
        #print(model.output_shape)

        model.add(layers.Conv2DTranspose(128, (3, 3), strides = (2, 2), padding = 'same', activation=activations.LeakyReLU))
        model.add(layers.BatchNormalization())
        #print(model.output_shape)

        model.add(layers.Conv2DTranspose(64, (3, 3), strides = (2, 2), activation=activations.LeakyReLU))
        model.add(layers.BatchNormalization())
        #print(model.output_shape)

        model.add(layers.Conv2DTranspose(32, (3, 3), strides = (2, 2), padding = 'same', activation=activations.LeakyReLU))
        model.add(layers.BatchNormalization())
        #print(model.output_shape)

        model.add(layers.Conv2DTranspose(1, (3, 3), strides = (2, 2), padding = 'same', activation=activations.tanh))
        #model.add(layers.BatchNormalization())

        #print(model.output_shape)
        return model

    def create_discriminator(self):
        model = Sequential()

        model.add(layers.GaussianNoise(0.01))
        model.add(layers.Conv2D(1, (3, 3), strides = (2, 2), padding = 'same', input_shape=[100, 100, 1]))
        model.add(layers.BatchNormalization())
        
        model.add(layers.Conv2D(32, (3, 3), strides = (2, 2), padding = 'same', activation=layers.LeakyReLU(0.2)))
        model.add(layers.BatchNormalization())
        #print(model.output_shape)

        model.add(layers.Conv2D(64, (3, 3), strides = (2, 2), activation=layers.LeakyReLU(0.2)))
        model.add(layers.BatchNormalization())
        #print(model.output_shape)

        model.add(layers.Conv2D(128, (3, 3), strides = (2, 2), padding = 'same', activation=layers.LeakyReLU(0.2)))
        model.add(layers.BatchNormalization())
        #print(model.output_shape)

        model.add(layers.Conv2D(256, (7, 7), strides = (2, 2), padding = 'same', activation=layers.LeakyReLU(0.2)))
        model.add(layers.BatchNormalization())
        #print(model.output_shape)

        #model.add(layers.Dense(256))
        model.add(layers.Flatten())

        model.add(layers.Dense(1))
        return model

    def discriminator_loss(self, real_output, fake_output):
        fake_output = tf.where(fake_output > 0.9, 0.9, fake_output)
        fake_output = tf.where(fake_output < -0.9, -0.9, fake_output)

        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)


    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([self.batch_size, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

            tf.print("gen_loss:", output_stream=sys.stdout)

            tf.print(gen_loss, output_stream=sys.stdout)
            tf.print("disc_loss:", output_stream=sys.stdout)

            tf.print(disc_loss, output_stream=sys.stdout)
            #self.gen_losses.append(gen_loss)
            #self.disc_losses.append(disc_loss)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))


    def train(self, dataset):
        for epoch, image_batch in zip(range(self.epochs)[int(self.step):], dataset):
            start = time.time()
            clear_output(wait=True)        
            #print("Epoch")
            self.checkpoint.step.assign_add(1)
            #for i, image_batch in enumerate(dataset):
                # print(image_batch.shape)
                # E0 = np.array(image_batch[0])
                # print(self.step)
                # from matplotlib import pyplot as plt
                # plt.imshow(E0)
                # plt.colorbar()
            self.train_step(image_batch)
                # if i >= 1:
                #     break
            # Save the model every 15 epochs
            if (epoch + 1) % self.save_interval == 0:
                # Produce images for the GIF as you go
                self.generate_and_save_images(self.generator,
                                        epoch + 1,
                                        self.seed)
                self.checkpoint.save(file_prefix = self.checkpoint_prefix)


            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start), flush=True)

        # Generate after the final epoch
        #clear_output(wait=True)
        self.generate_and_save_images(self.generator,
                                self.epochs,
                                self.seed)

        # fig = plt.figure(figsize=(8, 8))
        # plt.plot(np.arange(len(self.gen_losses)), self.gen_losses)
        # plt.plot(np.arange(len(self.disc_losses)), self.disc_losses)
        # plt.savefig('gan_images\\final_losses.png'.format(epoch))
        # plt.show()

    def generate_and_save_images(self, model, epoch, test_input):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = model(test_input, training=False)

        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        plt.savefig('gan_images\\image_at_epoch_{:04d}.png'.format(epoch))
        plt.close()
        #plt.show()



class WGAN:
    def __init__(self, latent_dim = 300, img_shape= (100, 100, 1)):
        self.latent_dim = latent_dim
        self.channels = 1
        self.img_shape = img_shape
        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.clip_value = 0.01
        optimizer = optimizers.RMSprop(learning_rate=0.00005)
        #optimizer = optimizers.Adam(learning_rate=0.08, beta_1=0.5, beta_2=0.9)


        # Build and compile the critic
        self.critic = self.build_critic()
        self.critic.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.critic.trainable = False

        # The critic takes generated images as input and determines validity
        valid = self.critic(img)

        # The combined model  (stacked generator and critic)
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])


    def train(self, dataset, epochs = 100, batch_size=128, sample_interval=100):
        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))
        for epoch in range(epochs):
            for _, img_batch in zip(range(self.n_critic), dataset):
                
                # Train critic
                
                noise = np.random.normal(0, 1, [batch_size, self.latent_dim])
                
                gen_imgs = self.generator.predict(noise)

                # Train the critic
                d_loss_real = self.critic.train_on_batch(img_batch, valid)
                d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
                # Clip critic weights
                for l in self.critic.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)

            # Train generator
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)



    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)


    # def build_generator(self):

    #     model = Sequential()

    #     model.add(layers.Dense(128 * 6 * 6, activation="relu", input_dim=self.latent_dim))
    #     model.add(layers.Reshape((6, 6, 128)))
    #     model.add(layers.UpSampling2D(interpolation='bilinear'))
    #     model.add(layers.Conv2D(128, kernel_size=2, padding="same"))
    #     model.add(layers.BatchNormalization(momentum=0.8))
    #     model.add(layers.Activation("relu"))
    #     model.add(layers.UpSampling2D(interpolation='bilinear'))

    #     model.add(layers.Conv2D(64, kernel_size=2, padding="same"))
    #     model.add(layers.BatchNormalization(momentum=0.8))
    #     model.add(layers.Activation("relu"))
    #     model.add(layers.UpSampling2D(interpolation='bilinear'))

    #     model.add(layers.Conv2D(32, kernel_size=2, padding="same"))
    #     model.add(layers.BatchNormalization(momentum=0.8))
    #     model.add(layers.Activation("relu"))
    #     model.add(layers.UpSampling2D(interpolation='bilinear'))
    #     model.add(layers.Conv2D(16, kernel_size=2, padding="same"))
    #     model.add(layers.BatchNormalization(momentum=0.8))
    #     model.add(layers.Activation("relu"))
    #     model.add(layers.Conv2D(self.channels, kernel_size=4, padding="same"))
    #     model.add(layers.Activation("tanh"))

    #     model.summary()

    #     noise = Input(shape=(self.latent_dim,))
    #     img = model(noise)

    #     return Model(noise, img)

    # def build_critic(self):

    #     model = Sequential()

    #     model.add(layers.Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
    #     model.add(layers.LeakyReLU(alpha=0.2))
    #     model.add(layers.Dropout(0.25))
    #     model.add(layers.Conv2D(32, kernel_size=3, strides=2, padding="same"))
    #     model.add(layers.ZeroPadding2D(padding=((0,1),(0,1))))
    #     model.add(layers.BatchNormalization(momentum=0.8))
    #     model.add(layers.LeakyReLU(alpha=0.2))
    #     model.add(layers.Dropout(0.25))
    #     model.add(layers.Conv2D(64, kernel_size=3, strides=2, padding="same"))
    #     model.add(layers.BatchNormalization(momentum=0.8))
    #     model.add(layers.LeakyReLU(alpha=0.2))
    #     model.add(layers.Dropout(0.25))
    #     model.add(layers.Conv2D(128, kernel_size=3, strides=1, padding="same"))
    #     model.add(layers.BatchNormalization(momentum=0.8))
    #     model.add(layers.LeakyReLU(alpha=0.2))
    #     model.add(layers.Dropout(0.25))
    #     model.add(layers.Flatten())
    #     model.add(layers.Dense(1))

    #     model.summary()

    #     img = Input(shape=self.img_shape)
    #     validity = model(img)

    #     return Model(img, validity)



    def build_generator(self):
        model = Sequential()
        init = initializers.RandomNormal(stddev = 0.02)
        model.add(layers.Dense(6*6*256, use_bias=False, input_shape=(self.latent_dim,), activation=layers.LeakyReLU(0.2)))
        #model.add(layers.BatchNormalization())
        #print(model.output_shape)
        model.add(layers.Reshape((6, 6, 256)))

        model.add(layers.Conv2DTranspose(512, (4, 4), strides = (2, 2), padding = 'same', kernel_initializer=init))
        model.add(layers.DepthwiseConv2D(kernel_size=(4, 4), use_bias=False))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU(0.2))

        model.add(layers.Conv2DTranspose(512, (4, 4), strides = (2, 2), padding = 'same', kernel_initializer=init))
        model.add(layers.DepthwiseConv2D(kernel_size=(4, 4), use_bias=False))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU(0.2))

        model.add(layers.Conv2DTranspose(256, (4, 4), strides = (2, 2), padding = 'same', kernel_initializer=init))
        model.add(layers.DepthwiseConv2D(kernel_size=(4, 4), use_bias=False))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU(0.2))

        model.add(layers.Conv2DTranspose(128, (4, 4), strides = (2, 2), padding = 'same', kernel_initializer=init))
        model.add(layers.DepthwiseConv2D(kernel_size=(4, 4), use_bias=False))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU(0.2))

        model.add(layers.Conv2DTranspose(64, (4, 4), strides = (2, 2), padding = 'same', kernel_initializer=init))
        model.add(layers.DepthwiseConv2D(kernel_size=(4, 4), use_bias=False))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU(0.2))
        #print(model.output_shape)

        # model.add(layers.Conv2DTranspose(32, (3, 3), strides = (2, 2), padding = 'same', activation=layers.LeakyReLU(0.2)))
        # model.add(layers.BatchNormalization(momentum=0.8))
        #print(model.output_shape)

        #model.add(layers.Conv2DTranspose(1, (4, 4), strides = (2, 2), padding = 'same'))
        model.add(layers.Conv2D(1, kernel_size=(3, 3), use_bias=True))
        model.add(layers.Activation(activation = activations.tanh))
        #model.add(layers.BatchNormalization())
        print(model.output_shape)

        #print(model.output_shape)
        noise = Input(shape=(self.latent_dim,))
        img = model(noise)
        return Model(noise, img)

    # def build_critic(self):
    #     model = Sequential()
    #     init = initializers.RandomNormal(stddev = 0.02)
    #     #model.add(layers.GaussianNoise(0.001, input_shape=self.img_shape))
    #     model.add(layers.Conv2D(2, kernel_size=(2, 2), use_bias=True, input_shape=self.img_shape, padding="same"))
    #     model.add(layers.LeakyReLU(0.2))

    #     model.add(layers.Conv2D(32, (4, 4), strides = (2, 2), padding = 'same', use_bias=False, kernel_initializer=init))
    #     model.add(layers.BatchNormalization(momentum=0.8))
    #     model.add(layers.LeakyReLU(0.2))
    #     #model.add(layers.Dropout(0.25))

    #     model.add(layers.Conv2D(2, kernel_size=(2, 2), use_bias=True, padding = 'same', kernel_initializer=init))
    #     model.add(layers.LeakyReLU(0.2))

    #     model.add(layers.Conv2D(128, (4, 4), strides = (2, 2), padding = 'same', use_bias=False, kernel_initializer=init))
    #     model.add(layers.BatchNormalization(momentum=0.8))
    #     model.add(layers.LeakyReLU(0.2))
    #     #model.add(layers.Dropout(0.25))
    #     model.add(layers.Conv2D(2, kernel_size=(2, 2), use_bias=True, padding = 'same', kernel_initializer=init))
    #     model.add(layers.LeakyReLU(0.2))


    #     model.add(layers.Conv2D(256, (4, 4), strides = (2, 2), padding = 'same', use_bias=False, kernel_initializer=init))
    #     model.add(layers.BatchNormalization(momentum=0.8))
    #     #model.add(layers.Dropout(0.25))
    #     model.add(layers.LeakyReLU(0.2))
    #     #model.add(layers.DepthwiseConv2D(kernel_size=(2, 2), use_bias=True))
        
    #     # model.add(layers.Conv2D(64, (4, 4), strides = (2, 2), activation=layers.LeakyReLU(0.2)))
    #     # model.add(layers.BatchNormalization(momentum=0.8))
    #     # model.add(layers.Dropout(0.25))
    #     # model.add(layers.LeakyReLU(0.2))
    #     #model.add(layers.DepthwiseConv2D(kernel_size=(2, 2), use_bias=True))
    #     #print(model.output_shape)

    #     #model.add(layers.Conv2D(1, kernel_size=(4, 4), strides = (1, 1)))
    #     #model.add(layers.Dense(256))
    #     model.add(layers.Flatten())
    #     #model.add(layers.Dense(512))
    #     model.add(layers.Dense(1))

    #     img = Input(shape=self.img_shape)
    #     validity = model(img)
    #     return Model(img, validity)

    # def build_critic(self):
    #     model = Sequential()
    #     init = initializers.RandomNormal(stddev = 0.02)

    #     #model.add(layers.DepthwiseConv2D(kernel_size=2, use_bias=False, padding="same", kernel_initializer=init, input_shape=self.img_shape))
    #     #model.add(layers.DepthwiseConv2D(kernel_size=(4, 4), use_bias=False))
    #     model.add(layers.Conv2D(8, kernel_size=9, strides = 2, use_bias=True, padding="same", input_shape=self.img_shape))#, kernel_initializer=init))
    #     model.add(layers.BatchNormalization(momentum=0.8))
    #     #model.add(layers.Conv2D(2, kernel_size=2, use_bias=False, padding="same", kernel_initializer=init))
    #     model.add(layers.LeakyReLU(0.2))
    #     model.add(layers.Dropout(0.25))


    #     #model.add(layers.DepthwiseConv2D(kernel_size=2, use_bias=False, padding="same", kernel_initializer=init))
    #     #model.add(layers.DepthwiseConv2D(kernel_size=(4, 4), use_bias=False))
    #     model.add(layers.Conv2D(32, kernel_size=7, strides = 2, use_bias=True, padding="same"))#, kernel_initializer=init))
    #     model.add(layers.BatchNormalization(momentum=0.8))
    #     model.add(layers.LeakyReLU(0.2))
    #     model.add(layers.Dropout(0.25))

    #     #model.add(layers.DepthwiseConv2D(kernel_size=2, use_bias=False, padding="same", kernel_initializer=init))
    #     #model.add(layers.DepthwiseConv2D(kernel_size=(4, 4), use_bias=False))
    #     model.add(layers.Conv2D(64, kernel_size=5, strides = 2, use_bias=True, padding="same"))#, kernel_initializer=init))
    #     model.add(layers.BatchNormalization(momentum=0.8))
    #     model.add(layers.LeakyReLU(0.2))
    #     model.add(layers.Dropout(0.25))

    #     model.add(layers.Conv2D(128, kernel_size=3, strides = 1, use_bias=True, padding="same"))#, kernel_initializer=init))
    #     model.add(layers.BatchNormalization(momentum=0.8))
    #     model.add(layers.LeakyReLU(0.2))
    #     model.add(layers.Dropout(0.25))

    #     model.add(layers.Flatten())
    #     model.add(layers.Dense(1))

    #     img = Input(shape=self.img_shape)
    #     validity = model(img)
    #     return Model(img, validity)

    def build_critic(self):

        model = Sequential()

        model.add(layers.Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.25))
        model.add(layers.Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(layers.ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.25))
        model.add(layers.Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.25))
        model.add(layers.Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.25))
        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("./gans/wgan/images/images_at_%d.png" % epoch)
        plt.close()
        #self.critic.save("./gans/wgan/critic/critic_at_%d.hdf5" % epoch)
        #self.generator.save("./gans/wgan/generator/generator_at_%d.hdf5" % epoch)



# Large amount of credit goes to:
# https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
# which I've used as a reference for this implementation

from __future__ import print_function, division
import tensorflow.keras.layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import RMSprop
from functools import partial

import tensorflow.keras.backend as K

import matplotlib.pyplot as plt

import sys

import numpy as np

class RandomWeightedAverage(tensorflow.keras.layers.Layer):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((32, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class WGANGP():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        optimizer = RMSprop(lr=0.00005)

        # Build the generator and critic
        self.generator = self.build_generator()
        self.critic = self.build_critic()

        #-------------------------------
        # Construct Computational Graph
        #       for the Critic
        #-------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_img = Input(shape=self.img_shape)

        # Noise input
        z_disc = Input(shape=(self.latent_dim,))
        # Generate image based of noise (fake sample)
        fake_img = self.generator(z_disc)

        # Discriminator determines validity of the real and fake images
        fake = self.critic(fake_img)
        valid = self.critic(real_img)

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage()([real_img, fake_img])
        # Determine validity of weighted sample
        validity_interpolated = self.critic(interpolated_img)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.critic_model = Model(inputs=[real_img, z_disc],
                            outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                              self.wasserstein_loss,
                                              partial_gp_loss],
                                        optimizer=optimizer,
                                        loss_weights=[1, 1, 10])
        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(self.latent_dim,))
        # Generate images based of noise
        img = self.generator(z_gen)
        # Discriminator determines validity
        valid = self.critic(img)
        # Defines generator model
        self.generator_model = Model(z_gen, valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)


    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=4, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_critic(self):

        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size, sample_interval=50):

        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake =  np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty
        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
                # Sample generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                # Train the critic
                d_loss = self.critic_model.train_on_batch([imgs, noise],
                                                                [valid, fake, dummy])

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.generator_model.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/mnist_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    wgan = WGANGP()
    wgan.train(epochs=30000, batch_size=32, sample_interval=100)