#GAN for synthetic environment


import tensorflow as tf
from tensorflow.keras import layers, Model

def build_gan_generators(latent_dim=100):
  inputs = tf.keras.Input(shape=(latent_dim,))
  x= layers.Dense(128*7*7)(inputs)
  x= layers.Reshape((7,7,128))(x)
  x= layers.Conv2DTranspose(128,(4,4), strides=(2,2),padding='same')(x)
  x=layers.BatchNormalization()(x)
  x=layers.Activation('relu')(x)
  x=layers.Conv2DTranspose(64,(4,4), strides=(2,2), padding='same')(x)
  x= layers.BatchNormalization()(x)
  x=layers.Activation('relu')(x)
  outputs= layers.Conv2D(3,(7,7), activation='tanh',padding = 'same')(x)
  return Model(inputs,outputs)
