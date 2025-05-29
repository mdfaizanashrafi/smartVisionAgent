#DQN model architecture

#DQN model.py

#archtitecture of the DeepQN model using tensorflow/keras.

import tensorflow as tf
from tensorflow.keras import layers, models

def build_dqn_model(input_shape=(84,84,1), num_actions=4):
  inputs= layers.Input(shape= input_shape)
  x= layers.Conv2D(32,8,strides=4, activation='relu')(inputs)
  x= layers.Conv2D(64,4,strides=2,activation='relu')(x) #learns more complex features from previous lauer output
  x=layers.Conv2D(64,3,strides=1,activation='relu')(x) #extract finer details without reducing resolution
  x= layers.Flatten()(x)  #converts the 3D output from the Convolutional layers into a 1D vector
  x= layers.Dense(512, activation='relu')(x)  # fully connected layer with 512 neurons
  outputs= layers.Dense(num_actions, activation='linear')(x)
  model= models.Model(inputs= inputs, outputs = outputs)  #connects the layers together into a full model
  return model
