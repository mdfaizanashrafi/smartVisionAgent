#AUtoML hyper parameter tuning

from kerastuner.tuners import RandomSearch
from dqn_model import build_tunable_dqn
from config import CONFIG

def build_tunable_dqn(hp):
  inputs= layers.Input(shape= CONFIG["input_shape"])
  x= layers.Conv2D(
      hp.Int('conv1_units',16,64,step=16),
      8,strides=4,activation='relu'
  )(inputs)
  x= layers.Conv2D(
      hp.Int('conv2_units',16,64,step=16),
      4,strides=2,activation='relu'
  )(x)
  x=layers.Conv2D(
      hp.Int('conv3_units',16,64,step=16),
      3,strides=1,activation='relu'
  )(x)
  x= layers.Flatten()(x)
  x= layers.Dense(
      hp.Int('dense_units',128,1024,step=128),activation='relu'
  )(x)
  outputs= layers.Dense(CONFIG["num_actions"], activation='linear')(x)
  model= models.Model(inputs=inputs, outputs=outputs)
  model.compile(optimizer=optimizers.Adam(hp.Choice('lr',[1e-4,2.5e-4.5e-4])), loss='mse')
  return model


def tune_hyperparameters():
  tuner= RandomSearch(
      build_tunable_dqn,
      objective='val_loss',
      max_trials=10,
      executions_per_trial=1,
      directory='./tuning',
      project_name="smartvision_dqn"
  )
  dummy_data= np.random.rand(100,84,84,1)
  dummy_labels= np.random.rand(100,CONFIG["num_actions"])
  tuner.search(dummy_data, dummy_labels, epochs=10,validation_split=0.2)
                              

