#Main RL agent logic

#agent.py

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import TensorBoard
import os
from config import CONFIG
from dqn_model import build_dqn_model

class DQNAgent:

  def __init__(self):
    self.policy_net= build_dqn_model()
    self.target_net= build_dqn_model()
    self.optimizer = optimizers.Adam(learning_rate= CONFIG["learning_rate"])
    self.update_target_network()
    self.memory= ReplayMemory(CONFIG["memory_capacity"])
    self.epsilon= CONFIG["epsilon_start"]
    self.steps_done=0
    self.summary_writer= tf.summary.create_file_writer(CONFIG["log_dir"])

  
  def update_target_network(self):
    self.target_net.set_weights(self.policy_net.get_weights())

  
  def select_action(self,state):
    self.steps_done +=1
    
    if random.random()< self.epsilon:
      return random.randint(0,CONFIG["num_actions"]-1)
    else:
      q_value= self.policy_net.predict(state[np.newaxis],verbose=0)
      return np.argmax(q_values[0])


  def store_transition(self,state, action,reward,next_state,done):
    self.memory.push(state,action,reward,next_state,done)


  def optimize(self):
    if len(self.memory) < CONFIG["batch_size"]:
      return

    transitions= self.memory.sample(CONFIG["batch_size"])
    batch= list(zip(*transitions))
    states= np.array(batch[0])
    actions= np.array(batch[1])
    rewards= np.array(batch[2])
    next_states= np.array(batch[3])
    dones= np.array(batch[4])

    q_next= self.target_net.predict(next_states, verbose=0)
    max_q_next= np.max(q_next, axis=1)
    targets= rewards+(1-dones)* CONFIG["gamma"]*max_q_next

    with tf.GradientTape() as tape:
      predictions= self.policy_net(states)
      loss= tf.reduce_mean(tf.square(predictions- targets))

    
    grads= tape.gradient(loss, self.policy_net.trainable_variables)
    self.optimizer.apply_gradients(zip(grads, self.policy_net.trainable_variables))


    #Decay Epsilon

    self.epsilon = max(
        CONFIG["epsilon_end"],
        CONFIG["epsilon_start"]-self.steps_done/CONFIG["epsilon_decay_steps"]
    )

    with self.summary_writer.as_default():
      tf.summary.scaler("loss", loss.numpy(), step= self.steps_done)
      tf.summary.scaler("epsilon",self.epsilon,step =self.steps_done)

  
  def save_model(self,path):
    self.policy_net.save_weights(path)

  def load_model(self,path):
    self.policy_net.load_weights(path)
    self.update_target_network()
