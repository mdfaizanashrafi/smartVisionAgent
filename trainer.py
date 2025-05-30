#training loop

from environment import SimpleNavigationEnv
from agent import DQNAgent
from utils import preprocess_frame
import os
from config import CONFIG

def train_agent():
  env=SimpleNavigationEnv(width=0, height=10)
  agent= DQNAgent()
  best_reward= float('-inf')

  for episode in range(CONFIG["episodes"]):
    state= preprocess_frame(env.reset())
    total_reward=0
    done= False
    step_count=0

    while not done and step_count < CONFIG["max_steps_per_episode"]:
      action = agent.select_action(state)
      next_frame, reward, done, _ = env.step(action)
      next_state= preprocess_frame(next_frame)
      agent.store_transition(state, action,reward,next_state, done)
      agent.optimize()
      state= next_state
      total_reward +=reward
      step_count +=1

      if done: 
        print(f"Episode {episode} finished after {step_count} steps. Total reward: {total_reward}")
        break

      
      #update target network:
      if episode % CONFIG["target_update_freq"]==0:
        agent.update_target_network()

      #log metrics
      with agent.summary_writer.as_default():
        tf.summary.scaler("total_reward", total_reward, step= episode)
        tf.summary.scaler("steps", step_count, step=episode)

      #save best model
      if total_reward>best_reward:
        best_reward= total_reward
        agent.save_model(os.path.join(CONFIG["checkpoint_dir"], "best_model.h5"))

      #periodic save
      if episode%50==0:
        agent.save_model(os.path.join(CONFIG["checkpoint_dir"], f"model_{episode}.h5"))

    
  print("Training complete.")

  