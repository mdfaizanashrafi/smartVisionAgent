#Live demo runner

import cv2
import numpy as np
from environment import SimpleNavigationEnv
from agent import DQNAgent
from utils import preprocess_frame
import os


def run_demo():
  env=SimpleNavigationEnv(width=10, height=10)
  agent= DQNAgent()
  try:
    agent.load_model(os.path.join(CONFIG["checkpoint_dir"], "best_model.h5"))
  except Exception as e:
    print("Failed to load model:",e)
    
  state= preprocess_frame(env.reset())
  done=False
  total_reward=0
  step=0

  fourcc= cv2.VideoWriter_fourcc(*'XVID')
  out= cv2.VideoWriter(CONFIG["demo_video_path"],fourcc,10.0,(env.cell_size*env.width, env.cell_size*env.height))

  while not done:
    frame= env._get_observation()
    out.write(frame)
    action= agent.select_action(state)
    next_frame, reward,done, _ = env.step(action)
    next_state= preprocess_frame(next_frame)
    state=next_state
    total_reward +=reward
    step +=1
    cv2.imshow('Demo', frame)

    if cv2.waitKey(100) & 0xFF == ord('q'):
      break

  print(f"Demo finished. Steps: {step}, Total Reward: {total_reward}")
  out.release()
  cv2.destroyAllWindows()

  