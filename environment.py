#Navigation environment

import numpy as np
import cv2
import random

class SimpleNavigationEnv:

  def __init__(self,width=10,height=10,cell_size=64):
    self.width= width
    self.height=height
    self.cell_size= cell_size
    self.reset()
  
  def reset(self):
    self.agent_pos=[random.randint(0,self.width-1), random.randint(0,self.height-1)]
    self.goal_pos=[random.randint(0,self.width-1), random.randint(0,self.height-1)]
    while self.agent_pos == self.goal_pos:
      self.goal_pos=[random.randint(0,self.width-1), random.randint(0,self.height-1)]
    return self._get_observation()

  def _get_observation(self):
    obs= np.zeros((self.height, self.width), dtype=np.uint8)
    obs[self.agent_pos[1], self.agent_pos[0]]=255
    obs[self.goal_pos[1], self.goal_pos[0]]=128
    obs=cv2.resize(obs,(self.cell_size*self.width, self.cell_size*self.height), interpolation= cv2.INTER_NEAREST)
    obs=cv2.cvtColor(obs, cv2.COLOR_GRAY2RGB)

    return obs

  def step(self, action):
    dx, dy=0,0
    if action==0: #up
      dy=-1
    elif action ==1: #down
      dy=1
    elif action ==2: #left
      dx=-1
    elif action ==3: #right
      dx=1


    new_x=max(0, min(self.width-1, self.agent_pos[0]+dx))
    new_y= max(0,min(self.height-1, self.agent_pos[1]+dy))
    self.agent_pos=[new_x,new_y]

    done=False
    reward= -0.1

    if self.agent_pos==self.goal_pos:
      reward= 10.0
      done=True

    return self._get_observation(), reward, done, {}