#Experience replay buffer

#replay_memory.py

from collections import deque
import random

class ReplayMemory:
  def __init__(self, capacity):
    self.memory= deque(maxlen=capacity)

  def push(self, state, action, reward, next_state, done):
    self.memory.append((state,action,reward,next_state, done))

  def sample(self, batch_size):
    return random.sample(self.memory, batch_size)

  def __len__(self):
    return len(self.memory)
