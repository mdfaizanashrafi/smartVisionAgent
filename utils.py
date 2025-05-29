#Utilities 

#Utils.py

import numpy as np
import cv2


#takes RGB image frame -> convert to grayscale -> resize it to NxN ->
#normalize the color range of (0,255) to (0,1) ->
# returns a single channel image ready for the DQN model

def preprocess_frame(frame):
  gray= cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  #convert the RGB to grayscale for simplification and reduce computational load
  resized= cv2.resize(gray, (84,84)) #resize the image
  normalized = resized/255.0  #normalize the pixel value
  return np.expand_dims(normalized, axis=-1)


#in DQN, we often stack multiple consecutive frames together to give the agent info
#about motion and velocity, since single frame doesnt show movement

def stack_frames(stacked_frames, frame, is_new_episode):
  if is_new_episode:
    stacked_frames = np.stack([frame]*1, axis=-1)
  else:
    stacked_frames= np.concatenate([stacked_frames[:,:,1:], frame], axis=-1)
  return stacked_frames
