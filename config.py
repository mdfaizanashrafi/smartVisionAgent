# config.py

CONFIG = {
    "input_shape": (84, 84, 1),  #input image are resized to 84x84 and converted to grayscale(1 channel)
    "num_actions": 4,  #num of actions the agent can take, up, down, right left
    "batch_size": 32,
    "gamma": 0.99,
    "epsilon_start": 1.0, #initial value of e for epsilon greedy strategy
    "epsilon_end": 0.1,  
    "epsilon_decay_steps": 10000,
    "memory_capacity": 100000,
    "target_update_freq": 1000,
    "learning_rate": 0.00025,
    "episodes": 1000, #total nmbr of episodes to train the  agent
    "max_steps_per_episode": 1000,  #prevents infinite loop and limits traingin duration per episode
    "log_dir": "./logs",
    "checkpoint_dir": "./checkpoints",
    "demo_video_path": "./demos/demo.avi",
    "tflite_model_path": "./models/agent.tflite"
}