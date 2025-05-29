#Central configuration and constants



CONFIG = {
    "input_shape": (84, 84, 1),
    "num_actions": 4,
    "batch_size": 32,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.1,
    "epsilon_decay_steps": 10000,
    "memory_capacity": 100000,
    "target_update_freq": 1000,
    "learning_rate": 0.00025,
    "episodes": 1000,
    "max_steps_per_episode": 1000,
    "log_dir": "./logs",
    "checkpoint_dir": "./checkpoints",
    "demo_video_path": "./demos/demo.avi",
    "tflite_model_path": "./models/agent.tflite"
}
