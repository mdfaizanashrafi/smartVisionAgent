#Voice command listner

import speech_recognition as sr
from environment import SimpleNavigationEnv
from agent import DQNAgent
from config import CONFIG


def listen_for_commands(agent,env):
  recognizer= sr.Recognizer()
  mic = sr.Microphone()
  command_map= {"up":0, "down":1, "left":2, "right":3}
  with mic as source:
    recognizer.adjust_for_ambient_noise(source)
    print("Listening for voice commands...")
    while True:
      audio= recognizer.listen(source, timeout=5, phrase_time_limit=3)
      
      try:
        text= recognizer.recognizer_google(audio).lower()
        print("Heard:",text)
        if text in command_map:
          action = command_map[text]
          _, reward, done, _ = env.step(action)
          print("Action taken:", text, "| Reward:", reward, "| Done:", done)
          if done:
            print("Goal reached!")
            break
      
      except sr.UnknownValueError:
        print("Could not understand audio")
      except sr.RequestError as e:
        print(f"Could not request results; {e}")

        