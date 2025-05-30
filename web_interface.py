#Flask based web UI

from flask import Flask, render_template_string, jsonify, request
import threading
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from trainer import train_agent
from demo import run_demo
from config import CONFIG

app= Flask(__name__)

@app.route('/')

def index():
  return render_template_string(HTML_TEMPLATE)

@app.route('/api/status')

def status():
  return jsonify({"status": "running", "agent":"SmartVision Agent v1.0"})

@app.route('/api/start_training', methods=['POST'])

def start_training():
  thread= threading.Thread(target=train_agent)
  thread.start()
  return jsonify({"status":"started"})

@app.route('/api/run_demo', methods=['POST'])

def start_demo():
  thread= threading.Thread(target=run_demo)
  thread.start()
  return jsonify({"status":"demo started"})

@app.route('/api/get_plot')
def get_plot():
  plt.figure(figsize=(10,4))
  plt.plot([random.random() for _ in range(50)])
  plt.title("Sample Reward Plot")
  buf= BytesIO()
  plt.savefig(buf, format='png')
  plt.close()
  data=base64.b64encode(buf.getvalue()).decode("utf-8")
  return f"data:image/png;base64, {data}"


#HTML

HTML_TEMPLATE= """
<!DOCTYPE html>
<html>
<head><title>SmartVision Agent</title></head>
<body>
<h1>SmartVision Agent</h1>
<p>This is an autonomous visual navigation system trained using Deep Q-Learning.</p>
<button onclick = "startTraining()"> Start Training </button>
<button onclick = "runDemo()"> Run Demo </button>
<img src= "/api/get_plot" alt="Reward Plot">
<script>

function startTraining(){
  fetch("/api/start_training", {method: "POST"});
}

function runDemo(){
  fetch("/api/run_demo", {method:"POST"});
}
</script>
</body>
</html>
"""
