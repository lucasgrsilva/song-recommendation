import pickle
import os
import hashlib
import time
import datetime
import threading
from flask import Flask, request, jsonify, send_file

app = Flask(__name__)
MODEL_PATH = os.getenv("MODEL_PATH")
APP_VERSION = os.getenv("APP_VERSION")
last_checksum = None
model_last_update = None
model = None

def load_model():
    global model
    print(f"Loading model from {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

def get_file_checksum(filepath):
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()        

def recommend_songs(input_songs):
    global model
    recommended_songs = set()

    track_rules = model["rules"]

    for rule in track_rules:
        if set(rule[0]).issubset(input_songs):
            recommended_songs.update(rule[1])
    
    recommended_songs = recommended_songs - set(input_songs)

    if not recommended_songs:
        recommended_songs = [song for song in model["track_popularity"].index if song not in input_songs]
  
    return list(recommended_songs)

@app.route('/api/recommend', methods=['POST'])
def recommend():
    global model_last_update
    data = request.get_json()
    input_songs = data.get('songs', [])
    
    if not input_songs:
        return jsonify({"error": "No songs provided"}), 400

    recommended_songs = recommend_songs(input_songs)

    file_path = '/app/recommendation/recommendations.txt'
    with open(file_path, 'w') as file:
        for song in recommended_songs:
            file.write(f"{song}\n")

    send_file(file_path, as_attachment=True)

    response = {
        "songs": recommended_songs,
        "version": APP_VERSION,
        "model_date": datetime.datetime.fromtimestamp(model_last_update).strftime('%Y-%m-%d %H:%M:%S')
    }

    return jsonify(response)

def model_watcher():
    global last_checksum
    global model_last_update
    while True:
        if os.path.exists(MODEL_PATH):
            new_checksum = get_file_checksum(MODEL_PATH)
            if new_checksum != last_checksum:
                print("Updating model :)")
                last_checksum = new_checksum
                model_last_update = time.time()
                load_model()
        time.sleep(5)

if __name__ == '__main__':
    watcher_thread = threading.Thread(target=model_watcher, daemon=True)
    watcher_thread.start()
    app.run()
