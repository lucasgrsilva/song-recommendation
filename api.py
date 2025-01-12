import pickle
import os
import hashlib
from flask import Flask, request, jsonify

app = Flask(__name__)
MODEL_PATH = os.getenv("MODEL_PATH")
APP_VERSION = os.getenv("APP_VERSION")
last_checksum = None
model = None

def load_model():
    global model
    app.logger.info(f"Loading model from {MODEL_PATH}")
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
  
    return list(recommended_songs)[:10]

@app.route('/api/recommender', methods=['POST'])
def recommend():
    global last_checksum

    new_checksum = get_file_checksum(MODEL_PATH)
    app.logger.info(f"M=recommend, last_checksum={last_checksum}, new_checksum={new_checksum}, model_path={MODEL_PATH}")
    if new_checksum != last_checksum:
        last_checksum = new_checksum
        app.logger.info("Updating model :)")
        load_model()

    data = request.get_json()
    input_songs = data.get('songs', [])
    
    if not input_songs:
        return jsonify({"error": "No songs provided"}), 400

    recommended_songs = recommend_songs(input_songs)

    response = {
        "songs": recommended_songs,
        "version": APP_VERSION,
        "model_date": model["last_update"]
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run()
