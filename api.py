import pickle
from flask import Flask, request, jsonify, send_file

app = Flask(__name__)

with open('./recommendation_model.pickle', 'rb') as model_file:
    app.model = pickle.load(model_file)

def recommend_songs(input_songs):
    model_data = app.model
    recommended_songs = set()

    track_rules = model_data["track_rules"]

    for rule in track_rules:
        if set(rule[0]).issubset(input_songs):
            recommended_songs.update(rule[1])
    
    recommended_songs = recommended_songs - set(input_songs)

    if not recommended_songs:
        recommended_songs = [song for song in model_data["track_popularity"].index if song not in input_songs]
  
    return list(recommended_songs)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    input_songs = data.get('songs', [])
    
    if not input_songs:
        return jsonify({"error": "No songs provided"}), 400

    recommended_songs = recommend_songs(input_songs)

    file_path = './recommendations.txt'
    with open(file_path, 'w') as file:
        for song in recommended_songs:
            file.write(f"{song}\n")

    send_file(file_path, as_attachment=True)

    response = {
        "songs": recommended_songs,
        "version": app.model["metadata"]["version"],
        "model_date": app.model["metadata"]["last_update"]
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
