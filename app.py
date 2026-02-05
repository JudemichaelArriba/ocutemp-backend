# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd


# Note: This python is not fully developed, some timestamp features are not included, 
# but will be added in the future. waiting for hardware to be ready and functional to test the model with real data. By: developer: Arriba

app = Flask(__name__)
CORS(app)


model = joblib.load("model/comfort_model_ver1.pk1") 

@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON like:
    {
        "rooms": [
            {"id": "room1", "temperature": 28, "humidity": 70},
            {"id": "room2", "temperature": 22, "humidity": 60}
        ]
    }
    Returns predictions for all rooms.
    """
    data = request.json
    rooms = data.get("rooms", [])
    results = []

    for room in rooms:
        room_id = room.get("id", "unknown")
        temp = room["temperature"]
        rh = room["humidity"]

    
        input_df = pd.DataFrame([{
            "Air temperature (C)": temp,
            "Relative humidity (%)": rh,
            "humidity_ratio": rh / 100,
            "temp_humidity_interaction": temp * rh
        }])

  
        prediction = model.predict(input_df)[0]
        confidence = max(model.predict_proba(input_df)[0]) * 100

      
        if prediction == "too_hot":
            recommended = max(18, temp - 2)
        elif prediction == "too_cold":
            recommended = min(26, temp + 2)
        else:
            recommended = temp

        results.append({
            "room_id": room_id,
            "comfort": prediction,
            "confidence": round(confidence, 2),
            "recommended_ac": recommended
        })

    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True)
