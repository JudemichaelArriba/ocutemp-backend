from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

API_KEY = os.environ.get("API_KEY")


model = joblib.load("model/ml-3.pkl")


def calculate_heat_index(T, RH):
    return (-8.784695
            + 1.61139411 * T
            + 2.338549 * RH
            - 0.14611605 * T * RH
            - 0.012308094 * T * T
            - 0.016424828 * RH * RH
            + 0.002211732 * T * T * RH
            + 0.00072546 * T * RH * RH
            - 0.000003582 * T * T * RH * RH)

def calculate_discomfort_index(T, RH):
   
    RH_ratio = RH / 100
    return T - 0.55 * (1 - RH_ratio) * (T - 14.5)



@app.route("/ping", methods=["GET"])
def health_check():
    client_key = request.headers.get("x-api-key")
    if not client_key or client_key != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    return jsonify({"status": "alive"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    client_key = request.headers.get("x-api-key")

    if not client_key or client_key != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.json
    rooms = data.get("rooms", [])

    results = []

    for room in rooms:
        room_id = room.get("id", "unknown")
        temp = room["temperature"]
        rh = room["humidity"]


        interaction = temp * (rh / 100)
        heat_index = calculate_heat_index(temp, rh)
        discomfort_index = calculate_discomfort_index(temp, rh)

        input_df = pd.DataFrame([{
            "temp": temp,
            "humidity": rh,
            "temp_humidity_interaction": interaction,
            "heat_index": heat_index,
            "discomfort_index": discomfort_index
        }])


        predicted_ac = model.predict(input_df)[0]


        predicted_ac = int(round(max(17, min(26, predicted_ac))))

        if heat_index >= 38:
            condition = "VERY HOT / HIGH HUMIDITY"
        elif heat_index >= 34:
            condition = "HOT"
        elif heat_index >= 30:
            condition = "WARM"
        elif heat_index >= 26:
            condition = "SLIGHTLY WARM"
        else:
            condition = "COMFORTABLE"

        results.append({
            "room_id": room_id,
            "condition": condition,
            "heat_index": round(heat_index, 2),
            "recommended_ac": predicted_ac
        })

    return jsonify(results)



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)