from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

API_KEY = os.environ.get("API_KEY")

# ml-4: Random Forest (monotonic) that predicts thermal sensation vote (TSV)
# from temperature + humidity. The setpoint is where it predicts neutral (TSV = 0).
model = joblib.load("model/ml-4.pkl")
model.n_jobs = 1  # one small batch per request; avoid thread start-up cost

CLAMP_MIN, CLAMP_MAX = 23.0, 26.0
SWEEP = np.round(np.arange(16.0, 32.0001, 0.05), 2)


def calculate_heat_index(T, RH):
    # Kept for the display-only "condition" label (VERY HOT / HOT / etc.).
    # No longer a model input.
    return (-8.784695
            + 1.61139411 * T
            + 2.338549 * RH
            - 0.14611605 * T * RH
            - 0.012308094 * T * T
            - 0.016424828 * RH * RH
            + 0.002211732 * T * T * RH
            + 0.00072546 * T * RH * RH
            - 0.000003582 * T * T * RH * RH)


def recommended_setpoint(rh):
    """First temperature at which ml-4 predicts a neutral vote (TSV >= 0) at this
    humidity, clamped to the ASHRAE-55 summer comfort band. The model is monotonic
    in temperature, so there is exactly one crossing."""
    tsv = model.predict(pd.DataFrame({"temp": SWEEP, "humidity": rh}))
    neutral = SWEEP[np.argmax(tsv >= 0)] if (tsv >= 0).any() else CLAMP_MAX
    return float(np.clip(neutral, CLAMP_MIN, CLAMP_MAX))


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

        # NEW: validate input instead of letting a KeyError become a 500.
        if "temperature" not in room or "humidity" not in room:
            return jsonify({"error": f"room '{room_id}' missing temperature/humidity"}), 400
        try:
            temp = float(room["temperature"])
            rh = float(room["humidity"])
        except (TypeError, ValueError):
            return jsonify({"error": f"room '{room_id}' has non-numeric temperature/humidity"}), 400

        heat_index = calculate_heat_index(temp, rh)

        # Predicted thermal sensation at the current room conditions (informational)
        tsv_input = pd.DataFrame([{"temp": temp, "humidity": rh}])
        predicted_tsv = float(model.predict(tsv_input)[0])

        # The setpoint served to the ESP32
        setpoint = recommended_setpoint(rh)
        predicted_ac = int(round(setpoint))

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
            "predicted_comfort_vote": round(predicted_tsv, 2),  # NEW, optional
            "recommended_ac": predicted_ac
        })

    return jsonify(results)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
