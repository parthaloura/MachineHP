from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load trained model (for now, we retrain it here, but ideally, you save & load it)
def train_model():
    np.random.seed(42)
    num_samples = 1000
    data = {
        "Vibration": np.random.normal(50, 10, num_samples),
        "Temperature": np.random.normal(75, 10, num_samples),
        "Humidity": np.random.uniform(30, 80, num_samples),
        "Pressure": np.random.normal(101325, 500, num_samples),
        "Acoustic": np.random.normal(60, 15, num_samples),
    }
    def determine_health(vibration, temp, humidity, pressure, acoustic):
        if vibration > 70 or temp > 90 or acoustic > 85:
            return 2  # Failure
        elif vibration > 60 or temp > 85 or acoustic > 75:
            return 1  # Warning
        else:
            return 0  # Normal
    data["Health_Status"] = [
        determine_health(data["Vibration"][i], data["Temperature"][i],
                         data["Humidity"][i], data["Pressure"][i],
                         data["Acoustic"][i])
        for i in range(num_samples)
    ]
    df = pd.DataFrame(data)
    X = df.drop(columns=["Health_Status"])
    y = df["Health_Status"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Train and save the model
model = train_model()

# Initialize Flask App
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        features = [[data["Vibration"], data["Temperature"], data["Humidity"], data["Pressure"], data["Acoustic"]]]
        prediction = model.predict(features)[0]
        health_status = {0: "Normal", 1: "Warning", 2: "Failure"}[prediction]
        return jsonify({"prediction": health_status})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
