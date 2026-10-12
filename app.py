from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("model/heart_disease_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/knowmore")
def knowmore():
    return render_template("knowmore.html")

@app.route("/heartdiseaseinfo")
def heartdiseaseinfo():
    return render_template("he_dis_info.html")

@app.route("/protect")
def protect():
    return render_template("protect_heart.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Handle Blood Pressure type and value
        bp_type = request.form["BloodPressureType"]
        bp_value = int(request.form["BloodPressureValue"])

        if bp_type == "Systolic":
            blood_pressure = bp_value
        elif bp_type == "Diastolic":
            blood_pressure = bp_value
        else:
            blood_pressure = 0  # fallback if something goes wrong

        # Collect data from form
        user_data = {
            "Age": int(request.form["Age"]),
            "Gender": request.form["Gender"],
            "BMI": float(request.form["BMI"]),
            "Smoking": request.form["Smoking"],
            "Alcohol": request.form["Alcohol"],
            "PhysicalActivity": request.form["PhysicalActivity"],
            "DietQuality": request.form["DietQuality"],
            "SleepHours": int(request.form["SleepHours"]),
            "StressLevel": request.form["StressLevel"],
            "BloodPressure": blood_pressure,
            "RestingHeartRate": int(request.form["RestingHeartRate"]),
            "ChestPain": request.form["ChestPain"],
            "ShortnessOfBreath": request.form["ShortnessOfBreath"],
            "FamilyHistory": request.form["FamilyHistory"]
        }

        # Convert to DataFrame
        df = pd.DataFrame([user_data])

        # Encode same as during model training
        df_encoded = pd.get_dummies(df)
        model_columns = model.feature_names_in_

        # Add missing columns
        for col in model_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        df_encoded = df_encoded[model_columns]

        # Make prediction
        prediction = model.predict(df_encoded)[0]
        result = "⚠️ High Risk of Heart Disease" if prediction == 1 else "✅ Low Risk of Heart Disease"

        # Generate personalized health tips
        tips = []

        if prediction == 1:
            if user_data["Smoking"] == "Yes":
                tips.append("🚭 Try to quit smoking gradually — it greatly reduces heart strain.")
            if user_data["Alcohol"] == "Daily":
                tips.append("🍷 Reduce alcohol to occasional levels — it helps lower blood pressure.")
            if user_data["PhysicalActivity"] == "Low":
                tips.append("🏃‍♂️ Start light workouts or brisk walking 30 mins/day.")
            if user_data["DietQuality"] == "Poor":
                tips.append("🥦 Improve your diet — add fruits, green veggies, and avoid fried food.")
            if user_data["StressLevel"] == "High":
                tips.append("🧘 Try meditation or yoga to manage stress.")
            if user_data["SleepHours"] < 6:
                tips.append("💤 Sleep at least 7–8 hours daily to support heart recovery.")
            if user_data["BMI"] > 30:
                tips.append("⚖️ Maintain a healthy weight through balanced diet and daily walks.")
            if not tips:
                tips.append("🩺 Consult a doctor for further evaluation and maintain a healthy routine.")
        else:
            tips.append("💚 You seem to have a healthy lifestyle! Keep it up.")
            tips.append("🏃 Continue regular physical activity and balanced diet.")
            tips.append("🩵 Go for routine health checkups once a year to stay safe.")

        suggestion_text = "<br>".join(tips)

        return render_template("index.html", prediction_text=result, suggestion_text=suggestion_text)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}", suggestion_text="")

if __name__ == "__main__":
    app.run(debug=True)
