import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# ----- Step 1: Create synthetic dataset -----
np.random.seed(42)

n_samples = 1000  # you can increase later for better accuracy

data = {
    "Age": np.random.randint(20, 80, n_samples),
    "Gender": np.random.choice(["Male", "Female"], n_samples),
    "BMI": np.random.uniform(18, 35, n_samples).round(1),
    "Smoking": np.random.choice(["Yes", "No"], n_samples),
    "Alcohol": np.random.choice(["Never", "Occasional", "Daily"], n_samples),
    "PhysicalActivity": np.random.choice(["Low", "Moderate", "High"], n_samples),
    "DietQuality": np.random.choice(["Poor", "Average", "Good"], n_samples),
    "SleepHours": np.random.randint(4, 10, n_samples),
    "StressLevel": np.random.choice(["Low", "Medium", "High"], n_samples),
    "BloodPressure": np.random.randint(90, 180, n_samples),
    "RestingHeartRate": np.random.randint(60, 110, n_samples),
    "ChestPain": np.random.choice(["Yes", "No"], n_samples),
    "ShortnessOfBreath": np.random.choice(["Yes", "No"], n_samples),
    "FamilyHistory": np.random.choice(["Yes", "No"], n_samples),
}

df = pd.DataFrame(data)

# ----- Step 2: Create target (HeartDisease) -----
# Just a simulated rule-based pattern to make it realistic
df["HeartDisease"] = (
    (df["Age"] > 50).astype(int)
    + (df["BMI"] > 30).astype(int)
    + (df["Smoking"] == "Yes").astype(int)
    + (df["ChestPain"] == "Yes").astype(int)
    + (df["ShortnessOfBreath"] == "Yes").astype(int)
    + (df["BloodPressure"] > 140).astype(int)
    + (df["FamilyHistory"] == "Yes").astype(int)
)
df["HeartDisease"] = (df["HeartDisease"] > 2).astype(int)

# ----- Step 3: Convert categorical to numeric -----
df_encoded = pd.get_dummies(df.drop("HeartDisease", axis=1), drop_first=True)
X = df_encoded
y = df["HeartDisease"]

# ----- Step 4: Train-Test Split -----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----- Step 5: Train Model -----
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ----- Step 6: Save Model -----
joblib.dump(model, "model/heart_disease_model.pkl")
print("✅ Model trained and saved successfully!")

# Optional: Show accuracy
accuracy = model.score(X_test, y_test) * 100
print(f"Model Accuracy: {accuracy:.2f}%")
