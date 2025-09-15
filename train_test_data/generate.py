import pandas as pd
import numpy as np

np.random.seed(42)
n = 700

data = {
    "Patient_ID": range(1, n+1),
    "Age": np.random.randint(10, 80, n),
    "Sex": np.random.choice(["Male", "Female"], n),
    "Frequent_Urination": np.random.choice([0,1], n, p=[0.3,0.7]),
    "Excessive_Thirst": np.random.choice([0,1], n, p=[0.4,0.6]),
    "Unexplained_Weight_Loss": np.random.choice([0,1], n, p=[0.6,0.4]),
    "Extreme_Hunger": np.random.choice([0,1], n, p=[0.5,0.5]),
    "Fatigue": np.random.choice([0,1], n, p=[0.4,0.6]),
    "Blurred_Vision": np.random.choice([0,1], n, p=[0.7,0.3]),
    "Slow_Healing_Sores": np.random.choice([0,1], n, p=[0.8,0.2]),
    "Tingling_Numbness": np.random.choice([0,1], n, p=[0.7,0.3]),
    "Family_History": np.random.choice(["Yes","No"], n, p=[0.5,0.5]),
    "Obesity": np.random.choice(["Yes","No"], n, p=[0.4,0.6])
}

# Generate diagnosis based on symptoms
diagnosis = []
for i in range(n):
    score = 0
    if data["Frequent_Urination"][i]: score += 1
    if data["Excessive_Thirst"][i]: score += 1
    if data["Unexplained_Weight_Loss"][i]: score += 2
    if data["Extreme_Hunger"][i]: score += 1
    if data["Fatigue"][i]: score += 1
    if data["Blurred_Vision"][i]: score += 1
    if data["Slow_Healing_Sores"][i]: score += 1
    if data["Tingling_Numbness"][i]: score += 1
    if data["Family_History"][i] == "Yes": score += 1
    if data["Obesity"][i] == "Yes": score += 1
    diagnosis.append(1 if score >= 4 else 0)

data["Diabetes_Diagnosis"] = diagnosis

df = pd.DataFrame(data)
file_path = "diabetes_symptoms.csv"
df.to_csv(file_path, index=False)
print(f"✅ Saved {file_path}")
