# measles_ml.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import joblib

# -------------------------
# 1) Load dataset
# -------------------------
CSV_PATH = "measles_symptoms.csv"  # Update path if needed
df = pd.read_csv(CSV_PATH)

# Drop non-feature column
if "Patient_ID" in df.columns:
    df = df.drop(columns=["Patient_ID"])

# -------------------------
# 2) Define X and y
# -------------------------
TARGET = "Measles_Diagnosis"
X = df.drop(columns=[TARGET])
y = df[TARGET]

# -------------------------
# 3) Preprocessing
# -------------------------
categorical_cols = [c for c in X.columns if X[c].dtype == "object"]
numeric_cols = [c for c in X.columns if c not in categorical_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), categorical_cols)
    ],
    remainder="passthrough",
    sparse_threshold=0
)

# -------------------------
# 4) Create pipeline
# -------------------------
clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", clf)
])

# -------------------------
# 5) Train/test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# 6) Train the model
# -------------------------
pipeline.fit(X_train, y_train)

# -------------------------
# 7) Predictions & evaluation
# -------------------------
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(5,4))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks([0,1], ["Negative","Positive"])
plt.yticks([0,1], ["Negative","Positive"])
plt.ylabel("True label")
plt.xlabel("Predicted label")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i,j], ha="center", va="center", color="white" if cm[i,j]>cm.max()/2 else "black")
plt.tight_layout()
plt.show()

# -------------------------
# 8) Feature importances
# -------------------------
feature_names = []
try:
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
except:
    feature_names = categorical_cols + numeric_cols

importances = pipeline.named_steps["classifier"].feature_importances_
if len(importances) == len(feature_names):
    idx = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in idx]
    sorted_importances = importances[idx]

    plt.figure(figsize=(8,6))
    plt.barh(sorted_features, sorted_importances)
    plt.gca().invert_yaxis()
    plt.xlabel("Feature Importance")
    plt.title("Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.show()

# -------------------------
# 9) Save the trained pipeline
# -------------------------
MODEL_FILE = "measles_pipeline.joblib"
joblib.dump(pipeline, MODEL_FILE)
print(f"Trained pipeline saved as: {MODEL_FILE}")

# -------------------------
# 10) Example: load and predict a single sample
# -------------------------
# loaded = joblib.load(MODEL_FILE)
# example = {
#     "Age": 5,
#     "Sex": "Male",
#     "Fever": 1,
#     "Cough": 1,
#     "Runny_Nose": 1,
#     "Red_Eyes": 1,
#     "Koplik_Spots": 1,
#     "Rash": 1,
#     "Fatigue": 1,
#     "Muscle_Aches": 0,
#     "Sore_Throat": 0,
#     "Vomiting": 0,
#     "Comorbidity": "None",
#     "Vaccinated": "No"
# }
# ex_df = pd.DataFrame([example])
# pred = loaded.predict(ex_df)[0]
# proba = loaded.predict_proba(ex_df)[0][1]
# print("Prediction:", pred, "Probability of Measles:", proba)
