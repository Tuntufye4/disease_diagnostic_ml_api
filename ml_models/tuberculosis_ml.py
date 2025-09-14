# train_tb_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    roc_curve, roc_auc_score
)
import matplotlib.pyplot as plt
import joblib
import os

# -------------------------
# CONFIG
# -------------------------
CSV_PATH = "tuberculosis_symptoms.csv"  # change if file is elsewhere
MODEL_OUT = "tb_diagnostics_pipeline.joblib"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# -------------------------
# 1) Load data
# -------------------------
df = pd.read_csv(CSV_PATH)

# Drop any non-feature columns if present (Patient_ID)
if "Patient_ID" in df.columns:
    df = df.drop(columns=["Patient_ID"])

# -------------------------
# 2) Define X and y
# -------------------------
TARGET = "TB_Diagnosis"
if TARGET not in df.columns:
    raise ValueError(f"{TARGET} column not found in {CSV_PATH}")

X = df.drop(columns=[TARGET])
y = df[TARGET]

# -------------------------
# 3) Preprocessing
# -------------------------
# specify categorical and numerical columns
categorical_cols = [col for col in X.columns if X[col].dtype == "object" or X[col].dtype.name == "category"]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

print(categorical_cols)   
print(numeric_cols)
# Build ColumnTransformer (OneHot encode categorical columns)
preproc = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), categorical_cols),
        # numeric columns are passed through unchanged
    ],
    remainder="passthrough",
    sparse_threshold=0
)

# Create pipeline: preprocessing + classifier
clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, class_weight="balanced")
pipeline = Pipeline([
    ("preproc", preproc),
    ("clf", clf)
])

# -------------------------
# 4) Train/test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# -------------------------
# 5) Fit pipeline
# -------------------------
pipeline.fit(X_train, y_train)

# -------------------------
# 6) Predictions & metrics
# -------------------------
y_pred = pipeline.predict(X_test)

# If classifier supports predict_proba, get probabilities for ROC
if hasattr(pipeline.named_steps["clf"], "predict_proba"):
    y_proba = pipeline.predict_proba(X_test)[:, 1]
else:
    # fallback to decision_function if available, else use predicted labels (not ideal)
    if hasattr(pipeline.named_steps["clf"], "decision_function"):
        y_proba = pipeline.decision_function(X_test)
    else:
        y_proba = y_pred  # not probabilistic

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# -------------------------
# 7) Plot confusion matrix (matplotlib)
# -------------------------
fig, ax = plt.subplots(figsize=(5,4))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
classes = ["No TB", "TB"]
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=classes, yticklabels=classes,
       xlabel="Predicted label", ylabel="True label",
       title="Confusion Matrix")
# annotate cells
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.show()

# -------------------------
# 8) ROC Curve & AUC
# -------------------------
try:
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
    print(f"AUC: {auc:.4f}")
except Exception as e:
    print("ROC/AUC could not be computed:", e)

# -------------------------
# 9) Feature importance
# -------------------------
# We need feature names after the preprocessing step
preproc_fitted = pipeline.named_steps["preproc"]
clf_fitted = pipeline.named_steps["clf"]

# get feature names created by the preprocessor
try:
    # sklearn >= 1.0: ColumnTransformer has get_feature_names_out
    feature_names = preproc_fitted.get_feature_names_out()
except Exception:
    # fallback: build names manually
    feature_names = []
    # handle categorical columns (onehot)
    if categorical_cols:
        ohe = preproc_fitted.named_transformers_["cat"]
        try:
            cat_names = ohe.get_feature_names_out(categorical_cols)
            feature_names.extend(cat_names)
        except Exception:
            # simpler fallback
            for c in categorical_cols:
                # attempt to get categories
                cats = getattr(ohe, "categories_", [None])
                feature_names.append(c)
    # add numeric columns
    feature_names.extend(numeric_cols)

# feature importances from classifier
importances = clf_fitted.feature_importances_
# Ensure equal length
if len(importances) == len(feature_names):
    # sort
    idx = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in idx]
    sorted_importances = importances[idx]

    plt.figure(figsize=(8,6))
    plt.barh(sorted_names, sorted_importances)
    plt.gca().invert_yaxis()
    plt.xlabel("Feature importance")
    plt.title("Feature importances (Random Forest)")
    plt.tight_layout()
    plt.show()
else:
    print("Warning: mismatch between number of feature importances and feature names.")

# -------------------------
# 10) Save the pipeline (preprocessing + model)
# -------------------------
joblib.dump(pipeline, MODEL_OUT)
print(f"Saved trained pipeline to: {MODEL_OUT}")

# -------------------------
# 11) Example: load and predict single sample
# -------------------------
# Example usage (uncomment to test)
# loaded = joblib.load(MODEL_OUT)
# example = {
#     "Age": 35,
#     "Sex": "Male",
#     "Cough_Duration_weeks": 3,
#     "Fever": 1,
#     "Night_Sweats": 1,
#     "Weight_Loss": 1,
#     "Fatigue": 1,
#     "Chest_Pain": 0,
#     "Hemoptysis": 0,
#     "HIV_Status": "Negative",
#     "Smoker": 0
# }
# ex_df = pd.DataFrame([example])
# pred = loaded.predict(ex_df)[0]
# proba = loaded.predict_proba(ex_df)[0][1] if hasattr(loaded.named_steps['clf'], 'predict_proba') else None
# print("Pred:", pred, "Proba:", proba)
    