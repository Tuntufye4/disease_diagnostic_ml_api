import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (    
    confusion_matrix, classification_report, accuracy_score,
    roc_curve, roc_auc_score
)
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# 1. Load dataset
df = pd.read_csv("cholera_symptoms.csv")

# 2. Features (X) and Target (y)
X = df.drop("cholera_diagnosis", axis=1)
y = df["cholera_diagnosis"]

# 3. Split into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # probabilities for ROC

# 6. Confusion Matrix + Reports
cm = confusion_matrix(y_test, y_pred)
print("Classification Report:\n", classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Plot confusion matrix
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Cholera", "Cholera"],
            yticklabels=["No Cholera", "Cholera"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# 7. ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.2f})", color="blue")
plt.plot([0,1], [0,1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

# 8. Feature Importance
importances = model.feature_importances_
feature_names = X.columns
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(x=feat_imp.values, y=feat_imp.index, palette="viridis")
plt.xlabel("Importance")
plt.ylabel("Symptom")
plt.title("Feature Importance in Cholera Diagnostics")
plt.show()

# 9. Save the model
joblib.dump(model, "cholera_diagnostics_model.joblib")
print("Model saved as cholera_diagnostics_model.joblib")
