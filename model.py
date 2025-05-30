import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
import sys
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Get file path from command line
if len(sys.argv) < 2:
    print("Usage: python model.py <file_path>")
    sys.exit(1)

file_path = sys.argv[1]

# ---- Load Dataset ----
df = pd.read_csv(file_path)

# ---- Data Preprocessing ----
for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].median())
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])

label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# ---- Create Graphs Folder ----
graph_dir = "graphs"
os.makedirs(graph_dir, exist_ok=True)

# ---- Train-Test Split ----
X = df.drop(columns=["Churn"])
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, "models/scaler.pkl")

# ---- XGBoost Model ----
xgb_model = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, subsample=0.8, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_preds)

# ---- ANN Model ----
ann_model = Sequential([
    Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

ann_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
ann_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1, callbacks=[reduce_lr])
ann_preds = (ann_model.predict(X_test) > 0.5).astype("int32")
ann_acc = accuracy_score(y_test, ann_preds)

# ---- Hybrid Model ----
weights = [0.6, 0.4]
hybrid_preds = (weights[0] * xgb_preds + weights[1] * ann_preds.flatten())
hybrid_preds = (hybrid_preds > 0.5).astype(int)
hybrid_acc = accuracy_score(y_test, hybrid_preds)

# ---- Save Results as JSON ----
results = {
    "XGBoost Accuracy": xgb_acc,
    "ANN Accuracy": ann_acc,
    "Hybrid Model Accuracy": hybrid_acc,
    "Confusion Matrix": confusion_matrix(y_test, hybrid_preds).tolist(),
    "Classification Report": classification_report(y_test, hybrid_preds, output_dict=True)
}

with open("output.json", "w") as f:
    json.dump(results, f, indent=4)

# ---- Save Models ----
ann_model.save("models/ann_model.h5")
xgb_model.save_model("models/xgb_model.json")

# ---- Generate and Save Graphs ----
sns.set_style("whitegrid")

# Churn Distribution
plt.figure(figsize=(8, 4))
sns.countplot(x=y)
plt.title("Churn Distribution")
plt.savefig(os.path.join(graph_dir, "churn_distribution.png"))
plt.close()

# Feature Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.savefig(os.path.join(graph_dir, "feature_correlation_heatmap.png"))
plt.close()

# XGBoost Feature Importance
plt.figure(figsize=(8, 5))
xgb.plot_importance(xgb_model, max_num_features=10)
plt.title("Top 10 Feature Importance in XGBoost")
plt.savefig(os.path.join(graph_dir, "xgboost_feature_importance.png"))
plt.close()

# Confusion Matrix
plt.figure(figsize=(6, 6))
sns.heatmap(confusion_matrix(y_test, hybrid_preds), annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(graph_dir, "confusion_matrix.png"))
plt.close()

print("Graphs saved successfully!")
