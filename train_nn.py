import pandas as pd
import numpy as np
import pickle
import json
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ─────────────────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────────────────
print("📱 Loading mobile phone dataset...")
df = pd.read_csv("data_2/train.csv")

X        = df.drop("price_range", axis=1)
y        = df["price_range"]
features = list(X.columns)

print(f"✅ Dataset: {len(df)} rows, {len(features)} features")
print(f"   Classes: {sorted(y.unique())}")

# ─────────────────────────────────────────────────────────
# Split & Scale
# ─────────────────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)

# ─────────────────────────────────────────────────────────
# Train Neural Network
# ─────────────────────────────────────────────────────────
print("\n🧠 Training MLP Neural Network (128→64→32)...")
model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    verbose=False,
)
model.fit(X_train_s, y_train)

y_pred = model.predict(X_val_s)
acc    = accuracy_score(y_val, y_pred)

print(f"\n✅ Validation Accuracy: {acc*100:.2f}%")
print("\n📊 Classification Report:")
print(classification_report(
    y_val, y_pred,
    target_names=["Class 0 (ถูกมาก)", "Class 1 (ปานกลาง)",
                  "Class 2 (สูง)", "Class 3 (Flagship)"]
))

# ─────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────
print("💾 Saving to models/ ...")

with open("models/mobile_nn_model.pkl", "wb") as f: pickle.dump(model,    f)
with open("models/mobile_scaler.pkl",   "wb") as f: pickle.dump(scaler,   f)
with open("models/mobile_features.pkl", "wb") as f: pickle.dump(features, f)

metrics = {
    "accuracy":   float(acc),
    "n_samples":  int(len(df)),
    "n_features": int(len(features)),
    "n_val":      int(len(y_val)),
}
with open("models/mobile_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\n✅ Saved files:")
print("   models/mobile_nn_model.pkl")
print("   models/mobile_scaler.pkl")
print("   models/mobile_features.pkl")
print("   models/mobile_metrics.json")
print("\n🎉 Done! Now run: streamlit run app.py")