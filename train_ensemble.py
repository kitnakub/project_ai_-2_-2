import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# โหลดข้อมูล
df = pd.read_csv("data_set/processed_new_laptop.csv")

# คำนวณ bins จากข้อมูลจริง
percentiles = [0, 15, 30, 45, 60, 75, 90, 100]
bins = [int(np.percentile(df["price"], p)) for p in percentiles]
# bins[0] = 0
bins[-1] = 999999
labels = []
for i in range(len(bins)-1):
    lo = bins[i] // 1000
    hi = bins[i+1] // 1000
    if bins[i+1] == 999999:
        labels.append(f"{lo}k+")
    else:
        labels.append(f"{lo}k-{hi}k")
joblib.dump({"bins": bins, "labels": labels}, "models/price_bins.pkl")
print(f"Bins: {bins}")
print(f"Labels: {labels}")

X = df.drop("price", axis=1)
y = df["price"]

print(f"Feature shape: {X.shape}")
print(f"Price range: {y.min():,.0f} - {y.max():,.0f} INR")

# แบ่ง train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# -------------------------------------------------------
# 3 โมเดล คนละประเภท 
# -------------------------------------------------------

# โมเดลที่ 1: Random Forest
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_leaf=3,
    max_features=0.6,
    random_state=42,
    n_jobs=-1
)

# โมเดลที่ 2: Gradient Boosting
gb = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    min_samples_leaf=5,
    subsample=0.8,
    random_state=42
)

# โมเดลที่ 3: Extra Trees (คนละประเภทกับ RF — สุ่ม threshold แบบสมบูรณ์)
et = ExtraTreesRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

# รวม 3 โมเดลเป็น Ensemble
ensemble = VotingRegressor([
    ("rf", rf),
    ("gb", gb),
    ("et", et)
])

print("\nTraining Ensemble Model (RF + GradientBoosting + ExtraTrees)...")
ensemble.fit(X_train, y_train)

# ประเมินผล
y_pred = ensemble.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"\n=== Test Set Results ===")
print(f"MAE:  {mae:,.0f} INR")
print(f"RMSE: {rmse:,.0f} INR")
print(f"R²:   {r2:.4f}")

# ตรวจสอบ overfit
y_train_pred = ensemble.predict(X_train)
r2_train = r2_score(y_train, y_train_pred)
print(f"\nTrain R²: {r2_train:.4f}")
print(f"Test  R²: {r2:.4f}")
if r2_train - r2 > 0.1:
    print("⚠️  อาจมี overfitting!")
else:
    print("✅ ไม่มีปัญหา overfitting")

# บันทึก model
joblib.dump(ensemble, "models/ensemble_model.pkl")
joblib.dump({"mae": mae, "rmse": rmse, "r2": r2}, "models/classifier_metrics.pkl")
print("\n✅ Saved ensemble_model.pkl")
print("✅ Saved classifier_metrics.pkl")
print("✅ Saved price_bins.pkl")