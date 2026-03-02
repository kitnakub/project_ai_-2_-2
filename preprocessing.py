import pandas as pd
import joblib
import os

os.makedirs("models", exist_ok=True)

# โหลดข้อมูล
df = pd.read_csv("data_set/data_new_laptop.csv")

# ลบ column ที่ไม่จำเป็น
df = df.drop(columns=["Unnamed: 0.1", "Unnamed: 0"])

# ลบ CPU เพราะมีค่าไม่ซ้ำมากเกินไป
# หมายเหตุ: ยังไม่ลบ 'name' ตอนนี้ เพราะต้องใช้ทำ is_gaming ก่อน
df = df.drop(columns=["CPU"])

# แปลง Ram: '8GB' → 8
df["Ram"] = df["Ram"].str.replace("GB", "").astype(int)

# แปลง ROM: '512GB' → 512, '1TB' → 1024, '2TB' → 2048
def parse_rom(val):
    val = str(val).strip()
    if "TB" in val:
        return int(val.replace("TB", "")) * 1024
    elif "GB" in val:
        return int(val.replace("GB", ""))
    return int(val)

df["ROM"] = df["ROM"].apply(parse_rom)

# -------------------------------------------------------
# ✨ ENGINEERED FEATURES ใหม่
# -------------------------------------------------------

# 1. is_gaming — ตรวจจาก GPU ที่มี RTX/GTX/RX หรือชื่อรุ่นมีคำว่า Gaming
gaming_gpu_keywords = ["RTX", "GTX", "RX 6500", "RX 6650", "RX6500", "RX6550"]
gaming_name_keywords = ["Gaming", "TUF", "ROG", "Nitro", "Predator",
                        "Victus", "Legion", "Katana", "Sword", "Bravo",
                        "Cyborg", "Pulse", "Omen", "Helios", "Aorus",
                        "Raider", "GF63", "Strix", "Zephyrus"]

df["is_gaming"] = (
    df["GPU"].str.contains("|".join(gaming_gpu_keywords), case=False, na=False) |
    df["name"].str.contains("|".join(gaming_name_keywords), case=False, na=False)
).astype(int)

# 2. is_premium_brand — แบรนด์ที่ขาย premium เกินสเปค
premium_brands = ["Apple", "Razer", "LG", "Microsoft", "Vaio",
                  "Samsung", "Huawei", "Honor"]
df["is_premium_brand"] = df["brand"].isin(premium_brands).astype(int)

# 3. gpu_tier — แบ่ง GPU เป็น 5 ระดับ (0=integrated, 1=entry, 2=mid, 3=high, 4=flagship)
def get_gpu_tier(gpu_str):
    g = str(gpu_str).upper()
    if any(k in g for k in ["RTX 4090", "RTX 3080", "RTX 4080"]):
        return 4
    elif any(k in g for k in ["RTX 4070", "RTX 3070", "RTX 3060"]):
        return 3
    elif any(k in g for k in ["RTX 4060", "RTX 4050", "RTX 3050", "RTX 2050",
                               "GTX 1650", "RX 6500", "RX 6650"]):
        return 2
    elif any(k in g for k in ["MX", "GTX 1050", "GEFORCE"]):
        return 1
    else:
        return 0  # integrated

df["gpu_tier"] = df["GPU"].apply(get_gpu_tier)

# 4. processor_tier — แบ่ง CPU เป็น 4 ระดับ
def get_processor_tier(proc_str):
    p = str(proc_str).upper()
    if any(k in p for k in ["I9", "RYZEN 9", "M2 MAX", "M2 PRO", "M1 MAX", "M1 PRO"]):
        return 4
    elif any(k in p for k in ["I7", "RYZEN 7", "M2", "M1"]):
        return 3
    elif any(k in p for k in ["I5", "RYZEN 5"]):
        return 2
    elif any(k in p for k in ["I3", "RYZEN 3"]):
        return 1
    else:
        return 0  # celeron, pentium, atom

df["processor_tier"] = df["processor"].apply(get_processor_tier)

# 5. ram_x_storage — interaction feature (RAM × ROM สะท้อนความ "ครบ" ของสเปค)
df["ram_x_storage"] = df["Ram"] * df["ROM"]

# 6. is_latest_gen — processor gen ใหม่ (12th/13th Intel หรือ 7th AMD)
def is_latest_gen(proc_str):
    p = str(proc_str)
    if any(k in p for k in ["13th", "12th", "7th Gen AMD", "M2"]):
        return 1
    return 0

df["is_latest_gen"] = df["processor"].apply(is_latest_gen)

print("✅ Feature engineering สำเร็จ:")
print(f"   is_gaming:        {df['is_gaming'].sum()} gaming laptops")
print(f"   is_premium_brand: {df['is_premium_brand'].sum()} premium brand laptops")
print(f"   gpu_tier dist:    {df['gpu_tier'].value_counts().to_dict()}")
print(f"   processor_tier:   {df['processor_tier'].value_counts().to_dict()}")
print(f"   is_latest_gen:    {df['is_latest_gen'].sum()} latest gen")

# ลบ 'name' หลังจากทำ feature engineering เสร็จแล้ว
df = df.drop(columns=["name"])

# เลือก features ที่ใช้
categorical_cols = ["brand", "processor", "GPU", "OS", "Ram_type", "ROM_type"]
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

print("Encoded shape:", df_encoded.shape)

# บันทึกข้อมูล
df_encoded.to_csv("data_set/processed_new_laptop.csv", index=False)

# บันทึก feature columns (ไม่รวม price)
feature_cols = df_encoded.drop("price", axis=1).columns.tolist()
joblib.dump(feature_cols, "models/feature_columns.pkl")

print(f"\nTotal features: {len(feature_cols)}")
print("Saved processed_new_laptop.csv")
print("Saved feature_columns.pkl")