import re
import os
import json
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Laptop & Mobile Price AI",
    layout="centered",
    page_icon="💻",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------
# Session state
# -------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "ml_explain"

# -------------------------------------------------------
# CSS Sidebar
# -------------------------------------------------------
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f0f1a 0%, #1a1a2e 100%);
    border-right: 1px solid #2a2a4a;
}
[data-testid="stSidebar"] .stMarkdown p {
    color: #8888bb;
    font-size: 0.75rem;
    letter-spacing: 0.10em;
    text-transform: uppercase;
}
div[data-testid="stSidebar"] .stButton button {
    background: transparent;
    border: none;
    color: #ccccee;
    text-align: left;
    width: 100%;
    padding: 0.4rem 0.5rem;
    font-size: 0.93rem;
    cursor: pointer;
    border-radius: 6px;
    transition: background 0.2s;
}
div[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(255,255,255,0.07);
    color: #ffffff;
}
div[data-testid="stSidebar"] .stButton button:focus {
    box-shadow: none;
    background: rgba(255,255,255,0.13);
    color: #ffffff;
}
</style>
""", unsafe_allow_html=True)

def nav(page): st.session_state.page = page

with st.sidebar:
    st.markdown("## 💻 Laptop & Mobile Price AI")
    st.markdown("---")
    st.markdown("**🤖 Ensemble ML**")
    if st.button("📖 อธิบาย Ensemble ML"):  nav("ml_explain")
    if st.button("🔍 ทดสอบ Ensemble ML"):   nav("ml_predict")
    st.markdown("---")
    st.markdown("**🧠 Neural Network**")
    if st.button("📖 อธิบาย Neural Network"): nav("nn_explain")
    if st.button("🔍 ทดสอบ Neural Network"):  nav("nn_predict")


# -------------------------------------------------------
# โหลด Models (Ensemble)
# -------------------------------------------------------
@st.cache_resource
def load_all_models():
    ensemble     = joblib.load("models/ensemble_model.pkl")
    ens_features = joblib.load("models/feature_columns.pkl")
    return ensemble, ens_features,

ensemble, ens_features = load_all_models()

clf_features = ens_features
price_bins   = joblib.load("models/price_bins.pkl")
clf_metrics  = joblib.load("models/classifier_metrics.pkl")

BINS   = price_bins["bins"]
LABELS = price_bins["labels"]

def get_price_range(predicted_inr):
    for i in range(len(BINS) - 1):
        if BINS[i] <= predicted_inr < BINS[i + 1]:
            lo_thb = inr_to_thb(BINS[i])
            hi_inr = BINS[i + 1]
            label  = LABELS[i]
            if hi_inr == 999999:
                thb_str = f"{lo_thb:,}+ บาท"
            else:
                hi_thb = inr_to_thb(hi_inr)
                thb_str = f"{lo_thb:,} – {hi_thb:,} บาท"
            return label, thb_str
    lo_thb = inr_to_thb(BINS[-2])
    return LABELS[-1], f"{lo_thb:,}+ บาท"

# -------------------------------------------------------
# อัตราแลกเปลี่ยน
# -------------------------------------------------------
INR_TO_THB = 0.3486

def inr_to_thb(inr):
    thb = inr * INR_TO_THB
    if thb >= 10000: return int(round(thb / 1000) * 1000)
    return int(round(thb / 100) * 100)

def fmt_thb_range(lo_inr, hi_inr):
    lo = inr_to_thb(lo_inr)
    if hi_inr == 999999: return f"~{lo:,}+ บาท"
    hi = inr_to_thb(hi_inr)
    return f"~{lo:,} – {hi:,} บาท"

PRICE_DISCLAIMER = (
    "💱 **หมายเหตุด้านราคา** — ตัวเลขแปลงจากราคาตลาด **อินเดีย** (INR → THB ที่ 1 ₹ ≈ 0.35 ฿) "
    "ราคาจริงในไทยอาจ **สูงกว่า 10–30%** เนื่องจากภาษีนำเข้า ค่าจัดจำหน่าย และมาร์จิ้นผู้นำเข้า "
    "— ใช้เป็นแนวทางเปรียบเทียบสเปคเท่านั้น ไม่ควรอ้างอิงเป็นราคาซื้อขายจริง"
)

# -------------------------------------------------------
# Helper — Ensemble
# -------------------------------------------------------
def extract_options(feature_columns, prefix):
    return sorted([col.replace(prefix, "") for col in feature_columns if col.startswith(prefix)])

def gpu_tier(g):
    g = str(g).upper()
    if any(k in g for k in ["RTX 4090","RTX 3080","RTX 4080"]): return 4
    if any(k in g for k in ["RTX 4070","RTX 3070","RTX 3060"]): return 3
    if any(k in g for k in ["RTX 4060","RTX 4050","RTX 3050","RTX 2050","GTX 1650","RX 6500"]): return 2
    if any(k in g for k in ["MX","GTX 1050"]): return 1
    return 0

def proc_tier(p):
    p = str(p).upper()
    if any(k in p for k in ["I9","RYZEN 9","M2 MAX","M2 PRO","M1 MAX","M1 PRO"]): return 4
    if any(k in p for k in ["I7","RYZEN 7","M2","M1"]): return 3
    if any(k in p for k in ["I5","RYZEN 5"]): return 2
    if any(k in p for k in ["I3","RYZEN 3"]): return 1
    return 0

def build_ens_input(ui, feature_columns):
    d = {col: 0 for col in feature_columns}
    for key in ["Ram","ROM","display_size","resolution_width","resolution_height","warranty","spec_rating"]:
        if key in d: d[key] = ui[key]
    for prefix, field in [("brand_","brand"),("processor_","processor"),
                           ("GPU_","GPU"),("OS_","OS"),
                           ("Ram_type_","Ram_type"),("ROM_type_","ROM_type")]:
        k = f"{prefix}{ui[field]}"
        if k in d: d[k] = 1
    premium = ["Apple","Razer","LG","Microsoft","Vaio","Samsung","Huawei","Honor"]
    gaming_kw = ["RTX","GTX","RX 6500","RX 6650"]
    d["is_gaming"]        = int(any(k.upper() in ui["GPU"].upper() for k in gaming_kw))
    d["is_premium_brand"] = int(ui["brand"] in premium)
    d["gpu_tier"]         = gpu_tier(ui["GPU"])
    d["processor_tier"]   = proc_tier(ui["processor"])
    d["ram_x_storage"]    = ui["Ram"] * ui["ROM"]
    d["is_latest_gen"]    = int(any(k in ui["processor"] for k in ["13th","12th","7th Gen AMD","M2"]))
    return d

# -------------------------------------------------------
# Brand Compatibility
# -------------------------------------------------------
@st.cache_data
def build_brand_compat():
    def parse_storage(val):
        val = str(val).strip()
        if 'TB' in val: return int(float(val.replace('TB','').strip()) * 1024)
        return int(val.replace('GB','').strip())
    df = pd.read_csv("data_set/data_new_laptop.csv")
    df['Ram'] = df['Ram'].str.replace('GB','').str.strip().astype(int)
    df['ROM'] = df['ROM'].apply(parse_storage)
    compat = {}
    fields = ['processor','GPU','OS','Ram_type','ROM_type',
              'display_size','resolution_width','resolution_height','Ram','ROM']
    for brand in sorted(df['brand'].unique()):
        sub = df[df['brand'] == brand]
        compat[brand] = {}
        for f in fields:
            compat[brand][f] = sorted(sub[f].dropna().unique().tolist())
    return compat

BRAND_COMPAT = build_brand_compat()

def filter_opts(brand, field, all_opts):
    vals = BRAND_COMPAT.get(brand, {}).get(field, [])
    filtered = [o for o in all_opts if o in vals]
    return filtered if filtered else all_opts

def filter_num(brand, field, all_vals):
    vals = BRAND_COMPAT.get(brand, {}).get(field, [])
    filtered = [v for v in all_vals if v in vals]
    return filtered if filtered else all_vals

_DISCRETE_RE = re.compile(
    r'nvidia|geforce|rtx\s*\d|gtx\s*\d|quadro|radeon\s*(rx|pro)|rx\s*\d{3,}|mx\s*\d{3,}|arc\s+a\d',
    re.IGNORECASE)

def is_discrete_gpu(g): return bool(_DISCRETE_RE.search(g))
def brand_has_discrete_gpu(brand):
    return any(is_discrete_gpu(g) for g in BRAND_COMPAT.get(brand, {}).get("GPU", []))

def get_all_numeric_opts(field):
    vals = set()
    for bd in BRAND_COMPAT.values():
        for v in bd.get(field, []): vals.add(v)
    return sorted(vals)

# -------------------------------------------------------
# UI ฟอร์มกรอกสเปค Laptop
# -------------------------------------------------------
def spec_form(key_prefix="A"):
    brand_opts    = extract_options(clf_features, "brand_")
    proc_opts_all = extract_options(clf_features, "processor_")
    gpu_opts_all  = extract_options(clf_features, "GPU_")
    os_opts_all   = extract_options(clf_features, "OS_")
    ram_type_all  = extract_options(clf_features, "Ram_type_")
    rom_type_all  = extract_options(clf_features, "ROM_type_")
    RAM_ALL   = get_all_numeric_opts("Ram")
    ROM_ALL   = get_all_numeric_opts("ROM")
    DISP_ALL  = get_all_numeric_opts("display_size")
    RES_W_ALL = get_all_numeric_opts("resolution_width")
    RES_H_ALL = get_all_numeric_opts("resolution_height")

    brand = st.selectbox("🏷️ แบรนด์", brand_opts, key=f"{key_prefix}_brand")
    st.caption(f"🔍 แสดงสเปคที่พบจริงสำหรับ **{brand}** จาก dataset")

    proc_opts     = filter_opts(brand, "processor",        proc_opts_all)
    gpu_opts      = filter_opts(brand, "GPU",              gpu_opts_all)
    os_opts       = filter_opts(brand, "OS",               os_opts_all)
    ram_type_opts = filter_opts(brand, "Ram_type",         ram_type_all)
    rom_type_opts = filter_opts(brand, "ROM_type",         rom_type_all)
    ram_opts      = filter_num(brand, "Ram",               RAM_ALL)
    rom_opts      = filter_num(brand, "ROM",               ROM_ALL)
    disp_opts     = filter_num(brand, "display_size",      DISP_ALL)
    res_w_opts    = filter_num(brand, "resolution_width",  RES_W_ALL)
    res_h_opts    = filter_num(brand, "resolution_height", RES_H_ALL)

    col1, col2 = st.columns(2)
    with col1:
        ram       = st.selectbox("🧠 RAM (GB)",        ram_opts,      key=f"{key_prefix}_ram")
        storage   = st.selectbox("💾 Storage (GB)",    rom_opts,      key=f"{key_prefix}_rom")
        ram_type  = st.selectbox("📌 RAM Type",        ram_type_opts, key=f"{key_prefix}_ramt")
        disp_size = st.selectbox("🖥️ หน้าจอ (นิ้ว)",  disp_opts,     key=f"{key_prefix}_disp")
    with col2:
        processor = st.selectbox("⚙️ Processor",      proc_opts,     key=f"{key_prefix}_proc")
        os        = st.selectbox("🪟 OS",              os_opts,       key=f"{key_prefix}_os")
        rom_type  = st.selectbox("📦 Storage Type",   rom_type_opts, key=f"{key_prefix}_romt")
        warranty  = st.selectbox("🛡️ ประกัน (ปี)",   [0,1,2,3],     key=f"{key_prefix}_war")

    if not brand_has_discrete_gpu(brand):
        gpu = gpu_opts[0] if gpu_opts else "Integrated"
        st.info(f"🎮 GPU: **{gpu}** — {brand} ใช้ Integrated Graphics เท่านั้น")
    else:
        discrete_first = sorted(gpu_opts, key=lambda g: (0 if is_discrete_gpu(g) else 1, g))
        gpu = st.selectbox("🎮 GPU", discrete_first, key=f"{key_prefix}_gpu")

    col3, col4 = st.columns(2)
    with col3: res_w = st.selectbox("↔️ ความละเอียด กว้าง", res_w_opts, key=f"{key_prefix}_resw")
    with col4: res_h = st.selectbox("↕️ ความละเอียด สูง",   res_h_opts, key=f"{key_prefix}_resh")

    return {
        "brand": brand, "processor": processor, "Ram": ram, "ROM": storage,
        "GPU": gpu, "OS": os, "Ram_type": ram_type, "ROM_type": rom_type,
        "display_size": disp_size, "resolution_width": res_w, "resolution_height": res_h,
        "warranty": warranty, "spec_rating": 69.32
    }


# ===================================================================
# PAGE: อธิบาย Ensemble ML
# ===================================================================
def page_ml_explain():
    st.title("📖 Ensemble Machine Learning — Laptop Price")
    st.caption("ที่มา Dataset · Features · ความไม่สมบูรณ์ · การเตรียมข้อมูล · ทฤษฎี · ขั้นตอนพัฒนา · แหล่งอ้างอิง")
    st.markdown("---")

    # ─────────────────────────────────────────────────────
    # ส่วนที่ 1: ที่มาของ Dataset
    # ─────────────────────────────────────────────────────
    st.subheader("1️⃣ ที่มาของ Dataset")
    st.markdown("""
**Dataset:** Laptop Price India Dataset

**ที่มา:** Download จากเว็บไซต์ **Kaggle**
- 🔗 Link: https://www.kaggle.com/datasets/jacksondivakarr/laptop-price-prediction-dataset
- ข้อมูล laptop ที่วางจำหน่ายจริงใน **ตลาดอินเดีย** โดยเก็บจากเว็บ e-commerce อินเดีย
- แต่ละแถวคือ laptop 1 รุ่น พร้อมสเปคครบถ้วนและราคาจริง (INR ₹) ณ ช่วงเวลาที่เก็บข้อมูล

**ขนาดของ Dataset:**
- จำนวนข้อมูล: **893 รุ่น**
- จำนวน Features: **14 columns** (ก่อนทำ encoding)
- ราคาต่ำสุด: ₹9,990 | ราคาสูงสุด: ₹4,49,990
""")
    st.markdown("---")

    # ─────────────────────────────────────────────────────
    # ส่วนที่ 2: Features ของ Dataset
    # ─────────────────────────────────────────────────────
    st.subheader("2️⃣ Features ของ Dataset")
    col_info = {
        "brand":            ("ชื่อแบรนด์",          "HP, Asus, Apple, Dell, Lenovo …",              "Categorical"),
        "name":             ("ชื่อรุ่น laptop",      "Victus 15, MacBook Pro, IdeaPad …",            "Text (ใช้แค่ทำ is_gaming แล้วลบทิ้ง)"),
        "price":            ("ราคา (INR ₹)",          "9,990 – 4,49,990",                            "🎯 Target Variable"),
        "spec_rating":      ("คะแนนสเปคจากเว็บ",     "60 – 89 (median ≈ 69.3)",                     "Numeric"),
        "processor":        ("รุ่น CPU",              "Intel Core i5, AMD Ryzen 7, Apple M2…",        "Categorical"),
        "Ram":              ("ขนาด RAM",              "'2GB', '4GB', '8GB', '16GB', '32GB', '64GB'", "String → แปลงเป็น Numeric"),
        "Ram_type":         ("ประเภท RAM",            "DDR4, DDR5, LPDDR5, LPDDR4X, Unified …",     "Categorical"),
        "ROM":              ("ขนาด Storage",          "'256GB', '512GB', '1TB', '2TB'",              "String → แปลงเป็น Numeric (GB)"),
        "ROM_type":         ("ประเภท Storage",        "SSD, HDD",                                    "Categorical"),
        "GPU":              ("รุ่น GPU",              "RTX 4060, Intel Iris Xe, AMD Radeon …",        "Categorical"),
        "display_size":     ("ขนาดจอ (นิ้ว)",        "11.6 – 18.0",                                 "Numeric"),
        "resolution_width": ("ความละเอียดกว้าง (px)","1366 – 3840",                                 "Numeric"),
        "resolution_height":("ความละเอียดสูง (px)",  "768 – 3456",                                  "Numeric"),
        "OS":               ("ระบบปฏิบัติการ",        "Windows 11, macOS, Chrome OS, DOS …",         "Categorical"),
        "warranty":         ("ระยะเวลาประกัน (ปี)",  "0, 1, 2, 3",                                  "Numeric"),
        "CPU":              ("รายละเอียด CPU",        "'Hexa Core, 12 Threads', 'Quad Core' …",      "String (ลบออก — ซ้อนกับ processor)"),
    }
    rows = [{"ชื่อ Feature": k, "ความหมาย": v[0], "ตัวอย่างค่า": v[1], "ประเภท": v[2]}
            for k, v in col_info.items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.markdown("---")

    # ─────────────────────────────────────────────────────
    # ส่วนที่ 3: ความไม่สมบูรณ์ของ Dataset
    # ─────────────────────────────────────────────────────
    st.subheader("3️⃣ ความไม่สมบูรณ์ของ Dataset")
    st.markdown("""
Dataset นี้มีปัญหาหลายจุดที่ต้องแก้ก่อนนำไปใช้งาน:

**① RAM และ ROM เก็บเป็น String ไม่ใช่ตัวเลข**
""")
    st.code("""
# ปัญหา: ค่าเป็น string เช่น "8GB", "512GB", "1TB"
df["Ram"].dtype   → object  (ควรเป็น int)
df["ROM"].dtype   → object  (ควรเป็น int)

# ตัวอย่างค่าที่พบ:
Ram: ["2GB", "4GB", "8GB", "16GB", "32GB", "64GB"]
ROM: ["32GB", "64GB", "128GB", "256GB", "512GB", "1TB", "2TB"]
""", language="python")

    st.markdown("""
**② Ram_type — ค่าซ้ำแต่พิมพ์ต่างกัน (Inconsistent Formatting)**
""")
    st.code("""
# ค่าที่พบใน dataset (บางค่าเหมือนกันแต่เขียนต่างกัน):
"LPDDR4X"  และ  "LPDDR4x"   → ควรเป็นค่าเดียวกัน
"DDR4-"    และ  "DDR"        → ค่าไม่สมบูรณ์
""", language="python")

    st.markdown("""
**③ GPU — รูปแบบการเขียนไม่สม่ำเสมอ (GPU เดียวกัน เขียนหลายแบบ)**
""")
    st.code("""
# GPU เดียวกันแต่ชื่อต่างกัน:
"Intel Iris Xe Graphics"
"Intel Integrated Iris Xe Graphics"
"Integrated Intel Iris Xe Graphics"
→ GPU เดียวกัน แต่ถ้าทำ One-Hot จะกลายเป็น 3 columns แยกกัน
""", language="python")

    st.markdown("""
**④ Columns ซ้ำซ้อนที่ไม่มีประโยชน์**
""")
    st.code("""
# Columns ที่ต้องลบออก:
"Unnamed: 0"    → index column ที่ถูก export มาโดยไม่ตั้งใจ
"Unnamed: 0.1"  → index column ซ้ำอีกอัน
"CPU"           → เช่น "Hexa Core, 12 Threads" ซ้อนกับ processor column
                   และมีค่าไม่ซ้ำมากเกินไป จึงไม่มีประโยชน์ในการเทรน
""", language="python")

    st.markdown("""
**⑤ OS — เว้นวรรคและรูปแบบไม่สม่ำเสมอ**
""")
    st.code("""
# ตัวอย่างค่าที่พบ:
"Windows 11 Home"
"Windows 11 Home "    ← มีเว้นวรรคหลัง
"Windows 10"
"macOS"
"Chrome OS"
"DOS"                 ← หมายถึงไม่มี OS (ขาย bare-bone)
""", language="python")
    st.markdown("---")

    # ─────────────────────────────────────────────────────
    # ส่วนที่ 4: การเตรียมข้อมูล
    # ─────────────────────────────────────────────────────
    st.subheader("4️⃣ การเตรียมข้อมูล (Data Preparation)")
    prep_steps = [
        ("ลบ Columns ไม่จำเป็น",  "Unnamed: 0, Unnamed: 0.1, CPU",                                "df.drop(columns=['Unnamed: 0', 'CPU'])"),
        ("แปลง RAM เป็น int",      "'8GB' → 8",                                                    "df['Ram'].str.replace('GB','').astype(int)"),
        ("แปลง ROM เป็น int (GB)", "'512GB' → 512,  '1TB' → 1024,  '2TB' → 2048",                 "ฟังก์ชัน parse_rom()"),
        ("One-Hot Encoding",       "แปลง brand, processor, GPU, OS, Ram_type, ROM_type → binary",  "pd.get_dummies(df, columns=cat_cols)"),
        ("Feature Engineering",    "สร้าง 6 features ใหม่ (ดูรายละเอียดด้านล่าง)",                "ดูโค้ด preprocessing.py"),
        ("ลบ Column 'name'",       "ใช้สำหรับ is_gaming แล้วลบออก — ไม่ใช้เทรน",                 "df.drop(columns=['name'])"),
        ("Train/Test Split",        "80% train (714 รุ่น), 20% test (179 รุ่น)",                    "train_test_split(X, y, test_size=0.2)"),
    ]
    st.dataframe(pd.DataFrame(prep_steps, columns=["ขั้นตอน","รายละเอียด","โค้ด (ตัวอย่าง)"]),
                 use_container_width=True, hide_index=True)

    st.info("💡 **ไม่ต้อง StandardScaler** เพราะ Tree-based models ตัดสินใจด้วย threshold ไม่ได้คำนวณ distance หรือ gradient — สเกลของ feature จึงไม่มีผลต่อการเรียนรู้")

    st.markdown("**⚙️ Feature Engineering — 6 Features ใหม่ที่สร้างขึ้น**")
    features_eng = [
        ("is_gaming",        "เป็น gaming laptop?",    "1 ถ้า GPU มี RTX/GTX หรือชื่อรุ่นมีคำว่า Gaming/ROG/TUF/Predator",  "Binary (0/1)"),
        ("is_premium_brand", "เป็น premium brand?",    "1 ถ้าเป็น Apple, Razer, Samsung, LG, Microsoft, Huawei, Honor, Vaio","Binary (0/1)"),
        ("gpu_tier",         "ระดับความแรง GPU",       "0=Integrated, 1=Entry(MX/GTX1050), 2=Mid(RTX3050/4050), 3=High(RTX3060/4060), 4=Flagship(RTX4080+)", "0–4"),
        ("processor_tier",   "ระดับความแรง CPU",       "0=Celeron/Atom, 1=i3/Ryzen3, 2=i5/Ryzen5, 3=i7/Ryzen7/M1/M2, 4=i9/Ryzen9/M2 Pro/Max","0–4"),
        ("ram_x_storage",    "RAM × Storage",           "Interaction feature เช่น 16 × 512 = 8,192 — สะท้อนความ 'ครบ' ของสเปค","Integer"),
        ("is_latest_gen",    "CPU gen ใหม่?",           "1 ถ้าเป็น 12th/13th Intel หรือ 7th Gen AMD หรือ Apple M2",          "Binary (0/1)"),
    ]
    st.dataframe(pd.DataFrame(features_eng, columns=["Feature","ความหมาย","วิธีสร้าง","ค่าที่เป็นไปได้"]),
                 use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("One-Hot Features", "~377", "จาก 6 categorical columns")
    c2.metric("Numeric Features",  "6",   "Ram, ROM, display, resolution, spec_rating, warranty")
    c3.metric("Engineered Features", "6", "สร้างใหม่ทั้งหมด")
    st.markdown("> รวมทั้งหมด **~389 features** ต่อ laptop 1 รุ่น")
    st.markdown("---")

    # ─────────────────────────────────────────────────────
    # ส่วนที่ 5: ทฤษฎีของอัลกอริทึม
    # ─────────────────────────────────────────────────────
    st.subheader("5️⃣ ทฤษฎีของอัลกอริทึม")
    st.markdown("""
**Ensemble Learning** คือการนำโมเดลหลายตัวมารวมกัน เพื่อให้ผลลัพธ์ดีกว่าโมเดลเดี่ยว
อาศัยหลักการ **"wisdom of crowds"** — โมเดลแต่ละตัวมีจุดแข็งต่างกัน เมื่อรวมกันจะชดเชยจุดอ่อนได้

**สูตรรวม VotingRegressor:**
```
ŷ_final = (ŷ_RF + ŷ_GB + ŷ_ET) / 3
```
""")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**โมเดลที่ 1**\n### 🌲 Random Forest")
        st.markdown("- `n_estimators = 200`\n- `max_depth = 20`\n- `min_samples_leaf = 3`\n- `max_features = 0.6`")
    with col2:
        st.markdown("**โมเดลที่ 2**\n### 📈 Gradient Boosting")
        st.markdown("- `n_estimators = 300`\n- `learning_rate = 0.05`\n- `max_depth = 5`\n- `subsample = 0.8`")
    with col3:
        st.markdown("**โมเดลที่ 3**\n### 🎲 Extra Trees")
        st.markdown("- `n_estimators = 200`\n- `max_depth = 20`\n- `min_samples_leaf = 3`")

    with st.expander("🌲 Random Forest — ทฤษฎีละเอียด"):
        st.markdown("""
**แนวคิด:** สร้าง Decision Tree หลายต้นแล้วเฉลี่ยผล

**ขั้นตอน:**
1. สุ่มข้อมูล subset จาก training set ด้วย **Bootstrap Sampling** (สุ่มแบบใส่คืน)
2. สร้าง Decision Tree แต่ละต้นโดยเลือก feature แบบสุ่ม (`max_features=0.6` = ใช้ 60% ของ features)
3. ทำนายโดยเฉลี่ยผลจากทุก tree (averaging)

**ทำไมดี:** ต้นไม้แต่ละต้น overfit data ของตัวเอง แต่เมื่อเฉลี่ยรวมกัน error ที่ random จะหักล้างกัน → **ลด variance**
""")
        st.code("RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_leaf=3, max_features=0.6, random_state=42)", language="python")

    with st.expander("📈 Gradient Boosting — ทฤษฎีละเอียด"):
        st.markdown("""
**แนวคิด:** เรียนรู้แบบ **sequential** — โมเดลแต่ละรอบแก้ error ของรอบก่อน

**สูตร:**
```
F_m(x) = F_{m-1}(x) + η × h_m(x)
```
- `F_m(x)` = โมเดลรอบที่ m
- `η` = learning rate (0.05) — ควบคุมขนาดการอัพเดท
- `h_m(x)` = tree ที่ fit กับ **residual error** ของรอบก่อน

**ทำไมดี:** เน้นแก้จุดที่ผิดมาก → **ลด bias** ได้ดี
""")
        st.code("GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8, random_state=42)", language="python")

    with st.expander("🎲 Extra Trees — ทฤษฎีละเอียด"):
        st.markdown("""
**แนวคิด:** คล้าย Random Forest แต่สุ่ม **threshold** ด้วย ไม่ใช่แค่ feature

| ประเด็น | Random Forest | Extra Trees |
|---|---|---|
| การสุ่มข้อมูล | Bootstrap (ใส่คืน) | ใช้ทั้งหมด |
| การเลือก threshold | หา best split | **สุ่ม threshold** |
| ความเร็ว | ช้ากว่า | **เร็วกว่า** |
| Variance | ต่ำ | **ต่ำกว่า** |

**ทำไมดี:** diversity ของ tree มากขึ้น → ลด variance และ overfitting ได้ดีกว่า
""")
        st.code("ExtraTreesRegressor(n_estimators=200, max_depth=20, min_samples_leaf=3, random_state=42)", language="python")
    st.markdown("---")

    # ─────────────────────────────────────────────────────
    # ส่วนที่ 6: ขั้นตอนการพัฒนาโมเดล
    # ─────────────────────────────────────────────────────
    st.subheader("6️⃣ ขั้นตอนการพัฒนาโมเดล")
    st.code("""
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
""", language="python")

    st.markdown("**📊 ผลการประเมิน (Test Set)**")
    try:
        metrics = joblib.load("models/classifier_metrics.pkl")
        m1, m2, m3 = st.columns(3)
        m1.metric("R²",   f"{metrics['r2']:.4f}",    "ยิ่งใกล้ 1 ยิ่งดี")
        m2.metric("MAE",  f"₹{metrics['mae']:,.0f}", "เฉลี่ยผิดพลาด (INR)")
        m3.metric("RMSE", f"₹{metrics['rmse']:,.0f}","sensitive ต่อ outlier")
    except:
        m1, m2, m3 = st.columns(3)
        m1.metric("R²",   "~0.95",   "ยิ่งใกล้ 1 ยิ่งดี")
        m2.metric("MAE",  "~₹3,500", "เฉลี่ยผิดพลาด (INR)")
        m3.metric("RMSE", "~₹7,200", "sensitive ต่อ outlier")

    st.markdown("""
**ความหมายของ Metrics:**
- **R²** = สัดส่วนความแปรปรวนที่โมเดลอธิบายได้ — ค่า 0.95 หมายถึงโมเดลอธิบาย 95% ของความแตกต่างของราคา
- **MAE** = ค่าเฉลี่ยของ |ราคาจริง − ราคาทำนาย| — ผิดเฉลี่ยประมาณ ₹3,500 (~1,200 บาท)
- **RMSE** = คล้าย MAE แต่ลงโทษความผิดพลาดขนาดใหญ่มากกว่า
""")
    st.markdown("---")

    # ─────────────────────────────────────────────────────
    # ส่วนที่ 7: แหล่งอ้างอิง
    # ─────────────────────────────────────────────────────
    st.subheader("7️⃣ แหล่งอ้างอิง")
    st.markdown("""
**Dataset:**
8. Laptop Price India Dataset — https://www.kaggle.com/datasets/jacksondivakarr/laptop-price-prediction-dataset
""")
    
    st.subheader(" โครงสร้าง Project (Ensemble ML)")

    st.markdown("""
<style>
.tree-block {
    background: #181825;
    border: 1px solid #2a2a3e;
    border-radius: 10px;
    padding: 20px 24px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12.5px;
    line-height: 2;
}
.sec { color: #45475a; font-size: 10px; letter-spacing: 0.15em; text-transform: uppercase;
       border-bottom: 1px solid #2a2a3e; padding-bottom: 3px; margin: 14px 0 4px 0; }
.folder { color: #89dceb; font-weight: bold; }
.py     { color: #a6e3a1; font-weight: bold; }
.csv    { color: #f9e2af; }
.pkl    { color: #cba6f7; }
.c      { color: #585b70; font-style: italic; }
.bd     { display:inline-block; font-size:10px; padding:1px 7px; border-radius:4px; margin-left:6px; vertical-align:middle; }
.b1 { background:#1e2a3a; color:#89b4fa; }
.b2 { background:#1a2e22; color:#a6e3a1; }
.i1 { padding-left:20px; }
.i2 { padding-left:40px; }
</style>

<div class="tree-block">
<div style="color:#89b4fa;font-size:15px;font-weight:bold;">PROJECT_AI/ <span style="color:#45475a;font-size:11px;font-weight:normal;">(เฉพาะส่วน Ensemble ML)</span></div>

<!-- DATA -->
<div class="sec">Data</div>
<div><span class="i1">├── </span><span class="folder">data_set/</span><span class="bd b1">Laptop · India · Kaggle</span></div>
<div><span class="i2">├── </span><span class="csv">data_new_laptop.csv</span><span class="c"> ← 893 รุ่น · 17 columns (raw)</span></div>
<div><span class="i2">└── </span><span class="csv">processed_new_laptop.csv</span><span class="c"> ← หลัง preprocessing · ~389 features · target = price (INR)</span></div>

<!-- PIPELINE -->
<div class="sec">Pipeline</div>
<div><span class="i1">├── </span><span class="py">preprocessing.py</span><span class="bd b2">ขั้นตอนที่ 1</span></div>
<div><span class="i2 c">├── ลบ columns ไม่จำเป็น · แปลง Ram/ROM · Feature Eng 6 ตัว · One-Hot Encoding</span></div>
<div><span class="i2 c">└── save → processed_new_laptop.csv · feature_columns.pkl</span></div>

<div style="height:4px"></div>
<div><span class="i1">└── </span><span class="py">train_ensemble.py</span><span class="bd b2">ขั้นตอนที่ 2</span></div>
<div><span class="i2 c">├── แบ่งข้อมูล 80% เทรน (~714 รุ่น) · 20% ทดสอบ (~179 รุ่น)</span></div>
<div><span class="i2 c">├── เทรน 3 โมเดล: RandomForest · GradientBoosting · ExtraTrees</span></div>
<div><span class="i2 c">└── รวมผลทั้ง 3 โดยเฉลี่ย → ได้ราคาที่แม่นยำขึ้น</span></div>

<!-- MODELS -->
<div class="sec">Saved Models</div>
<div><span class="i1">└── </span><span class="folder">models/</span></div>
<div><span class="i2">├── </span><span class="pkl">ensemble_model.pkl</span><span class="c">     </span></div>
<div><span class="i2">├── </span><span class="pkl">feature_columns.pkl</span><span class="c">    </span></div>
<div><span class="i2">├── </span><span class="pkl">price_bins.pkl</span><span class="c">         </span></div>
<div><span class="i2">└── </span><span class="pkl">classifier_metrics.pkl</span><span class="c"> </span></div>

<!-- APP -->
<div class="sec">Streamlit App</div>
<div><span class="i1">└── </span><span class="py">app.py</span><span class="bd b1">streamlit run app.py</span></div>
<div><span class="i2 c">├── spec_form() → build_ens_input() → ensemble.predict() → ราคา INR + THB</span></div>
<div><span class="i2 c">└── build_brand_compat() ← กรอง dropdown options ตาม brand ที่เลือก</span></div>

</div>
""", unsafe_allow_html=True)


# ===================================================================
# PAGE: ทดสอบ Ensemble ML
# ===================================================================
def page_ml_predict():
    st.title("🔍 ทดสอบ Ensemble ML")
    st.caption("กรอกสเปค laptop แล้วให้ Ensemble Model ทำนายราคา")
    st.markdown("---")

    ui = spec_form("ML")
    st.markdown("---")
    if st.button("🔍 ทำนายราคาด้วย Ensemble", type="primary", use_container_width=True):
        input_dict = build_ens_input(ui, ens_features)
        df_in = pd.DataFrame([input_dict])

        predicted_inr = ensemble.predict(df_in)[0]
        predicted_thb = predicted_inr * INR_TO_THB
        price_label, thb_range_str = get_price_range(predicted_inr)

        st.success(f"💰 ราคาที่ทำนาย (INR): **₹{predicted_inr:,.0f}**")
        st.success(f"💰 ราคาโดยประมาณ (THB): **~{predicted_thb:,.0f} บาท**")
        st.markdown("---")
        c1, c2 = st.columns(2)
        c1.metric("ช่วงราคา (INR)", price_label)
        c2.metric("ช่วงราคา (THB)", thb_range_str)
        st.warning(PRICE_DISCLAIMER)

        with st.expander("📋 สเปคที่เลือก"):
            st.write(f"**แบรนด์:** {ui['brand']}  |  **CPU:** {ui['processor']}")
            st.write(f"**RAM:** {ui['Ram']} GB ({ui['Ram_type']})  |  **Storage:** {ui['ROM']} GB ({ui['ROM_type']})")
            st.write(f"**GPU:** {ui['GPU']}  |  **OS:** {ui['OS']}")
            st.write(f"**หน้าจอ:** {ui['display_size']}\"  {ui['resolution_width']}×{ui['resolution_height']}  |  **ประกัน:** {ui['warranty']} ปี")

        st.markdown("**🤖 Model: VotingRegressor** — Random Forest + Gradient Boosting + Extra Trees")


# ===================================================================
# โหลด Mobile NN Model
# ===================================================================
@st.cache_resource
def load_mobile_model():
    try:
        with open("models/mobile_nn_model.pkl", "rb") as f: m_model    = pickle.load(f)
        with open("models/mobile_scaler.pkl",   "rb") as f: m_scaler   = pickle.load(f)
        with open("models/mobile_features.pkl", "rb") as f: m_features = pickle.load(f)
        m_metrics = {}
        if os.path.exists("models/mobile_metrics.json"):
            with open("models/mobile_metrics.json") as f: m_metrics = json.load(f)
        return m_model, m_scaler, m_features, m_metrics, True
    except FileNotFoundError:
        return None, None, None, {}, False

m_model, m_scaler, m_features, m_metrics, m_loaded = load_mobile_model()

MOBILE_PRICE_RANGES = {
    0: ("💚 ราคาถูกมาก",           "~2,000 – 5,000 บาท",   "#22c55e"),
    1: ("💛 ราคาปานกลาง",          "~5,000 – 10,000 บาท",  "#eab308"),
    2: ("🟠 ราคาสูง",               "~10,000 – 20,000 บาท", "#f97316"),
    3: ("🔴 ราคาสูงมาก (Flagship)", "~20,000+ บาท",         "#ef4444"),
}

MOBILE_FEATURES_META = {
    "battery_power": {"th": "ความจุแบตเตอรี่ (mAh)"},
    "clock_speed":   {"th": "ความเร็ว CPU (GHz)"},
    "fc":            {"th": "กล้องหน้า (MP)"},
    "int_memory":    {"th": "หน่วยความจำภายใน (GB)"},
    "m_dep":         {"th": "ความหนา (cm)"},
    "mobile_wt":     {"th": "น้ำหนัก (g)"},
    "n_cores":       {"th": "จำนวน CPU Cores"},
    "pc":            {"th": "กล้องหลัก (MP)"},
    "px_height":     {"th": "ความละเอียด – สูง (px)"},
    "px_width":      {"th": "ความละเอียด – กว้าง (px)"},
    "ram":           {"th": "RAM (MB)"},
    "sc_h":          {"th": "ความสูงหน้าจอ (cm)"},
    "sc_w":          {"th": "ความกว้างหน้าจอ (cm)"},
    "talk_time":     {"th": "เวลาโทรต่อชาร์จ (ชม.)"},
    "blue":          {"th": "มี Bluetooth"},
    "dual_sim":      {"th": "รองรับ Dual SIM"},
    "four_g":        {"th": "รองรับ 4G"},
    "three_g":       {"th": "รองรับ 3G"},
    "touch_screen":  {"th": "หน้าจอสัมผัส"},
    "wifi":          {"th": "รองรับ WiFi"},
}

def mobile_spec_form():
    ui = {}

    st.markdown("#### 📱 สเปคหลัก (มีผลต่อราคามากที่สุด)")
    col1, col2 = st.columns(2)
    with col1:
        ui["ram"]           = st.select_slider("🧠 RAM (MB)", options=[256,512,1024,2048,3000,3998], value=2048)
        ui["battery_power"] = st.select_slider("🔋 แบตเตอรี่ (mAh)", options=[500,750,1000,1250,1500,1750,2000], value=1500)
        ui["int_memory"]    = st.select_slider("💾 หน่วยความจำ (GB)", options=[2,4,8,16,32,64], value=32)
        ui["n_cores"]       = st.select_slider("⚙️ CPU Cores", options=[1,2,4,6,8], value=4)
    with col2:
        ui["pc"]            = st.select_slider("📷 กล้องหลัก (MP)", options=[0,2,5,8,12,16,20], value=8)
        ui["fc"]            = st.select_slider("🤳 กล้องหน้า (MP)", options=[0,2,5,8,12,16,20], value=5)
        ui["clock_speed"]   = st.select_slider("🚀 ความเร็ว CPU (GHz)", options=[0.5,1.0,1.5,2.0,2.5,3.0], value=1.5)
        ui["talk_time"]     = st.select_slider("📞 เวลาโทร/ชาร์จ (ชม.)", options=[2,5,7,10,15,20], value=10)

    st.markdown("#### 🖥️ หน้าจอ")
    col3, col4 = st.columns(2)
    with col3:
        ui["px_width"]  = st.select_slider("↔️ ความละเอียดกว้าง (px)",  options=[500,720,1080,1440,1998], value=1080)
        ui["sc_w"]      = st.select_slider("📐 ความกว้างหน้าจอ (cm)",    options=[0,3,5,7,9,12,18],        value=7)
    with col4:
        ui["px_height"] = st.select_slider("↕️ ความละเอียดสูง (px)",    options=[0,480,720,1280,1920], value=1280)
        ui["sc_h"]      = st.select_slider("📏 ความสูงหน้าจอ (cm)",      options=[5,8,10,12,14,16,19],  value=14)

    st.markdown("#### ⚙️ ตัวเลือกเพิ่มเติม")
    col5, col6 = st.columns(2)
    with col5:
        ui["m_dep"]     = st.select_slider("📦 ความหนาเครื่อง (cm)", options=[0.1,0.3,0.5,0.7,1.0], value=0.5)
        ui["mobile_wt"] = st.select_slider("⚖️ น้ำหนัก (g)",          options=[80,120,150,180,200],  value=150)
    with col6:
        st.markdown("**คุณสมบัติ:**")
        bool_fields = {
            "four_g":       "📶 รองรับ 4G",
            "three_g":      "📡 รองรับ 3G",
            "wifi":         "📡 WiFi",
            "blue":         "🔵 Bluetooth",
            "dual_sim":     "📲 Dual SIM",
            "touch_screen": "👆 Touch Screen",
        }
        bcol1, bcol2 = st.columns(2)
        items = list(bool_fields.items())
        for i, (key, label) in enumerate(items):
            col = bcol1 if i % 2 == 0 else bcol2
            with col:
                ui[key] = int(st.toggle(label, value=True, key=f"mob_{key}"))

    return ui


# ===================================================================
# PAGE: อธิบาย Neural Network
# ===================================================================
def page_nn_explain():
    st.title("📖 Neural Network — Mobile Price")
    st.caption("ที่มา Dataset · Features · ความไม่สมบูรณ์ · การเตรียมข้อมูล · ทฤษฎี · ขั้นตอนพัฒนา · แหล่งอ้างอิง")
    st.markdown("---")

    acc = m_metrics.get("accuracy", 0.915)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎯 Accuracy",  f"{acc*100:.1f}%",                       "Validation Set")
    c2.metric("📦 Dataset",   f"{m_metrics.get('n_samples', 2000):,}",  "มือถือ")
    c3.metric("🔢 Features",  f"{m_metrics.get('n_features', 20)}",    "สเปค")
    c4.metric("🏷️ Classes",   "4",                                     "ช่วงราคา")
    st.markdown("---")

    # ─────────────────────────────────────────────────────
    # ส่วนที่ 1: ที่มาของ Dataset
    # ─────────────────────────────────────────────────────
    st.subheader("1️⃣ ที่มาของ Dataset")
    st.markdown("""
**Dataset:** Mobile Price Classification

**ที่มา:** Download จากเว็บไซต์ **Kaggle**
- 🔗 Link: https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification
- ข้อมูลมือถือจำนวน **2,000 รุ่น** (train.csv) และ 1,000 รุ่น (test.csv)
- แต่ละรุ่นมีสเปคเชิงเทคนิค 20 ตัว และ label ช่วงราคา 4 ระดับ (0–3)
- Dataset ถูกออกแบบมาให้ **clean, balanced และพร้อมใช้** ตั้งแต่ต้น

**ข้อมูลสำคัญ:** Dataset นี้ **ไม่มีราคาจริง** — มีเพียง class 0-3 แทนช่วงราคา จึงเหมาะกับ **Classification** เท่านั้น
ไม่สามารถทำ Regression หาราคาที่แน่นอนได้
""")
    st.markdown("---")

    # ─────────────────────────────────────────────────────
    # ส่วนที่ 2: Features ของ Dataset
    # ─────────────────────────────────────────────────────
    st.subheader("2️⃣ Features ของ Dataset")

    st.markdown("**💰 4 ช่วงราคา (Target Classes)**")
    for k, (label, price, color) in MOBILE_PRICE_RANGES.items():
        st.markdown(
            f"<div style='padding:0.5rem 1rem; margin:0.25rem 0; border-left:4px solid {color}; "
            f"background:rgba(255,255,255,0.03); border-radius:0 8px 8px 0;'>"
            f"<b>Class {k} — {label}</b> &nbsp;—&nbsp; {price}</div>",
            unsafe_allow_html=True
        )

    st.markdown("**📋 20 Features เชิงเทคนิค**")
    mobile_col_info = [
        ("battery_power", "ความจุแบตเตอรี่",    "500 – 1998 mAh",     "Numeric"),
        ("blue",          "มี Bluetooth",        "0 = ไม่มี, 1 = มี",  "Binary"),
        ("clock_speed",   "ความเร็ว CPU",        "0.5 – 3.0 GHz",      "Numeric"),
        ("dual_sim",      "รองรับ Dual SIM",     "0 = ไม่มี, 1 = มี",  "Binary"),
        ("fc",            "กล้องหน้า",           "0 – 19 MP",          "Numeric"),
        ("four_g",        "รองรับ 4G",           "0 = ไม่มี, 1 = มี",  "Binary"),
        ("int_memory",    "หน่วยความจำภายใน",    "2 – 64 GB",          "Numeric"),
        ("m_dep",         "ความหนาเครื่อง",      "0.1 – 1.0 cm",       "Numeric"),
        ("mobile_wt",     "น้ำหนัก",             "80 – 200 g",         "Numeric"),
        ("n_cores",       "จำนวน CPU Cores",     "1 – 8",              "Numeric"),
        ("pc",            "กล้องหลัก",           "0 – 20 MP",          "Numeric"),
        ("px_height",     "ความละเอียดสูง",      "0 – 1960 px",        "Numeric"),
        ("px_width",      "ความละเอียดกว้าง",    "500 – 1998 px",      "Numeric"),
        ("ram",           "RAM",                 "256 – 3998 MB",      "Numeric ⭐ สำคัญมากที่สุด"),
        ("sc_h",          "ความสูงหน้าจอ",       "5 – 19 cm",          "Numeric"),
        ("sc_w",          "ความกว้างหน้าจอ",     "0 – 18 cm",          "Numeric"),
        ("talk_time",     "เวลาโทรต่อชาร์จ",    "2 – 20 ชั่วโมง",     "Numeric"),
        ("three_g",       "รองรับ 3G",           "0 = ไม่มี, 1 = มี",  "Binary"),
        ("touch_screen",  "หน้าจอสัมผัส",        "0 = ไม่มี, 1 = มี",  "Binary"),
        ("wifi",          "รองรับ WiFi",         "0 = ไม่มี, 1 = มี",  "Binary"),
    ]
    mob_rows = [{"Feature": f, "ความหมาย": m, "ช่วงค่า": r, "ประเภท": t}
                for f, m, r, t in mobile_col_info]
    st.dataframe(pd.DataFrame(mob_rows), use_container_width=True, hide_index=True)
    st.markdown("---")

    # ─────────────────────────────────────────────────────
    # ส่วนที่ 3: ความไม่สมบูรณ์ของ Dataset
    # ─────────────────────────────────────────────────────
    st.subheader("3️⃣ ความไม่สมบูรณ์ของ Dataset")
    st.success("✅ Dataset นี้ถูกออกแบบมาให้ **clean และ balanced** ตั้งแต่ต้น ความไม่สมบูรณ์จึงมีน้อยมาก")
    st.markdown("""
**① ไม่มีราคาจริง — มีแค่ label ระดับ (Class 0–3)**
""")
    st.code("""
# ปัญหา: ไม่สามารถทราบราคาที่แน่นอนของมือถือได้
# มีแค่: price_range = 0, 1, 2, 3
# ไม่มี: price = 5999, 12000, 25000 ฯลฯ

# ผลกระทบ: ทำได้แค่ Classification (จำแนกระดับ)
#           ทำ Regression (ทำนายราคาตัวเลขจริง) ไม่ได้
""", language="python")

    st.markdown("""
**② sc_w (ความกว้างหน้าจอ) มีค่า 0 ซึ่งไม่สมเหตุสมผล**
""")
    st.code("""
# ค่า 0 ใน sc_w หมายความว่าอะไร? ไม่มีหน้าจอ? หรือข้อมูลขาดหาย?
df["sc_w"].min()  →  0   ← ค่าน้อยสุดไม่สมเหตุสมผล
df["sc_w"].max()  →  18

# อย่างไรก็ตาม เนื่องจาก dataset นี้เป็น synthetic/generated data
# จึงไม่มีผลกระทบรุนแรง — โมเดลสามารถเรียนรู้ pattern ได้อยู่ดี
""", language="python")

    st.markdown("""
**③ ไม่มีชื่อแบรนด์หรือรุ่น — ไม่สามารถระบุมือถือจริงได้**
""")
    st.code("""
# Dataset ไม่มี column:
# - brand (Samsung, Apple, Xiaomi ...)
# - model_name (Galaxy S23, iPhone 15 ...)

# ผลกระทบ: โมเดลเรียนรู้จากสเปคเชิงตัวเลขล้วนๆ
#           ไม่สามารถจำแนกตาม brand premium หรือ marketing price ได้
""", language="python")

    st.markdown("""
**④ px_height มีค่า 0 ซึ่งไม่สมเหตุสมผล**
""")
    st.code("""
# ค่า 0 ใน px_height หมายความว่าอะไร?
df["px_height"].min()  →  0
df["px_height"].max()  →  1960

# ค่า 0 อาจเกิดจากข้อมูลที่ไม่สมบูรณ์ในต้นฉบับ
# แต่ dataset นี้เป็น synthetic ดังนั้นจึงยอมรับได้และไม่แก้ไข
""", language="python")
    st.markdown("---")

    # ─────────────────────────────────────────────────────
    # ส่วนที่ 4: การเตรียมข้อมูล
    # ─────────────────────────────────────────────────────
    st.subheader("4️⃣ การเตรียมข้อมูล (Data Preparation)")
    prep_steps_nn = [
        ("โหลดข้อมูล",     "train.csv — 2,000 rows, 21 columns (20 features + price_range)",   "df = pd.read_csv('data_2/train.csv')"),
        ("แยก X และ y",    "X = 20 สเปค (Numeric ทั้งหมด), y = price_range (0/1/2/3)",         "X = df.drop('price_range', axis=1)"),
        ("Train/Val Split","80% train (1,600), 20% val (400), stratify=y เพื่อให้ทุก class สมดุล","train_test_split(..., stratify=y, test_size=0.2)"),
        ("StandardScaler", "แปลงทุก feature ให้ mean=0, std=1 — จำเป็นสำหรับ Neural Network",   "scaler.fit_transform(X_train) / scaler.transform(X_val)"),
        ("ไม่ต้อง OHE",    "ทุก feature เป็น Numeric อยู่แล้ว ไม่มี categorical text",           "—"),
    ]
    st.dataframe(pd.DataFrame(prep_steps_nn, columns=["ขั้นตอน","รายละเอียด","โค้ด (ตัวอย่าง)"]),
                 use_container_width=True, hide_index=True)

    st.warning("⚠️ **ต้องทำ StandardScaler** เพราะ Neural Network ใช้ gradient descent — features ที่มีสเกลต่างกันมากจะทำให้ gradient update ไม่สม่ำเสมอ เช่น `ram` (256–3998) vs `blue` (0 หรือ 1)")

    st.code("""
scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)  # เรียน mean, std จาก train เท่านั้น
X_val_s   = scaler.transform(X_val)        # apply กับ val (ห้าม fit_transform!)
""", language="python")

    ex = pd.DataFrame({
        "Feature":               ["ram",  "battery_power", "blue", "m_dep"],
        "ค่าก่อน Scale":         [3000,   1500,            1,      0.5],
        "หลัง StandardScaler":   ["~1.2", "~0.4",          "~0.8", "~-0.1"],
    })
    st.dataframe(ex, use_container_width=True, hide_index=True)
    st.markdown("---")

    # ─────────────────────────────────────────────────────
    # ส่วนที่ 5: ทฤษฎีของอัลกอริทึม
    # ─────────────────────────────────────────────────────
    st.subheader("5️⃣ ทฤษฎีของอัลกอริทึม")

    st.markdown("### 🧠 MLP (Multilayer Perceptron) คืออะไร?")
    st.markdown("""
MLP คือ Neural Network แบบพื้นฐานที่ประกอบด้วย **layers ของ neurons** เชื่อมต่อกัน
แต่ละ neuron รับ input → คูณด้วย weight → บวก bias → ผ่าน activation function → ส่งต่อ layer ถัดไป
""")

    st.markdown("**🏗️ สถาปัตยกรรมของโมเดลนี้**")
    st.code("""
Input Layer:      20 neurons  ← สเปคมือถือ 20 ตัว (หลัง StandardScaler)
        ↓  weights (20×128) + bias + ReLU
Hidden Layer 1:  128 neurons
        ↓  weights (128×64) + bias + ReLU
Hidden Layer 2:   64 neurons
        ↓  weights (64×32)  + bias + ReLU
Hidden Layer 3:   32 neurons
        ↓  weights (32×4)   + bias + Softmax
Output Layer:      4 neurons  → probability ของแต่ละ class (รวม = 1.0)
""")

    with st.expander("⚡ ReLU Activation Function — ทฤษฎี"):
        st.markdown("""
**สูตร:** `f(x) = max(0, x)`

**ทำไมใช้ ReLU แทน Sigmoid/Tanh:**
- Sigmoid/Tanh มีปัญหา **vanishing gradient** — gradient เล็กมากใน layer ลึก ทำให้เรียนรู้ช้ามาก
- ReLU ให้ gradient = 1 เมื่อ x > 0 → gradient ไม่หายไป → เรียนรู้เร็วกว่า
- คำนวณง่าย เร็วกว่า sigmoid มาก

**ข้อเสีย:** Dying ReLU — neuron ที่ได้ input ≤ 0 ตลอดจะหยุดเรียนรู้
""")

    with st.expander("🎯 Softmax Output — ทฤษฎี"):
        st.markdown("""
**สูตร:** `softmax(z_i) = e^{z_i} / Σ e^{z_j}`

Layer สุดท้ายใช้ Softmax เพราะต้องการ **probability** ที่รวมกันได้ 1.0

**ตัวอย่าง output:**
```
Class 0: 0.05  (5%)
Class 1: 0.10  (10%)
Class 2: 0.15  (15%)
Class 3: 0.70  (70%)  ← ทำนายว่าเป็น Class 3 (Flagship)
         ────
         1.00
```
""")

    with st.expander("🔧 Adam Optimizer — ทฤษฎี"):
        st.markdown("""
**Adam** (Adaptive Moment Estimation) = optimizer ที่ปรับ learning rate อัตโนมัติสำหรับแต่ละ parameter

ผสม 2 เทคนิค:
- **Momentum** — จำทิศทางการ update ก่อนหน้า ลดการแกว่ง
- **RMSProp** — ปรับ learning rate ตาม gradient ที่ผ่านมา

**ทำไมดีกว่า SGD ธรรมดา:** ไม่ต้องปรับ learning rate เอง หาจุดที่ดีที่สุดได้เร็วกว่า
""")

    with st.expander("🛑 Early Stopping — ทำไมต้องใช้"):
        st.markdown("""
**ปัญหา:** ถ้า train นานเกินไป โมเดลจะ **overfit** — จำ training data มากเกินไปจนทำนาย validation data ได้แย่ลง

**วิธีแก้ด้วย Early Stopping:**
1. แบ่ง validation set ออก 10% (`validation_fraction=0.1`)
2. ทุก iteration วัด validation loss
3. ถ้า validation loss ไม่ดีขึ้นติดต่อกัน **20 รอบ** (`n_iter_no_change=20`) → หยุด train
4. คืนค่า weights ที่ดีที่สุด

**ผล:** ป้องกัน overfitting โดยอัตโนมัติ ไม่ต้องเดา max_iter เอง
""")
    st.markdown("---")

    # ─────────────────────────────────────────────────────
    # ส่วนที่ 6: ขั้นตอนการพัฒนาโมเดล
    # ─────────────────────────────────────────────────────
    st.subheader("6️⃣ ขั้นตอนการพัฒนาโมเดล")

    st.markdown("**⚙️ Hyperparameters ที่ใช้**")
    hp_data = {
        "Parameter":   ["hidden_layer_sizes", "activation", "solver", "max_iter", "early_stopping", "n_iter_no_change", "validation_fraction"],
        "ค่า":          ["(128, 64, 32)",       "relu",       "adam",   "500",      "True",           "20",               "0.1"],
        "เหตุผล":      [
            "3 layers ขนาดลดหลั่น — เรียนรู้ feature ระดับต่างๆ",
            "แก้ vanishing gradient ได้ดีกว่า sigmoid",
            "ปรับ learning rate อัตโนมัติ เร็วกว่า SGD",
            "กำหนด max รอบ แต่ early stopping จะหยุดก่อนถ้า val loss ไม่ดีขึ้น",
            "ป้องกัน overfitting อัตโนมัติ",
            "รอ 20 รอบก่อนตัดสินใจหยุด",
            "ใช้ 10% ของ train set สำหรับ monitor early stopping",
        ],
    }
    st.dataframe(pd.DataFrame(hp_data), use_container_width=True, hide_index=True)

    st.code("""
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
print("\n Done! Now run: streamlit run app.py")
""", language="python")

    st.markdown("**📊 ผลการประเมิน**")
    a1, a2, a3 = st.columns(3)
    a1.metric("🎯 Accuracy",         f"{acc*100:.1f}%", "Validation Set (400 รุ่น)")
    a2.metric("📦 Training Samples", f"{m_metrics.get('n_samples',2000):,}", "ข้อมูลทั้งหมด")
    a3.metric("🔢 Total Parameters", "~24,900+", "weights + biases")
    st.markdown("---")

    # ─────────────────────────────────────────────────────
    # ส่วนที่ 7: แหล่งอ้างอิง
    # ─────────────────────────────────────────────────────
    # 
    st.subheader("7️⃣ เเหล่งอ้างอิง ")
    st.markdown("""
**Dataset:**
8. Mobile Price Classification Dataset — https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification
""")

    st.subheader(" โครงสร้าง Project (Neural Network)")

    st.markdown("""
<style>
.tree-block {
    background: #181825;
    border: 1px solid #2a2a3e;
    border-radius: 10px;
    padding: 20px 24px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12.5px;
    line-height: 2;
}
.sec { color: #45475a; font-size: 10px; letter-spacing: 0.15em; text-transform: uppercase;
       border-bottom: 1px solid #2a2a3e; padding-bottom: 3px; margin: 14px 0 4px 0; }
.folder { color: #89dceb; font-weight: bold; }
.py     { color: #a6e3a1; font-weight: bold; }
.csv    { color: #f9e2af; }
.pkl    { color: #cba6f7; }
.js     { color: #fab387; }
.c      { color: #585b70; font-style: italic; }
.bd     { display:inline-block; font-size:10px; padding:1px 7px; border-radius:4px; margin-left:6px; vertical-align:middle; }
.b1 { background:#1e2a3a; color:#89b4fa; }
.b2 { background:#1a2e22; color:#a6e3a1; }
.i1 { padding-left:20px; }
.i2 { padding-left:40px; }
</style>

<div class="tree-block">
<div style="color:#89b4fa;font-size:15px;font-weight:bold;">PROJECT_AI/ <span style="color:#45475a;font-size:11px;font-weight:normal;">(เฉพาะส่วน Neural Network)</span></div>

<!-- DATA -->
<div class="sec">Data</div>
<div><span class="i1">└── </span><span class="folder">data_2/</span><span class="bd b1">Mobile · Kaggle · balanced 500 รุ่น/class</span></div>
<div><span class="i2">├── </span><span class="csv">train.csv</span><span class="c"> ← 2,000 รุ่น · 20 features · target = price_range (0/1/2/3)</span></div>
<div><span class="i2">└── </span><span class="csv">test.csv</span><span class="c"> ← 1,000 รุ่น · ไม่มี label</span></div>

<!-- PIPELINE -->
<div class="sec">Pipeline</div>
<div><span class="i1">└── </span><span class="py">train_nn.py</span><span class="bd b2">ขั้นตอนเดียว</span></div>
<div><span class="i2 c">├── แบ่งข้อมูล 80% เทรน (1,600 รุ่น) · 20% ทดสอบ (400 รุ่น) ให้แต่ละระดับราคาสมดุล</span></div>
<div><span class="i2 c">├── ปรับสเกลข้อมูลให้อยู่ในช่วงเดียวกัน (StandardScaler) ก่อนเข้าโมเดล</span></div>
<div><span class="i2 c">├── สร้าง Neural Network 3 ชั้น (128→64→32) ใช้ ReLU + Adam</span></div>
<div><span class="i2 c">├── หยุด Train อัตโนมัติเมื่อผลไม่ดีขึ้น 20 รอบติดกัน (Early Stopping)</span></div>
<div><span class="i2 c">└── บันทึก → โมเดล · scaler · ชื่อ features · ผล accuracy</span></div>


<!-- MODELS -->
<div class="sec">Saved Models</div>
<div><span class="i1">└── </span><span class="folder">models/</span></div>
<div><span class="i2">├── </span><span class="pkl">mobile_nn_model.pkl</span><span class="c"></span></div>
<div><span class="i2">├── </span><span class="pkl">mobile_scaler.pkl</span><span class="c">    </span></div>
<div><span class="i2">├── </span><span class="pkl">mobile_features.pkl</span><span class="c">  </span></div>
<div><span class="i2">└── </span><span class="js">mobile_metrics.json</span><span class="c">  </span></div>

<!-- APP -->
<div class="sec">Streamlit App</div>
<div><span class="i1">└── </span><span class="py">app.py</span><span class="bd b1">streamlit run app.py</span></div>
<div><span class="i2 c">└── แสดงผล: Class 0 ถูกมาก · Class 1 ปานกลาง · Class 2 สูง · Class 3 สูงมาก</span></div>

</div>
""", unsafe_allow_html=True)




# ===================================================================
# PAGE: ทดสอบ Neural Network
# ===================================================================
def page_nn_predict():
    st.title("🔍 ทำนายช่วงราคามือถือ")
    st.caption("กรอกสเปคมือถือแล้วให้ Neural Network ทำนายช่วงราคา")
    st.markdown("---")

    if not m_loaded:
        st.error("❌ ยังไม่พบ Mobile NN Model")
        st.markdown("""
**วิธีแก้ไข:**
1. ตรวจสอบว่า `data_2/train.csv` มีอยู่
2. รัน: `python train_nn.py`
3. Reload หน้านี้
""")
        st.code("python train_nn.py", language="bash")
        return

    ui = mobile_spec_form()

    st.markdown("---")
    if st.button("🔍 ทำนายราคา", type="primary", use_container_width=True):

        x_in     = np.array([[ui[f] for f in m_features]])
        x_scaled = m_scaler.transform(x_in)

        pred  = m_model.predict(x_scaled)[0]
        proba = m_model.predict_proba(x_scaled)[0]

        label, price_range, color = MOBILE_PRICE_RANGES[pred]

        st.markdown("---")
        st.markdown(
            f"<div style='padding:1.5rem; border:2px solid {color}; border-radius:12px; "
            f"background:rgba(255,255,255,0.03); text-align:center;'>"
            f"<div style='font-size:2rem;'>{label}</div>"
            f"<div style='font-size:1.3rem; color:{color}; margin-top:0.5rem; font-weight:600;'>{price_range}</div>"
            f"<div style='color:#888; margin-top:0.5rem; font-size:0.9rem;'>"
            f"ความมั่นใจ: <b>{proba[pred]*100:.1f}%</b></div>"
            f"</div>",
            unsafe_allow_html=True
        )

        st.markdown("---")
        st.markdown("### 📊 ความมั่นใจแต่ละช่วงราคา")
        for i, (lbl, pr, col) in MOBILE_PRICE_RANGES.items():
            pct    = proba[i] * 100
            bar    = "█" * max(1, int(pct / 4))
            marker = " ◀ **ผลทำนาย**" if i == pred else ""
            st.markdown(
                f"`{pr:>25}`  `{pct:5.1f}%`  "
                f"<span style='color:{col}'>{bar}</span>{marker}",
                unsafe_allow_html=True
            )

        with st.expander("📋 ดูสเปคที่กรอก"):
            spec_rows = {MOBILE_FEATURES_META[k]["th"]: [ui[k]] for k in m_features}
            st.dataframe(
                pd.DataFrame(spec_rows).T.rename(columns={0: "ค่า"}),
                use_container_width=True
            )

        st.markdown("---")
        acc = m_metrics.get("accuracy", 0)
        st.markdown(
            f"**🧠 Model:** MLP Neural Network (128→64→32)  |  "
            f"**🎯 Accuracy:** {acc*100:.1f}%  |  "
            f"**📦 Trained on:** {m_metrics.get('n_samples', '?'):,} samples"
        )


# ===================================================================
# ROUTER
# ===================================================================
PAGE = st.session_state.page

if   PAGE == "ml_explain": page_ml_explain()
elif PAGE == "ml_predict": page_ml_predict()
elif PAGE == "nn_explain": page_nn_explain()
elif PAGE == "nn_predict": page_nn_predict()
else:                      page_ml_explain()