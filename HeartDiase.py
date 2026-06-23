import streamlit as st
import numpy as np
import pandas as pd
import pickle

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Penyakit Jantung",
    page_icon="🫀",
    layout="wide",
)

# ── Load model, scaler, threshold ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('model_rf.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    try:
        with open('threshold.pkl', 'rb') as f:
            threshold = pickle.load(f)
    except FileNotFoundError:
        threshold = 0.50   # fallback jika file lama tidak ada threshold.pkl
    return model, scaler, threshold

model, scaler, THRESHOLD = load_model()

# ── Risk factor definitions ────────────────────────────────────────────────────
# Tiap entry: (label_display, fungsi evaluasi input → (is_risk: bool, keterangan: str))
def evaluate_risk_factors(inputs):
    age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal = inputs

    factors = []

    # Usia
    if age >= 60:
        factors.append(("⚠️ Usia ≥ 60 tahun", f"{age} tahun — faktor risiko signifikan", True))
    else:
        factors.append(("✅ Usia", f"{age} tahun — dalam rentang aman", False))

    # Tipe nyeri dada (cp)
    cp_labels = {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-anginal Pain", 3: "Asymptomatic"}
    if cp == 0:
        factors.append(("⚠️ Tipe Nyeri Dada", f"Typical Angina — indikator kuat penyakit jantung", True))
    elif cp == 3:
        factors.append(("⚠️ Tipe Nyeri Dada", f"Asymptomatic — seringkali tanda penyakit lanjut", True))
    else:
        factors.append(("✅ Tipe Nyeri Dada", f"{cp_labels[cp]} — risiko lebih rendah", False))

    # Tekanan darah
    if trestbps >= 140:
        factors.append(("⚠️ Tekanan Darah", f"{trestbps} mm Hg — hipertensi stage 2", True))
    elif trestbps >= 130:
        factors.append(("⚠️ Tekanan Darah", f"{trestbps} mm Hg — hipertensi stage 1", True))
    else:
        factors.append(("✅ Tekanan Darah", f"{trestbps} mm Hg — normal", False))

    # Kolesterol
    if chol >= 240:
        factors.append(("⚠️ Kolesterol", f"{chol} mg/dl — tinggi (≥240)", True))
    elif chol >= 200:
        factors.append(("⚠️ Kolesterol", f"{chol} mg/dl — borderline tinggi", True))
    else:
        factors.append(("✅ Kolesterol", f"{chol} mg/dl — normal", False))

    # Detak jantung maks
    max_expected = 220 - age
    pct = (thalach / max_expected) * 100
    if thalach < 120:
        factors.append(("⚠️ Detak Jantung Maks", f"{thalach} bpm — sangat rendah ({pct:.0f}% dari maks prediksi)", True))
    elif thalach < 140:
        factors.append(("⚠️ Detak Jantung Maks", f"{thalach} bpm — di bawah rata-rata untuk usia ini", True))
    else:
        factors.append(("✅ Detak Jantung Maks", f"{thalach} bpm — baik", False))

    # Angina saat olahraga
    if exang == 1:
        factors.append(("⚠️ Angina saat Olahraga", "Ada — indikator iskemia miokard", True))
    else:
        factors.append(("✅ Angina saat Olahraga", "Tidak ada", False))

    # Oldpeak
    if oldpeak >= 2.0:
        factors.append(("⚠️ Depresi ST (Oldpeak)", f"{oldpeak} — depresi signifikan, risiko tinggi", True))
    elif oldpeak >= 1.0:
        factors.append(("⚠️ Depresi ST (Oldpeak)", f"{oldpeak} — depresi ringan", True))
    else:
        factors.append(("✅ Depresi ST (Oldpeak)", f"{oldpeak} — normal", False))

    # Jumlah pembuluh darah (ca)
    if ca >= 2:
        factors.append(("⚠️ Pembuluh Darah Utama (ca)", f"{ca} pembuluh — penyempitan signifikan", True))
    elif ca == 1:
        factors.append(("⚠️ Pembuluh Darah Utama (ca)", f"{ca} pembuluh — penyempitan ringan", True))
    else:
        factors.append(("✅ Pembuluh Darah Utama (ca)", "0 pembuluh — bersih", False))

    # Thal
    thal_labels = {0: "Normal", 1: "Fixed Defect", 2: "Reversable Defect", 3: "Unknown"}
    if thal == 1:
        factors.append(("⚠️ Thalassemia (thal)", f"Fixed Defect — kerusakan permanen", True))
    elif thal == 2:
        factors.append(("⚠️ Thalassemia (thal)", f"Reversable Defect — perlu pemantauan", True))
    else:
        factors.append(("✅ Thalassemia (thal)", f"{thal_labels.get(thal, thal)}", False))

    return factors

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🫀 Prediksi Risiko Penyakit Jantung")
st.markdown(
    "Aplikasi ini menggunakan **Random Forest Classifier** untuk memprediksi risiko "
    "penyakit jantung berdasarkan data kesehatan pasien."
)
st.divider()

st.subheader("Masukkan Data Kesehatan Pasien")

col1, col2, col3 = st.columns(3)

with col1:
    age       = st.number_input("Usia", min_value=20, max_value=100, value=50)
    sex       = st.selectbox("Jenis Kelamin", options=[1, 0],
                              format_func=lambda x: "Laki-laki" if x == 1 else "Perempuan")
    cp        = st.selectbox("Tipe Nyeri Dada (cp)", options=[0, 1, 2, 3],
                              format_func=lambda x: {
                                  0: "0 – Typical Angina",
                                  1: "1 – Atypical Angina",
                                  2: "2 – Non-anginal Pain",
                                  3: "3 – Asymptomatic"
                              }[x])
    trestbps  = st.number_input("Tekanan Darah Istirahat (mm Hg)", min_value=80, max_value=220, value=120)
    chol      = st.number_input("Kolesterol (mg/dl)", min_value=100, max_value=600, value=200)

with col2:
    fbs       = st.selectbox("Gula Darah Puasa > 120 mg/dl", options=[0, 1],
                              format_func=lambda x: "Ya" if x == 1 else "Tidak")
    restecg   = st.selectbox("Hasil EKG Istirahat", options=[0, 1, 2],
                              format_func=lambda x: {
                                  0: "0 – Normal",
                                  1: "1 – ST-T Wave Abnormality",
                                  2: "2 – Left Ventricular Hypertrophy"
                              }[x])
    thalach   = st.number_input("Detak Jantung Maks (bpm)", min_value=60, max_value=220, value=150)
    exang     = st.selectbox("Angina saat Olahraga", options=[0, 1],
                              format_func=lambda x: "Ya" if x == 1 else "Tidak")

with col3:
    oldpeak   = st.number_input("Oldpeak (Depresi ST)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope     = st.selectbox("Slope Segmen ST", options=[0, 1, 2],
                              format_func=lambda x: {
                                  0: "0 – Downsloping",
                                  1: "1 – Flat",
                                  2: "2 – Upsloping"
                              }[x])
    ca        = st.selectbox("Jumlah Pembuluh Darah Utama (ca)", options=[0, 1, 2, 3])
    thal      = st.selectbox("Thalassemia (thal)", options=[0, 1, 2, 3],
                              format_func=lambda x: {
                                  0: "0 – Normal",
                                  1: "1 – Fixed Defect",
                                  2: "2 – Reversable Defect",
                                  3: "3 – Unknown"
                              }[x])

st.divider()

# ── Prediction ────────────────────────────────────────────────────────────────
if st.button("🔍 Prediksi Sekarang", type="primary", use_container_width=True):

    inputs = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    feature_names = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']

    input_df     = pd.DataFrame([inputs], columns=feature_names)
    input_scaled = scaler.transform(input_df)
    probability  = model.predict_proba(input_scaled)[0]
    prob_sakit   = probability[1]

    # Classify using tuned threshold
    is_berisiko  = prob_sakit >= THRESHOLD

    st.subheader("Hasil Prediksi")

    res_col, gauge_col = st.columns([2, 1])

    with res_col:
        if is_berisiko:
            st.error("### 🚨 Berisiko Penyakit Jantung")
            st.warning("Segera konsultasikan dengan dokter kardiologi untuk pemeriksaan lebih lanjut.")
        else:
            st.success("### ✅ Tidak Terdeteksi Risiko Signifikan")
            st.info("Tetap jaga pola hidup sehat dan lakukan pemeriksaan rutin.")

        # Probability bar
        st.markdown(f"**Probabilitas Risiko Penyakit Jantung: `{prob_sakit*100:.1f}%`**")
        st.progress(float(prob_sakit))
        st.caption(
            f"Threshold yang digunakan: {THRESHOLD:.2f} "
            f"(diturunkan dari default 0.50 untuk konteks medis — "
            f"meminimalkan False Negative)"
        )

    with gauge_col:
        # Simple risk level indicator
        if prob_sakit >= 0.70:
            risk_label, risk_color = "TINGGI", "🔴"
        elif prob_sakit >= THRESHOLD:
            risk_label, risk_color = "SEDANG", "🟠"
        elif prob_sakit >= 0.25:
            risk_label, risk_color = "RENDAH", "🟡"
        else:
            risk_label, risk_color = "MINIMAL", "🟢"

        st.markdown(
            f"""
            <div style='text-align:center; padding: 20px; border-radius: 10px;
                        border: 2px solid #444; background: #1e1e1e;'>
                <div style='font-size: 40px'>{risk_color}</div>
                <div style='font-size: 22px; font-weight: bold; margin-top: 8px'>
                    {risk_label}
                </div>
                <div style='color: #aaa; font-size: 13px'>Tingkat Risiko</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ── Risk Factor Breakdown ─────────────────────────────────────────────────
    st.divider()
    st.subheader("📋 Analisis Faktor Risiko")

    risk_factors   = evaluate_risk_factors(inputs)
    n_risk         = sum(1 for _, _, is_r in risk_factors if is_r)
    n_total        = len(risk_factors)

    st.markdown(f"Ditemukan **{n_risk} dari {n_total}** faktor yang berpotensi berisiko:")

    col_risk, col_safe = st.columns(2)

    with col_risk:
        st.markdown("**Faktor Risiko yang Terdeteksi**")
        has_risk = False
        for label, detail, is_r in risk_factors:
            if is_r:
                st.markdown(f"- {label}: *{detail}*")
                has_risk = True
        if not has_risk:
            st.markdown("_Tidak ada faktor risiko yang terdeteksi._")

    with col_safe:
        st.markdown("**Faktor dalam Kondisi Normal**")
        has_safe = False
        for label, detail, is_r in risk_factors:
            if not is_r:
                st.markdown(f"- {label}: *{detail}*")
                has_safe = True
        if not has_safe:
            st.markdown("_Semua faktor menunjukkan risiko._")

st.divider()
st.caption(
    "⚠️ **Disclaimer:** Prediksi ini adalah alat bantu berbasis data dan **tidak menggantikan "
    "diagnosis medis profesional**. Selalu konsultasikan kondisi kesehatan Anda dengan dokter."
)
