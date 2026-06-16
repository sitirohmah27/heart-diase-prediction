import streamlit as st
import numpy as np
import pickle

# Load model dan scaler
@st.cache_resource
def load_model():
    with open('model_rf.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# Judul aplikasi
st.title('Prediksi Risiko Penyakit Jantung')
st.markdown('Aplikasi ini menggunakan algoritma **Random Forest** untuk memprediksi risiko penyakit jantung berdasarkan data kesehatan pasien.')
st.divider()

# Input data pasien
st.subheader('Masukkan Data Kesehatan Pasien')

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input('Usia', min_value=20, max_value=100, value=50)
    sex = st.selectbox('Jenis Kelamin', options=[1, 0], format_func=lambda x: 'Laki-laki' if x == 1 else 'Perempuan')
    cp = st.selectbox('Tipe Nyeri Dada (cp)', options=[0, 1, 2, 3],
                      help='0=Typical Angina, 1=Atypical Angina, 2=Non-anginal Pain, 3=Asymptomatic')
    trestbps = st.number_input('Tekanan Darah (mm Hg)', min_value=80, max_value=220, value=120)
    chol = st.number_input('Kolesterol (mg/dl)', min_value=100, max_value=600, value=200)

with col2:
    fbs = st.selectbox('Gula Darah Puasa > 120 mg/dl', options=[0, 1],
                       format_func=lambda x: 'Ya' if x == 1 else 'Tidak')
    restecg = st.selectbox('Hasil EKG Istirahat', options=[0, 1, 2],
                           help='0=Normal, 1=ST-T Wave Abnormality, 2=Left Ventricular Hypertrophy')
    thalach = st.number_input('Detak Jantung Maks', min_value=60, max_value=220, value=150)
    exang = st.selectbox('Angina saat Olahraga', options=[0, 1],
                         format_func=lambda x: 'Ya' if x == 1 else 'Tidak')

with col3:
    oldpeak = st.number_input('Oldpeak (Depresi ST)', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox('Slope ST', options=[0, 1, 2],
                         help='0=Downsloping, 1=Flat, 2=Upsloping')
    ca = st.selectbox('Jumlah Pembuluh Darah Utama (ca)', options=[0, 1, 2, 3])
    thal = st.selectbox('Thal', options=[0, 1, 2, 3],
                        help='0=Normal, 1=Fixed Defect, 2=Reversable Defect, 3=Unknown')

st.divider()

# Tombol prediksi
if st.button('Prediksi Sekarang', type='primary', use_container_width=True):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    st.subheader('Hasil Prediksi')
    if prediction == 1:
        st.error(f'**Berisiko Penyakit Jantung**')
        st.metric('Probabilitas Berisiko', f'{probability[1]*100:.1f}%')
        st.warning('Segera konsultasikan dengan dokter untuk pemeriksaan lebih lanjut.')
    else:
        st.success(f'**Tidak Berisiko Penyakit Jantung**')
        st.metric('Probabilitas Tidak Berisiko', f'{probability[0]*100:.1f}%')
        st.info('Tetap jaga pola hidup sehat dan lakukan pemeriksaan rutin.')

st.divider()
st.caption('Disclaimer: Prediksi ini hanya alat bantu dan tidak menggantikan diagnosis medis profesional.')
