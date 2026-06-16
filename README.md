# Laporan Proyek Machine Learning

## Identitas Kelompok

* Siti Rohmah Ramadhanti (2330511013)
* Prima Pratama Putra (2330511022)

---

# Project Overview

Penyakit jantung merupakan salah satu penyebab kematian tertinggi di dunia. Menurut World Health Organization (WHO), penyakit kardiovaskular menyebabkan sekitar 17,9 juta kematian setiap tahun atau sekitar 32% dari total kematian global. Oleh karena itu, deteksi dini terhadap risiko penyakit jantung menjadi langkah penting untuk mengurangi angka kematian dan meningkatkan kualitas hidup masyarakat.

Perkembangan teknologi Machine Learning memungkinkan pemanfaatan data kesehatan pasien untuk membantu proses prediksi penyakit secara lebih cepat dan efisien. Dengan menganalisis berbagai faktor kesehatan seperti usia, tekanan darah, kadar kolesterol, detak jantung maksimum, dan riwayat nyeri dada, model Machine Learning dapat digunakan untuk mengidentifikasi individu yang berisiko mengalami penyakit jantung.

Pada proyek ini digunakan algoritma Random Forest Classifier untuk membangun model klasifikasi yang mampu memprediksi keberadaan penyakit jantung berdasarkan data kesehatan pasien. Model yang dihasilkan kemudian dievaluasi menggunakan metrik Accuracy, Precision, Recall, dan F1-Score.

---

# Business Understanding

## Problem Statements

1. Bagaimana membangun model Machine Learning yang dapat memprediksi risiko penyakit jantung berdasarkan data kesehatan pasien?
2. Faktor kesehatan apa saja yang paling berpengaruh terhadap prediksi penyakit jantung?

## Goals

1. Membangun model klasifikasi untuk memprediksi risiko penyakit jantung.
2. Mengevaluasi performa model menggunakan metrik klasifikasi.
3. Mengidentifikasi fitur yang berpengaruh terhadap hasil prediksi.

## Solution Approach

Pendekatan yang digunakan pada proyek ini adalah algoritma **Random Forest Classifier**. Algoritma ini dipilih karena mampu memberikan performa yang baik pada permasalahan klasifikasi, relatif tahan terhadap overfitting, serta dapat digunakan untuk mengetahui tingkat kepentingan masing-masing fitur.

Evaluasi model dilakukan menggunakan:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

# Data Understanding

Dataset yang digunakan adalah **Heart Disease Dataset** yang berisi informasi kesehatan pasien dan status penyakit jantung.

Jumlah data awal:

* 1025 baris data
* 14 kolom

## Deskripsi Fitur

| Kolom    | Keterangan                   |
| -------- | ---------------------------- |
| age      | Usia pasien                  |
| sex      | Jenis kelamin                |
| cp       | Tipe nyeri dada              |
| trestbps | Tekanan darah saat istirahat |
| chol     | Kadar kolesterol             |
| fbs      | Gula darah puasa             |
| restecg  | Hasil elektrokardiogram      |
| thalach  | Detak jantung maksimum       |
| exang    | Angina akibat olahraga       |
| oldpeak  | Depresi ST                   |
| slope    | Kemiringan segmen ST         |
| ca       | Jumlah pembuluh darah utama  |
| thal     | Tipe thalassemia             |
| target   | Status penyakit jantung      |

## Kondisi Data

Hasil eksplorasi data menunjukkan:

* Tidak ditemukan missing values.
* Ditemukan 723 data duplikat.
* Data duplikat dihapus pada tahap data cleaning.
* Distribusi kelas target relatif seimbang antara pasien yang memiliki penyakit jantung dan yang tidak memiliki penyakit jantung.

---

# Data Preparation

Tahapan persiapan data yang dilakukan adalah sebagai berikut:

## 1. Data Cleaning

Dilakukan penghapusan data duplikat untuk meningkatkan kualitas dataset.

Jumlah data:

* Sebelum cleaning: 1025 data
* Setelah cleaning: 302 data

## 2. Pemisahan Fitur dan Target

Fitur (X) terdiri dari 13 atribut kesehatan pasien.

Target (y):

* 0 = Tidak memiliki penyakit jantung
* 1 = Memiliki penyakit jantung

## 3. Standardisasi Data

Data dinormalisasi menggunakan **StandardScaler** agar setiap fitur memiliki skala yang sebanding dan tidak mendominasi proses pembelajaran model.

## 4. Train-Test Split

Dataset dibagi menjadi:

* Data Training: 241 data (80%)
* Data Testing: 61 data (20%)

Pembagian data dilakukan menggunakan `random_state = 42`.

---

# Modeling

## Algoritma

Model yang digunakan adalah **Random Forest Classifier**.

Random Forest merupakan metode ensemble learning yang membangun banyak decision tree dan menggabungkan hasil prediksi setiap pohon menggunakan mekanisme majority voting.

### Parameter Model

* n_estimators = 100
* max_depth = 10
* random_state = 42

### Kelebihan Random Forest

* Mampu menangani data dengan banyak fitur.
* Mengurangi risiko overfitting dibandingkan Decision Tree tunggal.
* Memiliki performa yang baik untuk kasus klasifikasi.
* Dapat digunakan untuk mengetahui tingkat kepentingan fitur (feature importance).

---

# Evaluation

Model dievaluasi menggunakan data testing.

## Hasil Evaluasi

| Metrik    | Nilai  |
| --------- | ------ |
| Accuracy  | 73.77% |
| Precision | 74%    |
| Recall    | 79%    |
| F1-Score  | 76%    |

Hasil tersebut menunjukkan bahwa model mampu melakukan klasifikasi risiko penyakit jantung dengan performa yang cukup baik pada dataset yang digunakan.

## Confusion Matrix

Confusion Matrix digunakan untuk melihat jumlah prediksi yang benar maupun salah pada setiap kelas.

Dalam konteks prediksi penyakit jantung, kesalahan False Negative perlu diperhatikan karena dapat menyebabkan pasien yang sebenarnya berisiko tidak terdeteksi oleh sistem.

## Feature Importance

Berdasarkan hasil analisis feature importance, beberapa fitur yang memiliki kontribusi besar terhadap prediksi penyakit jantung antara lain:

* cp (Chest Pain Type)
* thalach (Maximum Heart Rate)
* oldpeak
* ca

---

# Deployment

Model telah diimplementasikan dalam bentuk aplikasi web sederhana menggunakan **Streamlit**.

Aplikasi memungkinkan pengguna memasukkan data kesehatan pasien dan memperoleh hasil prediksi risiko penyakit jantung secara langsung berdasarkan model Random Forest yang telah dilatih.

### Cara Menjalankan Aplikasi

```bash
streamlit run HeartDiase.py
```

---

# Struktur Repository

```text
heart-disease-prediction/
│
├── HeartDiase_ML.ipynb
├── HeartDiase.py
├── heart.csv
├── model_rf.pkl
├── scaler.pkl
├── requirements.txt
└── README.md
```

---

# Kesimpulan

Pada proyek ini berhasil dibangun model Machine Learning menggunakan algoritma Random Forest Classifier untuk memprediksi risiko penyakit jantung berdasarkan data kesehatan pasien.

Setelah dilakukan proses pembersihan data, persiapan data, pelatihan model, dan evaluasi, model memperoleh tingkat akurasi sebesar 73,77%. Hasil tersebut menunjukkan bahwa Machine Learning dapat dimanfaatkan sebagai alat bantu dalam mendukung deteksi dini penyakit jantung berdasarkan data kesehatan yang tersedia.

---

# Referensi

1. World Health Organization (WHO). Cardiovascular Diseases (CVDs).
2. Heart Disease Dataset – Kaggle.
3. Scikit-Learn Documentation.
4. Han, J., Kamber, M., & Pei, J. Data Mining: Concepts and Techniques.
