import streamlit as st
import joblib
import numpy as np


st.set_page_config(page_title="Prediksi Penjualan F&B", layout="centered")

st.title("🍽️ Aplikasi Prediksi Penjualan F&B")
st.write("Aplikasi untuk memperkirakan jumlah penjualan harian berdasarkan fitur operasional.")


@st.cache_resource
def load_model():
    model = joblib.load('model_penjualan.pth')
    return model

model = load_model()

st.sidebar.header("📝 Parameter Input Data")


day_of_week = st.sidebar.selectbox("Hari (0=Senin, 1=Selasa, ..., 6=Minggu)", [0, 1, 2, 3, 4, 5, 6])
is_weekend = 1 if day_of_week >= 5 else 0
month = st.sidebar.slider("Bulan (1-12)", 1, 12, 1)
day = st.sidebar.slider("Tanggal (1-31)", 1, 31, 1)
lag_1 = st.sidebar.number_input("Penjualan Hari Sebelumnya (Lag 1)", value=50.0)
lag_7 = st.sidebar.number_input("Penjualan 7 Hari Lalu (Lag 7)", value=50.0)
rolling_mean_7 = st.sidebar.number_input("Rata-rata 7 Hari (Rolling Mean 7)", value=50.0)


if st.sidebar.button("Hitung Prediksi"):
    input_data = np.array([[day_of_week, is_weekend, month, day, lag_1, lag_7, rolling_mean_7]])
    prediction = model.predict(input_data)
    
    st.subheader("📊 Hasil Prediksi:")
    st.success(f"Perkiraan Jumlah Penjualan: **{round(prediction[0], 2)} Unit**")