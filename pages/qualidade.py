import streamlit as st
from utils.image_processing import calcular_snr

def show(dicom_data, image_array):
    st.header("Métricas de Qualidade de Imagem")
    st.write("Análise básica de qualidade da imagem DICOM.")

    snr = calcular_snr(image_array)
    st.metric("SNR (Signal-to-Noise Ratio)", f"{snr:.2f}")
