import streamlit as st
import numpy as np

def show(dicom_data, image_array):
    st.header("Métricas de Qualidade de Imagem")
    st.write("Análise básica de qualidade da imagem DICOM.")
    snr = np.mean(image_array) / (np.std(image_array) + 1e-6)
    st.metric("SNR (Signal-to-Noise Ratio)", f"{snr:.2f}")
