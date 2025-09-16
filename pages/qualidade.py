import streamlit as st
import numpy as np

def calculate_quality_metrics(image_array):
    snr = np.mean(image_array) / (np.std(image_array) + 1e-6)
    stnr = f"{snr:.2f}"
    return snr

def enhanced_quality_metrics_tab(dicom_data, image_array):
    st.subheader("Métricas de Qualidade de Imagem")
    snr = calculate_quality_metrics(image_array)
    st.metric("SNR (Signal-to-Noise Ratio)", snr)

def show(dicom_data, image_array):
    st.title("Qualidade")
    enhanced_quality_metrics_tab(dicom_data, image_array)
