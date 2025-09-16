import streamlit as st
import numpy as np
import plotly.express as px

def calculate_extended_statistics(image_array):
    flattened = image_array.flatten()
    stats = {
        'Média': np.mean(flattened),
        'Mediana': np.median(flattened),
        'Desvio Padrão': np.std(flattened),
        'Mínimo': np.min(flattened),
        'Máximo': np.max(flattened),
    }
    return stats

def enhanced_statistics_tab(dicom_data, image_array):
    st.subheader("Análise Estatística Avançada")
    stats = calculate_extended_statistics(image_array)
    for k, v in stats.items():
        st.metric(k, f"{v:.2f}")
    fig = px.histogram(image_array.flatten(), nbins=50, title="Histograma de Intensidades")
    st.plotly_chart(fig, use_container_width=True)

def show(dicom_data, image_array):
    st.title("Estatísticas")
    enhanced_statistics_tab(dicom_data, image_array)
