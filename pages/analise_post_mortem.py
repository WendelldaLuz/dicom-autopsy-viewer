import streamlit as st
import numpy as np

def show(dicom_data, image_array):
    st.header("Análise Post-Mortem Avançada")
    st.write("Aqui você pode implementar as análises post-mortem detalhadas.")
  
    mean_val = np.mean(image_array)
    std_val = np.std(image_array)
    st.metric("Média da Imagem", f"{mean_val:.2f}")
    st.metric("Desvio Padrão", f"{std_val:.2f}")
