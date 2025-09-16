import streamlit as st
import numpy as np

def show(dicom_data, image_array):
    st.header("Análise Técnica Forense")
    st.write("Aqui você pode adicionar análises técnicas avançadas da imagem DICOM.")
    # Exemplo simples: mostrar estatísticas básicas técnicas
    st.metric("Média da Imagem", f"{np.mean(image_array):.2f}")
    st.metric("Desvio Padrão", f"{np.std(image_array):.2f}")
