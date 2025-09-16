import streamlit as st
import numpy as np

def show(dicom_data, image_array):
    st.header("Estatísticas Básicas da Imagem")
    st.metric("Dimensões", f"{image_array.shape[0]} x {image_array.shape[1]}")
    st.metric("Valor Mínimo", f"{np.min(image_array)}")
    st.metric("Valor Máximo", f"{np.max(image_array)}")
    st.metric("Média", f"{np.mean(image_array):.2f}")
    st.metric("Desvio Padrão", f"{np.std(image_array):.2f}")
