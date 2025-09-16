import streamlit as st
import numpy as np
from skimage import filters

def qualidade_tab(dicom_data, image_array: np.ndarray):
    st.header("✅ Métricas de Qualidade da Imagem")

    try:
        # Exemplo: cálculo de contraste local via filtro Sobel
        sobel_edges = filters.sobel(image_array)
        contrast_metric = np.mean(sobel_edges)
        st.write(f"Contraste local médio (Sobel): {contrast_metric:.4f}")

        st.image(sobel_edges, caption="Mapa de Bordas (Sobel)", use_container_width=True)
    except Exception as e:
        st.error(f"Erro ao calcular métricas de qualidade: {e}")
