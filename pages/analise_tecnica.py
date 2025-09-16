import streamlit as st
import numpy as np
import plotly.express as px

def enhanced_technical_analysis_tab(dicom_data, image_array: np.ndarray):
    st.header("🛠️ Análise Técnica Avançada")

    st.markdown("### Histogramas e Distribuição de Intensidades")
    try:
        hist_data = image_array.flatten()
        fig = px.histogram(hist_data, nbins=100, title="Histograma de Intensidades (HU)")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Erro ao gerar histograma: {e}")

    st.markdown("### Estatísticas Básicas")
    try:
        st.write(f"Média: {np.mean(image_array):.2f} HU")
        st.write(f"Mediana: {np.median(image_array):.2f} HU")
        st.write(f"Desvio Padrão: {np.std(image_array):.2f} HU")
        st.write(f"Valor Máximo: {np.max(image_array):.2f} HU")
        st.write(f"Valor Mínimo: {np.min(image_array):.2f} HU")
    except Exception as e:
        st.error(f"Erro ao calcular estatísticas: {e}")

    st.markdown("### Visualização 3D (se aplicável)")
    st.info("Implementação futura: visualização 3D com VTK ou PyVista.")
