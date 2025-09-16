import streamlit as st
import numpy as np
import pandas as pd

def estatisticas_tab(dicom_data, image_array: np.ndarray):
    st.header("📊 Estatísticas Detalhadas")

    try:
        data = {
            "Média": np.mean(image_array),
            "Mediana": np.median(image_array),
            "Desvio Padrão": np.std(image_array),
            "Máximo": np.max(image_array),
            "Mínimo": np.min(image_array),
            "Percentil 25": np.percentile(image_array, 25),
            "Percentil 75": np.percentile(image_array, 75),
        }
        df_stats = pd.DataFrame(data.items(), columns=["Métrica", "Valor"])
        st.table(df_stats)
    except Exception as e:
        st.error(f"Erro ao calcular estatísticas: {e}")
