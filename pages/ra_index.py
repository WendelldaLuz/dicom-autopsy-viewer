import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from io import BytesIO

def generate_advanced_ra_index_data(image_array: np.ndarray):
    """
    Gera dados detalhados do RA-Index com categorização de risco e tipos teciduais.
    """
    h, w = image_array.shape
    grid_size = 8
    h_step, w_step = h // grid_size, w // grid_size

    ra_data = {
        'coords': [],
        'ra_values': [],
        'risk_categories': [],
        'tissue_types': [],
        'intensities': []
    }

    def categorize_risk(mean_intensity):
        if mean_intensity < -500:
            return 'Baixo', 'Gás/Ar'
        elif -500 <= mean_intensity < 0:
            return 'Baixo', 'Gordura'
        elif 0 <= mean_intensity < 100:
            return 'Médio', 'Tecido Mole'
        elif 100 <= mean_intensity < 400:
            return 'Médio', 'Músculo'
        elif 400 <= mean_intensity < 1000:
            return 'Alto', 'Osso'
        else:
            return 'Crítico', 'Metal/Implante'

    for i in range(grid_size):
        for j in range(grid_size):
            region = image_array[i * h_step:(i + 1) * h_step, j * w_step:(j + 1) * w_step]
            mean_intensity = np.mean(region)
            std_intensity = np.std(region)
            intensity_factor = min(abs(mean_intensity) / 1000, 1.0)
            variation_factor = min(std_intensity / 500, 1.0)
            center_distance = np.sqrt((i - grid_size / 2) ** 2 + (j - grid_size / 2) ** 2)
            position_factor = 1 - (center_distance / (grid_size / 2))
            ra_value = (intensity_factor * 0.5 + variation_factor * 0.3 + position_factor * 0.2) * 100
            risk_category, tissue_type = categorize_risk(mean_intensity)

            ra_data['coords'].append((i, j))
            ra_data['ra_values'].append(ra_value)
            ra_data['risk_categories'].append(risk_category)
            ra_data['tissue_types'].append(tissue_type)
            ra_data['intensities'].append(mean_intensity)

    return ra_data, grid_size

def enhanced_ra_index_tab(dicom_data, image_array):
    st.header("📈 RA-Index - Análise de Risco Aprimorada")

    ra_data, grid_size = generate_advanced_ra_index_data(image_array)

    st.markdown("### Estatísticas Gerais do RA-Index")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_ra = np.mean(ra_data['ra_values'])
        st.metric("RA-Index Médio", f"{avg_ra:.1f}")
    with col2:
        max_ra = np.max(ra_data['ra_values'])
        st.metric("RA-Index Máximo", f"{max_ra:.1f}")
    with col3:
        risk_counts = pd.Series(ra_data['risk_categories']).value_counts()
        critical_count = risk_counts.get('Crítico', 0)
        st.metric("Regiões Críticas", critical_count)
    with col4:
        high_risk_count = risk_counts.get('Alto', 0)
        st.metric("Regiões Alto Risco", high_risk_count)

    st.markdown("### Mapas de Calor Avançados")
    col1, col2 = st.columns(2)
    with col1:
        ra_matrix = np.array(ra_data['ra_values']).reshape(grid_size, grid_size)
        fig1 = go.Figure(data=go.Heatmap(
            z=ra_matrix,
            colorscale='RdYlBu_r',
            showscale=True,
            text=ra_matrix.round(1),
            texttemplate="%{text}",
            textfont={"size": 12, "color": "white"},
            hoverongaps=False
        ))
        fig1.update_layout(title="Mapa de Calor - RA-Index", xaxis_title="Região X", yaxis_title="Região Y", height=500)
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        tissue_mapping = {'Gás/Ar': 1, 'Gordura': 2, 'Tecido Mole': 3, 'Músculo': 4, 'Osso': 5, 'Metal/Implante': 6}
        tissue_matrix = np.array([tissue_mapping[t] for t in ra_data['tissue_types']]).reshape(grid_size, grid_size)
        fig2 = go.Figure(data=go.Heatmap(
            z=tissue_matrix,
            colorscale='viridis',
            showscale=True,
            text=np.array(ra_data['tissue_types']).reshape(grid_size, grid_size),
            texttemplate="%{text}",
            textfont={"size": 8, "color": "white"},
            hoverongaps=False
        ))
        fig2.update_layout(title="Mapa de Tipos de Tecido", xaxis_title="Região X", yaxis_title="Região Y", height=500)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Análise de Distribuição de Risco")
    col1, col2 = st.columns(2)
    with col1:
        fig3 = go.Figure(data=[go.Pie(
            labels=list(risk_counts.index),
            values=list(risk_counts.values),
            hole=.3,
            marker_colors=['#FF4B4B', '#FFA500', '#FFFF00', '#90EE90']
        )])
        fig3.update_layout(title="Distribuição de Categorias de Risco", height=400)
        st.plotly_chart(fig3, use_container_width=True)
    with col2:
        fig4 = go.Figure()
        fig4.add_trace(go.Histogram(
            x=ra_data['ra_values'],
            nbinsx=20,
            name="RA-Index",
            marker_color='lightcoral',
            opacity=0.7
        ))
        fig4.add_vline(x=np.mean(ra_data['ra_values']), line_dash="dash", line_color="red", annotation_text="Média")
        fig4.add_vline(x=np.percentile(ra_data['ra_values'], 90), line_dash="dash", line_color="orange", annotation_text="P90")
        fig4.update_layout(title="Distribuição de Valores RA-Index", xaxis_title="RA-Index", yaxis_title="Frequência", height=400)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("### Análise Temporal Simulada")
    time_points = ['T0', 'T1', 'T2', 'T3', 'T4', 'T5']
    temporal_data = {level: [] for level in ['Crítico', 'Alto', 'Médio', 'Baixo']}
    base_counts = risk_counts.to_dict()
    for i, _ in enumerate(time_points):
        variation = 1 + 0.1 * np.sin(i * np.pi / 3) + np.random.normal(0, 0.05)
        for risk_level in temporal_data.keys():
            base_value = base_counts.get(risk_level, 0)
            temporal_data[risk_level].append(max(0, int(base_value * variation)))

    fig5 = go.Figure()
    colors = {'Crítico': 'red', 'Alto': 'orange', 'Médio': 'yellow', 'Baixo': 'green'}
    for risk_level, values in temporal_data.items():
        fig5.add_trace(go.Scatter(
            x=time_points,
            y=values,
            mode='lines+markers',
            name=risk_level,
            line=dict(color=colors[risk_level], width=3),
            marker=dict(size=8)
        ))
    fig5.update_layout(title="Evolução Temporal das Categorias de Risco", xaxis_title="Ponto Temporal", yaxis_title="Número de Regiões", height=400, hovermode='x unified')
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("### Exportar Dados RA-Index")
    if st.button("Gerar Relatório RA-Index"):
        df_export = pd.DataFrame({
            'Região_X': [coord[0] for coord in ra_data['coords']],
            'Região_Y': [coord[1] for coord in ra_data['coords']],
            'RA_Index': ra_data['ra_values'],
            'Categoria_Risco': ra_data['risk_categories'],
            'Tipo_Tecido': ra_data['tissue_types'],
            'Intensidade_Média': ra_data['intensities']
        })
        csv_buffer = BytesIO()
        df_export.to_csv(csv_buffer, index=False, encoding='utf-8')
        csv_buffer.seek(0)
        st.download_button(
            label="Baixar Dados RA-Index (CSV)",
            data=csv_buffer,
            file_name=f"ra_index_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        st.success("Relatório RA-Index preparado para download!")
