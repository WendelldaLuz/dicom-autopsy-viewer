import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px

def calculate_extended_statistics(image_array: np.ndarray) -> dict:
    """Calcula estatísticas descritivas e avançadas da imagem."""
    flattened = image_array.flatten()
    return {
        'Média': np.mean(flattened),
        'Mediana': np.median(flattened),
        'Desvio Padrão': np.std(flattened),
        'Erro Padrão': stats.sem(flattened),
        'Mínimo': np.min(flattened),
        'Máximo': np.max(flattened),
        'Amplitude': np.ptp(flattened),
        'Percentil 5': np.percentile(flattened, 5),
        'Percentil 25': np.percentile(flattened, 25),
        'Percentil 75': np.percentile(flattened, 75),
        'Percentil 95': np.percentile(flattened, 95),
        'IQR': np.percentile(flattened, 75) - np.percentile(flattened, 25),
        'Assimetria': stats.skew(flattened),
        'Curtose': stats.kurtosis(flattened),
        'Coeficiente de Variação': np.std(flattened) / np.mean(flattened) if np.mean(flattened) != 0 else 0
    }

def create_histogram_with_normal_fit(image_array: np.ndarray) -> go.Figure:
    flattened = image_array.flatten()
    mu, sigma = np.mean(flattened), np.std(flattened)
    x_range = np.linspace(np.min(flattened), np.max(flattened), 200)
    pdf = stats.norm.pdf(x_range, mu, sigma)
    scale_factor = len(flattened) * (np.max(flattened) - np.min(flattened)) / 100

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=flattened, nbinsx=100, name="Dados", opacity=0.7, marker_color='lightblue'))
    fig.add_trace(go.Scatter(x=x_range, y=pdf * scale_factor, mode='lines', name="Distribuição Normal", line=dict(color='red', width=2)))
    fig.update_layout(title="Histograma com Ajuste de Distribuição Normal", xaxis_title="Unidades Hounsfield (HU)", yaxis_title="Frequência", height=400)
    return fig

def create_qq_plot(image_array: np.ndarray) -> go.Figure:
    flattened = np.sort(image_array.flatten())
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(flattened)))
    sample_quantiles = np.percentile(flattened, np.linspace(1, 99, len(flattened)))
    min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
    max_val = max(theoretical_quantiles.max(), sample_quantiles.max())

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=theoretical_quantiles, y=sample_quantiles, mode='markers', name='Quantis Amostrais'))
    fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Referência', line=dict(color='red', dash='dash')))
    fig.update_layout(title="QQ Plot - Análise de Normalidade", xaxis_title="Quantis Teóricos", yaxis_title="Quantis Amostrais", height=400)
    return fig

def calculate_regional_statistics(image_array: np.ndarray, grid_size: int) -> pd.DataFrame:
    h, w = image_array.shape
    h_step, w_step = h // grid_size, w // grid_size
    regional_data = []
    for i in range(grid_size):
        for j in range(grid_size):
            region = image_array[i * h_step:(i + 1) * h_step, j * w_step:(j + 1) * w_step]
            if region.size > 0:
                regional_data.append({
                    'Região': f"{i + 1}-{j + 1}",
                    'X': j,
                    'Y': i,
                    'Média': np.mean(region),
                    'Mediana': np.median(region),
                    'Desvio Padrão': np.std(region),
                    'Mínimo': np.min(region),
                    'Máximo': np.max(region),
                    'Assimetria': stats.skew(region.flatten()),
                    'Área (%)': (region.size / image_array.size) * 100
                })
    return pd.DataFrame(regional_data)

def create_regional_heatmap(regional_stats: pd.DataFrame, grid_size: int) -> go.Figure:
    mean_matrix = np.zeros((grid_size, grid_size))
    for _, row in regional_stats.iterrows():
        i, j = int(row['Y']), int(row['X'])
        mean_matrix[i, j] = row['Média']
    fig = go.Figure(data=go.Heatmap(z=mean_matrix, colorscale='viridis', showscale=True,
                                    text=[[f"Média: {mean_matrix[i, j]:.1f}\nRegião: {i + 1}-{j + 1}" for j in range(grid_size)] for i in range(grid_size)],
                                    texttemplate="%{text}", textfont={"size": 10}))
    fig.update_layout(title="Mapa de Calor Regional - Valores Médios por Região", xaxis_title="Região X", yaxis_title="Região Y", height=500)
    return fig

def create_spatial_correlation_analysis(image_array: np.ndarray) -> go.Figure:
    from scipy import signal
    small_array = image_array[::max(1, image_array.shape[0] // 100), ::max(1, image_array.shape[1] // 100)]
    correlation = signal.correlate2d(small_array, small_array, mode='same')
    fig = go.Figure(data=go.Heatmap(z=correlation, colorscale='viridis', showscale=True))
    fig.update_layout(title="Matriz de Autocorrelação Espacial", height=400)
    return fig

def create_variogram_analysis(image_array: np.ndarray) -> go.Figure:
    import numpy as np
    from scipy.spatial.distance import pdist
    h, w = image_array.shape
    n_points = min(1000, h * w)
    indices = np.random.choice(h * w, n_points, replace=False)
    y_coords, x_coords = np.unravel_index(indices, (h, w))
    values = image_array.flatten()[indices]
    distances = pdist(np.column_stack([x_coords, y_coords]))
    value_diffs = pdist(values[:, None])
    squared_diffs = value_diffs ** 2
    max_distance = np.sqrt(h ** 2 + w ** 2) / 2
    bins = np.linspace(0, max_distance, 20)
    variogram_vals = np.zeros(len(bins) - 1)
    for i in range(len(bins) - 1):
        mask = (distances >= bins[i]) & (distances < bins[i + 1])
        if np.any(mask):
            variogram_vals[i] = np.mean(squared_diffs[mask]) / 2
    bin_centers = (bins[:-1] + bins[1:]) / 2
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bin_centers, y=variogram_vals, mode='lines+markers', name='Variograma Experimental'))
    fig.update_layout(title="Variograma Experimental", xaxis_title="Distância (pixels)", yaxis_title="Semivariância", height=400)
    return fig

def enhanced_statistics_tab(dicom_data, image_array):
    st.header(" Análise Estatística Avançada")

    with st.expander(" Referências Científicas (Normas ABNT)"):
        st.markdown("""
        - SILVA, W. L. Análise quantitativa de alterações post-mortem por tomografia computadorizada. 2023.
        - EGGER, C. et al. Development and validation of a postmortem radiological alteration index. Int J Legal Med, 2012.
        - ALTAIMIRANO, R. Técnicas de imagem aplicadas à tanatologia forense. Revista de Medicina Legal, 2022.
        """)

    tab_basic, tab_advanced, tab_regional, tab_correlation = st.tabs([
        "Estatísticas Básicas", "Análises Avançadas", "Análise Regional", "Correlação Espacial"
    ])

    with tab_basic:
        st.subheader("Estatísticas Descritivas")
        stats_data = calculate_extended_statistics(image_array)
        df_stats = pd.DataFrame(stats_data.items(), columns=["Métrica", "Valor"])
        st.table(df_stats)

        fig_hist = create_histogram_with_normal_fit(image_array)
        st.plotly_chart(fig_hist, use_container_width=True)

        fig_qq = create_qq_plot(image_array)
        st.plotly_chart(fig_qq, use_container_width=True)

    with tab_advanced:
        st.subheader("Análise Avançada")
        fig_heatmap = create_spatial_correlation_analysis(image_array)
        st.plotly_chart(fig_heatmap, use_container_width=True)

        fig_variogram = create_variogram_analysis(image_array)
        st.plotly_chart(fig_variogram, use_container_width=True)

    with tab_regional:
        st.subheader("Análise Regional")
        grid_size = st.slider("Tamanho da grade para análise regional", 2, 8, 4)
        regional_stats = calculate_regional_statistics(image_array, grid_size)
        fig_regional = create_regional_heatmap(regional_stats, grid_size)
        st.plotly_chart(fig_regional, use_container_width=True)
        st.dataframe(regional_stats, use_container_width=True)

    with tab_correlation:
        st.subheader("Correlação Espacial")
        fig_corr = create_spatial_correlation_analysis(image_array)
        st.plotly_chart(fig_corr, use_container_width=True)
