import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

def generate_ra_index_data(image_array):
    grid_size = 8
    h, w = image_array.shape
    h_step, w_step = h // grid_size, w // grid_size
    ra_values = []
    for i in range(grid_size):
        for j in range(grid_size):
            region = image_array[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step]
            mean_intensity = np.mean(region)
            ra_values.append(mean_intensity)
    return ra_values, grid_size

def enhanced_ra_index_tab(dicom_data, image_array):
    st.subheader("RA-Index - Análise de Risco Aprimorada")
    ra_values, grid_size = generate_ra_index_data(image_array)
    ra_matrix = np.array(ra_values).reshape(grid_size, grid_size)
    fig = go.Figure(data=go.Heatmap(
        z=ra_matrix,
        colorscale='RdYlBu_r',
        showscale=True,
        text=ra_matrix.round(1),
        texttemplate="%{text}",
        textfont={"size": 12, "color": "white"},
    ))
    fig.update_layout(title="Mapa de Calor - RA-Index", height=400)
    st.plotly_chart(fig, use_container_width=True)

def show(dicom_data, image_array):
    st.title("RA Index")
    enhanced_ra_index_tab(dicom_data, image_array)
