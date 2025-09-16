import streamlit as st
import plotly.express as px

def show(dicom_data, image_array):
    st.header("Visualização Avançada")
    fig = px.imshow(image_array, color_continuous_scale='gray')
    st.plotly_chart(fig, use_container_width=True)
