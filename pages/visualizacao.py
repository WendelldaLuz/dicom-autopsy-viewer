import streamlit as st

def show(dicom_data, image_array):
    st.title("Visualização")
    st.write("Visualização avançada da imagem DICOM com janelamento e ferramentas colorimétricas.")
    st.image(image_array, caption="Imagem DICOM", use_column_width=True)
