import streamlit as st
import tempfile
import pydicom
from pages import visualizacao, estatisticas  # Importar páginas que vamos criar

def main():
    st.set_page_config(page_title="DICOM Autopsy Viewer PRO", layout="wide")

    st.title("DICOM Autopsy Viewer PRO")

    uploaded_file = st.sidebar.file_uploader("Carregue um arquivo DICOM", type=['dcm', 'dicom'])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as tmp:
            tmp.write(uploaded_file.read())
            dicom_data = pydicom.dcmread(tmp.name)
        image_array = dicom_data.pixel_array

        page = st.sidebar.selectbox("Selecione a página", ["Visualização", "Estatísticas"])

        if page == "Visualização":
            visualizacao.show(dicom_data, image_array)
        elif page == "Estatísticas":
            estatisticas.show(dicom_data, image_array)
    else:
        st.info("Por favor, carregue um arquivo DICOM para começar.")

if __name__ == "__main__":
    main()
