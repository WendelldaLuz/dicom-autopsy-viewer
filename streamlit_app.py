import streamlit as st
import tempfile
import pydicom
from pages import (
    visualizacao,
    estatisticas,
    analise_tecnica,
    qualidade,
    analise_post_mortem,
    ra_index,
    relatorios
)  # Importar todas as páginas

def main():
    st.set_page_config(page_title="DICOM Autopsy Viewer PRO", layout="wide")

    st.title("DICOM Autopsy Viewer PRO")

    uploaded_file = st.sidebar.file_uploader("Carregue um arquivo DICOM", type=['dcm', 'dicom'])

    if uploaded_file:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as tmp:
                tmp.write(uploaded_file.read())
                tmp.flush()
                dicom_data = pydicom.dcmread(tmp.name)
            image_array = dicom_data.pixel_array
        except Exception as e:
            st.error(f"Erro ao ler o arquivo DICOM: {e}")
            return

        page = st.sidebar.selectbox("Selecione a página", [
            "Visualização",
            "Estatísticas",
            "Análise Técnica",
            "Qualidade",
            "Análise Post-Mortem",
            "RA Index",
            "Relatórios"
        ])

        if page == "Visualização":
            visualizacao.show(dicom_data, image_array)
        elif page == "Estatísticas":
            estatisticas.show(dicom_data, image_array)
        elif page == "Análise Técnica":
            analise_tecnica.show(dicom_data, image_array)
        elif page == "Qualidade":
            qualidade.show(dicom_data, image_array)
        elif page == "Análise Post-Mortem":
            analise_post_mortem.show(dicom_data, image_array)
        elif page == "RA Index":
            ra_index.show(dicom_data, image_array)
        elif page == "Relatórios":
            relatorios.show(dicom_data, image_array)
    else:
        st.info("Por favor, carregue um arquivo DICOM para começar.")

if __name__ == "__main__":
    main()
