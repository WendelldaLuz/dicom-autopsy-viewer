import streamlit as st
from datetime import datetime

def enhanced_reporting_tab(dicom_data, image_array, user_data=None):
    st.subheader("Relatórios Completos")
    st.write("Aqui você pode implementar a geração e visualização de relatórios detalhados.")
    st.markdown(f"Relatório gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

def show(dicom_data, image_array):
    st.title("Relatórios")
    enhanced_reporting_tab(dicom_data, image_array)
