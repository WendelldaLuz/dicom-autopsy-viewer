import streamlit as st
from io import BytesIO
from datetime import datetime
import csv
import json

def generate_csv_report(report_data):
    output = BytesIO()
    writer = csv.writer(output)
    writer.writerow(["DICOM AUTOPSY VIEWER PRO - RELATÓRIO DE ANÁLISE"])
    writer.writerow(["Data de Geração", datetime.now().strftime('%d/%m/%Y %H:%M')])
    writer.writerow([])
    for section, content in report_data.items():
        writer.writerow([section])
        if isinstance(content, dict):
            for k, v in content.items():
                writer.writerow([k, v])
        else:
            writer.writerow([str(content)])
        writer.writerow([])
    output.seek(0)
    return output

def enhanced_reporting_tab(dicom_data, image_array):
    st.subheader("Relatórios Completos")
    report_name = st.text_input("Nome do Relatório", value=f"Relatorio_{datetime.now().strftime('%Y%m%d_%H%M')}")
    format_option = st.selectbox("Formato de Exportação", ["CSV"])

    if st.button("Gerar Relatório Completo"):
        # Exemplo: coletar dados de análise (implemente conforme seu código)
        report_data = {
            "Metadados": {"PatientName": getattr(dicom_data, "PatientName", "N/A")},
            "Estatísticas": {"Média HU": float(image_array.mean())},
            "Análise Post-Mortem": {"IPM Estimado": "Exemplo"},
            # Adicione outras seções conforme implementado
        }

        if format_option == "CSV":
            report_file = generate_csv_report(report_data)
            st.download_button(
                label=f"Download do Relatório ({format_option})",
                data=report_file,
                file_name=f"{report_name}.csv",
                mime="text/csv",
                use_container_width=True
            )
