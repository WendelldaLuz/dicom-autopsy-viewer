import streamlit as st
from io import BytesIO
from datetime import datetime
import csv
import json
import uuid
import logging

def generate_csv_report(report_data: dict, report_name: str) -> BytesIO:
    output = BytesIO()
    writer = csv.writer(output)
    writer.writerow(["DICOM AUTOPSY VIEWER PRO - RELATÓRIO DE ANÁLISE"])
    writer.writerow(["Nome do Relatório", report_name])
    writer.writerow(["Data de Geração", datetime.now().strftime('%d/%m/%Y %H:%M')])
    writer.writerow(["ID do Relatório", report_data.get('report_id', 'N/A')])
    writer.writerow([])
    for section, content in report_data.items():
        if section in ['report_id', 'generated_at']:
            continue
        writer.writerow([section.upper()])
        if isinstance(content, dict):
            for k, v in content.items():
                writer.writerow([k, v])
        else:
            writer.writerow([str(content)])
        writer.writerow([])
    output.seek(0)
    return output

def generate_html_report(report_data: dict, report_name: str) -> BytesIO:
    html_content = f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <title>{report_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 900px; margin: auto; padding: 20px; background: #fff; color: #000; }}
            h1, h2 {{ border-bottom: 2px solid #000; padding-bottom: 5px; }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
            th {{ background-color: #f0f0f0; }}
        </style>
    </head>
    <body>
        <h1>DICOM AUTOPSY VIEWER PRO</h1>
        <h2>Relatório de Análise Forense</h2>
        <p><strong>Nome do Relatório:</strong> {report_name}</p>
        <p><strong>Data de Geração:</strong> {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
        <p><strong>ID do Relatório:</strong> {report_data.get('report_id', 'N/A')}</p>
    """

    for section, content in report_data.items():
        if section in ['report_id', 'generated_at']:
            continue
        html_content += f"<h3>{section.upper()}</h3>"
        if isinstance(content, dict):
            html_content += "<table><thead><tr><th>Campo</th><th>Valor</th></tr></thead><tbody>"
            for k, v in content.items():
                html_content += f"<tr><td>{k}</td><td>{v}</td></tr>"
            html_content += "</tbody></table>"
        else:
            html_content += f"<p>{content}</p>"

    html_content += """
        <footer><p>Relatório gerado por DICOM Autopsy Viewer PRO v3.0 - © 2025</p></footer>
    </body>
    </html>
    """
    return BytesIO(html_content.encode('utf-8'))

def generate_pdf_report(report_data: dict, report_name: str) -> BytesIO:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib import colors
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
        styles = getSampleStyleSheet()
        story = []
        story.append(Paragraph("DICOM AUTOPSY VIEWER PRO", styles['Title']))
        story.append(Paragraph("Relatório de Análise Forense", styles['Heading2']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Nome do Relatório: {report_name}", styles['Normal']))
        story.append(Paragraph(f"Data de Geração: {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles['Normal']))
        story.append(Paragraph(f"ID do Relatório: {report_data.get('report_id', 'N/A')}", styles['Normal']))
        story.append(Spacer(1, 24))

        for section, content in report_data.items():
            if section in ['report_id', 'generated_at']:
                continue
            story.append(Paragraph(section.upper(), styles['Heading3']))
            if isinstance(content, dict):
                data = [["Campo", "Valor"]] + [[str(k), str(v)] for k, v in content.items()]
                table = Table(data, hAlign='LEFT')
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ]))
                story.append(table)
            else:
                story.append(Paragraph(str(content), styles['Normal']))
            story.append(Spacer(1, 12))

        doc.build(story)
        buffer.seek(0)
        return buffer
    except ImportError:
        st.error("Biblioteca ReportLab não instalada. Instale para gerar relatórios PDF.")
        return BytesIO(b"")

def save_report_to_db(user_email, report_name, report_data_bytes, parameters):
    import sqlite3
    import json
    import logging
    try:
        conn = sqlite3.connect("dicom_viewer.db")
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO reports (user_email, report_name, report_data, parameters)
            VALUES (?, ?, ?, ?)
        """, (user_email, report_name, report_data_bytes, json.dumps(parameters)))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logging.error(f"Erro ao salvar relatório: {e}")
        return False

def get_user_reports(user_email):
    import sqlite3
    import logging
    try:
        conn = sqlite3.connect("dicom_viewer.db")
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, report_name, generated_at
            FROM reports
            WHERE user_email = ?
            ORDER BY generated_at DESC
        """, (user_email,))
        reports = cursor.fetchall()
        conn.close()
        return reports
    except Exception as e:
        logging.error(f"Erro ao recuperar relatórios: {e}")
        return []

def enhanced_reporting_tab(dicom_data, image_array, user_data):
    st.header("📄 Relatórios Completos")

    report_tab1, report_tab2, report_tab3 = st.tabs(["Gerar Relatório", "Relatórios Salvos", "Configurações"])

    with report_tab1:
        st.markdown("### Personalizar Relatório")
        col1, col2 = st.columns(2)
        with col1:
            report_name = st.text_input("Nome do Relatório", value=f"Análise_{datetime.now().strftime('%Y%m%d_%H%M')}")
            report_type = st.selectbox("Tipo de Relatório", ["Completo", "Forense", "Qualidade", "Estatístico", "Técnico"])
            include_sections = st.multiselect(
                "Seções a Incluir",
                ["Metadados", "Estatísticas", "Análise Técnica", "Qualidade", "Análise Post-Mortem", "RA-Index", "Visualizações", "Imagens"],
                default=["Metadados", "Estatísticas", "Análise Técnica", "Qualidade", "Análise Post-Mortem", "RA-Index"]
            )
        with col2:
            format_option = st.selectbox("Formato de Exportação", ["PDF", "HTML", "CSV"])
            include_3d = st.checkbox("Incluir visualizações 3D", value=True)
            include_heatmaps = st.checkbox("Incluir mapas de calor", value=True)
            include_graphs = st.checkbox("Incluir gráficos estatísticos", value=True)

        if st.button("Gerar Relatório Completo", type="primary", use_container_width=True):
            with st.spinner("Gerando relatório..."):
