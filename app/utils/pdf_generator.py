from fpdf import FPDF
import pandas as pd

def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
        
    replacements = {
        'ă': 'a', 'â': 'a', 'î': 'i', 'ș': 's', 'ț': 't',
        'Ă': 'A', 'Â': 'A', 'Î': 'I', 'Ș': 'S', 'Ț': 'T',
        '–': '-', '”': '"', '„': '"' 
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
        
    return text.encode('latin-1', 'replace').decode('latin-1')

class ExamPDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 15)
        self.cell(0, 10, clean_text('SmarTest - Subiect Examen IA'), border=False, align='C')
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Pagina {self.page_no()}', align='C')

def create_pdf(problem_type, requirement, matrix_data, filename="subiect.pdf"):
    pdf = ExamPDF()
    pdf.add_page()
    
    pdf.set_font("Helvetica", size=12)
    safe_type = clean_text(f"Tip Problema: {problem_type}")
    pdf.cell(0, 10, safe_type, ln=True)
    pdf.ln(5)

    safe_req = clean_text(requirement)
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 7, safe_req)
    pdf.ln(10)

    pdf.set_font("Courier", size=10)
    
    if not matrix_data or len(matrix_data) == 0:
        return bytes(pdf.output())

    col_width = pdf.epw / len(matrix_data[0])
    row_height = 10

    for row in matrix_data:
        for item in row:
            text = str(item).replace("(", "").replace(")", "")
            safe_cell = clean_text(text)
            pdf.cell(col_width, row_height, safe_cell, border=1, align='C')
        pdf.ln(row_height)

    pdf.ln(20)
    pdf.set_font("Helvetica", 'I', 10)
    pdf.cell(0, 10, clean_text("Spatiu pentru rezolvare:"), ln=True)
    pdf.rect(x=10, y=pdf.get_y(), w=190, h=100)

    return bytes(pdf.output())