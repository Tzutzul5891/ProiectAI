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

def create_pdf(problem_type, requirement, matrix_data, filename="subiect.pdf", hanoi_state=None):
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

    # Check if this is Tower of Hanoi
    if hanoi_state is not None:
        # Draw Tower of Hanoi visualization
        pdf.set_font("Courier", size=9)
        
        num_pegs = len(hanoi_state)
        max_disks = max([max(hanoi_state[i]) if hanoi_state[i] else 0 for i in range(num_pegs)])
        
        peg_names = ["A", "B", "C", "D"][:num_pegs]
        
        # Draw peg labels
        start_y = pdf.get_y()
        peg_spacing = 45
        start_x = 15
        
        for peg_idx in range(num_pegs):
            x_pos = start_x + peg_idx * peg_spacing
            pdf.set_xy(x_pos, start_y)
            pdf.cell(40, 5, f"Tija {peg_names[peg_idx]}", align='C')
        
        pdf.ln(7)
        
        # Draw disks from bottom to top (reversed order)
        # We need to draw all pegs aligned, so find max height
        max_height = max([len(hanoi_state[i]) for i in range(num_pegs)])
        if max_height == 0:
            max_height = 1
        
        # Draw each level from bottom (disk position) to top
        for level in range(max_height, 0, -1):
            current_y = pdf.get_y()
            
            for peg_idx in range(num_pegs):
                x_pos = start_x + peg_idx * peg_spacing
                disks = hanoi_state[peg_idx]
                
                # Check if this peg has a disk at this level
                if len(disks) >= level:
                    # Get the disk size at this position (0 is bottom, -1 is top)
                    disk_size = disks[level - 1]
                    disk_repr = "=" * (disk_size * 2)
                else:
                    # No disk at this level, show peg rod
                    disk_repr = "|"
                
                pdf.set_xy(x_pos, current_y)
                pdf.cell(40, 5, disk_repr, align='C')
            
            pdf.ln(5)
        
        # Draw base for all pegs
        current_y = pdf.get_y()
        for peg_idx in range(num_pegs):
            x_pos = start_x + peg_idx * peg_spacing
            pdf.set_xy(x_pos, current_y)
            pdf.cell(40, 5, "#" * 12, align='C')
        
        pdf.ln(15)
    else:
        # Original matrix-based visualization
        pdf.set_font("Courier", size=10)
        
        if not matrix_data or len(matrix_data) == 0:
            val = pdf.output(dest='S')
            return bytes(val) if isinstance(val, (bytes, bytearray)) else val.encode('latin-1')

        # Calculate effective page width (page width minus margins)
        effective_width = pdf.w - 2 * pdf.l_margin
        col_width = effective_width / len(matrix_data[0])
        row_height = 10

        for row in matrix_data:
            for item in row:
                text = str(item).replace("(", "").replace(")", "")
                safe_cell = clean_text(text)
                pdf.cell(col_width, row_height, safe_cell, border=1, align='C')
            pdf.ln(row_height)

        pdf.ln(20)
    
    # Space for solution
    pdf.set_font("Helvetica", 'I', 10)
    pdf.cell(0, 10, clean_text("Spatiu pentru rezolvare:"), ln=True)
    pdf.rect(x=10, y=pdf.get_y(), w=190, h=100)

    val = pdf.output(dest='S')
    return bytes(val) if isinstance(val, (bytes, bytearray)) else val.encode('latin-1')