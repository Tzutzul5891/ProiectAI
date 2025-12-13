from fpdf import FPDF
import pandas as pd
from typing import Any, Iterable

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
    def __init__(self, header_title: str | None = None, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._header_title = header_title or "SmarTest - Subiect Examen IA"

    def header(self):
        self.set_font('Helvetica', 'B', 15)
        self.cell(0, 10, clean_text(self._header_title), border=False, align='C')
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Pagina {self.page_no()}', align='C')

def _draw_hanoi_state(pdf: FPDF, hanoi_state: dict[int, list[int]] | None) -> None:
    if not hanoi_state:
        return

    pdf.set_font("Courier", size=9)

    num_pegs = len(hanoi_state)
    peg_names = ["A", "B", "C", "D"][:num_pegs]

    # Draw peg labels
    start_y = pdf.get_y()
    peg_spacing = 45
    start_x = 15

    for peg_idx in range(num_pegs):
        x_pos = start_x + peg_idx * peg_spacing
        pdf.set_xy(x_pos, start_y)
        pdf.cell(40, 5, clean_text(f"Tija {peg_names[peg_idx]}"), align="C")

    pdf.ln(7)

    # Draw disks from bottom to top; align all pegs by max height
    max_height = max((len(hanoi_state[i]) for i in range(num_pegs)), default=1)
    if max_height == 0:
        max_height = 1

    for level in range(max_height, 0, -1):
        current_y = pdf.get_y()

        for peg_idx in range(num_pegs):
            x_pos = start_x + peg_idx * peg_spacing
            disks = hanoi_state.get(peg_idx, [])

            if len(disks) >= level:
                disk_size = disks[level - 1]
                disk_repr = "=" * (disk_size * 2)
            else:
                disk_repr = "|"

            pdf.set_xy(x_pos, current_y)
            pdf.cell(40, 5, clean_text(disk_repr), align="C")

        pdf.ln(5)

    # Draw base for all pegs
    current_y = pdf.get_y()
    for peg_idx in range(num_pegs):
        x_pos = start_x + peg_idx * peg_spacing
        pdf.set_xy(x_pos, current_y)
        pdf.cell(40, 5, clean_text("#" * 12), align="C")

    pdf.ln(15)


def _draw_matrix_grid(pdf: FPDF, matrix_data: Any) -> None:
    pdf.set_font("Courier", size=10)

    if not matrix_data:
        return

    # Expect list-of-lists-ish; keep defensive to avoid crashes.
    try:
        rows = list(matrix_data)
        if not rows:
            return
        first_row = list(rows[0])
        if not first_row:
            return
    except Exception:
        pdf.multi_cell(0, 6, clean_text(str(matrix_data)))
        pdf.ln(5)
        return

    effective_width = pdf.w - 2 * pdf.l_margin
    col_width = effective_width / max(1, len(first_row))
    row_height = 10

    for row in rows:
        try:
            items = list(row)
        except Exception:
            items = [row]
        for item in items:
            text = str(item).replace("(", "").replace(")", "")
            pdf.cell(col_width, row_height, clean_text(text), border=1, align="C")
        pdf.ln(row_height)

    pdf.ln(10)


def create_pdf(problem_type, requirement, matrix_data, filename="subiect.pdf", hanoi_state=None):
    """Backwards-compatible single-question PDF generator."""

    pdf = ExamPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 10, clean_text(f"Tip Problema: {problem_type}"), ln=True)
    pdf.ln(5)

    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 7, clean_text(requirement))
    pdf.ln(10)

    if hanoi_state is not None:
        _draw_hanoi_state(pdf, hanoi_state)
    else:
        _draw_matrix_grid(pdf, matrix_data)

    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(0, 10, clean_text("Spatiu pentru rezolvare:"), ln=True)
    pdf.rect(x=10, y=pdf.get_y(), w=190, h=100)

    return pdf.output(dest="S").encode("latin-1")


def _extract_question_fields(question: Any) -> dict[str, Any]:
    """Support both `Question` objects and dict payloads."""

    if isinstance(question, dict):
        return {
            "id": question.get("id", ""),
            "type": question.get("type", ""),
            "prompt_text": question.get("prompt_text", ""),
            "data": question.get("data"),
            "correct_answer": question.get("correct_answer"),
            "correct_explanation": question.get("correct_explanation", ""),
            "metadata": question.get("metadata") or {},
        }

    return {
        "id": getattr(question, "id", ""),
        "type": getattr(question, "type", ""),
        "prompt_text": getattr(question, "prompt_text", ""),
        "data": getattr(question, "data", None),
        "correct_answer": getattr(question, "correct_answer", None),
        "correct_explanation": getattr(question, "correct_explanation", ""),
        "metadata": getattr(question, "metadata", {}) or {},
    }


def create_test_pdf(
    questions: Iterable[Any],
    *,
    include_answer_key: bool = False,
    header_title: str | None = None,
) -> bytes:
    """Create a paginated PDF for a list of questions.

    - When `include_answer_key=False`, the PDF contains only prompts + data and space to solve.
    - When `include_answer_key=True`, the PDF contains solutions + explanations (no blank space).
    """

    questions_list = list(questions)
    title = header_title or ("SmarTest - Answer Key" if include_answer_key else "SmarTest - Test IA")

    pdf = ExamPDF(header_title=title)

    total = max(1, len(questions_list))
    for idx, q in enumerate(questions_list, start=1):
        fields = _extract_question_fields(q)
        metadata = fields.get("metadata") or {}

        pdf.add_page()

        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, clean_text(f"Intrebarea {idx}/{total}"), ln=True)
        pdf.set_font("Helvetica", size=11)
        type_label = metadata.get("ui_label") or fields.get("type") or ""
        if type_label:
            pdf.cell(0, 7, clean_text(f"Tip: {type_label}"), ln=True)

        pdf.ln(2)
        prompt_text = fields.get("prompt_text") or ""
        pdf.multi_cell(0, 7, clean_text(prompt_text))
        pdf.ln(5)

        hanoi_state = metadata.get("initial_state")
        if hanoi_state is not None:
            _draw_hanoi_state(pdf, hanoi_state)
        else:
            _draw_matrix_grid(pdf, fields.get("data"))

        if include_answer_key:
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, clean_text("Raspuns corect:"), ln=True)
            pdf.set_font("Courier", size=9)
            pdf.multi_cell(0, 5, clean_text(str(fields.get("correct_answer"))))

            pdf.ln(1)
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, clean_text("Explicatie:"), ln=True)
            pdf.set_font("Helvetica", size=10)
            pdf.multi_cell(0, 5, clean_text(str(fields.get("correct_explanation") or "")))
        else:
            pdf.set_font("Helvetica", "I", 10)
            pdf.cell(0, 10, clean_text("Spatiu pentru rezolvare:"), ln=True)
            pdf.rect(x=10, y=pdf.get_y(), w=190, h=90)

    return pdf.output(dest="S").encode("latin-1")
