import csv
from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(48, 43, 99)
        self.cell(0, 10, "Retail Sales Dataset Report", border=False, align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "I", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 6, "AI Data Analysis Assistant - Dataset Summary", border=False, align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

def generate_dataset_pdf(csv_filepath, pdf_filepath):
    pdf = PDF(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_font("Helvetica", size=9)

    # Read CSV
    with open(csv_filepath, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        print("CSV is empty!")
        return

    headers = rows[0]
    data = rows[1:]

    # Define column widths (A4 printable width ~ 190mm)
    col_widths = [22, 42, 38, 25, 18, 45]

    # Render Header Row
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(99, 102, 241)
    pdf.set_text_color(255, 255, 255)

    for i, col in enumerate(headers):
        pdf.cell(col_widths[i], 8, str(col), border=1, align="C", fill=True)
    pdf.ln()

    # Render Data Rows
    pdf.set_font("Helvetica", size=8)
    pdf.set_text_color(30, 30, 30)

    for fill_toggle, row in enumerate(data):
        fill = (fill_toggle % 2 == 1)
        if fill:
            pdf.set_fill_color(240, 242, 255)
        else:
            pdf.set_fill_color(255, 255, 255)

        for i, val in enumerate(row):
            align = "C" if i in [0, 3, 4] else "L"
            pdf.cell(col_widths[i], 6, str(val), border=1, align=align, fill=True)
        pdf.ln()

    pdf.output(pdf_filepath)
    print(f"Successfully created PDF: {pdf_filepath}")

if __name__ == "__main__":
    generate_dataset_pdf("dataset.csv", "dataset.pdf")
