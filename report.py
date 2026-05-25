from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import datetime
import numpy as np

def generate_pdf(filename, normal, pneumonia, result_text):

    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 18)
    c.drawString(180, 750, "AI Medical Imaging Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, 700, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    c.drawString(50, 670, "--------------------------------------------")

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 640, "Diagnosis Result:")

    c.setFont("Helvetica", 12)
    c.drawString(70, 615, result_text)

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 570, "Confidence Scores:")

    c.setFont("Helvetica", 12)
    c.drawString(70, 545, f"Normal: {normal*100:.2f}%")
    c.drawString(70, 525, f"Pneumonia: {pneumonia*100:.2f}%")

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 480, "AI Summary:")

    summary = ""
    if pneumonia > normal:
        summary = "AI model detected signs consistent with Pneumonia. Recommend clinical verification."
    else:
        summary = "No signs of pneumonia detected. Lungs appear normal based on AI analysis."

    c.setFont("Helvetica", 12)
    c.drawString(70, 455, summary)

    c.save()