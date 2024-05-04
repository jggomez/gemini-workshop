import google.generativeai as genai
import os
from rich.console import Console
from dotenv import load_dotenv
from PyPDF2 import PdfReader

load_dotenv()

console = Console()

if __name__ == "__main__":
    google_api_key = str(os.getenv("GOOGLE_API_KEY"))
    genai.configure(api_key=google_api_key)

    path_file = "data/files/example.pdf"
    text = ""
    with open(path_file, "rb") as pdf_file:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text()

    text = text[500:12000]

    console.print(text)

    model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
    prompt = f"Provide a brief summary of this text: {text}"
    response = model.generate_content(prompt, stream=True)
    for chunk in response:
        console.print(chunk.text)
