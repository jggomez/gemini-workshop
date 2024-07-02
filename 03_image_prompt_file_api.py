import google.generativeai as genai
import os
from rich.console import Console
from dotenv import load_dotenv
import PIL.Image

load_dotenv()

console = Console()

if __name__ == "__main__":
    google_api_key = str(os.getenv("GOOGLE_API_KEY"))
    genai.configure(api_key=google_api_key)

    ref_image = genai.upload_file(path="data/images/animales_mexico.jpeg",
                                  display_name="animales_mexico")

    print(f"Uploaded file '{ref_image.name}' as: {ref_image.uri}")

    model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
    prompt = [
        "Write a nature blog with images and references with the following image:", ref_image]
    response = model.generate_content(prompt, stream=True)
    for chunk in response:
        console.print(chunk.text)
