import google.generativeai as genai
import os
from rich.console import Console
from dotenv import load_dotenv

load_dotenv()

console = Console()

if __name__ == "__main__":
    google_api_key = str(os.getenv("GOOGLE_API_KEY"))
    genai.configure(api_key=google_api_key)

    ref_audio = genai.upload_file(path="data/audios/audio_example.mp3",
                                  display_name="audio_example")

    print(f"Uploaded file '{ref_audio.name}' as: {ref_audio.uri}")

    model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
    prompt = [
        "Provide a brief summary:", ref_audio]
    response = model.generate_content(prompt, stream=True)
    for chunk in response:
        console.print(chunk.text)
