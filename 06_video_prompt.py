import time
import google.generativeai as genai
import os
from rich.console import Console
from dotenv import load_dotenv

load_dotenv()

console = Console()


if __name__ == "__main__":
    google_api_key = str(os.getenv("GOOGLE_API_KEY"))
    genai.configure(api_key=google_api_key)

    video_path = "data/videos/video_example.mp4"
    video_file = genai.upload_file(path=video_path)

    while video_file.state.name == "PROCESSING":
        print('Waiting for video to be processed.')
        time.sleep(2)
        video_file = genai.get_file(video_file.name)

    console.log(f"File uploaded successfully {video_file.uri}")

    model = genai.GenerativeModel('models/gemini-1.5-flash')
    response = model.generate_content(
        ["Provide a brief summary over the video:", video_file],
        stream=True)

    for chunk in response:
        console.print(chunk.text)
