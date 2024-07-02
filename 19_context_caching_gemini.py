import os
import time
import datetime
import google.generativeai as genai
from google.generativeai import caching
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

    cache_1 = caching.CachedContent.create(
        model='models/gemini-1.5-flash-001',
        display_name="video_example_id",
        system_instruction="You are an expert video analyzer",
        contents=[video_file],
        ttl=datetime.timedelta(minutes=5),
    )

    cache_2 = caching.CachedContent.create(
        model='models/gemini-1.5-flash-001',
        display_name="video_example_id_2",
        system_instruction="You are an expert in linguistics",
        contents=[video_file],
        ttl=datetime.timedelta(minutes=5),
    )

    for cache in caching.CachedContent.list():
        console.print(cache)

    model = genai.GenerativeModel.from_cached_content(cached_content=cache_1)
    response = model.generate_content(["Create a summary for this video"])
    console.print(response.usage_metadata)
    console.print(response.text)

    response = model.generate_content(["Create questions for this video"])
    console.print(response.usage_metadata)
    console.print(response.text)

    cache_2.update(ttl=datetime.timedelta(minutes=10))
    model = genai.GenerativeModel.from_cached_content(cached_content=cache_2)
    response = model.generate_content(["Get named entities for this video"])
    console.print(response.usage_metadata)
    console.print(response.text)

    cache_2.delete()
    cache_1.delete()
