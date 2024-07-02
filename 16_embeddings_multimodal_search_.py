from google.cloud import aiplatform
from rich.console import Console
from dotenv import load_dotenv
from google.cloud.firestore_v1.vector import Vector
from google.cloud import firestore_v1
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from vertexai.vision_models import MultiModalEmbeddingModel, Video

load_dotenv()

console = Console()

db = firestore_v1.Client(database="embeddings")
content_multimodal_collection = db.collection('ContentMultimodal')
url_media = "https://media.wordbox.ai/series/"


def read_csv_and_get_values(file_path):
    """
    Reads a CSV file and returns a list of dictionaries, where each dictionary represents a row in the CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        list: A list of dictionaries, where each dictionary represents a row in the CSV file.
    """

    import csv

    with open(file_path, 'r', encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)

    rows.remove(rows[0])

    return rows


def create_csv(file_path, data):
    """
    Creates a CSV file from a list of dictionaries.

    Args:
        file_path (str): The path to the CSV file.
        data (list): A list of dictionaries, where each dictionary represents a row in the CSV file.
    """

    import csv

    with open(file_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)


def find_nearest_neighbors(video_id, text, embeddings):
    vector_query = content_multimodal_collection.find_nearest(
        vector_field="embeddings",
        query_vector=Vector(embeddings),
        distance_measure=DistanceMeasure.COSINE,
        limit=20
    )

    docs = (vector_query.stream())
    data = []
    data.append({
        "text": text,
        "url": f"{url_media}{video_id}"
    })
    for doc in docs:
        data.append({
            "text": doc.to_dict()["text"],
            "url": doc.to_dict()["url"]
        })

    return data


def download_video(video_url, video_path):
    """
    Downloads a video from a URL and saves it to a file.

    Args:
        video_url (str): The URL of the video.
        video_path (str): The path to the file where the video will be saved.
    """

    import requests

    response = requests.get(video_url, stream=True)
    with open(video_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


if __name__ == "__main__":
    aiplatform.init(project='wordboxdev', location='us-central1')

    videos = read_csv_and_get_values("test_multimodal.csv")

    video_phrases = [item[0] for item in videos]
    video_ids = [item[1] for item in videos]

    model_name = "multimodalembedding"
    model = MultiModalEmbeddingModel.from_pretrained(model_name)
    video_segment_config = {
        "startOffsetSec": 0,
        "endOffsetSec": 30
    }
    count_results = 0

    for video_id, video_phrase in zip(video_ids, video_phrases):
        download_video(f"{url_media}{video_id}", f"videos/{video_id}.mp4")
        video = Video.load_from_file(f"videos/{video_id}.mp4")
        embeddings = model.get_embeddings(
            video=video,
            contextual_text=video_phrase,
        )
        embedding = embeddings.video_embeddings[0].embedding
        console.print(str(embedding)[:50], '... TRIMMED ...')
        data = find_nearest_neighbors(
            video_id, video_phrase, embedding)
        console.print(data)
        create_csv(f"result_multimodal__{count_results}.csv", data)
        count_results += 1
