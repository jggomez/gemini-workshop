from google.cloud import aiplatform
from rich.console import Console
from dotenv import load_dotenv
from vertexai.vision_models import MultiModalEmbeddingModel, Video
from google.cloud import firestore_v1
from google.cloud.firestore_v1.vector import Vector

load_dotenv()

console = Console()


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


def process_list_with_batches(list_to_process, batch_size=50):
    """
    Processes a list with batches of a given size.

    Args:
        list_to_process (list): The list to process.
        batch_size (int, optional): The size of each batch. Defaults to 500.

    Returns:
        list: A list of batches.
    """

    batches = []
    for i in range(0, len(list_to_process), batch_size):
        batches.append(list_to_process[i:i + batch_size])
    return batches


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


def process_list_with_batches(list_to_process, batch_size=50):
    """
    Processes a list with batches of a given size.

    Args:
        list_to_process (list): The list to process.
        batch_size (int, optional): The size of each batch. Defaults to 500.

    Returns:
        list: A list of batches.
    """

    batches = []
    for i in range(0, len(list_to_process), batch_size):
        batches.append(list_to_process[i:i + batch_size])
    return batches


if __name__ == "__main__":
    aiplatform.init(project='wordboxdev', location='us-central1')

    rows = read_csv_and_get_values("southpark2.csv")

    model_name = "multimodalembedding"
    console.print(f"Using model {model_name}...")

    model = MultiModalEmbeddingModel.from_pretrained(model_name)
    video_segment_config = {
        "startOffsetSec": 0,
        "endOffsetSec": 30
    }

    batches = process_list_with_batches(rows)

    db = firestore_v1.Client(database="embeddings")
    batch_firestore = db.batch()
    content_text_collection = db.collection('ContentMultimodal')

    url_media = "https://media.wordbox.ai/series/"

    for batch in batches:
        console.print(f"Processing batch of {len(batch)} rows...")
        texts = [item[0] for item in batch]
        text_ids = [item[1] for item in batch]
        embeddings_to_save = []

        for text, text_id in zip(texts, text_ids):
            console.print(f"Processing text {text_id}...")
            download_video(f"{url_media}{text_id}", f"videos/{text_id}.mp4")
            video = Video.load_from_file(f"videos/{text_id}.mp4")

            embeddings = model.get_embeddings(
                video=video,
                contextual_text=text,
            )

            for video_embedding in embeddings.video_embeddings:
                embedding = video_embedding.embedding
                console.print(str(embedding)[:50], '... TRIMMED ...')

                ref_doc = content_text_collection.document()
                batch_firestore.set(ref_doc, {
                    'text': text,
                    'id': text_id,
                    'url': f"{url_media}{text_id}",
                    'embeddings': Vector(embedding)
                })

        batch_firestore.commit()
