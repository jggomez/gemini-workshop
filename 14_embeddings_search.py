from google.cloud import aiplatform
from rich.console import Console
from dotenv import load_dotenv
from google.cloud.firestore_v1.vector import Vector
from vertexai.language_models import TextEmbeddingModel
from google.cloud import firestore_v1
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure

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


def find_nearest_neighbors(texts, phrase_embeddings, name_file):
    count_results = 0

    db = firestore_v1.Client(database="embeddings")
    content_text_collection = db.collection('ContentTexts')

    for text, phrase_embedding in zip(texts, phrase_embeddings):
        # Requires vector index
        vector_query = content_text_collection.find_nearest(
            vector_field="embeddings",
            query_vector=Vector(phrase_embedding.values),
            distance_measure=DistanceMeasure.COSINE,
            limit=50
        )

        docs = (vector_query.stream())
        data = []
        data.append({
            "text": text
        })
        for doc in docs:
            data.append({
                "text": doc.to_dict()["text"]
            })
        create_csv(f"{name_file}_{count_results}.csv", data)
        count_results += 1


if __name__ == "__main__":
    aiplatform.init(project='wordboxdev', location='us-central1')

    phrases_test = read_csv_and_get_values("test-phrases.csv")
    #phrases_test_es = read_csv_and_get_values("phrases-test-es.csv")

    texts = [item[0] for item in phrases_test]
    #texts_es = [item[0] for item in phrases_test_es]

    model_name = "text-embedding-004"
    dimensionality: int = 768

    kwargs = dict(
        output_dimensionality=dimensionality) if dimensionality else {}
    model = TextEmbeddingModel.from_pretrained(model_name)

    phrase_embeddings = model.get_embeddings(texts, **kwargs)
    find_nearest_neighbors(texts, phrase_embeddings, "result_phrases_en")

    #phrase_embeddings = model.get_embeddings(texts_es, **kwargs)
    #find_nearest_neighbors(texts_es, phrase_embeddings, "result_phrases_es")
