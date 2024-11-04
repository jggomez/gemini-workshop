from google.cloud import aiplatform
from rich.console import Console
from dotenv import load_dotenv
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from google.cloud import firestore_v1
from google.cloud.firestore_v1.vector import Vector

load_dotenv()

console = Console()

# create a function read a csv and get values


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

# create a function for processing a list with batches of 500 rows


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


def save_to_pickle(file_path, data):
    """
    Saves data to a pickle file.

    Args:
        file_path (str): The path to the pickle file.
        data (any): The data to save.
    """

    import pickle

    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    aiplatform.init(project='wordboxdev', location='us-central1')

    rows = read_csv_and_get_values("office.csv")

    model_name = "text-multilingual-embedding-002"
    task = "SEMANTIC_SIMILARITY"
    dimensionality: int = 768
    console.print(f"Using model {model_name}...")
    console.print(f"Using task {task}...")
    console.print(f"Using dimensionality {dimensionality}...")

    model = TextEmbeddingModel.from_pretrained(model_name)
    kwargs = dict(
        output_dimensionality=dimensionality) if dimensionality else {}

    batches = process_list_with_batches(rows)

    db = firestore_v1.Client(database="embeddings")
    batch_firestore = db.batch()
    content_text_collection = db.collection('ContentEmbeddings')
    all_embeddings = []

    for batch in batches:
        console.print(f"Processing batch of {len(batch)} rows...")

        texts = [item[0] for item in batch]
        content_ids = [item[1] for item in batch]
        interest_ids = [item[2] for item in batch]
        content_difficulties = [item[3] for item in batch]

        inputs = [TextEmbeddingInput(text, task) for text in texts]
        embeddings = model.get_embeddings(inputs, **kwargs)

        for text, content_id, interest_id, content_difficulty, embedding in zip(texts, content_ids, interest_ids, content_difficulties, embeddings):
            console.print(str(embedding.values)[:50], '... TRIMMED ...')
            ref_doc = content_text_collection.document()
            batch_firestore.set(ref_doc, {
                'text': text,
                'contentid': content_id,
                'interestid': interest_id,
                'contentdifficulty': content_difficulty,
                'embeddingtext': Vector(embedding.values)
            })
            all_embeddings.append(embedding.values)

        batch_firestore.commit()

    save_to_pickle("embeddings_southpark.pkl", np.array(all_embeddings))
