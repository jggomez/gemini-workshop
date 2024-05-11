from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from rich.console import Console
from dotenv import load_dotenv
from sentence_transformers import util

load_dotenv()

console = Console()


if __name__ == "__main__":
    google_api_key = str(os.getenv("GOOGLE_API_KEY"))

    contents = [
        "What is a Hackathon?",
        "Software developers should do a program for solving a problem.",
        "I'm really into play soccer"]

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=google_api_key)
    embeddings_vector = embeddings.embed_documents(contents)

    console.print(len(embeddings_vector))

    console.print(embeddings[0])

    for embedding in embeddings_vector:
        console.print(str(embedding)[:50], '... TRIMMED ...')

    cosine_scores = util.cos_sim(embeddings, embeddings)

    pairs = []

    for i in range(len(cosine_scores)-1):
        for j in range(i+1, len(cosine_scores)):
            pairs.append({'index': [i, j], "score": cosine_scores[i][j]})

        pairs = sorted(pairs, key=lambda x: x["score"], reverse=True)

    for pair in pairs[0:5]:
        i, j = pair['index']
        console.print(f"{contents[i]} -- {contents[j]} -> Score: {pair['score']}")
