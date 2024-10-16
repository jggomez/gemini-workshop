from google.cloud import aiplatform
from google.cloud.firestore_v1.vector import Vector
from google.cloud import firestore_v1
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from vertexai.vision_models import MultiModalEmbeddingModel
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
import gradio as gr


db = firestore_v1.Client(database="embeddings")
content_multimodal_collection = db.collection('ContentMultimodal')
content_text_collection = db.collection('ContentTexts')
url_media = "https://media/series/"


def get_recommend_videos(embeddings, search_videos=True):

    if search_videos:
        vector_query = content_multimodal_collection.find_nearest(
            vector_field="embeddings",
            query_vector=Vector(embeddings),
            distance_measure=DistanceMeasure.COSINE,
            limit=20
        )
    else:
        vector_query = content_text_collection.find_nearest(
            vector_field="embeddings",
            query_vector=Vector(embeddings),
            distance_measure=DistanceMeasure.COSINE,
            limit=20
        )

    docs = (vector_query.stream())
    videos = []
    videos_text = []
    for doc in docs:
        if (doc.to_dict()["text"] in videos_text):
            continue
        videos_text.append(doc.to_dict()["text"])
        videos.append({
            "text": doc.to_dict()["text"],
            "url": f"{url_media}{doc.to_dict()['id']}"
        })

    return videos[:10]


def search_videos(text):
    model_name = "multimodalembedding"
    model = MultiModalEmbeddingModel.from_pretrained(model_name)
    embeddings = model.get_embeddings(
        contextual_text=text,
    )
    videos = get_recommend_videos(embeddings.text_embedding, True)
    recommend_videos = []
    for video in videos:
        print(video["text"], video["url"])
        recommend_videos.extend([video["text"], video["url"]])

    return tuple(recommend_videos)


def search_texts(text):
    model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    TextEmbeddingInput(text, "SEMANTIC_SIMILARITY")
    kwargs = dict(
        output_dimensionality=768) if 768 else {}
    embeddings = model.get_embeddings(
        [TextEmbeddingInput(text, "SEMANTIC_SIMILARITY")], **kwargs)

    videos = get_recommend_videos(embeddings[0].values, False)
    recommend_texts = []
    for video in videos:
        print(video["text"], video["url"])
        recommend_texts.extend([video["text"], video["url"]])

    return tuple(recommend_texts)


def get_ui():
    ui_search_videos = []
    ui_search_texts = []
    for _ in range(10):
        ui_search_videos.extend([gr.Label(), gr.Video()])
        ui_search_texts.extend([gr.Label(), gr.Video()])

    return gr.TabbedInterface(
        [gr.Interface(
            fn=search_videos,
            inputs=[gr.Textbox(label="Search videos...", lines=3),],
            outputs=ui_search_videos,
            allow_flagging="never"),
         gr.Interface(
            fn=search_texts,
            inputs=[gr.Textbox(label="Search texts...", lines=3),],
            outputs=ui_search_texts,
            allow_flagging="never"),],
        ["Search Videos", "Search Texts",],
    )


if __name__ == "__main__":
    aiplatform.init(project='wordboxdev', location='us-central1')
    print("Starting server...")
    iface = get_ui().launch(server_name="0.0.0.0", server_port=8080)
