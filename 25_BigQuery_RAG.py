import vertexai
from langchain.chains import RetrievalQA
from langchain.globals import set_debug
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_community import BigQueryVectorStore
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from rich import print


PROJECT_ID = "PROJECT_ID"
LOCATION = "us-central1"
DATASET = "rag_test"
TABLE = "softwarearchitecturetopics"

vertexai.init(project=PROJECT_ID, location=LOCATION)

embedding_model = VertexAIEmbeddings(
    model_name="text-embedding-004", project=PROJECT_ID
)


def ingestPDF(file):
    loader = PyPDFLoader(file)
    documents = loader.load()

    for document in documents:
        doc_md = document.metadata
        print(doc_md)
        document_name = doc_md["source"].split("/")[-1]
        print(document_name)
        document.metadata = {"document_name": document_name}

    print(f"# of documents loaded (pre-chunking) = {len(documents)}")
    return documents


def chunking(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    )
    doc_splits = text_splitter.split_documents(documents)

    for idx, split in enumerate(doc_splits):
        split.metadata["chunk"] = idx

    print(f"# of documents = {len(doc_splits)}")
    print(doc_splits[0].metadata)
    return doc_splits


def get_bigquery_vector_store():
    bq_store = BigQueryVectorStore(
        project_id=PROJECT_ID,
        location=LOCATION,
        dataset_name=DATASET,
        table_name=TABLE,
        embedding=embedding_model,
    )

    return bq_store.as_retriever()


def create_bigquery_vector_store(doc_splits):
    bq_store = BigQueryVectorStore(
        project_id=PROJECT_ID,
        location=LOCATION,
        dataset_name=DATASET,
        table_name=TABLE,
        embedding=embedding_model,
    )
    bq_store.add_documents(documents=doc_splits)
    langchain_retriever = bq_store.as_retriever()
    return langchain_retriever


def get_llm_response(query, langchain_retriever):
    llm = VertexAI(model_name="gemini-1.5-flash-002")

    retrieval_qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=langchain_retriever
    )
    response = retrieval_qa.invoke(query)
    print("\n################ Final Answer ################\n")
    print(response["result"])


def batch_search():
    bq_store = BigQueryVectorStore(
        project_id=PROJECT_ID,
        location=LOCATION,
        dataset_name=DATASET,
        table_name=TABLE,
        embedding=embedding_model,
    )

    results = bq_store.batch_search(
        queries=["What is the C4 model?",
                 "What is a container in C4 Model?"],
    )

    print(results)


def sync_feature_online_store():
    bq_store = BigQueryVectorStore(
        project_id=PROJECT_ID,
        location=LOCATION,
        dataset_name=DATASET,
        table_name=TABLE,
        embedding=embedding_model,
    )

    vertex_fs = bq_store.to_vertex_fs_vector_store()
    vertex_fs.sync_data()
    return vertex_fs


def feature_online_store_similarity_search(vertex_fs):
    print(vertex_fs.similarity_search("What is the C4 model?"))


def feature_online_store_retriever(vertex_fs):
    langchain_retriever = vertex_fs.as_retriever()
    results = langchain_retriever.invoke("What is C4 Model?")
    print(results)


def feature_online_store_cleaning_up(vertex_fs):
    vertex_fs.feature_view.delete()
    vertex_fs.online_store.delete()


if __name__ == "__main__":
    set_debug(True)
    documents = ingestPDF("resource/visualising-software-architecture.pdf")
    doc_splits = chunking(documents)
    langchain_retriever = create_bigquery_vector_store(doc_splits=doc_splits)
    langchain_retriever = get_bigquery_vector_store()
    get_llm_response(
        query="What is the C4 model?",
        langchain_retriever=langchain_retriever
    )
    batch_search()
    vertex_fs = sync_feature_online_store()
    feature_online_store_similarity_search(vertex_fs)
    feature_online_store_retriever(vertex_fs)
    feature_online_store_cleaning_up(vertex_fs)
