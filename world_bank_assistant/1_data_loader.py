# general utilities
from os import environ

# text processing utilities
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

# vector storage
import pinecone

# langchain dependencies
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.schema import Document
from langchain.vectorstores import Pinecone

# model dependencies
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import SentenceTransformerEmbeddings


PINECONE_INDEX = "world-bank-indicators"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def get_embedding_tool():
    #return OpenAIEmbeddings(openai_api_key=environ.get("OPENAI_API_KEY"))
    return SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)


def initialize_pinecone():
    pinecone.init(api_key=environ.get("PINECONE_API_KEY"), environment="gcp-starter")
    

def load_embeddings(texts):
    embeddings_api = get_embedding_tool()
    vector_database = Pinecone.from_texts(
        texts, embeddings_api, index_name=PINECONE_INDEX
    )
    return vector_database


def load_csv_data(file_path: str):
    loader = CSVLoader(
        file_path=file_path, encoding="utf-8",
        csv_args={'delimiter': ','}
    )
    return loader.load()


def split_docs(documents, chunk_size=1000, chunk_overlap=150):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs



if __name__ == "__main__":
    data_path = "./data/world_bank/formatted_Series_metadata.csv"
    input_texts = split_docs(load_csv_data(data_path))

    #Create an instance of the Pinecone class for the specified index, using the document splits, embeddings, and index name.
    index = load_embeddings(input_texts)