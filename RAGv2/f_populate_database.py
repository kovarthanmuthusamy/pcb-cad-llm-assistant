import argparse
import os
import shutil
from pathlib import Path
import glob

#from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

#from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_community.document_loaders import UnstructuredExcelLoader

CHROMA_PATH = "chroma"
DATA_PATH = "data"

def get_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings

def load_documents(DATA_PATH):
    doc_list=[]
    for file_name in os.listdir(DATA_PATH):
        file_path = Path(os.path.join(DATA_PATH, file_name))
        if os.path.isfile(file_path):  # Check if it's a file
            print(file_path)
            if(file_path.suffix==".pdf"):
                document_loader = PyPDFLoader(file_path)
                doc_list.append(document_loader.load())
            elif(file_path.suffix==".csv"):
                document_loader = CSVLoader(file_path)
                doc_list.append(document_loader.load()) 
            elif(file_path.suffix==".docx"):
                document_loader = Docx2txtLoader(file_path)
                doc_list.append(document_loader.load()) 
            elif(file_path.suffix==".xlsx"):
                document_loader = UnstructuredExcelLoader(file_path=file_path,mode="elements")
                doc_list.append(document_loader.load()) 
            elif(file_path.suffix==".pptx"):
                document_loader = UnstructuredPowerPointLoader(file_path)
                doc_list.append(document_loader.load()) 
            else:
                {}
    return [item for doc in doc_list for item in doc]


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        #db.persist()
    else:
        print("No new documents to add")


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database(CHROMA_PATH=CHROMA_PATH):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("Database Cleared...")

def clear_data(DATA_PATH=DATA_PATH):
    """Deletes all files in a folder but keeps the folder itself."""
    
    # Get all files in the folder
    files = glob.glob(os.path.join(DATA_PATH, "*"))  # Includes all files

    for file in files:
        if os.path.isfile(file):  # Check if it's a file (not a subfolder)
            os.remove(file)  # Delete file
            print(f"Deleted: {file}")

    print("Data Cleared...")



def write_chroma(CHROMA_PATH=CHROMA_PATH ,DATA_PATH=DATA_PATH):
    # Create (or update) the data store.
    documents = load_documents(DATA_PATH)
    chunks = split_documents(documents)
    add_to_chroma(chunks)
