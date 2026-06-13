from w_get_search import get_search
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from bs4 import BeautifulSoup
import re
import os
os.getenv("USER_AGENT")



#Preprocess the text by removing HTML tags, extra spaces, and special characters
def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML tags
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces/newlines
    text = re.sub(r'[^\w\s,.!?]', '', text)  # Keep only words, spaces, and punctuation
    return text.strip()


#Function Driver
def store_data(query="eCadstar",num_results=2,searcher="duckDuckGo",isLink=0,CHROMA_PATH = "./chroma_web"):
    if(isLink):
        results=query
    else:
        results=get_search(query,searcher,num_results=num_results)  
    #WebBased scrapper
    loader = WebBaseLoader(results,continue_on_failure=True)
    loader.requests_per_second = 1
    data = loader.aload()

    # Apply cleaning to all loaded documents
    cleaned_docs = [clean_text(doc.page_content) for doc in data]

    # Define text chunking strategy
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Each chunk is 500 characters
        chunk_overlap=50  # Overlapping for better context
    )
    
    # Split the cleaned text
    chunked_docs = text_splitter.create_documents(cleaned_docs)

    #Store Documents in ChromaDB
    Chroma.from_documents(documents=chunked_docs, embedding=GPT4AllEmbeddings(), persist_directory=CHROMA_PATH)
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=GPT4AllEmbeddings())
    print("Number of documents stored:", vectorstore._collection.count())
    return results
