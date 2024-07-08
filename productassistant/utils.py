import os
import PyPDF2
import json
import traceback
import pinecone

from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

def read_file():
    try:
        loader = PyPDFDirectoryLoader("pdfs")
        data=loader.load()            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        text_chunks = text_splitter.split_documents(data)
        return text_chunks
    
    except Exception as e:
        raise Exception("Error reading the pdf file")
    

def create_index_in_pinecone(index_name):
    
    PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    if index_name not in pc.list_indexes().names():
        pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-1'
        ) 
    )