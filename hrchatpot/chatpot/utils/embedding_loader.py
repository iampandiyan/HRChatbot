import fitz  # PyMuPDF
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import os
import zipfile
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
import logging


def load_and_embed_policies(pdf_folder="chatpot/policies"):
    documents = []

    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            path = os.path.join(pdf_folder, filename)
            doc = fitz.open(path)
            text = ""
            for page in doc:
                text += page.get_text()
            documents.append(Document(page_content=text, metadata={"source": filename}))

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    vectorstore.save_local("chatpot/vectorstore")


def process_policy_documents(zip_file_path, vectorstore_path="chatpot/vectorstore"):
    #extract_path = os.path.splitext(zip_file_path)[0]

    #logging.info(f"Extracting {zip_file_path} to {extract_path}")
    #logger = logging.getLogger("TESTING")
    #logger.debug(f"Extracting {zip_file_path} to {extract_path}")
    # Unzip the file
    #with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    #    zip_ref.extractall(extract_path)

    # Load documents
    documents = []
    for filename in os.listdir(zip_file_path):
        if filename.endswith(".pdf"):
            path = os.path.join(zip_file_path, filename)
            doc = fitz.open(path)
            text = ""
            for page in doc:
                text += page.get_text()
            documents.append(Document(page_content=text, metadata={"source": filename}))

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    vectorstore.save_local("chatpot/vectorstore")

    return True

def load_and_save_embeddings(folder_path):
    loader = DirectoryLoader(folder_path, glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(docs, embeddings)
    vectordb.save_local("chatpot/vectorstore")
