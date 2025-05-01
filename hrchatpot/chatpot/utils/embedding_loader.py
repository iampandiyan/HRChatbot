import fitz  # PyMuPDF
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import os

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
