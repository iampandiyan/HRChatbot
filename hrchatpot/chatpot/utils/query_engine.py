from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os

def answer_query(query):
    # Load the embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load the FAISS vector store
    vectordb = FAISS.load_local("chatpot/vectorstore", embeddings,allow_dangerous_deserialization=True)

    # Create a retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # Set Mistral API configuration (replace with your actual key and endpoint)
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    mistral_base_url = "https://api.mistral.ai/v1"  # or your custom deployment

    # Use Mistral with LangChain (OpenAI-compatible interface)
    llm = ChatOpenAI(
        model="mistral-large-latest",
        temperature=0.3,
        openai_api_base=mistral_base_url,
        openai_api_key=mistral_api_key
    )

    # Build the QA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    
    # Run the query
    result = qa_chain.run(query)
    return result
