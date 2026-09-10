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

    # Set Together AI configuration
    together_api_key = os.getenv("TOGETHER_API_KEY")
    together_base_url = "https://api.together.ai/v1"

    # Use Together AI with LangChain (OpenAI-compatible interface)
    # reasoning_effort="low" + explicit max_tokens keep gpt-oss's internal
    # chain-of-thought from eating the whole token budget before it writes
    # the final answer (that's why the previous reasoning model came back blank).
    llm = ChatOpenAI(
        model="openai/gpt-oss-20b",
        temperature=0.3,
        max_tokens=512,
        openai_api_base=together_base_url,
        openai_api_key=together_api_key,
        model_kwargs={"reasoning_effort": "low"}
    )

    # Build the QA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

    # Run the query
    response = qa_chain.invoke({"query": query})
    result = response["result"]

    print(f"Question: {query}")
    print("Retrieved Chunks:")
    for i, doc in enumerate(response["source_documents"], 1):
        print(f"--- Chunk {i} ({doc.metadata.get('source', 'unknown')}) ---")
        print(doc.page_content)

    return result
