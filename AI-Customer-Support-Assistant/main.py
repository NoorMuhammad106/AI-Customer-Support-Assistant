import os
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain import OpenAI, PromptTemplate
from scraper import scrape_and_prepare
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ACTIVELOOP_TOKEN = os.getenv("ACTIVELOOP_TOKEN")
ACTIVELOOP_ORG_ID = os.getenv("ACTIVELOOP_ORG_ID")
DATASET_NAME = os.getenv("DATASET_NAME")

# URLs to scrape
urls = [
    "https://beebom.com/what-is-nft-explained/",
    "https://beebom.com/how-delete-spotify-account/",
    "https://beebom.com/how-download-gif-twitter/",
    "https://beebom.com/how-use-chatgpt-linux-terminal/",
    "https://beebom.com/how-save-instagram-story-with-music/",
    "https://beebom.com/how-install-pip-windows/",
    "https://beebom.com/how-check-disk-usage-linux/",
]

# Create vector store
def create_vector_store(urls):
    """
    Scrapes content, generates embeddings, and stores them in DeepLake.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    dataset_path = f"hub://{ACTIVELOOP_ORG_ID}/{DATASET_NAME}"

    # Scrape and prepare documents
    docs = scrape_and_prepare(urls)

    # Create Deep Lake vector store
    db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)
    db.add_documents(docs)

    return db

# Initialize DeepLake
st.title("AI Customer Support Assistant")
st.sidebar.header("Configuration")
with st.sidebar:
    if st.button("Initialize Vector Store"):
        with st.spinner("Setting up the vector store..."):
            db = create_vector_store(urls)
            st.success("Vector store initialized successfully!")

# User Query
query = st.text_input("Ask your question:")
if query:
    st.write("Processing your query...")
    # Retrieve relevant chunks
    db = DeepLake(dataset_path=f"hub://{ACTIVELOOP_ORG_ID}/{DATASET_NAME}")
    docs = db.similarity_search(query)
    retrieved_chunks = [doc.page_content for doc in docs]

    # Format the prompt
    chunks_formatted = "\n\n".join(retrieved_chunks)
    template = """You are an exceptional customer support chatbot that gently answers questions.

    You know the following context information:

    {chunks_formatted}

    Answer the following question from a customer. Use only information from the previous context. Do not invent.

    Question: {query}

    Answer:"""
    prompt = PromptTemplate(
        input_variables=["chunks_formatted", "query"], template=template
    )
    prompt_formatted = prompt.format(chunks_formatted=chunks_formatted, query=query)

    # Generate answer
    llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)
    answer = llm(prompt_formatted)

    # Display the answer
    st.success("Answer:")
    st.write(answer)
