from langchain.document_loaders import SeleniumURLLoader
from langchain.text_splitter import CharacterTextSplitter

def scrape_and_prepare(urls):
    """
    Scrapes content from the provided URLs and splits them into smaller chunks.
    Args:
        urls (list): List of URLs to scrape content from.

    Returns:
        list: List of document chunks.
    """
    # Load documents from URLs
    loader = SeleniumURLLoader(urls=urls)
    docs_not_split = loader.load()

    # Split documents into smaller chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(docs_not_split)

    return docs
