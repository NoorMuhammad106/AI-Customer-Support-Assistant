# AI Customer Support Assistant

An AI-powered customer support assistant that uses LangChain, OpenAI, and DeepLake to answer user queries by leveraging context from a set of technical articles.

## Features
- Scrapes and processes technical articles
- Stores embeddings in a DeepLake vector store
- Retrieves relevant chunks based on user queries
- Generates accurate answers using OpenAI GPT-3
- Interactive Streamlit-based user interface

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/username/AI-Customer-Support-Assistant.git
   cd AI-Customer-Support-Assistant
2. Install dependencies:
   ``` 
   pip install -r requirements.txt
3. Set up environment variables: Create a .env file and add your API keys:
   ``` 
   OPENAI_API_KEY=your_openai_api_key
    ACTIVELOOP_TOKEN=your_activeloop_token
    ACTIVELOOP_ORG_ID=your_activeloop_org_id
    DATASET_NAME=langchain_course_customer_support
    
4. Run the Streamlit app:
   ```bash 
   streamlit run main.py

## Usage
Initialize the vector store using the sidebar button.
Enter your query in the text box and press Enter.
View the AI-generated answer based on the scraped knowledge base.

## License
This project is licensed under the MIT License.