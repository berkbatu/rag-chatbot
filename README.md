# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that allows you to upload your own documents and chat with them using Pinecone for vector storage and OpenAI for the language model.

## Features

- Upload and process documents in various formats (PDF, TXT, CSV, Markdown)
- Store document embeddings in Pinecone vector database
- Chat with your documents using OpenAI's language models
- Organize documents in namespaces
- View source documents for each response
- Modern and intuitive Streamlit web interface

## Requirements

- Python 3.8+
- OpenAI API key
- Pinecone API key

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/rag-chatbot.git
cd rag-chatbot
```

2. Install the dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX_NAME=your_pinecone_index_name
```

## Usage

1. Run the application:
```bash
python main.py
```

2. Open your browser and go to `http://localhost:8501`

3. Enter your API keys in the sidebar (if not already in the `.env` file)

4. Upload your documents and process them

5. Start chatting with your documents!

## How It Works

1. **Document Processing**: Documents are loaded and split into chunks using LangChain's document loaders and text splitters.

2. **Vector Storage**: Document chunks are embedded using OpenAI's embeddings and stored in Pinecone.

3. **Retrieval**: When you ask a question, the system retrieves the most relevant document chunks from Pinecone.

4. **Generation**: OpenAI's language model generates a response based on the retrieved document chunks and the conversation history.

## Project Structure

- `main.py`: Main entry point for the application
- `app/app.py`: Streamlit web application
- `app/utils/document_processor.py`: Document processing utilities
- `app/utils/vector_store.py`: Vector store management utilities
- `app/utils/chatbot.py`: RAG chatbot implementation

## License

MIT 