import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()


class VectorStoreManager:
    """Utility class for managing the Pinecone vector store."""

    def __init__(
        self,
        openai_api_key=None,
        pinecone_api_key=None,
        pinecone_environment=None,
        pinecone_index_name=None,
    ):
        """Initialize the vector store manager with API keys from environment variables or parameters."""
        # Use provided keys or fall back to environment variables
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.pinecone_api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
        self.pinecone_environment = pinecone_environment or os.getenv(
            "PINECONE_ENVIRONMENT"
        )
        self.pinecone_index_name = pinecone_index_name or os.getenv(
            "PINECONE_INDEX_NAME"
        )

        if not all(
            [
                self.openai_api_key,
                self.pinecone_api_key,
                self.pinecone_index_name,
            ]
        ):
            raise ValueError(
                "Missing required environment variables. Please check your .env file."
            )

        # Use text-embedding-3-large which produces 3072-dimension embeddings
        self.embeddings = OpenAIEmbeddings(
            api_key=self.openai_api_key, model="text-embedding-3-large"
        )
        # Initialize Pinecone client with API key
        self.pc = Pinecone(api_key=self.pinecone_api_key)

        # Get the index directly
        try:
            self.index = self.pc.Index(self.pinecone_index_name)
            print(
                f"Successfully connected to Pinecone index: {self.pinecone_index_name}"
            )
        except Exception as e:
            print(f"Error connecting to Pinecone index: {e}")
            self.index = None

    def initialize_index(self, dimension: int = 3072):
        """
        Initialize the Pinecone index if it doesn't exist.

        Args:
            dimension: Dimension of the embeddings (3072 for your existing index)
        """
        # Check if index exists
        try:
            existing_indexes = [index.name for index in self.pc.list_indexes()]

            if self.pinecone_index_name not in existing_indexes:
                # Create the index
                self.pc.create_index(
                    name=self.pinecone_index_name,
                    dimension=dimension,
                    metric="cosine",
                )
                print(f"Created new Pinecone index: {self.pinecone_index_name}")
                # Connect to the newly created index
                self.index = self.pc.Index(self.pinecone_index_name)
            else:
                print(f"Using existing Pinecone index: {self.pinecone_index_name}")
                # Make sure we're connected to the index
                if self.index is None:
                    self.index = self.pc.Index(self.pinecone_index_name)
        except Exception as e:
            raise ValueError(f"Error initializing Pinecone index: {e}")

    def get_vector_store(self):
        """
        Get the Pinecone vector store.

        Returns:
            PineconeVectorStore instance
        """
        if self.index is None:
            raise ValueError(
                "Pinecone index not initialized. Call initialize_index() first."
            )

        # Create the vector store using the index directly
        return PineconeVectorStore(
            index=self.index,
            embedding=self.embeddings,
        )

    def add_documents(self, documents: List[Document], namespace: Optional[str] = None):
        """
        Add documents to the vector store.

        Args:
            documents: List of Document objects to add
            namespace: Optional namespace for the documents
        """
        vector_store = self.get_vector_store()

        # Add documents to the vector store
        vector_store.add_documents(documents=documents, namespace=namespace)

        print(f"Added {len(documents)} document chunks to Pinecone")

    def similarity_search(
        self, query: str, k: int = 4, namespace: Optional[str] = None
    ):
        """
        Perform a similarity search in the vector store.

        Args:
            query: The query string
            k: Number of results to return
            namespace: Optional namespace to search in

        Returns:
            List of Document objects
        """
        vector_store = self.get_vector_store()

        return vector_store.similarity_search(query=query, k=k, namespace=namespace)
