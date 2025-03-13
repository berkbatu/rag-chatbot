import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

load_dotenv()


class RAGChatbot:
    """Utility class for the RAG chatbot."""

    def __init__(
        self, vector_store, model_name: str = "gpt-3.5-turbo", api_key: str = None
    ):
        """
        Initialize the RAG chatbot.

        Args:
            vector_store: The vector store to use for retrieval
            model_name: The OpenAI model to use
            api_key: OpenAI API key (optional, will use env var if not provided)
        """
        # Use provided API key or fall back to environment variable
        self.openai_api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.openai_api_key:
            raise ValueError("Missing OpenAI API key. Please check your .env file.")

        self.vector_store = vector_store
        self.model_name = model_name

        # Initialize the language model
        self.llm = ChatOpenAI(
            model_name=model_name, temperature=0.7, api_key=self.openai_api_key
        )

        # Initialize conversation memory with explicit output_key
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",  # Explicitly set the output key to "answer"
        )

        # Initialize the conversational chain
        self.chain = self._create_chain()

    def _create_chain(self):
        """
        Create the conversational retrieval chain.

        Returns:
            ConversationalRetrievalChain
        """
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 4}),
            memory=self.memory,
            return_source_documents=True,
        )

    def chat(self, query: str, namespace: Optional[str] = None):
        """
        Chat with the RAG chatbot.

        Args:
            query: The user's query
            namespace: Optional namespace to search in

        Returns:
            Dict containing the response and source documents
        """
        # If namespace is provided, update the retriever
        if namespace:
            self.chain.retriever = self.vector_store.as_retriever(
                search_kwargs={"k": 4, "namespace": namespace}
            )

        # Get the response
        try:
            response = self.chain({"question": query})

            return {
                "answer": response["answer"],
                "source_documents": response["source_documents"],
            }
        except Exception as e:
            error_message = f"Error generating response: {str(e)}"
            return {
                "answer": error_message,
                "source_documents": [],
            }

    def reset_conversation(self):
        """Reset the conversation history."""
        self.memory.clear()
