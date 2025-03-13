import os
from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
)


class DocumentProcessor:
    """Utility class for processing documents of various formats."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document processor.

        Args:
            chunk_size: The size of each text chunk
            chunk_overlap: The overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def load_document(self, file_path: str) -> List[Document]:
        """
        Load a document based on its file extension.

        Args:
            file_path: Path to the document

        Returns:
            List of Document objects
        """
        _, file_extension = os.path.splitext(file_path)

        if file_extension.lower() == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_extension.lower() == ".txt":
            loader = TextLoader(file_path)
        elif file_extension.lower() == ".csv":
            loader = CSVLoader(file_path)
        elif file_extension.lower() in [".md", ".markdown"]:
            loader = UnstructuredMarkdownLoader(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")

        return loader.load()

    def process_documents(self, file_paths: List[str]) -> List[Document]:
        """
        Process multiple documents and split them into chunks.

        Args:
            file_paths: List of paths to documents

        Returns:
            List of chunked Document objects
        """
        documents = []

        for file_path in file_paths:
            try:
                docs = self.load_document(file_path)
                documents.extend(docs)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        return self.text_splitter.split_documents(documents)
