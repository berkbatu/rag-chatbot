import os
import streamlit as st
import tempfile
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from utils.document_processor import DocumentProcessor
from utils.vector_store import VectorStoreManager
from utils.chatbot import RAGChatbot

# Load environment variables
load_dotenv()

# Check if running in Streamlit Cloud
is_streamlit_cloud = os.environ.get("IS_STREAMLIT_CLOUD", False)

# Set page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide" if not is_streamlit_cloud else "centered",
    initial_sidebar_state="expanded" if not is_streamlit_cloud else "collapsed",
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chatbot" not in st.session_state:
    st.session_state.chatbot = None

if "vector_store_manager" not in st.session_state:
    st.session_state.vector_store_manager = None

if "namespaces" not in st.session_state:
    st.session_state.namespaces = []

if "current_namespace" not in st.session_state:
    st.session_state.current_namespace = None

if "initialization_attempted" not in st.session_state:
    st.session_state.initialization_attempted = False

# Pre-fill API keys from environment variables or Streamlit secrets
if "openai_api_key" not in st.session_state:
    # Try to get from Streamlit secrets first, then fall back to env vars
    st.session_state.openai_api_key = (
        st.secrets.get("OPENAI_API_KEY", "")
        if hasattr(st, "secrets")
        else os.getenv("OPENAI_API_KEY", "")
    )

if "pinecone_api_key" not in st.session_state:
    st.session_state.pinecone_api_key = (
        st.secrets.get("PINECONE_API_KEY", "")
        if hasattr(st, "secrets")
        else os.getenv("PINECONE_API_KEY", "")
    )

if "pinecone_environment" not in st.session_state:
    st.session_state.pinecone_environment = (
        st.secrets.get("PINECONE_ENVIRONMENT", "")
        if hasattr(st, "secrets")
        else os.getenv("PINECONE_ENVIRONMENT", "")
    )

if "pinecone_index_name" not in st.session_state:
    st.session_state.pinecone_index_name = (
        st.secrets.get("PINECONE_INDEX_NAME", "")
        if hasattr(st, "secrets")
        else os.getenv("PINECONE_INDEX_NAME", "")
    )


# Function to initialize the vector store manager
def initialize_vector_store():
    try:
        # Create the vector store manager with current session state values
        vector_store_manager = VectorStoreManager(
            openai_api_key=st.session_state.openai_api_key,
            pinecone_api_key=st.session_state.pinecone_api_key,
            pinecone_environment=st.session_state.pinecone_environment,
            pinecone_index_name=st.session_state.pinecone_index_name,
        )

        # Initialize the index
        try:
            vector_store_manager.initialize_index()
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg and "Invalid API Key" in error_msg:
                st.error(
                    "Invalid Pinecone API Key. Please check your API key and try again."
                )
            elif "404" in error_msg:
                st.error(
                    f"Index not found: {vector_store_manager.pinecone_index_name}. Please check your index name."
                )
            else:
                st.error(f"Error initializing Pinecone index: {e}")
            return False

        # Store the vector store manager in session state
        st.session_state.vector_store_manager = vector_store_manager

        # Initialize the chatbot
        try:
            vector_store = vector_store_manager.get_vector_store()
            st.session_state.chatbot = RAGChatbot(
                vector_store, api_key=st.session_state.openai_api_key
            )

            # Set default namespace if available
            st.session_state.namespaces = ["default"]
            st.session_state.current_namespace = "default"

            return True
        except Exception as e:
            st.error(f"Error initializing chatbot: {e}")
            return False
    except Exception as e:
        st.error(f"Error initializing vector store: {e}")
        return False


# Function to process uploaded files
def process_files(files, namespace):
    try:
        # Create a temporary directory to store the uploaded files
        with tempfile.TemporaryDirectory() as temp_dir:
            file_paths = []

            # Save the uploaded files to the temporary directory
            for file in files:
                file_path = os.path.join(temp_dir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                file_paths.append(file_path)

            # Process the documents
            document_processor = DocumentProcessor()
            documents = document_processor.process_documents(file_paths)

            # Add the documents to the vector store
            st.session_state.vector_store_manager.add_documents(
                documents, namespace=namespace
            )

            # Add the namespace to the list if it's not already there
            if namespace not in st.session_state.namespaces:
                st.session_state.namespaces.append(namespace)
                st.session_state.current_namespace = namespace

            return len(documents)
    except Exception as e:
        st.error(f"Error processing files: {e}")
        return 0


# Function to save API keys to .env file
def save_api_keys(
    openai_api_key, pinecone_api_key, pinecone_environment, pinecone_index_name
):
    try:
        with open(".env", "w") as f:
            f.write(f"# OpenAI API Key\n")
            f.write(f"OPENAI_API_KEY={openai_api_key}\n\n")
            f.write(f"# Pinecone API Key - Make sure this is your complete API key\n")
            f.write(
                f"# It should look like: pcsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n"
            )
            f.write(f"PINECONE_API_KEY={pinecone_api_key}\n\n")
            f.write(f"# Pinecone Environment and Index Name\n")
            f.write(f"PINECONE_ENVIRONMENT={pinecone_environment}\n")
            f.write(f"PINECONE_INDEX_NAME={pinecone_index_name}\n")

        # Update session state
        st.session_state.openai_api_key = openai_api_key
        st.session_state.pinecone_api_key = pinecone_api_key
        st.session_state.pinecone_environment = pinecone_environment
        st.session_state.pinecone_index_name = pinecone_index_name

        return True
    except Exception as e:
        st.error(f"Error saving API keys: {e}")
        return False


# Main application
def main():
    # Check if we're in iframe mode (embedded)
    is_iframe = st.query_params.get("embedded", "false").lower() == "true"

    if is_iframe:
        # Simplified interface for iframe embedding
        st.markdown(
            "<h3 style='text-align: center;'>AI Assistant</h3>", unsafe_allow_html=True
        )
    else:
        # Full interface
        st.title("🤖 RAG Chatbot")

    # Try to initialize the vector store on startup if not already attempted
    if not st.session_state.initialization_attempted:
        st.session_state.initialization_attempted = True
        with st.spinner("Initializing vector store..."):
            initialize_vector_store()

    # Only show sidebar in full mode
    if not is_iframe:
        with st.sidebar:
            st.header("Configuration")

            # API key input
            openai_api_key = st.text_input(
                "OpenAI API Key", value=st.session_state.openai_api_key, type="password"
            )
            pinecone_api_key = st.text_input(
                "Pinecone API Key",
                value=st.session_state.pinecone_api_key,
                type="password",
                help="Enter your full Pinecone API key. It should start with 'pcsk_'",
            )
            pinecone_environment = st.text_input(
                "Pinecone Environment", value=st.session_state.pinecone_environment
            )
            pinecone_index_name = st.text_input(
                "Pinecone Index Name", value=st.session_state.pinecone_index_name
            )

            # Save API keys to .env file
            if st.button("Save API Keys"):
                if save_api_keys(
                    openai_api_key,
                    pinecone_api_key,
                    pinecone_environment,
                    pinecone_index_name,
                ):
                    st.success("API keys saved successfully!")
                    # Initialize the vector store
                    if initialize_vector_store():
                        st.success("Vector store initialized successfully!")

            st.divider()

            # File upload section
            st.header("Upload Documents")

            # Namespace input
            namespace = st.text_input("Namespace", "default")

            # File uploader
            uploaded_files = st.file_uploader(
                "Upload documents",
                type=["pdf", "txt", "csv", "md"],
                accept_multiple_files=True,
            )

            # Process the uploaded files
            if uploaded_files and st.button("Process Documents"):
                if st.session_state.vector_store_manager is None:
                    if not initialize_vector_store():
                        st.error("Please initialize the vector store first!")
                        return

                num_chunks = process_files(uploaded_files, namespace)
                if num_chunks > 0:
                    st.success(
                        f"Successfully processed {len(uploaded_files)} files into {num_chunks} chunks!"
                    )

            st.divider()

            # Namespace selection
            if st.session_state.namespaces:
                st.header("Select Namespace")
                selected_namespace = st.selectbox(
                    "Namespace",
                    options=st.session_state.namespaces,
                    index=(
                        st.session_state.namespaces.index(
                            st.session_state.current_namespace
                        )
                        if st.session_state.current_namespace
                        in st.session_state.namespaces
                        else 0
                    ),
                )

                if st.button("Set Namespace"):
                    st.session_state.current_namespace = selected_namespace
                    st.success(f"Namespace set to {selected_namespace}")

            # Reset conversation button
            if st.session_state.chatbot:
                if st.button("Reset Conversation"):
                    st.session_state.chatbot.reset_conversation()
                    st.session_state.messages = []
                    st.success("Conversation reset!")

            # Reinitialize button
            if st.button("Reinitialize Vector Store"):
                if initialize_vector_store():
                    st.success("Vector store reinitialized successfully!")

    # Main chat interface
    if not is_iframe:
        st.header("Chat")

    # Display status (only in full mode)
    if not is_iframe:
        if st.session_state.chatbot is None:
            st.warning(
                "Vector store not initialized. Please check your API keys and reinitialize the vector store."
            )
        else:
            st.success("Vector store initialized. You can start chatting!")

    # Custom styling for iframe mode
    if is_iframe:
        st.markdown(
            """
        <style>
        .stApp {
            background-color: transparent;
        }
        .main .block-container {
            padding-top: 0;
            padding-bottom: 0;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Display source documents if available (only in full mode)
            if (
                not is_iframe
                and "source_documents" in message
                and message["source_documents"]
            ):
                with st.expander("Source Documents"):
                    for i, doc in enumerate(message["source_documents"]):
                        st.markdown(f"**Source {i+1}:**")
                        st.markdown(f"```\n{doc.page_content}\n```")
                        st.markdown(f"**Metadata:** {doc.metadata}")
                        st.divider()

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Check if chatbot is initialized
        if st.session_state.chatbot is None:
            st.error("Please initialize the vector store and upload documents first!")
            return

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response from chatbot
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot.chat(
                    prompt, namespace=st.session_state.current_namespace
                )

                st.markdown(response["answer"])

                # Display source documents (only in full mode)
                if not is_iframe and response["source_documents"]:
                    with st.expander("Source Documents"):
                        for i, doc in enumerate(response["source_documents"]):
                            st.markdown(f"**Source {i+1}:**")
                            st.markdown(f"```\n{doc.page_content}\n```")
                            st.markdown(f"**Metadata:** {doc.metadata}")
                            st.divider()

        # Add assistant message to chat history
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response["answer"],
                "source_documents": response["source_documents"],
            }
        )


if __name__ == "__main__":
    main()
