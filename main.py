import os
import subprocess
import sys


def check_dependencies():
    """Check if all dependencies are installed."""
    try:
        import streamlit
        import pinecone
        import openai
        import langchain
        import dotenv

        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        return False


def install_dependencies():
    """Install dependencies from requirements.txt."""
    print("Installing dependencies...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
    )
    print("Dependencies installed successfully!")


def main():
    """Main entry point for the application."""
    # Check if dependencies are installed
    if not check_dependencies():
        install_dependencies()

    # Run the Streamlit app
    print("Starting RAG Chatbot...")
    subprocess.call(["streamlit", "run", "app/app.py"])


if __name__ == "__main__":
    main()
