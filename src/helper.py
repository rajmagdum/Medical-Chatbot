import os
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from openai import error as openai_error
import math

# Load environment variables from .env file
load_dotenv()

# Function to create and return the OpenAIEmbeddings object
def get_openai_embeddings():
    try:
        # Load proxy settings from environment (optional)
        # http_proxy = os.getenv("HTTP_PROXY")
        # https_proxy = os.getenv("HTTPS_PROXY")

        # Create an OpenAIEmbeddings object, without proxy settings
        embedding_model = OpenAIEmbeddings(
            model="text-embedding-ada-002",  # Specify the model for embeddings
            openai_api_key=os.getenv("OPENAI_API_KEY"),  # Load OpenAI API key from environment
            openai_api_base=os.getenv("OPENAI_API_BASE"),  # Load API base URL from environment
            # openai_proxy=http_proxy if http_proxy else https_proxy  # Load proxy if available (commented out)
        )
        return embedding_model

    except openai_error.OpenAIError as e:
        print(f"Error initializing OpenAI embeddings: {str(e)}")
        return None

# Optional function to download embeddings for multiple texts in batches (if needed)
def download_openai_embeddings(texts, batch_size=20):
    """
    Generate OpenAI embeddings for a list of texts in batches.
    
    Args:
    texts (list of str): The texts to embed.
    batch_size (int): The number of texts to process in each API call.

    Returns:
    list of list of float: A list containing the embeddings for each text.
    """
    try:
        embedding_model = get_openai_embeddings()
        if embedding_model is None:
            return None

        all_embeddings = []
        num_batches = math.ceil(len(texts) / batch_size)

        for i in range(num_batches):
            batch = texts[i * batch_size : (i + 1) * batch_size]
            
            # Embed the batch using the embedding model's embed_documents method
            batch_embeddings = embedding_model.embed_documents(batch)

            if len(batch_embeddings) != len(batch):
                print(f"Error: Expected {len(batch)} embeddings but got {len(batch_embeddings)}.")
                return None

            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    except Exception as e:
        print(f"Error downloading embeddings: {str(e)}")
        return None

# Function to load and preprocess a PDF
def load_pdf(pdf_directory):
    # Implementation for loading a PDF goes here
    pass

# Function to split the extracted text into smaller chunks
def text_split(text, chunk_size=500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
