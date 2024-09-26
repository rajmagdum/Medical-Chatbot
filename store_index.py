from src.helper import load_pdf, text_split, download_openai_embeddings
import faiss
import os
import numpy as np
from dotenv import load_dotenv

# Load environment variables, including proxies
load_dotenv()

# Load and preprocess the PDF data
extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)

# Create OpenAI embeddings for each text chunk
texts = [t.page_content for t in text_chunks]

# Call the embedding function to get embeddings for all text chunks
# Batch size is set to avoid hitting the API payload limit (413 error)
text_embeddings = download_openai_embeddings(texts, batch_size=20)

# Ensure that embeddings are returned in the correct format
if text_embeddings is None or len(text_embeddings) == 0:
    raise ValueError("Embeddings could not be generated. Please check your OpenAI API key and embeddings function.")

# Initialize FAISS index
d = len(text_embeddings[0])  # Dimension of the embeddings
index = faiss.IndexFlatL2(d)

# Convert the list of embeddings to a NumPy array
text_embeddings_np = np.array(text_embeddings)

# Add embeddings to the FAISS index
index.add(text_embeddings_np)

# Save the FAISS index
faiss.write_index(index, "medical_bot.index")

print("FAISS index has been created and saved as 'medical_bot.index'")