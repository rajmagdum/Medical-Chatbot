from flask import Flask, render_template, jsonify, request
from src.helper import get_openai_embeddings  # Updated to import the function
from langchain.vectorstores import FAISS
import faiss
import os
import numpy as np
from PyPDF2 import PdfReader
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.docstore.document import Document
from dotenv import load_dotenv
from src.prompt import prompt_template  # Ensure this points to the right prompt
import asyncio

app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

# Path to your PDF file and FAISS index file
pdf_file_path = "data/Medical_book.pdf"
index_file = "medical_bot.index"

# Step 1: Extract text from the PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Step 2: Split the extracted text into smaller chunks (e.g., paragraphs or sentences)
def split_text_into_chunks(text, chunk_size=500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Initialize the embedding model
embedding_model = get_openai_embeddings()  # Initialize the embedding model here

# Check if FAISS index already exists
if os.path.exists(index_file):
    print(f"Loading FAISS index from {index_file}")
    index = faiss.read_index(index_file)

    # Load the document text to associate with the FAISS index
    pdf_text = extract_text_from_pdf(pdf_file_path)
    chunks = split_text_into_chunks(pdf_text)
    documents = [Document(page_content=chunk) for chunk in chunks]
else:
    print(f"FAISS index not found, creating new index.")
    # Extract and split the text from the PDF
    pdf_text = extract_text_from_pdf(pdf_file_path)
    chunks = split_text_into_chunks(pdf_text)

    # Convert the chunks into LangChain Document objects
    documents = [Document(page_content=chunk) for chunk in chunks]

    # Step 3: Embed the text chunks using the embedding model
    document_embeddings = embedding_model.embed_documents([chunk for chunk in chunks])

    # Check if embeddings were returned successfully
    if document_embeddings is None:
        raise ValueError("Failed to obtain document embeddings.")

    # Step 4: Convert embeddings to NumPy arrays for FAISS compatibility
    document_embeddings_np = np.array(document_embeddings)

    # Step 5: Initialize FAISS - Create a new index with the embedding dimension
    embedding_dim = len(document_embeddings[0])  # Using len() to get the dimensionality of the embedding
    index = faiss.IndexFlatL2(embedding_dim)  # L2 distance (Euclidean)

    # Step 6: Add the embeddings to the FAISS index
    index.add(document_embeddings_np)

    # Step 7: Save the index to a file
    faiss.write_index(index, index_file)
    print(f"Created and saved FAISS index to {index_file}")

# Manually create index_to_docstore_id mapping
index_to_docstore_id = {i: str(i) for i in range(len(chunks))}

# Create the InMemoryDocstore
docstore = InMemoryDocstore({str(i): Document(page_content=chunk) for i, chunk in enumerate(chunks)})

# Initialize FAISS vector store with the embedding model
docsearch = FAISS(embedding_model.embed_query, index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id)

# Define the prompt template
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs = {"prompt": PROMPT}

# Load the LLM (GPT-3.5 turbo or any specified model) using environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_base = os.getenv("OPENAI_API_BASE")

llm = OpenAI(model="gpt-35-turbo", api_key=openai_api_key, api_base=openai_api_base, temperature=0.8, max_tokens=256)

# Create the RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

# Warm-up the model for faster first response
def warm_up_model():
    dummy_query = "Hello"
    try:
        result = qa({"query": dummy_query})
        print("Model warmed up successfully.")
        print("Warm-up result:", result)
    except Exception as e:
        print(f"Error during model warm-up: {e}")

warm_up_model()  # Call warm-up when the server starts

# Asynchronous query function
async def async_query(query):
    result = qa({"query": query})
    # Remove the <|im_end|> token from the response
    clean_result = result["result"].replace("<|im_end|>", "").strip()
    return clean_result

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
async def chat():
    msg = request.form["msg"]
    input = msg
    print(f"User input: {input}")

    # Query the RetrievalQA system asynchronously
    result = await async_query(input)
    print("Response:", result)

    return jsonify(result)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True, threaded=True)