from flask import Flask, render_template, jsonify, request
from src.helper import get_openai_embeddings
from langchain_community.vectorstores import FAISS
import faiss
import os
import numpy as np
from PyPDF2 import PdfReader
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI  # Updated import from langchain_openai
from langchain.chains import RetrievalQA
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.docstore.document import Document
from dotenv import load_dotenv
from src.prompt import prompt_template
import asyncio
import openai

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
embedding_model = get_openai_embeddings()

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
    pdf_text = extract_text_from_pdf(pdf_file_path)
    chunks = split_text_into_chunks(pdf_text)
    documents = [Document(page_content=chunk) for chunk in chunks]
    document_embeddings = embedding_model.embed_documents([chunk for chunk in chunks])

    if document_embeddings is None:
        raise ValueError("Failed to obtain document embeddings.")

    document_embeddings_np = np.array(document_embeddings)
    embedding_dim = len(document_embeddings[0])
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(document_embeddings_np)
    faiss.write_index(index, index_file)
    print(f"Created and saved FAISS index to {index_file}")

# Create the document store and index map
index_to_docstore_id = {i: str(i) for i in range(len(chunks))}
docstore = InMemoryDocstore({str(i): Document(page_content=chunk) for i, chunk in enumerate(chunks)})

# Update this line to pass the full embedding model, not just the embedding function
docsearch = FAISS(embedding_model, index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id)

# Create the PromptTemplate object
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Define the chain type kwargs
chain_type_kwargs = {"prompt": PROMPT}

# Initialize the OpenAI LLM (ensure you're using the correct import)
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.8, max_tokens=256)

# Initialize the QA system using the FAISS retriever
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

# Function to warm up the model
def warm_up_model():
    dummy_query = "Hello"
    try:
        result = qa.invoke({"query": dummy_query})  # Updated from __call__ to invoke
        print("Model warmed up successfully.")
        print("Warm-up result:", result)
    except Exception as e:
        print(f"Error during model warm-up: {e}")

# Call warm-up to ensure the model is ready
warm_up_model()

# Async function to handle user queries
async def async_query(query):
    result = qa.invoke({"query": query})  # Updated from __call__ to invoke
    clean_result = result["result"].replace("<|im_end|>", "").strip()
    return clean_result

# Flask route for the home page
@app.route("/")
def index():
    return render_template('chat.html')

# Flask route for handling user queries (POST request)
@app.route("/get", methods=["POST"])
async def chat():
    msg = request.form["msg"]
    input = msg
    print(f"User input: {input}")
    result = await async_query(input)
    print("Response:", result)
    return jsonify(result)

# Run the Flask app
if __name__ == '__main__':
    # Use the PORT environment variable or default to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True, threaded=True)
