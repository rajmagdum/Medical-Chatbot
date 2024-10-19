# Medical Chatbot with FAISS and OpenAI GPT

This is a medical chatbot built using Flask, FAISS (Facebook AI Similarity Search), and OpenAI's GPT models. The chatbot leverages FAISS for efficient retrieval of text from a medical PDF document, which is embedded into vector space. The chatbot uses OpenAI's GPT model to generate responses to user queries.

## Features
- Extracts and embeds text from medical PDFs.
- Uses FAISS for efficient vector-based retrieval.
- Employs OpenAI's GPT models for natural language responses.
- Simple web interface using Flask.

## Requirements
You need the following tools installed to run this project:
- Python 3.8+
- FAISS for efficient vector search.
- Flask for serving the web application.
- PyPDF2 for PDF text extraction.
- LangChain for integrating with FAISS and OpenAI's GPT models.

## Python Dependencies
The required Python packages are listed in `requirements.txt`:
```bash
langchain==0.0.332
flask
flask[async]
pypdf2
python-dotenv
faiss-cpu
openai==0.28.0
tiktoken
-e .
```
Setup Instructions
Step 1: Clone the Repository
```bash
git clone https://github.com/your-repo/medical-chatbot.git
cd medical-chatbot

```
Step 2: Install Dependencies
It is recommended to use a virtual environment. You can install the required packages using:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

Step 3: Prepare the Medical PDF
Place your medical PDF (e.g., Medical_book.pdf) in the data/ directory. Ensure that the path to the PDF in app.py is correct, or adjust the pdf_file_path in app.py if necessary.

Step 4: Set Up Environment Variables
Create a .env file in the project root with the following values:
```bash
# .env file
OPENAI_API_KEY=your-openai-api-key
OPENAI_API_BASE=https://api.openai.com/v1
```
If using HTTP/HTTPS proxies, also add:
```bash
HTTP_PROXY=http://your-http-proxy
HTTPS_PROXY=https://your-https-proxy
```
Step 5: Create FAISS Index
Run the script to generate the FAISS index for the medical PDF:
```bash
python store_index.py
```
Step 6: Running the Application
To run the Flask application locally, use the following command:
```bash
python app.py
```
