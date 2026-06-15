from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

# Step 1 - Load all PDFs from documents folder
documents = []
for filename in os.listdir("documents/"):
    if filename.lower().endswith(".pdf"):
        loader = PyPDFLoader(f"documents/{filename}")
        documents.extend(loader.load())
        print(f"Loaded: {filename}")

# Step 2 - Split text into smaller chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
print(f"Total chunks created: {len(chunks)}")

# Step 3 - Create embeddings and store in ChromaDB
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="vectorstore/"
)
print("Vector store saved successfully!")