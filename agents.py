import os
import json

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from state import LegalAuditState
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(
    model="llama-3.3-70b-versatile"
)


# Load the vector store from disk
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="vectorstore/",
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})



def type_identifier(state: LegalAuditState):
    prompt = f"""You are an expert Indian legal document analyst.
    
    Identify the type of this legal document.
    Common Indian legal document types: rent_agreement, rti_application, vendor_contract, consumer_complaint.
    Return only the document type in snake_case, nothing else. If unsure return unknown.

    Document:
    {state['raw_document']}"""

    result = model.invoke(prompt)
    return {"document_type": result.content.strip().lower()}

def meaning_extractor(state : LegalAuditState):
    prompt = f"""You are an expert Indian legal document analyst.

        I will give you a raw legal document and its type. Your task is to extract all important information from the document.

        Rules:
        - Return ONLY a valid JSON object, nothing else
        - No explanation, no markdown, no backticks, no additional text
        - Just raw JSON starting with {{ and ending with }}
        - Extract all parties, dates, amounts, addresses, and key terms
        - Keep values concise, not full sentences

        Document Type: {state['document_type']}

        Document:
        {state['raw_document']}

        Return only JSON:"""

    result = model.invoke(prompt)
    cleaned = result.content.strip()
    cleaned = cleaned.strip()
    parsed = json.loads(cleaned)
    return {"extracted_clauses": parsed}

def law_fetcher(state: LegalAuditState):
    # Use retry_query if available, otherwise build from extracted clauses
    if state["retry_query"]:
        query = state["retry_query"]
    else:
        query = f"Laws related to {state['document_type']} in India covering {str(state['extracted_clauses'])}"
    
    # Search vector store
    docs = retriever.invoke(query)
    
    # Extract just the text from each retrieved document
    retrieved_laws = [doc.page_content for doc in docs]
    
    return {"retrieved_laws": retrieved_laws}