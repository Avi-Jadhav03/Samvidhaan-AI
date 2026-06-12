import os
import json

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from state import LegalAuditState


load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(
    model="llama-3.3-70b-versatile"
)

def type_identifier(state: LegalAuditState):
    prompt = f"Think like a professional Indian lawyer, identify the document type. Give me a single word string, nothing additional.\n\nDocument:\n{state['raw_document']}"

    result = model.invoke(prompt)
    return {"document_type":result.content}

def meaning_extractor(state : LegalAuditState):
    prompt = f"im giving you a raw document and its type ,you task is to extract the meaning in key value pair  in structured format and return the output in json format. every meaning should be extracted with value not having in detail description, just main word\n\nDocument:\n{state['raw_document']}\n\nType of Document:\n{state['document_type']}"

    result = model.invoke(prompt)
    parsed = json.loads(result.content)
    return {"extracted_clauses": parsed}