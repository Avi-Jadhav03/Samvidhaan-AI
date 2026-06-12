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

