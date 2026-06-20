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
    if state["retry_query"]:
        query = state["retry_query"]
        docs = retriever.invoke(query)
    else:
        # Build specific queries from extracted clauses
        clauses_str = str(state['extracted_clauses'])
        
        queries = [
            f"landlord tenant rights obligations {state['document_type']}",
            f"rent payment terms security deposit India law",
            f"notice period termination lease agreement India",
            f"registration requirements {state['document_type']} India"
        ]
        
        # Search for each query and combine results
        all_docs = []
        for q in queries:
            docs = retriever.invoke(q)
            all_docs.extend(docs)
        
        # Remove duplicates
        seen = set()
        unique_docs = []
        for doc in all_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_docs.append(doc)
        
        docs = unique_docs

    retrieved_laws = [doc.page_content for doc in docs]
    return {"retrieved_laws": retrieved_laws}


def cross_checker(state: LegalAuditState):
    prompt = f"""You are an expert Indian legal document auditor.

I will give you extracted information from a legal document and relevant Indian laws.
Your job is to cross-check the document against the laws and find all problems.

Look for:
- Missing mandatory clauses that Indian law REQUIRES
- Clauses that Indian law PROHIBITS but are present in the document
- Illegal or risky terms
- Procedural errors

Important rules for analysis:
- If a law says "shall not contain X" → flag if document HAS X, not if it is missing
- If a law says "must contain X" → flag if document is MISSING X
- Read the law carefully before deciding if something is missing or prohibited
- Never flip the meaning of what the law says

Output rules:
- Return ONLY a valid JSON array, nothing else
- No explanation, no markdown, no backticks
- Each finding must have these exact keys: "issue", "law_reference", "severity", "suggested_fix"
- Severity must be "high", "medium" or "low"
- Keep language simple and plain English

Extracted Document Information:
{state['extracted_clauses']}

Relevant Indian Laws:
{state['retrieved_laws']}

Return only JSON array:"""

    result = model.invoke(prompt)
    cleaned = result.content.strip()
    parsed = json.loads(cleaned)
    return {"analyst_findings": parsed}

def hallucination_guard(state: LegalAuditState):
    prompt = f"""You are a strict legal AI verifier.

I will give you the original document, analyst findings, and retrieved laws.
Your job is to verify each finding by checking TWO things:
1. Is this finding actually true based on what the document says?
2. Is this finding supported by the retrieved laws?

If a finding claims something is missing but it IS in the document → REJECT it
If a finding claims something exists but it is NOT in the document → REJECT it  
If ALL findings are accurate and law-backed → return "pass"
If ANY finding is wrong → return "re-fetch" with a specific query to get better laws

Rules:
- Return ONLY valid JSON, no markdown, no backticks
- Return exactly: {{"verdict": "pass", "retry_query": ""}} or {{"verdict": "re-fetch", "retry_query": "specific query"}}

Original Document:
{state['raw_document']}

Analyst Findings:
{state['analyst_findings']}

Retrieved Laws:
{state['retrieved_laws']}

Return only JSON:"""

    result = model.invoke(prompt)
    cleaned = result.content.strip()
    parsed = json.loads(cleaned)
    
    return {
        "guard_verdict": parsed["verdict"],
        "retry_query": parsed["retry_query"],
        "retry_count": state["retry_count"] + 1
    }


def report_generator(state: LegalAuditState):
    prompt = f"""You are a helpful legal assistant explaining legal issues to a common person in India.

I will give you a list of legal findings from a document audit.
Your job is to convert these into a simple, clear audit report.

Rules:
- Write in plain English, no legal jargon
- Be empathetic - the user is not a lawyer
- Return ONLY valid JSON, no markdown, no backticks
- Return exactly this structure:
{{
    "overall_risk": "High/Medium/Low",
    "summary": "2-3 line plain English summary of the document's condition",
    "issues": [
        {{
            "issue": "what is wrong in simple words",
            "why_it_matters": "why this could harm the user",
            "how_to_fix": "simple actionable fix"
        }}
    ]
}}

Document Type: {state['document_type']}
Findings: {state['analyst_findings']}

Return only JSON:"""

    result = model.invoke(prompt)
    cleaned = result.content.strip()
    parsed = json.loads(cleaned)
    return {"final_report": str(parsed)}