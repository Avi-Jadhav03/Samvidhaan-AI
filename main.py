test_document = """
RENT AGREEMENT

This agreement is made between Ram Sharma (Owner) and Shyam Patel (Tenant).
Property Address: 123, MG Road, Pune, Maharashtra.
Monthly Rent: Rs. 8000
Agreement Start Date: 1st January 2024
"""

from state import LegalAuditState
from agents import type_identifier, meaning_extractor

from agents import law_fetcher



test_state = {
    "raw_document": test_document,
    "document_type": "",
    "extracted_clauses": [],
    "retrieved_laws": [],
    "analyst_findings": [],
    "guard_verdict": "",
    "retry_count": 0,
    "retry_query": "",
    "final_report": ""
}

# Run agent 1
result1 = type_identifier(test_state)
print("Type:", result1)
test_state.update(result1)

# Run agent 2
result2 = meaning_extractor(test_state)
print("Extracted:", result2)
test_state.update(result2)

# Run agent 3
result3 = law_fetcher(test_state)
print("Laws:", result3)