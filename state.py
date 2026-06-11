from typing import TypedDict

class LegalAuditState(TypedDict):
    raw_document: str                      #user uploaded file
    document_type: str                     #type like - RTI,House rent
    extracted_clauses: list[dict]          #structure extracted data from user document
    retrieved_laws: list[str]              #retrive the similar laws from documents folder
    analyst_findings: list[dict]           #cross-check after finding result
    guard_verdict: str                     #pass or re-fetch the laws 
    retry_count: int                       #count the retry times
    retry_query: str                       #which specific query to retry 
    final_report: str                      #final result