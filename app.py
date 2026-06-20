from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from llm.main import app as pipeline
import tempfile
import os
import ast
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.post("/audit")
async def audit_document(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    # Extract text from PDF
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    raw_text = " ".join([page.page_content for page in pages])

    # Clean up temp file
    os.unlink(tmp_path)

    # Run through pipeline
    initial_state = {
        "raw_document": raw_text,
        "document_type": "",
        "extracted_clauses": [],
        "retrieved_laws": [],
        "analyst_findings": [],
        "guard_verdict": "",
        "retry_count": 0,
        "retry_query": "",
        "final_report": ""
    }

    result = pipeline.invoke(initial_state)
    return {"report": ast.literal_eval(result["final_report"])}