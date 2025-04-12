from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

from ..script_parser.core import ScriptParser

app = FastAPI(title="AI Video Teacher - Script Parser")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize script parser
script_parser = ScriptParser()

class ScriptResponse(BaseModel):
    scenes: list
    metadata: dict

@app.post("/parse-script", response_model=ScriptResponse)
async def parse_script(file: UploadFile = File(...)):
    """
    Parse an uploaded script file and return structured data for video generation.
    """
    try:
        # Read the uploaded file
        content = await file.read()
        script_text = content.decode("utf-8")
        
        # Parse the script
        result = script_parser.parse_script(script_text)
        
        return ScriptResponse(
            scenes=result["scenes"],
            metadata=result["metadata"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 