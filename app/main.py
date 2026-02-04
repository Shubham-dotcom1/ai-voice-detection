from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
import traceback

from .auth import verify_api_key
from .audio_processor import decode_base64_audio, validate_audio
from .detector import VoiceDetector

# Pydantic Models
class VoiceDetectionRequest(BaseModel):
    language: Literal["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
    audioFormat: Literal["mp3"] = "mp3"
    audioBase64: str

class VoiceDetectionResponse(BaseModel):
    status: Literal["success"] = "success"
    language: str
    classification: Literal["AI_GENERATED", "HUMAN"]
    confidenceScore: float = Field(ge=0.0, le=1.0)
    explanation: str

class ErrorResponse(BaseModel):
    status: Literal["error"] = "error"
    message: str

# FastAPI App
app = FastAPI(
    title="AI Voice Detection API",
    description="Detect AI-generated vs Human voices",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def health_check():
    return {
        "status": "healthy",
        "service": "AI Voice Detection API",
        "version": "1.0.0",
        "supported_languages": ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
    }

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/api/voice-detection", response_model=VoiceDetectionResponse)
async def detect_voice(
    request: VoiceDetectionRequest,
    api_key: str = Depends(verify_api_key)
):
    try:
        supported_languages = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
        if request.language not in supported_languages:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language. Must be one of: {supported_languages}"
            )
        
        if not request.audioBase64 or len(request.audioBase64) < 100:
            raise HTTPException(
                status_code=400,
                detail="Invalid or empty audio data"
            )
        
        try:
            y, sr = decode_base64_audio(request.audioBase64)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to decode audio: {str(e)}"
            )
        
        try:
            validate_audio(y, sr)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        detector = VoiceDetector(y, sr, request.language)
        result = detector.detect()
        
        return VoiceDetectionResponse(
            status="success",
            language=request.language,
            classification=result["classification"],
            confidenceScore=result["confidenceScore"],
            explanation=result["explanation"]
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )