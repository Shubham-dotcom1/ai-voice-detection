from fastapi import Header, HTTPException
import os

# API Keys - Add your own keys here
VALID_API_KEYS = {
    os.getenv("API_KEY", "sk_test_123456789"),
    os.getenv("API_KEY_2", "sk_live_guvi_hackathon_2024"),
    "sk_test_123456789"
}

async def verify_api_key(x_api_key: str = Header(..., alias="x-api-key")):
    """Validate API key from request header"""
    if x_api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return x_api_key