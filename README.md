# AI Voice Detection API

Detect AI-generated vs Human voices in Tamil, English, Hindi, Malayalam, and Telugu.

## Endpoint

POST /api/voice-detection

## Headers
- Content-Type: application/json
- x-api-key: sk_test_123456789

## Request
```json
{
    "language": "Tamil",
    "audioFormat": "mp3",
    "audioBase64": "base64_audio_here"
}

{
    "status": "success",
    "language": "Tamil",
    "classification": "AI_GENERATED",
    "confidenceScore": 0.87,
    "explanation": "Unnatural pitch consistency detected"
}