import requests
import base64
import os

# LOCAL API (faster!)
API_URL = "http://127.0.0.1:8000/api/voice-detection"
API_KEY = "sk_test_123456789"

print("🎤 AI Voice Detection - LOCAL Test")
print("="*50)

audio_file = "sample voice 1.mp3"

print(f"\nTesting with: {audio_file}")
print("="*50)

# Check file exists
if not os.path.exists(audio_file):
    print(f"❌ File not found: {audio_file}")
    exit()

# Load audio
with open(audio_file, "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode('utf-8')
print(f"✅ File loaded, size: {len(audio_base64)} characters")

# Send request
print("📡 Sending request to LOCAL API...")

response = requests.post(
    API_URL,
    headers={
        "Content-Type": "application/json",
        "x-api-key": API_KEY
    },
    json={
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": audio_base64
    },
    timeout=300
)

print(f"\n📊 Response:")
print(f"Status Code: {response.status_code}")

result = response.json()

if response.status_code == 200:
    print(f"\n🎯 RESULT:")
    print(f"   Status: {result.get('status')}")
    print(f"   Language: {result.get('language')}")
    print(f"   Classification: {result.get('classification')}")
    print(f"   Confidence: {result.get('confidenceScore')}")
    print(f"   Explanation: {result.get('explanation')}")
else:
    print(f"❌ Error: {result}")

print("\n" + "="*50)
print("✅ Test completed!")
print("="*50)