import requests
import base64
import os

API_URL = "https://ai-voice-detection-ww3g.onrender.com/api/voice-detection"
API_KEY = "sk_test_123456789"

def test_with_audio_file(file_path, language="English"):
    """Test API with a real audio file"""
    
    print(f"\n{'='*50}")
    print(f"Testing with: {file_path}")
    print(f"Language: {language}")
    print('='*50)
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    # Read and encode audio file
    try:
        with open(file_path, "rb") as audio_file:
            audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')
        print(f"✅ File loaded, size: {len(audio_base64)} characters (base64)")
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return
    
    # Make API request
    try:
        print("📡 Sending request to API...")
        response = requests.post(
            API_URL,
            headers={
                "Content-Type": "application/json",
                "x-api-key": API_KEY
            },
            json={
                "language": language,
                "audioFormat": "mp3",
                "audioBase64": audio_base64
            },
            timeout=120
        )
        
        print(f"\n📊 Response:")
        print(f"Status Code: {response.status_code}")
        
        result = response.json()
        
        if response.status_code == 200:
            print(f"\n🎯 RESULT:")
            print(f"   Classification: {result.get('classification', 'N/A')}")
            print(f"   Confidence: {result.get('confidenceScore', 'N/A')}")
            print(f"   Explanation: {result.get('explanation', 'N/A')}")
        else:
            print(f"❌ Error: {result}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

print("🎤 AI Voice Detection - Real Audio Test")
print("="*50)

# YOUR AUDIO FILE HERE 👇
audio_file = "sample voice 1.mp3"

# Test with English (change language if needed: Tamil, Hindi, Malayalam, Telugu)
test_with_audio_file(audio_file, language="English")

print("\n" + "="*50)
print("✅ Test completed!")
print("="*50)