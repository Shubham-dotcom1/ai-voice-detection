import requests

print("Starting tests...")

# Your live API URL
API_URL = "https://ai-voice-detection-ww3g.onrender.com"

try:
    # Test 1: Health Check
    print("=" * 50)
    print("Test 1: Health Check")
    print("=" * 50)
    
    response = requests.get(API_URL, timeout=30)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

except Exception as e:
    print(f"Error: {e}")

try:
    # Test 2: API with Key
    print("\n" + "=" * 50)
    print("Test 2: API with Key")
    print("=" * 50)
    
    response = requests.post(
        f"{API_URL}/api/voice-detection",
        headers={
            "Content-Type": "application/json",
            "x-api-key": "sk_test_123456789"
        },
        json={
            "language": "English",
            "audioFormat": "mp3",
            "audioBase64": "dGVzdGluZw=="
        },
        timeout=30
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 50)
print("Tests completed!")
print("=" * 50)