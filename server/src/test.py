import requests

# Replace with your actual server's IP address and port
server_url = "http://44.211.155.206:8000"

# Example 1: Testing the /process endpoint with ADD_USER request
def test_add_user():
    url = f"{server_url}/process"
    headers = {
        'key': 'kivsa_ehad',  # Replace with your actual API key
        'request-type': 'ADD_USER',
        'voice_description': 'Test Voice'
    }
    files = {
        'files[]': open('output.mp4', 'rb')  # Replace with the path to your audio file
    }
    data = {
        'voice': 'test_voice',
        'files_encrypted': 'false'
    }
    response = requests.post(url, headers=headers, files=files, data=data)
    print("ADD_USER response:", response.json())

# Example 2: Testing the /process endpoint with INPUT request
def test_generate_speech():
    url = f"{server_url}/process"
    headers = {
        'key': 'your_api_key',  # Replace with your actual API key
        'request-type': 'INPUT'
    }
    data = {
        'text': 'Hello, this is a test.',
        'voice': 'test_voice'
    }
    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        with open('output_speech.mp3', 'wb') as f:
            f.write(response.content)
        print("Speech generated and saved as output_speech.mp3")
    else:
        print("Error generating speech:", response.json())

# Example 3: Testing the /audio endpoint
def test_audio_stream():
    url = f"{server_url}/audio"
    headers = {
        'key': 'your_api_key'  # Replace with your actual API key
    }
    response = requests.post(url, headers=headers)
    print("Audio stream response:", response.json())

# Run tests
test_add_user()
test_generate_speech()
test_audio_stream()
