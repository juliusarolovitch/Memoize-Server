# Memoize Labs API Documentation

The current implementation of the API is essentially a proxy server to the ElevenLabs API. All it does is provide a Flask-based request interface for text-to-speech and voice cloning. This will change as we add more of our features. 

The necessary parts to construct an API call are as follows:

+ Headers:
  + 'key' - Your Memoize personal API key
  + 'request-type':
    + Specifying 'ADD_USER' as your request type should be used for training and saving a new voice clone.
    + Specifying 'INPUT' as your request type should be used for text input for text to speech (TTS).
+ Data:
  + 'text' - this is only mandatory to specify when operating in INPUT mode, i.e. when you're inputting text for TTS.
  + 'voice' - the name of the voice you are referencing. In INPUT mode, it is the name of the voice you want the text to be read in. In ADD_USER mode, it is the name of the voice you want to add.

### Basic Security: Implementing Key-Specific Voices:

The current implementation has a framework so that API users will not be able to access or see the voices of other users for simple use cases. This is implemented by concatenating the API user's ID to the name of the voice they are using on the backend. i.e. for an API user with key of 'sheep' using a voice named 'cow', the backend voice will be titled 'sheep_cow'. Additionally, when we fetch all voices for a user, it is as simple as tracking down all voices that start with the necessary API key. There are obvious vulnerabilities, though. 

## Example Usage:

### Simple TTS Request With No Error Checking:
Use at your own risk. This is primarily for understanding the necessary parts to construct the API call. 

```python
url = 'http://127.0.0.1:5000/process'
headers = {'key': 'kivsa_ehad', 'request-type': 'INPUT'}
text = "Hello! 你好! Hola! नमस्ते! Bonjour! こんにちは! مرحبا! 안녕하세요! Ciao! Cześć! Привіт! வணக்கம்!"
voice = 'Julius'
data = {'text': text, 'voice': voice}
response = requests.post(url, headers=headers, data=data)

with open('speech.mp3', 'wb') as f:
    f.write(response.content)
print("Speech generated successfully, saved as speech.mp3")
```

### TTS Request With Error Checking:
All necessary error checking implemented and will print the error upon return of the API if necessary. 

```python
import requests

def generate_speech(url, headers, text, voice):
    data = {'text': text, 'voice': voice}

    try:
        response = requests.post(url, headers=headers, data=data)

        if response.status_code == 200:
            with open('speech.mp3', 'wb') as f:
                f.write(response.content)
            print("Speech generated successfully, saved as speech.mp3")
        else:
            print("Failed to generate speech")
            print("Response status code:", response.status_code)
            try:
                print("Response:", response.json())
            except requests.exceptions.JSONDecodeError:
                print("Response content is not in JSON format")
                print("Response content:", response.text)
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {str(e)}")

url = 'http://127.0.0.1:5000/process'
headers = {'key': 'kivsa_ehad', 'request-type': 'INPUT'}
text = "Hello! 你好! Hola! नमस्ते! Bonjour! こんにちは! مرحبا! 안녕하세요! Ciao! Cześć! Привіт! வணக்கம்!"
voice = 'Julius'

generate_speech(url, headers, text, voice)
```

### Simple Voice Cloning Request With No Error Checking

```python
url = 'http://127.0.0.1:5000/process'
headers = {'key': 'kivsa_ehad',
           'request-type': 'ADD_USER', 'voice_description': ''}
file_paths = ['output.mp4']
files = [('files[]', open(file_path, 'rb')) for file_path in file_paths]
data = {'voice': 'Julius'}
requests.post(url, files=files, headers=headers, data=data)
```

### Voice Cloning Request With Error Checking

```python

import requests

def add_user(url, headers, file_paths):
    files = [('files[]', open(file_path, 'rb')) for file_path in file_paths]
    data = {'voice': 'Julius'}

    try:
        response = requests.post(url, files=files, headers=headers, data=data)

        for _, file_handle in files:
            file_handle.close()

        if response.status_code == 200:
            print("Voice cloned successfully")
            print("Response:", response.json())
        else:
            print("Failed to clone voice")
            print("Response status code:", response.status_code)
            try:
                print("Response:", response.json())
            except requests.exceptions.JSONDecodeError:
                print("Response content is not in JSON format")
                print("Response content:", response.text)
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {str(e)}")

url = 'http://127.0.0.1:5000/process'
headers = {'key': 'kivsa_ehad', 'request-type': 'ADD_USER', 'voice_description': ''}
file_paths = ['output.mp4']

add_user(url, headers, file_paths)
```
