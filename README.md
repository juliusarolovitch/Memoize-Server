# Memoize Labs API Documentation

#### Usage Note: This API is available to be called via an AWS instance or to be run locally

#### Structural Note: 
local_server.py contains the up-to-date version for the local server. remote_server.py contains the up-to-date version for the remote server. Assign a 3rd-party reviewer when merging changes into either, and note that the remote server will require pulling from GitHub and restarting to reflect changes. Only 5 files should exist in the repository: the remote and local server scripts, the gitignore, the library requirements, and the README. Note that you will need the .env file to run this server locally, but it cannot be posted to the public repository due to API keys. 

_Fun fact: you can install all necessary libraries by running_
```bash
pip install -r requirements.txt
```
#####_Not so fun fact: when you add a new library import, you have to add it and its version to requirements.txt_

#### And now for the useful stuff...

The current implementation of the API is essentially a proxy server to the ElevenLabs API. All it does is provide a Flask-based request interface for text-to-speech and voice cloning. This will change as we add more of our features. 

The necessary parts to construct an API call are as follows:

+ Headers:
  + 'key' - Your Memoize personal API key. 'MEMOIZE_KEY' is used throughout the code as a placeholder. 
  + 'request-type':
    + Specifying 'ADD_USER' as your request type should be used for training and saving a new voice clone.
    + Specifying 'INPUT' as your request type should be used for text input for text to speech (TTS).
+ Data:
  + 'text' - this is only mandatory to specify when operating in INPUT mode, i.e. when you're inputting text for TTS.
  + 'voice' - the name of the voice you are referencing. In INPUT mode, it is the name of the voice you want the text to be read in. In ADD_USER mode, it is the name of the voice you want to add.

### Basic Security: Implementing Key-Specific Voices:

The current implementation has a framework so that API users will not be able to access or see the voices of other users for simple use cases. This is implemented by concatenating the API user's ID to the name of the voice they are using on the backend. i.e. for an API user with key of 'sheep' using a voice named 'cow', the backend voice will be titled 'sheep_cow'. Additionally, when we fetch all voices for a user, it is as simple as tracking down all voices that start with the necessary API key. There are obvious vulnerabilities, though. 

### Notes on the Remote Server:
The remote server is hosted via AWS EC2. It's public IP address is 54.82.15.158. It supports up to 4 concurrent requests-subsequent requests may be queued or rejected. 

## Example Usage:

You don't have to train a voice right away to use text-to-speech. This is because all users have access to my voice via the ElevenLabs API, which you can try. It's ID is 'Julius'. 
Training a voice requires at least 30 seconds of quality recording, preferably with a variety of expressions in the recording. Note that it currently struggles with strong accents. 

### _The below examples will work for local servers. To call the remote server, simply replace the url with 'http://54.82.15.158:5000/process'_

### Simple TTS Request With No Error Checking:
Use at your own risk. This is primarily for understanding the necessary parts to construct the API call. 
The text can be in any of the 12 languages that both the ElevenLabs API and ChatGPT support, including English, Spanish, French, German, Italian, Portugese, Dutch, Russian, Chinese, Japanese, Korean, and Arabic. Text to be read is passed in through the data array under the 'text' parameter. 

```python
url = 'http://127.0.0.1:5000/process'
headers = {'key': 'MEMOIZE_KEY', 'request-type': 'INPUT'}
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
The text can be in any of the 12 languages that both the ElevenLabs API and ChatGPT support, including English, Spanish, French, German, Italian, Portugese, Dutch, Russian, Chinese, Japanese, Korean, and Arabic. Text to be read is passed in through the data array under the 'text' parameter. 

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
headers = {'key': 'MEMOIZE_KEY', 'request-type': 'INPUT'}
text = "Hello! 你好! Hola! नमस्ते! Bonjour! こんにちは! مرحبا! 안녕하세요! Ciao! Cześć! Привіт! வணக்கம்!"
voice = 'Julius'

generate_speech(url, headers, text, voice)
```

### Simple Voice Cloning Request With No Error Checking:
Use at your own risk. This is primarily for understanding the necessary parts to construct the API call. 
The file paths used to generate a voice clone are to be contained in the file_paths array. They are then opened and the data is fed through the API to the server. 
Voice cloning benefits from adding a brief description of the person, which can be passed in as a 'voice_description' parameter within the header array. 


```python
url = 'http://127.0.0.1:5000/process'
headers = {'key': 'MEMOIZE_KEY',
           'request-type': 'ADD_USER', 'voice_description': ''}
file_paths = ['output.mp4']
files = [('files[]', open(file_path, 'rb')) for file_path in file_paths]
data = {'voice': 'Julius'}
requests.post(url, files=files, headers=headers, data=data)
```

### Voice Cloning Request With Error Checking
All necessary error checking implemented and will print the error upon return of the API if necessary. 
The file paths used to generate a voice clone are to be contained in the file_paths array. They are then opened and the data is fed through the API to the server. 
Voice cloning benefits from adding a brief description of the person, which can be passed in as a 'voice_description' parameter within the header array. 

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
headers = {'key': 'MEMOIZE_KEY', 'request-type': 'ADD_USER', 'voice_description': ''}
# The paths of the audio recordings you're looking to train on within your local directory
file_paths = ['output.mp4']

add_user(url, headers, file_paths)
```
