# Memoize Labs API Documentation

Usage:

Simple Voice Generation Request:

'''
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

# URL of the Flask API
url = 'http://127.0.0.1:5000/process'
headers = {'key': 'kivsa_ehad', 'request-type': 'INPUT'}
text = "Hello! 你好! Hola! नमस्ते! Bonjour! こんにちは! مرحبا! 안녕하세요! Ciao! Cześć! Привіт! வணக்கம்!"
voice = 'Julius'

# Generate speech
generate_speech(url, headers, text, voice)

'''
