from flask import Flask, request, jsonify, send_file
import os
import logging
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv, find_dotenv
from io import BytesIO

load_dotenv(find_dotenv('keys.env'))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4'}

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

logging.basicConfig(level=logging.DEBUG)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower(
           ) in app.config['ALLOWED_EXTENSIONS']


def authenticate_api_key(request):
    provided_api_key = request.headers.get('key')
    valid_api_keys = os.getenv('FLASK_API_KEYS').split(',')
    app.logger.debug(f"Provided API key: {provided_api_key}")
    app.logger.debug(f"Valid API keys: {valid_api_keys}")
    return provided_api_key in valid_api_keys


def get_all_voices(client):
    try:
        response = client.voices.get_all()
        return response.voices
    except Exception as e:
        app.logger.error(f"Error getting voices: {str(e)}")
        return []


@app.route('/process', methods=['POST'])
def process_request():
    if not authenticate_api_key(request):
        return jsonify({"error": "Unauthorized"}), 401

    request_type = request.headers.get('request-type')

    if request_type == 'ADD_USER':
        return add_user(request)
    elif request_type == 'INPUT':
        return generate_speech(request)
    else:
        return jsonify({"error": "Invalid request type"}), 400


def add_user(request):
    if 'files[]' not in request.files:
        app.logger.error("No files part in the request")
        return jsonify({"error": "No files part in the request"}), 400

    files = request.files.getlist('files[]')

    if len(files) == 0:
        app.logger.error("No files uploaded")
        return jsonify({"error": "No files uploaded"}), 400

    file_paths = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            file_paths.append(filepath)
        else:
            app.logger.error(f"File {file.filename} is not allowed")
            return jsonify({"error": f"File {file.filename} is not allowed"}), 400

    try:
        api_key = os.getenv('ELEVEN_API_KEY')
        client = ElevenLabs(api_key=api_key)
        provided_api_key = request.headers.get('key')
        provided_name = request.form.get('voice')
        voice_name = f"{provided_api_key}_{provided_name}"
        print(f"Requested voice name: {voice_name}")

        provided_voice_description = request.headers.get('voice_description')

        voices = get_all_voices(client)
        for voice in voices:
            if voice.name == voice_name:
                return jsonify({"message": "Voice already exists", "voice": voice})

        voice = client.clone(
            name=voice_name,
            description=provided_voice_description,
            files=file_paths,
        )
        return jsonify({"message": "Voice cloned successfully", "voice": voice})
    except Exception as e:
        app.logger.error(f"Error cloning voice: {str(e)}")
        return jsonify({"error": f"Error cloning voice: {str(e)}"}), 500


def generate_speech(request):
    try:
        api_key = os.getenv('ELEVEN_API_KEY')
        client = ElevenLabs(api_key=api_key)

        text = request.form.get('text')
        submitted_api_key = request.headers.get('key')
        print(f"Submitted API key: {submitted_api_key}")
        voice_name = "".join(
            (submitted_api_key, "_", request.form.get('voice')))
        print("hello!")

        print(f"REQUESTED VOICE NAME: {voice_name}\n\n\n\n\n\n")
        if not text:
            return jsonify({"error": "Text is required"}), 400

        audio_generator = client.generate(
            text=text,
            voice=voice_name,
            model='eleven_multilingual_v2'
        )

        audio = b''.join(audio_generator)

        audio_io = BytesIO(audio)
        audio_io.seek(0)

        return send_file(audio_io, mimetype='audio/mpeg', as_attachment=True, download_name='speech.mp3')
    except Exception as e:
        app.logger.error(f"Error generating speech: {str(e)}")
        return jsonify({"error": f"Error generating speech: {str(e)}"}), 500


if __name__ == '__main__':
    print("FLASK_API_KEYS:", os.getenv('FLASK_API_KEYS'))
    print("ELEVEN_API_KEY:", os.getenv('ELEVEN_API_KEY'))
    app.run(debug=False)
