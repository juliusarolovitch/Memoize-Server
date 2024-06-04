from flask import Flask, request, jsonify, send_file
import os
import logging
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv, find_dotenv
from io import BytesIO
from GPT import Text

load_dotenv(find_dotenv('keys.env'))


class Server:
    def __init__(self):
        self.app = Flask(__name__)
        self.app.config['UPLOAD_FOLDER'] = 'uploads'
        self.app.config['ALLOWED_EXTENSIONS'] = {'mp4'}
        self.setupLogging()
        self.setupUploadFolder()
        self.setupRoutes()

    def setupLogging(self):
        logging.basicConfig(level=logging.DEBUG)

    def setupUploadFolder(self):
        if not os.path.exists(self.app.config['UPLOAD_FOLDER']):
            os.makedirs(self.app.config['UPLOAD_FOLDER'])

    def setupRoutes(self):
        self.app.add_url_rule('/process', 'processRequest',
                              self.processRequest, methods=['POST'])

    def allowedFile(self, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in self.app.config['ALLOWED_EXTENSIONS']

    def authenticateApiKey(self, request):
        provided_api_key = request.headers.get('key')
        valid_api_keys = os.getenv('FLASK_API_KEYS').split(',')
        self.app.logger.debug(f"Provided API key: {provided_api_key}")
        self.app.logger.debug(f"Valid API keys: {valid_api_keys}")
        return provided_api_key in valid_api_keys

    def getAllVoices(self, client):
        try:
            response = client.voices.get_all()
            return response.voices
        except Exception as e:
            self.app.logger.error(f"Error getting voices: {str(e)}")
            return []

    def processRequest(self):
        if not self.authenticateApiKey(request):
            return jsonify({"error": "Unauthorized"}), 401

        request_type = request.headers.get('request-type')

        if request_type == 'ADD_USER':
            return self.addUser(request)
        elif request_type == 'INPUT':
            return self.generateSpeech(request)
        else:
            return jsonify({"error": "Invalid request type"}), 400

    def addUser(self, request):
        if 'files[]' not in request.files:
            self.app.logger.error("No files part in the request")
            return jsonify({"error": "No files part in the request"}), 400

        files = request.files.getlist('files[]')

        if len(files) == 0:
            self.app.logger.error("No files uploaded")
            return jsonify({"error": "No files uploaded"}), 400

        file_paths = []
        for file in files:
            if file and self.allowedFile(file.filename):
                filename = file.filename
                filepath = os.path.join(
                    self.app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                file_paths.append(filepath)
            else:
                self.app.logger.error(f"File {file.filename} is not allowed")
                return jsonify({"error": f"File {file.filename} is not allowed"}), 400

        try:
            api_key = os.getenv('ELEVEN_API_KEY')
            client = ElevenLabs(api_key=api_key)
            provided_api_key = request.headers.get('key')
            provided_name = request.form.get('voice')
            voice_name = f"{provided_api_key}_{provided_name}"
            self.app.logger.debug(f"Requested voice name: {voice_name}")

            provided_voice_description = request.headers.get(
                'voice_description')

            voices = self.getAllVoices(client)
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
            self.app.logger.error(f"Error cloning voice: {str(e)}")
            return jsonify({"error": f"Error cloning voice: {str(e)}"}), 500

    def generateSpeech(self, request):
        try:
            api_key = os.getenv('ELEVEN_API_KEY')
            client = ElevenLabs(api_key=api_key)

            text = request.form.get('text')
            submitted_api_key = request.headers.get('key')
            self.app.logger.debug(f"Submitted API key: {submitted_api_key}")
            voice_name = f"{submitted_api_key}_{request.form.get('voice')}"
            self.app.logger.debug(f"Requested voice name: {voice_name}")

            if not text:
                return jsonify({"error": "Text is required"}), 400
            
            # convert text to GPT response
            text = Text(text, "OPENAI_API_KEY").to_gpt()
            # now text is GPT's response

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
            self.app.logger.error(f"Error generating speech: {str(e)}")
            return jsonify({"error": f"Error generating speech: {str(e)}"}), 500

    def run(self):
        self.app.run(debug=False)


if __name__ == '__main__':
    server = Server()
    server.run()
