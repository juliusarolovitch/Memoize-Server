from flask import Flask, request, jsonify, send_file
from flask_socketio import SocketIO, emit, disconnect
import os
import logging
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
from io import BytesIO
from pydub import AudioSegment
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from src.GPT import Text
import base64
import numpy as np
import sounddevice as sd
import soundfile as sf
from memoize_audio_processing import memoizeAudioProccessing
from src.finetune import FineTune
from src.vision import Images, Video
import hashlib
import time
import threading 
from src.finetune import FineTune
from src.vision import Images, Video
import hashlib
import wave
load_dotenv()

class processingThread(threading.Thread):
    def __init__(self, audio_file_path, samples_dir, output_dir): 
        threading.Thread.__init__(self)
        self.audio= audio_file_path
        self.samples_dir = samples_dir
        self.output_dir = output_dir
        self.processor = memoizeAudioProccessing()

    def run(self):
        results = self.processor.multispeaker_silence(self.audio, self.samples_dir, self.output_dir)
        return
    
# WebSocket Manager to handle multiple clients
class WebSocketManager:
    def __init__(self, base_audio_dir, logger):
        self.clients = {}
        self.audio_buffers = {}  # Buffer to hold incoming audio data
        self.chunk_size = 5 * 44100  # 5 seconds of audio at 44100 sample rate (samples/sec)
        self.sample_rate = 44100
        self.channels = 1
        self.width = 2  # Sample width in bytes (16-bit)
        self.client_dirs = {}  # Dictionary to store client directories
        self.api_keys = [os.getenv('FLASK_API_KEYS')]
        self.base_audio_dir = base_audio_dir
        self.logger = logger
        self.audio_processor = memoizeAudioProccessing()
        self.reference_audio_folder = os.path.join(os.getcwd(), 'samples')
        self.transcriptions_folder = os.path.join(os.getcwd(), 'inference')

    def add_client(self, sid, api_key):
        print("API Keys:", self.api_keys)
        if api_key not in self.api_keys:
            self.logger.warning(f'Unauthorized access with API key: {api_key}')
            disconnect()
            return

        self.clients[sid] = {
            'audio_data': [],
            'file_count': 0  # Counter for saved audio files
        }
        self.audio_buffers[sid] = b''  # Initialize buffer as bytes object
        self.client_dirs[sid] = os.path.join(self.base_audio_dir, sid)
        self.create_client_directory(sid)
        self.logger.info(f'Client added: {sid}')

    def remove_client(self, sid):
        if sid in self.clients:
            chunk_data = self.audio_buffers[sid]
            self.save_audio_chunk(sid, chunk_data)  # Save any remaining audio data
            del self.clients[sid]
            del self.audio_buffers[sid]
            del self.client_dirs[sid]
            self.logger.info(f'Client removed: {sid}')

    def add_audio_data(self, sid, data):
        if sid in self.clients:
            self.audio_buffers[sid] += data.tobytes()  # Append audio chunk as bytes

            # Check if enough data for a chunk
            while len(self.audio_buffers[sid]) >= self.chunk_size:
                chunk_data = self.audio_buffers[sid][:self.chunk_size]
                self.audio_buffers[sid] = self.audio_buffers[sid][self.chunk_size:]

                self.save_audio_chunk(sid, chunk_data)

    def save_audio_chunk(self, sid, chunk_data):
        file_count = self.clients[sid]['file_count']
        file_path = os.path.join(self.client_dirs[sid], f'chunk_{file_count + 1}.wav')

        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.width)
            wf.setframerate(self.sample_rate)
            wf.writeframes(chunk_data)
        
        sid_output_folder = os.path.join(self.client_dirs[sid], "transcription")
        if not os.path.exists(sid_output_folder):
            os.makedirs(sid_output_folder)
        
        # Audio processing for transcription and speaker detection occurs here
        processor = processingThread(file_path, self.reference_audio_folder, sid_output_folder)
        processor.start()

        self.logger.info(f'Chunk {file_count + 1} saved for client {sid} at {file_path}')
        self.clients[sid]['file_count'] += 1

    def create_client_directory(self, sid):
        client_dir = self.client_dirs[sid]
        if not os.path.exists(client_dir):
            os.makedirs(client_dir)

    def get_audio_data(self, sid):
        if sid in self.clients:
            return self.clients[sid]['audio_data']

class Server:
    def __init__(self):
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, async_mode='eventlet')
        self.app.config['UPLOAD_FOLDER'] = 'uploads'
        self.app.config['ALLOWED_EXTENSIONS'] = {'mp3', 'enc', 'mp4'}
        self.app.config['AUDIO_INFERENCE_FOLDER'] = 'inference'
        self.setupLogging()
        self.setupFolders()
        self.setupRoutes()
        self.setupSocketHandlers()
        self.ws_manager = WebSocketManager(self.app.config['AUDIO_INFERENCE_FOLDER'], self.app.logger)
        self.encryption_key = base64.urlsafe_b64decode(
            os.getenv('ENCRYPTION_KEY') + '===')
        self.audio_processor = memoizeAudioProccessing()
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.llm = 'gpt-4o'
        self.current_text = "" #current text coming from environment
        self.reference_audio_folder = os.path.join(os.getcwd(), "samples")
        self.transcriptions_folder = os.path.join(os.getcwd(), "inference")
        self.server_dir = os.getcwd()
        
    def derive_iv(self, data):
        hash_digest = hashlib.sha256(data.encode()).digest()
        return hash_digest[:16]
    
    def encrypt(self, data):
        iv = self.derive_iv(data)

        cipher = Cipher(algorithms.AES(self.encryption_key), modes.CFB(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        encrypted_data = encryptor.update(data.encode()) + encryptor.finalize()
        return base64.urlsafe_b64encode(iv + encrypted_data).decode()

    def decrypt(self, encrypted_data):
        encrypted_data_bytes = base64.urlsafe_b64decode(encrypted_data)
        iv = encrypted_data_bytes[:16]
        ciphertext = encrypted_data_bytes[16:]
        cipher = Cipher(algorithms.AES(self.encryption_key), modes.CFB(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        decrypted_data = decryptor.update(ciphertext) + decryptor.finalize()
        return decrypted_data.decode()

    def setupLogging(self):
        logging.basicConfig(level=logging.DEBUG)

    def setupFolders(self):
        if not os.path.exists(self.app.config['UPLOAD_FOLDER']):
            os.makedirs(self.app.config['UPLOAD_FOLDER'])
        if not os.path.exists(self.app.config['AUDIO_INFERENCE_FOLDER']): 
            os.makedirs(self.app.config['AUDIO_INFERENCE_FOLDER'])

    def setupRoutes(self):
        self.app.add_url_rule('/process', 'processRequest', self.processRequest, methods=['POST'])
        # self.app.add_url_rule('/audio', 'audioStream', self.audioStream, methods=['POST'])
        # self.app.add_url_rule('/voice_sample', 'recordTrainingAudio', self.recordTrainingAudio, methods=['POST'])

    def setupSocketHandlers(self):
        @self.socketio.on('connect')
        def handle_connect():
            sid = request.sid
            api_key = request.args.get('api_key')

            if api_key in self.ws_manager.api_keys:
                self.ws_manager.add_client(sid, api_key)
                self.app.logger.info(f'Client connected: {sid}')
                emit('connect_ack', {'status': 'connected'})
            else: 
                self.logger.warning(f'Invalid API Key provided {sid}')
                disconnect()
                return 
            

        @self.socketio.on('disconnect')
        def handle_disconnect():
            sid = request.sid
            self.ws_manager.remove_client(sid)
            self.app.logger.info(f'Client disconnected: {sid}')

        @self.socketio.on('audio_data')
        def handle_audio(data):
            sid = request.sid
            audio_chunk = np.frombuffer(data['audio'], dtype=np.int16)
            self.ws_manager.add_audio_data(sid, audio_chunk)
            emit('ack_audio', {'status': 'received'})
            #self.logger.info(f'Received audio data from client {sid}')

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

    def decrypt_file(self, file_path, encryption_key):
        with open(file_path, 'rb') as file:
            encrypted_data = file.read()

        iv = encrypted_data[:16]
        ciphertext = encrypted_data[16:]
        cipher = Cipher(algorithms.AES(encryption_key),
                        modes.CFB(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        decrypted_data = decryptor.update(ciphertext) + decryptor.finalize()

        decrypted_file_path = file_path.rstrip('.enc')
        with open(decrypted_file_path, 'wb') as file:
            file.write(decrypted_data)

        return decrypted_file_path

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
        total_duration = 0
        files_encrypted = request.form.get(
            'files_encrypted', 'false') == 'true'

        for file in files:
            if file and self.allowedFile(file.filename):
                filename = file.filename
                filepath = os.path.join(
                    self.app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                if files_encrypted:
                    filepath = self.decrypt_file(filepath, self.encryption_key)

                file_paths.append(filepath)

                try:
                    audio = AudioSegment.from_file(filepath)
                    duration = len(audio) / 1000  # Convert to seconds
                    total_duration += duration

                    self.app.logger.debug(
                        f"File {filename} duration: {duration} seconds")
                except Exception as e:
                    self.app.logger.error(
                        f"File {filename} is corrupted or invalid: {str(e)}")
                    return jsonify({"error": f"File {filename} is corrupted or invalid"}), 400
            else:
                self.app.logger.error(f"File {file.filename} is not allowed")
                return jsonify({"error": f"File {file.filename} is not allowed"}), 400

        if total_duration < 30:
            self.app.logger.error(
                "Total duration of files is less than 30 seconds")
            return jsonify({"error": "Total duration of files must be at least 30 seconds"}), 400

        try:
            api_key = os.getenv('ELEVEN_API_KEY')
            client = ElevenLabs(api_key=api_key)
            provided_api_key = request.headers.get('key')
            provided_name = request.form.get('voice')
            encrypted_voice_name = self.encrypt(
                f"{provided_api_key}~{provided_name}")
            self.app.logger.debug(
                f"Requested encrypted voice name: {encrypted_voice_name}")

            provided_voice_description = request.headers.get(
                'voice_description')

            voices = self.getAllVoices(client)
            for voice in voices:
                if voice.name == encrypted_voice_name:
                    return jsonify({"message": "Voice already exists", "voice": voice})

            voice = client.clone(
                name=encrypted_voice_name,
                description=provided_voice_description,
                files=file_paths,
            )
            return jsonify({"message": "Voice cloned successfully", "voice": voice})
        except Exception as e:
            self.app.logger.error(f"Error cloning voice: {str(e)}")
            return jsonify({"error": f"Error cloning voice: {str(e)}"}), 500
        
    def finetuneResponses(self, training_file):
        fntune = FineTune(training_file, self.openai_api_key)
        self.llm = fntune.train()
        return self.llm
    
    def scene_to_text(self, path):
        # describe video feed, local
        video = Video(path, 'Describe what you see in the video.', self.openai_api_key)
        return video.video_prompt(frame_rate=50)
    
    def format_prompt(self, user, speaker_data, video = False, video_path = None):

        prompt = "User: " + user + "."
        scene = "Unknown"

        if video:
            scene = self.scene_to_text(video_path)
        str_to_add = "Scene: " + scene

        prompt += str_to_add

        for speaker in list(speaker.keys()):
            msg = speaker_data[speaker]
            str_to_add = speaker + ": " + msg 
            prompt += str_to_add

        return prompt

    def send_prompts(self):
        curr_text = self.current_text # or query it
        time.sleep(1.00)
        next_text = self.current_text

        if curr_text == next_text:
            self.generateSpeech("INPUT")  # call prompt
        else: # keep the code running, speaker till speaking
            pass

    def generateSpeech(self, request):
        try:
            api_key = os.getenv('ELEVEN_API_KEY')
            client = ElevenLabs(api_key=api_key)

            text = request.form.get('text')
            submitted_api_key = request.headers.get('key')
            self.app.logger.debug(f"Submitted API key: {submitted_api_key}")
            provided_name = request.form.get('voice')
            encrypted_voice_name = self.encrypt(
                f"{submitted_api_key}~{provided_name}")
            self.app.logger.debug(
                f"Requested encrypted voice name: {encrypted_voice_name}")

            if not text:
                return jsonify({"error": "Text is required"}), 400
            
            #format prompt to final input format
            #call self.format_prompt

            # convert text to GPT response
            text = Text(text, self.openai_api_key, self.llm).to_gpt()

            audio_generator = client.generate(
                text=text,
                voice=encrypted_voice_name,
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
        self.socketio.run(self.app, host='0.0.0.0', port=8000, debug=False)
        self.app.logger.debug("Inference server started")

if __name__ == '__main__':
    server = Server()
    server.run()