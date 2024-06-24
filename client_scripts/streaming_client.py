import socketio
import pyaudio
import wave
import threading

# Constants for audio recording
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

# Create a Socket.IO client
sio = socketio.Client()

def send_audio():
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* Recording audio...")

    while True:
        try:
            data = stream.read(CHUNK)
            sio.emit('audio_data', {'audio': data}, callback=acknowledgement)
        except KeyboardInterrupt:
            break

    print("* Audio streaming stopped.")

    stream.stop_stream()
    stream.close()
    p.terminate()

def acknowledgement():
    print("Server acknowledged receiving audio chunk")

@sio.event
def connect():
    print('Connected to server')
    threading.Thread(target=send_audio, daemon=True).start()

@sio.event
def disconnect():
    print('Disconnected from server')

if __name__ == '__main__':
    api_key = 'MEMOIZE_KEY'
    sio.connect('http://0.0.0.0:8000?api_key=' + api_key)
    sio.wait()
