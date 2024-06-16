import pyaudio
import requests
import time

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

print("Recording...")

try:
    while True:
        try:
            data = stream.read(CHUNK)
            stream.stop_stream()
            stream.close()
            audio.terminate()
            response = requests.post('http://127.0.0.1:5000/voice_sample', data=data, headers={'key': 'MEMOIZE_KEY'})
            print(response.text)
        except OSError as e:
            # if e.errno == -9981:
            #     CHUNK //= 2
            #     print("Recording...")
            #     stream = audio.open(format=FORMAT, channels=CHANNELS,
            #                         rate=RATE, input=True,
            #                         frames_per_buffer=CHUNK)
            # else:
            raise 

        time.sleep(0.05) 
except KeyboardInterrupt:
    print("Stopping...")


