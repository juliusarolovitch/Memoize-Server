import asyncio
import websockets
import json
import base64
import shutil
import os
import subprocess
import time
from openai import AsyncOpenAI

OPENAI_API_KEY = 'replace'
ELEVENLABS_API_KEY = 'replace'
VOICE_ID = '21m00Tcm4TlvDq8ikWAM'

aclient = AsyncOpenAI(api_key=OPENAI_API_KEY)

def is_installed(lib_name):
    return shutil.which(lib_name) is not None

async def text_chunker(chunks):
    """Split text into chunks, ensuring to not break sentences."""
    splitters = (".", ",", "?", "!", ";", ":", "—", "-", "(", ")", "[", "]", "}", " ")
    buffer = ""

    async for text in chunks:
        if text is None:
            continue

        if buffer.endswith(splitters):
            yield buffer + " "
            buffer = text
        elif text.startswith(splitters):
            yield buffer + text[0] + " "
            buffer = text[1:]
        else:
            buffer += text

    if buffer:
        yield buffer + " "

async def stream(audio_stream, start_time):
    """Stream audio data using ffplay player."""
    if not is_installed("ffplay"):
        raise ValueError(
            "ffplay not found, necessary to stream audio. "
            "Install instructions: https://ffmpeg.org/download.html"
        )

    ffplay_process = subprocess.Popen(
        ["ffplay", "-nodisp", "-autoexit", "-"],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    first_audio_chunk = True
    async for chunk in audio_stream:
        if chunk:
            if first_audio_chunk:
                first_audio_chunk = False
                elapsed_time = time.time() - start_time
                print(f"Time to first audio: {elapsed_time:.2f} seconds")
            ffplay_process.stdin.write(chunk)
            ffplay_process.stdin.flush()

    if ffplay_process.stdin:
        ffplay_process.stdin.close()
    ffplay_process.wait()

async def text_to_speech_input_streaming(voice_id, text_iterator, start_time):
    """Send text to ElevenLabs API and stream the returned audio."""
    uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input?model_id=eleven_turbo_v2"

    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps({
            "text": " ",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
            "xi_api_key": ELEVENLABS_API_KEY,
        }))

        async def listen():
            """Listen to the websocket for audio data and stream it."""
            while True:
                try:
                    message = await websocket.recv()
                    data = json.loads(message)
                    if data.get("audio"):
                        yield base64.b64decode(data["audio"])
                    elif data.get('isFinal'):
                        break
                except websockets.exceptions.ConnectionClosed:
                    print("Connection closed")
                    break

        listen_task = asyncio.create_task(stream(listen(), start_time))

        async for text in text_chunker(text_iterator):
            await websocket.send(json.dumps({"text": text, "try_trigger_generation": True}))

        await websocket.send(json.dumps({"text": ""}))

        await listen_task

async def chat_completion(query):
    """Retrieve text from OpenAI and pass it to the text-to-speech function."""
    response = await aclient.chat.completions.create(model='gpt-3.5-turbo', messages=[{'role': 'user', 'content': query}],
    temperature=1, stream=True)

    async def text_iterator():
        async for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content is not None:
                yield delta.content

    start_time = time.time()  
    await text_to_speech_input_streaming(VOICE_ID, text_iterator(), start_time)

if __name__ == "__main__":
    user_query = "Tell me about why Carnegie Mellon is the best school on Earth"
    asyncio.run(chat_completion(user_query))
