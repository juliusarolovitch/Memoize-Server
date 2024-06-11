import os
import whisper
from pydub import AudioSegment
from pyannote.audio import Pipeline
from speechbrain.inference.speaker import SpeakerRecognition
from transformers import AutoTokenizer
import ssl
import urllib.request
import requests
from io import BytesIO


class memoizeAudioProccessing:
    def __init__(self, whisper_model_size="base"):
        self.whisper_model = whisper.load_model(whisper_model_size)
        self.verification = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb"
        )

    def diarize(self, input_audio_path, output_dir):
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="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")
        diarization = pipeline(input_audio_path)
        audio = AudioSegment.from_file(input_audio_path)

        speaker_segments = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            speaker_segments[speaker].append((turn.start, turn.end))

        for speaker, segments in speaker_segments.items():
            for i, (start, end) in enumerate(segments):
                segment = audio[start * 1000:end * 1000]
                segment_path = os.path.join(
                    output_dir, f"{speaker}_segment_{i}.wav")
                segment.export(segment_path, format="wav")
        
        return speaker_segments

    def transcribe(self, segment_path):
        result = self.whisper_model.transcribe(segment_path)
        return result['text']
    
    def transcribe_chunk(self, audio_chunk):
        result = self.whisper_model.transcribe(BytesIO(audio_chunk))
        return result['text']

    def targetSpeakerClassification(self, segment_path, reference_audio_path):
        _, prediction = self.verification.verify_files(
            reference_audio_path, segment_path)
        return prediction

    def process(self, input_audio_path, output_dir, reference_audio_path):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        speaker_segments = self.diarize(input_audio_path, output_dir)
        transcriptions = []

        for speaker, segments in speaker_segments.items():
            for i, (start, end) in enumerate(segments):
                segment_path = os.path.join(
                    output_dir, f"{speaker}_segment_{i}.wav")
                transcription = self.transcribe(segment_path)
                is_target_speaker = self.targetSpeakerClassification(
                    segment_path, reference_audio_path)

                if is_target_speaker:
                    transcriptions.append(f"{transcription} (Target Speaker)")
                else:
                    transcriptions.append(f"{transcription} (Other Speaker)")
        file_path = os.path.join(output_dir, "transcription.txt")
        with open(file_path, "w") as f:
            for transcription in transcriptions:
                f.write(transcription + "\n")
        return file_path

    def tokenize(self, transcription_file_path):
        with open(transcription_file_path, "r") as f:
            transcriptions = f.read()

        tokens = self.tokenizer(
            transcriptions, return_tensors="pt", padding=True, truncation=True)
        return tokens

