import os
import whisper
from pydub import AudioSegment, silence
from pydub.utils import make_chunks
from pyannote.audio import Pipeline
from speechbrain.inference.speaker import SpeakerRecognition
from transformers import AutoTokenizer

from io import BytesIO
import numpy as np
import torch
import torchaudio
import tempfile


class memoizeAudioProccessing:
    def __init__(self, whisper_model_size="base"):
        self.whisper_model = whisper.load_model(whisper_model_size)
        self.verification = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb"
        )

    def diarize(self, input_audio_path, output_dir):
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="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")
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
    
    def diarize_in_memory(self, input_audio_path):
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="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")
        diarization = pipeline(input_audio_path)
        audio = AudioSegment.from_file(input_audio_path)

        speaker_segments = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            segment = audio[turn.start * 1000:turn.end * 1000]
            segment_buffer = BytesIO()
            segment.export(segment_buffer, format="wav")
            speaker_segments[speaker].append(segment_buffer.getvalue())

        return speaker_segments

    def transcribe(self, segment_path):
        result = self.whisper_model.transcribe(segment_path)
        return result['text']
    
    def transcribe_in_memory(self, audio_bytes):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            temp_audio_file.write(audio_bytes)
            temp_audio_file_path = temp_audio_file.name
        
        # Transcribe using Whisper
        result = self.whisper_model.transcribe(temp_audio_file_path)
        
        # Remove the temporary file
        os.remove(temp_audio_file_path)
        
        return result['text']
    
    def transcribe_chunk(self, audio_chunk):
        result = self.whisper_model.transcribe(BytesIO(audio_chunk))
        return result['text']

    def targetSpeakerClassification(self, segment_path, reference_audio_path):
        _, prediction = self.verification.verify_files(
            reference_audio_path, segment_path)
        return prediction
    
    def targetSpeakerClassificationMem(self, segment_bytes, reference_audio_path):
        segment_buffer = BytesIO(segment_bytes)
        segment_tensor, segment_sample_rate = torchaudio.load(segment_buffer, format="wav")

        # Load reference audio and convert to tensor using torchaudio
        reference_tensor, reference_sample_rate = torchaudio.load(reference_audio_path)

        print("work till here")
        prediction = self.verification.verify_batch(segment_tensor, reference_tensor)
        return prediction
    
    def speakerClass(self, segment_byts, reference_audio_path):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            temp_audio_file.write(segment_byts)
            temp_audio_file_path = temp_audio_file.name
        
        _, prediction = self.verification.verify_files(reference_audio_path, temp_audio_file_path)
        
        # Remove the temporary file
        os.remove(temp_audio_file_path)
        
        return prediction

    def process(self, input_audio_path, output_dir, reference_audio_paths):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        speaker_segments = self.diarize(input_audio_path, output_dir)
        transcriptions = []

        for speaker, segments in speaker_segments.items():
            for i, (start, end) in enumerate(segments):
                segment_path = os.path.join(
                    output_dir, f"{i}.wav")
                transcription = self.transcribe(segment_path)
                detected_speaker = False
                for file in os.listdir(reference_audio_paths):
                    is_target_speaker = self.targetSpeakerClassification(segment_path, os.path.join(reference_audio_paths,file))
                    if is_target_speaker: 
                        transcriptions.append(f"{transcription} ({file})")
                        detected_speaker = True

                if not detected_speaker:
                    transcriptions.append(f"{transcription} (Unknown Speaker)")
                
        file_path = os.path.join(output_dir, "transcription.txt")
        with open(file_path, "w") as f:
            for transcription in transcriptions:
                f.write(transcription + "\n")
        return file_path
    
    def multispeaker(self, input_audio_path, reference_audio_paths, output_dir):
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="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")
        diarization = pipeline(input_audio_path)
        audio = AudioSegment.from_file(input_audio_path)

        speaker_segments = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            segment = audio[turn.start * 1000: turn.end * 1000]
            segment_buffer = BytesIO()
            segment.export(segment_buffer, format="wav")
            speaker_segments[speaker].append(segment_buffer.getvalue())

        transcriptions = []

        for speaker, segments in speaker_segments.items():
            for i, segment_bytes in enumerate(segments):
                transcription = self.transcribe_in_memory(segment_bytes)
                #print("transcription worked")
                detected_speaker = False
                for file in os.listdir(reference_audio_paths):
                    is_target_speaker = self.speakerClass(segment_bytes, os.path.join(reference_audio_paths, file))
                    #print("speaker detection worked")
                    if is_target_speaker: 
                        transcriptions.append(f"{transcription} ({file})")
                        detected_speaker = True
                        break

                if not detected_speaker:
                    transcriptions.append(f"{transcription} (Unknown Speaker)")

        file_path = os.path.join(output_dir, "transcription.txt")
        if os.path.exists(file_path):
            mode = 'a'
        else: 
            mode = 'w'

        with open(file_path, mode) as f:
            for transcription in transcriptions:
                f.write(transcription + "\n")
        return file_path
    
    def multispeaker_silence(self, input_audio_path, reference_audio_paths, output_dir):
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="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")
        diarization = pipeline(input_audio_path)
        audio = AudioSegment.from_file(input_audio_path)

        speaker_segments = {}
        all_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            segment = audio[turn.start * 1000: turn.end * 1000]
            segment_buffer = BytesIO()
            segment.export(segment_buffer, format="wav")
            speaker_segments[speaker].append((segment_buffer.getvalue(), turn.start, turn.end))
            all_segments.append((turn.start, turn.end, speaker))

        transcriptions = []

        # Sort all segments to detect silence patches
        all_segments.sort()

        # Add initial silence if the first segment does not start at the beginning
        if all_segments[0][0] > 0:
            transcriptions.append("<Break>")

        for i in range(len(all_segments)):
            start, end, speaker = all_segments[i]

            # Detect silence between segments
            if i > 0 and all_segments[i-1][1] < start:
                transcriptions.append("<Break>")

            # Process speaker segments
            for segment_bytes, seg_start, seg_end in speaker_segments[speaker]:
                transcription = self.transcribe_in_memory(segment_bytes)
                detected_speaker = False
                for file in os.listdir(reference_audio_paths):
                    is_target_speaker = self.speakerClass(segment_bytes, os.path.join(reference_audio_paths, file))
                    if is_target_speaker: 
                        transcriptions.append(f"{transcription} ({file})")
                        detected_speaker = True
                        break

                if not detected_speaker:
                    transcriptions.append(f"{transcription} (Unknown Speaker)")

            # Detect silence at the end of the last segment
            if i == len(all_segments) - 1 and end < len(audio) / 1000:
                transcriptions.append("<Break>")

        file_path = os.path.join(output_dir, "transcription.txt")
        mode = 'a' if os.path.exists(file_path) else 'w'

        with open(file_path, mode) as f:
            for transcription in transcriptions:
                f.write(transcription + "\n")

        return file_path

    def tokenize(self, transcription_file_path):
        with open(transcription_file_path, "r") as f:
            transcriptions = f.read()

        tokens = self.tokenizer(
            transcriptions, return_tensors="pt", padding=True, truncation=True)
        return tokens

