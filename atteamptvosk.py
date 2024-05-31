import os
import wave
import json
import contextlib
from pydub import AudioSegment
from pyannote.audio import Pipeline
from vosk import Model, KaldiRecognizer
import tempfile

def transcribe_audio(file_path, vosk_model_path):
    # Load the audio file using pydub
    audio = AudioSegment.from_file(file_path)
    
    # Save the audio file as a WAV file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav_file:
        temp_wav_path = temp_wav_file.name
        audio.export(temp_wav_path, format='wav')
    
    # Load the pre-trained speaker diarization model
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
    
    # Perform speaker diarization
    diarization = pipeline(temp_wav_path)
    
    # Load VOSK model for speech recognition
    vosk_model = Model(vosk_model_path)
    
    transcriptions = []
    
    # Process each segment and transcribe the speech
    with contextlib.closing(wave.open(temp_wav_path, 'r')) as audio_file:
        frame_rate = audio_file.getframerate()
        
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            start_time = segment.start
            end_time = segment.end
            
            # Extract the audio segment
            start_frame = int(start_time * frame_rate)
            end_frame = int(end_time * frame_rate)
            audio_file.setpos(start_frame)
            segment_frames = audio_file.readframes(end_frame - start_frame)
            
            # Save the segment as a temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_segment_file:
                temp_segment_path = temp_segment_file.name
                with wave.open(temp_segment_path, 'w') as temp_segment_wav:
                    temp_segment_wav.setparams(audio_file.getparams())
                    temp_segment_wav.writeframes(segment_frames)
            
            # Transcribe the audio segment using VOSK
            with wave.open(temp_segment_path, "rb") as segment_wav:
                rec = KaldiRecognizer(vosk_model, segment_wav.getframerate())
                rec.SetWords(True)
                transcription = []
                
                while True:
                    data = segment_wav.readframes(4000)
                    if len(data) == 0:
                        break
                    if rec.AcceptWaveform(data):
                        result = json.loads(rec.Result())
                        transcription.append(result.get('text', ''))
                final_result = json.loads(rec.FinalResult())
                transcription.append(final_result.get('text', ''))
                text = ' '.join(transcription).strip()
                transcriptions.append((speaker, start_time, end_time, text))
            
            # Clean up the temporary segment file
            os.remove(temp_segment_path)
    
    # Clean up the temporary WAV file
    os.remove(temp_wav_path)
    
    # Print transcriptions
    for transcription in transcriptions:
        speaker, start_time, end_time, text = transcription
        print(f"Speaker {speaker} ({start_time:.2f} - {end_time:.2f}): {text}")

# Example usage
file_path = './small.mp3'
vosk_model_path = 'vosk-model-small-en-us-0.15'  # Path to the VOSK model directory
transcribe_audio(file_path, vosk_model_path)

