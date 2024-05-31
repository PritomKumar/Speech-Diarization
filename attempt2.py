import os
import wave
import contextlib
from pydub import AudioSegment
import speech_recognition as sr
from pyannote.audio import Pipeline
import tempfile
from config import AUTH_TOKEN

def transcribe_audio(file_path):
    # Load the audio file using pydub
    audio = AudioSegment.from_file(file_path)
    
    # Save the audio file as a WAV file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav_file:
        temp_wav_path = temp_wav_file.name
        audio.export(temp_wav_path, format='wav')
    
    # Load the pre-trained speaker diarization model
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token= AUTH_TOKEN)
    
    # Perform speaker diarization
    diarization = pipeline(temp_wav_path)
    
    recognizer = sr.Recognizer()
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
            
            # Transcribe the audio segment
            with sr.AudioFile(temp_segment_path) as source:
                audio_data = recognizer.record(source)
                try:
                    text = recognizer.recognize_google(audio_data)
                    transcriptions.append((speaker, start_time, end_time, text))
                except sr.UnknownValueError:
                    transcriptions.append((speaker, start_time, end_time, "[Inaudible]"))
                except sr.RequestError as e:
                    transcriptions.append((speaker, start_time, end_time, f"[Error: {e}]"))
            
            # Clean up the temporary segment file
            os.remove(temp_segment_path)
    
    # Clean up the temporary WAV file
    os.remove(temp_wav_path)
    
    # Print transcriptions
    result_string = ""
    for transcription in transcriptions:
        speaker, start_time, end_time, text = transcription
        print(f"Speaker {speaker} ({start_time:.2f} - {end_time:.2f}): {text}")
        transcript_values = f"Speaker {speaker} ({start_time:.2f} - {end_time:.2f}): {text}\n"
        result_string += transcript_values

    result_filename = "result_text_multiple.txt"
    with open(result_filename, "w") as text_file:
        text_file.write(result_string)

# Example usage
# file_path = '.\small.mp3' # 2 speakers
file_path = '.\multiple.mp3' # multiple speakers
transcribe_audio(file_path)
