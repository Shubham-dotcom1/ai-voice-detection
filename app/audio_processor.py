import base64
import io
import numpy as np
import librosa
from pydub import AudioSegment

def decode_base64_audio(audio_base64: str) -> tuple:
    """
    Decode base64 MP3 audio to numpy array
    Returns: (audio_array, sample_rate)
    """
    try:
        # Remove data URL prefix if present
        if "," in audio_base64:
            audio_base64 = audio_base64.split(",")[1]
        
        # Decode base64 to bytes
        audio_bytes = base64.b64decode(audio_base64)
        
        # Convert MP3 to WAV using pydub
        audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
        
        # Convert to mono if stereo
        if audio_segment.channels > 1:
            audio_segment = audio_segment.set_channels(1)
        
        # Export to WAV format in memory
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)
        
        # Load with librosa at 16kHz
        y, sr = librosa.load(wav_io, sr=16000, mono=True)
        
        return y, sr
    
    except Exception as e:
        raise ValueError(f"Failed to process audio: {str(e)}")


def validate_audio(y: np.ndarray, sr: int) -> bool:
    """Validate audio meets minimum requirements"""
    duration = len(y) / sr
    
    if duration < 0.5:
        raise ValueError("Audio too short. Minimum 0.5 seconds required.")
    
    if duration > 300:
        raise ValueError("Audio too long. Maximum 5 minutes allowed.")
    
    if np.max(np.abs(y)) < 0.001:
        raise ValueError("Audio appears to be silent or corrupted.")
    
    return True