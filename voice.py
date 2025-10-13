"""
Voice Interface Abstraction
Provides swappable speech-to-text and text-to-speech providers.
"""

import whisper
import numpy as np
import sounddevice as sd
import soundfile as sf
from typing import Optional
import tempfile
import os

from config import (
    STT_PROVIDER, STT_MODEL, 
    TTS_PROVIDER, TTS_VOICE,
    SAMPLE_RATE, CHANNELS,
    SILENCE_THRESHOLD, SILENCE_DURATION
)


class VoiceInterface:
    """Unified interface for speech-to-text and text-to-speech."""
    
    def __init__(self, stt_provider: str = None, tts_provider: str = None):
        self.stt_provider = stt_provider or STT_PROVIDER
        self.tts_provider = tts_provider or TTS_PROVIDER
        
        # Initialize STT
        if self.stt_provider == "whisper":
            print(f"Loading Whisper model: {STT_MODEL}...")
            self.stt_model = whisper.load_model(STT_MODEL)
            print("✓ Whisper ready")
        else:
            raise ValueError(f"Unknown STT provider: {self.stt_provider}")
        
        # Initialize TTS
        if self.tts_provider == "piper":
            print("Loading Piper TTS...")
            try:
                from piper import PiperVoice
                self.tts_model = PiperVoice.load(TTS_VOICE)
                print(f"✓ Piper ready with voice: {TTS_VOICE}")
            except Exception as e:
                print(f"⚠ Piper voice not found: {e}")
                print("TTS will be disabled. Download voices from:")
                print("https://github.com/rhasspy/piper/releases")
                self.tts_model = None
        else:
            raise ValueError(f"Unknown TTS provider: {self.tts_provider}")
    
    def listen(self, duration: Optional[int] = None) -> str:
        """
        Record audio from microphone and transcribe to text.
        
        Args:
            duration: Max recording duration in seconds (None = auto-stop on silence)
        
        Returns:
            Transcribed text
        """
        print("\n🎤 Listening...")
        
        if duration:
            # Fixed duration recording
            audio = sd.rec(
                int(duration * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype='float32'
            )
            sd.wait()
        else:
            # Auto-stop on silence
            audio = self._record_until_silence()
        
        print("🔄 Transcribing...")
        
        # Transcribe directly from audio array (no file needed!)
        # Flatten stereo to mono if needed
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Whisper expects float32 in range [-1, 1]
        audio = audio.flatten().astype('float32')
        
        # Transcribe directly
        result = self.stt_model.transcribe(audio, fp16=False)
        text = result["text"].strip()
        return text
    
    def speak(self, text: str):
        """
        Convert text to speech and play.
        
        Args:
            text: Text to speak
        """
        if not self.tts_model:
            print(f"\n[TTS disabled] Would say: {text}\n")
            return
        
        print("🔊 Speaking...")
        
        try:
            # Collect all audio chunks
            audio_bytes = b''
            sample_rate = SAMPLE_RATE
            
            for chunk in self.tts_model.synthesize(text):
                sample_rate = chunk.sample_rate
                audio_bytes += chunk.audio_int16_bytes
            
            # Convert bytes to numpy array
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            
            # Convert to float32 for sounddevice
            audio_float = audio_int16.astype(np.float32) / 32768.0
            
            # Play audio
            sd.play(audio_float, sample_rate)
            sd.wait()
            
        except Exception as e:
            print(f"⚠ TTS error: {e}")
            print(f"Would say: {text}")
    
    def _record_until_silence(self) -> np.ndarray:
        """
        Record audio until silence is detected.
        
        Returns:
            Recorded audio as numpy array
        """
        chunks = []
        silence_chunks = 0
        chunk_duration = 0.1  # 100ms chunks
        chunk_samples = int(SAMPLE_RATE * chunk_duration)
        max_silence_chunks = int(SILENCE_DURATION / chunk_duration)
        
        # Callback to capture audio chunks
        def callback(indata, frames, time, status):
            if status:
                print(f"Audio status: {status}")
            chunks.append(indata.copy())
        
        # Start recording
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            callback=callback,
            blocksize=chunk_samples
        ):
            print("Speak now... (will auto-stop on silence)")
            
            while True:
                sd.sleep(int(chunk_duration * 1000))
                
                if len(chunks) > 0:
                    # Check if current chunk is silent
                    current_chunk = chunks[-1]
                    volume = np.abs(current_chunk).mean()
                    
                    if volume < SILENCE_THRESHOLD:
                        silence_chunks += 1
                        if silence_chunks >= max_silence_chunks:
                            print("✓ Silence detected, stopping...")
                            break
                    else:
                        silence_chunks = 0  # Reset on sound
                    
                    # Safety: max 30 seconds
                    if len(chunks) > 300:
                        print("⚠ Max recording time reached")
                        break
        
        # Combine all chunks
        if chunks:
            audio = np.concatenate(chunks, axis=0)
            return audio
        else:
            return np.array([])


def test_voice():
    """Test the voice interface."""
    
    print("=" * 70)
    print("Voice Interface Test")
    print("=" * 70)
    
    voice = VoiceInterface()
    
    print("\nTest 1: Record and transcribe")
    print("Press Enter when ready to speak...")
    input()
    
    text = voice.listen()
    print(f"\nYou said: {text}")
    
    print("\nTest 2: Text-to-speech")
    voice.speak("Hello! I am your AI assistant with voice capabilities.")
    
    print("\n✓ Voice test complete!")


if __name__ == "__main__":
    test_voice()