"""
Download Piper TTS voices
Run this once to get voice files for text-to-speech.
"""

import os
import urllib.request
from pathlib import Path

# Voice URLs
VOICES = {
    "en_US-amy-medium": {
        "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx",
        "json": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx.json"
    }
}

def download_voice(voice_name):
    """Download voice files."""
    
    # Create voices directory
    voices_dir = Path("voices")
    voices_dir.mkdir(exist_ok=True)
    
    voice_files = VOICES[voice_name]
    
    print(f"Downloading {voice_name}...")
    
    for file_type, url in voice_files.items():
        filename = f"{voice_name}.{file_type}"
        filepath = voices_dir / filename
        
        if filepath.exists():
            print(f"  ✓ {filename} already exists")
            continue
        
        print(f"  Downloading {filename}...")
        
        try:
            urllib.request.urlretrieve(url, filepath)
            print(f"  ✓ {filename} downloaded")
        except Exception as e:
            print(f"  ✗ Failed to download {filename}: {e}")
            return False
    
    print(f"\n✓ {voice_name} ready!")
    return True

if __name__ == "__main__":
    print("=" * 70)
    print("Piper TTS Voice Downloader")
    print("=" * 70)
    
    download_voice("en_US-amy-medium")
    
    print("\nVoice files saved to ./voices/")
    print("You're ready to use voice chat!")