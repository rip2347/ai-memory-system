# Copy this to config.py and customize

# LLM Settings
LLM_PROVIDER = "ollama"  # or "openai", "anthropic"
# MODEL_NAME = "qwen2.5:32b"
MODEL_NAME = "qwen2.5:7b"
OLLAMA_URL = "http://localhost:11434"

# Memory Settings
MEMORY_DIR = "./memories"
MAX_MEMORIES_TO_RETRIEVE = 5

# Model Parameters
TEMPERATURE = 0.7
MAX_TOKENS = 2000

# Voice Settings
STT_PROVIDER = "whisper"
STT_MODEL = "small"  # tiny, base, small, medium, large

TTS_PROVIDER = "piper"
TTS_VOICE = "./voices/en_US-amy-medium.onnx"

# Audio Settings
SAMPLE_RATE = 16000
CHANNELS = 1
SILENCE_THRESHOLD = 0.02  # Adjust based on your mic sensitivity
SILENCE_DURATION = 1.5    # Seconds of silence before stopping recording