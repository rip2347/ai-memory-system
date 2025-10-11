# Copy this to config.py and customize

# LLM Settings
LLM_PROVIDER = "ollama"  # or "openai", "anthropic"
MODEL_NAME = "qwen2.5:32b"
OLLAMA_URL = "http://localhost:11434"

# Memory Settings
MEMORY_DIR = "./memories"
MAX_MEMORIES_TO_RETRIEVE = 5

# Model Parameters
TEMPERATURE = 0.7
MAX_TOKENS = 2000
