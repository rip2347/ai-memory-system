# AI Memory System

Long-term conversational memory for AI assistants.

## Setup

1. Install Ollama: `curl -fsSL https://ollama.com/install.sh | sh`
2. Pull model: `ollama pull qwen2.5:32b`
3. Copy config: `cp config.example.py config.py`
4. Install dependencies: `pip install requests`

## Usage

Coming soon...

## Architecture

- `llm.py` - LLM abstraction layer
- `conversation.py` - Process and save conversations  
- `memory.py` - Retrieve relevant memories
- `config.py` - Your local settings

## Privacy

Conversations stored in `memories/` are gitignored and stay local.
