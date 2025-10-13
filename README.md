# AI Memory System

Long-term conversational memory for AI assistants with voice support.

## Features

- 🧠 **Semantic Memory**: Retrieves relevant past conversations automatically
- 🎤 **Voice Input**: Speech-to-text using Whisper (local, private)
- 🔊 **Voice Output**: Text-to-speech using Piper (local, private)
- 🔄 **Swappable Providers**: Easy to switch between LLM providers (Ollama, OpenAI, Anthropic)
- 🔒 **Privacy First**: All data stays local - no cloud required

## Setup

### 1. Install Ollama
bash
curl -fsSL https://ollama.com/install.sh | sh

### 2. Pull a Model
ollama pull qwen2.5:7b

### 3. Clone Repository
git clone https://github.com/rip2347/ai-memory-system.git
cd ai-memory-system

### 4. Install Dependencies
pip install -r requirements.txt

### 5. Download Voice Files (Optional - for voice chat)
python download_voices.py
Usage
Text Chat
python chat.py
Voice Chat
python voice_chat.py

Press ENTER to start recording
Speak naturally - auto-stops on silence
AI responds with voice (toggle with /voice)
Type exit to quit and save

## Architecture

llm.py - LLM abstraction layer (swappable providers)
conversation.py - Process and save conversations
memory.py - Retrieve relevant memories using semantic search
voice.py - Voice interface abstraction (STT/TTS)
voice_chat.py - Voice chat interface
chat.py - Text chat interface
config.py - Configuration settings

## Configuration
Edit config.py to customize:

## LLM provider and model
Voice settings (STT/TTS providers)
Memory settings
Audio parameters

# Privacy

All conversations stored in memories/ (gitignored, stays local)
Voice processing runs locally (no cloud APIs)
Your data never leaves your machine

# Requirements

Python 3.8+
Ollama (or OpenAI/Anthropic API keys)
Microphone (for voice input)
Speakers (for voice output)

License
MIT
