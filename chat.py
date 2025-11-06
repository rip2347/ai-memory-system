"""
Chat Interface
Simple chat with your local LLM.
"""

import sys
import requests
import time
from datetime import datetime
from typing import List, Dict

from llm import ask_llm
from config import TEMPERATURE, OLLAMA_URL

# Display settings
ASSISTANT_NAME = "Cortana"  # Change to whatever you want
USER_NAME = "You"          # Or use your actual name

# Colors
USER_COLOR = '\033[94m'    # Blue
AI_COLOR = '\033[91m'      # Red
RESET = '\033[0m'


class ChatSession:
    """Interactive chat session."""
    
    def __init__(self, model_name: str = None):
        self.messages: List[Dict[str, str]] = []
        self.response_times: List[Dict] = []
        self.session_start = datetime.now()
        self.model_name = model_name
    
    def start(self):
        """Start the chat session."""
        
        print("=" * 70)
        print("AI Chat")
        print("=" * 70)
        
        # Get available models and let user choose
        if not self.model_name:
            self.model_name = self._select_model()
            if not self.model_name:
                return
        
        print(f"\n✓ Using model: {self.model_name}")
        
        print("\n" + "=" * 70)
        print("Commands:")
        print("  /exit or /quit - Exit chat")
        print("  /clear - Clear current conversation")
        print("  /model - Switch model")
        print("=" * 70)
        print("\nStart chatting! Type your message and press Enter.\n")
        
        # Main chat loop
        self.chat_loop()
    
    def _select_model(self) -> str:
        """Let user select which model to use."""
        
        try:
            # Get available models from Ollama
            response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            response.raise_for_status()
            models = response.json().get("models", [])

            # Sort models by size (smallest first)
            models = sorted(models, key=lambda m: m.get('size', 0))
            
            if not models:
                print("✗ No models found. Please pull a model first:")
                print("  ollama pull qwen2.5:7b")
                return None
            
            # Display available models
            print("\nAvailable models:")
            print("-" * 70)
            for i, model in enumerate(models, 1):
                model_name = model['name']
                size = model.get('size', 0) / (1024**3)
                print(f"  {i}. {model_name} ({size:.1f} GB)")
            
            print("-" * 70)
            
            # Get user selection
            while True:
                try:
                    choice = input("\nSelect model number (or press Enter for #1): ").strip()
                    
                    if not choice:
                        choice = 1
                    else:
                        choice = int(choice)
                    
                    if 1 <= choice <= len(models):
                        selected = models[choice - 1]['name']
                        return selected
                    else:
                        print(f"Please enter a number between 1 and {len(models)}")
                
                except ValueError:
                    print("Please enter a valid number")
                except KeyboardInterrupt:
                    print("\nCancelled.")
                    return None
        
        except Exception as e:
            print(f"✗ Error connecting to Ollama: {e}")
            print("Make sure Ollama is running")
            return None
    
    def chat_loop(self):
        """Main chat loop."""
        
        while True:
            try:
                # Get user input
                # user_input = input("You: ").strip()
                # user_input = input("\033[1mYou:\033[0m ").strip()
                print(f"{USER_COLOR}{USER_NAME}:{RESET} ", end='', flush=True)
                user_input = input().strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    if self._handle_command(user_input):
                        break
                    continue
                
                # Add user message to history
                self.messages.append({
                    "role": "user",
                    "content": user_input,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Get AI response
                response_start = time.time()
                response = self._generate_response(user_input)
                response_time = time.time() - response_start

                # Track response data
                response_data = {
                    "prompt_word_count": len(user_input.split()),
                    "response_word_count": len(response.split()),
                    "response_time": round(response_time, 2),
                    "words_per_second": round(len(response.split()) / response_time, 2) if response_time > 0 else 0
                }
                self.response_times.append(response_data)
                                
                # Add AI response to history
                self.messages.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Display response
                # print(f"\nAI: {response}\n")
                print(f"\n{AI_COLOR}{ASSISTANT_NAME}:{RESET} {response}\n")
                print(f"({response_data['response_word_count']} words in {response_time:.2f}s = {response_data['words_per_second']:.1f} w/s)\n")

            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"\n✗ Error: {e}")
                print("Continuing...\n")
    
    def _generate_response(self, user_input: str) -> str:
        """Generate AI response."""
        
        # Build conversation history (last 10 messages)
        recent_history = self.messages[-10:]
        history_text = ""
        for msg in recent_history:
            role = msg['role'].capitalize()
            history_text += f"{role}: {msg['content']}\n"
        
        # Build prompt
        system_prompt = "You are a helpful AI assistant."
        
        full_prompt = f"""{history_text}
User: {user_input}
Assistant:"""
        
        # Get response from LLM
        response = ask_llm(
            full_prompt,
            system_prompt=system_prompt,
            temperature=TEMPERATURE,
            model_override=self.model_name
        )
        
        return response
    
    def _handle_command(self, command: str) -> bool:
        """Handle special commands. Returns True if should exit."""
        
        cmd = command.lower()
        
        if cmd in ['/exit', '/quit', '/q']:
            print("\nGoodbye! 👋")
            return True
        
        elif cmd == '/clear':
            self.messages = []
            print("\n✓ Conversation cleared\n")
        
        elif cmd == '/model':
            print("\nSwitching model...")
            new_model = self._select_model()
            if new_model:
                self.model_name = new_model
                print(f"✓ Now using: {self.model_name}\n")
        
        elif cmd in ['/help', '/h']:
            self._show_help()
        
        else:
            print(f"Unknown command: {command}")
            print("Type /help for available commands")
        
        return False
    
    def _show_help(self):
        """Show help message."""
        
        print("\n" + "=" * 70)
        print("Available Commands:")
        print("=" * 70)
        print("  /exit, /quit     - Exit chat")
        print("  /clear           - Clear conversation")
        print("  /model           - Switch model")
        print("  /help            - Show this help")
        print("=" * 70 + "\n")


def main():
    """Main entry point."""
    
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print("AI Chat")
            print("\nUsage: python chat.py")
            print("\nStarts an interactive chat session.")
            return
    
    session = ChatSession()
    session.start()


if __name__ == "__main__":
    main()