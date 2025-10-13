"""
Voice Chat Interface
Chat with your AI using voice input and output.
"""

import sys
from datetime import datetime
from typing import List, Dict

from chat import ChatSession
from voice import VoiceInterface
from config import TEMPERATURE


class VoiceChatSession(ChatSession):
    """Interactive voice chat session with memory."""
    
    def __init__(self, model_name: str = None):
        super().__init__(model_name)
        self.voice = VoiceInterface()
    
    def start(self):
        """Start the voice chat session."""
        
        print("=" * 70)
        print("🎤 Voice Chat with Memory")
        print("=" * 70)
        
        # Get available models and let user choose
        if not self.model_name:
            self.model_name = self._select_model()
            if not self.model_name:
                return
        
        print(f"\n✓ Using model: {self.model_name}")
        
        # Show memory stats
        stats = self.retriever.get_memory_stats()
        if stats['total_conversations'] > 0:
            print(f"\n📚 Memory loaded: {stats['total_conversations']} past conversations")
            print(f"   Most discussed: {', '.join([t[0] for t in stats['top_topics'][:3]])}")
        else:
            print("\n📚 No previous conversations found. Starting fresh!")
        
        print("\n" + "=" * 70)
        print("Commands:")
        print("  Press SPACE then ENTER - Start recording")
        print("  Type 'exit' or 'quit' - Save and exit")
        print("  Type 'text' - Switch to text mode")
        print("  Type any text - Send text message (no voice)")
        print("=" * 70)
        print("\nReady for voice input!\n")
        
        # Main voice loop
        self.voice_loop()
    
    def voice_loop(self):
        """Main voice interaction loop."""
        
        while True:
            try:
                # Prompt for input
                user_input = input("🎤 Press SPACE+ENTER to talk (or type command): ").strip()
                
                # Handle text commands
                if user_input.lower() in ['exit', 'quit', '/exit', '/quit']:
                    print("\nSaving conversation and exiting...")
                    self._save_and_exit()
                    break
                
                elif user_input.lower() == 'text':
                    print("\n📝 Switching to text mode...")
                    self.chat_loop()  # Use parent's text chat loop
                    continue
                
                elif user_input == '' or user_input == ' ':
                    # Voice input
                    user_text = self.voice.listen()
                    
                    if not user_text:
                        print("⚠ No speech detected, try again")
                        continue
                    
                    print(f"\n💬 You: {user_text}")
                
                elif user_input.startswith('/'):
                    # Handle special commands
                    if self._handle_command(user_input):
                        break
                    continue
                
                else:
                    # Text input (without voice)
                    user_text = user_input
                    print(f"\n💬 You: {user_text}")
                
                # Add user message to history
                self.messages.append({
                    "role": "user",
                    "content": user_text,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Generate AI response
                print("\n🤔 Thinking...")
                response = self._generate_response(user_text)
                
                # Add AI response to history
                self.messages.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Display and speak response
                print(f"\n🤖 AI: {response}\n")
                self.voice.speak(response)
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Saving conversation...")
                self._save_and_exit()
                break
            except Exception as e:
                print(f"\n✗ Error: {e}")
                print("Continuing...\n")


def main():
    """Main entry point."""
    
    # Handle command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print("Voice Chat with Memory")
            print("\nUsage: python voice_chat.py")
            print("\nStarts an interactive voice chat session.")
            print("Speak naturally - the system auto-detects silence.")
            return
    
    # Start voice chat session
    session = VoiceChatSession()
    session.start()


if __name__ == "__main__":
    main()