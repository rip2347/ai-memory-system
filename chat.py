"""
Chat Interface with Automatic Memory
Chat with your local LLM with full conversational continuity.
"""

import sys
import requests
import time
from datetime import datetime
from typing import List, Dict

from llm import ask_llm
from memory import MemoryRetriever
from conversation import ConversationProcessor
from config import TEMPERATURE, OLLAMA_URL


class ChatSession:
    """Interactive chat session with memory."""
    
    def __init__(self, model_name: str = None):
        self.messages: List[Dict[str, str]] = []
        self.response_times: List[Dict] = []
        self.retriever = MemoryRetriever()
        self.processor = ConversationProcessor()
        self.session_start = datetime.now()
        self.model_name = model_name  # Store selected model
    
    def start(self):
        """Start the chat session."""
        
        print("=" * 70)
        print("AI Chat with Memory")
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
        print("  /exit or /quit - Save and exit")
        print("  /clear - Clear current conversation (doesn't delete memory)")
        print("  /memories - Show relevant memories for current context")
        print("  /stats - Show memory statistics")
        print("  /model - Switch model")
        print("=" * 70)
        print("\nStart chatting! Type your message and press Enter.\n")
        
        # Main chat loop
        self.chat_loop()
    
    def _select_model(self) -> str:
        """
        Let user select which model to use.
        Returns the selected model name, or None if selection failed.
        """
        
        try:
            # Get available models from Ollama
            response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            response.raise_for_status()
            models = response.json().get("models", [])
            
            if not models:
                print("✗ No models found. Please pull a model first:")
                print("  ollama pull qwen2.5:7b")
                return None
            
            # Display available models
            print("\nAvailable models:")
            print("-" * 70)
            for i, model in enumerate(models, 1):
                model_name = model['name']
                size = model.get('size', 0) / (1024**3)  # Convert to GB
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
            print("Make sure Ollama is running: systemctl status ollama")
            return None
    
    def chat_loop(self):
        """Main chat loop."""
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    if self._handle_command(user_input):
                        break  # Exit if command returns True
                    continue
                
                # Add user message to history
                self.messages.append({
                    "role": "user",
                    "content": user_input,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Get AI response with memory
                response_start = time.time()
                response = self._generate_response(user_input)
                response_time = time.time() - response_start

                # Track detailed response data
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
                print(f"\nAI: {response}\n")
                print(f"(Response: {len(response.split())} words in {response_time:.2f}s = {response_data['words_per_second']:.1f} w/s)\n")

            except KeyboardInterrupt:
                print("\n\nInterrupted. Saving conversation...")
                self._save_and_exit()
                break
            except Exception as e:
                print(f"\n✗ Error: {e}")
                print("Continuing...\n")
    
    def _generate_response(self, user_input: str) -> str:
        """Generate AI response with relevant memory context."""
        
        # Retrieve relevant memories
        context = self.retriever.get_context_for_query(user_input, max_memories=3)
        
        # Build conversation history (last 10 messages for context window management)
        recent_history = self.messages[-10:]
        history_text = ""
        for msg in recent_history:
            role = msg['role'].capitalize()
            history_text += f"{role}: {msg['content']}\n"
        
        # Build full prompt with memory and history
        if context:
            system_prompt = f"""You are a helpful AI assistant with memory of past conversations.

{context}

Use this context from previous conversations to provide informed, continuous responses.
Reference past discussions naturally when relevant."""
        else:
            system_prompt = "You are a helpful AI assistant."
        
        # Construct the full prompt
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
        """
        Handle special commands.
        
        Returns:
            True if should exit, False otherwise
        """
        
        cmd = command.lower()
        
        if cmd in ['/exit', '/quit', '/q']:
            print("\nSaving conversation and exiting...")
            self._save_and_exit()
            return True
        
        elif cmd == '/clear':
            self.messages = []
            print("\n✓ Conversation cleared (memory preserved)\n")
        
        elif cmd == '/memories':
            self._show_relevant_memories()
        
        elif cmd == '/stats':
            self._show_stats()

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
    
    def _show_relevant_memories(self):
        """Show relevant memories based on current conversation."""
        
        if not self.messages:
            print("\n📚 No current conversation to find memories for.\n")
            return
        
        # Get last few user messages as context
        recent_user_messages = [
            msg['content'] for msg in self.messages[-5:]
            if msg['role'] == 'user'
        ]
        
        if not recent_user_messages:
            print("\n📚 No user messages yet.\n")
            return
        
        query = " ".join(recent_user_messages)
        memories = self.retriever.search(query, max_results=5)
        
        if not memories:
            print("\n📚 No relevant memories found.\n")
            return
        
        print("\n" + "=" * 70)
        print("📚 Relevant Memories:")
        print("=" * 70)
        
        for i, memory in enumerate(memories, 1):
            print(f"\n{i}. {memory['title']} ({memory['timestamp'][:10]})")
            print(f"   Topics: {', '.join(memory['topics'][:3])}")
            print(f"   Summary: {memory['summaries']['brief']}")
        
        print("\n" + "=" * 70 + "\n")
    
    def _show_stats(self):
        """Show memory statistics."""
        
        stats = self.retriever.get_memory_stats()
        
        print("\n" + "=" * 70)
        print("📊 Memory Statistics:")
        print("=" * 70)
        print(f"Total conversations: {stats['total_conversations']}")
        print(f"Total topics: {stats['total_topics']}")
        print(f"Total insights: {stats['total_insights']}")
        
        if stats['date_range']:
            print(f"Date range: {stats['date_range']['earliest']} to {stats['date_range']['latest']}")
        
        if stats.get('top_topics'):
            print(f"\nTop topics:")
            for topic, count in stats['top_topics'][:5]:
                print(f"  • {topic} ({count}x)")
        
        print("=" * 70 + "\n")
    
    def _show_help(self):
        """Show help message."""
        
        print("\n" + "=" * 70)
        print("Available Commands:")
        print("=" * 70)
        print("  /exit, /quit     - Save conversation and exit")
        print("  /clear           - Clear current conversation")
        print("  /memories        - Show relevant past conversations")
        print("  /stats           - Show memory statistics")
        print("  /model           - Switch to different model")
        print("  /help            - Show this help message")
        print("=" * 70 + "\n")
    
    def _save_and_exit(self):
        """Save the conversation and exit."""
        
        if not self.messages:
            print("No conversation to save. Exiting.")
            return
        
        # Format conversation for saving
        conversation_text = self._format_conversation()
        
        # Generate title from first user message
        first_user_msg = next(
            (msg['content'] for msg in self.messages if msg['role'] == 'user'),
            None
        )
        
        if first_user_msg:
            # Use first 50 chars of first message as title
            title = first_user_msg[:50]
            if len(first_user_msg) > 50:
                title += "..."
        else:
            title = f"Chat {self.session_start.strftime('%Y-%m-%d %H:%M')}"
        
        # Process and save with timing
        print("\n💾 Processing conversation into memory...")
        start_time = time.time()
        
        try:
            memory = self.processor.process_conversation(
                conversation_text,
                title=title,
                verbose=False,
                model_name=self.model_name,
                response_times=self.response_times
            )
            
            processing_time = time.time() - start_time
            
            # Calculate total conversation stats
            total_words = sum(len(msg['content'].split()) for msg in self.messages)
            conversation_duration = (datetime.now() - self.session_start).total_seconds()
            overall_wps = total_words / conversation_duration if conversation_duration > 0 else 0

            print(f"\n✓ Conversation saved: {memory['id']}")
            print(f"   Topics: {', '.join(memory['topics'][:3])}")
            print(f"   Insights: {len(memory['insights'])}")
            print(f"   Messages: {len(self.messages)}")
            print(f"   Total: {total_words} words in {conversation_duration:.1f}s ({overall_wps:.1f} w/s)")
            print(f"   Processing time: {processing_time:.2f}s")
            print("\nGoodbye! 👋")
            
        except Exception as e:
            print(f"✗ Error saving conversation: {e}")
            print("Conversation not saved.")
    
    def _format_conversation(self) -> str:
        """Format messages into readable text."""
        
        lines = []
        for msg in self.messages:
            role = "You" if msg['role'] == 'user' else "AI"
            lines.append(f"{role}: {msg['content']}")
        
        return "\n\n".join(lines)


def main():
    """Main entry point."""
    
    # Handle command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print("AI Chat with Memory")
            print("\nUsage: python chat.py")
            print("\nStarts an interactive chat session with automatic memory.")
            print("All conversations are automatically processed and saved.")
            return
    
    # Start chat session
    session = ChatSession()
    session.start()


if __name__ == "__main__":
    main()