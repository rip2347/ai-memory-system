"""
Conversation Processor
Analyzes conversations and extracts structured memory data.
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

from llm import ask_llm
from config import MEMORY_DIR


class ConversationProcessor:
    """Processes conversations and extracts structured memory."""
    
    def __init__(self):
        self.memory_dir = Path(MEMORY_DIR)
        self.memory_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.memory_dir / "raw").mkdir(exist_ok=True)
        (self.memory_dir / "processed").mkdir(exist_ok=True)
    
    def process_conversation(self, conversation_text: str, 
                           title: str = None, verbose: bool = True,
                           model_name: str = None,
                           response_times: List[float] = None) -> Dict[str, Any]:
        """
        Process a conversation and extract structured memory.
        
        Args:
            conversation_text: The full conversation text
            title: Optional title for the conversation
            verbose: Whether to print progress messages
            model_name: Optional model to use for processing (uses config default if None)
            
        Returns:
            Dict containing processed memory data
        """

        start_time = time.time()
        
        if verbose:
            print("Processing conversation...")
            print("-" * 50)
        
        # Generate conversation ID
        timestamp = datetime.now()
        convo_id = timestamp.strftime("%Y%m%d_%H%M%S")
        
        # Extract structured information using the LLM
        if verbose:
            print("Extracting key insights...")
        insights = self._extract_insights(conversation_text, model_name)
        
        if verbose:
            print("Identifying topics...")
        topics = self._extract_topics(conversation_text, model_name)
        
        if verbose:
            print("Extracting key concepts...")
        concepts = self._extract_concepts(conversation_text, model_name)
        
        if verbose:
            print("Generating summaries...")
        summaries = self._generate_summaries(conversation_text, model_name)

        processing_time = time.time() - start_time
        
        # Build memory structure
        memory = {
            "id": convo_id,
            "timestamp": timestamp.isoformat(),
            "title": title or f"Conversation {convo_id}",
            "insights": insights,
            "topics": topics,
            "concepts": concepts,
            "summaries": summaries,
            "metadata": {
                "length": len(conversation_text),
                "word_count": len(conversation_text.split()),
                "processed_at": datetime.now().isoformat(),
                "processing_time_seconds": round(processing_time, 2),
                "processing_words_per_second": round(len(conversation_text.split()) / processing_time, 2),
                "model_used": model_name or "default",
                "response_times": response_times or [],
                "average_response_time": round(sum(r['response_time'] for r in response_times)/len(response_times), 2) if response_times else None,
                "average_words_per_second": round(sum(r['words_per_second'] for r in response_times)/len(response_times), 2) if response_times else None
            }
        }
        
        # Save raw conversation
        raw_path = self.memory_dir / "raw" / f"{convo_id}.txt"
        with open(raw_path, 'w') as f:
            f.write(conversation_text)
        
        # Save processed memory
        processed_path = self.memory_dir / "processed" / f"{convo_id}.json"
        with open(processed_path, 'w') as f:
            json.dump(memory, f, indent=2)
        
        if verbose:
            print("-" * 50)
            print(f"✓ Conversation saved: {convo_id}")
            print(f"  Processing time: {processing_time:.2f} seconds")
            print(f"  Word count: {memory['metadata']['word_count']}")
            print(f"  Model: {memory['metadata']['model_used']}")
            print(f"  Raw: {raw_path}")
            print(f"  Processed: {processed_path}")
        
        return memory
    
    def _extract_insights(self, text: str, model_name: str = None) -> List[str]:
        """Extract key insights from the conversation."""
        
        prompt = f"""Analyze this conversation and extract 3-5 key insights or important points.
Focus on the main ideas, conclusions, or realizations discussed.

Conversation:
{text}

Provide the insights as a JSON array of strings.
Example: ["insight 1", "insight 2", "insight 3"]

Only return the JSON array, nothing else."""

        try:
            # Use fast processing model
            response = ask_llm(prompt, temperature=0.3, model_override=model_name)
            # Try to parse JSON from response
            insights = json.loads(response)
            return insights if isinstance(insights, list) else []
        except:
            # Fallback: treat each line as an insight
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            return lines[:5]
    
    def _extract_topics(self, text: str, model_name: str = None) -> List[str]:
        """Extract main topics/themes from the conversation."""
        
        prompt = f"""Identify the main topics or themes discussed in this conversation.
List 3-7 topics as short phrases (2-4 words each).

Conversation:
{text}

Provide the topics as a JSON array of strings.
Example: ["artificial intelligence", "consciousness", "free will"]

Only return the JSON array, nothing else."""

        try:
            response = ask_llm(prompt, temperature=0.3, model_override=model_name)
            topics = json.loads(response)
            return topics if isinstance(topics, list) else []
        except:
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            return lines[:7]
    
    def _extract_concepts(self, text: str, model_name: str = None) -> List[str]:
        """Extract key concepts mentioned in the conversation."""
        
        prompt = f"""Extract important concepts, terms, or ideas mentioned in this conversation.
Include technical terms, philosophical concepts, or specific references.
List 5-10 concepts.

Conversation:
{text}

Provide the concepts as a JSON array of strings.
Example: ["neural networks", "emergence", "determinism"]

Only return the JSON array, nothing else."""

        try:
            response = ask_llm(prompt, temperature=0.3, model_override=model_name)
            concepts = json.loads(response)
            return concepts if isinstance(concepts, list) else []
        except:
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            return lines[:10]
    
    def _generate_summaries(self, text: str, model_name: str = None) -> Dict[str, str]:
        """Generate multi-level summaries of the conversation."""
        
        # Brief summary (1-2 sentences)
        brief_prompt = f"""Summarize this conversation in 1-2 sentences. Capture the essence.

Conversation:
{text}

Summary:"""
        
        brief = ask_llm(brief_prompt, temperature=0.3, model_override=model_name)
        
        # Detailed summary (1 paragraph)
        detailed_prompt = f"""Provide a detailed summary of this conversation in one paragraph.
Include main points, key arguments, and conclusions.

Conversation:
{text}

Summary:"""
        
        detailed = ask_llm(detailed_prompt, temperature=0.3, model_override=model_name)
        
        return {
            "brief": brief.strip(),
            "detailed": detailed.strip()
        }
    
    def load_memory(self, convo_id: str) -> Dict[str, Any]:
        """Load a processed memory by ID."""
        
        processed_path = self.memory_dir / "processed" / f"{convo_id}.json"
        
        if not processed_path.exists():
            raise FileNotFoundError(f"Memory not found: {convo_id}")
        
        with open(processed_path, 'r') as f:
            return json.load(f)
    
    def list_memories(self) -> List[Dict[str, Any]]:
        """List all stored memories with basic info."""
        
        processed_dir = self.memory_dir / "processed"
        memories = []
        
        for path in processed_dir.glob("*.json"):
            with open(path, 'r') as f:
                memory = json.load(f)
                memories.append({
                    "id": memory["id"],
                    "timestamp": memory["timestamp"],
                    "title": memory["title"],
                    "topics": memory["topics"]
                })
        
        # Sort by timestamp (newest first)
        memories.sort(key=lambda x: x["timestamp"], reverse=True)
        return memories


def main():
    """CLI interface for processing conversations."""
    
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python conversation.py <command> [args]")
        print("\nCommands:")
        print("  save <file>        Process and save a conversation from file")
        print("  list               List all stored conversations")
        print("  show <id>          Show details of a conversation")
        print("\nExample:")
        print("  python conversation.py save my_chat.txt")
        return
    
    processor = ConversationProcessor()
    command = sys.argv[1]
    
    if command == "save":
        if len(sys.argv) < 3:
            print("Error: Please provide a file path")
            print("Usage: python conversation.py save <file>")
            return
        
        file_path = sys.argv[2]
        
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return
        
        with open(file_path, 'r') as f:
            conversation_text = f.read()
        
        memory = processor.process_conversation(conversation_text)
        
        print("\nMemory created:")
        print(f"Topics: {', '.join(memory['topics'])}")
        print(f"Insights: {len(memory['insights'])}")
        print(f"\nBrief summary:")
        print(memory['summaries']['brief'])
    
    elif command == "list":
        memories = processor.list_memories()
        
        if not memories:
            print("No conversations stored yet.")
            return
        
        print(f"\nStored conversations ({len(memories)}):")
        print("-" * 70)
        
        for memory in memories:
            print(f"{memory['id']} - {memory['title']}")
            print(f"  Topics: {', '.join(memory['topics'][:3])}")
            print()
    
    elif command == "show":
        if len(sys.argv) < 3:
            print("Error: Please provide a conversation ID")
            print("Usage: python conversation.py show <id>")
            return
        
        convo_id = sys.argv[2]
        
        try:
            memory = processor.load_memory(convo_id)
            
            print(f"\n{memory['title']}")
            print("=" * 70)
            print(f"ID: {memory['id']}")
            print(f"Date: {memory['timestamp']}")
            print(f"\nTopics: {', '.join(memory['topics'])}")
            print(f"\nKey Insights:")
            for i, insight in enumerate(memory['insights'], 1):
                print(f"  {i}. {insight}")
            print(f"\nBrief Summary:")
            print(f"  {memory['summaries']['brief']}")
            print(f"\nDetailed Summary:")
            print(f"  {memory['summaries']['detailed']}")
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
    
    else:
        print(f"Unknown command: {command}")
        print("Run 'python conversation.py' for usage info")


if __name__ == "__main__":
    main()