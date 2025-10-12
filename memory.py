"""
Memory Retrieval System
Searches and retrieves relevant past conversations based on current context.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta

from llm import ask_llm
from config import MEMORY_DIR, MAX_MEMORIES_TO_RETRIEVE


class MemoryRetriever:
    """Retrieves relevant memories based on current context."""
    
    def __init__(self):
        self.memory_dir = Path(MEMORY_DIR)
        self.processed_dir = self.memory_dir / "processed"
    
    def search(self, query: str, max_results: int = None) -> List[Dict[str, Any]]:
        """
        Search for relevant memories based on a query.
        
        Args:
            query: The search query or current topic
            max_results: Maximum number of results to return
            
        Returns:
            List of relevant memories, ranked by relevance
        """
        
        if max_results is None:
            max_results = MAX_MEMORIES_TO_RETRIEVE
        
        # Load all memories
        all_memories = self._load_all_memories()
        
        if not all_memories:
            return []
        
        # Score each memory for relevance
        scored_memories = []
        for memory in all_memories:
            score = self._calculate_relevance_score(query, memory)
            if score > 0:
                scored_memories.append((score, memory))
        
        # Sort by score (highest first)
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        
        # Return top results (without scores)
        return [memory for score, memory in scored_memories[:max_results]]
    
    def get_context_for_query(self, query: str, max_memories: int = None) -> str:
        """
        Get formatted context from relevant memories for a query.
        This is what gets injected into the chat.
        
        Args:
            query: Current user query or topic
            max_memories: Maximum memories to include
            
        Returns:
            Formatted context string to inject into prompt
        """
        
        relevant_memories = self.search(query, max_memories)
        
        if not relevant_memories:
            return ""
        
        # Build context string
        context_parts = ["Based on our previous conversations:\n"]
        
        for i, memory in enumerate(relevant_memories, 1):
            context_parts.append(f"\n--- Memory {i} ({memory['timestamp'][:10]}) ---")
            context_parts.append(f"Topics: {', '.join(memory['topics'][:3])}")
            context_parts.append(f"Summary: {memory['summaries']['brief']}")
            
            # Include key insights
            if memory['insights']:
                context_parts.append(f"Key points:")
                for insight in memory['insights'][:2]:
                    context_parts.append(f"  - {insight}")
        
        return "\n".join(context_parts)
    
    def _load_all_memories(self) -> List[Dict[str, Any]]:
        """Load all processed memories from disk."""
        
        if not self.processed_dir.exists():
            return []
        
        memories = []
        for path in self.processed_dir.glob("*.json"):
            try:
                with open(path, 'r') as f:
                    memory = json.load(f)
                    memories.append(memory)
            except Exception as e:
                print(f"Warning: Could not load {path}: {e}")
        
        return memories
    
    def _calculate_relevance_score(self, query: str, memory: Dict[str, Any]) -> float:
        """
        Calculate how relevant a memory is to the current query.
        Uses multiple heuristics.
        
        Returns:
            Float score (higher = more relevant)
        """
        
        score = 0.0
        query_lower = query.lower()
        
        # 1. Topic matching (highest weight)
        for topic in memory['topics']:
            if topic.lower() in query_lower or query_lower in topic.lower():
                score += 10.0
        
        # 2. Concept matching
        for concept in memory.get('concepts', []):
            if concept.lower() in query_lower or query_lower in concept.lower():
                score += 5.0
        
        # 3. Check summary and insights for keyword overlap
        text_to_check = (
            memory['summaries']['brief'] + " " +
            " ".join(memory.get('insights', []))
        ).lower()
        
        # Simple keyword matching
        query_words = set(query_lower.split())
        text_words = set(text_to_check.split())
        overlap = query_words & text_words
        score += len(overlap) * 0.5
        
        # 4. Recency bonus (more recent = slightly higher score)
        try:
            timestamp = datetime.fromisoformat(memory['timestamp'])
            days_old = (datetime.now() - timestamp).days
            
            # Decay: Recent convos get small boost
            if days_old < 7:
                score += 2.0
            elif days_old < 30:
                score += 1.0
        except:
            pass
        
        # 5. Length/importance heuristic
        # Longer conversations with more insights might be more valuable
        num_insights = len(memory.get('insights', []))
        if num_insights >= 3:
            score += 1.0
        
        return score
    
    def find_similar_topics(self, topics: List[str], max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Find memories that discuss similar topics.
        
        Args:
            topics: List of topic strings to match
            max_results: Maximum results to return
            
        Returns:
            List of relevant memories
        """
        
        all_memories = self._load_all_memories()
        scored_memories = []
        
        for memory in all_memories:
            score = 0
            for query_topic in topics:
                for memory_topic in memory['topics']:
                    # Exact match
                    if query_topic.lower() == memory_topic.lower():
                        score += 5
                    # Partial match
                    elif query_topic.lower() in memory_topic.lower() or memory_topic.lower() in query_topic.lower():
                        score += 2
            
            if score > 0:
                scored_memories.append((score, memory))
        
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [memory for score, memory in scored_memories[:max_results]]
    
    def get_recent_memories(self, days: int = 7, max_results: int = 10) -> List[Dict[str, Any]]:
        """Get memories from the last N days."""
        
        all_memories = self._load_all_memories()
        cutoff = datetime.now() - timedelta(days=days)
        
        recent = []
        for memory in all_memories:
            try:
                timestamp = datetime.fromisoformat(memory['timestamp'])
                if timestamp >= cutoff:
                    recent.append(memory)
            except:
                continue
        
        # Sort by timestamp (newest first)
        recent.sort(key=lambda x: x['timestamp'], reverse=True)
        return recent[:max_results]
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about stored memories."""
        
        all_memories = self._load_all_memories()
        
        if not all_memories:
            return {
                "total_conversations": 0,
                "total_topics": 0,
                "total_insights": 0,
                "date_range": None
            }
        
        # Collect all topics and insights
        all_topics = set()
        total_insights = 0
        timestamps = []
        
        for memory in all_memories:
            all_topics.update(memory.get('topics', []))
            total_insights += len(memory.get('insights', []))
            timestamps.append(memory['timestamp'])
        
        timestamps.sort()
        
        return {
            "total_conversations": len(all_memories),
            "total_topics": len(all_topics),
            "total_insights": total_insights,
            "date_range": {
                "earliest": timestamps[0][:10],
                "latest": timestamps[-1][:10]
            } if timestamps else None,
            "top_topics": self._get_top_topics(all_memories, top_n=10)
        }
    
    def _get_top_topics(self, memories: List[Dict[str, Any]], top_n: int = 10) -> List[Tuple[str, int]]:
        """Get most frequently discussed topics."""
        
        topic_counts = {}
        for memory in memories:
            for topic in memory.get('topics', []):
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        # Sort by count
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_topics[:top_n]


def main():
    """CLI interface for memory retrieval."""
    
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python memory.py <command> [args]")
        print("\nCommands:")
        print("  search <query>     Search for relevant memories")
        print("  topics <topic>     Find conversations about a topic")
        print("  recent [days]      Show recent conversations (default: 7 days)")
        print("  stats              Show memory statistics")
        print("  context <query>    Get formatted context for a query")
        print("\nExamples:")
        print("  python memory.py search consciousness")
        print("  python memory.py topics 'free will'")
        print("  python memory.py recent 14")
        return
    
    retriever = MemoryRetriever()
    command = sys.argv[1]
    
    if command == "search":
        if len(sys.argv) < 3:
            print("Error: Please provide a search query")
            return
        
        query = " ".join(sys.argv[2:])
        print(f"Searching for: {query}")
        print("-" * 70)
        
        results = retriever.search(query)
        
        if not results:
            print("No relevant memories found.")
            return
        
        print(f"\nFound {len(results)} relevant conversation(s):\n")
        
        for i, memory in enumerate(results, 1):
            print(f"{i}. {memory['title']} ({memory['timestamp'][:10]})")
            print(f"   Topics: {', '.join(memory['topics'][:3])}")
            print(f"   Summary: {memory['summaries']['brief']}")
            print()
    
    elif command == "topics":
        if len(sys.argv) < 3:
            print("Error: Please provide a topic")
            return
        
        topic = " ".join(sys.argv[2:])
        results = retriever.find_similar_topics([topic])
        
        if not results:
            print(f"No conversations found about '{topic}'")
            return
        
        print(f"\nConversations about '{topic}':\n")
        
        for i, memory in enumerate(results, 1):
            print(f"{i}. {memory['title']} ({memory['timestamp'][:10]})")
            print(f"   Topics: {', '.join(memory['topics'])}")
            print()
    
    elif command == "recent":
        days = 7
        if len(sys.argv) >= 3:
            try:
                days = int(sys.argv[2])
            except:
                print("Error: Days must be a number")
                return
        
        results = retriever.get_recent_memories(days)
        
        if not results:
            print(f"No conversations in the last {days} days.")
            return
        
        print(f"\nConversations from last {days} days:\n")
        
        for memory in results:
            print(f"{memory['title']} ({memory['timestamp'][:10]})")
            print(f"  Topics: {', '.join(memory['topics'][:3])}")
            print()
    
    elif command == "stats":
        stats = retriever.get_memory_stats()
        
        print("\nMemory Statistics")
        print("=" * 70)
        print(f"Total conversations: {stats['total_conversations']}")
        print(f"Total topics discussed: {stats['total_topics']}")
        print(f"Total insights extracted: {stats['total_insights']}")
        
        if stats['date_range']:
            print(f"Date range: {stats['date_range']['earliest']} to {stats['date_range']['latest']}")
        
        if stats.get('top_topics'):
            print(f"\nMost discussed topics:")
            for topic, count in stats['top_topics']:
                print(f"  {topic}: {count} conversation(s)")
    
    elif command == "context":
        if len(sys.argv) < 3:
            print("Error: Please provide a query")
            return
        
        query = " ".join(sys.argv[2:])
        context = retriever.get_context_for_query(query)
        
        if not context:
            print("No relevant context found.")
            return
        
        print("\nContext for query:")
        print("=" * 70)
        print(context)
    
    else:
        print(f"Unknown command: {command}")
        print("Run 'python memory.py' for usage info")


if __name__ == "__main__":
    main()