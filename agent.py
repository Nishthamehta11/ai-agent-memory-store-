"""
AI Agent with Persistent Memory
================================
A conversational AI agent that remembers past interactions using
vector embeddings stored in FAISS. Each conversation turn is embedded
and retrieved as context — giving the agent genuine long-term memory.

Architecture:
    User input
        → embed with sentence-transformers
        → FAISS similarity search (retrieve relevant memories)
        → build prompt: [system] + [retrieved memories] + [current input]
        → LLM generates response
        → store new (input, response) pair as memory

No API key required — uses free HuggingFace models throughout.

Author: Nishtha Mehta
"""

import os
import json
import time
import pickle
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline as hf_pipeline
except ImportError:
    os.system("pip install faiss-cpu sentence-transformers transformers torch -q")
    import faiss
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline as hf_pipeline


# ── Config ────────────────────────────────────────────────────────────────────

EMBED_MODEL    = "all-MiniLM-L6-v2"     # 384-dim, fast, accurate
QA_MODEL       = "deepset/roberta-base-squad2"
MEMORY_FILE    = "agent_memory.pkl"
MEMORY_LOG     = "memory_log.json"
TOP_K_MEMORIES = 3                       # how many past memories to retrieve
EMBED_DIM      = 384
MIN_SCORE      = 0.25                    # ignore low-relevance memories


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class Memory:
    """A single stored memory unit."""
    id: str
    timestamp: str
    user_input: str
    agent_response: str
    embedding: Optional[np.ndarray] = None   # not serialised to JSON

    def to_text(self) -> str:
        """Convert memory to retrieval-ready text chunk."""
        return f"User said: {self.user_input}\nAgent replied: {self.agent_response}"

    def to_dict(self) -> dict:
        """JSON-safe dict (excludes numpy array)."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "user_input": self.user_input,
            "agent_response": self.agent_response,
        }


# ── Vector Memory Store ────────────────────────────────────────────────────────

class VectorMemoryStore:
    """
    FAISS-backed memory store.

    Stores conversation turns as dense embeddings.
    On query, retrieves the top-k most semantically similar past turns.

    Why FAISS?
    - Exact cosine search on 384-dim vectors
    - Sub-millisecond retrieval even at 10K+ memories
    - No server required — pure in-process library
    """

    def __init__(self, embedder: SentenceTransformer):
        self.embedder = embedder
        self.index = faiss.IndexFlatIP(EMBED_DIM)  # Inner Product = cosine on normalised vecs
        self.memories: list[Memory] = []

    def _embed(self, text: str) -> np.ndarray:
        vec = self.embedder.encode([text], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(vec)
        return vec

    def add(self, memory: Memory):
        """Embed and store a memory."""
        vec = self._embed(memory.to_text())
        memory.embedding = vec
        self.index.add(vec)
        self.memories.append(memory)

    def search(self, query: str, k: int = TOP_K_MEMORIES) -> list[tuple[Memory, float]]:
        """Retrieve top-k memories most relevant to query."""
        if self.index.ntotal == 0:
            return []

        query_vec = self._embed(query)
        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and float(score) >= MIN_SCORE:
                results.append((self.memories[idx], float(score)))
        return results

    def save(self, path: str = MEMORY_FILE):
        """Persist index + memories to disk."""
        data = {
            "index": faiss.serialize_index(self.index),
            "memories": [
                {**m.to_dict(), "embedding": m.embedding.tolist()}
                for m in self.memories
            ],
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str = MEMORY_FILE):
        """Restore index + memories from disk."""
        if not Path(path).exists():
            return
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.index = faiss.deserialize_index(data["index"])
        self.memories = []
        for m in data["memories"]:
            emb = np.array(m.pop("embedding"), dtype="float32").reshape(1, -1)
            mem = Memory(**m, embedding=emb)
            self.memories.append(mem)
        print(f"  ✓ Loaded {len(self.memories)} memories from disk")


# ── LLM Backend ───────────────────────────────────────────────────────────────

class LLMBackend:
    """
    Free HuggingFace QA model used as a response generator.
    Falls back to echo mode if model fails to load.
    """

    def __init__(self):
        self.model = None
        self._load()

    def _load(self):
        try:
            print("  Loading language model (first run ~1 min)...")
            self.model = hf_pipeline(
                "question-answering",
                model=QA_MODEL,
                tokenizer=QA_MODEL
            )
            print("  ✓ Language model ready")
        except Exception as e:
            print(f"  ⚠ Model load failed ({e}) — using echo mode")

    def respond(self, question: str, context: str) -> str:
        """Generate a response grounded in the provided context."""
        if self.model is None or not context.strip():
            return f"[No context available] I don't have relevant memories for: {question}"
        try:
            result = self.model(question=question, context=context, max_answer_len=200)
            return result["answer"] if result["score"] > 0.01 else \
                   f"Based on our conversation: {context[:150]}..."
        except Exception:
            return f"I recall: {context[:200]}..."


# ── Agent ─────────────────────────────────────────────────────────────────────

class MemoryAgent:
    """
    Conversational AI agent with persistent vector memory.

    Each turn:
    1. Search memory for relevant past conversations
    2. Build context from retrieved memories
    3. Generate response grounded in context
    4. Store this turn as a new memory
    5. Persist to disk after every turn
    """

    def __init__(self, agent_name: str = "Aria"):
        self.name = agent_name
        self.session_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        self.turn_count = 0

        print(f"\n🤖 Initialising {self.name}...")
        print("  Loading embedding model...")
        self.embedder = SentenceTransformer(EMBED_MODEL)
        print("  ✓ Embedding model ready")

        self.memory_store = VectorMemoryStore(self.embedder)
        self.memory_store.load()   # restore previous sessions if available

        self.llm = LLMBackend()
        print(f"\n✅ {self.name} is ready. Total memories: {len(self.memory_store.memories)}\n")

    def _make_memory_id(self) -> str:
        return f"{self.session_id}_{self.turn_count:04d}"

    def _build_context(self, retrieved: list[tuple[Memory, float]]) -> str:
        """Assemble retrieved memories into a single context string."""
        if not retrieved:
            return ""
        parts = ["[Relevant past conversations:]\n"]
        for mem, score in retrieved:
            parts.append(f"- {mem.to_text()} (relevance: {score:.2f})")
        return "\n".join(parts)

    def chat(self, user_input: str) -> dict:
        """Process one conversation turn."""
        self.turn_count += 1
        t0 = time.time()

        # Step 1: retrieve relevant memories
        retrieved = self.memory_store.search(user_input)
        context = self._build_context(retrieved)

        # Step 2: generate response
        if context:
            response = self.llm.respond(user_input, context)
        else:
            response = f"I don't have memories about that yet. Tell me more!"

        # Step 3: store this turn
        memory = Memory(
            id=self._make_memory_id(),
            timestamp=datetime.now().isoformat(),
            user_input=user_input,
            agent_response=response,
        )
        self.memory_store.add(memory)
        self.memory_store.save()

        # Step 4: log to JSON
        self._log(memory)

        elapsed = round((time.time() - t0) * 1000, 1)
        return {
            "response": response,
            "memories_retrieved": len(retrieved),
            "top_memory_score": round(retrieved[0][1], 3) if retrieved else 0.0,
            "total_memories": len(self.memory_store.memories),
            "response_time_ms": elapsed,
        }

    def _log(self, memory: Memory):
        """Append memory to JSON log file."""
        log = []
        if Path(MEMORY_LOG).exists():
            with open(MEMORY_LOG) as f:
                try:
                    log = json.load(f)
                except json.JSONDecodeError:
                    log = []
        log.append(memory.to_dict())
        with open(MEMORY_LOG, "w") as f:
            json.dump(log, f, indent=2)

    def show_memories(self, n: int = 5):
        """Display the n most recent memories."""
        memories = self.memory_store.memories[-n:]
        print(f"\n📚 Last {len(memories)} memories:")
        for m in memories:
            print(f"  [{m.timestamp[:19]}] User: {m.user_input[:60]}...")
            print(f"                    Agent: {m.agent_response[:60]}...")

    def forget_all(self):
        """Wipe all memories (useful for testing)."""
        self.memory_store = VectorMemoryStore(self.embedder)
        for f in [MEMORY_FILE, MEMORY_LOG]:
            if Path(f).exists():
                os.remove(f)
        print("🗑️  All memories wiped.")


# ── Demo ──────────────────────────────────────────────────────────────────────

def run_demo():
    """Demonstrate the agent's memory across a multi-turn conversation."""
    agent = MemoryAgent(agent_name="Aria")

    # Seed some memories first
    seed_conversations = [
        ("My name is Nishtha and I'm a 3rd year CSE student.",
         "Nice to meet you Nishtha! I'll remember you're in 3rd year CSE."),
        ("I am applying to Tredence for an AI Engineering internship.",
         "That's exciting! Tredence is a great company for AI engineering work."),
        ("I built a FastAPI service with Redis caching for LLM responses.",
         "Impressive — async FastAPI with Redis caching shows production thinking."),
        ("I know Python, PyTorch, FAISS, and RAG pipeline design.",
         "Great stack for AI engineering — FAISS + RAG is exactly what LLM systems need."),
        ("My L&T internship involved automating reports saving 20 man-hours weekly.",
         "Automating 20 man-hours weekly at L&T as a 2nd year student is genuinely impressive."),
    ]

    print("📝 Seeding agent with initial memories...")
    from dataclasses import asdict
    for user_msg, agent_msg in seed_conversations:
        mem = Memory(
            id=agent._make_memory_id(),
            timestamp=datetime.now().isoformat(),
            user_input=user_msg,
            agent_response=agent_msg,
        )
        agent.memory_store.add(mem)
        agent._log(mem)
        agent.turn_count += 1
    agent.memory_store.save()
    print(f"  ✓ {len(seed_conversations)} memories seeded\n")

    # Now demonstrate retrieval
    test_queries = [
        "What is my name and what am I studying?",
        "Which company am I applying to?",
        "What backend technologies do I know?",
        "Tell me about my internship experience.",
        "What AI/ML skills do I have?",
    ]

    print("=" * 60)
    print("🧠 Memory Retrieval Demo")
    print("=" * 60)

    results = []
    for query in test_queries:
        print(f"\n❓ {query}")
        result = agent.chat(query)
        print(f"💡 {result['response']}")
        print(f"   Memories retrieved: {result['memories_retrieved']} | "
              f"Top score: {result['top_memory_score']} | "
              f"Time: {result['response_time_ms']}ms")
        results.append({"query": query, **result})

    # Save demo results
    with open("demo_results.json", "w") as f:
        json.dump(results, f, indent=2)

    agent.show_memories(n=5)
    print(f"\n✅ Demo complete. {len(agent.memory_store.memories)} total memories stored.")
    print("   Files saved: agent_memory.pkl, memory_log.json, demo_results.json")


def run_interactive():
    """Live interactive chat with the memory agent."""
    agent = MemoryAgent()
    print(f"Chat with {agent.name}. Type 'memories' to see stored memories, 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if user_input.lower() == "memories":
            agent.show_memories()
            continue
        if user_input.lower() == "forget":
            agent.forget_all()
            continue

        result = agent.chat(user_input)
        print(f"\n{agent.name}: {result['response']}")
        print(f"  [memories: {result['memories_retrieved']} retrieved | "
              f"total: {result['total_memories']} | "
              f"{result['response_time_ms']}ms]\n")


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        run_interactive()
    else:
        run_demo()
