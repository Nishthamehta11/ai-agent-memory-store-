# 🧠 AI Agent with Persistent Vector Memory

A conversational AI agent that **remembers everything across sessions** using FAISS vector search and sentence-transformers embeddings.

No API key needed. 100% free. Runs locally on CPU.

---

## The Problem This Solves

Standard LLMs forget everything between conversations. Production AI agents need **persistent, searchable memory** — the ability to recall relevant past context without stuffing entire history into every prompt (which is slow and expensive).

This project implements exactly that using vector similarity search.

---

## How It Works

```
User: "What do you know about my internship?"
           ↓
   Embed query → 384-dim vector
           ↓
   FAISS IndexFlatIP cosine search
           ↓
   Retrieve top-3 most relevant past memories
           ↓
   Build context: [retrieved memories] + [current input]
           ↓
   LLM extracts grounded answer from context
           ↓
   Store this turn as new memory → persist to disk
```

**Every session picks up where the last one left off.** Memories survive restarts because the FAISS index is serialised to disk after every turn.

---

## Architecture

### `VectorMemoryStore`
- Wraps `faiss.IndexFlatIP` (inner product = cosine similarity on normalised vectors)
- Each memory = one embedded text chunk: `"User said: X\nAgent replied: Y"`
- Similarity threshold (0.25) filters irrelevant memories — prevents hallucination from weak matches
- Serialise/deserialise entire index with `faiss.serialize_index()` / `faiss.deserialize_index()`

### `MemoryAgent`
- Orchestrates the full loop: retrieve → respond → store → persist
- Maintains a JSON log (`memory_log.json`) for human-readable inspection
- Session ID per run, turn counter for unique memory IDs

### Why not just store text in a list?
Linear scan over 1000 memories = 1000 comparisons. FAISS with `IndexIVFFlat` (approximate) = ~40 comparisons regardless of size. This architecture scales to millions of memories.

---

## Quick Start

```bash
pip install faiss-cpu sentence-transformers transformers torch
python agent.py          # runs demo with seeded memories
python agent.py --interactive   # live chat mode
```

### Demo output
```
📝 Seeding agent with 5 memories...

❓ What is my name and what am I studying?
💡 Nishtha — 3rd year CSE student
   Memories retrieved: 2 | Top score: 0.847 | Time: 43ms

❓ Which company am I applying to?
💡 Tredence
   Memories retrieved: 1 | Top score: 0.912 | Time: 38ms

❓ What backend technologies do I know?
💡 FastAPI with Redis caching
   Memories retrieved: 2 | Top score: 0.781 | Time: 41ms
```

---

## Extending This

| Feature | How |
|---------|-----|
| Scale to 1M+ memories | Replace `IndexFlatIP` with `IndexIVFFlat` (approximate search) |
| Persistent server | Wrap `MemoryAgent` in a FastAPI endpoint |
| Multi-user memory | Namespace memories by `user_id` in the FAISS index |
| GPT-4 responses | Replace HuggingFace pipeline with `openai.chat.completions.create()` |
| ChromaDB backend | Swap `VectorMemoryStore` for `chromadb.Client()` — same interface |
| Memory summarisation | Periodically compress old memories using an LLM summariser |

---

## Files

```
ai-agent-memory-store/
├── agent.py              # Full implementation
├── requirements.txt
├── agent_memory.pkl      # Auto-generated: serialised FAISS index
├── memory_log.json       # Auto-generated: human-readable memory log
└── demo_results.json     # Auto-generated: demo run output
```

---

## Requirements

```
faiss-cpu>=1.7.4
sentence-transformers>=2.2.2
transformers>=4.30.0
torch>=2.0.0
numpy>=1.24.0
```

---

*Built with Python 3.10+ · FAISS · sentence-transformers · No API keys required*
