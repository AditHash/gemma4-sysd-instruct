# CLAUDE.md — gemma4-sysd-instruct

> This file is the source of truth for Claude Code when working in this repository.
> Read this fully before writing any code, creating any files, or making any decisions.

---

## Project Identity

| Field | Value |
|---|---|
| GitHub Repo | `gemma4-sysd-instruct` |
| HuggingFace Dataset | `AditHash/sysd-instruct` |
| HuggingFace Model | `AditHash/gemma4-sysd-instruct` |
| Base Model | `unsloth/gemma-4-E4B-it` |
| Language | Python 3.11+ |
| GPU Target | Google Colab T4 (free tier) |

---

## Repo Description

```
A fine-tuned Gemma 4 (E4B) model specialized in system design —
covering distributed systems, scalability, API design, database
selection, caching strategies, and backend architecture patterns.
Trained on a synthetically generated, critic-filtered dataset of
structured system design Q&A. Built for developers preparing for
MAANG interviews and production architecture decisions.
```

## HuggingFace Dataset Card Description

```
High-quality synthetic instruction dataset for system design and
backend architecture. Generated via an agentic pipeline
(Generator → Critic → Rewriter → Deduplicator) using Groq API.
Covers distributed systems, HLD/LLD patterns, database tradeoffs,
caching, API design, message queues, load balancing, and
scalability estimation. Alpaca-style JSONL format,
compatible with Unsloth + SFTTrainer.
```

---

## Project Overview

**What this is:**
An end-to-end open-source pipeline that:
1. Generates a high-quality synthetic instruction dataset for system design
2. Fine-tunes Google's Gemma 4 E4B on that dataset using Unsloth + QLoRA (**text only**)
3. Exposes the fine-tuned model via a FastAPI inference endpoint

**Scope — TEXT ONLY:**
This project fine-tunes **text capabilities only**. Vision and audio modalities are
explicitly out of scope. Do not add image/audio inputs to the dataset,
do not use `FastVisionModel`, do not include multimodal training code.
Use `FastModel` (text) not `FastVisionModel` (multimodal).

**The unique problem being solved:**
Base Gemma 4 answers system design questions generically and inconsistently.
This fine-tuned model gives structured, opinionated, interview-ready answers
in a consistent format every single time — covering capacity estimation,
HLD, LLD, component tradeoffs, and failure modes.

**Why this is unique:**
- No fine-tuned open model exists specifically for system design interviews
- Dataset is 100% synthesizable — no proprietary data needed
- Covers two domains (classic distributed systems + backend architecture)
  that MAANG interviews actually test
- Developer building this dataset simultaneously prepares for MAANG interviews

**Domain coverage (Classic System Design + Backend Architecture):**

| Domain | Examples |
|---|---|
| Classic System Design | URL shortener, rate limiter, distributed cache, notification system |
| Distributed Systems | CAP theorem, consistent hashing, sharding, replication |
| Backend Architecture | API design, REST vs gRPC, microservices vs monolith |
| Database Patterns | SQL vs NoSQL, indexing, sharding, connection pooling |
| Caching | LRU vs LFU, Redis patterns, cache invalidation strategies |
| Scalability | Back-of-envelope estimation, load balancing, horizontal scaling |

---

## Repo Structure

```
gemma4-sysd-instruct/
│
├── CLAUDE.md                        ← You are here
├── README.md                        ← Public-facing documentation
├── requirements.txt                 ← All Python dependencies
├── .env.example                     ← Template for environment variables
├── .gitignore
│
├── dataset_generator/               ← Phase 1: Agentic data pipeline
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── generator_agent.py       ← Generates raw Q&A pairs (temp 0.8)
│   │   ├── critic_agent.py          ← Scores quality (temp 0.1)
│   │   ├── rewriter_agent.py        ← Rewrites rejected samples (temp 0.4)
│   │   └── deduplicator.py          ← Removes near-duplicate samples
│   │
│   ├── topics/
│   │   └── topics.json              ← Seed topic list (25 topics)
│   │
│   ├── schemas/
│   │   └── sample.py                ← Pydantic schema for dataset sample
│   │
│   ├── pipeline.py                  ← Main orchestrator
│   ├── config.py                    ← Temperatures, model names, thresholds
│   └── push_to_hub.py               ← Uploads final JSONL to HuggingFace
│
├── finetuning/                      ← Phase 2: Gemma 4 fine-tuning
│   ├── train.ipynb                  ← Google Colab notebook (primary)
│   ├── train.py                     ← Script version of same pipeline
│   ├── eval.py                      ← Evaluation: all 4 layers
│   ├── eval_questions.json          ← Fixed 20-question domain test suite
│   ├── eval_results.md              ← Auto-generated results (committed post-run)
│   └── config.py                    ← LoRA rank, batch size, epochs, LR etc.
│
├── inference/                       ← Phase 3: Serving the fine-tuned model
│   ├── app.py                       ← FastAPI app
│   ├── model_loader.py              ← Loads adapter from HuggingFace Hub
│   ├── schemas.py                   ← Request/response Pydantic models
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── data/                            ← Local data (gitignored)
│   ├── raw/                         ← Raw generated samples before filtering
│   ├── filtered/                    ← Post-critic, pre-dedup samples
│   ├── final/
│   │   ├── train.jsonl              ← Used for fine-tuning
│   │   └── test.jsonl               ← FROZEN — used only for evaluation
│   └── eval/                        ← Eval outputs and comparison tables
│
└── scripts/
    ├── run_pipeline.sh              ← One-shot dataset generation script
    └── validate_dataset.py          ← Schema validation before pushing
```

---

## Tech Stack

| Layer | Tool | Why |
|---|---|---|
| Dataset generation LLM | Groq API (llama-3.3-70b-versatile) | Free tier, very fast |
| Fallback LLM | AWS Bedrock (Claude 3 Haiku) | Optional, higher quality |
| Deduplication | sentence-transformers (local) | No API cost |
| Fine-tuning | Unsloth + SFTTrainer (HuggingFace TRL) | 2x faster, 60% less VRAM |
| Base model | `unsloth/gemma-4-E4B-it` | Runs on Colab T4 free |
| Model class | `FastModel` (text-only) | NOT FastVisionModel |
| Quantization | QLoRA 4-bit | Fits in 16GB T4 VRAM |
| Training platform | Google Colab (T4) / Kaggle (P100) | Free GPU |
| Inference | FastAPI | Familiar, production-ready |
| Containerization | Docker | Familiar, deployable to AWS |
| Dataset hosting | HuggingFace Hub | Standard |
| Model hosting | HuggingFace Hub | Standard |
| Language | Python 3.11+ | Primary language |

---

## Base Model Decision — `unsloth/gemma-4-E4B-it` NOT `google/gemma-4-E4B-it`

This is a deliberate, documented decision. Do not switch to the Google/HF version.

**They are the same weights. Different packaging.**
Google releases Gemma 4 → HuggingFace hosts at `google/gemma-4-E4B-it` →
Unsloth patches bugs, optimizes, re-hosts at `unsloth/gemma-4-E4B-it`.

| Property | `google/gemma-4-E4B-it` | `unsloth/gemma-4-E4B-it` |
|---|---|---|
| Weights | Original bf16 | Same weights |
| Quantization | None (~8GB download) | Dynamic 4-bit (~2.5GB) |
| Training speed | Baseline | ~1.5x–2x faster |
| VRAM usage | High (won't fit T4) | ~60% less ✅ |
| Gemma 4 bug fixes | Raw, bugs present | All patched ✅ |
| Fine-tuning setup | Needs manual PEFT config | Plug and play ✅ |
| Colab T4 compatible | ❌ OOM | ✅ Works |

**Rule:** Always load from `unsloth/gemma-4-E4B-it`. Never from `google/` or `huggingface/`.

**Ollama is NOT used for fine-tuning.**
Ollama is an inference runtime for running models locally — no role in training.
After fine-tuning, the adapter CAN be exported to GGUF and loaded in Ollama
for local inference — optional post-training step only.

---

## What Fine-tuning Does and Does NOT Do

**Fine-tuning does NOT make a model smarter or more powerful.**
It makes a model more focused and consistent in a specific domain.

```
Base Gemma 4 E4B    →  knows everything, generalist, inconsistent style
Fine-tuned Gemma 4  →  same knowledge, answers like a system design expert
                        in a consistent, structured, interview-ready format
```

| Property | Base Model | Fine-tuned |
|---|---|---|
| Raw intelligence | Full | Same |
| General knowledge | Full | Same |
| System design tone | Generic, wishy-washy | Opinionated, structured |
| Answer format | Inconsistent | Always: Requirements → Estimation → HLD → LLD → Tradeoffs |
| Tradeoff reasoning | Surface level | Production-grade with justification |
| Response quality on domain | 6/10 | 8/10 |

**The real power is in the data pipeline.**
A model trained on 2000 high-quality, critic-filtered samples beats
a model trained on 500 lazy GPT dumps every single time.
The Generator → Critic → Rewriter pipeline is what separates this
dataset from everything else on HuggingFace.

---

## Environment Variables

All secrets live in `.env`. Never commit `.env` — only `.env.example`.

```env
# .env.example

# Groq (primary LLM for dataset generation)
GROQ_API_KEY=your_groq_api_key_here

# AWS Bedrock (optional fallback)
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_DEFAULT_REGION=us-east-1

# HuggingFace (for pushing dataset + model)
HF_TOKEN=your_hf_token_here
HF_DATASET_REPO=AditHash/sysd-instruct
HF_MODEL_REPO=AditHash/gemma4-sysd-instruct

# Pipeline config overrides (optional)
MAX_SAMPLES_PER_TOPIC=20
CRITIC_THRESHOLD=14
DEDUP_SIMILARITY_THRESHOLD=0.85
```

---

## Topic Seeds (`dataset_generator/topics/topics.json`)

```json
{
  "topics": [
    "Design a URL shortening service like bit.ly",
    "Design a rate limiter for an API gateway",
    "Design a distributed cache system",
    "Design a notification system for mobile and email",
    "Design a message queue system like Kafka",
    "Design an API gateway from scratch",
    "Design a ride sharing service like Uber",
    "Design a search autocomplete system",
    "Design a leaderboard system for a gaming platform",
    "Design a distributed job scheduler",
    "Design a file storage system like Amazon S3",
    "Design a payment processing system",
    "Design a logging and monitoring system",
    "SQL vs NoSQL — when to use which",
    "CAP theorem with real world tradeoffs and examples",
    "Database sharding strategies — horizontal vs vertical",
    "Consistent hashing — how it works and when to use it",
    "Load balancing algorithms — round robin vs least connections vs consistent hash",
    "REST vs GraphQL vs gRPC — tradeoffs and when to pick each",
    "Caching strategies — LRU vs LFU, write-through vs write-back",
    "Microservices vs monolith — decision framework",
    "API versioning strategies in production",
    "Database indexing strategies — B-tree vs LSM tree",
    "Event-driven vs request-driven architecture tradeoffs",
    "Back of envelope estimation techniques for system design interviews"
  ]
}
```

---

## Dataset Format

All samples must conform to Alpaca-style JSONL format.
Every line is a valid JSON object. No trailing commas. UTF-8 encoded.

```json
{
  "instruction": "Design a URL shortening service like bit.ly. Handle 100M URLs, 10:1 read to write ratio.",
  "input": "",
  "output": "## Requirements Clarification\n\nFunctional:\n- Shorten a long URL\n- Redirect to original\n\nNon-functional:\n- Latency < 100ms\n- Availability 99.99%\n\n## Capacity Estimation\n\nWrites: ~317/sec. Reads: ~3170/sec. Storage: 50GB.\n\n## High Level Design\n..."
}
```

**Every output MUST follow this 5-section structure (enforce in generator prompt):**
1. `## Requirements Clarification` — functional + non-functional
2. `## Capacity Estimation` — real numbers, units, back-of-envelope math
3. `## High Level Design` — components and data flow
4. `## Deep Dive` — at least one component in detail
5. `## Tradeoffs and Alternatives` — what was NOT chosen and why

**Rules:**
- `instruction` → always non-empty, a clear system design question or tradeoff
- `input` → empty string `""` unless additional constraints are given
- `output` → must follow 5-section structure, minimum 300 words
- Every capacity estimation must contain concrete numbers
- No vague answers — every tradeoff must have a justification
- No "it depends" without immediately explaining what it depends on

---

## Agent Specifications

### Generator Agent (`generator_agent.py`)
- **Model:** `llama-3.3-70b-versatile` via Groq
- **Temperature:** `0.8` — high diversity is the goal
- **Input:** One topic string from `topics.json`
- **Output:** List of 20 raw `{instruction, input, output}` dicts
- **System prompt must enforce:**
  - Domain: system design + backend architecture only
  - Output format: 5-section structure (Requirements → Estimation → HLD → Deep Dive → Tradeoffs)
  - Concrete numbers in every capacity estimation
  - Opinionated recommendations — no "it depends" without explanation
  - Output must be valid JSON array only, nothing else
- **Retry logic:** 3 retries on JSON parse failure with exponential backoff

### Critic Agent (`critic_agent.py`)
- **Model:** `llama-3.3-70b-versatile` via Groq
- **Temperature:** `0.1` — must be deterministic
- **Input:** One sample dict
- **Output:**
  ```json
  {
    "technical_accuracy": 4,
    "structure": 5,
    "depth": 3,
    "has_numbers": 5,
    "total": 17,
    "verdict": "keep",
    "reason": "Good structure and estimation but shallow deep dive"
  }
  ```
- **Scoring (each 1–5):**
  - `technical_accuracy`: Is the design technically sound?
  - `structure`: Does it follow the 5-section format?
  - `depth`: Is the deep dive genuinely detailed?
  - `has_numbers`: Does capacity estimation have real numbers?
- **Threshold:** `total >= 14` out of 20 → keep. Below 14 → rewriter.
- **verdict** must be exactly `"keep"` or `"rewrite"` — nothing else

### Rewriter Agent (`rewriter_agent.py`)
- **Model:** `llama-3.3-70b-versatile` via Groq
- **Temperature:** `0.4`
- **Input:** Rejected sample + critic scores + reason
- **Output:** Rewritten `{instruction, input, output}`
- **After rewrite:** Run through critic again. Still < 14 → discard.
- **Max rewrites per sample:** 1 — never loop more than once

### Deduplicator (`deduplicator.py`)
- **Model:** `all-MiniLM-L6-v2` from `sentence-transformers` (local, no API cost)
- **Method:** Encode all `instruction` fields → cosine similarity matrix
- **Threshold:** similarity > `0.85` → keep first, discard rest
- **Run AFTER** critic + rewriter, BEFORE push to hub

---

## Pipeline Orchestration (`pipeline.py`)

```
topics.json
    │
    ▼  (for each topic)
Generator Agent  →  20 raw samples × 25 topics = 500 raw
    │
    ▼  (for each sample)
Critic Agent  →  score out of 20
    │
    ├── score >= 14  →  keep
    └── score < 14   →  Rewriter Agent
                              │
                              ▼
                        Critic Agent (re-score)
                              │
                              ├── score >= 14  →  keep
                              └── score < 14   →  discard
    │
    ▼
Deduplicator  →  remove near-duplicates
    │
    ▼
90/10 train/test split
    │
    ├── data/final/train.jsonl  →  fine-tuning
    └── data/final/test.jsonl   →  FROZEN, evaluation only
    │
    ▼
push_to_hub.py  →  AditHash/sysd-instruct
```

**Target:** 500–700 clean, diverse, high-quality samples

---

## Fine-tuning Configuration (`finetuning/config.py`)

```python
# CRITICAL: Text-only fine-tuning
# from unsloth import FastModel      ← CORRECT
# from unsloth import FastVisionModel  ← WRONG for this project

BASE_MODEL = "unsloth/gemma-4-E4B-it"
DATASET_REPO = "AditHash/sysd-instruct"

# QLoRA — E4B QLoRA recommended over E2B LoRA (per Unsloth docs)
LOAD_IN_4BIT = True
LORA_RANK = 16
LORA_ALPHA = 16
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
LORA_DROPOUT = 0.0           # 0 is optimal for Unsloth

# Training
MAX_SEQ_LENGTH = 2048
EPOCHS = 3
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4    # Effective batch = 8
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.03
LR_SCHEDULER = "cosine"
OPTIMIZER = "adamw_8bit"

# Unsloth specific — CRITICAL
# Always "unsloth" not True — reduces VRAM + extends context
USE_GRADIENT_CHECKPOINTING = "unsloth"

# Chat template
# "gemma-4" for E2B/E4B (non-thinking)
# "gemma-4-thinking" ONLY for 26B and 31B
CHAT_TEMPLATE = "gemma-4"

OUTPUT_DIR = "./outputs"
PUSH_TO_HUB = True
HUB_MODEL_ID = "AditHash/gemma4-sysd-instruct"
```

**VRAM requirements (official Unsloth docs):**

| Model | VRAM | Free tier? |
|---|---|---|
| E2B LoRA | 8–10GB | ✅ Colab T4 |
| E4B QLoRA | ~10GB | ✅ Colab T4 |
| E4B LoRA 16-bit | 17GB | ❌ Needs Colab Pro |
| 26B-A4B LoRA | >40GB | ❌ Needs A100 |
| 31B QLoRA | 22GB | ❌ Needs Colab Pro |

**Colab T4 estimate:** ~45–90 minutes for 600 samples, 3 epochs

---

## Known Gemma 4 Bugs — Fixed by Unsloth

Do NOT use raw transformers + PEFT for Gemma 4 training. Unsloth patches all of these.

### Bug 1: `use_cache=False` produces garbage outputs (CRITICAL for E4B)
E2B/E4B share KV state across layers. When `use_cache=False` (vanilla QLoRA default),
KV-shared layers produce garbage outputs. Unsloth fixes this automatically.
Never set `use_cache=False` manually.

### Bug 2: Gradient accumulation inflates loss
Standard TRL incorrectly reports loss as 100–300 instead of 13–15.
Always use `use_gradient_checkpointing = "unsloth"`.

### Bug 3: Audio float16 overflow on T4
Not relevant — text-only project. Documented for awareness.

### Bug 4: IndexError on 31B and 26B-A4B
Not relevant for E4B. Documented for awareness.

### Normal behavior — do NOT panic
- E2B and E4B loss of **13–15 is completely normal**
- Panic only if loss is 100+ (gradient bug) or 0.0 (data issue)
- 26B/31B normal range: 1–3

---

## Evaluation Strategy — Raw vs Fine-tuned (`finetuning/eval.py`)

Run every evaluation on BOTH models with identical prompts:
- **Baseline:** `unsloth/gemma-4-E4B-it` (raw)
- **Fine-tuned:** `AditHash/gemma4-sysd-instruct` (with adapter)

Use `data/final/test.jsonl` only — frozen, never seen during training.

---

### Layer 1: Automated Metrics

**Perplexity** — lower = model finds domain more natural
```python
# Expected: 20–40% lower perplexity on system design text
def compute_perplexity(model, tokenizer, texts: list[str]) -> float: ...
```

**ROUGE-L** — output overlap with reference answers
```python
from rouge_score import rouge_scorer
# Fine-tuned should score 15–25% higher
```

**Structure Compliance** — does output follow the 5-section format?
```python
REQUIRED_SECTIONS = ["Requirements", "Capacity", "High Level", "Deep Dive", "Tradeoff"]
# Fine-tuned target: 90%+ compliance
# Base model typical: 30–40%
```

---

### Layer 2: LLM-as-Judge (Groq, blind, free)

Temperature `0.1`. Judge does not know which is base vs fine-tuned.

```python
JUDGE_PROMPT = """
You are a senior software engineer and system design interviewer.
Evaluate two responses to a system design question. Blind evaluation.

Question: {question}
Response A: {response_a}
Response B: {response_b}

Score each (1-10):
1. Technical accuracy
2. Structure and clarity
3. Depth of explanation
4. Concrete numbers and estimation
5. Tradeoff reasoning

Output ONLY valid JSON:
{
  "response_a": {"technical_accuracy":0,"structure":0,"depth":0,"estimation":0,"tradeoffs":0,"total":0},
  "response_b": {"technical_accuracy":0,"structure":0,"depth":0,"estimation":0,"tradeoffs":0,"total":0},
  "winner": "A" | "B" | "tie",
  "reason": "one sentence"
}
"""
# Run on 50 questions from test set
# Report: win rate, avg score delta, per-category breakdown
```

---

### Layer 3: Fixed Domain Test Suite

20 hand-crafted questions, identical every run. Tracks improvement across iterations.

```json
// finetuning/eval_questions.json
[
  "Design a URL shortening service handling 100M URLs with 10:1 read/write ratio",
  "How would you implement rate limiting for a public API with 1M requests/day?",
  "What are the tradeoffs between SQL and NoSQL for a social media feed?",
  "Explain consistent hashing and when you would use it",
  "Design a notification system that handles email, SMS, and push notifications",
  "How do you approach database sharding? What are the strategies?",
  "Design a distributed cache. How do you handle cache invalidation?",
  "REST vs gRPC — when would you choose each in a microservices system?",
  "Design a leaderboard system that updates in real-time for 10M users",
  "How do you handle the CAP theorem in a real distributed system?",
  "Design a message queue. When would you use Kafka vs RabbitMQ vs SQS?",
  "What caching strategy would you use for a read-heavy e-commerce product page?",
  "Design an API gateway. What features does it need in production?",
  "How do you estimate storage requirements for 1M images uploaded per day?",
  "Design a distributed job scheduler that handles 100K jobs per minute",
  "Microservices vs monolith — walk me through your decision framework",
  "How does a load balancer decide which server to route traffic to?",
  "Design a search autocomplete system for a search engine",
  "How would you design the database schema for a ride-sharing app?",
  "Explain back-of-envelope estimation and walk through a real example"
]
```

---

### Layer 4: Side-by-Side Comparison Table (auto-generated for README)

```markdown
| Question | Base Score | Fine-tuned Score | Delta | Winner |
|---|---|---|---|---|
| URL Shortener | 6.2 | 8.7 | +2.5 | Fine-tuned ✅ |
| Rate Limiter | 5.8 | 8.1 | +2.3 | Fine-tuned ✅ |
| SQL vs NoSQL | 7.1 | 8.9 | +1.8 | Fine-tuned ✅ |
| avg | 6.1 | 8.4 | +2.3 | +38% improvement |
```

Goes directly into README and LinkedIn post.

### Success Criteria

| Metric | Minimum | Good |
|---|---|---|
| Perplexity reduction | 15% lower | 30%+ lower |
| ROUGE-L improvement | +0.05 | +0.15 |
| Structure compliance | 70%+ | 90%+ |
| LLM-Judge win rate | 60% | 75%+ |
| Domain test avg delta | +1.5 pts | +2.5 pts |

### Running Evaluation

```bash
python finetuning/eval.py \
  --base-model unsloth/gemma-4-E4B-it \
  --finetuned-model AditHash/gemma4-sysd-instruct \
  --test-split data/final/test.jsonl \
  --output eval_results.md
```

---

## Inference API (`inference/app.py`)

```
POST /design
  body:     { "question": "Design a rate limiter for 1M requests/day" }
  response: { "answer": "...", "model": "gemma4-sysd-instruct", "latency_ms": 230 }

GET /health
  response: { "status": "ok", "model_loaded": true }
```

- Model loads once at startup via `lifespan` context manager
- Adapter pulled from HuggingFace Hub on first start
- Input validated with Pydantic
- Max input: 512 tokens
- Max output: 2048 tokens (system design answers are long)

---

## Code Conventions

- Python 3.11+
- Type hints on all function signatures — no exceptions
- Pydantic v2 for all data validation and schemas
- `async/await` throughout — no sync blocking calls
- `loguru` for logging — never `print()`, never stdlib `logging`
- f-strings only — no `.format()` or `%`
- Max line length: 100 characters
- Google-style docstrings on all public functions and classes
- Always `pathlib.Path` — never `os.path`
- Always `httpx` async — never `requests`
- `json.dumps(ensure_ascii=False)` for all JSONL writes
- `tenacity` for all API retry logic with exponential backoff
- Stdlib → third party → local import grouping (one blank line between)
- No wildcard imports

---

## What NOT to Do

- Do NOT use `FastVisionModel` — text-only, use `FastModel`
- Do NOT add image or audio fields to dataset schema
- Do NOT hardcode API keys, tokens, or secrets
- Do NOT commit `data/` directory (gitignored)
- Do NOT commit `.env` (only `.env.example`)
- Do NOT use `print()` — use `loguru`
- Do NOT run deduplication before critic scoring
- Do NOT use QLoRA on 26B-A4B (MoE — use 16-bit LoRA)
- Do NOT set `use_gradient_checkpointing = True` — always `"unsloth"`
- Do NOT set `use_cache = False` manually — let Unsloth handle it
- Do NOT push dataset with fewer than 400 samples
- Do NOT evaluate on training data — `test.jsonl` only
- Do NOT generate samples outside system design / backend architecture domain

---

## Official References

| Resource | URL |
|---|---|
| Unsloth Gemma 4 Fine-tuning Guide | https://unsloth.ai/docs/models/gemma-4/train |
| Unsloth Core Code-based Guide | https://unsloth.ai/docs/models/gemma-4/train#unsloth-core-code-based-guide |
| E4B Text Only Colab (primary reference) | https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma4_(E4B)-Text.ipynb |
| E4B Vision + Text Colab (reference only) | https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma4_(E4B)-Vision.ipynb |
| Gemma 4 E2B Vision Colab (reference only) | https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma4_(E2B)-Vision.ipynb |
| Gemma 4 31B on Kaggle | https://www.kaggle.com/code/danielhanchen/gemma4-31b-unsloth |
| Gradient Accumulation Bug Fix Blog | https://unsloth.ai/blog/gradient |

Always check these before changing any training parameters or model loading code.

---

## Running the Pipeline

```bash
# 1. Install dependencies
uv sync

# 2. Set up environment
cp .env.example .env
# Fill in GROQ_API_KEY and HF_TOKEN

# 3. Generate dataset
uv run python dataset_generator/pipeline.py

# 4. Validate before push
uv run python scripts/validate_dataset.py --input data/final/dataset.jsonl

# 5. Push to HuggingFace
python dataset_generator/push_to_hub.py

# 6. Fine-tune (open on Google Colab)
# finetuning/train.ipynb
# Or locally:
python finetuning/train.py

# 7. Evaluate
python finetuning/eval.py \
  --base-model unsloth/gemma-4-E4B-it \
  --finetuned-model AditHash/gemma4-sysd-instruct \
  --test-split data/final/test.jsonl \
  --output eval_results.md

# 8. Serve
cd inference && docker-compose up --build
```

---

## Git Conventions

```
<type>: <short description>

Types: feat | fix | data | train | eval | docs | refactor | test | chore
```

Examples:
```
feat: add critic agent with groq integration
data: expand topics.json with 5 more system design topics
train: add cosine LR scheduler to finetuning config
eval: add structure compliance metric to eval pipeline
fix: handle json parse failure in generator retry logic
```

Branches: `main` (stable) → `dev` (active) → `feature/<name>`

---

## Milestones

| Week | Deliverable |
|---|---|
| Week 1 | All 4 agents + pipeline runs end-to-end + dataset on HuggingFace |
| Week 2 | Gemma 4 E4B fine-tuned on Colab + adapter on HuggingFace |
| Week 3 | Evaluation complete + results table in README |
| Week 4 | FastAPI inference + Dockerfile + deployed on AWS EC2 |

---

## Resume Bullets

```
• Identified gap in open-source LLM ecosystem — no model fine-tuned
  specifically for system design interviews — and built one end-to-end

• Built multi-agent synthetic data pipeline (Generator → Critic → Rewriter →
  Deduplicator) using Groq API, producing 600+ critic-filtered system design
  Q&A samples published at AditHash/sysd-instruct on HuggingFace

• Fine-tuned Gemma 4 E4B using Unsloth QLoRA (4-bit) on Google Colab T4,
  achieving 38% improvement over base model on structured domain evaluation

• Published open-source model adapter AditHash/gemma4-sysd-instruct on
  HuggingFace with full eval results and blind comparison methodology

• Built async FastAPI inference endpoint, containerized with Docker,
  deployed on AWS EC2
```