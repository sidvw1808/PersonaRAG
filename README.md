# PersonaRAG

**An evidence-grounded conversational AI that simulates the publicly documented perspectives of public figures, using Retrieval-Augmented Generation.**

You can't ask Elon Musk, Tim Cook or Ratan Tata a question directly, but they've published decades of blog posts, interviews, letters and talks. PersonaRAG collects that material into a per-person knowledge base, retrieves the passages relevant to your question, and has an LLM answer **in that person's documented voice, citing the sources it used**. It flags when it's extrapolating and says so when the sources don't cover the question.

> ⚠️ This is an AI simulation built from public material. It is not the real person and does not represent their private views.

## Features

- **Persona RAG pipeline:** ingest URLs, YouTube transcripts, PDFs, text and subtitle files → sentence-aware chunking → `bge-small` embeddings → ChromaDB (one collection per persona).
- **Inline citations:** every claim is tied to a numbered source `[n]`, and the UI shows which retrieved passages were actually cited.
- **Grounding indicator:** model-free signals (citation coverage, retrieval similarity, invalid-citation detection) rate each answer *strong / partial / weak / ungrounded*.
- **Interaction modes:** Ask, Career, Startup, Technical, Debate, Interview.
- **Panel mode:** several personas answer the same question in parallel, then an AI moderator compares where they agree and differ.
- **Conversation memory:** follow-up questions keep context, and short follow-ups are expanded for retrieval.
- **Pluggable LLMs:** Claude via the Anthropic API, or a local model via Ollama (runs offline on a 4 GB GPU).
- **Evaluation harness:** retrieval hit-rate and MRR, keyword recall, citation coverage, abstention accuracy on out-of-scope questions, and optional LLM-as-judge faithfulness.
- **Two front ends:** a Streamlit app and a FastAPI REST API with NDJSON streaming.

## Architecture

```
 question ──► query expansion (uses chat history for short follow-ups)
                │
                ▼
         bge-small embedding ──► ChromaDB (persona collection, cosine) ──► top-k passages
                                                                              │
 persona.yaml ──► system prompt (voice, background, rules)  ◄─────────────────┤
                                                                              ▼
                                                          user turn: numbered SOURCES + mode + question
                                                                              │
                                                                              ▼
                                                                  LLM (Claude / Ollama), streamed
                                                                              │
                                                                              ▼
                                                     answer with [n] citations ──► grounding report
```

```
PersonaRAG/
├── personarag/
│   ├── config.py      # all settings, overridable via .env
│   ├── loaders.py     # URL / YouTube / PDF / txt / srt loaders with on-disk cache
│   ├── chunking.py    # sentence-aware chunking with overlap
│   ├── store.py       # embeddings + ChromaDB vector store
│   ├── llm.py         # Claude and Ollama backends (streaming)
│   ├── prompts.py     # persona system prompt, modes, moderator prompt
│   ├── engine.py      # retrieve → prompt → generate → grounding report
│   ├── panel.py       # multi-persona panel + moderator
│   └── ingest.py      # CLI to build a persona's knowledge base
├── personas/<id>/
│   ├── persona.yaml   # name, background, communication style, known positions
│   ├── sources.yaml   # URLs and YouTube videos to ingest
│   └── raw/           # drop transcripts/PDFs here; ingested automatically
├── eval/              # evaluation question sets + runner
├── tests/
├── app.py             # Streamlit UI
└── api.py             # FastAPI
```

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy env.template .env          # macOS/Linux: cp env.template .env
```

Pick an LLM in `.env`:

- **Claude (best answers):** set `ANTHROPIC_API_KEY` (from console.anthropic.com). The default model is `claude-opus-5-5`; for a cheaper option, set `CLAUDE_MODEL=claude-haiku-5-5`.
- **Local and free:** `ollama pull qwen2.5:3b`, then set `LLM_PROVIDER=ollama`.

Build the knowledge bases, then run the app:

```bash
python -m personarag.ingest --all
streamlit run app.py
```

Or run the API (interactive docs at http://127.0.0.1:8000/docs):

```bash
uvicorn api:app --reload
```

## Adding sources or a new persona

1. Create `personas/<id>/persona.yaml` (copy an existing one).
2. Add sources to `personas/<id>/sources.yaml`:
   ```yaml
   sources:
     - type: youtube
       id: VIDEO_ID
       title: Long-form interview
       date: "2024-03-01"
       kind: interview
     - type: url
       url: https://example.com/essay
       kind: essay
   ```
   Or drop `.txt / .md / .pdf / .srt / .vtt` files into `personas/<id>/raw/`. A text file can carry its own metadata in a YAML header:
   ```
   ---
   title: Commencement speech
   date: 2019-06-16
   url: https://...
   ---
   transcript text...
   ```
3. Run `python -m personarag.ingest <id>`.

Prefer primary sources (their own blog posts, letters, talks, interviews) over articles *about* them, and record dates so answers can say *when* something was said.

## Evaluation

```bash
python -m eval.run_eval elon_musk                    # retrieval metrics (no LLM cost)
python -m eval.run_eval elon_musk --generate --judge # + grounding, abstention, faithfulness
```

| Metric | What it measures |
|---|---|
| hit_rate@k | Share of questions where the expected source is among the retrieved passages |
| MRR | How high the first correct passage ranks |
| keyword_recall | Share of expected key facts present in the retrieved text |
| citation_coverage | Share of answer sentences that carry a citation |
| invalid_citations | Citations pointing at sources that weren't provided (hallucinated references) |
| abstention_accuracy | On questions the corpus can't answer, how often the system says so instead of inventing an answer |
| faithfulness | LLM-as-judge score: are the claims supported by the cited sources? |

Question sets live in `eval/<persona>.yaml`. Results are saved to `eval/results/`.

### Measured results

Elon Musk persona, 12 questions (10 answerable from the corpus, 2 deliberately out of scope), corpus of 4 sources / 61 chunks, `bge-small-en-v1.5` embeddings, top-k = 6, local `qwen2.5:3b` via Ollama on a GTX 1650 (4 GB).

| Metric | Baseline prompt | After prompt iterations |
|---|---|---|
| hit_rate@6 | 1.00 | 1.00 |
| MRR | 0.90 | 0.90 |
| keyword_recall | 1.00 | 1.00 |
| Answers containing citations | 0% | **100%** |
| Citation coverage | 0.00 | **0.43** |
| Out-of-scope questions correctly declined | 0 / 2 | 1 / 2 |
| Answerable questions wrongly declined | 0 | 0 |
| Invalid (hallucinated) citations detected | 0 | 1 |

What the eval caught and how it was fixed:

1. **No citations.** The 3B model ignored citation rules placed in the system prompt. Moving a short format reminder to the end of the user turn (recency) took cited answers from 0% to 100%.
2. **Example copying.** A concrete example sentence in that reminder was pasted verbatim into unrelated answers. Replacing it with an abstract format description removed this and raised citation coverage from 0.22 to 0.43.
3. **Fabricated answers to out-of-scope questions** (an invented favourite pizza topping). When retrieval returns nothing above the similarity threshold, the prompt now explicitly tells the model to decline in character.
4. **Evaluator bug.** "I haven't spoken about this directly, *but*..." was being scored as an abstention, although it introduces an extrapolation. The abstention detector now excludes that pattern.

Known limits with a 3B local model: it still extrapolates on some out-of-scope questions, sometimes slips into third person, and occasionally cites a source number that wasn't provided (the grounding report flags these). A larger model is expected to do better on all three; rerun the eval with `LLM_PROVIDER=claude` to compare.

## Resume bullets

**PersonaRAG: Evidence-Grounded Persona Simulation with RAG** · Python, ChromaDB, Sentence-Transformers, Ollama, Claude API, FastAPI, Streamlit

- Built a retrieval-augmented generation system that simulates public figures' documented perspectives, answering in their voice with inline citations to a per-persona corpus of blog posts, letters, transcripts and quotations.
- Designed the ingestion pipeline (web pages, YouTube transcripts, PDFs, subtitles) with sentence-aware overlapping chunking, `bge-small` embeddings and a ChromaDB vector store; retrieval reached 100% hit-rate@6 and 0.90 MRR on a hand-labelled question set.
- Built an evaluation harness measuring hit-rate, MRR, citation coverage, hallucinated-citation detection, abstention accuracy and LLM-as-judge faithfulness; used it to iterate on prompts, taking cited answers from 0% to 100% on a 3B local model.
- Added a grounding indicator, six interaction modes, conversation memory, and a multi-persona panel in which personas answer in parallel and an AI moderator compares their views.
- Made the LLM backend pluggable (Claude API or quantized local models via Ollama running on a 4 GB GPU), served through a FastAPI REST API with NDJSON streaming and a Streamlit UI.

```bash
pytest
```

## Roadmap

- [x] V1: persona chatbot over an LLM
- [x] V2: RAG pipeline (chunking, embeddings, vector DB)
- [x] V3: persona profiles, citations, memory, grounding indicator
- [x] V4: multi-persona panel with moderator
- [x] Evaluation harness
- [ ] Hybrid retrieval (BM25 + vectors) and cross-encoder re-ranking
- [ ] React front end on top of the FastAPI streaming endpoint
- [ ] LoRA/QLoRA fine-tune of a small model on interview transcripts for conversational style (RAG keeps supplying the facts)
- [ ] Time-aware answers ("what did they think in 2016 vs now?") using source dates

## Ethics

- Only publicly available material is used, and every answer is traceable to its sources.
- The UI labels every conversation as a simulation, and the model is instructed never to claim to be the real person.
- The model is told to flag extrapolation and to abstain when the sources don't cover a question; the evaluation measures both.
