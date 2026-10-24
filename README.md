# PersonaRAG

**An evidence-grounded conversational AI that simulates the publicly documented perspectives of public figures, using Retrieval-Augmented Generation.**

You can't ask Elon Musk, Tim Cook or Ratan Tata a question directly, but they've published decades of blog posts, interviews, letters and talks. PersonaRAG collects that material into a per-person knowledge base, retrieves the passages relevant to your question, and has an LLM answer **in that person's documented voice, citing the sources it used**. It flags when it's extrapolating and says so when the sources don't cover the question.

> ⚠️ This is an AI simulation built from public material. It is not the real person and does not represent their private views.

![PersonaRAG demo: asking Elon Musk a question, the answer streaming in with citations, and opening the cited passage](docs/demo.gif)

▶ **[Watch the full 2-minute demo (MP4)](docs/demo.mp4)**: chat with citations, a grounding badge, an out-of-scope question, and a three-persona panel with a moderator. Recorded on a laptop with the local `qwen2.5:3b` model; the panel's waiting time is sped up 8×.

## Features

- **Persona RAG pipeline:** ingest blogs, YouTube videos, podcasts, PDFs, text and subtitle files → sentence-aware chunking → `bge-small` embeddings → ChromaDB (one collection per persona).
- **Video and podcast ingestion:** YouTube captions are fetched directly; podcasts that aren't on YouTube, local audio files and videos without captions are transcribed locally with Whisper (`faster-whisper` + `yt-dlp`).
- **Hybrid retrieval with re-ranking:** BM25 keyword search and embedding search are merged with Reciprocal Rank Fusion, then a cross-encoder re-orders the candidates. Measured on the eval: hit-rate 0.90 → 1.00, MRR 0.70 → 0.78.
- **Inline citations:** every claim is tied to a numbered source `[n]`, and the UI shows which retrieved passages were actually cited.
- **Grounding indicator:** model-free signals (citation coverage, retrieval similarity, invalid-citation detection) rate each answer *strong / partial / weak / ungrounded*.
- **Interaction modes:** Ask, Career, Startup, Technical, Debate, Interview.
- **Panel mode:** several personas answer the same question in parallel, then an AI moderator compares where they agree and differ.
- **Conversation memory:** follow-up questions keep context, and short follow-ups are expanded for retrieval.
- **Pluggable LLMs:** Claude via the Anthropic API, or a local model via Ollama (runs offline on a 4 GB GPU).
- **Evaluation harness:** retrieval hit-rate and MRR, keyword recall, citation coverage, abstention accuracy on out-of-scope questions, and optional LLM-as-judge faithfulness.
- **React web app:** streaming answers, clickable citation chips that open the exact source passage, grounding badges, a live panel view where each panelist appears as they finish, and a responsive layout for phones. Built with React 19, TypeScript and Vite on top of the FastAPI streaming API.
- **API and alternatives:** FastAPI REST API with NDJSON streaming (`/chat/stream`, `/panel/stream`) and OpenAPI docs; a Streamlit app is kept for quick experiments.

## Architecture

```
 question ──► query expansion (uses chat history for short follow-ups)
                │
                ▼
         ┌─► bge-small embedding ──► ChromaDB (cosine) ──► top 20 ─┐
         │                                                          ├─► RRF fusion ──► relevance gate
         └─► BM25 keyword index ───────────────────────────► top 20 ─┘   (cosine ≥ 0.5)
                                                                              │
                                              cross-encoder re-ranks top 12 ──► top-k passages
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
├── frontend/          # React + TypeScript + Vite web app
│   └── src/
│       ├── api.ts         # typed API client and NDJSON stream reader
│       ├── App.tsx        # layout: personas, modes, knowledge base, tabs
│       └── components/    # Chat, Panel, Answer (citations), SourcesDrawer, GroundingBadge
├── tests/
├── app.py             # Streamlit UI (quick experiments)
└── api.py             # FastAPI: REST + streaming endpoints, serves the built web app
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

Build the knowledge bases:

```bash
python -m personarag.ingest --all
```

### Run the web app

One-time build of the React app (needs Node 18+):

```bash
cd frontend && npm install && npm run build && cd ..
```

Then a single server runs everything. Open http://127.0.0.1:8000 for the app, or http://127.0.0.1:8000/docs for the API docs:

```bash
uvicorn api:app --port 8000
```

### Developing the frontend

Run the API and the Vite dev server side by side; Vite proxies `/api/*` to port 8000 and hot-reloads on save:

```bash
uvicorn api:app --port 8000
cd frontend && npm run dev          # http://localhost:5173
```

The Streamlit app is still available for quick experiments: `streamlit run app.py`.

### Back up the knowledge bases

```bash
python -m personarag.snapshot export     # data/snapshot/*.jsonl: every chunk's text and metadata
python -m personarag.snapshot restore    # rebuild the vector store from the snapshot
```

Restoring re-embeds the text, so it works across ChromaDB versions and never re-downloads sources. It also repairs a corrupted on-disk index.

### Re-record the demo video

```bash
pip install playwright imageio-ffmpeg && python -m playwright install chromium
cd frontend && npm run build && cd ..
uvicorn api:app --port 8000              # in another terminal
python scripts/record_demo.py            # -> docs/demo.mp4 and docs/demo.gif
```

The script drives the real app in a headless browser (visible cursor, typed questions, on-screen captions), then edits the segments with ffmpeg.

## How videos and podcasts become knowledge

The LLM never watches or listens to anything: it only ever reads text. Every source is turned into a transcript at ingestion time, then chunked and embedded like any article.

```
YouTube link ──► captions available? ──yes──► YouTube transcript (human or auto-generated)
                        │
                        no
                        ▼
Podcast page / MP3 / caption-less video ──► yt-dlp downloads audio ──► Whisper speech-to-text
                                                                              │
Blog / letter / transcript page ──► trafilatura extracts article text          │
                                                                              ▼
                                          text ──► chunks ──► embeddings ──► vector DB
```

Transcripts are cached in `personas/<id>/cache/`, so each video is downloaded or transcribed only once.

Limits to be aware of:

- **Speaker attribution.** YouTube captions have no speaker labels, so an interviewer's questions are mixed into the transcript. Prefer sources with labelled transcripts (such as lexfridman.com) or solo talks where possible.
- **Auto-caption quality.** Auto-generated captions misspell names and have no punctuation. Add `whisper: true` to a YouTube entry to transcribe it with Whisper instead.
- **Speed.** Whisper `base.en` on CPU transcribes about 4 minutes of audio per minute. Set `WHISPER_DEVICE=cuda` or `WHISPER_MODEL=small.en` in `.env` to trade speed for accuracy.

### Current corpus

| Persona | Sources |
|---|---|
| Elon Musk (275 chunks) | 3 Tesla blog posts, Lex Fridman #400 transcript, 3 TED interviews, Stanford eCorner talk, Wikiquote |
| Mark Zuckerberg (348 chunks) | "Open Source AI Is the Path Forward" letter, Lex Fridman #383 and #398, 2 Dwarkesh Podcast episodes, Acquired interview, Harvard commencement 2017 |
| Tim Cook (20 chunks) | 2016 customer letter on encryption, Stanford 2019 and MIT 2017 commencement speeches, Wikiquote |
| Ratan Tata (94 chunks) | Computer History Museum oral history, Stanford GSB View From The Top, 2 CNBC-TV18 interviews |

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
     - type: audio                          # transcribed with Whisper
       url: https://example.com/episode.mp3 # or a podcast episode page, or path: raw/episode.mp3
       kind: podcast
   ```
   Or drop `.txt / .md / .pdf / .srt / .vtt` files, or `.mp3 / .m4a / .wav` audio, into `personas/<id>/raw/`. A text file can carry its own metadata in a YAML header:
   ```
   ---
   title: Commencement speech
   date: 2019-06-16
   url: https://...
   ---
   transcript text...
   ```
3. Stop the Streamlit app, then run `python -m personarag.ingest <id>` (the app and the ingester can't rebuild the same ChromaDB collection at the same time).

Only use the person's own words: talks, interviews, letters and posts. Articles or summaries *about* them get retrieved and voiced in the first person as if they had said it.

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

**Current numbers** (expanded 22-question set, 275-chunk corpus, hybrid + re-rank retrieval, same `qwen2.5:3b`): every answer cites sources, citation coverage **0.68**, **0** invalid citations, 0 answerable questions wrongly declined, 1 of 2 out-of-scope questions declined.

> **Scorer fix.** The original citation-coverage scorer split text on sentence boundaries *before* attaching citations, so a citation written after the full stop ("…cheaper cars. [1]") was counted as belonging to no sentence. This under-counted coverage in the table above (0.43); the fixed scorer is covered by a unit test.

What the eval caught and how it was fixed:

1. **No citations.** The 3B model ignored citation rules placed in the system prompt. Moving a short format reminder to the end of the user turn (recency) took cited answers from 0% to 100%.
2. **Example copying.** A concrete example sentence in that reminder was pasted verbatim into unrelated answers. Replacing it with an abstract format description removed this and raised citation coverage from 0.22 to 0.43.
3. **Fabricated answers to out-of-scope questions** (an invented favourite pizza topping). When retrieval returns nothing above the similarity threshold, the prompt now explicitly tells the model to decline in character.
4. **Evaluator bug.** "I haven't spoken about this directly, *but*..." was being scored as an abstention, although it introduces an extrapolation. The abstention detector now excludes that pattern.

### Retrieval: vector vs hybrid vs re-ranked

`python -m eval.run_eval <persona> --compare` runs every retrieval mode on the same questions (no LLM calls). Elon Musk set: 20 answerable questions over 211 chunks (blog posts, Lex Fridman transcript, TED interviews, Wikiquote).

| Mode | hit_rate@6 | MRR | keyword_recall | ms / query (CPU) |
|---|---|---|---|---|
| Vector only | 0.90 | 0.70 | 0.92 | 52 |
| Hybrid (BM25 + vector, RRF) | 0.95 | 0.73 | 1.00 | 65 |
| **Hybrid + cross-encoder re-rank** (default) | **1.00** | **0.78** | **1.00** | 820 |

- BM25 recovers passages that use exact names and terms the embedding model under-weights (Grok, Diablo, Optimus), which fixed the vector-only misses.
- The cross-encoder (`ms-marco-MiniLM-L-6-v2`) reads the question and passage together, so it ranks the passage that actually answers the question above ones that merely share the topic.
- Re-ranking all ~40 fused candidates at full length cost 4.0 s per query on CPU. Capping it to the top 12 candidates truncated to 256 tokens brought that to 0.8 s with MRR 0.79 → 0.78.
Mark Zuckerberg set: 11 answerable questions over 348 chunks (2024 letter, two Lex Fridman episodes, two Dwarkesh episodes, Acquired, Harvard commencement).

| Mode | hit_rate@6 | MRR | keyword_recall |
|---|---|---|---|
| Vector only | 1.00 | 0.87 | 0.91 |
| Hybrid (BM25 + vector, RRF) | 0.91 | 0.85 | 1.00 |
| **Hybrid + cross-encoder re-rank** | **1.00** | **0.89** | **1.00** |

- Hybrid retrieval alone is slightly *worse* here: BM25 pulls in long podcast passages that repeat keywords like "AI" and "open", and one Meta AI question lost its correct passage from the top 6. The re-ranker recovers it, which is why `hybrid_rerank` is the default rather than `hybrid`.
- **Label correction.** On the first run, re-ranking appeared to lower MRR (0.87 → 0.80). Reading the promoted passages showed they were correct answers in his own words from podcasts added later (Dwarkesh on competing with China, Acquired on Apple vs open platforms), while the labels listed only the 2024 letter. Those two questions now accept both sources. Lesson: when the corpus grows, eval labels must be re-audited, or better retrieval gets scored as worse.
- Choose a mode with `RETRIEVAL_MODE=vector|hybrid|hybrid_rerank` in `.env`; use `hybrid` on slow machines.

**Effect of growing the corpus.** Expanding the Elon Musk corpus from 61 to 211 chunks (adding a Lex Fridman transcript and TED interviews) kept hit_rate@6 at 1.00, but MRR fell from 0.90 to 0.72 and keyword recall from 1.00 to 0.89: the new long-form interviews also discuss Tesla's strategy and compete with the primary blog posts for the top rank. This regression is what motivated the hybrid retrieval and re-ranking above.

Known limits with a 3B local model: it still extrapolates on some out-of-scope questions, sometimes slips into third person, and occasionally cites a source number that wasn't provided (the grounding report flags these). A larger model is expected to do better on all three; rerun the eval with `LLM_PROVIDER=claude` to compare.

## Resume bullets

**PersonaRAG: Evidence-Grounded Persona Simulation with RAG** · Python, ChromaDB, Sentence-Transformers, Ollama, Claude API, FastAPI, Streamlit

- Built a retrieval-augmented generation system that simulates public figures' documented perspectives, answering in their voice with inline citations to a per-persona corpus of blog posts, letters, transcripts and quotations.
- Designed the ingestion pipeline (blogs, YouTube videos, podcasts via Whisper speech-to-text, PDFs) with sentence-aware overlapping chunking, `bge-small` embeddings and a ChromaDB vector store.
- Implemented hybrid retrieval (BM25 + dense vectors fused with Reciprocal Rank Fusion) and cross-encoder re-ranking, raising hit-rate@6 from 0.90 to 1.00 and MRR from 0.70 to 0.78, and cut re-ranking latency 5× (4.0 s → 0.8 s on CPU) by bounding the candidate pool.
- Built an evaluation harness measuring hit-rate, MRR, citation coverage, hallucinated-citation detection, abstention accuracy and LLM-as-judge faithfulness; used it to iterate on prompts, taking cited answers from 0% to 100% on a 3B local model.
- Added a grounding indicator, six interaction modes, conversation memory, and a multi-persona panel in which personas answer in parallel and an AI moderator compares their views.
- Built a React + TypeScript web app on a FastAPI NDJSON streaming API: token-by-token answers, clickable citations that open the source passage, grounding badges, and a live multi-persona panel; the LLM backend is pluggable (Claude API or a quantized local model via Ollama on a 4 GB GPU).

```bash
pytest
```

## Roadmap

- [x] V1: persona chatbot over an LLM
- [x] V2: RAG pipeline (chunking, embeddings, vector DB)
- [x] V3: persona profiles, citations, memory, grounding indicator
- [x] V4: multi-persona panel with moderator
- [x] Evaluation harness
- [x] Video and podcast ingestion with Whisper fallback
- [ ] Speaker diarization so interviewer questions are not attributed to the persona
- [x] Hybrid retrieval (BM25 + vectors) and cross-encoder re-ranking
- [x] React front end on top of the FastAPI streaming endpoint
- [ ] LoRA/QLoRA fine-tune of a small model on interview transcripts for conversational style (RAG keeps supplying the facts)
- [ ] Time-aware answers ("what did they think in 2016 vs now?") using source dates

## Ethics

- Only publicly available material is used, and every answer is traceable to its sources.
- The UI labels every conversation as a simulation, and the model is instructed never to claim to be the real person.
- The model is told to flag extrapolation and to abstain when the sources don't cover a question; the evaluation measures both.
