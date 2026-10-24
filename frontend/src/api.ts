// Typed client for the PersonaRAG FastAPI backend.

const BASE = import.meta.env.VITE_API_URL ?? "/api";

export interface Persona {
  id: string;
  name: string;
  tagline: string;
  avatar: string;
  domains: string[];
  chunks: number;
}

export interface Source {
  n: number;
  label: string;
  title: string;
  kind?: string;
  date?: string;
  url?: string;
  score: number;
  rerank: number | null;
  text: string;
}

export interface KnowledgeSource {
  source_id: string;
  title: string;
  kind?: string;
  date?: string;
  url?: string;
  chunks: number;
}

export type GroundingLevel = "strong" | "partial" | "weak" | "ungrounded";

export interface Grounding {
  level: GroundingLevel;
  cited: number[];
  invalid_citations: number[];
  citation_coverage: number;
  top_similarity: number;
  sources_retrieved: number;
}

export interface ServerConfig {
  llm: string;
  embedding_model: string;
  retrieval_mode: string;
  top_k: number;
}

export interface Turn {
  role: "user" | "assistant";
  content: string;
}

export type ChatEvent =
  | { type: "sources"; sources: Source[] }
  | { type: "token"; text: string }
  | { type: "done"; grounding: Grounding }
  | { type: "error"; message: string };

export interface PanelAnswer {
  persona: string;
  answer: string;
  grounding: Grounding;
  sources: Source[];
}

export type PanelEvent =
  | ({ type: "answer" } & PanelAnswer)
  | { type: "summary"; text: string }
  | { type: "error"; message: string };

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(BASE + path);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
}

export const api = {
  config: () => getJSON<ServerConfig>("/config"),
  personas: () => getJSON<Persona[]>("/personas"),
  modes: () => getJSON<Record<string, string>>("/modes"),
  sources: (persona: string) => getJSON<KnowledgeSource[]>(`/personas/${persona}/sources`),
};

/** POST a JSON body and yield each NDJSON line of the streamed response as an object. */
async function* streamNDJSON<T>(path: string, body: unknown, signal?: AbortSignal): AsyncGenerator<T> {
  const res = await fetch(BASE + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) {
    const detail = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText} ${detail}`.trim());
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let newline: number;
    while ((newline = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, newline).trim();
      buffer = buffer.slice(newline + 1);
      if (line) yield JSON.parse(line) as T;
    }
  }
  if (buffer.trim()) yield JSON.parse(buffer) as T;
}

export const streamChat = (
  req: { persona: string; question: string; mode: string; history: Turn[] },
  signal?: AbortSignal,
) => streamNDJSON<ChatEvent>("/chat/stream", req, signal);

export const streamPanel = (
  req: { personas: string[]; question: string; mode: string },
  signal?: AbortSignal,
) => streamNDJSON<PanelEvent>("/panel/stream", req, signal);
