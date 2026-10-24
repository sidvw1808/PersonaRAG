import { useEffect, useRef, useState } from "react";
import { ApiError, streamChat, type Persona, type Turn } from "../api";
import { uid, type DrawerState, type Message } from "../types";
import { Answer } from "./Answer";
import { GroundingBadge } from "./GroundingBadge";

const STARTERS: Record<string, string[]> = {
  elon_musk: [
    "Why did Tesla start with an expensive sports car?",
    "Why is Grok modeled after The Hitchhiker's Guide?",
    "Why does humanity need to become multiplanetary?",
  ],
  mark_zuckerberg: [
    "Why should frontier AI models be open source?",
    "What are Codec Avatars?",
    "How should the US compete with China on AI?",
  ],
  tim_cook: [
    "Why did Apple refuse to unlock the iPhone for the FBI?",
    "What advice did you give the Stanford class of 2019?",
    "Why is privacy a human right?",
  ],
  ratan_tata: [
    "What advice do you give young entrepreneurs?",
    "Why did the Tata Group expand beyond India?",
    "How do you choose which startups to invest in?",
  ],
};

interface Props {
  persona: Persona;
  mode: string;
  messages: Message[];
  setMessages: (update: (prev: Message[]) => Message[]) => void;
  drawer: DrawerState | null;
  openDrawer: (state: DrawerState) => void;
}

export function Chat({ persona, mode, messages, setMessages, drawer, openDrawer }: Props) {
  const [input, setInput] = useState("");
  const abortRef = useRef<AbortController | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const busy = messages.some((m) => m.status === "streaming");

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  // Abort an in-flight answer when switching persona or leaving the view.
  useEffect(() => () => abortRef.current?.abort(), [persona.id]);

  const patch = (id: string, fn: (m: Message) => Message) =>
    setMessages((prev) => prev.map((m) => (m.id === id ? fn(m) : m)));

  async function send(question: string) {
    question = question.trim();
    if (!question || busy) return;
    setInput("");
    const history: Turn[] = messages
      .filter((m) => m.status === "done")
      .map((m) => ({ role: m.role, content: m.content }));
    const answerId = uid();
    setMessages((prev) => [
      ...prev,
      { id: uid(), role: "user", content: question, status: "done" },
      { id: answerId, role: "assistant", content: "", status: "streaming" },
    ]);

    const controller = new AbortController();
    abortRef.current = controller;
    try {
      for await (const ev of streamChat({ persona: persona.id, question, mode, history }, controller.signal)) {
        if (ev.type === "sources") patch(answerId, (m) => ({ ...m, sources: ev.sources }));
        else if (ev.type === "token") patch(answerId, (m) => ({ ...m, content: m.content + ev.text }));
        else if (ev.type === "done") patch(answerId, (m) => ({ ...m, grounding: ev.grounding, status: "done" }));
        else if (ev.type === "error") patch(answerId, (m) => ({ ...m, status: "error", error: ev.message }));
      }
    } catch (err) {
      const aborted = (err as Error).name === "AbortError";
      patch(answerId, (m) => ({
        ...m,
        status: aborted && m.content ? "done" : "error",
        error: aborted
          ? "Stopped."
          : err instanceof ApiError
            ? err.message
            : `Could not reach the API: ${(err as Error).message}`,
      }));
    } finally {
      abortRef.current = null;
    }
  }

  const show = (m: Message, n: number | null) =>
    m.sources &&
    openDrawer({ title: persona.name, sources: m.sources, cited: m.grounding?.cited ?? [], active: n });

  return (
    <div className="chat">
      <div className="chat-scroll">
        {messages.length === 0 && (
          <div className="empty">
            <div className="empty-avatar" aria-hidden>
              {persona.avatar}
            </div>
            <h2>{persona.name}</h2>
            <p className="muted">{persona.tagline}</p>
            {persona.chunks === 0 ? (
              <p className="warn">
                No sources ingested yet. Run <code>python -m personarag.ingest {persona.id}</code>.
              </p>
            ) : (
              <div className="starters">
                {(STARTERS[persona.id] ?? []).map((q) => (
                  <button key={q} className="starter" onClick={() => send(q)}>
                    {q}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}

        {messages.map((m) =>
          m.role === "user" ? (
            <div key={m.id} className="msg msg-user">
              <div className="bubble">{m.content}</div>
            </div>
          ) : (
            <div key={m.id} className="msg msg-assistant">
              <div className="avatar" aria-hidden>
                {persona.avatar}
              </div>
              <div className="msg-body">
                <div className="msg-name">{persona.name}</div>
                {m.content ? (
                  <Answer
                    text={m.content}
                    sourceCount={m.sources?.length ?? 0}
                    activeCitation={drawer && drawer.sources === m.sources ? drawer.active : null}
                    onCite={(n) => show(m, n)}
                  />
                ) : m.status === "streaming" ? (
                  <div className="thinking">
                    <span />
                    <span />
                    <span />
                    {m.sources ? `Read ${m.sources.length} passages, writing…` : "Searching the knowledge base…"}
                  </div>
                ) : null}
                {m.error && <div className={m.status === "error" ? "error" : "muted small"}>{m.error}</div>}
                {m.status !== "streaming" && m.sources && (
                  <div className="msg-foot">
                    {m.grounding && <GroundingBadge grounding={m.grounding} />}
                    <button className="link-btn" onClick={() => show(m, null)}>
                      {m.sources.length} sources
                    </button>
                  </div>
                )}
              </div>
            </div>
          ),
        )}
        <div ref={endRef} />
      </div>

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          send(input);
        }}
      >
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send(input);
            }
          }}
          placeholder={`Ask ${persona.name} anything…`}
          rows={1}
          aria-label="Your question"
        />
        {busy ? (
          <button type="button" className="send stop" onClick={() => abortRef.current?.abort()}>
            Stop
          </button>
        ) : (
          <button type="submit" className="send" disabled={!input.trim()}>
            Send
          </button>
        )}
      </form>
    </div>
  );
}
