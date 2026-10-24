import { useRef, useState } from "react";
import { ApiError, streamPanel, type PanelAnswer, type Persona } from "../api";
import type { DrawerState } from "../types";
import { Answer } from "./Answer";
import { GroundingBadge } from "./GroundingBadge";

interface Props {
  personas: Persona[];
  mode: string;
  openDrawer: (state: DrawerState) => void;
}

export function Panel({ personas, mode, openDrawer }: Props) {
  const ready = personas.filter((p) => p.chunks > 0);
  const [chosen, setChosen] = useState<string[]>(ready.slice(0, 3).map((p) => p.id));
  const [question, setQuestion] = useState("");
  const [asked, setAsked] = useState<{ question: string; personas: string[] } | null>(null);
  const [answers, setAnswers] = useState<Record<string, PanelAnswer>>({});
  const [summary, setSummary] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const byId = Object.fromEntries(personas.map((p) => [p.id, p]));

  const toggle = (id: string) =>
    setChosen((c) => (c.includes(id) ? c.filter((x) => x !== id) : c.length < 5 ? [...c, id] : c));

  async function run() {
    const q = question.trim();
    if (!q || chosen.length < 2 || running) return;
    setAsked({ question: q, personas: chosen });
    setAnswers({});
    setSummary(null);
    setError(null);
    setRunning(true);
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      for await (const ev of streamPanel({ personas: chosen, question: q, mode }, controller.signal)) {
        if (ev.type === "answer") setAnswers((a) => ({ ...a, [ev.persona]: ev }));
        else if (ev.type === "summary") setSummary(ev.text);
        else if (ev.type === "error") setError(ev.message);
      }
    } catch (err) {
      if ((err as Error).name !== "AbortError")
        setError(err instanceof ApiError ? err.message : `Could not reach the API: ${(err as Error).message}`);
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="panel-view">
      <section className="panel-setup">
        <div className="eyebrow">Panelists (2 to 5)</div>
        <div className="chips">
          {personas.map((p) => (
            <button
              key={p.id}
              className={`chip${chosen.includes(p.id) ? " chip-on" : ""}`}
              onClick={() => toggle(p.id)}
              disabled={p.chunks === 0}
              aria-pressed={chosen.includes(p.id)}
              title={p.chunks === 0 ? "No sources ingested" : p.tagline}
            >
              <span aria-hidden>{p.avatar}</span> {p.name}
            </button>
          ))}
        </div>
        <form
          className="panel-ask"
          onSubmit={(e) => {
            e.preventDefault();
            run();
          }}
        >
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Should I drop out of college to start a company?"
            aria-label="Question for the panel"
          />
          {running ? (
            <button type="button" className="send stop" onClick={() => abortRef.current?.abort()}>
              Stop
            </button>
          ) : (
            <button type="submit" className="send" disabled={!question.trim() || chosen.length < 2}>
              Ask the panel
            </button>
          )}
        </form>
      </section>

      {asked && (
        <>
          <h3 className="panel-question">“{asked.question}”</h3>
          <div className="panel-grid">
            {asked.personas.map((pid) => {
              const p = byId[pid];
              const a = answers[pid];
              return (
                <article key={pid} className="panel-card">
                  <header>
                    <span className="avatar" aria-hidden>
                      {p.avatar}
                    </span>
                    <div>
                      <div className="msg-name">{p.name}</div>
                      <div className="muted small">{p.tagline}</div>
                    </div>
                  </header>
                  {a ? (
                    <>
                      <Answer
                        text={a.answer}
                        sourceCount={a.sources.length}
                        onCite={(n) =>
                          openDrawer({ title: p.name, sources: a.sources, cited: a.grounding.cited, active: n })
                        }
                      />
                      <div className="msg-foot">
                        <GroundingBadge grounding={a.grounding} />
                        <button
                          className="link-btn"
                          onClick={() =>
                            openDrawer({ title: p.name, sources: a.sources, cited: a.grounding.cited, active: null })
                          }
                        >
                          {a.sources.length} sources
                        </button>
                      </div>
                    </>
                  ) : (
                    <div className="skeleton" aria-label="Waiting for answer">
                      <span />
                      <span />
                      <span />
                    </div>
                  )}
                </article>
              );
            })}
          </div>
          <section className="moderator">
            <div className="eyebrow">Moderator</div>
            {summary ? (
              <Answer text={summary} sourceCount={0} citations={false} />
            ) : running ? (
              <p className="muted">
                {Object.keys(answers).length < asked.personas.length
                  ? `Waiting for ${asked.personas.length - Object.keys(answers).length} panelist(s)…`
                  : "Comparing the answers…"}
              </p>
            ) : null}
          </section>
        </>
      )}
      {error && <div className="error">{error}</div>}
    </div>
  );
}
