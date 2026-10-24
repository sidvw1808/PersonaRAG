import { useEffect, useRef } from "react";
import type { Source } from "../api";

interface Props {
  title: string;
  sources: Source[];
  cited: number[];
  active: number | null;
  onSelect: (n: number) => void;
  onClose: () => void;
}

export function SourcesDrawer({ title, sources, cited, active, onSelect, onClose }: Props) {
  const refs = useRef<Record<number, HTMLElement | null>>({});

  useEffect(() => {
    if (active != null) refs.current[active]?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [active]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <aside className="drawer" aria-label="Sources">
      <header className="drawer-head">
        <div>
          <div className="eyebrow">Sources</div>
          <h3>{title}</h3>
        </div>
        <button className="icon-btn" onClick={onClose} aria-label="Close sources">
          ✕
        </button>
      </header>
      <p className="drawer-note">
        {cited.length} of {sources.length} retrieved passages were cited. Passages are what the model was given to
        answer from.
      </p>
      <div className="drawer-list">
        {sources.map((s) => (
          <article
            key={s.n}
            ref={(el) => {
              refs.current[s.n] = el;
            }}
            className={`source${active === s.n ? " source-active" : ""}${cited.includes(s.n) ? " source-cited" : ""}`}
            onClick={() => onSelect(s.n)}
          >
            <div className="source-head">
              <span className="source-n">{s.n}</span>
              <div className="source-title">
                {s.url ? (
                  <a href={s.url} target="_blank" rel="noreferrer" onClick={(e) => e.stopPropagation()}>
                    {s.title}
                  </a>
                ) : (
                  s.title
                )}
                <div className="source-meta">
                  {[s.kind, s.date].filter(Boolean).join(" · ")}
                  {cited.includes(s.n) ? <span className="tag">cited</span> : <span className="tag tag-muted">not cited</span>}
                </div>
              </div>
            </div>
            <p className="source-text">{s.text}</p>
            <div className="source-scores">
              similarity {s.score.toFixed(2)}
              {s.rerank != null && <> · rerank {s.rerank > 0 ? "+" : ""}{s.rerank.toFixed(1)}</>}
            </div>
          </article>
        ))}
        {sources.length === 0 && <p className="muted">No passages passed the relevance threshold.</p>}
      </div>
    </aside>
  );
}
