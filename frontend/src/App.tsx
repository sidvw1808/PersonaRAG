import { useCallback, useEffect, useState } from "react";
import { api, type KnowledgeSource, type Persona, type ServerConfig } from "./api";
import { Chat } from "./components/Chat";
import { Panel } from "./components/Panel";
import { SourcesDrawer } from "./components/SourcesDrawer";
import type { DrawerState, Message } from "./types";

type Tab = "chat" | "panel";

function load<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function save(key: string, value: unknown) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* storage unavailable: preferences just won't persist */
  }
}

export default function App() {
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [modes, setModes] = useState<Record<string, string>>({ ask: "" });
  const [config, setConfig] = useState<ServerConfig | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [personaId, setPersonaId] = useState<string>(() => load("personaId", "elon_musk"));
  const [mode, setMode] = useState<string>(() => load("mode", "ask"));
  const [tab, setTab] = useState<Tab>("chat");
  const [chats, setChats] = useState<Record<string, Message[]>>({});
  const [drawer, setDrawer] = useState<DrawerState | null>(null);
  const [kb, setKb] = useState<KnowledgeSource[] | null>(null);
  const [kbOpen, setKbOpen] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    Promise.all([api.personas(), api.modes(), api.config()])
      .then(([p, m, c]) => {
        setPersonas(p);
        setModes(m);
        setConfig(c);
        setPersonaId((id) => (p.some((x) => x.id === id) ? id : p[0]?.id));
      })
      .catch((e: Error) => setLoadError(e.message));
  }, []);

  useEffect(() => save("personaId", personaId), [personaId]);
  useEffect(() => save("mode", mode), [mode]);

  useEffect(() => {
    if (!kbOpen || !personaId) return;
    setKb(null);
    api.sources(personaId).then(setKb).catch(() => setKb([]));
  }, [kbOpen, personaId]);

  const persona = personas.find((p) => p.id === personaId);
  const setMessages = useCallback(
    (update: (prev: Message[]) => Message[]) =>
      setChats((all) => ({ ...all, [personaId]: update(all[personaId] ?? []) })),
    [personaId],
  );
  const openDrawer = useCallback((state: DrawerState) => setDrawer(state), []);
  const closeDrawer = useCallback(() => setDrawer(null), []);

  if (loadError) {
    return (
      <div className="boot">
        <h1>PersonaRAG</h1>
        <p className="error">Can't reach the API ({loadError}).</p>
        <p className="muted">
          Start it from the project folder: <code>.venv\Scripts\uvicorn api:app --port 8000</code>
        </p>
      </div>
    );
  }
  if (!persona) return <div className="boot muted">Loading personas…</div>;

  return (
    <div className={`app${drawer ? " with-drawer" : ""}`}>
      <aside className={`sidebar${menuOpen ? " open" : ""}`}>
        <div className="brand">
          <span className="brand-mark" aria-hidden>
            ◎
          </span>
          PersonaRAG
        </div>
        <p className="muted small">Evidence-grounded conversations with AI simulations of public figures.</p>

        <div className="eyebrow">Persona</div>
        <nav className="persona-list">
          {personas.map((p) => (
            <button
              key={p.id}
              className={`persona${p.id === personaId ? " persona-on" : ""}`}
              onClick={() => {
                setPersonaId(p.id);
                setTab("chat");
                setDrawer(null);
                setMenuOpen(false);
              }}
            >
              <span className="persona-avatar" aria-hidden>
                {p.avatar}
              </span>
              <span className="persona-text">
                <span className="persona-name">{p.name}</span>
                <span className="persona-sub">{p.chunks ? `${p.chunks} passages` : "no sources yet"}</span>
              </span>
            </button>
          ))}
        </nav>

        <div className="eyebrow">Mode</div>
        <div className="modes" role="radiogroup" aria-label="Mode">
          {Object.keys(modes).map((m) => (
            <button
              key={m}
              role="radio"
              aria-checked={mode === m}
              className={`mode${mode === m ? " mode-on" : ""}`}
              onClick={() => setMode(m)}
              title={modes[m]}
            >
              {m}
            </button>
          ))}
        </div>

        <button className="kb-toggle" onClick={() => setKbOpen((o) => !o)} aria-expanded={kbOpen}>
          {kbOpen ? "▾" : "▸"} Knowledge base
        </button>
        {kbOpen && (
          <ul className="kb">
            {kb === null && <li className="muted small">Loading…</li>}
            {kb?.map((s) => (
              <li key={s.source_id}>
                {s.url ? (
                  <a href={s.url} target="_blank" rel="noreferrer">
                    {s.title}
                  </a>
                ) : (
                  s.title
                )}
                <span className="muted small">
                  {" "}
                  {[s.kind, s.date].filter(Boolean).join(" · ")} · {s.chunks}
                </span>
              </li>
            ))}
          </ul>
        )}

        <div className="sidebar-foot">
          <p className="disclaimer">
            AI simulation built from public material. Not the real person, and not their private views.
          </p>
          {config && (
            <p className="muted tiny">
              {config.llm}
              <br />
              {config.retrieval_mode} · {config.embedding_model}
            </p>
          )}
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <button className="icon-btn menu-btn" onClick={() => setMenuOpen((o) => !o)} aria-label="Menu">
            ☰
          </button>
          <div className="tabs" role="tablist">
            <button role="tab" aria-selected={tab === "chat"} className={tab === "chat" ? "tab tab-on" : "tab"} onClick={() => setTab("chat")}>
              Chat · {persona.name}
            </button>
            <button role="tab" aria-selected={tab === "panel"} className={tab === "panel" ? "tab tab-on" : "tab"} onClick={() => setTab("panel")}>
              Panel
            </button>
          </div>
          {tab === "chat" && (chats[personaId]?.length ?? 0) > 0 && (
            <button className="link-btn" onClick={() => setChats((c) => ({ ...c, [personaId]: [] }))}>
              New chat
            </button>
          )}
        </header>

        {tab === "chat" ? (
          <Chat
            key={personaId}
            persona={persona}
            mode={mode}
            messages={chats[personaId] ?? []}
            setMessages={setMessages}
            drawer={drawer}
            openDrawer={openDrawer}
          />
        ) : (
          <Panel personas={personas} mode={mode} openDrawer={openDrawer} />
        )}
      </main>

      {drawer && (
        <SourcesDrawer
          {...drawer}
          onSelect={(n) => setDrawer((d) => (d ? { ...d, active: n } : d))}
          onClose={closeDrawer}
        />
      )}
      {menuOpen && <div className="scrim" onClick={() => setMenuOpen(false)} />}
    </div>
  );
}
