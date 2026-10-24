import { Fragment, type ReactNode } from "react";

interface Props {
  text: string;
  sourceCount: number;
  activeCitation?: number | null;
  onCite?: (n: number) => void;
  /** false renders [n] as plain text (e.g. the moderator quoting panelists' citations). */
  citations?: boolean;
}

// Splits on citations like [3], bold **x** and italic *x*; everything else is plain text.
const INLINE = /(\[\d+\]|\*\*[^*]+\*\*|\*[^*\s][^*]*\*)/g;

function renderInline(text: string, props: Props): ReactNode[] {
  return text.split(INLINE).map((part, i) => {
    const cite = part.match(/^\[(\d+)\]$/);
    if (cite && props.citations !== false) {
      const n = Number(cite[1]);
      const valid = n >= 1 && n <= props.sourceCount;
      return (
        <button
          key={i}
          type="button"
          className={`cite${valid ? "" : " cite-invalid"}${props.activeCitation === n ? " cite-active" : ""}`}
          onClick={() => valid && props.onCite?.(n)}
          title={valid ? `Show source ${n}` : `Source ${n} was not provided to the model`}
          aria-label={valid ? `Source ${n}` : `Invalid citation ${n}`}
        >
          {n}
        </button>
      );
    }
    if (part.startsWith("**") && part.endsWith("**")) return <strong key={i}>{part.slice(2, -2)}</strong>;
    if (part.startsWith("*") && part.endsWith("*") && part.length > 2) return <em key={i}>{part.slice(1, -1)}</em>;
    return <Fragment key={i}>{part}</Fragment>;
  });
}

/** Minimal, safe markdown: paragraphs, bullet lists, headings, bold/italic, clickable citations. */
export function Answer(props: Props) {
  const blocks = props.text.trim().split(/\n{2,}/);
  return (
    <div className="answer">
      {blocks.map((block, b) => {
        const lines = block.split("\n");
        if (lines.every((l) => /^\s*([-*•]|\d+\.)\s+/.test(l))) {
          return (
            <ul key={b}>
              {lines.map((l, i) => (
                <li key={i}>{renderInline(l.replace(/^\s*([-*•]|\d+\.)\s+/, ""), props)}</li>
              ))}
            </ul>
          );
        }
        const heading = block.match(/^#{1,4}\s+(.*)$/);
        if (heading) return <h4 key={b}>{renderInline(heading[1], props)}</h4>;
        return (
          <p key={b}>
            {lines.map((l, i) => (
              <Fragment key={i}>
                {i > 0 && <br />}
                {renderInline(l, props)}
              </Fragment>
            ))}
          </p>
        );
      })}
    </div>
  );
}
