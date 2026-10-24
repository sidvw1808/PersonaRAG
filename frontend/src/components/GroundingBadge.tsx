import type { Grounding } from "../api";

const LABELS: Record<Grounding["level"], string> = {
  strong: "Strongly grounded",
  partial: "Partially grounded",
  weak: "Weakly grounded",
  ungrounded: "Not grounded in sources",
};

export function GroundingBadge({ grounding }: { grounding: Grounding }) {
  const tip =
    `${Math.round(grounding.citation_coverage * 100)}% of sentences cite a source · ` +
    `top retrieval similarity ${grounding.top_similarity.toFixed(2)}` +
    (grounding.invalid_citations.length ? ` · ${grounding.invalid_citations.length} invalid citation(s)` : "");
  return (
    <span className={`badge badge-${grounding.level}`} title={tip}>
      <span className="badge-dot" aria-hidden />
      {LABELS[grounding.level]}
      <span className="badge-meta">{Math.round(grounding.citation_coverage * 100)}% cited</span>
    </span>
  );
}
