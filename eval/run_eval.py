"""Evaluate retrieval quality and (optionally) answer grounding for a persona.

    python -m eval.run_eval elon_musk                 # retrieval only: free, no LLM calls
    python -m eval.run_eval elon_musk --generate      # also generate answers and score grounding
    python -m eval.run_eval elon_musk --generate --judge   # plus LLM-as-judge faithfulness

Question file: eval/<persona>.yaml (see eval/elon_musk.yaml).
"""

import argparse
import json
import re
import time
from pathlib import Path

import yaml

from personarag.engine import PersonaEngine
from personarag.llm import complete

EVAL_DIR = Path(__file__).parent
ABSTAIN = re.compile(
    r"(doesn'?t cover|don'?t cover|not (?:in|covered by) (?:the|my) (?:sources|material)|"
    # "I haven't spoken about this." counts; "I haven't spoken about this directly, but..." is
    # an extrapolation, not an abstention.
    r"haven'?t (?:\w+ )?(?:spoken|talked|said|weighed in|commented)[^,.]*\.(?!\s*(?:but|however))|"
    r"no (?:relevant )?(?:sources|information)|none of the (?:[\w-]+ ){0,3}(?:sources|material)|"
    r"(?:sources|material) (?:don'?t|do not) (?:say|cover|address)|can'?t speak to)",
    re.I,
)

JUDGE_SYSTEM = """You grade whether an answer is faithful to its sources.
Reply with JSON only: {"faithful": <0.0-1.0>, "unsupported_claims": ["..."]}.
1.0 = every factual claim is supported by the numbered sources or is explicitly flagged as extrapolation.
0.0 = the main claims are fabricated or contradict the sources."""


def retrieval_metrics(hits, item) -> dict:
    expected = set(item.get("expect_sources", []))
    ranks = [i for i, h in enumerate(hits, 1) if h.metadata.get("source_id") in expected]
    joined = " ".join(h.text.lower() for h in hits)
    kws = item.get("keywords", [])
    return {
        "hit": bool(ranks) if expected else None,
        "rr": (1 / ranks[0]) if ranks else 0.0,
        "keyword_recall": (sum(k.lower() in joined for k in kws) / len(kws)) if kws else None,
    }


def judge(engine, answer) -> float | None:
    sources = "\n\n".join(f"[{i}] {h.text}" for i, h in enumerate(answer.hits, 1))
    raw = complete(
        engine.llm,
        JUDGE_SYSTEM,
        [{"role": "user", "content": f"SOURCES:\n{sources}\n\nANSWER:\n{answer.text}"}],
        max_tokens=800,
    )
    match = re.search(r"\{.*\}", raw, re.S)
    try:
        return float(json.loads(match.group(0))["faithful"]) if match else None
    except (ValueError, KeyError, json.JSONDecodeError):
        return None


def mean(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("persona")
    parser.add_argument("--generate", action="store_true", help="generate answers (uses the LLM)")
    parser.add_argument("--judge", action="store_true", help="LLM-as-judge faithfulness (implies --generate)")
    args = parser.parse_args()
    args.generate = args.generate or args.judge

    items = yaml.safe_load((EVAL_DIR / f"{args.persona}.yaml").read_text(encoding="utf-8"))["questions"]
    engine = PersonaEngine()
    rows = []
    for item in items:
        q = item["question"]
        answerable = item.get("answerable", True)
        hits = engine.retrieve(args.persona, q)
        row = {"question": q, "answerable": answerable, **retrieval_metrics(hits, item)}
        if args.generate:
            ans = engine.ask(args.persona, q)
            g = ans.grounding
            row.update(
                answer=ans.text,
                grounding=g["level"],
                citation_coverage=g["citation_coverage"] if answerable else None,
                invalid_citations=len(g["invalid_citations"]),
                abstained=bool(ABSTAIN.search(ans.text)),
            )
            if args.judge and answerable:
                row["faithfulness"] = judge(engine, ans)
        rows.append(row)
        status = "PASS" if row["hit"] else ("  - " if row["hit"] is None else "MISS")
        print(f"{status} {q[:70]}")

    answerable_rows = [r for r in rows if r["answerable"]]
    summary = {
        "persona": args.persona,
        "questions": len(rows),
        "hit_rate@k": mean([float(r["hit"]) for r in answerable_rows if r["hit"] is not None]),
        "mrr": mean([r["rr"] for r in answerable_rows if r["hit"] is not None]),
        "keyword_recall": mean([r["keyword_recall"] for r in answerable_rows]),
    }
    if args.generate:
        unanswerable = [r for r in rows if not r["answerable"]]
        summary.update(
            citation_coverage=mean([r["citation_coverage"] for r in answerable_rows]),
            answers_with_citations=mean([float(r["grounding"] != "ungrounded") for r in answerable_rows]),
            invalid_citations=sum(r["invalid_citations"] for r in rows),
            abstention_accuracy=mean([float(r["abstained"]) for r in unanswerable]),
            false_abstentions=sum(r["abstained"] for r in answerable_rows),
        )
        if args.judge:
            summary["faithfulness"] = mean([r.get("faithfulness") for r in answerable_rows])

    out_dir = EVAL_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{args.persona}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n| metric | value |\n|---|---|")
    for k, v in summary.items():
        print(f"| {k} | {v} |")
    print(f"\nFull results: {out}")


if __name__ == "__main__":
    main()
