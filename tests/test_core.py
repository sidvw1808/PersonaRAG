from personarag.chunking import chunk_text, split_sentences
from personarag.engine import grounding_report
from personarag.loaders import _split_front_matter, _strip_subtitles
from personarag.store import Hit


def _hit(score=0.7):
    return Hit(id="x", text="t", metadata={"title": "T"}, score=score)


def test_chunks_respect_size_and_overlap():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    chunks = chunk_text(text, max_words=50, overlap_words=10)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 60 for c in chunks)
    # The last sentence of one chunk reappears at the start of the next.
    first_tail = chunks[0].split(". ")[-1]
    assert first_tail.rstrip(".") in chunks[1]


def test_unpunctuated_transcript_still_splits():
    text = "word " * 1000
    assert len(split_sentences(text)) > 1


def test_front_matter_parsed():
    meta, body = _split_front_matter("---\ntitle: Talk\ndate: 2020\n---\nHello there.")
    assert meta == {"title": "Talk", "date": 2020}
    assert body.strip() == "Hello there."


def test_subtitles_stripped():
    vtt = "WEBVTT\n\n1\n00:00:01.000 --> 00:00:02.000\nHello\n\n2\n00:00:02.000 --> 00:00:03.000\nHello\nworld"
    assert _strip_subtitles(vtt) == "Hello world"


def test_grounding_levels():
    hits = [_hit(0.8), _hit(0.7)]
    strong = grounding_report("We should go to Mars [1]. Rockets must be reusable [2].", hits)
    assert strong["level"] == "strong" and strong["cited"] == [1, 2]

    none = grounding_report("I think this is a great idea for everyone involved.", hits)
    assert none["level"] == "ungrounded"

    bad = grounding_report("A claim with a bogus citation here [7].", hits)
    assert bad["invalid_citations"] == [7] and bad["level"] == "ungrounded"
