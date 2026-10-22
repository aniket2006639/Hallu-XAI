"""Human-readable keyword cues derived from the existing verification signals."""
from dataclasses import dataclass, asdict
import re

from app.pipeline.verification import STOPWORDS, _NUMBER, _tokens, extract_entities


@dataclass(frozen=True)
class TokenCue:
    token: str
    token_type: str
    found: bool
    note: str
    start: int
    end: int

    def as_dict(self):
        value = asdict(self)
        value["status"] = "found" if self.found else "MISSING"
        del value["start"]
        del value["end"]
        return value


def _number_texts(text):
    return [match.group(0) for match in _NUMBER.finditer(text)]


def _find_span(text, value, start=0):
    match = re.search(re.escape(value), text[start:], re.IGNORECASE)
    if not match:
        return start, start
    return start + match.start(), start + match.end()


def _content_candidates(claim):
    return [
        (token, "word", start, end)
        for match in re.finditer(r"[A-Za-z][A-Za-z'-]*", claim)
        for token, start, end in [(match.group(0), match.start(), match.end())]
        if token.lower() not in STOPWORDS and len(token) > 1
    ]


def unsupported_token_cues(claim, evidence_texts, entities=None, max_tokens=25):
    """Return cheap FOUND/MISSING cues for entities, numbers, and content words.

    This deliberately does not claim that a missing token proves a hallucination:
    retrieval can miss valid evidence. The returned spans are used only for display.
    """
    joined = " ".join(evidence_texts)
    lower_evidence = joined.lower()
    entities = extract_entities(claim) if entities is None else entities
    candidates = []

    for entity in entities:
        start, end = _find_span(claim, entity)
        found = entity.lower() in lower_evidence
        candidates.append(TokenCue(entity, "entity", found,
                                   "" if found else "entity is missing from evidence",
                                   start, end))

    for number in _NUMBER.finditer(claim):
        value = number.group(0)
        normalized = value.replace(",", "")
        found = normalized in {n.replace(",", "") for n in _number_texts(joined)}
        note = ""
        if not found:
            evidence_numbers = _number_texts(joined)
            if evidence_numbers:
                note = f"evidence says {evidence_numbers[0]}"
        candidates.append(TokenCue(value, "number", found, note, number.start(), number.end()))

    entity_spans = {(cue.start, cue.end) for cue in candidates}
    for token, token_type, start, end in _content_candidates(claim):
        if any(start >= left and end <= right for left, right in entity_spans):
            continue
        found = token.lower() in set(_tokens(joined))
        candidates.append(TokenCue(token, token_type, found,
                                   "" if found else "content word is missing from evidence",
                                   start, end))

    # Keep all entities/numbers, then prefer missing and distinctive words.
    candidates.sort(key=lambda cue: (
        cue.found,
        cue.token_type == "word",
        len(cue.token),
    ))
    return candidates[:max_tokens]


def highlight_claim(claim, cues, color=True):
    """Return the claim with missing cues red and found cues green."""
    if not color:
        return claim
    spans = sorted((cue.start, cue.end, cue.found) for cue in cues if cue.end > cue.start)
    parts, cursor = [], 0
    for start, end, found in spans:
        if start < cursor:
            continue
        parts.append(claim[cursor:start])
        colour = "\033[32m" if found else "\033[31m"
        parts.append(f"{colour}{claim[start:end]}\033[0m")
        cursor = end
    parts.append(claim[cursor:])
    return "".join(parts)
