"""Stage 3: split an answer into small, individually checkable claims."""
import re
import logging

logger = logging.getLogger(__name__)
_nlp = None

_BULLET = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+")


def get_nlp():
    """Load spaCy once. Returns None (and we fall back to regex) if the model is missing."""
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
        except Exception as exc:  # spaCy or the model not installed
            logger.warning("spaCy model unavailable (%s); using regex fallback. "
                           "Run: python -m spacy download en_core_web_sm", exc)
            _nlp = False
    return _nlp or None


def split_into_claims(text: str, min_words: int = 4) -> list:
    text = (text or "").strip()
    if not text:
        return []

    nlp = get_nlp()
    if nlp:
        sentences = [s.text for s in nlp(text).sents]
    else:
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)

    claims = []
    for sentence in sentences:
        for line in sentence.split("\n"):
            line = _BULLET.sub("", line).strip()
            if len(line.split()) >= min_words:
                claims.append(line)
    return claims
