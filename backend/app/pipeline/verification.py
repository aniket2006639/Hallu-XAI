"""Stage 4: score one claim against its evidence using several independent signals."""
import re
import logging
import numpy as np

from app import config
from app.pipeline.fact_atomization import get_nlp

logger = logging.getLogger(__name__)

STOPWORDS = set("""a an the and or but if of to in on at by for with from as is are was were be been being it its
this that these those he she they them his her their we you i not no do does did has have had will would can could
should may might also than then so such into over under about after before during between which who whom whose what
when where why how""".split())

_NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
_ENTITY_LABELS = {"PERSON", "ORG", "GPE", "LOC", "NORP", "FAC", "EVENT", "WORK_OF_ART",
                  "PRODUCT", "LAW", "LANGUAGE"}


def _tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def _numbers(text):
    return {n.replace(",", "") for n in _NUMBER.findall(text)}


def extract_entities(text):
    nlp = get_nlp()
    if nlp:
        return list(dict.fromkeys(e.text for e in nlp(text).ents if e.label_ in _ENTITY_LABELS))
    found = re.findall(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*", text)
    return list(dict.fromkeys(f for f in found if f.lower() not in STOPWORDS))


# ---------- the simple, cheap signals (no models needed) ----------
def lexical_overlap(claim, evidence):
    words = [t for t in _tokens(claim) if t not in STOPWORDS]
    if not words:
        return 1.0
    ev = set(_tokens(evidence))
    return sum(w in ev for w in words) / len(words)


def number_consistency(claim, evidence):
    nums = _numbers(claim)
    if not nums:
        return 1.0
    ev = _numbers(evidence)
    return sum(n in ev for n in nums) / len(nums)


def entity_coverage(entities, evidence):
    if not entities:
        return 1.0
    ev = evidence.lower()
    return sum(e.lower() in ev for e in entities) / len(entities)


# ---------- the model-based signals ----------
class Verifier:
    def __init__(self):
        self._tok = None
        self._model = None
        self._idx = None

    @property
    def is_loaded(self):
        return self._model is not None and self._tok is not None and self._idx is not None

    @staticmethod
    def _label_index(labels, accepted):
        for index, label in labels.items():
            normalized = re.sub(r"[^a-z]", "", str(label).lower())
            if normalized in accepted:
                return index
        return None

    def load(self):
        if self._model is not None:
            return
        import torch  # noqa: F401
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        logger.info("Loading NLI model %s ...", config.NLI_MODEL)
        self._tok = AutoTokenizer.from_pretrained(config.NLI_MODEL)
        self._model = AutoModelForSequenceClassification.from_pretrained(config.NLI_MODEL).eval()
        labels = {int(i): name for i, name in self._model.config.id2label.items()}
        entailment = self._label_index(labels, {"entailment", "entail"})
        contradiction = self._label_index(labels, {"contradiction", "contradict"})
        if entailment is None or contradiction is None:
            raise ValueError(f"NLI model labels must include entailment and contradiction; got {labels}")
        self._idx = {"entailment": entailment, "contradiction": contradiction}

    def nli_probs(self, premises, hypotheses):
        """Returns an array (n, 2): columns = [P(entailment), P(contradiction)]."""
        import torch
        self.load()
        batch = self._tok(list(premises), list(hypotheses), return_tensors="pt",
                          padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            probs = torch.softmax(self._model(**batch).logits, dim=-1).numpy()
        return probs[:, [self._idx["entailment"], self._idx["contradiction"]]]

    def entails(self, a, b):
        """True if text a logically implies text b (used by semantic entropy)."""
        p = self.nli_probs([a], [b])[0]
        return bool(p[0] > p[1] and p[0] > 0.5)

    def score_claim(self, claim, evidence_texts, entities=None):
        """Returns the feature dict (keys must match features.FEATURE_NAMES)."""
        if not evidence_texts:
            return {n: 0.0 for n in ("nli_entailment", "nli_contradiction", "semantic_similarity",
                                     "lexical_overlap", "entity_coverage", "number_consistency")}
        from app.pipeline.retrieval import embed

        entities = extract_entities(claim) if entities is None else entities
        joined = " ".join(evidence_texts)

        nli = self.nli_probs(evidence_texts, [claim] * len(evidence_texts))
        vecs = embed([claim] + list(evidence_texts))
        similarity = float(np.max(vecs[1:] @ vecs[0]))

        return {
            "nli_entailment": float(nli[:, 0].max()),
            "nli_contradiction": float(nli[:, 1].max()),
            "semantic_similarity": max(0.0, similarity),
            "lexical_overlap": lexical_overlap(claim, joined),
            "entity_coverage": entity_coverage(entities, joined),
            "number_consistency": number_consistency(claim, joined),
        }
