"""Wires every stage together. This is the only place that knows the order of the pipeline."""
import time
import logging

from app import config
from app.pipeline.fact_atomization import split_into_claims
from app.pipeline.retrieval import Retriever
from app.pipeline.verification import Verifier, extract_entities
from app.pipeline.classifier import HallucinationClassifier
from app.pipeline.explainability import Explainer
from app.pipeline.features import FEATURE_NAMES, to_vector

logger = logging.getLogger(__name__)


def verdict_for(score):
    if score >= config.HIGH_RISK:
        return "likely_hallucinated"
    if score < config.LOW_RISK:
        return "supported"
    return "uncertain"


class HalluPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.verifier = Verifier()
        self.classifier = HallucinationClassifier()
        self.explainer = Explainer(self.classifier)

    def warmup(self):
        """Load the heavy models now (at server start) instead of on the first request."""
        self.retriever.warmup()
        self.verifier.load()
        from app.pipeline.retrieval import get_embedder
        get_embedder()

    def analyze(self, question, answer, use_wikipedia=None):
        started = time.time()
        stage_ms = {
            "claim_split": 0,
            "retrieval": 0,
            "feature_computation": 0,
            "classification": 0,
            "explanation": 0,
        }
        split_started = time.perf_counter()
        claims = split_into_claims(answer)[: config.MAX_CLAIMS]
        stage_ms["claim_split"] = int((time.perf_counter() - split_started) * 1000)
        results = []

        for claim in claims:
            entities = extract_entities(claim)
            stage_started = time.perf_counter()
            evidence = self.retriever.retrieve(claim, question, entities, use_wikipedia)
            stage_ms["retrieval"] += int((time.perf_counter() - stage_started) * 1000)

            stage_started = time.perf_counter()
            features = self.verifier.score_claim(claim, [e["text"] for e in evidence], entities)
            stage_ms["feature_computation"] += int((time.perf_counter() - stage_started) * 1000)

            vector = to_vector(features)
            stage_started = time.perf_counter()
            score = float(self.classifier.predict([vector])[0])
            stage_ms["classification"] += int((time.perf_counter() - stage_started) * 1000)

            stage_started = time.perf_counter()
            explanation = self.explainer.explain(vector, score)
            stage_ms["explanation"] += int((time.perf_counter() - stage_started) * 1000)
            results.append({
                "claim": claim,
                "score": round(score, 4),
                "verdict": verdict_for(score),
                "features": {k: round(v, 4) for k, v in features.items()},
                "explanation": explanation,
                "evidence": [{"text": e["text"][:400], "source": e["source"]} for e in evidence[:3]],
            })

        flagged = [r for r in results if r["verdict"] == "likely_hallucinated"]
        uncertain = [r for r in results if r["verdict"] == "uncertain"]
        warnings = []
        if self.classifier.is_placeholder:
            warnings.append("Classifier is a synthetic placeholder: scores are illustrative until it is "
                            "trained on real labelled data.")
        if not claims:
            warnings.append("No checkable claims were found in the text.")

        return {
            "overall": {
                "num_claims": len(results),
                "num_flagged": len(flagged),
                "num_uncertain": len(uncertain),
                "max_score": max((r["score"] for r in results), default=0.0),
                "summary": (f"{len(flagged)} of {len(results)} claims flagged as likely hallucinated, "
                            f"{len(uncertain)} uncertain."),
            },
            "claims": results,
            "meta": {
                "classifier": self.classifier.meta,
                "feature_names": FEATURE_NAMES,
                "latency_ms": int((time.time() - started) * 1000),
                "latency_by_stage_ms": stage_ms,
                "warnings": warnings,
            },
        }
