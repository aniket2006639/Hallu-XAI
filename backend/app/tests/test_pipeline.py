"""Run from the backend folder:  python -m pytest app/tests -v
These tests need no model downloads - the heavy models are replaced by stubs."""
import math
import numpy as np
import pytest

from app.pipeline.fact_atomization import split_into_claims
from app.pipeline import verification as ver
from app.pipeline.features import FEATURE_NAMES, to_vector
from app.pipeline.semantic_entropy import semantic_entropy, cluster_by_meaning
from app.pipeline.classifier import synthetic_dataset, train
from app.pipeline.explainability import Explainer


# ---------- claim splitting ----------
def test_split_into_claims_basic():
    text = "The Eiffel Tower is in Paris. It was completed in 1889.\n- Short.\n- It is about 330 metres tall."
    claims = split_into_claims(text)
    assert "The Eiffel Tower is in Paris." in claims
    assert any("1889" in c for c in claims)
    assert any(c.startswith("It is about 330") for c in claims)   # bullet marker stripped
    assert "Short." not in claims                                  # too short to be a claim


def test_split_keeps_leading_year():
    assert split_into_claims("1998 was a very important year for the company.") == \
        ["1998 was a very important year for the company."]


def test_split_empty():
    assert split_into_claims("   ") == []


# ---------- cheap signals ----------
def test_number_consistency():
    ev = "The tower was completed in 1889 and is 330 metres tall. Cost: 7,799,401 francs."
    assert ver.number_consistency("It was completed in 1889.", ev) == 1.0
    assert ver.number_consistency("It was completed in 1998.", ev) == 0.0
    assert ver.number_consistency("It cost 7,799,401 francs.", ev) == 1.0
    assert ver.number_consistency("It is very tall.", ev) == 1.0     # no numbers -> neutral


def test_nli_label_index_accepts_standard_labels():
    assert ver.Verifier._label_index({0: "CONTRADICTION", 1: "ENTAILMENT"}, {"entailment"}) == 1
    assert ver.Verifier._label_index({0: "contradict", 1: "entail"}, {"contradict"}) == 0
    assert ver.Verifier._label_index({0: "neutral"}, {"entailment"}) is None


def test_lexical_overlap_and_entities():
    ev = "Guido van Rossum created Python in 1991."
    assert ver.lexical_overlap("Guido van Rossum created Python.", ev) == 1.0
    assert ver.lexical_overlap("Elon Musk invented Java.", ev) < 0.3
    assert ver.entity_coverage(["Guido van Rossum"], ev) == 1.0
    assert ver.entity_coverage(["Elon Musk"], ev) == 0.0
    assert ver.entity_coverage([], ev) == 1.0


# ---------- semantic entropy ----------
def test_semantic_entropy_agree_vs_scatter():
    same = lambda a, b: a.split()[-1] == b.split()[-1]         # toy "entails": same last word
    assert semantic_entropy(["it is Paris", "Paris", "so Paris"], same) == 0.0
    scattered = ["Paris", "Rome", "Berlin", "Madrid"]
    assert math.isclose(semantic_entropy(scattered, same), math.log(4))
    assert len(cluster_by_meaning(["a Paris", "b Paris", "c Rome"], same)) == 2


# ---------- classifier + SHAP ----------
@pytest.fixture(scope="module")
def trained():
    X, y = synthetic_dataset(n=3000)
    model, meta = train(X, y, "test")

    class Holder:                       # stands in for HallucinationClassifier
        pass
    h = Holder()
    h.model, h.meta = model, meta
    return h


SUPPORTED = dict(nli_entailment=0.95, nli_contradiction=0.01, semantic_similarity=0.85,
                 lexical_overlap=0.9, entity_coverage=1.0, number_consistency=1.0)
CONTRADICTED = dict(nli_entailment=0.02, nli_contradiction=0.95, semantic_similarity=0.7,
                    lexical_overlap=0.6, entity_coverage=1.0, number_consistency=0.0)


def test_classifier_sanity(trained):
    assert trained.meta["holdout_auroc"] > 0.9
    p = trained.model.predict_proba(np.array([to_vector(SUPPORTED), to_vector(CONTRADICTED)]))[:, 1]
    assert p[0] < 0.2 and p[1] > 0.8


def test_monotone_constraints_hold(trained):
    """More contradiction must never LOWER the hallucination probability."""
    base = dict(SUPPORTED)
    probs = []
    for c in np.linspace(0, 1, 11):
        base["nli_contradiction"] = c
        probs.append(trained.model.predict_proba(np.array([to_vector(base)]))[0, 1])
    assert all(b >= a - 1e-9 for a, b in zip(probs, probs[1:]))


@pytest.mark.parametrize("force_fallback", [False, True])
def test_shap_explanation_adds_up(trained, force_fallback):
    ex = Explainer(trained)
    if force_fallback:
        ex._shap = None                                          # exercise the built-in TreeSHAP path
    vec = to_vector(CONTRADICTED)
    prob = float(trained.model.predict_proba(np.array([vec]))[0, 1])
    out = ex.explain(vec, prob)

    # SHAP guarantee: base value + sum of contributions == model output (in log-odds)
    margin = out["base_value"] + sum(c["shap"] for c in out["contributions"])
    assert math.isclose(1 / (1 + math.exp(-margin)), prob, abs_tol=0.01)
    assert out["contributions"][0]["feature"] in ("nli_contradiction", "nli_entailment", "number_consistency")
    assert out["reason"].startswith("Flagged mainly because")
    assert {c["feature"] for c in out["contributions"]} == set(FEATURE_NAMES)


# ---------- the HTTP route (pipeline replaced by a stub) ----------
def test_analyze_route(monkeypatch):
    from fastapi.testclient import TestClient
    from app import config
    monkeypatch.setattr(config, "PRELOAD_MODELS", False)
    from app.main import app

    class StubPipeline:
        classifier = type("C", (), {"meta": {"source": "stub"}})()
        retriever = type("R", (), {"passages": []})()
        verifier = type("V", (), {"is_loaded": False})()
        def analyze(self, q, a, w=None):
            return {"overall": {"num_claims": 1}, "claims": [{"claim": a}], "meta": {}}

    with TestClient(app) as client:
        app.state.pipeline = StubPipeline()
        ok = client.post("/analyze", json={"question": "q", "answer": "The sky is blue today."})
        assert ok.status_code == 200 and ok.json()["claims"][0]["claim"] == "The sky is blue today."
        assert client.post("/analyze", json={"answer": "   "}).status_code == 422
        assert client.post("/analyze", json={"answer": "x" * 9000}).status_code == 413
        assert client.get("/health").json()["classifier"] == "stub"
