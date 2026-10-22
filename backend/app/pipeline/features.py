"""The single source of truth for the feature vector fed to the classifier."""

FEATURE_NAMES = [
    "nli_entailment",       # best entailment probability of the claim vs any evidence passage
    "nli_contradiction",    # worst contradiction probability of the claim vs any evidence passage
    "semantic_similarity",  # best cosine similarity between claim and an evidence passage
    "lexical_overlap",      # share of the claim's content words found in the evidence
    "entity_coverage",      # share of the claim's named entities found in the evidence
    "number_consistency",   # share of the claim's numbers found in the evidence
]

# +1: higher value pushes toward "hallucinated"; -1: higher value pushes toward "supported".
# Used as XGBoost monotone constraints so the model can never learn something nonsensical
# like "more contradiction => less hallucination".
MONOTONE_DIRECTION = {
    "nli_entailment": -1,
    "nli_contradiction": +1,
    "semantic_similarity": -1,
    "lexical_overlap": -1,
    "entity_coverage": -1,
    "number_consistency": -1,
}


def to_vector(feature_dict: dict) -> list:
    return [float(feature_dict[name]) for name in FEATURE_NAMES]
