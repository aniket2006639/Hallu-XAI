"""Stage 6: explain WHY a claim got its score, using SHAP.

SHAP gives each feature a signed contribution (in log-odds). Positive = pushed toward
"hallucinated", negative = pushed toward "supported". Contributions + base value = the model's output.
"""
import logging
import numpy as np

from app import config
from app.pipeline.features import FEATURE_NAMES

logger = logging.getLogger(__name__)

# (feature, raises_risk) -> plain-English phrase
PHRASES = {
    ("nli_entailment", True): "the evidence does not clearly support the claim",
    ("nli_entailment", False): "the evidence logically supports the claim",
    ("nli_contradiction", True): "the evidence contradicts the claim",
    ("nli_contradiction", False): "nothing in the evidence contradicts the claim",
    ("semantic_similarity", True): "the claim is semantically far from the evidence",
    ("semantic_similarity", False): "the claim closely matches the evidence",
    ("lexical_overlap", True): "few of the claim's key words appear in the evidence",
    ("lexical_overlap", False): "the claim's key words appear in the evidence",
    ("entity_coverage", True): "named entities in the claim are missing from the evidence",
    ("entity_coverage", False): "all named entities in the claim appear in the evidence",
    ("number_consistency", True): "numbers/dates in the claim do not match the evidence",
    ("number_consistency", False): "numbers/dates in the claim match the evidence",
}


class Explainer:
    def __init__(self, classifier):
        self.classifier = classifier
        self._shap = None
        try:
            import shap
            self._shap = shap.TreeExplainer(classifier.model)
        except Exception as exc:
            logger.warning("shap.TreeExplainer unavailable (%s); using XGBoost's built-in TreeSHAP.", exc)

    def _shap_values(self, x):
        """Returns (per-feature contributions, base value) in log-odds."""
        x = np.asarray(x, dtype=float).reshape(1, -1)
        if self._shap is not None:
            try:
                values = self._shap.shap_values(x)
                if isinstance(values, list):
                    values = values[-1]
                base = np.ravel(self._shap.expected_value)[-1]
                return np.asarray(values).reshape(-1)[: len(FEATURE_NAMES)], float(base)
            except Exception as exc:
                logger.warning("shap failed (%s); falling back to built-in TreeSHAP.", exc)
        import xgboost as xgb
        contribs = self.classifier.model.get_booster().predict(xgb.DMatrix(x), pred_contribs=True)[0]
        return contribs[:-1], float(contribs[-1])

    def explain(self, feature_values, probability):
        x = np.asarray(feature_values, dtype=float)
        shap_vals, base = self._shap_values(x)

        contributions = [
            {
                "feature": name,
                "value": round(float(x[i]), 4),
                "shap": round(float(shap_vals[i]), 4),
                "direction": "raises_risk" if shap_vals[i] > 0 else "lowers_risk",
                "meaning": PHRASES[(name, bool(shap_vals[i] > 0))],
            }
            for i, name in enumerate(FEATURE_NAMES)
        ]
        contributions.sort(key=lambda c: abs(c["shap"]), reverse=True)
        return {
            "base_value": round(base, 4),
            "contributions": contributions,
            "reason": self._reason(contributions, probability),
        }

    @staticmethod
    def _reason(contributions, probability):
        raising = [c for c in contributions if c["shap"] > 0.1]
        lowering = [c for c in contributions if c["shap"] < -0.1]
        if probability >= config.HIGH_RISK:
            top = [c["meaning"] for c in raising[:2]] or ["the combined signals look weak"]
            return "Flagged mainly because " + " and ".join(top) + "."
        if probability < config.LOW_RISK:
            top = [c["meaning"] for c in lowering[:2]] or ["no signal looked suspicious"]
            return "Looks supported because " + " and ".join(top) + "."
        parts = []
        if raising:
            parts.append("against: " + raising[0]["meaning"])
        if lowering:
            parts.append("for: " + lowering[0]["meaning"])
        return "Uncertain - " + "; ".join(parts) + "." if parts else "Uncertain - mixed signals."
