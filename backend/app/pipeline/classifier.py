"""Stage 5: turn the feature vector into P(hallucinated) with XGBoost.

On first start there is no trained model, so we bootstrap one from SYNTHETIC data just so the whole
pipeline runs end to end. That is a placeholder - replace it with a model trained on real labelled
data (e.g. HaluEval) by running:

    python -m app.pipeline.classifier --csv path/to/features.csv

CSV columns: the 6 feature names in features.py plus `label` (1 = hallucinated, 0 = supported).
"""
import argparse
import json
import logging
import numpy as np
from xgboost import XGBClassifier

from app import config
from app.pipeline.features import FEATURE_NAMES, MONOTONE_DIRECTION

logger = logging.getLogger(__name__)


def synthetic_dataset(n=6000, seed=42):
    """Plausible feature distributions for supported vs contradicted vs unsupported claims."""
    rng = np.random.default_rng(seed)
    b = rng.beta
    n_pos = n // 2
    n_neg = n - n_pos
    n_con = n_pos // 2
    n_uns = n_pos - n_con

    def num(size, p_exact, a, c):
        return np.where(rng.random(size) < p_exact, 1.0, b(a, c, size))

    supported = np.column_stack([b(6, 2, n_neg), b(1, 12, n_neg), b(7, 2.5, n_neg),
                                 b(6, 2, n_neg), b(7, 1.5, n_neg), num(n_neg, 0.93, 2, 3)])
    contradicted = np.column_stack([b(1, 8, n_con), b(7, 2, n_con), b(5, 3, n_con),
                                    b(4, 3, n_con), b(4, 2, n_con), num(n_con, 0.5, 2, 4)])
    unsupported = np.column_stack([b(1.5, 7, n_uns), b(1, 10, n_uns), b(2.5, 5, n_uns),
                                   b(2, 4, n_uns), b(2, 3, n_uns), num(n_uns, 0.4, 1.5, 4)])
    X = np.vstack([supported, contradicted, unsupported])
    y = np.concatenate([np.zeros(n_neg), np.ones(n_pos)])
    order = rng.permutation(len(y))
    return X[order], y[order]


def train(X, y, source):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = XGBClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.08, subsample=0.9,
        monotone_constraints=tuple(MONOTONE_DIRECTION[n] for n in FEATURE_NAMES),
        eval_metric="logloss", random_state=42,
    )
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    meta = {
        "source": source,
        "n_samples": int(len(y)),
        "holdout_accuracy": round(float(accuracy_score(y_te, proba > 0.5)), 4),
        "holdout_auroc": round(float(roc_auc_score(y_te, proba)), 4),
        "features": FEATURE_NAMES,
    }
    return model, meta


def save(model, meta):
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(config.CLASSIFIER_PATH))
    config.CLASSIFIER_META_PATH.write_text(json.dumps(meta, indent=2))


class HallucinationClassifier:
    def __init__(self):
        self.model = XGBClassifier()
        if config.CLASSIFIER_PATH.exists():
            self.model.load_model(str(config.CLASSIFIER_PATH))
            self.meta = (json.loads(config.CLASSIFIER_META_PATH.read_text())
                         if config.CLASSIFIER_META_PATH.exists() else {"source": "unknown"})
            logger.info("Loaded classifier (%s)", self.meta.get("source"))
        else:
            logger.warning("No trained classifier found - bootstrapping a SYNTHETIC placeholder model.")
            X, y = synthetic_dataset()
            self.model, self.meta = train(X, y, source="synthetic-placeholder")
            save(self.model, self.meta)

    @property
    def is_placeholder(self):
        return self.meta.get("source") == "synthetic-placeholder"

    def predict(self, X):
        """X: array (n, n_features) -> P(hallucinated) for each row."""
        return self.model.predict_proba(np.asarray(X, dtype=float))[:, 1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the hallucination classifier")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--synthetic", action="store_true", help="train on synthetic placeholder data")
    group.add_argument("--csv", help="CSV with the feature columns + `label`")
    args = parser.parse_args()

    if args.synthetic:
        X, y = synthetic_dataset()
        source = "synthetic-placeholder"
    else:
        import pandas as pd
        df = pd.read_csv(args.csv)
        X, y, source = df[FEATURE_NAMES].to_numpy(), df["label"].to_numpy(), f"csv:{args.csv}"
    model, meta = train(X, y, source)
    save(model, meta)
    print(json.dumps(meta, indent=2))
