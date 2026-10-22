r"""Smoke-test the live API's NLI and numeric sanity signals.

Run from PowerShell with the backend server already running:
    backend\venv\Scripts\python.exe scripts\smoke_test.py
"""
import argparse
import sys

import requests


FEATURE_NAMES = (
    "nli_entailment",
    "nli_contradiction",
    "semantic_similarity",
    "lexical_overlap",
    "entity_coverage",
    "number_consistency",
)


def check_case(base_url, name, question, answer):
    response = requests.post(
        f"{base_url.rstrip('/')}/analyze",
        json={"question": question, "answer": answer, "use_wikipedia": False},
        timeout=300,
    )
    response.raise_for_status()
    data = response.json()
    if not data["claims"]:
        raise AssertionError(f"{name}: API returned no claims")
    claim = data["claims"][0]
    features = claim["features"]
    missing = set(FEATURE_NAMES) - set(features)
    if missing:
        raise AssertionError(f"{name}: missing features {sorted(missing)}")
    print(f"{name}: score={claim['score']:.4f}, verdict={claim['verdict']}")
    print("  " + ", ".join(f"{key}={features[key]:.4f}" for key in FEATURE_NAMES))
    return claim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    failures = []
    try:
        health = requests.get(f"{args.base_url.rstrip('/')}/health", timeout=30)
        health.raise_for_status()
        print("health: PASS", health.json())
        true_claim = check_case(
            args.base_url,
            "true claim",
            "Where is the Eiffel Tower?",
            "The Eiffel Tower is in Paris, France.",
        )
        false_claim = check_case(
            args.base_url,
            "false year",
            "When was the Eiffel Tower completed?",
            "The Eiffel Tower was completed in 1998.",
        )
        if true_claim["features"]["nli_entailment"] < 0.5:
            failures.append("true claim has low nli_entailment")
        if (
            false_claim["features"]["nli_contradiction"] < 0.5
            and false_claim["features"]["number_consistency"] > 0.5
        ):
            failures.append("false year has neither high contradiction nor low number consistency")
    except (requests.RequestException, KeyError, AssertionError) as exc:
        failures.append(str(exc))

    if failures:
        print("FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())