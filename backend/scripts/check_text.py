"""Check pasted text with the same pipeline used by the FastAPI API.

Run from the backend directory, for example:
    python scripts/check_text.py --text "The Eiffel Tower was completed in 1889."
"""
import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app import config
from app.pipeline.orchestrator import HalluPipeline
from app.pipeline.keyword_attribution import highlight_claim, unsupported_token_cues


def build_parser():
    parser = argparse.ArgumentParser(description="Check an AI answer for likely hallucinations.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", help="Answer text to check.")
    source.add_argument("--file", type=Path, help="Read the answer from a UTF-8 text file.")
    source.add_argument("--stdin", action="store_true", help="Read the answer from standard input.")
    parser.add_argument("--interactive", action="store_true",
                        help="Check answers in a loop until 'quit'.")
    parser.add_argument("--question", default="", help="Question that provides context for the answer.")
    parser.add_argument("--no-wikipedia", action="store_true", help="Disable Wikipedia retrieval.")
    parser.add_argument("--evidence-file", type=Path,
                        help="Reserved for the evidence-backed attribution phase.")
    parser.add_argument("--top-keywords", type=int, default=5,
                        help="Reserved for the keyword attribution phase.")
    parser.add_argument("--max-tokens", type=int, default=25,
                        help="Reserved for the keyword attribution phase.")
    parser.add_argument("--threshold", type=float, help="Override the saved hallucination threshold.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--lime", action="store_true", help="Reserved for the optional LIME phase.")
    parser.add_argument("--allow-placeholder", action="store_true",
                        help="Allow the synthetic placeholder model (illustrative results only).")
    parser.add_argument("--no-color", action="store_true", help="Disable coloured terminal output.")
    parser.add_argument("--verbose", action="store_true", help="Print known pipeline limitations.")
    parser.add_argument("--selftest", action="store_true", help="Reserved for the retraining sanity check.")
    return parser


def check_model_preconditions(allow_placeholder=False):
    """Return saved metadata, refusing missing, incomplete, or placeholder models."""
    model_path = config.CLASSIFIER_PATH
    meta_path = config.CLASSIFIER_META_PATH
    if not model_path.exists() or not meta_path.exists():
        raise RuntimeError(
            f"No complete trained model found. Expected {model_path} and {meta_path}. "
            "Train the classifier before running this tool."
        )

    try:
        json.loads(model_path.read_text(encoding="utf-8"))
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "The saved model appears incomplete or is being written. "
            "Wait for training to finish and try again."
        ) from exc

    if not isinstance(metadata, dict) or not metadata.get("source"):
        raise RuntimeError("The saved model metadata is incomplete; refusing to run.")

    if metadata["source"] == "synthetic-placeholder" and not allow_placeholder:
        raise RuntimeError(
            "WARNING: the saved classifier is a synthetic-placeholder model. "
            "Results would be illustrative only. Re-run with --allow-placeholder "
            "to proceed, or train a real model first."
        )
    return metadata


def read_answer(args):
    if args.text is not None:
        return args.text
    if args.file is not None:
        return args.file.read_text(encoding="utf-8-sig")
    if args.stdin:
        return sys.stdin.read()
    if args.interactive:
        return None

    print("Paste the answer, then enter an empty line (or press Ctrl+Z then Enter):")
    lines = []
    for line in sys.stdin:
        if not line.rstrip("\r\n"):
            break
        lines.append(line.rstrip("\r\n"))
    return "\n".join(lines)


def run_analysis(pipeline, answer, question="", use_wikipedia=True):
    """Delegate scoring to HalluPipeline so CLI and API use identical scores."""
    return pipeline.analyze(question, answer.strip(), use_wikipedia)


def _verdict(score, threshold):
    if score >= threshold:
        return "LIKELY HALLUCINATED"
    if score < config.LOW_RISK:
        return "SUPPORTED"
    return "UNCERTAIN"


def _display_result(result, threshold, no_color=False):
    console = Console(no_color=no_color or not sys.stdout.isatty())
    print()
    for claim in result["claims"]:
        score = float(claim["score"])
        cues = claim.get("keywords", [])
        console.print(Text.from_ansi(f"Claim: {claim['claim_display']}"))
        console.print(
            f"Verdict: {_verdict(score, threshold)} | risk: {score:.1%} | "
            f"threshold: {threshold:.1%}"
        )
        table = Table(title="Top keywords")
        for column in ("token", "type", "in evidence?", "note"):
            table.add_column(column)
        for cue in cues:
            table.add_row(cue["token"], cue["token_type"], cue["status"], cue["note"] or "-")
        console.print(table)
        print(f"SHAP explanation: {claim['explanation']['reason']}")
        print("Features: " + ", ".join(
            f"{name}={value:.4f}" for name, value in claim["features"].items()
        ))
        print("Evidence:")
        if claim["evidence"]:
            for evidence in claim["evidence"]:
                print(f"  [{evidence['source']}] {evidence['text'][:200]}")
        else:
            print("  (none retrieved)")
        print()

    flagged = [
        claim for claim in result["claims"]
        if float(claim["score"]) >= threshold
    ]
    riskiest = max(result["claims"], key=lambda claim: claim["score"], default=None)
    print(f"OVERALL: {len(flagged)} of {len(result['claims'])} claims flagged.")
    if riskiest:
        print(f"Riskiest claim: {riskiest['claim']} ({float(riskiest['score']):.1%})")
    print(f"Hallucination detected: {'YES' if flagged else 'NO'}")


def process_one(args, pipeline, answer, metadata):
    answer = (answer or "").strip()
    if not answer:
        raise ValueError("No answer text was provided.")
    if len(answer) > config.MAX_ANSWER_CHARS:
        raise ValueError(f"Text too long ({len(answer)} chars, max {config.MAX_ANSWER_CHARS}).")

    threshold = metadata.get("threshold", config.HIGH_RISK) if args.threshold is None else args.threshold
    if not 0 <= threshold <= 1:
        raise ValueError("--threshold must be between 0 and 1.")
    result = run_analysis(pipeline, answer, args.question, not args.no_wikipedia)
    for claim in result["claims"]:
        cues = unsupported_token_cues(
            claim["claim"],
            [e["text"] for e in claim["evidence"]],
            max_tokens=args.max_tokens,
        )
        claim["keywords"] = [cue.as_dict() for cue in cues[:args.top_keywords]]
        claim["claim_display"] = highlight_claim(
            claim["claim"], cues[:args.top_keywords],             color=not args.json and not args.no_color and sys.stdout.isatty()
        )
    if args.json:
        result["cli"] = {
            "threshold": threshold,
            "model": metadata,
            "phase": 1,
        }
        print(json.dumps(result, indent=2))
    else:
        _display_result(result, threshold, args.no_color)
    return any(float(claim["score"]) >= threshold for claim in result["claims"])


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        metadata = check_model_preconditions(args.allow_placeholder)
        if args.evidence_file is not None:
            raise ValueError("--evidence-file will be implemented in Phase 3.")
        if args.lime:
            raise ValueError("--lime will be implemented in Phase 3.")
        if args.selftest:
            raise ValueError("--selftest will be implemented in Phase 5.")

        pipeline = HalluPipeline()
        if args.verbose:
            print("Limitations: scores are model signals, not proof; inspect the evidence.")
            print("Retrieval can miss supporting evidence, and opinions or recent events may be unverifiable.")
            print("Claims with fewer than 4 words are skipped by the claim splitter.")

        answer = read_answer(args)
        if args.interactive:
            while True:
                if answer is None:
                    print("Paste an answer (or type 'quit'):")
                    answer = sys.stdin.readline()
                    if not answer:
                        break
                    answer = answer.strip()
                if answer.lower() == "quit":
                    break
                process_one(args, pipeline, answer, metadata)
                answer = None
            return 0

        return 1 if process_one(args, pipeline, answer, metadata) else 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
