import json

import pytest

from app import config
from scripts import check_text
from app.pipeline.keyword_attribution import highlight_claim, unsupported_token_cues


def test_model_preconditions_reject_placeholder(monkeypatch, tmp_path):
    model_path = tmp_path / "xgb_classifier.json"
    meta_path = tmp_path / "xgb_meta.json"
    model_path.write_text(json.dumps({"model": "complete"}), encoding="utf-8")
    meta_path.write_text(json.dumps({"source": "synthetic-placeholder"}), encoding="utf-8")
    monkeypatch.setattr(config, "CLASSIFIER_PATH", model_path)
    monkeypatch.setattr(config, "CLASSIFIER_META_PATH", meta_path)

    with pytest.raises(RuntimeError, match="synthetic-placeholder"):
        check_text.check_model_preconditions()


def test_model_preconditions_reject_incomplete_model(monkeypatch, tmp_path):
    model_path = tmp_path / "xgb_classifier.json"
    meta_path = tmp_path / "xgb_meta.json"
    model_path.write_text("{", encoding="utf-8")
    meta_path.write_text(json.dumps({"source": "real"}), encoding="utf-8")
    monkeypatch.setattr(config, "CLASSIFIER_PATH", model_path)
    monkeypatch.setattr(config, "CLASSIFIER_META_PATH", meta_path)

    with pytest.raises(RuntimeError, match="incomplete"):
        check_text.check_model_preconditions()


def test_cli_delegates_to_pipeline_for_score_parity():
    class StubPipeline:
        def analyze(self, question, answer, use_wikipedia):
            return {"claims": [{"score": 0.73}], "question": question,
                    "answer": answer, "use_wikipedia": use_wikipedia}

    result = check_text.run_analysis(StubPipeline(), "A claim.", "A question.", False)
    assert result["claims"][0]["score"] == 0.73
    assert result["question"] == "A question."
    assert result["use_wikipedia"] is False


def test_unsupported_tokens_marks_missing_number_and_correction():
    cues = unsupported_token_cues(
        "The tower was completed in 1998.",
        ["The tower was completed in 1889."],
        entities=[],
    )
    number = next(cue for cue in cues if cue.token == "1998")
    assert number.found is False
    assert number.note == "evidence says 1889"


def test_unsupported_tokens_marks_found_entity_and_missing_word():
    cues = unsupported_token_cues(
        "The Eiffel Tower is purple.",
        ["The Eiffel Tower is in Paris."],
        entities=["Eiffel Tower"],
    )
    by_token = {cue.token: cue for cue in cues}
    assert by_token["Eiffel Tower"].found is True
    assert by_token["purple"].found is False
    assert "\033[31mpurple\033[0m" in highlight_claim("The Eiffel Tower is purple.", cues)
