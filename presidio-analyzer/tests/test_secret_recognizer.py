import pytest

from presidio_analyzer.predefined_recognizers.secrets_recognizer import SecretsRecognizer
from tests import assert_result

@pytest.fixture(scope="module")
def recognizer():
    return SecretsRecognizer()

@pytest.mark.parametrize(
    "text, expected_len, entities, expected_positions, score",
    [
        # fmt: off
        ("TOKEN: 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", 1, ["GitHub Token"], ((7, 47),), 1),
        # Add more test cases here
        # fmt: on
    ],
)
def test_when_all_secrets_then_succeed(
    spacy_nlp_engine,
    text,
    expected_len,
    entities,
    expected_positions,
    score,
    recognizer,
):
    nlp_artifacts = spacy_nlp_engine.process_text(text, "en")
    results = recognizer.analyze(text, entities, nlp_artifacts=nlp_artifacts)
    assert len(results) == expected_len
    for i, (res, (st_pos, fn_pos)) in enumerate(zip(results, expected_positions)):
        assert_result(res, entities[i], st_pos, fn_pos, score)