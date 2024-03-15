# File: test_credit_card_cvv_recognizer.py
import pytest

from tests import assert_result
from presidio_analyzer.predefined_recognizers import CreditCardCVVRecognizer

@pytest.fixture(scope="module")
def cc_cvv_recognizer():
    return CreditCardCVVRecognizer()

@pytest.fixture(scope="module")
def entities():
    return ["CREDIT_CARD_CVV"]

@pytest.mark.parametrize(
    "text, expected_len, expected_scores, expected_res",
    [
        ("The cvv of my visa card is 123", 1, (), ((27, 30),)),
        ("My mastercard has a security code of 4567", 1, (), ((35, 39),)),
    ],
)
def test_when_all_credit_card_cvv_then_succeed(
    text,
    expected_len,
    expected_scores,
    expected_res,
    cc_cvv_recognizer,
    entities,
):
    results = cc_cvv_recognizer.analyze(text, entities)
    assert len(results) == expected_len

    for res, expected_score, (start, end) in zip(results, expected_scores, expected_res):
        assert res.score == expected_score
        assert_result(res, entities[0], start, end, expected_score)