import pytest

from tests import assert_result
from presidio_analyzer.predefined_recognizers import CreditCardExpiryRecognizer

@pytest.fixture(scope="module")
def cc_expiry_recognizer():
    return CreditCardExpiryRecognizer()

@pytest.fixture(scope="module")
def entities():
    return ["CREDIT_CARD_EXPIRY"]

@pytest.mark.parametrize(
    "text, expected_len, expected_scores, expected_res",
    [
        # fmt: off
        ("03/2025", 1, (), [(0, 7)]),
        ("03/25", 1, (), [(0, 7)]),
        ("03-25", 1, (), [(0, 7)]),
        ("March 2025", 1, (), [(0, 10)]),
        ("202503", 1, (), [(0, 6)]),
        ("Mar-25", 1, (), [(0, 6)]),
        ("03.25", 1, (), [(0, 7)]),
        ("Mar 2025", 1, (), [(0, 8)]),
        ("The credit card expiry is 03/2025", 1, (), [(24, 31)]),
        ("This is a visa card with expiration date 03-25", 1, (), [(39, 46)]),
        ("My mastercard is valid until March 2025", 1, (), [(30, 40)]),
        ("The amex card has an end date of 202503", 1, (), [(32, 38)]),
        ("The discover card expires in Mar-25", 1, (), [(30, 36)]), 
        ("The jcb card has a date of 03.25", 1, (), [(26, 33)]),
        ("The diners card is valid until Mar 2025", 1, (), [(30, 38)]),
        # fmt: on
    ],
)
def test_when_all_credit_card_expiry_then_succeed(
    text,
    expected_len,
    expected_scores,
    expected_res,
    cc_expiry_recognizer,
    entities,
):
    results = cc_expiry_recognizer.analyze(text, entities)
    print("results: ", results)
    assert len(results) == expected_len
    for res, expected_score in zip(results, expected_scores):
        assert res.score == expected_score
    # for res, (start, end) in zip(results, expected_res):
    #     assert_result(res, entities[0], start, end, expected_score)