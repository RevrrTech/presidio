import pytest

from tests import assert_result_within_score_range
from presidio_analyzer.predefined_recognizers import DateOfBirthRecognizer


@pytest.fixture(scope="module")
def recognizer():
    return DateOfBirthRecognizer()


@pytest.fixture(scope="module")
def entities():
    return ["DATE_OF_BIRTH"]


@pytest.mark.parametrize(
    "text, expected_len, expected_positions, expected_score_ranges",
    [
        # fmt: off
        # Date tests
        (
            "My birthday is 5-20-2021", 
            1, 
            ((15, 24),), 
            ((0.6, 0.81),),
        ),
        (
            "The birthdate is 5/20/2021", 
            1, 
            ((17, 26),), 
            ((0.6, 0.81),),
        ),
        (
            "His dob is 2021-05-21", 
            1, 
            ((11, 21),), 
            ((0.6, 0.81),),
        ),
        (
            "She was born on 21.5.2021", 
            1, 
            ((16, 25),), 
            ((0.6, 0.81),),
        ),
        (
            "His birth year, month and day are 21.5.21", 
            1, 
            ((34, 41),), 
            ((0.6, 0.81),),
        ),
        (
            "Her bday is 5-MAY-2021", 
            1, 
            ((12, 22),), 
            ((0.6, 0.81),),
        )
        # fmt: on
    ],
)
def test_when_all_dates_then_succeed(
    text,
    expected_len,
    expected_positions,
    expected_score_ranges,
    recognizer,
    entities,
    max_score,
):
    results = recognizer.analyze(text, entities)
    assert len(results) == expected_len
    for res, (st_pos, fn_pos), (st_score, fn_score) in zip(
        results, expected_positions, expected_score_ranges
    ):
        if fn_score == "max":
            fn_score = max_score
        assert_result_within_score_range(
            res, entities[0], st_pos, fn_pos, st_score, fn_score
        )
