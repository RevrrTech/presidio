from typing import List, Tuple, Optional

from presidio_analyzer import Pattern, PatternRecognizer

class CreditCardCVVRecognizer(PatternRecognizer):
    """
    Recognize credit card CVV numbers using regex.

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    :param replacement_pairs: List of tuples with potential replacement values
    for different strings to be used during pattern matching.
    """

    PATTERNS = [
        Pattern(
            "Credit Card CVV",
            r"\b\d{3,4}\b",
            0.5,
        ),
    ]

    CONTEXT = [
        "credit",
        "card",
        "visa",
        "mastercard",
        "cc",
        "amex",
        "discover",
        "jcb",
        "diners",
        "maestro",
        "instapayment",
        "cvv",
        "cvc",
        "cvv2",
        "cvc2",
        "cid",
        "security", #security verification code
        "code",
        "verification",
        "value", #card verification value
        "verification number",
        "Card Identification Number"
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "en",
        supported_entity: str = "CREDIT_CARD_CVV",
        replacement_pairs: Optional[List[Tuple[str, str]]] = None,
    ):
        patterns = patterns if patterns else self.PATTERNS
        context = context if context else self.CONTEXT
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns,
            context=context,
            supported_language=supported_language,
        )