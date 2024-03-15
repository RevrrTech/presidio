from typing import List, Tuple, Optional

from presidio_analyzer import Pattern, PatternRecognizer

class CreditCardExpiryRecognizer(PatternRecognizer):
    """
    Recognize credit card expiry dates using regex.

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    :param replacement_pairs: List of tuples with potential replacement values
    for different strings to be used during pattern matching.
    """

    PATTERNS = [
        Pattern(
            "Credit Card Expiry (MM/YY or MM/YYYY)",
            r"\b(0[1-9]|1[0-2])/(19|20)?\d{2}\b",
            0.5,
        ),
        Pattern(
            "Credit Card Expiry (MM-YY)",
            r"\b(0[1-9]|1[0-2])-(19|20)?\d{2}\b",
            0.5,
        ),
        Pattern(
            "Credit Card Expiry (Month Year)",
            r"\b(January|February|March|April|May|June|July|August|September|October|November|December) (19|20)?\d{2}\b",
            0.5,
        ),
        Pattern(
            "Credit Card Expiry (YYMM)",
            r"\b(19|20)?\d{2}(0[1-9]|1[0-2])\b",
            0.5,
        ),
        Pattern(
            "Credit Card Expiry (MMM-YY)",
            r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-(19|20)?\d{2}\b",
            0.5,
        ),
        Pattern(
            "Credit Card Expiry (MM.YY)",
            r"\b(0[1-9]|1[0-2]).(19|20)?\d{2}\b",
            0.5,
        ),
        Pattern(
            "Credit Card Expiry (Month/YYYY)",
            r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) (19|20)?\d{2}\b",
            0.5,
        ),
    ]

    CONTEXT = [
        "credit",
        "card",
        "visa",
        "mastercard",
        "cc ",
        "amex",
        "discover",
        "jcb",
        "diners",
        "maestro",
        "instapayment",
        "expiry",
        "expiration",
        "valid",
        "end",
        "date",
        "month",
        "year",
        "mm/yy",
        "mm-yy",
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "en",
        supported_entity: str = "CREDIT_CARD_EXPIRY",
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