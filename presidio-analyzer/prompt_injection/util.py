import importlib
import json
import logging
import re
import sys
from functools import lru_cache
from typing import Any, Dict, List, Literal, Optional

LOGGER = logging.getLogger(__name__)

def _get_library_name() -> str:
    return __name__.split(".")[0]

def get_logger(name: Optional[str] = None) -> Any:
    """
    Return a logger with the specified name.
    """

    if name is None:
        name = _get_library_name()

    return logging.getLogger(name)


# Detect pytorch device
@lru_cache(maxsize=None)  # Unbounded cache
def device():
    torch = lazy_load_dep("torch")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")

def lazy_load_dep(import_name: str, package_name: Optional[str] = None):
    """Helper function to lazily load optional dependencies. If the dependency is not
    present, the function will raise an error _when used_.

    NOTE: This wrapper adds a warning message at import time.
    """

    if package_name is None:
        package_name = import_name

    spec = importlib.util.find_spec(import_name)
    if spec is None:
        LOGGER.warning(
            f"Optional feature dependent on missing package: {import_name} was initialized.\n"
            f"Use `pip install {package_name}` to install the package if running locally."
        )

    return importlib.import_module(import_name)

def calculate_risk_score(score: float, threshold: float) -> float:
    if score > threshold:
        return 1.0

    risk_score = round(abs(score - threshold) / threshold, 1)
    # Ensure risk score is between 0 and 1
    return min(max(risk_score, 0), 1)

def chunk_text(text: str, chunk_size: int) -> List[str]:
    text = text.strip()
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def chunk_text_by_sentences(text: str, max_chunk_size: int) -> List[str]:
    nltk = lazy_load_dep("nltk")

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    sentences = nltk.sent_tokenize(text.strip())

    chunks = []
    chunk = []
    chunk_size = 0

    for sentence in sentences:
        sentence_length = len(sentence)
        if chunk_size + sentence_length <= max_chunk_size:
            chunk.append(sentence)
            chunk_size += sentence_length
        else:
            if chunk:  # Check if chunk is non-empty
                chunks.append(" ".join(chunk))
            chunk = [sentence]
            chunk_size = sentence_length

    if chunk:  # Don't forget the last chunk, and check if it's non-empty
        chunks.append(" ".join(chunk))

    return [chunk for chunk in chunks if chunk.strip()]


def split_text_by_sentences(text: str) -> List[str]:
    nltk = lazy_load_dep("nltk")

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    return nltk.sent_tokenize(text.strip())


def split_text_to_word_chunks(
    input_length: int, chunk_length: int, overlap_length: int
) -> List[List]:
    """The function calculates chunks of text with size chunk_length. Each chunk has overlap_length number of
    words to create context and continuity for the model

    :param input_length: Length of input_ids for a given text
    :type input_length: int
    :param chunk_length: Length of each chunk of input_ids.
    Should match the max input length of the transformer model
    :type chunk_length: int
    :param overlap_length: Number of overlapping words in each chunk
    :type overlap_length: int
    :return: List of start and end positions for individual text chunks
    :rtype: List[List]
    """
    if input_length < chunk_length:
        return [[0, input_length]]
    if chunk_length <= overlap_length:
        LOGGER.warning(
            "overlap_length should be shorter than chunk_length, setting overlap_length to by half of chunk_length"
        )
        overlap_length = chunk_length // 2
    return [
        [i, min([i + chunk_length, input_length])]
        for i in range(0, input_length - overlap_length, chunk_length - overlap_length)
    ]


def truncate_tokens_head_tail(tokens, max_length=512, head_length=128, tail_length=382):
    if len(tokens) > max_length:
        head_tokens = tokens[:head_length]
        tail_tokens = tokens[-tail_length:]
        tokens = head_tokens + tail_tokens
    return tokens


url_pattern = re.compile(
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
)


def extract_urls(text: str) -> List[str]:
    """
    Extracts URLs from the given text.
    """
    return url_pattern.findall(text)


def remove_markdown(text):
    # Patterns to remove various Markdown elements
    patterns = [
        r"\*\*([^\*]+)\*\*",  # Bold
        r"\*([^\*]+)\*",  # Italic
        r"\!\[[^\]]+\]\([^\)]+\)",  # Images
        r"\[[^\]]+\]\([^\)]+\)",  # Links
        r"\#{1,6}\s",  # Headers
        r"\>+",  # Blockquotes
        r"`{1,3}[^`]+`{1,3}",  # Inline code and code blocks
        r"\n{2,}",  # Multiple newlines
    ]

    clean_text = text
    for pattern in patterns:
        clean_text = re.sub(pattern, "", clean_text)

    # Extra cleanup for simpler elements
    clean_text = re.sub(r"\*|\_|\`", "", clean_text)

    return clean_text