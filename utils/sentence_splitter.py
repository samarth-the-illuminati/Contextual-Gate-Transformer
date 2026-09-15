import re
from typing import List


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text.strip())

    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]