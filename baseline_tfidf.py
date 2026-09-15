import re
from pathlib import Path
from typing import List

import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.text_loader import load_text_from_file
from utils.sentence_splitter import split_sentences

nltk.download("punkt", quiet=True)

class OfflineSummarizer:
    def __init__(self, top_k: int = 3):
        self.top_k = top_k

    def summarize(self, text: str) -> List[str]:
        sentences = split_sentences(text)
        if len(sentences) <= self.top_k:
            return sentences

        vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
        X = vectorizer.fit_transform(sentences)
        sim_matrix = cosine_similarity(X)
        scores = sim_matrix.sum(axis=1)

        top_indices = sorted(
            sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)[:self.top_k]
        )
        return [sentences[i] for i in top_indices]


def summarize_file(input_path: str, top_k: int = 3) -> List[str]:
    text = load_text_from_file(input_path)
    summarizer = OfflineSummarizer(top_k=top_k)
    return summarizer.summarize(text)


def main():
    sample_text = """
    Artificial intelligence is changing how students study, write, and summarize documents.
    Long reports can be difficult to process on low-power devices because traditional models can require large memory.
    Mamba is a state-space architecture designed for efficient sequence modeling and linear scaling with sequence length.
    This makes it attractive for offline summarization on laptops and edge devices.
    A hybrid transformer plus Mamba design can use a pretrained transformer to create sentence embeddings.
    A Mamba layer can then model sentence order and long-range dependencies.
    For an undergraduate project, extractive summarization is a practical first step.
    The project can run fully offline and still demonstrate privacy, speed, and efficiency.
    """

    summarizer = OfflineSummarizer(top_k=3)
    summary = summarizer.summarize(sample_text)

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary))

    pd.DataFrame({"summary_sentence": summary}).to_csv(out_dir / "summary.csv", index=False)
    print("\n".join(summary))


if __name__ == "__main__":
    main()