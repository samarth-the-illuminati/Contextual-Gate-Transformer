import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.sentence_splitter import split_sentences
from utils.text_loader import load_text_from_file


class SemanticMMRSummarizer:
    """
    Transformer-based extractive summarizer.

    It combines:
    - semantic relevance from MiniLM embeddings
    - TF-IDF keyword importance
    - sentence position
    - MMR redundancy control
    """

    def __init__(
        self,
        top_k: int = 3,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        mmr_lambda: float = 0.75,
    ):
        self.top_k = top_k
        self.mmr_lambda = mmr_lambda
        self.encoder = SentenceTransformer(model_name)

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)

        minimum = values.min()
        maximum = values.max()

        if maximum - minimum < 1e-8:
            return np.ones_like(values)

        return (values - minimum) / (maximum - minimum)

    def _semantic_scores(
        self,
        sentences: List[str],
        embeddings: np.ndarray,
    ) -> np.ndarray:
        document_embedding = embeddings.mean(axis=0, keepdims=True)
        scores = cosine_similarity(embeddings, document_embedding).reshape(-1)
        return self._normalize(scores)

    def _keyword_scores(self, sentences: List[str]) -> np.ndarray:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            lowercase=True,
        )

        matrix = vectorizer.fit_transform(sentences)
        scores = np.asarray(matrix.sum(axis=1)).reshape(-1)

        return self._normalize(scores)

    def _position_scores(self, sentence_count: int) -> np.ndarray:
        if sentence_count == 1:
            return np.ones(1, dtype=np.float32)

        positions = np.arange(sentence_count, dtype=np.float32)
        scores = 1.0 - (positions / (sentence_count - 1))

        return scores

    def _mmr_select(
        self,
        relevance_scores: np.ndarray,
        embeddings: np.ndarray,
    ) -> List[int]:
        sentence_count = len(relevance_scores)
        similarity_matrix = cosine_similarity(embeddings)

        selected = []
        remaining = set(range(sentence_count))

        while remaining and len(selected) < min(self.top_k, sentence_count):
            best_index = None
            best_score = -float("inf")

            for index in remaining:
                relevance = relevance_scores[index]

                if not selected:
                    redundancy = 0.0
                else:
                    redundancy = max(
                        similarity_matrix[index][selected_index]
                        for selected_index in selected
                    )

                mmr_score = (
                    self.mmr_lambda * relevance
                    - (1.0 - self.mmr_lambda) * redundancy
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_index = index

            selected.append(best_index)
            remaining.remove(best_index)

        return sorted(selected)

    def rank_sentences(
        self,
        sentences: List[str],
    ) -> Tuple[List[int], np.ndarray]:
        embeddings = self.encoder.encode(
            sentences,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        semantic = self._semantic_scores(sentences, embeddings)
        keyword = self._keyword_scores(sentences)
        position = self._position_scores(len(sentences))

        # Initial multi-factor score.
        # Mamba score will be added in the next phase.
        relevance = (
            0.60 * semantic
            + 0.25 * keyword
            + 0.15 * position
        )

        selected_indices = self._mmr_select(relevance, embeddings)
        return selected_indices, relevance

    def summarize(self, text: str) -> List[str]:
        sentences = split_sentences(text)

        if not sentences:
            return []

        if len(sentences) <= self.top_k:
            return sentences

        selected_indices, _ = self.rank_sentences(sentences)
        return [sentences[index] for index in selected_indices]


def summarize_file(input_path: str, top_k: int = 3) -> List[str]:
    text = load_text_from_file(input_path)
    summarizer = SemanticMMRSummarizer(top_k=top_k)
    return summarizer.summarize(text)


def main():
    input_path = Path("data/samples/sample1.txt")

    if not input_path.exists():
        print(f"File not found: {input_path}")
        return

    summarizer = SemanticMMRSummarizer(top_k=3)
    text = load_text_from_file(str(input_path))
    summary = summarizer.summarize(text)

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    summary_text = "\n".join(summary)

    (output_dir / "semantic_summary.txt").write_text(
        summary_text,
        encoding="utf-8",
    )

    pd.DataFrame(
        {
            "summary_sentence": summary,
        }
    ).to_csv(
        output_dir / "semantic_summary.csv",
        index=False,
    )

    print("\nSemantic MMR Summary:\n")
    print(summary_text)


if __name__ == "__main__":
    main()