import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.sentence_splitter import split_sentences
from utils.text_loader import load_text_from_file


class LightweightSelectiveSSM(nn.Module):
    """
    Small CPU-friendly selective state-space sequence layer.

    Input:
        [batch_size, sentence_count, embedding_dimension]

    Output:
        [batch_size, sentence_count, hidden_dimension]
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.state_candidate = nn.Linear(hidden_dim, hidden_dim)
        self.update_gate = nn.Linear(hidden_dim, hidden_dim)

        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = torch.tanh(self.input_projection(x))

        batch_size, sequence_length, hidden_dim = projected.shape

        state = torch.zeros(
            batch_size,
            hidden_dim,
            device=projected.device,
            dtype=projected.dtype,
        )

        outputs = []

        for time_step in range(sequence_length):
            current_input = projected[:, time_step, :]

            candidate = torch.tanh(
                self.state_candidate(current_input)
            )

            gate = torch.sigmoid(
                self.update_gate(current_input)
            )

            state = gate * candidate + (1.0 - gate) * state
            outputs.append(state.unsqueeze(1))

        sequence_output = torch.cat(outputs, dim=1)
        sequence_output = self.output_projection(sequence_output)

        return self.norm(sequence_output + projected)


class MambaSentenceRanker(nn.Module):
    """
    Receives Transformer sentence embeddings and returns
    one raw sentence-importance score per sentence.

    These are raw logits. Do not apply sigmoid here because
    BCEWithLogitsLoss will apply it internally during training.
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.ssm = LightweightSelectiveSSM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
        )

        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        contextual_embeddings = self.ssm(embeddings)
        raw_scores = self.scorer(contextual_embeddings).squeeze(-1)
        return raw_scores


class MambaTransformerSummarizer:
    """
    Offline Mamba–Transformer extractive summarizer.

    Transformer:
        Creates sentence-level semantic embeddings.

    Lightweight Selective SSM:
        Models sentence-to-sentence context.

    Multi-factor relevance:
        Mamba relevance + semantic relevance +
        TF-IDF keyword score + sentence position.

    MMR:
        Avoids repeated or highly similar sentences.
    """

    def __init__(
        self,
        top_k: int = 3,
        transformer_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        hidden_dim: int = 128,
        mmr_lambda: float = 0.75,
        model_path: str = "models/mamba_ranker.pt",
    ):
        self.top_k = top_k
        self.mmr_lambda = mmr_lambda

        self.encoder = SentenceTransformer(transformer_name)
        self.encoder.eval()

        self.model = MambaSentenceRanker(
            input_dim=384,
            hidden_dim=hidden_dim,
        )

        checkpoint_path = Path(model_path)

        if checkpoint_path.exists():
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
            )

            self.model.load_state_dict(
                checkpoint["model_state_dict"]
            )

            print(
                f"Loaded trained Mamba ranker from: "
                f"{checkpoint_path}"
            )
        else:
            print(
                "Warning: No trained Mamba checkpoint found. "
                "Using randomly initialized Mamba weights."
            )

        self.model.eval()

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)

        minimum = values.min()
        maximum = values.max()

        if maximum - minimum < 1e-8:
            return np.ones_like(values)

        return (values - minimum) / (maximum - minimum)

    def _semantic_scores(
        self,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        document_embedding = embeddings.mean(
            axis=0,
            keepdims=True,
        )

        scores = cosine_similarity(
            embeddings,
            document_embedding,
        ).reshape(-1)

        return self._normalize(scores)

    def _keyword_scores(
        self,
        sentences: List[str],
    ) -> np.ndarray:
        try:
            vectorizer = TfidfVectorizer(
                stop_words="english",
                lowercase=True,
            )

            matrix = vectorizer.fit_transform(sentences)

            scores = np.asarray(
                matrix.sum(axis=1)
            ).reshape(-1)

            return self._normalize(scores)

        except ValueError:
            return np.ones(
                len(sentences),
                dtype=np.float32,
            )

    def _position_scores(
        self,
        sentence_count: int,
    ) -> np.ndarray:
        if sentence_count <= 1:
            return np.ones(
                sentence_count,
                dtype=np.float32,
            )

        positions = np.arange(
            sentence_count,
            dtype=np.float32,
        )

        return 1.0 - 0.5 * (
            positions / (sentence_count - 1)
        )

    def _mamba_scores(
        self,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        embedding_tensor = torch.tensor(
            embeddings,
            dtype=torch.float32,
        ).unsqueeze(0)

        with torch.inference_mode():
            raw_scores = self.model(
                embedding_tensor
            ).squeeze(0).numpy()

        return self._normalize(raw_scores)

    def _mmr_select(
        self,
        relevance_scores: np.ndarray,
        embeddings: np.ndarray,
    ) -> List[int]:
        sentence_count = len(relevance_scores)
        similarity_matrix = cosine_similarity(embeddings)

        selected = []
        remaining = set(range(sentence_count))

        target_count = min(
            self.top_k,
            sentence_count,
        )

        while remaining and len(selected) < target_count:
            best_index = None
            best_score = -float("inf")

            for index in remaining:
                relevance = relevance_scores[index]

                if not selected:
                    redundancy = 0.0
                else:
                    redundancy = max(
                        similarity_matrix[index][chosen_index]
                        for chosen_index in selected
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

        semantic_scores = self._semantic_scores(embeddings)
        keyword_scores = self._keyword_scores(sentences)
        position_scores = self._position_scores(len(sentences))
        mamba_scores = self._mamba_scores(embeddings)

        relevance_scores = (
            0.40 * mamba_scores
            + 0.30 * semantic_scores
            + 0.20 * keyword_scores
            + 0.10 * position_scores
        )

        selected_indices = self._mmr_select(
            relevance_scores,
            embeddings,
        )

        return selected_indices, relevance_scores

    def summarize(self, text: str) -> List[str]:
        sentences = split_sentences(text)

        if not sentences:
            return []

        if len(sentences) <= self.top_k:
            return sentences

        selected_indices, _ = self.rank_sentences(sentences)

        return [
            sentences[index]
            for index in selected_indices
        ]


def summarize_file(
    input_path: str,
    top_k: int = 3,
) -> List[str]:
    text = load_text_from_file(input_path)

    summarizer = MambaTransformerSummarizer(
        top_k=top_k,
    )

    return summarizer.summarize(text)


def main():
    input_path = Path("data/samples/sample1.txt")

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return

    text = load_text_from_file(str(input_path))

    summarizer = MambaTransformerSummarizer(
        top_k=3,
    )

    summary = summarizer.summarize(text)

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    summary_text = "\n".join(summary)

    (output_dir / "mamba_summary.txt").write_text(
        summary_text,
        encoding="utf-8",
    )

    pd.DataFrame(
        {
            "summary_sentence": summary,
        }
    ).to_csv(
        output_dir / "mamba_summary.csv",
        index=False,
    )

    print("\nMamba–Transformer Summary:\n")
    print(summary_text)


if __name__ == "__main__":
    main()