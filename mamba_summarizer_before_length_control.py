import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from pathlib import Path
from typing import Dict, List, Tuple

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
    Lightweight selective state-space sequence module.

    It processes the sequence of sentence embeddings and produces
    contextual sentence representations used by the ranking layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.input_projection = nn.Linear(
            input_dim,
            hidden_dim,
        )

        self.state_candidate = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        self.update_gate = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        self.output_projection = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        self.norm = nn.LayerNorm(
            hidden_dim
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        projected = torch.tanh(
            self.input_projection(x)
        )

        batch_size, sequence_length, hidden_dim = (
            projected.shape
        )

        state = torch.zeros(
            batch_size,
            hidden_dim,
            dtype=projected.dtype,
            device=projected.device,
        )

        outputs = []

        for time_step in range(sequence_length):
            current_input = projected[
                :,
                time_step,
                :,
            ]

            candidate = torch.tanh(
                self.state_candidate(current_input)
            )

            gate = torch.sigmoid(
                self.update_gate(current_input)
            )

            state = (
                gate * candidate
                + (1.0 - gate) * state
            )

            outputs.append(
                state.unsqueeze(1)
            )

        sequence_output = torch.cat(
            outputs,
            dim=1,
        )

        sequence_output = self.output_projection(
            sequence_output
        )

        return self.norm(
            sequence_output + projected
        )


class MambaSentenceRanker(nn.Module):
    """
    Scores each sentence after contextual processing by the
    lightweight selective state-space module.
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

    def forward(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        contextual_embeddings = self.ssm(
            embeddings
        )

        raw_scores = self.scorer(
            contextual_embeddings
        ).squeeze(-1)

        return raw_scores


def normalize_scores(
    values: np.ndarray,
) -> np.ndarray:
    """
    Applies min-max normalization to sentence-level scores.
    """

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    if len(values) == 0:
        return values

    minimum = values.min()
    maximum = values.max()

    if maximum - minimum < 1e-8:
        return np.ones_like(values)

    return (
        values - minimum
    ) / (
        maximum - minimum
    )


def calculate_semantic_scores(
    embeddings: np.ndarray,
) -> np.ndarray:
    """
    Measures each sentence's cosine similarity with the mean
    embedding of the document.
    """

    document_embedding = embeddings.mean(
        axis=0,
        keepdims=True,
    )

    scores = cosine_similarity(
        embeddings,
        document_embedding,
    ).reshape(-1)

    return normalize_scores(scores)


def calculate_keyword_scores(
    sentences: List[str],
) -> np.ndarray:
    """
    Computes a TF-IDF-based importance score per sentence.
    """

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            lowercase=True,
        )

        matrix = vectorizer.fit_transform(
            sentences
        )

        scores = np.asarray(
            matrix.sum(axis=1)
        ).reshape(-1)

        return normalize_scores(scores)

    except ValueError:
        return np.ones(
            len(sentences),
            dtype=np.float32,
        )


def calculate_position_scores(
    sentence_count: int,
) -> np.ndarray:
    """
    Gives a gentle preference to earlier sentences.

    The first sentence receives 1.0 and the last sentence receives
    0.5. This is intentionally a weak prior rather than a strict
    Lead-3 rule.
    """

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


def calculate_mamba_scores(
    embeddings: np.ndarray,
    ranker: MambaSentenceRanker,
) -> np.ndarray:
    """
    Obtains normalized contextual Mamba ranking scores.
    """

    embedding_tensor = torch.tensor(
        embeddings,
        dtype=torch.float32,
    ).unsqueeze(0)

    with torch.inference_mode():
        raw_scores = ranker(
            embedding_tensor
        ).squeeze(0).numpy()

    return normalize_scores(raw_scores)


def select_top_k_sentences(
    relevance_scores: np.ndarray,
    top_k: int,
) -> List[int]:
    """
    Selects the highest-scoring sentences and returns their original
    document order for a readable extractive summary.
    """

    sentence_count = len(relevance_scores)

    selected_count = min(
        top_k,
        sentence_count,
    )

    selected_indices = np.argsort(
        relevance_scores
    )[-selected_count:]

    return sorted(
        int(index)
        for index in selected_indices
    )


def lead3_summarize(
    document: str,
    max_sentences: int = 3,
) -> str:
    """
    Lead-3 baseline: returns the first max_sentences
    from the document.
    """

    sentences = split_sentences(document)

    return " ".join(
        sentences[:max_sentences]
    )


def extractive_summarize(
    document: str,
    encoder: SentenceTransformer,
    ranker: MambaSentenceRanker,
    max_sentences: int = 3,
    use_semantic_relevance: bool = True,
    use_keyword_relevance: bool = True,
    use_position_relevance: bool = True,
    use_entity_relevance: bool = False,
    use_mmr_diversity: bool = False,
) -> str:
    """
    Produces an extractive summary.

    The default configuration is the selected final configuration:

    - Mamba score
    - MiniLM semantic relevance
    - TF-IDF keyword importance
    - sentence-position relevance
    - top-k sentence selection

    `use_entity_relevance` and `use_mmr_diversity` remain accepted
    for compatibility with older ablation scripts, but are ignored in
    this lightweight final implementation.
    """

    del use_entity_relevance
    del use_mmr_diversity

    sentences = split_sentences(document)

    if not sentences:
        return ""

    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    embeddings = encoder.encode(
        sentences,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    mamba_scores = calculate_mamba_scores(
        embeddings=embeddings,
        ranker=ranker,
    )

    if use_semantic_relevance:
        semantic_scores = calculate_semantic_scores(
            embeddings
        )
    else:
        semantic_scores = np.ones(
            len(sentences),
            dtype=np.float32,
        )

    if use_keyword_relevance:
        keyword_scores = calculate_keyword_scores(
            sentences
        )
    else:
        keyword_scores = np.ones(
            len(sentences),
            dtype=np.float32,
        )

    if use_position_relevance:
        position_scores = calculate_position_scores(
            len(sentences)
        )
    else:
        position_scores = np.ones(
            len(sentences),
            dtype=np.float32,
        )

    relevance_scores = (
        0.35 * mamba_scores
        + 0.25 * semantic_scores
        + 0.15 * keyword_scores
        + 0.10 * position_scores
    )

    selected_indices = select_top_k_sentences(
        relevance_scores=relevance_scores,
        top_k=max_sentences,
    )

    return " ".join(
        sentences[index]
        for index in selected_indices
    )


class MambaTransformerSummarizer:
    """
    Final lightweight CPU-friendly extractive summarizer.

    Default ranking configuration:
    - trained Mamba contextual score
    - MiniLM semantic relevance
    - TF-IDF keyword score
    - sentence position score
    - top-k extraction without MMR
    """

    def __init__(
        self,
        top_k: int = 3,
        transformer_name: str = (
            "sentence-transformers/all-MiniLM-L6-v2"
        ),
        hidden_dim: int = 128,
        model_path: str = "models/mamba_ranker.pt",
    ):
        self.top_k = top_k

        self.encoder = SentenceTransformer(
            transformer_name
        )

        self.encoder.eval()

        checkpoint_path = Path(model_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                "Trained Mamba ranker checkpoint not found: "
                f"{checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        input_dim = checkpoint.get(
            "embedding_dim",
            384,
        )

        checkpoint_hidden_dim = checkpoint.get(
            "hidden_dim",
            hidden_dim,
        )

        self.model = MambaSentenceRanker(
            input_dim=input_dim,
            hidden_dim=checkpoint_hidden_dim,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

        print(
            "Loaded trained Mamba ranker from: "
            f"{checkpoint_path}"
        )

    def rank_sentences(
        self,
        sentences: List[str],
    ) -> Tuple[List[int], Dict[str, np.ndarray]]:
        """
        Computes individual score components and selects the final
        top-k sentences.
        """

        embeddings = self.encoder.encode(
            sentences,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        mamba_scores = calculate_mamba_scores(
            embeddings=embeddings,
            ranker=self.model,
        )

        semantic_scores = calculate_semantic_scores(
            embeddings
        )

        keyword_scores = calculate_keyword_scores(
            sentences
        )

        position_scores = calculate_position_scores(
            len(sentences)
        )

        combined_scores = (
            0.35 * mamba_scores
            + 0.25 * semantic_scores
            + 0.15 * keyword_scores
            + 0.10 * position_scores
        )

        selected_indices = select_top_k_sentences(
            relevance_scores=combined_scores,
            top_k=self.top_k,
        )

        score_details = {
            "mamba_score": mamba_scores,
            "semantic_score": semantic_scores,
            "keyword_score": keyword_scores,
            "position_score": position_scores,
            "combined_score": combined_scores,
        }

        return selected_indices, score_details

    def summarize(
        self,
        text: str,
    ) -> Tuple[List[str], pd.DataFrame]:
        """
        Returns selected sentences and a detailed score table.
        """

        sentences = split_sentences(text)

        if not sentences:
            return [], pd.DataFrame()

        if len(sentences) <= self.top_k:
            result = pd.DataFrame(
                {
                    "sentence_index": range(
                        len(sentences)
                    ),
                    "sentence": sentences,
                    "selected": [True] * len(sentences),
                }
            )

            return sentences, result

        selected_indices, score_details = (
            self.rank_sentences(sentences)
        )

        selected_set = set(selected_indices)

        result = pd.DataFrame(
            {
                "sentence_index": range(len(sentences)),
                "sentence": sentences,
                "mamba_score": score_details[
                    "mamba_score"
                ],
                "semantic_score": score_details[
                    "semantic_score"
                ],
                "keyword_score": score_details[
                    "keyword_score"
                ],
                "position_score": score_details[
                    "position_score"
                ],
                "combined_score": score_details[
                    "combined_score"
                ],
                "selected": [
                    index in selected_set
                    for index in range(len(sentences))
                ],
            }
        )

        summary = [
            sentences[index]
            for index in selected_indices
        ]

        return summary, result


def summarize_file(
    input_path: str,
    top_k: int = 3,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Loads a supported text document and summarizes it.
    """

    text = load_text_from_file(input_path)

    summarizer = MambaTransformerSummarizer(
        top_k=top_k,
    )

    return summarizer.summarize(text)


def main() -> None:
    input_path = Path(
        "data/samples/sample1.txt"
    )

    if not input_path.exists():
        print(
            f"Input file not found: {input_path}"
        )
        return

    text = load_text_from_file(
        str(input_path)
    )

    summarizer = MambaTransformerSummarizer(
        top_k=3,
    )

    summary, score_details = summarizer.summarize(
        text
    )

    output_dir = Path("output")

    output_dir.mkdir(
        exist_ok=True
    )

    summary_text = "\n".join(summary)

    (output_dir / "mamba_summary.txt").write_text(
        summary_text,
        encoding="utf-8",
    )

    score_details.to_csv(
        output_dir / "mamba_sentence_scores.csv",
        index=False,
    )

    print("\nFinal LiteMambaSum Summary:\n")

    print(summary_text)

    print(
        "\nDetailed ranking scores saved to: "
        "output/mamba_sentence_scores.csv"
    )


if __name__ == "__main__":
    main()