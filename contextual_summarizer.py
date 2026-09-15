import os
import re

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


MAX_CANDIDATE_WORDS = 60
MIN_CANDIDATE_WORDS = 4


class LightweightSelectiveSSM(nn.Module):
    """
    Lightweight learned contextual gating layer.

    This is a compact recurrent contextual layer for sentence ranking.
    It is not a formal implementation of the published Mamba selective
    state-space architecture.
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
    Backward-compatible class name for the trained learned contextual
    sentence ranker.

    The name is retained so existing training and checkpoint-loading
    scripts do not break. In documentation, call this a learned
    contextual sentence ranker rather than a formal Mamba model.
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
    Min-max normalizes a vector of sentence scores.
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


def clean_candidate_text(
    text: str,
) -> str:
    """
    Removes citation markers and normalizes whitespace while preserving
    the original extractive content as much as possible.
    """

    text = re.sub(
        r"\[[^\]]{1,30}\]",
        "",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def split_long_sentence(
    sentence: str,
    maximum_words: int = MAX_CANDIDATE_WORDS,
) -> List[str]:
    """
    Splits abnormally long sentence units into smaller candidate chunks.

    It first attempts to split at semicolons, colons, em dashes, and
    sentence-ending punctuation. If a piece remains long, it creates
    fixed-size word chunks. This preserves extractive behavior while
    preventing one very long unit from consuming an entire summary.
    """

    cleaned_sentence = clean_candidate_text(
        sentence
    )

    if not cleaned_sentence:
        return []

    words = cleaned_sentence.split()

    if len(words) <= maximum_words:
        return [cleaned_sentence]

    pieces = re.split(
        r"(?<=[.;:!?])\s+|(?<=—)\s+",
        cleaned_sentence,
    )

    output_chunks = []

    for piece in pieces:
        piece = piece.strip()

        if not piece:
            continue

        piece_words = piece.split()

        if len(piece_words) <= maximum_words:
            output_chunks.append(piece)
            continue

        for start in range(
            0,
            len(piece_words),
            maximum_words,
        ):
            chunk_words = piece_words[
                start:start + maximum_words
            ]

            chunk = " ".join(chunk_words).strip()

            if chunk:
                output_chunks.append(chunk)

    return output_chunks


def create_candidate_units(
    text: str,
    maximum_words: int = MAX_CANDIDATE_WORDS,
    minimum_words: int = MIN_CANDIDATE_WORDS,
) -> List[str]:
    """
    Creates clean, bounded-length extractive candidate units.

    Normal sentences remain unchanged except for citation/whitespace
    cleanup. Long units are split into smaller chunks. Extremely short
    fragments are retained only if filtering would otherwise leave no
    candidates.
    """

    raw_sentences = split_sentences(text)

    all_candidates = []

    for sentence in raw_sentences:
        all_candidates.extend(
            split_long_sentence(
                sentence,
                maximum_words=maximum_words,
            )
        )

    if not all_candidates:
        return []

    filtered_candidates = [
        candidate
        for candidate in all_candidates
        if len(candidate.split()) >= minimum_words
    ]

    if filtered_candidates:
        return filtered_candidates

    return all_candidates


def calculate_semantic_scores(
    embeddings: np.ndarray,
) -> np.ndarray:
    """
    Scores each candidate by cosine similarity to the mean embedding
    for the full document.
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
    candidates: List[str],
) -> np.ndarray:
    """
    Calculates TF-IDF keyword importance for each candidate unit.
    """

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            lowercase=True,
        )

        matrix = vectorizer.fit_transform(
            candidates
        )

        scores = np.asarray(
            matrix.sum(axis=1)
        ).reshape(-1)

        return normalize_scores(scores)

    except ValueError:
        return np.ones(
            len(candidates),
            dtype=np.float32,
        )


def calculate_position_scores(
    candidate_count: int,
) -> np.ndarray:
    """
    Gives a small relevance preference to early candidate units.

    First candidate: 1.0
    Last candidate: 0.5
    """

    if candidate_count <= 1:
        return np.ones(
            candidate_count,
            dtype=np.float32,
        )

    positions = np.arange(
        candidate_count,
        dtype=np.float32,
    )

    return 1.0 - 0.5 * (
        positions / (candidate_count - 1)
    )


def calculate_contextual_scores(
    embeddings: np.ndarray,
    ranker: MambaSentenceRanker,
) -> np.ndarray:
    """
    Gets learned contextual importance scores from the trained ranker.
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


def select_top_k_candidates(
    relevance_scores: np.ndarray,
    top_k: int,
) -> List[int]:
    """
    Selects the top-scoring candidates and restores original candidate
    order for a readable extractive summary.
    """

    candidate_count = len(relevance_scores)

    selected_count = min(
        top_k,
        candidate_count,
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
    Lead-3 baseline using bounded candidate units.

    It is intended for the Streamlit demo. The benchmark's original
    Lead-3 baseline should remain unchanged unless all models are
    rerun under this preprocessing policy.
    """

    candidates = create_candidate_units(document)

    return " ".join(
        candidates[:max_sentences]
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
    Runs the learned contextual-ranking hybrid.

    The reproducible final ranking configuration uses:
    - learned contextual score
    - semantic relevance
    - TF-IDF keyword importance
    - sentence-position relevance
    - direct top-k selection

    Entity and MMR flags are retained only for compatibility with
    benchmarking scripts. They are not active in this implementation.
    """

    del use_entity_relevance
    del use_mmr_diversity

    candidates = create_candidate_units(document)

    if not candidates:
        return ""

    if len(candidates) <= max_sentences:
        return " ".join(candidates)

    embeddings = encoder.encode(
        candidates,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    contextual_scores = calculate_contextual_scores(
        embeddings=embeddings,
        ranker=ranker,
    )

    if use_semantic_relevance:
        semantic_scores = calculate_semantic_scores(
            embeddings
        )
    else:
        semantic_scores = np.ones(
            len(candidates),
            dtype=np.float32,
        )

    if use_keyword_relevance:
        keyword_scores = calculate_keyword_scores(
            candidates
        )
    else:
        keyword_scores = np.ones(
            len(candidates),
            dtype=np.float32,
        )

    if use_position_relevance:
        position_scores = calculate_position_scores(
            len(candidates)
        )
    else:
        position_scores = np.ones(
            len(candidates),
            dtype=np.float32,
        )

    relevance_scores = (
        0.35 * contextual_scores
        + 0.25 * semantic_scores
        + 0.15 * keyword_scores
        + 0.10 * position_scores
    )

    selected_indices = select_top_k_candidates(
        relevance_scores=relevance_scores,
        top_k=max_sentences,
    )

    return " ".join(
        candidates[index]
        for index in selected_indices
    )


class MambaTransformerSummarizer:
    """
    Backward-compatible public summarizer class.

    The implementation is a lightweight hybrid with:
    - MiniLM sentence embeddings
    - learned contextual sentence gating/ranking
    - semantic relevance
    - TF-IDF keyword importance
    - sentence-position relevance

    It is not presented as a formal published Mamba implementation.
    """

    def __init__(
        self,
        top_k: int = 3,
        transformer_name: str = (
            "sentence-transformers/all-MiniLM-L6-v2"
        ),
        hidden_dim: int = 128,
        model_path: str = "models/contextual_ranker.pt",
    ):
        self.top_k = top_k

        self.encoder = SentenceTransformer(
            transformer_name
        )

        self.encoder.eval()

        checkpoint_path = Path(model_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                "Trained contextual-ranking checkpoint not found: "
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
            "Loaded trained contextual ranker from: "
            f"{checkpoint_path}"
        )

    def rank_sentences(
        self,
        candidates: List[str],
    ) -> Tuple[List[int], Dict[str, np.ndarray]]:
        """
        Scores candidate units with the final four-factor ranking
        configuration.
        """

        embeddings = self.encoder.encode(
            candidates,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        contextual_scores = calculate_contextual_scores(
            embeddings=embeddings,
            ranker=self.model,
        )

        semantic_scores = calculate_semantic_scores(
            embeddings
        )

        keyword_scores = calculate_keyword_scores(
            candidates
        )

        position_scores = calculate_position_scores(
            len(candidates)
        )

        combined_scores = (
            0.35 * contextual_scores
            + 0.25 * semantic_scores
            + 0.15 * keyword_scores
            + 0.10 * position_scores
        )

        selected_indices = select_top_k_candidates(
            relevance_scores=combined_scores,
            top_k=self.top_k,
        )

        score_details = {
            "contextual_score": contextual_scores,
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
        Produces an extractive summary and a detailed ranking table.
        """

        candidates = create_candidate_units(text)

        if not candidates:
            return [], pd.DataFrame()

        if len(candidates) <= self.top_k:
            result = pd.DataFrame(
                {
                    "candidate_index": range(
                        len(candidates)
                    ),
                    "candidate_text": candidates,
                    "selected": [True] * len(candidates),
                }
            )

            return candidates, result

        selected_indices, score_details = self.rank_sentences(
            candidates
        )

        selected_set = set(selected_indices)

        result = pd.DataFrame(
            {
                "candidate_index": range(len(candidates)),
                "candidate_text": candidates,
                "word_count": [
                    len(candidate.split())
                    for candidate in candidates
                ],
                "contextual_score": score_details[
                    "contextual_score"
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
                    for index in range(len(candidates))
                ],
            }
        )

        summary = [
            candidates[index]
            for index in selected_indices
        ]

        return summary, result


def summarize_file(
    input_path: str,
    top_k: int = 3,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Loads a TXT or PDF file using the project's text loader and runs
    the final learned contextual-ranking summarizer.
    """

    text = load_text_from_file(input_path)

    summarizer = MambaTransformerSummarizer(
        top_k=top_k,
    )

    return summarizer.summarize(text)


def main() -> None:
    """
    Command-line test using the sample text file.
    """

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

    (output_dir / "contextual_summary.txt").write_text(
        summary_text,
        encoding="utf-8",
    )

    score_details.to_csv(
        output_dir / "contextual_sentence_scores.csv",
        index=False,
    )

    print("\nGenerated Extractive Summary:\n")

    print(summary_text)

    print(
        "\nDetailed sentence-ranking scores saved to: "
        "output/contextual_sentence_scores.csv"
    )


if __name__ == "__main__":
    main()