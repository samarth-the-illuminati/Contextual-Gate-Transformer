import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from contextual_summarizer import (
    MambaSentenceRanker,
    create_candidate_units,
)


DATASET_PATH = Path(
    "data/training/xsum_train_subset.json"
)

MODEL_PATH = Path(
    "models/contextual_ranker.pt"
)

HISTORY_PATH = Path(
    "output/contextual_ranker_training_history.csv"
)

EMBEDDING_DIM = 384
HIDDEN_DIM = 128

EPOCHS = 15
LEARNING_RATE = 0.001

TOP_POSITIVE_CANDIDATES = 3
VALIDATION_RATIO = 0.20
RANDOM_SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dataset(
    path: Path,
) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def create_top_k_weak_labels(
    document_candidates: List[str],
    reference_summary: str,
    encoder: SentenceTransformer,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates weak extractive labels using the same candidate-unit
    preprocessing used during inference.

    The top candidate units most semantically similar to the human
    reference summary are assigned positive labels.
    """

    reference_candidates = create_candidate_units(
        reference_summary
    )

    document_embeddings = encoder.encode(
        document_candidates,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    if not reference_candidates:
        labels = np.zeros(
            len(document_candidates),
            dtype=np.float32,
        )

        labels[0] = 1.0

        return labels, document_embeddings

    reference_embeddings = encoder.encode(
        reference_candidates,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    similarity_matrix = cosine_similarity(
        document_embeddings,
        reference_embeddings,
    )

    maximum_similarities = similarity_matrix.max(
        axis=1
    )

    positive_count = min(
        TOP_POSITIVE_CANDIDATES,
        len(document_candidates),
    )

    positive_indices = np.argsort(
        maximum_similarities
    )[-positive_count:]

    labels = np.zeros(
        len(document_candidates),
        dtype=np.float32,
    )

    labels[positive_indices] = 1.0

    return labels, document_embeddings


def prepare_examples(
    dataset: List[Dict],
    encoder: SentenceTransformer,
) -> List[Dict]:
    """
    Converts documents into training examples.

    Candidate units are created with create_candidate_units(), matching
    the preprocessing used by the Streamlit app and benchmark scripts.
    """

    examples = []

    for index, item in enumerate(
        dataset,
        start=1,
    ):
        document_candidates = create_candidate_units(
            item["document"]
        )

        if len(document_candidates) < 2:
            continue

        labels, embeddings = create_top_k_weak_labels(
            document_candidates=document_candidates,
            reference_summary=item["summary"],
            encoder=encoder,
        )

        examples.append(
            {
                "id": item["id"],
                "embeddings": embeddings,
                "labels": labels,
                "candidate_count": len(
                    document_candidates
                ),
                "positive_count": int(
                    labels.sum()
                ),
            }
        )

        if index % 25 == 0 or index == len(dataset):
            print(
                f"Prepared {index}/{len(dataset)} documents"
            )

    return examples


def split_examples(
    examples: List[Dict],
) -> Tuple[List[Dict], List[Dict]]:
    shuffled_examples = examples.copy()

    random.shuffle(shuffled_examples)

    validation_size = max(
        1,
        int(
            len(shuffled_examples)
            * VALIDATION_RATIO
        ),
    )

    validation_examples = shuffled_examples[
        :validation_size
    ]

    training_examples = shuffled_examples[
        validation_size:
    ]

    return training_examples, validation_examples


def calculate_positive_weight(
    examples: List[Dict],
) -> torch.Tensor:
    positive_count = sum(
        example["labels"].sum()
        for example in examples
    )

    total_count = sum(
        len(example["labels"])
        for example in examples
    )

    negative_count = total_count - positive_count

    if positive_count <= 0:
        return torch.tensor(
            1.0,
            dtype=torch.float32,
        )

    positive_weight = negative_count / positive_count

    return torch.tensor(
        positive_weight,
        dtype=torch.float32,
    )


def evaluate_model(
    model: MambaSentenceRanker,
    examples: List[Dict],
    loss_function: nn.Module,
) -> float:
    model.eval()

    total_loss = 0.0

    with torch.inference_mode():
        for example in examples:
            embedding_tensor = torch.tensor(
                example["embeddings"],
                dtype=torch.float32,
            ).unsqueeze(0)

            label_tensor = torch.tensor(
                example["labels"],
                dtype=torch.float32,
            ).unsqueeze(0)

            raw_scores = model(
                embedding_tensor
            )

            loss = loss_function(
                raw_scores,
                label_tensor,
            )

            total_loss += loss.item()

    return total_loss / len(examples)


def save_checkpoint(
    model: MambaSentenceRanker,
    epoch: int,
    validation_loss: float,
) -> None:
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "embedding_dim": EMBEDDING_DIM,
            "hidden_dim": HIDDEN_DIM,
            "epoch": epoch,
            "validation_loss": validation_loss,
            "top_positive_candidates": (
                TOP_POSITIVE_CANDIDATES
            ),
            "candidate_preprocessing": {
                "method": "create_candidate_units",
                "maximum_candidate_words": 60,
                "minimum_candidate_words": 4,
            },
        },
        MODEL_PATH,
    )


def train_model(
    training_examples: List[Dict],
    validation_examples: List[Dict],
) -> Tuple[MambaSentenceRanker, List[Dict]]:
    model = MambaSentenceRanker(
        input_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    positive_weight = calculate_positive_weight(
        training_examples
    )

    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=positive_weight
    )

    print(
        f"\nPositive-class weight: "
        f"{positive_weight.item():.3f}"
    )

    best_validation_loss = float("inf")
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()

        total_training_loss = 0.0

        random.shuffle(training_examples)

        for example in training_examples:
            embedding_tensor = torch.tensor(
                example["embeddings"],
                dtype=torch.float32,
            ).unsqueeze(0)

            label_tensor = torch.tensor(
                example["labels"],
                dtype=torch.float32,
            ).unsqueeze(0)

            optimizer.zero_grad()

            raw_scores = model(
                embedding_tensor
            )

            loss = loss_function(
                raw_scores,
                label_tensor,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            total_training_loss += loss.item()

        average_training_loss = (
            total_training_loss
            / len(training_examples)
        )

        validation_loss = evaluate_model(
            model=model,
            examples=validation_examples,
            loss_function=loss_function,
        )

        history.append(
            {
                "epoch": epoch,
                "training_loss": (
                    average_training_loss
                ),
                "validation_loss": validation_loss,
            }
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train loss: {average_training_loss:.4f} | "
            f"Validation loss: {validation_loss:.4f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss

            save_checkpoint(
                model=model,
                epoch=epoch,
                validation_loss=validation_loss,
            )

            print(
                "  Saved new best contextual-ranker "
                "checkpoint."
            )

    return model, history


def main() -> None:
    set_seed(RANDOM_SEED)

    if not DATASET_PATH.exists():
        print(
            f"Training dataset not found: "
            f"{DATASET_PATH}"
        )
        return

    print("Loading MiniLM sentence encoder...")

    encoder = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    encoder.eval()

    print("Loading XSum training subset...")

    dataset = load_dataset(DATASET_PATH)

    print(
        f"Documents loaded: {len(dataset)}"
    )

    print(
        "\nGenerating candidate-unit embeddings and "
        "weak labels..."
    )

    examples = prepare_examples(
        dataset=dataset,
        encoder=encoder,
    )

    if len(examples) < 2:
        print(
            "Not enough valid examples for training."
        )
        return

    training_examples, validation_examples = (
        split_examples(examples)
    )

    print(
        f"\nTraining examples: "
        f"{len(training_examples)}"
    )

    print(
        f"Validation examples: "
        f"{len(validation_examples)}"
    )

    print(
        "\nTraining lightweight contextual ranker "
        "on CPU..."
    )

    _, history = train_model(
        training_examples=training_examples,
        validation_examples=validation_examples,
    )

    HISTORY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_dataframe = pd.DataFrame(history)

    history_dataframe.to_csv(
        HISTORY_PATH,
        index=False,
    )

    checkpoint_size_mb = (
        MODEL_PATH.stat().st_size
        / (1024 * 1024)
    )

    best_epoch = min(
        history,
        key=lambda row: row["validation_loss"],
    )

    print("\nTraining completed successfully.")

    print(
        f"Best validation loss: "
        f"{best_epoch['validation_loss']:.4f} "
        f"at epoch {best_epoch['epoch']}"
    )

    print(
        f"Trained checkpoint: {MODEL_PATH}"
    )

    print(
        f"Checkpoint size: "
        f"{checkpoint_size_mb:.2f} MB"
    )

    print(
        f"Training history: {HISTORY_PATH}"
    )


if __name__ == "__main__":
    main()