import json
import os
import time
import tracemalloc
from pathlib import Path
from typing import Callable, Dict, List, Tuple

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import pandas as pd
import torch
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer

from contextual_summarizer import (
    MambaSentenceRanker,
    extractive_summarize,
    lead3_summarize,
)


EVALUATION_PATH = Path(
    "data/evaluation/test_dataset.json"
)

MODEL_PATH = Path(
    "models/contextual_ranker.pt"
)

OUTPUT_PATH = Path(
    "output/ablation_results_rouge_score.csv"
)

ROUGE_SCORER = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=True,
)


def load_evaluation_data(
    path: Path,
) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def calculate_rouge_scores(
    generated: str,
    reference: str,
) -> Dict[str, float]:
    """
    Uses rouge-score for every ablation variant.

    Reference is supplied first and generated summary second,
    which is the expected rouge-score API order.
    """

    if not generated.strip() or not reference.strip():
        return {
            "rouge1_f1": 0.0,
            "rouge2_f1": 0.0,
            "rougeL_f1": 0.0,
        }

    scores = ROUGE_SCORER.score(
        reference,
        generated,
    )

    return {
        "rouge1_f1": scores["rouge1"].fmeasure,
        "rouge2_f1": scores["rouge2"].fmeasure,
        "rougeL_f1": scores["rougeL"].fmeasure,
    }


def measure_time_and_memory(
    function: Callable,
    *args,
    **kwargs,
) -> Tuple[str, float, float]:
    """
    Measures runtime and Python allocations for one inference call.

    The memory value is an allocation delta from tracemalloc, not
    total operating-system process RAM.
    """

    tracemalloc.start()

    start_time = time.perf_counter()

    result = function(
        *args,
        **kwargs,
    )

    end_time = time.perf_counter()

    current_memory, _ = tracemalloc.get_traced_memory()

    tracemalloc.stop()

    duration_seconds = end_time - start_time
    memory_delta_mb = current_memory / (1024 * 1024)

    return result, duration_seconds, memory_delta_mb


def run_lead3_benchmark(
    examples: List[Dict],
) -> List[Dict]:
    results = []

    for index, example in enumerate(
        examples,
        start=1,
    ):
        summary, duration, memory_delta = (
            measure_time_and_memory(
                lead3_summarize,
                example["document"],
                max_sentences=3,
            )
        )

        rouge_scores = calculate_rouge_scores(
            generated=summary,
            reference=example["summary"],
        )

        results.append(
            {
                "model": "Lead-3 Baseline",
                "document_id": example["id"],
                "time_seconds": duration,
                "python_memory_delta_mb": memory_delta,
                **rouge_scores,
            }
        )

        if index % 5 == 0 or index == len(examples):
            print(
                f"Lead-3: {index}/{len(examples)}"
            )

    return results


def run_ablation_benchmark(
    examples: List[Dict],
    encoder: SentenceTransformer,
    ranker: MambaSentenceRanker,
) -> List[Dict]:
    """
    Runs only configurations supported by the current reproducible
    final implementation.

    The final selected variant contains:
    Mamba-style learned contextual ranking + semantic relevance
    + TF-IDF keyword importance + sentence-position relevance.
    """

    variants = [
        {
            "name": "Contextual ranker only",
            "use_semantic": False,
            "use_keyword": False,
            "use_position": False,
        },
        {
            "name": "Contextual ranker + semantic",
            "use_semantic": True,
            "use_keyword": False,
            "use_position": False,
        },
        {
            "name": (
                "Contextual ranker + semantic + "
                "TF-IDF + position"
            ),
            "use_semantic": True,
            "use_keyword": True,
            "use_position": True,
        },
    ]

    results = []

    for variant in variants:
        print(
            f"\nRunning ablation: {variant['name']}"
        )

        for index, example in enumerate(
            examples,
            start=1,
        ):
            summary, duration, memory_delta = (
                measure_time_and_memory(
                    extractive_summarize,
                    document=example["document"],
                    encoder=encoder,
                    ranker=ranker,
                    max_sentences=3,
                    use_semantic_relevance=(
                        variant["use_semantic"]
                    ),
                    use_keyword_relevance=(
                        variant["use_keyword"]
                    ),
                    use_position_relevance=(
                        variant["use_position"]
                    ),
                    use_entity_relevance=False,
                    use_mmr_diversity=False,
                )
            )

            rouge_scores = calculate_rouge_scores(
                generated=summary,
                reference=example["summary"],
            )

            results.append(
                {
                    "model": f"Proposed ({variant['name']})",
                    "document_id": example["id"],
                    "time_seconds": duration,
                    "python_memory_delta_mb": memory_delta,
                    **rouge_scores,
                }
            )

            if index % 5 == 0 or index == len(examples):
                print(
                    f"{variant['name']}: "
                    f"{index}/{len(examples)}"
                )

    return results


def create_average_table(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    average_results = (
        dataframe.groupby(
            "model",
            sort=False,
        )
        .agg(
            {
                "time_seconds": "mean",
                "python_memory_delta_mb": "mean",
                "rouge1_f1": "mean",
                "rouge2_f1": "mean",
                "rougeL_f1": "mean",
            }
        )
        .reset_index()
    )

    return average_results.rename(
        columns={
            "time_seconds": "avg_time_seconds",
            "python_memory_delta_mb": (
                "avg_python_memory_delta_mb"
            ),
            "rouge1_f1": "avg_rouge1_f1",
            "rouge2_f1": "avg_rouge2_f1",
            "rougeL_f1": "avg_rougeL_f1",
        }
    )


def main() -> None:
    if not EVALUATION_PATH.exists():
        print(
            "Evaluation dataset not found: "
            f"{EVALUATION_PATH}"
        )
        return

    if not MODEL_PATH.exists():
        print(
            "Trained ranking checkpoint not found: "
            f"{MODEL_PATH}"
        )
        return

    print("Loading evaluation data...")

    examples = load_evaluation_data(
        EVALUATION_PATH
    )

    print(
        f"Documents loaded: {len(examples)}"
    )

    if not examples:
        print("Evaluation dataset is empty.")
        return

    print("\nLoading MiniLM encoder...")

    encoder = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    encoder.eval()

    print("Loading trained contextual ranker...")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    ranker = MambaSentenceRanker(
        input_dim=checkpoint["embedding_dim"],
        hidden_dim=checkpoint["hidden_dim"],
    )

    ranker.load_state_dict(
        checkpoint["model_state_dict"]
    )

    ranker.eval()

    print("\nRunning Lead-3 baseline...")

    lead3_results = run_lead3_benchmark(
        examples
    )

    print("\nRunning reproducible ablations...")

    ablation_results = run_ablation_benchmark(
        examples=examples,
        encoder=encoder,
        ranker=ranker,
    )

    all_results = lead3_results + ablation_results

    dataframe = pd.DataFrame(all_results)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    average_results = create_average_table(
        dataframe
    )

    print("\nAblation benchmark completed.")

    print(
        "Detailed results saved to: "
        f"{OUTPUT_PATH}"
    )

    print("\nAverage Results:\n")

    print(
        average_results.to_string(
            index=False,
            float_format="%.4f",
        )
    )


if __name__ == "__main__":
    main()