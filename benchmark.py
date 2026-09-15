import gc
import json
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import pandas as pd
import psutil
from rouge_score import rouge_scorer

from baseline_tfidf import OfflineSummarizer
from semantic_mmr_baseline import SemanticMMRSummarizer
from contextual_summarizer import MambaTransformerSummarizer
from utils.sentence_splitter import split_sentences


EVALUATION_FILE = Path("data/evaluation/test_dataset.json")
OUTPUT_DIR = Path("output")

NUMBER_OF_DOCUMENTS = 30
SUMMARY_SENTENCES = 3
WARMUP_DOCUMENTS = 1

PROCESS = psutil.Process(os.getpid())

ROUGE_SCORER = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=True,
)


def load_evaluation_data() -> List[Dict]:
    with open(EVALUATION_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data[:NUMBER_OF_DOCUMENTS]


def get_rss_memory_mb() -> float:
    return PROCESS.memory_info().rss / (1024 * 1024)


def calculate_rouge(
    prediction: str,
    reference: str,
) -> Dict[str, float]:
    scores = ROUGE_SCORER.score(
        reference,
        prediction,
    )

    return {
        "rouge1_f1": scores["rouge1"].fmeasure,
        "rouge2_f1": scores["rouge2"].fmeasure,
        "rougeL_f1": scores["rougeL"].fmeasure,
    }


def get_file_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0

    return path.stat().st_size / (1024 * 1024)


def get_folder_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0

    total_bytes = sum(
        file.stat().st_size
        for file in path.rglob("*")
        if file.is_file()
    )

    return total_bytes / (1024 * 1024)


def classify_document_length(
    sentence_count: int,
) -> str:
    if sentence_count <= 10:
        return "short"

    if sentence_count <= 25:
        return "medium"

    return "long"


def run_tfidf(
    model: OfflineSummarizer,
    document: str,
) -> Tuple[List[str], pd.DataFrame]:
    summary = model.summarize(document)
    return summary, pd.DataFrame()


def run_semantic(
    model: SemanticMMRSummarizer,
    document: str,
) -> Tuple[List[str], pd.DataFrame]:
    summary = model.summarize(document)
    return summary, pd.DataFrame()


def run_mamba(
    model: MambaTransformerSummarizer,
    document: str,
) -> Tuple[List[str], pd.DataFrame]:
    return model.summarize(document)


def benchmark_model(
    model_name: str,
    model: object,
    runner: Callable,
    evaluation_data: List[Dict],
    model_size_mb: float,
) -> Tuple[List[Dict], List[Dict]]:
    print(f"\nBenchmarking: {model_name}")

    benchmark_rows = []
    summary_rows = []

    for index, item in enumerate(evaluation_data, start=1):
        document = item["document"]
        reference = item["summary"]

        sentence_count = len(
            split_sentences(document)
        )

        gc.collect()

        memory_before_mb = get_rss_memory_mb()

        start_time = time.perf_counter()

        summary_sentences, _ = runner(
            model,
            document,
        )

        elapsed_seconds = (
            time.perf_counter() - start_time
        )

        memory_after_mb = get_rss_memory_mb()
        memory_delta_mb = (
            memory_after_mb - memory_before_mb
        )

        generated_summary = " ".join(
            summary_sentences
        )

        rouge_values = calculate_rouge(
            prediction=generated_summary,
            reference=reference,
        )

        row = {
            "model": model_name,
            "document_id": item["id"],
            "document_number": index,
            "document_length_group": classify_document_length(
                sentence_count
            ),
            "sentence_count": sentence_count,
            "reference_word_count": len(
                reference.split()
            ),
            "generated_word_count": len(
                generated_summary.split()
            ),
            "inference_time_seconds": elapsed_seconds,
            "ram_before_mb": memory_before_mb,
            "ram_after_mb": memory_after_mb,
            "ram_delta_mb": memory_delta_mb,
            "model_size_mb": model_size_mb,
            **rouge_values,
        }

        benchmark_rows.append(row)

        summary_rows.append(
            {
                "model": model_name,
                "document_id": item["id"],
                "document_number": index,
                "reference_summary": reference,
                "generated_summary": generated_summary,
            }
        )

        print(
            f"[{index:02d}/{len(evaluation_data)}] "
            f"{item['id']} | "
            f"{sentence_count} sentences | "
            f"{elapsed_seconds:.3f}s | "
            f"ROUGE-L: {rouge_values['rougeL_f1']:.3f}"
        )

    return benchmark_rows, summary_rows


def create_aggregate_results(
    benchmark_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    metrics = [
        "inference_time_seconds",
        "ram_after_mb",
        "ram_delta_mb",
        "model_size_mb",
        "rouge1_f1",
        "rouge2_f1",
        "rougeL_f1",
    ]

    aggregate = (
        benchmark_dataframe
        .groupby("model")[metrics]
        .mean()
        .reset_index()
    )

    aggregate = aggregate.rename(
        columns={
            "inference_time_seconds": "avg_time_seconds",
            "ram_after_mb": "avg_ram_after_mb",
            "ram_delta_mb": "avg_ram_delta_mb",
            "model_size_mb": "model_size_mb",
            "rouge1_f1": "avg_rouge1_f1",
            "rouge2_f1": "avg_rouge2_f1",
            "rougeL_f1": "avg_rougeL_f1",
        }
    )

    return aggregate


def main():
    if not EVALUATION_FILE.exists():
        print(
            f"Evaluation file not found: "
            f"{EVALUATION_FILE}"
        )
        return

    OUTPUT_DIR.mkdir(exist_ok=True)

    evaluation_data = load_evaluation_data()

    if not evaluation_data:
        print("No evaluation examples found.")
        return

    print(
        f"Loaded {len(evaluation_data)} "
        f"evaluation documents."
    )

    print("\nLoading models once before benchmarking...")

    tfidf_model = OfflineSummarizer(
        top_k=SUMMARY_SENTENCES
    )

    semantic_model = SemanticMMRSummarizer(
        top_k=SUMMARY_SENTENCES
    )

    mamba_model = MambaTransformerSummarizer(
        top_k=SUMMARY_SENTENCES
    )

    print("\nPerforming warm-up inference...")

    warmup_document = evaluation_data[
        :WARMUP_DOCUMENTS
    ][0]["document"]

    run_tfidf(tfidf_model, warmup_document)
    run_semantic(semantic_model, warmup_document)
    run_mamba(mamba_model, warmup_document)

    checkpoint_size_mb = get_file_size_mb(
        Path("models/contextual_ranker.pt")
    )

    minilm_cache_size_mb = get_folder_size_mb(
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--sentence-transformers--all-MiniLM-L6-v2"
    )

    mamba_total_size_mb = (
        minilm_cache_size_mb + checkpoint_size_mb
    )

    print(
        f"\nMiniLM cached model size: "
        f"{minilm_cache_size_mb:.2f} MB"
    )

    print(
        f"Mamba ranking checkpoint size: "
        f"{checkpoint_size_mb:.2f} MB"
    )

    all_benchmark_rows = []
    all_summary_rows = []

    configurations = [
        (
            "TF-IDF Baseline",
            tfidf_model,
            run_tfidf,
            0.0,
        ),
        (
            "MiniLM + MMR Baseline",
            semantic_model,
            run_semantic,
            minilm_cache_size_mb,
        ),
        (
            "Proposed contextual Ranking Hybrid",
            mamba_model,
            run_mamba,
            mamba_total_size_mb,
        ),
    ]

    for model_name, model, runner, size_mb in configurations:
        benchmark_rows, summary_rows = benchmark_model(
            model_name=model_name,
            model=model,
            runner=runner,
            evaluation_data=evaluation_data,
            model_size_mb=size_mb,
        )

        all_benchmark_rows.extend(benchmark_rows)
        all_summary_rows.extend(summary_rows)

    benchmark_dataframe = pd.DataFrame(
        all_benchmark_rows
    )

    summary_dataframe = pd.DataFrame(
        all_summary_rows
    )

    aggregate_dataframe = create_aggregate_results(
        benchmark_dataframe
    )

    detailed_path = (
        OUTPUT_DIR / "benchmark_detailed_results.csv"
    )

    summary_path = (
        OUTPUT_DIR / "benchmark_generated_summaries.csv"
    )

    aggregate_path = (
        OUTPUT_DIR / "benchmark_average_results.csv"
    )

    benchmark_dataframe.to_csv(
        detailed_path,
        index=False,
    )

    summary_dataframe.to_csv(
        summary_path,
        index=False,
    )

    aggregate_dataframe.to_csv(
        aggregate_path,
        index=False,
    )

    print("\nBenchmark completed successfully.")

    print(
        f"Detailed metrics: {detailed_path}"
    )

    print(
        f"Generated summaries: {summary_path}"
    )

    print(
        f"Average results: {aggregate_path}"
    )

    print("\nAverage Results:\n")
    print(
        aggregate_dataframe.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )


if __name__ == "__main__":
    main()