from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "output/benchmark_detailed_results.csv"
)

OUTPUT_PATH = Path(
    "output/benchmark_length_group_results.csv"
)


def main() -> None:
    if not INPUT_PATH.exists():
        print(
            f"Benchmark data not found: {INPUT_PATH}"
        )
        return

    dataframe = pd.read_csv(INPUT_PATH)

    required_columns = [
        "model",
        "document_id",
        "document_length_group",
        "sentence_count",
        "inference_time_seconds",
        "rouge1_f1",
        "rouge2_f1",
        "rougeL_f1",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        print(
            "The benchmark CSV is missing required columns: "
            f"{missing_columns}"
        )
        return

    length_order = [
        "short",
        "medium",
        "long",
    ]

    dataframe["document_length_group"] = pd.Categorical(
        dataframe["document_length_group"],
        categories=length_order,
        ordered=True,
    )

    grouped_results = (
        dataframe.groupby(
            [
                "model",
                "document_length_group",
            ],
            observed=True,
            sort=False,
        )
        .agg(
            document_count=(
                "document_id",
                "count",
            ),
            average_sentence_count=(
                "sentence_count",
                "mean",
            ),
            average_inference_time_seconds=(
                "inference_time_seconds",
                "mean",
            ),
            average_rouge1_f1=(
                "rouge1_f1",
                "mean",
            ),
            average_rouge2_f1=(
                "rouge2_f1",
                "mean",
            ),
            average_rougeL_f1=(
                "rougeL_f1",
                "mean",
            ),
        )
        .reset_index()
    )

    grouped_results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nDocument-Length Results:\n")

    print(
        grouped_results.to_string(
            index=False,
            float_format="%.4f",
        )
    )

    print(
        f"\nSaved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()