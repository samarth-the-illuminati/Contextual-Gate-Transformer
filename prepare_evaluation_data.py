from pathlib import Path
import json

from datasets import load_dataset


OUTPUT_DIR = Path("data/evaluation")
OUTPUT_FILE = OUTPUT_DIR / "test_dataset.json"

DATASET_NAME = "EdinburghNLP/xsum"
SPLIT = "test"
NUMBER_OF_EXAMPLES = 30
SEED = 42


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset: {DATASET_NAME}")
    print(f"Split: {SPLIT}")
    print(f"Examples to save: {NUMBER_OF_EXAMPLES}")

    dataset = load_dataset(
        DATASET_NAME,
        split=SPLIT,
    )

    dataset = dataset.shuffle(seed=SEED)
    subset = dataset.select(
        range(NUMBER_OF_EXAMPLES)
    )

    evaluation_examples = []

    for item in subset:
        document = item["document"].strip()
        summary = item["summary"].strip()

        if not document or not summary:
            continue

        evaluation_examples.append(
            {
                "id": item["id"],
                "document": document,
                "summary": summary,
            }
        )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            evaluation_examples,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nEvaluation data saved successfully.")
    print(f"File: {OUTPUT_FILE}")
    print(f"Valid examples: {len(evaluation_examples)}")

    print("\nFirst example preview:")
    first = evaluation_examples[0]

    print(f"ID: {first['id']}")
    print(f"Document: {first['document'][:300]}...")
    print(f"Reference summary: {first['summary']}")


if __name__ == "__main__":
    main()