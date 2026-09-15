import json
from pathlib import Path

from datasets import load_dataset


TRAINING_OUTPUT = Path(
    "data/training/xsum_train_subset.json"
)

EVALUATION_FILE = Path(
    "data/evaluation/test_dataset.json"
)

DATASET_NAME = "EdinburghNLP/xsum"
TRAIN_SPLIT = "train"

NUMBER_OF_TRAINING_EXAMPLES = 300
SEED = 123


def load_evaluation_ids() -> set:
    if not EVALUATION_FILE.exists():
        return set()

    with open(
        EVALUATION_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        evaluation_data = json.load(file)

    return {
        str(item["id"])
        for item in evaluation_data
    }


def main():
    TRAINING_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluation_ids = load_evaluation_ids()

    print(f"Loading dataset: {DATASET_NAME}")
    print(f"Split: {TRAIN_SPLIT}")
    print(
        f"Requested training examples: "
        f"{NUMBER_OF_TRAINING_EXAMPLES}"
    )
    print(
        f"Evaluation IDs excluded: "
        f"{len(evaluation_ids)}"
    )

    dataset = load_dataset(
        DATASET_NAME,
        split=TRAIN_SPLIT,
    )

    dataset = dataset.shuffle(seed=SEED)

    training_examples = []

    for item in dataset:
        document_id = str(item["id"])
        document = item["document"].strip()
        summary = item["summary"].strip()

        if document_id in evaluation_ids:
            continue

        if not document or not summary:
            continue

        training_examples.append(
            {
                "id": document_id,
                "document": document,
                "summary": summary,
            }
        )

        if len(training_examples) >= NUMBER_OF_TRAINING_EXAMPLES:
            break

    with open(
        TRAINING_OUTPUT,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            training_examples,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nTraining data saved successfully.")
    print(f"File: {TRAINING_OUTPUT}")
    print(
        f"Valid training examples: "
        f"{len(training_examples)}"
    )

    first = training_examples[0]

    print("\nFirst example preview:")
    print(f"ID: {first['id']}")
    print(f"Document: {first['document'][:300]}...")
    print(
        f"Reference summary: "
        f"{first['summary']}"
    )


if __name__ == "__main__":
    main()