#!/usr/bin/env python3
"""Train SetFit multi-label classifier from labeled samples.

Input: data/labeled-samples.json (from label-chunks.py)
Output: data/models/setfit-tagger/ (saved model)

Takes ~5 min on M2 Mac.
"""

import json
import sys
from pathlib import Path
from collections import Counter

DATA_FILE = Path(__file__).parent.parent / "data" / "labeled-samples.json"
MODEL_DIR = Path(__file__).parent.parent / "data" / "models" / "setfit-tagger"
TAXONOMY_FILE = Path(__file__).parent.parent / "src" / "zikaron" / "taxonomy.json"


def load_taxonomy() -> list[str]:
    with open(TAXONOMY_FILE) as f:
        data = json.load(f)
    labels = []
    for cat, info in data["categories"].items():
        for label in info["labels"]:
            labels.append(label)
    return sorted(labels)


def main():
    if not DATA_FILE.exists():
        print(f"No labeled samples found: {DATA_FILE}")
        print("Run: python scripts/pre-label.py && python scripts/label-chunks.py")
        sys.exit(1)

    with open(DATA_FILE) as f:
        samples = json.load(f)

    # Filter to reviewed samples with labels
    reviewed = [s for s in samples if s.get("reviewed") and s.get("human_labels")]
    print(f"Reviewed samples with labels: {len(reviewed)}/{len(samples)}")

    if len(reviewed) < 20:
        print(f"Need at least 20 labeled samples, have {len(reviewed)}")
        sys.exit(1)

    # Load taxonomy for label ordering
    all_labels = load_taxonomy()
    label_to_idx = {l: i for i, l in enumerate(all_labels)}
    print(f"Taxonomy: {len(all_labels)} labels")

    # Count label distribution
    label_counts = Counter()
    for s in reviewed:
        for label in s["human_labels"]:
            label_counts[label] += 1

    print("\nLabel distribution:")
    for label, count in label_counts.most_common(20):
        print(f"  {label}: {count}")

    # Prepare training data
    texts = [s["content"][:512] for s in reviewed]  # Truncate for model

    # Multi-label: each sample has a set of labels
    # SetFit expects list of label sets
    label_sets = []
    for s in reviewed:
        label_set = set()
        for label in s["human_labels"]:
            if label in label_to_idx:
                label_set.add(label)
        label_sets.append(label_set)

    print(f"\nTraining on {len(texts)} samples...")

    # Import SetFit
    try:
        from setfit import SetFitModel, Trainer, TrainingArguments
        from datasets import Dataset
    except ImportError:
        print("Missing deps: pip install setfit datasets")
        sys.exit(1)

    # For multi-label, we train one model per label (one-vs-rest)
    # Or use SetFit's multi-label support
    # SetFit v1.x supports multi-label natively

    # Create binary labels for each active label
    active_labels = [l for l, c in label_counts.items() if c >= 3]  # Need at least 3 examples
    print(f"Training on {len(active_labels)} active labels (3+ examples each)")

    # Prepare dataset with multi-label encoding
    # SetFit expects a "label" column with list of binary values per sample
    encoded_labels = []
    for ls in label_sets:
        encoded_labels.append([1 if label in ls else 0 for label in active_labels])

    ds_dict = {"text": texts, "label": encoded_labels}
    dataset = Dataset.from_dict(ds_dict)

    # Split train/eval (80/20)
    split = dataset.train_test_split(test_size=0.2, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]

    print(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    # Train SetFit model
    model = SetFitModel.from_pretrained(
        "BAAI/bge-small-en-v1.5",
        multi_target_strategy="one-vs-rest",
    )

    args = TrainingArguments(
        output_dir=str(MODEL_DIR),
        batch_size=16,
        num_epochs=2,
        num_iterations=20,  # SetFit uses contrastive pairs
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        column_mapping={"text": "text", "label": "label"},
    )

    print("\nTraining...")
    trainer.train()

    # Evaluate
    metrics = trainer.evaluate(eval_ds)
    print(f"\nEval metrics: {metrics}")

    # Save model + metadata
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MODEL_DIR))

    # Save label mapping
    meta = {
        "active_labels": active_labels,
        "all_labels": all_labels,
        "label_counts": dict(label_counts),
        "train_size": len(train_ds),
        "eval_size": len(eval_ds),
        "metrics": metrics,
    }
    with open(MODEL_DIR / "training-meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nModel saved: {MODEL_DIR}")
    print(f"Next: python scripts/classify-all.py")


if __name__ == "__main__":
    main()
