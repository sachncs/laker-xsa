#!/usr/bin/env python3
"""
Real NLP Benchmark: Sentiment Classification with XSA+LAKER.

Uses the IMDB movie review dataset (or AG News if available) to evaluate
attention mechanisms on a real-world NLP task.

This tests whether XSA+LAKER's mathematical properties translate to
practical benefits on standard NLP benchmarks.

Usage:
    # With IMDB (requires datasets package)
    python -m examples.nlp_sentiment_benchmark --dataset imdb --max-length 256

    # With AG News (text classification)
    python -m examples.nlp_sentiment_benchmark --dataset agnews --max-length 512

    # Quick test with synthetic data
    python -m examples.nlp_sentiment_benchmark --dataset synthetic --num-samples 500
"""

from __future__ import annotations

import argparse
import json
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from laker_xsa.config import XSA_LAKER_Config
from laker_xsa.model.full_model import XSALAKERTransformer

# Try to import datasets, fall back to synthetic
try:
    from datasets import load_dataset

    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


class SentimentDataset(Dataset):
    """Sentiment classification dataset."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: Any,
        max_length: int = 256,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        text = self.texts[idx]
        label = self.labels[idx]

        # Simple tokenization: split on whitespace, map to vocab ids
        tokens = self.tokenizer(text, self.max_length)
        return tokens, label


class SimpleTokenizer:
    """Simple word-level tokenizer with vocabulary."""

    def __init__(self, vocab_size: int = 10000) -> None:
        self.vocab_size = vocab_size
        self.word2idx: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word: Dict[int, str] = {0: "<PAD>", 1: "<UNK>"}
        self.next_idx = 2

    def fit(self, texts: List[str]) -> None:
        """Build vocabulary from texts."""
        word_counts: Dict[str, int] = {}
        for text in texts:
            for word in text.lower().split():
                word_counts[word] = word_counts.get(word, 0) + 1

        # Sort by frequency and take top vocab_size - 2
        sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
        for word, _ in sorted_words[: self.vocab_size - 2]:
            self.word2idx[word] = self.next_idx
            self.idx2word[self.next_idx] = word
            self.next_idx += 1

    def __call__(self, text: str, max_length: int) -> torch.Tensor:
        """Tokenize text to fixed-length tensor."""
        tokens = []
        for word in text.lower().split():
            tokens.append(self.word2idx.get(word, 1))  # 1 = <UNK>

        # Pad or truncate
        if len(tokens) < max_length:
            tokens.extend([0] * (max_length - len(tokens)))
        else:
            tokens = tokens[:max_length]

        return torch.tensor(tokens, dtype=torch.long)


def create_synthetic_dataset(
    num_samples: int,
    max_length: int,
    vocab_size: int,
) -> Tuple[List[str], List[int]]:
    """
    Create synthetic sentiment dataset.

    Positive reviews contain words from positive set.
    Negative reviews contain words from negative set.
    """
    positive_words = [
        "great",
        "excellent",
        "amazing",
        "wonderful",
        "fantastic",
        "love",
        "best",
        "perfect",
        "beautiful",
        "brilliant",
    ]
    negative_words = [
        "terrible",
        "awful",
        "horrible",
        "worst",
        "bad",
        "hate",
        "boring",
        "stupid",
        "waste",
        "poor",
    ]
    neutral_words = [
        "the",
        "a",
        "is",
        "was",
        "it",
        "this",
        "that",
        "movie",
        "film",
        "and",
        "but",
        "very",
        "really",
        "so",
        "just",
    ]

    texts = []
    labels = []

    for i in range(num_samples):
        label = i % 2  # Alternate positive/negative

        # Create review with sentiment words + neutral fillers
        sentiment_words = positive_words if label == 1 else negative_words
        length = random.randint(max_length // 4, max_length // 2)

        words = []
        for _ in range(length):
            if random.random() < 0.3:  # 30% sentiment words
                words.append(random.choice(sentiment_words))
            else:
                words.append(random.choice(neutral_words))

        texts.append(" ".join(words))
        labels.append(label)

    return texts, labels


def load_imdb_dataset(
    max_samples: int = 5000,
    max_length: int = 256,
) -> Tuple[List[str], List[int], int]:
    """Load IMDB dataset."""
    if not HAS_DATASETS:
        print("datasets package not installed, falling back to synthetic")
        texts, labels = create_synthetic_dataset(max_samples, max_length, 10000)
        return texts, labels, 10000

    try:
        dataset = load_dataset("imdb")
        train_texts = dataset["train"]["text"][:max_samples]
        train_labels = dataset["train"]["label"][:max_samples]
        test_texts = dataset["test"]["text"][: max_samples // 2]
        test_labels = dataset["test"]["label"][: max_samples // 2]

        # Build tokenizer vocabulary
        tokenizer = SimpleTokenizer(vocab_size=5000)
        tokenizer.fit(train_texts + test_texts)

        return train_texts, train_labels, test_texts, test_labels, tokenizer
    except Exception as e:
        print(f"Error loading IMDB: {e}, falling back to synthetic")
        texts, labels = create_synthetic_dataset(max_samples, max_length, 10000)
        return texts, labels, 10000


def load_agnews_dataset(
    max_samples: int = 5000,
    max_length: int = 256,
) -> Tuple:
    """Load AG News classification dataset."""
    if not HAS_DATASETS:
        texts, labels = create_synthetic_dataset(max_samples, max_length, 10000)
        return texts, labels, 10000

    try:
        dataset = load_dataset("ag_news")
        train_texts = dataset["train"]["text"][:max_samples]
        train_labels = dataset["train"]["label"][:max_samples]
        test_texts = dataset["test"]["text"][: max_samples // 2]
        test_labels = dataset["test"]["label"][: max_samples // 2]

        tokenizer = SimpleTokenizer(vocab_size=5000)
        tokenizer.fit(train_texts + test_texts)

        return train_texts, train_labels, test_texts, test_labels, tokenizer
    except Exception as e:
        print(f"Error loading AG News: {e}, falling back to synthetic")
        texts, labels = create_synthetic_dataset(max_samples, max_length, 10000)
        return texts, labels, 10000


class SentimentClassifier(nn.Module):
    """Transformer-based sentiment classifier."""

    def __init__(
        self,
        config: XSA_LAKER_Config,
        num_layers: int,
        vocab_size: int,
        max_seq_len: int,
        num_classes: int = 2,
        dropout: float = 0.1,
        attention_type: str = "standard",
    ) -> None:
        super().__init__()

        # Create transformer WITHOUT output projection (we need hidden states)
        self.transformer = XSALAKERTransformer(
            config,
            num_layers=num_layers,
            d_ff=config.d_model * 4,
            vocab_size=vocab_size,  # For embedding
            max_seq_len=max_seq_len,
            dropout=dropout,
            attention_type=attention_type,
        )
        # Remove output projection since we do classification
        self.transformer.output_proj = None

        # Classification head: use [CLS]-equivalent (first position)
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(config.d_model, num_classes),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        # Get transformer output (hidden states)
        hidden = self.transformer(input_ids)

        # Pool: use first token (like CLS)
        pooled = hidden[:, 0, :]

        # Classify
        logits = self.classifier(pooled)
        return logits


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
) -> Dict[str, List[float]]:
    """Train classifier and return history."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01
    )
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for input_ids, labels in train_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()

        train_losses.append(epoch_loss / len(train_loader))
        train_accs.append(correct / total)

        # Validation
        model.eval()
        epoch_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for input_ids, labels in val_loader:
                input_ids = input_ids.to(device)
                labels = labels.to(device)

                logits = model(input_ids)
                loss = criterion(logits, labels)

                epoch_loss += loss.item()
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.numel()

        val_losses.append(epoch_loss / len(val_loader))
        val_accs.append(correct / total)

        print(
            f"  Epoch {epoch + 1}: Train Loss={train_losses[-1]:.4f}, "
            f"Train Acc={train_accs[-1]:.4f}, Val Acc={val_accs[-1]:.4f}"
        )

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
    }


def evaluate_classifier(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate classifier on test set."""
    model.eval()
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for input_ids, labels in test_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits = model(input_ids)
            preds = logits.argmax(dim=-1)

            correct += (preds == labels).sum().item()
            total += labels.numel()

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    accuracy = correct / total

    # Compute F1 score for binary classification
    tp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 1)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    return accuracy, f1


def run_nlp_benchmark(
    dataset_name: str = "synthetic",
    max_length: int = 256,
    d_model: int = 128,
    num_heads: int = 4,
    num_layers: int = 4,
    num_epochs: int = 20,
    learning_rate: float = 1e-3,
    num_samples: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run NLP sentiment benchmark."""
    torch.manual_seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Load dataset
    print(f"\nLoading {dataset_name} dataset...")

    if dataset_name == "synthetic":
        train_texts, train_labels = create_synthetic_dataset(
            num_samples, max_length, vocab_size=1000
        )
        test_texts, test_labels = create_synthetic_dataset(
            num_samples // 4, max_length, vocab_size=1000
        )
        tokenizer = SimpleTokenizer(vocab_size=1000)
        tokenizer.fit(train_texts)
        vocab_size = 1000
        num_classes = 2

    elif dataset_name == "imdb":
        result = load_imdb_dataset(max_samples=num_samples, max_length=max_length)
        train_texts, train_labels, test_texts, test_labels, tokenizer = result
        vocab_size = len(tokenizer.word2idx)
        num_classes = 2

    elif dataset_name == "agnews":
        result = load_agnews_dataset(max_samples=num_samples, max_length=max_length)
        train_texts, train_labels, test_texts, test_labels, tokenizer = result
        vocab_size = len(tokenizer.word2idx)
        num_classes = 4  # AG News has 4 classes

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"  Train samples: {len(train_texts)}")
    print(f"  Test samples: {len(test_texts)}")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Num classes: {num_classes}")
    print(f"  Max length: {max_length}")

    # Create datasets
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, max_length)
    test_dataset = SentimentDataset(test_texts, test_labels, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8)

    results: Dict[str, Any] = {
        "dataset": dataset_name,
        "max_length": max_length,
        "d_model": d_model,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "vocab_size": vocab_size,
        "num_classes": num_classes,
        "num_epochs": num_epochs,
        "attention_types": {},
    }

    for attention_type in ["standard", "fused"]:
        print(f"\n{'=' * 60}")
        print(f"Training {attention_type} attention model...")
        print("=" * 60)

        config = XSA_LAKER_Config(
            d_model=d_model,
            num_heads=num_heads,
            num_iterations=10,
            preconditioner_rank=d_model // 16,
            kernel_type="rbf",
            xsa_mode="subtract_projection",
            dropout=0.1,
        )

        model = SentimentClassifier(
            config=config,
            num_layers=num_layers,
            vocab_size=vocab_size,
            max_seq_len=max_length,
            num_classes=num_classes,
            dropout=0.1,
            attention_type=attention_type,
        )

        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {total_params:,}")

        history = train_classifier(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,  # Use test as val for simplicity
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=device,
        )

        test_accuracy, test_f1 = evaluate_classifier(model, test_loader, device)

        results["attention_types"][attention_type] = {
            "test_accuracy": test_accuracy,
            "test_f1": test_f1,
            "final_train_loss": history["train_losses"][-1],
            "final_val_accuracy": history["val_accs"][-1],
            "train_accs": history["train_accs"],
            "val_accs": history["val_accs"],
            "total_params": total_params,
        }

        print(f"\n{attention_type.upper()} Results:")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        print(f"  Test F1 Score: {test_f1:.4f}")
        print(f"  Final Train Loss: {history['train_losses'][-1]:.4f}")

    # Compute comparison
    if (
        "standard" in results["attention_types"]
        and "fused" in results["attention_types"]
    ):
        std_acc = results["attention_types"]["standard"]["test_accuracy"]
        fused_acc = results["attention_types"]["fused"]["test_accuracy"]
        std_f1 = results["attention_types"]["standard"]["test_f1"]
        fused_f1 = results["attention_types"]["fused"]["test_f1"]

        results["comparison"] = {
            "accuracy_improvement": fused_acc - std_acc,
            "accuracy_improvement_pct": ((fused_acc - std_acc) / max(std_acc, 0.01))
            * 100,
            "f1_improvement": fused_f1 - std_f1,
            "f1_improvement_pct": ((fused_f1 - std_f1) / max(std_f1, 0.01)) * 100,
        }

        print(f"\n{'=' * 60}")
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print(
            f"Accuracy Improvement: {results['comparison']['accuracy_improvement']:.4f} "
            f"({results['comparison']['accuracy_improvement_pct']:.1f}%)"
        )
        print(
            f"F1 Improvement: {results['comparison']['f1_improvement']:.4f} "
            f"({results['comparison']['f1_improvement_pct']:.1f}%)"
        )

    return results


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="NLP Sentiment Benchmark")
    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        choices=["synthetic", "imdb", "agnews"],
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    print("=" * 60)
    print("NLP Sentiment Classification Benchmark")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Max Length: {args.max_length}")
    print(
        f"Model: d_model={args.d_model}, heads={args.num_heads}, layers={args.num_layers}"
    )

    results = run_nlp_benchmark(
        dataset_name=args.dataset,
        max_length=args.max_length,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
