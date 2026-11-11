"""Train byte-level BPE and save its ordered merge rules."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

from bpe import train_bpe


def save_merges(path: Path, merges: dict[tuple[int, int], int], metadata: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rules = [[left, right, token_id] for (left, right), token_id in merges.items()]
    path.write_text(json.dumps({"merges": rules, "metadata": metadata or {}}, indent=2) + "\n", encoding="utf-8")


def load_merges(path: Path) -> dict[tuple[int, int], int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {(left, right): token_id for left, right, token_id in data["merges"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer.")
    parser.add_argument("--input", type=Path, default=Path("data/train.txt"))
    parser.add_argument("--iterations", type=int, required=True, help="Number of merge operations")
    parser.add_argument("--output", type=Path, default=Path("tokenizers/polish-bpe.json"))
    parser.add_argument("--max-chars", type=int, help="Use the first N Unicode characters")
    args = parser.parse_args()
    if args.iterations < 0:
        parser.error("--iterations must be non-negative")

    if args.max_chars is not None and args.max_chars <= 0:
        parser.error("--max-chars must be positive")
    text = args.input.read_bytes().decode("utf-8")
    if args.max_chars is not None:
        text = text[:args.max_chars]
    sample = text.encode("utf-8")
    start = perf_counter()
    _, merges = train_bpe(text, args.iterations)
    duration = perf_counter() - start
    metadata = {
        "corpus_characters": len(text),
        "corpus_bytes": len(sample),
        "corpus_sha256": hashlib.sha256(sample).hexdigest(),
        "requested_merges": args.iterations,
        "merge_count": len(merges),
        "vocab_size": 256 + len(merges),
        "training_seconds": duration,
    }
    save_merges(args.output, merges, metadata)
    print(f"Saved {len(merges)} merges to {args.output} (vocabulary: {256 + len(merges)} tokens)")


if __name__ == "__main__":
    main()
