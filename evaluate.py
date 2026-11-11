"""Compare a saved byte-level BPE tokenizer with cl100k_base."""

import argparse
import hashlib
import json
from pathlib import Path

import tiktoken

from bpe import build_vocab
from train import load_merges


def encode_for_evaluation(text: str, merges: dict[tuple[int, int], int]) -> list[int]:
    # Each character represents one token ID, not a Unicode text character.
    tokens = text.encode('utf-8').decode('latin-1')
    for (left, right), token_id in merges.items():
        tokens = tokens.replace(chr(left) + chr(right), chr(token_id))
    return [ord(token) for token in tokens]


def measure(text: str, tokens: list[int], decoder) -> dict:
    word_count = len(text.split())
    token_count = len(tokens)
    return {
        'word_count': word_count,
        'token_count': token_count,
        'fertility': token_count / word_count if word_count else None,
        'characters_per_token': len(text) / token_count if token_count else None,
        'round_trip': decoder(tokens) == text,
    }


def main():
    parser = argparse.ArgumentParser(description='Compare BPE and cl100k_base on the same text.')
    parser.add_argument('--input', type=Path, default=Path('data/test.txt'))
    parser.add_argument('--tokenizer', type=Path, default=Path('tokenizers/polish-bpe-1024.json'))
    parser.add_argument('--output', type=Path, default=Path('results/metrics.json'))
    args = parser.parse_args()
    raw = args.input.read_bytes()
    text = raw.decode('utf-8')
    merges = load_merges(args.tokenizer)
    vocab = build_vocab(merges)
    reference = tiktoken.get_encoding('cl100k_base')
    ours = measure(text, encode_for_evaluation(text, merges),
                   lambda ids: b''.join(vocab[token] for token in ids).decode('utf-8'))
    baseline = measure(text, reference.encode_ordinary(text), reference.decode)
    result = {
        'input': {'path': str(args.input), 'bytes': len(raw), 'characters': len(text),
                  'sha256': hashlib.sha256(raw).hexdigest(),
                  'word_definition': 'Non-empty sequences separated by Python str.split() whitespace'},
        'model_sha256': hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
        'tiktoken_version': tiktoken.__version__,
        'tokenizers': {str(args.tokenizer): ours, 'cl100k_base': baseline},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
