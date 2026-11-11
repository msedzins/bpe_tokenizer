from bpe import build_vocab, decode, encode, merge, train_bpe


def test_merge_replaces_matching_pairs():
    assert merge([97, 98, 97, 98], (97, 98), 256) == [256, 256]


def test_train_bpe_merges_most_frequent_pair():
    tokens, merges = train_bpe("abab", 1)

    assert tokens == [256, 256]
    assert merges == {(97, 98): 256}


def test_train_bpe_empty_text():
    tokens, merges = train_bpe("", 3)

    assert tokens == []
    assert merges == {}


def test_train_bpe_single_byte_has_no_pairs_to_merge():
    tokens, merges = train_bpe("a", 3)

    assert tokens == [97]
    assert merges == {}


def test_train_bpe_zero_iterations_keeps_input_tokens():
    tokens, merges = train_bpe("ab", 0)

    assert tokens == [97, 98]
    assert merges == {}


def test_encode_applies_earlier_merge_before_more_frequent_pair():
    merges = {(98, 99): 256, (97, 98): 257}
    assert encode("abcab", merges) == [97, 256, 257]


def test_build_vocab_includes_bytes_and_composed_tokens():
    vocab = build_vocab({(98, 99): 256, (97, 256): 257})
    assert len(vocab) == 258
    assert all(vocab[i] == bytes([i]) for i in range(256))
    assert vocab[256] == b"bc"
    assert vocab[257] == b"abc"


def test_round_trip_for_utf8_and_whitespace():
    _, merges = train_bpe("Zażółć gęślą jaźń 🙂 🙂", 20)
    vocab = build_vocab(merges)
    for text in ("", "Zażółć gęślą jaźń", "🙂🙃", "  a\n\tb  ", "Łódź 🙂"):
        assert decode(encode(text, merges), vocab) == text


def test_saved_merges_keep_order_and_encoding(tmp_path):
    from train import load_merges, save_merges

    merges = {(98, 99): 256, (97, 98): 257, (97, 256): 258}
    path = tmp_path / "tokenizer.json"
    save_merges(path, merges)
    loaded = load_merges(path)
    assert list(loaded.items()) == list(merges.items())
    assert encode("abcab ą🙂", loaded) == encode("abcab ą🙂", merges)


def test_training_cli_records_selected_sample(tmp_path):
    import hashlib
    import json
    import subprocess
    import sys

    source = tmp_path / "source.txt"
    source.write_bytes("ąababa\r\nextra".encode("utf-8"))
    output = tmp_path / "model.json"
    subprocess.run(
        [sys.executable, "train.py", "--input", str(source), "--iterations", "2",
         "--max-chars", "6", "--output", str(output)],
        check=True,
    )
    data = json.loads(output.read_text(encoding="utf-8"))
    sample = "ąababa".encode("utf-8")
    assert data["metadata"]["corpus_characters"] == 6
    assert data["metadata"]["corpus_bytes"] == len(sample)
    assert data["metadata"]["corpus_sha256"] == hashlib.sha256(sample).hexdigest()
    assert data["metadata"]["merge_count"] == len(data["merges"])
    assert data["metadata"]["vocab_size"] == 256 + len(data["merges"])
    assert data["metadata"]["training_seconds"] >= 0
