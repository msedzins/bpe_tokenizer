from bpe import merge, train_bpe


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
