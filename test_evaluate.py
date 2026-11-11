import random

from bpe import encode, train_bpe
from evaluate import encode_for_evaluation, measure


def test_evaluation_encoding_matches_bpe():
    rng = random.Random(42)
    alphabet = 'abc ąć🙂\n'
    for _ in range(40):
        corpus = ''.join(rng.choices(alphabet, k=80))
        text = ''.join(rng.choices(alphabet, k=60))
        _, merges = train_bpe(corpus, 30)
        assert encode_for_evaluation(text, merges) == encode(text, merges)


def test_measure_counts_whitespace_delimited_words():
    text = 'ą  b!\n🙂'
    result = measure(text, list(text.encode('utf-8')), lambda _: text)
    assert result['word_count'] == 3
    assert result['token_count'] == len(text.encode('utf-8'))
    assert result['fertility'] == len(text.encode('utf-8')) / 3
    assert result['characters_per_token'] == len(text) / len(text.encode('utf-8'))
    assert result['round_trip'] is True
