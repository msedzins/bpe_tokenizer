# Byte-level BPE tokenizer

Edukacyjna implementacja byte-level BPE w Pythonie, przygotowana na warsztat
[Slayer Labs](https://github.com/slayerlabs/tokenizer). Tokenizer zaczyna od
256 tokenów odpowiadających pojedynczym bajtom. Podczas treningu uczy się reguł
scalania sąsiednich par. Używa tych reguł do kodowania tekstu, a słownik bajtów
pozwala odtworzyć tekst w UTF-8.

## Przykład

Uruchom z katalogu repozytorium. Skrypt trenuje tokenizer na krótkim tekście,
koduje drugi tekst i odtwarza go z listy ID tokenów.

```sh
python3 bpe.py
```

## Testy

Zainstaluj `pytest` w swoim środowisku Pythona i uruchom testy.

```sh
python3 -m pip install pytest
python3 -m pytest
```
