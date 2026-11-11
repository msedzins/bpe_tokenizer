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

## Korpus i podział

Źródłem jest [Wikipedia w Polish DynaWord](https://huggingface.co/datasets/SlayerLab/polish-dynaword/tree/ad1e59fbee7055326706a6f1709473bf22ac34ef/data/wikipedia), rewizja `ad1e59fbee7055326706a6f1709473bf22ac34ef`. [Karta źródła](https://huggingface.co/datasets/SlayerLab/polish-dynaword/blob/ad1e59fbee7055326706a6f1709473bf22ac34ef/data/wikipedia/wikipedia.md) opisuje pochodzenie i licencję danych.

Zainstaluj `pyarrow` i uruchom jedno polecenie z katalogu repozytorium:

```sh
python3 -m pip install pyarrow
python3 prepare_corpus.py
```

Skrypt pobiera plik Parquet (około 1,46 GB), sprawdza jego SHA-256 i wybiera kolejne niepuste dokumenty do osiągnięcia 50 000 000 bajtów tekstu UTF-8 wraz z separatorami. Co dziesiąty dokument trafia do `data/test.txt`, a pozostałe do `data/train.txt`. Dokumenty są oddzielone dwiema nowymi liniami. Skrypt zapisuje rozmiary, liczbę dokumentów i sumy SHA-256 w `data/metadata.json`. Pliki powstają lokalnie w ignorowanym przez Git katalogu `data/`. Ponowne uruchomienie nadpisuje pliki wynikowe.

Oczekiwane wyniki dla przypiętej rewizji:

| Plik | Dokumenty | Bajty | SHA-256 |
| --- | ---: | ---: | --- |
| `train.txt` | 8 324 | 44 966 880 | `bbca42db24d0fc13a0a23065373989012192ab662d96c5090112119cc703383f` |
| `test.txt` | 925 | 5 042 301 | `e864d613a6606fda47ca6e954c13301d6b63b7fa2e291ba77ed35e85349b5b4e` |

## Trening i zapis tokenizera

Po przygotowaniu korpusu uruchom trening, podając liczbę scaleń:

```sh
python3 train.py --input data/train.txt --iterations 100 --output tokenizers/polish-bpe.json
```

`--iterations` oznacza liczbę scaleń. Słownik ma 256 tokenów bazowych oraz
faktycznie wykonane scalenia. Trening może zakończyć się wcześniej, jeśli w
tekście nie ma kolejnych par. Duży korpus i liczba scaleń wydłużają trening.
Plik JSON zawiera listę reguł `[lewy_id, prawy_id, nowe_id]` w kolejności ich
powstania. Tokenizer można wczytać i użyć bez ponownego treningu:

```python
from pathlib import Path
from bpe import build_vocab, decode, encode
from train import load_merges

merges = load_merges(Path("tokenizers/polish-bpe.json"))
tokens = encode("Zażółć gęślą jaźń", merges)
text = decode(tokens, build_vocab(merges))
```

Przykładowe warianty wytrenowano na pierwszym milionie znaków `data/train.txt`.
Próbka ma 1 057 373 bajty i SHA-256
`ae26495cab63e4d1bd31ed624dca356668754db901d9d98381db6efdf70b7ff4`.
Polecenia do powtórzenia treningu:

```sh
python3 train.py --input data/train.txt --max-chars 1000000 --iterations 768 --output tokenizers/polish-bpe-1024.json
python3 train.py --input data/train.txt --max-chars 1000000 --iterations 3840 --output tokenizers/polish-bpe-4096.json
```

Metadane zapisane w JSON zawierają rozmiar i SHA-256 próbki, żądaną i wykonaną
liczbę scaleń, rozmiar słownika i czas treningu w sekundach. Czas nie obejmuje
wczytywania danych ani zapisu pliku. Kolejne uruchomienie nadpisuje artefakt.

## Ewaluacja

Zainstaluj `tiktoken` i uruchom pomiar dla obu zapisanych tokenizerów:

```sh
python3 -m pip install tiktoken==0.12.0
python3 evaluate.py --tokenizer tokenizers/polish-bpe-1024.json --output results/metrics.json
python3 evaluate.py --tokenizer tokenizers/polish-bpe-4096.json --output results/metrics-4096.json
```

Oba tokenizery są mierzone na całym `data/test.txt`. Słowo to niepusty
fragment rozdzielony białymi znakami według `str.split()` w Pythonie.
Fertility to liczba tokenów całego tekstu podzielona przez liczbę takich słów.
Znaki na token to liczba znaków Unicode w tekście podzielona przez liczbę
jego tokenów. Round-trip sprawdza dokładną zgodność odtworzonego tekstu.
Porównanie z `cl100k_base` używa `encode_ordinary`. Pierwsze uruchomienie
może pobrać dane kodowania `cl100k_base`.

Dla przyspieszenia ewaluacji własny tokenizer wykonuje kolejne reguły na
ciągu, w którym każdy znak reprezentuje ID tokenu. Test porównuje ten sposób
z `bpe.encode`. Wyniki obejmują SHA-256 danych i artefaktu oraz wersję
`tiktoken`.

### Wyniki i fertility

Fertility to średnia liczba tokenów całego tekstu przypadająca na jedno słowo
według definicji opisanej powyżej. Niższa wartość oznacza mniej tokenów dla
tego samego tekstu. W pomiarze na zbiorze testowym z polskiej Wikipedii
(4 771 412 znaków i 648 222 słowa) otrzymano:

| Tokenizer | Tokeny | Fertility | Znaki na token | Round-trip |
| --- | ---: | ---: | ---: | --- |
| BPE 1024 | 2 233 770 | 3,446 | 2,136 | zgodny |
| BPE 4096 | 1 646 458 | 2,540 | 2,898 | zgodny |
| `cl100k_base` | 1 823 293 | 2,813 | 2,617 | zgodny |

Oba własne warianty trenowano na tej samej próbce. Wariant 4096 ma 3840
wyuczonych scaleń, a wariant 1024 ma ich 768. Dodatkowe reguły pozwalają mu
scalać więcej fragmentów tekstu. W tym pomiarze wariant 4096 użył około 26,3%
mniej tokenów niż wariant 1024.

Wariant 4096 użył też około 9,7% mniej tokenów niż `cl100k_base`. Hipotezą jest
dopasowanie reguł do polskiego tekstu Wikipedii. Na wynik mogą wpływać także
różnice w pretokenizacji. Własny BPE przetwarza ciąg bajtów bez pretokenizacji,
więc może tworzyć tokeny obejmujące spacje i fragmenty sąsiednich słów. Wyniki
dotyczą przygotowanego zbioru testowego z Wikipedii. Do oceny na innych typach
tekstu potrzebny jest osobny pomiar.
