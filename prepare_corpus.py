"""Download a pinned Wikipedia corpus and split complete documents."""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import pyarrow.parquet as pq

REVISION = "ad1e59fbee7055326706a6f1709473bf22ac34ef"
URL = f"https://huggingface.co/datasets/SlayerLab/polish-dynaword/resolve/{REVISION}/data/wikipedia/wikipedia.parquet"
PARQUET_SHA256 = "6fec912de221285b1d3de1e813b66aca9ebfb62782e64a73bf48783e8a013fa9"
TARGET_BYTES = 50_000_000


def main():
    data = Path(__file__).resolve().parent / "data"
    data.mkdir(exist_ok=True)
    parquet = data / "wikipedia.parquet"
    if not parquet.exists():
        temporary = data / "wikipedia.parquet.download"
        with urlopen(URL) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(parquet)
    digest = hashlib.file_digest(parquet.open("rb"), "sha256").hexdigest()
    if digest != PARQUET_SHA256:
        raise ValueError("Nieprawidłowa suma SHA-256 pliku źródłowego")

    paths = {name: data / f"{name}.txt" for name in ("train", "test")}
    temporary = {name: path.with_suffix(".tmp") for name, path in paths.items()}
    counts = {name: {"documents": 0, "bytes": 0} for name in paths}
    total = 0
    with temporary["train"].open("wb") as train, temporary["test"].open("wb") as test:
        outputs = {"train": train, "test": test}
        for batch in pq.ParquetFile(parquet).iter_batches(columns=["text"]):
            for text in batch.column(0).to_pylist():
                if not isinstance(text, str) or not text:
                    continue
                index = counts["train"]["documents"] + counts["test"]["documents"]
                split = "test" if index % 10 == 0 else "train"
                document = text.encode("utf-8") + b"\n\n"
                outputs[split].write(document)
                counts[split]["documents"] += 1
                counts[split]["bytes"] += len(document)
                total += len(document)
                if total >= TARGET_BYTES:
                    break
            if total >= TARGET_BYTES:
                break
    if total < TARGET_BYTES:
        raise ValueError("Korpus jest mniejszy od wymaganej próbki")
    for name, path in paths.items():
        temporary[name].replace(path)
        with path.open("rb") as file:
            counts[name]["sha256"] = hashlib.file_digest(file, "sha256").hexdigest()
    metadata = {"source": URL, "source_sha256": digest, "target_bytes": TARGET_BYTES, "split": counts}
    (data / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
