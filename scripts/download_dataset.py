"""Baixa o Chest X-Ray Dataset (keremberke/chest-xray-classification) do Hugging Face.

Extrai os splits no layout esperado por `DatasetRepository`:
    data/{train,valid,test}/{NORMAL,PNEUMONIA}/*.jpg

Uso:
    python scripts/download_dataset.py [--data-dir data]
"""

import argparse
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "keremberke/chest-xray-classification"
SPLITS = ["train", "valid", "test"]


def download_dataset(data_dir: Path) -> None:
    for split in SPLITS:
        split_dir = data_dir / split
        if split_dir.exists() and any(split_dir.rglob("*.jpg")):
            print(f"[ok] {split_dir} já existe, pulando.")
            continue

        zip_path = hf_hub_download(REPO_ID, f"data/{split}.zip", repo_type="dataset")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(split_dir)

        counts = {
            c.name: len(list(c.glob("*.jpg"))) for c in sorted(split_dir.iterdir()) if c.is_dir()
        }
        print(f"[ok] {split_dir}: {counts}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    try:
        # Usa os certificados do sistema (antivírus com inspeção SSL quebram o certifi)
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass

    download_dataset(args.data_dir)


if __name__ == "__main__":
    main()
