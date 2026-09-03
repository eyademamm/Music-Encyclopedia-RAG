"""Download and verify the pinned local embedding model."""

import hashlib
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "Xenova/bge-small-en-v1.5"
# Pin the revision used to build the committed embeddings so fresh installs
# cannot silently switch to newer model weights.
MODEL_REVISION = "ea104dacec62c0de699686887e3f920caeb4f3e3"
EXPECTED_SHA256 = {
    "onnx/model.onnx": "828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35",
    "tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    local_dir = Path(__file__).resolve().parent.parent / "models" / "bge-small-en-v1.5"
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {REPO_ID} at revision {MODEL_REVISION} to {local_dir}")
    for filename, expected_sha256 in EXPECTED_SHA256.items():
        path = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                revision=MODEL_REVISION,
                local_dir=local_dir,
            )
        )
        actual_sha256 = sha256(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"Checksum mismatch for {filename}: expected {expected_sha256}, got {actual_sha256}"
            )
        print(f"Verified {filename}")

    print("Model download complete.")

if __name__ == "__main__":
    main()
