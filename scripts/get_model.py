import hashlib
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
LOCK_PATH = ROOT / "model.lock"


def read_lock():
    if not LOCK_PATH.exists():
        raise FileNotFoundError("model.lock not found. Please add one before downloading models.")
    data = tomllib.loads(LOCK_PATH.read_text(encoding="utf-8"))
    repo_id = data["repo_id"]
    filename = data["filename"]
    sha256 = data.get("sha256", "").strip() or None
    return repo_id, filename, sha256


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    load_dotenv()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    repo_id, filename, expected_sha = read_lock()
    token = os.getenv("HUGGINGFACE_TOKEN")

    print(f"Downloading {filename} from {repo_id} ...")
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(MODELS_DIR),
        local_dir_use_symlinks=False,
        token=token,
    )
    local_path = Path(local_path)

    if expected_sha:
        print("Verifying SHA-256...")
        actual_sha = sha256_file(local_path)
        if actual_sha.lower() != expected_sha.lower():
            print("SHA mismatch detected!", file=sys.stderr)
            print(f"Expected: {expected_sha}", file=sys.stderr)
            print(f"Actual:   {actual_sha}", file=sys.stderr)
            sys.exit(2)
        print("SHA-256 OK.")

    print(f"Model ready at: {local_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
