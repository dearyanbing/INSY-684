"""Download the IBM Telco Customer Churn CSV into data/ (no Kaggle account needed)."""
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from churn.config import DATA_URL, RAW_DATA_PATH  # noqa: E402


def main() -> None:
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if RAW_DATA_PATH.exists():
        print(f"Already have {RAW_DATA_PATH}")
        return
    print(f"Downloading {DATA_URL}")
    urllib.request.urlretrieve(DATA_URL, RAW_DATA_PATH)
    print(f"Saved to {RAW_DATA_PATH}")


if __name__ == "__main__":
    main()
