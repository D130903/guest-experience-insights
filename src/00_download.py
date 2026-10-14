"""Step 0 - Download the Inside Airbnb Melbourne snapshot (16 Sep 2026) into data/."""
from pathlib import Path
from urllib.request import urlretrieve

BASE = "https://data.insideairbnb.com/australia/vic/melbourne/2026-09-16"
FILES = {"listings.csv.gz": f"{BASE}/data/listings.csv.gz",
         "reviews.csv.gz": f"{BASE}/data/reviews.csv.gz"}

if __name__ == "__main__":
    data = Path(__file__).resolve().parents[1] / "data"
    data.mkdir(exist_ok=True)
    for name, url in FILES.items():
        if not (data / name).exists():
            print(f"downloading {name} ...")
            urlretrieve(url, data / name)
    print("data ready")
