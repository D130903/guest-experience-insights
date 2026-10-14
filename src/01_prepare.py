"""Step 1 - Clean the raw Inside Airbnb data for Melbourne.

Inputs  (data/):    listings.csv.gz, reviews.csv.gz
Outputs (outputs/): listings_clean.parquet, reviews_clean.parquet

What it does
- Listings: parse price, flag superhosts, group property types into
  "Hotel & serviced" vs "Private home", and flag listings within 1.5 km of
  Crown Melbourne (Southbank precinct).
- Reviews: drop empty and automated reviews (host cancellations), keep only
  English text, and attach each review's listing attributes.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "outputs"
OUT.mkdir(exist_ok=True)

CROWN_LAT, CROWN_LON = -37.8236, 144.9580   # Crown Melbourne, Southbank
PRECINCT_KM = 1.5

HOTEL_TYPES = ("hotel", "serviced apartment", "aparthotel", "resort")

ENGLISH_WORDS = {"the", "and", "was", "is", "to", "a", "very", "great", "stay",
                 "place", "we", "it", "of", "in", "for", "with", "clean", "nice"}


def haversine_km(lat, lon, lat0, lon0):
    lat, lon, lat0, lon0 = map(np.radians, (lat, lon, lat0, lon0))
    a = (np.sin((lat - lat0) / 2) ** 2
         + np.cos(lat) * np.cos(lat0) * np.sin((lon - lon0) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def is_english(text: str) -> bool:
    words = text.lower().split()
    if len(words) < 3:
        return False
    return sum(w.strip(".,!?") in ENGLISH_WORDS for w in words) / len(words) >= 0.08


def prepare_listings() -> pd.DataFrame:
    raw = pd.read_csv(DATA / "listings.csv.gz", low_memory=False)
    score_cols = [c for c in raw.columns if c.startswith("review_scores_")]
    df = raw[["id", "name", "neighbourhood_cleansed", "latitude", "longitude",
              "property_type", "room_type", "accommodates", "price",
              "host_is_superhost", "number_of_reviews", *score_cols]].copy()

    df = df.rename(columns={"id": "listing_id", "neighbourhood_cleansed": "lga"})
    df["price"] = pd.to_numeric(df["price"].str.replace(r"[$,]", "", regex=True),
                                errors="coerce")
    df["superhost"] = df["host_is_superhost"].eq("t")
    df["stay_type"] = np.where(
        df["property_type"].str.lower().str.contains("|".join(HOTEL_TYPES), na=False),
        "Hotel & serviced", "Private home")
    df["km_to_crown"] = haversine_km(df["latitude"], df["longitude"], CROWN_LAT, CROWN_LON)
    df["near_crown"] = df["km_to_crown"] <= PRECINCT_KM
    df = df.drop(columns="host_is_superhost")
    df.columns = [c.replace("review_scores_", "score_") for c in df.columns]
    return df


def prepare_reviews(listings: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(DATA / "reviews.csv.gz",
                     usecols=["listing_id", "id", "date", "comments"],
                     parse_dates=["date"])
    n_raw = len(df)
    df = df.dropna(subset=["comments"])
    df["comments"] = (df["comments"].str.replace(r"<br\s*/?>", " ", regex=True)
                      .str.replace(r"\s+", " ", regex=True).str.strip())
    automated = df["comments"].str.contains(
        r"^The host canceled this reservation|This is an automated posting", regex=True)
    df = df[~automated & df["comments"].str.len().ge(10)]
    df = df[df["comments"].map(is_english)]
    df = df.rename(columns={"id": "review_id"})
    df = df.merge(listings[["listing_id", "lga", "stay_type", "near_crown", "room_type"]],
                  on="listing_id", how="inner")
    print(f"reviews: {n_raw:,} raw -> {len(df):,} clean English reviews")
    return df


if __name__ == "__main__":
    listings = prepare_listings()
    listings.to_parquet(OUT / "listings_clean.parquet", index=False)
    print(f"listings: {len(listings):,}  "
          f"near Crown: {listings.near_crown.sum():,}  "
          f"hotel & serviced: {(listings.stay_type == 'Hotel & serviced').sum():,}")
    prepare_reviews(listings).to_parquet(OUT / "reviews_clean.parquet", index=False)
