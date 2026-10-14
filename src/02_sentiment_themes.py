"""Step 2 - Score guest sentiment and tag guest-journey themes.

Inputs  (outputs/): reviews_clean.parquet
Outputs (outputs/): reviews_scored.parquet, sentences_themed.parquet

Method
- Every review gets a VADER sentiment score (-1 very negative .. +1 very positive).
- Reviews from the last 24 months are split into sentences. Each sentence is
  scored and tagged with the guest-journey themes it mentions (keyword rules
  below). A sentence scoring <= -0.05 that mentions a theme counts as a
  complaint about that theme.
"""
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

WINDOW_START = pd.Timestamp("2024-10-01")   # last 24 months of the Sep 2026 scrape
NEG, POS = -0.05, 0.05

# Guest-journey themes, ordered the way a guest moves through a stay.
THEMES = {
    "Booking & communication": r"communicat|respon(d|s|ded)|repl(y|ied)|messag|instruction|booking|booked|cancel",
    "Arrival & check-in": r"check[- ]?in|check[- ]?out|\bkeys?\b|lock ?box|key ?safe|\bcode\b|arriv|\baccess\b|front desk|reception",
    "Cleanliness": r"clean|dirty|dust|stain|\bhairs?\b|spotless|tidy|mou?ld|smell|hygien|filthy|grime",
    "Sleep quality & noise": r"noise|noisy|\bloud|quiet|sleep|\bbeds?\b|mattress|pillow|party|parties|thin walls",
    "Room comfort": r"spacious|cramped|comfort|\bviews?\b|heat(er|ing)|air ?con|cold|\bhot\b|shower|bathroom|water pressure|temperature",
    "Facilities & amenities": r"wi-?fi|internet|\btv\b|netflix|\bpool\b|\bgym\b|kitchen|washing machine|laundry|\blifts?\b|elevator|amenit|towel|toiletr",
    "Staff & hospitality": r"friendly|welcom|helpful|hospitab|hospitality|\bstaff\b|service|\brude\b|accommodating|attentive|unhelpful",
    "Food & dining": r"breakfast|restaurant|\bcafes?\b|coffee|\bfood\b|dining|dinner|\bbars?\b",
    "Location & transport": r"locat|walking distance|\btrams?\b|\btrains?\b|station|\bcbd\b|transport|public transport",
    "Parking": r"parking|car ?park|park the car|\bparked\b",
    "Value for money": r"\bprice|value|expensive|worth|money|cheap|\bfees?\b|overpriced|afford",
    "Listing accuracy": r"as described|as advertised|photos|pictures|misleading|accurate|not as expected",
}
THEME_RE = {k: re.compile(v, re.I) for k, v in THEMES.items()}
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

# VADER is a general-purpose lexicon. These fixes stop common hospitality
# phrases being misread (found by reading samples of "negative" sentences).
LEXICON_FIXES = {"stop": 0.0, "stops": 0.0, "war": 0.0, "hard": -0.3, "fault": -0.5}
PHRASE_FIXES = [
    (r"\b(could|can|would)(n['’]?t| ?not) (have )?(be|been|get|got|ask for|wish for|fault|recommend)"
     r"( it| this( place)?)?( any| much| a| more)? ?(better|more|highly)", "excellent"),
    (r"\bnot (too |that |very )?far\b", "close"),
    (r"\b(no|zero|nothing to) (complaints?|complain about|issues?|problems?|negatives?)\b", "great"),
    (r"\b(did ?n['’]?t|did not|does ?n['’]?t|does not) (bother|disappoint|affect)( us| me)?\b", "fine"),
    (r"\bnot disappoint(ed|ing)?\b", "great"),
    (r"\bno (hassle|fuss|worries)\b", "easy"),
]
PHRASE_RE = [(re.compile(p, re.I), r) for p, r in PHRASE_FIXES]

_analyzer = None


def _init():
    global _analyzer
    _analyzer = SentimentIntensityAnalyzer()
    _analyzer.lexicon.update(LEXICON_FIXES)


def polarity(text: str) -> float:
    for rx, repl in PHRASE_RE:
        text = rx.sub(repl, text)
    return _analyzer.polarity_scores(text)["compound"]


def score_reviews(texts):
    return [polarity(t) for t in texts]


def score_sentences(rows):
    """rows: list of (review_id, text) -> list of (review_id, sentence, compound, themes)."""
    out = []
    for review_id, text in rows:
        for sent in SENT_SPLIT.split(text):
            if len(sent) < 8:
                continue
            themes = [k for k, rx in THEME_RE.items() if rx.search(sent)]
            if not themes:
                continue
            out.append((review_id, sent, polarity(sent),
                        "|".join(themes)))
    return out


def chunks(seq, n):
    size = max(1, len(seq) // n)
    return [seq[i:i + size] for i in range(0, len(seq), size)]


if __name__ == "__main__":
    reviews = pd.read_parquet(OUT / "reviews_clean.parquet")

    with Pool(initializer=_init) as pool:
        parts = pool.map(score_reviews, chunks(reviews["comments"].tolist(), 64))
        reviews["sentiment"] = np.concatenate(parts)
        reviews["sentiment_label"] = pd.cut(reviews["sentiment"], [-1.01, NEG, POS, 1.01],
                                            labels=["Negative", "Neutral", "Positive"])

        recent = reviews[reviews["date"] >= WINDOW_START]
        rows = list(zip(recent["review_id"], recent["comments"]))
        sent_parts = pool.map(score_sentences, chunks(rows, 64))

    sentences = pd.DataFrame([r for part in sent_parts for r in part],
                             columns=["review_id", "sentence", "sentiment", "themes"])
    sentences = sentences.merge(
        reviews[["review_id", "listing_id", "date", "lga", "stay_type", "near_crown"]],
        on="review_id")

    reviews.drop(columns="comments").to_parquet(OUT / "reviews_scored.parquet", index=False)
    sentences.to_parquet(OUT / "sentences_themed.parquet", index=False)
    print(f"scored {len(reviews):,} reviews; {len(recent):,} in the 24-month window; "
          f"{len(sentences):,} theme-tagged sentences")
    print(reviews["sentiment_label"].value_counts(normalize=True).round(3).to_string())
