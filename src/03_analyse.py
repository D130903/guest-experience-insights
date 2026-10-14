"""Step 3 - Turn scored reviews into Guest Experience insights.

Inputs  (outputs/): listings_clean.parquet, reviews_scored.parquet, sentences_themed.parquet
Outputs (outputs/): dashboard_data.json plus one CSV per table (for Excel / Power BI)

Questions answered
1. What do guests complain about most, by stage of the guest journey?
2. Which complaints hurt the guest's overall impression most (severity)?
3. Which parts of the stay drive the overall star rating (key-driver analysis)?
4. How does the Crown / Southbank precinct compare with the rest of Melbourne,
   and hotel-style stays with private homes?
5. Is the experience getting better or worse month to month?
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
NEG = -0.05
WINDOW_START = pd.Timestamp("2024-10-01")
WINDOW_END = pd.Timestamp("2026-09-01")   # last full month before the Sep 2026 scrape

JOURNEY_ORDER = ["Booking & communication", "Arrival & check-in", "Cleanliness",
                 "Sleep quality & noise", "Room comfort", "Facilities & amenities",
                 "Staff & hospitality", "Food & dining", "Location & transport",
                 "Parking", "Value for money", "Listing accuracy"]

SUBSCORES = {"score_cleanliness": "Cleanliness", "score_checkin": "Check-in",
             "score_communication": "Communication", "score_location": "Location",
             "score_value": "Value", "score_accuracy": "Accuracy"}

SEGMENTS = {
    "All Melbourne": lambda d: d.index == d.index,
    "Near Crown (1.5 km)": lambda d: d["near_crown"],
    "Rest of Melbourne": lambda d: ~d["near_crown"],
    "Hotel & serviced": lambda d: d["stay_type"].eq("Hotel & serviced"),
    "Private home": lambda d: d["stay_type"].eq("Private home"),
}

def explode_themes(sentences: pd.DataFrame) -> pd.DataFrame:
    return (sentences.assign(theme=sentences["themes"].str.split("|"))
            .explode("theme").drop(columns="themes"))


def theme_table(reviews, themed):
    """Per theme: how often guests mention it, how often they complain, and severity."""
    n = len(reviews)
    complaints = themed[themed["sentiment"] <= NEG]
    review_sent = reviews.set_index("review_id")["sentiment"]
    rows = []
    for theme in JOURNEY_ORDER:
        t = themed[themed["theme"] == theme]
        c = complaints[complaints["theme"] == theme]
        complaining_reviews = c["review_id"].unique()
        rows.append({
            "theme": theme,
            "mention_rate": t["review_id"].nunique() / n,
            "complaints_per_1000": len(complaining_reviews) / n * 1000,
            "negative_share_of_mentions": (t["sentiment"] <= NEG).mean(),
            "avg_review_sentiment_when_complaint":
                review_sent.reindex(complaining_reviews).mean(),
        })
    return pd.DataFrame(rows)


def key_drivers(listings):
    """Which sub-scores move the overall rating? Standardised regression + importance/performance."""
    cols = list(SUBSCORES)
    d = listings[listings["number_of_reviews"] >= 10].dropna(subset=cols + ["score_rating"])
    X = (d[cols] - d[cols].mean()) / d[cols].std()
    y = (d["score_rating"] - d["score_rating"].mean()) / d["score_rating"].std()
    model = LinearRegression().fit(X, y)
    out = pd.DataFrame({
        "driver": [SUBSCORES[c] for c in cols],
        "impact": model.coef_,
        "correlation": [d[c].corr(d["score_rating"]) for c in cols],
        "avg_score": [d[c].mean() for c in cols],
        "pct_below_4_5": [(d[c] < 4.5).mean() for c in cols],
    }).sort_values("impact", ascending=False)
    return out, model.score(X, y), len(d)


def monthly_trend(reviews, complaint_ids):
    r = reviews[(reviews["date"] >= WINDOW_START) & (reviews["date"] < WINDOW_END)].copy()
    r["month"] = r["date"].dt.to_period("M").dt.to_timestamp()
    r["has_complaint"] = r["review_id"].isin(complaint_ids)
    r["segment"] = np.where(r["near_crown"], "Near Crown (1.5 km)", "Rest of Melbourne")
    return (r.groupby(["segment", "month"])["has_complaint"].agg(["mean", "size"])
            .reset_index().rename(columns={"mean": "complaint_rate", "size": "reviews"}))


def lga_table(listings, reviews, complaint_ids, themed_complaints):
    r = reviews.assign(has_complaint=reviews["review_id"].isin(complaint_ids))
    top_theme = (themed_complaints.groupby(["lga", "theme"]).size().reset_index(name="n")
                 .sort_values("n").groupby("lga").tail(1).set_index("lga")["theme"])
    rated = listings[listings["number_of_reviews"] >= 5]
    t = pd.DataFrame({
        "listings": listings.groupby("lga").size(),
        "reviews_24m": r.groupby("lga").size(),
        "avg_rating": rated.groupby("lga")["score_rating"].mean(),
        "complaint_rate": r.groupby("lga")["has_complaint"].mean(),
        "top_complaint": top_theme,
    }).dropna(subset=["reviews_24m"])
    return t[t["reviews_24m"] >= 1000].sort_values("complaint_rate").reset_index(names="lga")


def curated_verbatims(sentences):
    """Look up analyst-selected quotes (src/curated_verbatims.csv) in the near-Crown review text.

    The shortlist was read from random samples of complaint and praise sentences per
    theme; quotes naming people were excluded. Each match must exist in the data.
    """
    picks = pd.read_csv(ROOT / "src" / "curated_verbatims.csv")
    near = sentences.loc[sentences["near_crown"], "sentence"].drop_duplicates()
    quotes = []
    for match in picks["match"]:
        found = near[near.str.contains(match, regex=False)]
        if found.empty:
            raise ValueError(f"curated quote not found in data: {match!r}")
        quotes.append(found.iloc[0])
    return picks.drop(columns="match").assign(quote=quotes)


def quarterly_volume(reviews):
    r = reviews[(reviews["date"] >= "2016-01-01") & (reviews["date"] < "2026-07-01")]
    q = r.groupby(r["date"].dt.to_period("Q")).size()
    return pd.DataFrame({"quarter": q.index.astype(str), "reviews": q.values})


def event_months(reviews):
    """Review volume by calendar month near Crown - a proxy for visitor demand."""
    r = reviews[(reviews["date"] >= "2024-01-01") & (reviews["date"] < "2026-01-01")
                & reviews["near_crown"]]
    m = r.groupby(r["date"].dt.month).size() / 2
    return pd.DataFrame({"month": pd.to_datetime(m.index, format="%m").strftime("%b"),
                         "avg_reviews": m.values.round()})


def january_effect(reviews, themed_complaints):
    """Near Crown: complaint rate per theme in January (summer + Australian Open) vs other months."""
    r = reviews[reviews["near_crown"]]
    c = themed_complaints[themed_complaints["near_crown"]]
    jan_r, jan_c = r["date"].dt.month.eq(1), c["date"].dt.month.eq(1)
    out = pd.DataFrame({
        "january_per_1000": c[jan_c].groupby("theme")["review_id"].nunique() / jan_r.sum() * 1000,
        "other_months_per_1000": c[~jan_c].groupby("theme")["review_id"].nunique() / (~jan_r).sum() * 1000,
    })
    out["uplift"] = out["january_per_1000"] / out["other_months_per_1000"] - 1
    return out.sort_values("uplift", ascending=False).reset_index(names="theme")


def records(df):
    return json.loads(df.to_json(orient="records", date_format="iso", double_precision=4))


if __name__ == "__main__":
    listings = pd.read_parquet(OUT / "listings_clean.parquet")
    reviews_all = pd.read_parquet(OUT / "reviews_scored.parquet")
    sentences = pd.read_parquet(OUT / "sentences_themed.parquet")

    reviews = reviews_all[reviews_all["date"] >= WINDOW_START]
    themed = explode_themes(sentences)
    themed_complaints = themed[themed["sentiment"] <= NEG]
    complaint_ids = set(themed_complaints["review_id"])

    themes_by_segment = {}
    for name, mask in SEGMENTS.items():
        r = reviews[mask(reviews)]
        t = themed[themed["review_id"].isin(r["review_id"])]
        tbl = theme_table(r, t)
        tbl.insert(0, "segment", name)
        themes_by_segment[name] = tbl
    theme_all = pd.concat(themes_by_segment.values())

    drivers, r2, n_drivers = key_drivers(listings)
    trend = monthly_trend(reviews, complaint_ids)
    lgas = lga_table(listings, reviews, complaint_ids, themed_complaints)
    verbatims = curated_verbatims(sentences)
    volume = quarterly_volume(reviews_all)
    events = event_months(reviews_all)
    january = january_effect(reviews, themed_complaints)

    superhost = (listings[listings["number_of_reviews"] >= 10]
                 .groupby("superhost")[["score_rating", "score_communication",
                                        "score_cleanliness"]].mean())

    def seg_kpis(mask):
        r = reviews[mask(reviews)]
        l = listings[mask(listings) & (listings["number_of_reviews"] >= 5)]
        return {"reviews": int(len(r)),
                "listings": int(mask(listings).sum()),
                "avg_rating": float(l["score_rating"].mean()),
                "complaint_rate": float(r["review_id"].isin(complaint_ids).mean()),
                "negative_review_rate": float((r["sentiment"] <= NEG).mean())}

    data = {
        "meta": {"window": f"{WINDOW_START:%b %Y} - Sep 2026",
                 "reviews_all_time": int(len(reviews_all)),
                 "reviews_window": int(len(reviews)),
                 "sentences_tagged": int(len(sentences)),
                 "listings": int(len(listings)),
                 "driver_r2": r2, "driver_n": n_drivers},
        "kpis": {name: seg_kpis(mask) for name, mask in SEGMENTS.items()},
        "themes": records(theme_all),
        "drivers": records(drivers),
        "trend": records(trend),
        "lgas": records(lgas),
        "verbatims": records(verbatims),
        "volume": records(volume),
        "event_months": records(events),
        "january": records(january),
        "superhost": records(superhost.reset_index()),
    }
    (OUT / "dashboard_data.json").write_text(json.dumps(data, indent=1), encoding="utf-8")

    theme_all.to_csv(OUT / "theme_summary.csv", index=False)
    drivers.to_csv(OUT / "key_drivers.csv", index=False)
    trend.to_csv(OUT / "monthly_complaint_trend.csv", index=False)
    lgas.to_csv(OUT / "lga_scorecard.csv", index=False)
    verbatims.to_csv(OUT / "verbatims.csv", index=False)
    january.to_csv(OUT / "january_effect.csv", index=False)

    pd.set_option("display.width", 200)
    print(json.dumps(data["kpis"], indent=1))
    print(theme_all.round(3).to_string())
    print(drivers.round(3).to_string(), "R2", round(r2, 3), "n", n_drivers)
    print(lgas.round(3).to_string())
    print(superhost.round(3))
    print(events.to_string())
    print(january.round(3).to_string())
    print(trend.groupby("segment")["complaint_rate"].agg(["first", "last", "mean"]))
