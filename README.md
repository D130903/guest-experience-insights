# Melbourne Guest Experience Insights: a Voice of Guest analysis

**What guests complain about, where in the stay it happens, and what to fix first.**

I analysed 947,000 written guest reviews of Melbourne short-stay accommodation, the way a hotel
Guest Experience (GX) team would read its own Voice of Guest feedback, with a focus on the
Crown Melbourne / Southbank precinct.

- **Interactive dashboard:** `docs/index.html` (published with GitHub Pages)
- **Insights report for stakeholders:** [`INSIGHTS_REPORT.md`](INSIGHTS_REPORT.md)
- **Data:** [Inside Airbnb](https://insideairbnb.com/get-the-data/), Melbourne snapshot 16 Sep 2026:
  25,715 listings and 1.04M reviews (2010 - Sep 2026). Public data, CC BY 4.0.

## Key findings (reviews Oct 2024 - Sep 2026)

| | |
|---|---|
| Average star rating | **4.78 / 5**, but **5.9%** of reviews still contain a complaint |
| Crown precinct (within 1.5 km) | **7.0%** of reviews contain a complaint vs **5.3%** for the rest of Melbourne (**+31%**) |
| Arrival & check-in near Crown | **20.0** complaints per 1,000 reviews vs 11.6 elsewhere (**+73%**) |
| Hotel & serviced stays | Highest complaint rate (**8.2%**) and a lower rating (**4.58** vs 4.79 for private homes) |
| What drives the star rating | **Accuracy, cleanliness and value** (regression R² = 0.89); location barely matters |
| Worst month near Crown | **January** in both years (9.0% in Jan 2026): value complaints +57%, facilities +49%, room comfort +44% |

## How it works

```
src/00_download.py          download the Inside Airbnb files into data/
src/01_prepare.py           clean listings + reviews, keep English reviews, flag the Crown precinct and hotel-style stays
src/02_sentiment_themes.py  VADER sentiment for every review; split recent reviews into sentences and tag 12 guest-journey themes
src/03_analyse.py           complaint rates, severity, key-driver regression, trends, area scorecard, January effect
src/04_build_dashboard.py   build docs/index.html from outputs/dashboard_data.json
```

1. **Cleaning:** removed empty and automated reviews (host cancellations) and kept English text (947,165 reviews).
   Listings within 1.5 km of Crown Melbourne (haversine distance) are tagged as the Crown precinct, and hotels,
   serviced apartments and aparthotels are tagged as hotel-style stays.
2. **Sentiment:** each review and sentence gets a VADER score from -1 to +1. VADER misreads some hospitality
   phrases ("tram **stop**", "couldn't be better"), so I added lexicon and phrase fixes after reading samples.
3. **Themes:** 12 guest-journey themes (booking, arrival, cleanliness, sleep & noise, room comfort, facilities,
   staff, food, location, parking, value, accuracy) are tagged with keyword rules at the sentence level. A sentence
   scoring -0.05 or lower that mentions a theme counts as a complaint about it. Working at the sentence level matters:
   only 1.5% of reviews are negative overall, because most complaints sit inside otherwise positive reviews
   ("great location, but the bed was awful").
4. **Severity:** for each theme, the average sentiment of the whole review when that complaint appears.
   A low score means the complaint ruins the stay.
5. **Key drivers:** standardised linear regression of each listing's overall rating on its six category ratings
   (13,280 listings with 10+ reviews).

## Validation and limitations

- **Complaint precision:** in a manual check of 60 random complaint sentences, 45 (75%) were genuine complaints
  (`outputs/precision_check.csv`). The rest were mostly neutral context sentences or praise the model misread.
  Rates are best read as comparisons (segment vs segment, month vs month) rather than exact counts.
- Keyword theme tagging is transparent and easy to explain, but it misses complaints that use other words.
  A next step would be a trained classifier (e.g. a fine-tuned transformer) checked against a labelled sample.
- Airbnb reviews skew positive, and guests who had a bad stay may not leave a review at all.
- Listing category ratings are lifetime averages, so the driver analysis is cross-sectional (listing level).
- This is a public proxy dataset. It is not Crown data, and it shows the method a GX team would apply to its own surveys,
  complaints and online reviews.

## Run it

```bash
pip install -r requirements.txt
python src/00_download.py
python src/01_prepare.py
python src/02_sentiment_themes.py   # ~3 minutes on 8 cores
python src/03_analyse.py
python src/04_build_dashboard.py
```

Tables for Excel / Power BI are written to `outputs/*.csv`.

Built by Dhruv Rawat. Tools: Python, pandas, scikit-learn, VADER, Plotly.
