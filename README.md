# SmartReads — Genre‑Specific Book Recommender

A content‑based recommender that suggests books **within a chosen genre**, then ranks by a weighted blend of:
- **Similarity** on `bookDesc + bookGenres` (TF‑IDF + Cosine)
- **Popularity** (scaled `ratingCount` + `reviewCount`)
- **Quality** (`bookRating`)

## Dataset
Goodreads Best Books Ever with Recommendations (Kaggle). This repo includes a trimmed CSV (`data/raw/goodreads_genre_clean_5k.csv`) derived from your upload for reproducibility.

## Quickstart
```bash
pip install -r requirements.txt
python app.py --genre "science fiction" --query "space opera" --topk 10
# or open notebooks/genre_recommender.ipynb
```

## Scoring Weights
You can tune the contribution of each component:
```bash
python app.py --genre "fantasy" --weights 0.5,0.3,0.2
#           similarity, popularity, quality
```

## Notes
- Genres are emphasized in the text features to keep the system **genre‑specific**.
- Replace the data file with the full dataset anytime; the code will adapt as long as columns exist.