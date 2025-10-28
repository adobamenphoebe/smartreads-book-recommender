
import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

TITLE_COL = "bookTitle"
DESC_COL = "bookDesc"
GENRE_COL = "bookGenres"
AUTH_COL = "bookAuthors"
RATING_COL = "bookRating"
RC_COL = "ratingCount"
REV_COL = "reviewCount"

def build_features(df):
    # combine genres + description; genres are weighted by repeating tokens
    genres = df[GENRE_COL].fillna('').astype(str).str.replace('[\[\]\'\"]','', regex=True).str.replace(',', ' ', regex=False)
    desc = df[DESC_COL].fillna('').astype(str)
    # emphasize genres slightly
    combined = (genres + ' ' + genres + ' ' + desc).values
    tfidf = TfidfVectorizer(stop_words='english', max_features=50000, ngram_range=(1,2))
    X = tfidf.fit_transform(combined)
    return tfidf, X

def filter_by_genre(df, genre_query):
    # case-insensitive containment match on the raw genre string
    mask = df[GENRE_COL].astype(str).str.lower().str.contains(genre_query.lower(), na=False)
    return df[mask].copy()

def rank(df_filt, X_filt, query_text=None, topk=10, weights=(0.6, 0.25, 0.15)):
    # Similarity component
    if query_text and query_text.strip():
        from sklearn.feature_extraction.text import TfidfVectorizer
        # rebuild small tfidf on filtered set to transform query consistently
        genres = df_filt[GENRE_COL].fillna('').astype(str).str.replace('[\[\]\'\"]','', regex=True).str.replace(',', ' ', regex=False)
        desc = df_filt[DESC_COL].fillna('').astype(str)
        combined = (genres + ' ' + genres + ' ' + desc).values
        tfidf_local = TfidfVectorizer(stop_words='english', max_features=50000, ngram_range=(1,2))
        X_local = tfidf_local.fit_transform(combined)
        qv = tfidf_local.transform([query_text])
        sim = cosine_similarity(qv, X_local).ravel()
    else:
        # if no query given, use item popularity/quality only
        sim = np.zeros(len(df_filt), dtype=float)

    # Popularity: ratingCount + reviewCount (scaled 0-1)
    pop_cols = []
    if RC_COL in df_filt.columns:
        pop_cols.append(RC_COL)
    if REV_COL in df_filt.columns and REV_COL != RC_COL:
        pop_cols.append(REV_COL)
    if pop_cols:
        pop = df_filt[pop_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).sum(axis=1).values.reshape(-1,1)
        pop = MinMaxScaler().fit_transform(pop).ravel()
    else:
        pop = np.zeros(len(df_filt), dtype=float)

    # Quality: rating (scaled 0-1)
    if RATING_COL in df_filt.columns:
        qual = pd.to_numeric(df_filt[RATING_COL], errors='coerce').fillna(0.0).values.reshape(-1,1)
        qual = MinMaxScaler().fit_transform(qual).ravel()
    else:
        qual = np.zeros(len(df_filt), dtype=float)

    w_sim, w_pop, w_qual = weights
    score = w_sim*sim + w_pop*pop + w_qual*qual
    order = np.argsort(-score)[:topk]
    cols = [c for c in [TITLE_COL, AUTH_COL, GENRE_COL, RATING_COL, RC_COL, REV_COL] if c in df_filt.columns]
    out = df_filt.iloc[order][cols].copy()
    out['score'] = score[order]
    return out

def main():
    ap = argparse.ArgumentParser(description="SmartReads: Genre‑Specific Recommender")
    ap.add_argument("--data", default="data/raw/goodreads_genre_clean_5k.csv")
    ap.add_argument("--genre", required=True, help="Genre keyword, e.g., 'science fiction'")
    ap.add_argument("--query", default="", help="Optional free‑text query to refine matches")
    ap.add_argument("--topk", type=int, default=10, help="Number of results")
    ap.add_argument("--weights", default="0.6,0.25,0.15", help="Weights for similarity,popularity,quality")
    args = ap.parse_args()

    df = pd.read_csv(args.data, dtype=str)
    wf = [float(x) for x in args.weights.split(",")]
    # Filter by genre
    df_filt = filter_by_genre(df, args.genre)
    if df_filt.empty:
        print("No books found for genre:", args.genre)
        return
    # Build features on filtered set for efficiency
    tfidf, X = build_features(df_filt)
    # Rank
    recs = rank(df_filt, X, query_text=args.query, topk=args.topk, weights=tuple(wf))
    print(recs.to_string(index=False))

if __name__ == "__main__":
    main()
