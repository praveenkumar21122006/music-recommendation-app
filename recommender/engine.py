import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, StandardScaler

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "songs.csv")

class MusicRecommender:
    def __init__(self, csv_path=DATA_PATH):
        self.df = pd.read_csv(csv_path)
        self.df['combined_text'] = (
            self.df['genre'].astype(str) + " " +
            self.df['mood'].astype(str) + " " +
            self.df['artist'].astype(str) + " " +
            self.df['genre'].astype(str)  # boost genre weight
        )
        # TF-IDF on combined text
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(self.df['combined_text'])

        # Numerical features scaler
        num_cols = ['danceability','energy','valence','tempo','popularity','year']
        scaler = MinMaxScaler()
        self.num_features = scaler.fit_transform(self.df[num_cols])
        # Weighted numerical similarity
        # give more weight to danceability/energy/valence
        weights = np.array([1.2, 1.2, 1.2, 0.5, 0.8, 0.3])
        self.num_features_weighted = self.num_features * weights

        # Combined similarity matrix (cached)
        tfidf_sim = cosine_similarity(self.tfidf_matrix)
        num_sim = cosine_similarity(self.num_features_weighted)
        # Blend: 60% text, 40% numeric
        self.sim_matrix = 0.6 * tfidf_sim + 0.4 * num_sim

        # For collaborative mock: user -> liked ids mapping will be passed at runtime
        # Precompute popularity fallback
        self.popularity_sorted = self.df.sort_values(by='popularity', ascending=False)

    def search(self, query: str, limit: int = 10):
        if not query:
            return self.df.head(limit).to_dict(orient='records')
        q = query.lower()
        mask = (
            self.df['title'].str.lower().str.contains(q) |
            self.df['artist'].str.lower().str.contains(q) |
            self.df['genre'].str.lower().str.contains(q) |
            self.df['mood'].str.lower().str.contains(q)
        )
        results = self.df[mask]
        # rank by popularity if many results
        results = results.sort_values(by='popularity', ascending=False)
        return results.head(limit).to_dict(orient='records')

    def get_song(self, song_id: int):
        row = self.df[self.df['id'] == song_id]
        if row.empty:
            return None
        return row.iloc[0].to_dict()

    def recommend_for_songs(self, liked_ids, n=10, mood=None, genre=None, exclude_ids=None):
        """
        Content-based: aggregate similarity to liked songs.
        Optionally filter by mood/genre.
        """
        if exclude_ids is None:
            exclude_ids = set(liked_ids)
        else:
            exclude_ids = set(exclude_ids) | set(liked_ids)

        if not liked_ids:
            # Cold start: popularity + diversity
            candidates = self.df.copy()
            if mood:
                candidates = candidates[candidates['mood'].str.lower() == mood.lower()]
            if genre:
                candidates = candidates[candidates['genre'].str.lower() == genre.lower()]
            if candidates.empty:
                candidates = self.df.copy()
            # add diversity: take top popularity but shuffle genres
            candidates = candidates.sort_values(by='popularity', ascending=False).head(30)
            return candidates.head(n).to_dict(orient='records')

        # get indices of liked songs
        liked_indices = []
        for lid in liked_ids:
            idx = self.df.index[self.df['id'] == lid].tolist()
            if idx:
                liked_indices.append(idx[0])

        if not liked_indices:
            return self.recommend_for_songs([], n=n, mood=mood, genre=genre)

        # mean similarity to liked songs
        scores = np.mean(self.sim_matrix[liked_indices, :], axis=0)

        # popularity boost (0.1 weight)
        pop_norm = self.df['popularity'].values / 100.0
        scores = scores * 0.9 + pop_norm * 0.1

        # create scored df
        scored = self.df.copy()
        scored['score'] = scores

        # filter mood/genre if requested
        if mood:
            # soft filter: boost mood matches rather than hard filter, but if user selects, filter
            scored = scored[scored['mood'].str.lower() == mood.lower()] if len(scored[scored['mood'].str.lower()==mood.lower()]) >= n else scored
            # if we filtered and still have results, keep; else boost
            if mood.lower() in scored['mood'].str.lower().values:
                # boost
                scored.loc[scored['mood'].str.lower() == mood.lower(), 'score'] += 0.15

        if genre:
            if len(scored[scored['genre'].str.lower() == genre.lower()]) >= n:
                scored = scored[scored['genre'].str.lower() == genre.lower()]
            else:
                scored.loc[scored['genre'].str.lower() == genre.lower(), 'score'] += 0.15

        scored = scored[~scored['id'].isin(exclude_ids)]
        scored = scored.sort_values(by='score', ascending=False)

        # add reasoning
        results = scored.head(n).to_dict(orient='records')
        # attach score and reason
        for r in results:
            # find most similar liked song
            r_idx = self.df.index[self.df['id'] == r['id']].tolist()[0]
            best_liked_idx = max(liked_indices, key=lambda i: self.sim_matrix[i, r_idx])
            best_song = self.df.iloc[best_liked_idx]
            reason = f"Similar to '{best_song['title']}'"
            if r['genre'] == best_song['genre']:
                reason += f" • Same genre ({r['genre']})"
            if r['mood'] == best_song['mood']:
                reason += f" • Same mood ({r['mood']})"
            r['reason'] = reason
            r['score'] = round(float(r['score']), 3)

        return results

    def recommend_for_user(self, liked_ids, history_ids=None, n=10, mood=None, genre=None):
        if history_ids is None:
            history_ids = []
        exclude = set(liked_ids) | set(history_ids)
        return self.recommend_for_songs(liked_ids, n=n, mood=mood, genre=genre, exclude_ids=exclude)

    def similar_songs(self, song_id, n=8):
        idx_list = self.df.index[self.df['id'] == song_id].tolist()
        if not idx_list:
            return []
        idx = idx_list[0]
        scores = self.sim_matrix[idx]
        scored = self.df.copy()
        scored['score'] = scores
        scored = scored[scored['id'] != song_id].sort_values(by='score', ascending=False)
        results = scored.head(n).to_dict(orient='records')
        for r in results:
            r['score'] = round(float(r['score']), 3)
            r['reason'] = f"Similar vibe to selected track"
        return results

    def get_genres(self):
        return sorted(self.df['genre'].unique().tolist())

    def get_moods(self):
        return sorted(self.df['mood'].unique().tolist())

    def get_stats(self):
        return {
            "total_songs": len(self.df),
            "total_artists": self.df['artist'].nunique(),
            "total_genres": self.df['genre'].nunique(),
            "total_moods": self.df['mood'].nunique(),
        }
