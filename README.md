# Musica — Personalized Music Recommendation System

> Address the need for personalized and accurate music recommendations with tailored suggestions based on preferences, mood, and listening history.

A full-stack, hybrid recommendation engine + modern web UI. Content-based filtering (TF-IDF + audio-feature similarity) with popularity & mood/genre re-ranking. Handles cold-start, search, and similar-track discovery.

![Stack](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python)
![ML](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn)

---

## ✨ Features

- **Personalized recommendations** — like songs → instant tailored picks
- **Hybrid scoring**: `0.6·TF-IDF(genre+mood+artist) + 0.4·cosine(danceability,energy,valence,tempo,popularity,year)` + popularity boost + mood/genre re-rank
- **Cold-start fallback** — popular & diverse picks when no history
- **Search** by title/artist/genre/mood, filter by genre/mood, trending/chill/energetic tabs
- **Similar tracks** — click cover to find similar vibe
- **Persisted likes** via localStorage, history exclusion
- **Explainability** — each recommendation shows `Similar to 'X' • Same genre/mood` + score

## 📁 Project Structure

```
music-recommendation-app/
├── data/
│   └── songs.csv              # 100 tracks across 14 genres, 9 moods, audio features
├── recommender/
│   ├── __init__.py
│   └── engine.py              # MusicRecommender core (TF-IDF + cosine)
├── backend/
│   └── main.py                # FastAPI API + static frontend serving
├── frontend/
│   └── index.html             # Vanilla JS SPA — no build step
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

```bash
pip install -r requirements.txt  # or: pip install --break-system-packages -r requirements.txt
uvicorn backend.main:app --reload --port 8000
# open http://127.0.0.1:8000
```

## 🔌 API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` / `/api/stats` | Health & dataset stats |
| GET | `/api/songs?q=&genre=&mood=&limit=` | Search / list & filter |
| GET | `/api/songs/{id}` | Get single track |
| GET | `/api/songs/{id}/similar?n=` | Similar tracks |
| POST | `/api/recommend` | Recommendations |
| GET | `/api/genres` / `/api/moods` | Taxonomies |

**POST /api/recommend** body:
```json
{
  "liked_ids": [1, 3, 11],
  "history_ids": [],
  "n": 10,
  "mood": "Energetic",
  "genre": "Pop"
}
```

**Example:**
```bash
curl -X POST http://127.0.0.1:8000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"liked_ids":[1,3],"n":5}'
```

## 🧠 How It Works

1. **Text features**: `genre + mood + artist` → TF-IDF vector → cosine similarity
2. **Audio features**: `[danceability, energy, valence, tempo, popularity, year]` min-max scaled, weighted `[1.2,1.2,1.2,0.5,0.8,0.3]` → cosine similarity
3. **Fusion**: `sim = 0.6*text_sim + 0.4*audio_sim`
4. **Personalization**: mean similarity to liked songs, plus `0.9*sim + 0.1*popularity`, plus mood/genre boost, exclude liked/history, rank.

No external APIs needed — fully offline, extensible to Spotify API.

## 🧪 Verify

```bash
python3 -c "from recommender.engine import MusicRecommender; r=MusicRecommender(); print(r.recommend_for_songs([1,3], n=3))"
```

## 🔮 Next Steps

- Collaborative filtering with real user-item matrix (ALS / NMF)
- Spotify API enrichment (audio_features, user top tracks & OAuth)
- Deep learning embeddings (autoencoders / transformers)
- Playlist generation & mood-sequence recommendation

## 📄 License

MIT — use freely for learning/demo.
