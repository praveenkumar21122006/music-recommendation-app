from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from recommender.engine import MusicRecommender

app = FastAPI(title="Music Recommendation System", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

recommender = MusicRecommender()

class RecommendRequest(BaseModel):
    liked_ids: List[int] = []
    history_ids: List[int] = []
    n: int = 10
    mood: Optional[str] = None
    genre: Optional[str] = None

@app.get("/api/health")
def health():
    return {"status": "ok", "stats": recommender.get_stats()}

@app.get("/api/songs")
def list_songs(q: Optional[str] = None, limit: int = 20, genre: Optional[str]=None, mood: Optional[str]=None):
    if q:
        return recommender.search(q, limit=limit)
    df = recommender.df
    if genre:
        df = df[df['genre'].str.lower() == genre.lower()]
    if mood:
        df = df[df['mood'].str.lower() == mood.lower()]
    df = df.sort_values(by='popularity', ascending=False)
    return df.head(limit).to_dict(orient='records')

@app.get("/api/songs/{song_id}")
def get_song(song_id: int):
    song = recommender.get_song(song_id)
    if not song:
        raise HTTPException(status_code=404, detail="Song not found")
    return song

@app.get("/api/songs/{song_id}/similar")
def similar(song_id: int, n: int = 8):
    return recommender.similar_songs(song_id, n=n)

@app.post("/api/recommend")
def recommend(req: RecommendRequest):
    results = recommender.recommend_for_user(
        liked_ids=req.liked_ids,
        history_ids=req.history_ids,
        n=req.n,
        mood=req.mood,
        genre=req.genre
    )
    return {"recommendations": results, "count": len(results)}

@app.get("/api/genres")
def genres():
    return recommender.get_genres()

@app.get("/api/moods")
def moods():
    return recommender.get_moods()

@app.get("/api/stats")
def stats():
    return recommender.get_stats()

# Serve frontend static files
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

