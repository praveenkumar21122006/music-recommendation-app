#!/bin/bash
echo "Starting Musica — http://127.0.0.1:8000"
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
