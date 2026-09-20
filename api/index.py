import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from backend.main import app
# Vercel expects 'app' to be exposed for ASGI
