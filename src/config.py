"""
Centralized configuration for the Sports Quiz Agent.
Loads secrets from a local .env file so keys never live in source code.
"""

import os
from dotenv import load_dotenv

# Load variables from .env into the environment
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Path where the persistent ChromaDB vector store is saved
CHROMA_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")

# Path to the offline JSON knowledge base
SPORTS_FACTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "sports_facts.json"
)

SUPPORTED_SPORTS = ["Cricket", "Football", "Tennis", "Badminton", "Basketball"]
DIFFICULTY_LEVELS = ["Easy", "Medium", "Hard"]


def validate_config():
    """Returns a list of human-readable warnings about missing configuration."""
    warnings = []
    if not OPENAI_API_KEY:
        warnings.append(
            "OPENAI_API_KEY is missing. Add it to a .env file (see .env.example)."
        )
    return warnings
