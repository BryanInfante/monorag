"""Configuration for the Inspector END example agent.

Centralizes environment variable resolution, defaults, and UI constants.
"""

import os
from pathlib import Path

# --- LLM Configuration ---

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY") or ""
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or None

# --- Collection ---

MONORAG_COLLECTION = os.getenv("MONORAG_COLLECTION", "inspector-end")
MONORAG_DB_PATH = os.getenv("MONORAG_DB_PATH") or None

# --- Paths ---

EXAMPLE_DIR = Path(__file__).parent
AGENTS_MD_PATH = EXAMPLE_DIR / "agents.md"

# --- UI Constants ---

AGENT_NAME = "Inspector END"
AGENT_DESCRIPTION = (
    "Expert assistant specialized in Non-Destructive Testing (NDT) standards. "
    "Ask questions about inspection criteria, procedures, and acceptance standards."
)

SUGGESTED_QUESTIONS = [
    "What are the acceptance criteria for liquid penetrant indications?",
    "What surface preparation is required before penetrant testing?",
    "What is the minimum dwell time for penetrant application?",
    "How should excess penetrant be removed?",
    "What are the environmental requirements for penetrant testing?",
]
