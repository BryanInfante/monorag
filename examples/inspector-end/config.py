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

# --- Admin ---

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")

# --- Paths ---

EXAMPLE_DIR = Path(__file__).parent
AGENTS_MD_PATH = EXAMPLE_DIR / "agents.md"

# --- UI Constants ---

AGENT_NAME = "Inspector END"
AGENT_SUBTITLE = "Asistente experto en normas de END"
AGENT_DESCRIPTION = (
    "Expert assistant specialized in Non-Destructive Testing (NDT) standards. "
    "Ask questions about inspection criteria, procedures, and acceptance standards."
)

SUGGESTED_QUESTIONS = [
    "¿Cuáles son los criterios de aceptación para UT según ASME Sección V?",
    "¿Cómo se calibra la inspección por partículas magnéticas?",
    "¿Qué preparación de superficie requiere el ensayo por líquidos penetrantes?",
    "Explica los requisitos de densidad de película radiográfica.",
    "¿Cuáles son los requisitos de calificación para END Nivel II?",
]

EMPTY_STATE_TITLE = "¿Cómo puedo ayudarte con tu inspección?"
EMPTY_STATE_DESCRIPTION = (
    "Pregunta sobre criterios de aceptación, calibración, "
    "calificación o cualquier procedimiento en las normas de END indexadas."
)

DISCLAIMER = (
    "Inspector END puede cometer errores. "
    "Verifica los criterios críticos con la norma original."
)
