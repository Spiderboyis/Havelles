"""
Configuration settings for the Customer Voice Intelligence Agent.
Centralizes all configuration parameters for easy management.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Project Paths ───────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
VECTORDB_DIR = BASE_DIR / "vectordb"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
VECTORDB_DIR.mkdir(exist_ok=True)

# ─── LLM Configuration ──────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))

# ─── Havells Product Categories ─────────────────────────────────
HAVELLS_PRODUCT_CATEGORIES = {
    "fans": {
        "subcategories": ["ceiling_fan", "table_fan", "wall_fan", "pedestal_fan", "exhaust_fan"],
        "common_aspects": ["air_delivery", "noise", "speed_control", "design", "motor_quality",
                           "energy_efficiency", "remote_control", "installation", "durability", "wobble"]
    },
    "water_heaters": {
        "subcategories": ["instant_geyser", "storage_geyser", "solar_heater"],
        "common_aspects": ["heating_speed", "temperature_control", "safety", "energy_efficiency",
                           "build_quality", "installation", "thermostat", "leakage", "capacity", "durability"]
    },
    "air_purifiers": {
        "subcategories": ["room_purifier", "car_purifier"],
        "common_aspects": ["filter_quality", "coverage_area", "noise_level", "air_quality_display",
                           "filter_replacement", "design", "effectiveness", "smart_features"]
    },
    "mixer_grinders": {
        "subcategories": ["mixer_grinder", "juicer_mixer", "hand_blender"],
        "common_aspects": ["grinding_performance", "motor_power", "jar_quality", "noise",
                           "durability", "ease_of_cleaning", "safety_lock", "speed_settings"]
    },
    "lighting": {
        "subcategories": ["led_bulb", "led_panel", "downlighter", "smart_light", "decorative"],
        "common_aspects": ["brightness", "color_temperature", "energy_saving", "lifespan",
                           "flickering", "heat_generation", "design", "smart_connectivity"]
    }
}

# ─── Sentiment Analysis Config ───────────────────────────────────
SENTIMENT_LABELS = ["very_negative", "negative", "neutral", "positive", "very_positive"]
SENTIMENT_THRESHOLDS = {
    "very_negative": (-1.0, -0.6),
    "negative": (-0.6, -0.1),
    "neutral": (-0.1, 0.1),
    "positive": (0.1, 0.6),
    "very_positive": (0.6, 1.0)
}

# ─── Topic Modeling Config ───────────────────────────────────────
BERTOPIC_MIN_TOPIC_SIZE = 5
BERTOPIC_NR_TOPICS = "auto"
UMAP_N_NEIGHBORS = 15
UMAP_N_COMPONENTS = 5
HDBSCAN_MIN_CLUSTER_SIZE = 5

# ─── Vector Store Config ─────────────────────────────────────────
CHROMA_COLLECTION_NAME = "havells_reviews"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RETRIEVAL = 15

# ─── Agent Configuration ─────────────────────────────────────────
MAX_AGENT_ITERATIONS = 10
AGENT_TIMEOUT_SECONDS = 120
GROUNDING_CONFIDENCE_THRESHOLD = 0.7
