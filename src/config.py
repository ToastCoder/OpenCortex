# OpenCortex
# src/config.py — Centralised accessor for JSON config files.
# Loads parameters.json and prompts.json once at import time and exposes them
# through getter functions to prevent scattered mutations across the codebase.

import json
import os

from utils.logger import setup_logger

logger = setup_logger("config")

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")

# Load model / RAG / audio parameters
with open(os.path.join(CONFIG_DIR, "parameters.json")) as f:
    _params = json.load(f)

# Load prompt templates and normalise list-valued prompts into joined strings
with open(os.path.join(CONFIG_DIR, "prompts.json")) as f:
    _prompts_raw = json.load(f)
    _prompts = {}
    for key, value in _prompts_raw.items():
        if isinstance(value, list):
            _prompts[key] = "\n".join(value)
        else:
            _prompts[key] = value


def get_params():
    """Return the full parameters dictionary (live reference)."""
    return _params


def get_prompts():
    """Return the full prompts dictionary."""
    return _prompts


def set_vision_model(model_name):
    """Update the vision model used for image analysis at runtime."""
    _params["llm"]["vision_model"] = model_name
