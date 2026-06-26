# OpenCortex
# src/database.py — Thin MongoDB connection manager.
# Owns the client lifecycle; higher-level modules (auth, chat_history) consume
# the connection through get_collection() rather than duplicating client setup.

from pymongo import MongoClient
from utils.logger import setup_logger

logger = setup_logger("database")


class MongoManager:
    """Manages a single MongoClient and exposes the opencortex database."""

    def __init__(self, url="mongodb://mongodb:27017/"):
        """Connect to MongoDB. Fails gracefully — consumers check is_connected."""
        try:
            self.client = MongoClient(url, serverSelectionTimeoutMS=5000)
            self.client.server_info()  # Raises if unreachable
            self._db = self.client["opencortex"]
            logger.info("Successfully connected to MongoDB.")
        except Exception as e:
            logger.error(f"MongoDB connection failed: {e}")
            self.client = None
            self._db = None

    @property
    def is_connected(self):
        """Whether the client is healthy and the database is available."""
        return self.client is not None and self._db is not None

    def get_collection(self, name):
        """Return a PyMongo Collection handle, or None if not connected."""
        return self._db[name] if self.is_connected else None
