# OpenCortex
# src/chat_history.py — Persistence layer for conversation histories.
# Each user's messages are stored as an append-only array in a single
# MongoDB document, keyed by username.

from utils.logger import setup_logger

logger = setup_logger("chat_history")


class ChatHistory:
    """Read, append, and clear chat messages per user."""

    def __init__(self, db):
        self.db = db

    def save_message(self, username, role, content):
        """Append a single message to the user's conversation array."""
        conversations = self.db.get_collection("conversations")
        conversations.update_one(
            {"username": username},
            {"$push": {"messages": {"role": role, "content": content}}},
            upsert=True,
        )
        logger.info(f"Message saved for user {username}.")

    def get_history(self, username):
        """Return all stored messages for the user, newest-last."""
        conversations = self.db.get_collection("conversations")
        user_data = conversations.find_one({"username": username})
        return user_data["messages"] if user_data else []

    def clear_history(self, username):
        """Remove the entire conversation document for the user."""
        conversations = self.db.get_collection("conversations")
        conversations.delete_one({"username": username})
