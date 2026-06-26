# OpenCortex
# src/auth.py — User authentication layer.
# Handles registration (bcrypt-hashed password storage) and login verification.
# Depends on MongoManager for the underlying users collection.

import bcrypt
from utils.logger import setup_logger

logger = setup_logger("auth")


class AuthManager:
    """Create and verify user credentials against MongoDB."""

    def __init__(self, db):
        self.db = db

    def create_user(self, username, password):
        """Register a new user. Returns (success: bool, message: str)."""
        if not self.db.is_connected:
            return False, "Database connection unavailable."

        users = self.db.get_collection("users")
        if users.find_one({"username": username}):
            return False, "Username already exists."

        # Hash the password with a per-user salt before persisting
        hashed_pw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
        users.insert_one({"username": username, "password": hashed_pw})
        logger.info(f"User {username} created successfully.")
        return True, "User created successfully."

    def verify_user(self, username, password):
        """Authenticate against stored hash. Returns (success: bool, message: str)."""
        if not self.db.is_connected:
            return False, "Database connection unavailable."

        users = self.db.get_collection("users")
        user = users.find_one({"username": username})
        if not user:
            return False, "User not found."

        if bcrypt.checkpw(password.encode("utf-8"), user["password"]):
            logger.info(f"User {username} logged in successfully.")
            return True, "Login successful."

        logger.error(f"User {username} login failed.")
        return False, "Invalid password."
