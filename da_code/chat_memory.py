"""Flexible chat memory management for da_code with PostgreSQL integration and fallbacks."""

import logging
import os
from typing import Optional

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_postgres import PostgresChatMessageHistory
from langchain_community.chat_message_histories import FileChatMessageHistory
from pathlib import Path

logger = logging.getLogger(__name__)



class ChatMemoryManager:
    """Flexible chat memory manager with PostgreSQL primary, in-memory fallback."""

    def __init__(self, session_id: str):
        """Initialize memory manager with session ID."""
        self.session_id = session_id
        self.chat_history: Optional[BaseChatMessageHistory] = None
        self.memory_type = "unknown"
        self._create_chat_history()

    def get_chat_history(self) -> BaseChatMessageHistory:
        """Get chat history with automatic fallback strategy."""
        if self.chat_history is None:
            self.chat_history = self._create_chat_history()

        return self.chat_history

    def _create_chat_history(self) -> BaseChatMessageHistory:
        """Create chat history with fallback strategy: PostgreSQL -> File -> In-Memory."""
        logger.debug(f"Creating chat history for session: {self.session_id}")

        # Strategy 1: PostgreSQL (preferred for production)
        logger.debug("Attempting PostgreSQL chat memory...")
        postgres_history = self._try_postgres()
        if postgres_history:
            logger.debug("PostgreSQL chat memory successfully created")
            return postgres_history

        # Strategy 2: File-based (persistence without database)
        logger.debug("PostgreSQL failed, attempting file-based chat memory...")
        file_history = self._try_file()
        if file_history:
            logger.debug("File-based chat memory successfully created")
            return file_history

        # Strategy 3: In-memory (fallback)
        logger.debug("All persistent storage failed, using in-memory chat history")
        logger.info("Using in-memory chat history (no persistence)")
        self.memory_type = "memory"
        return ChatMessageHistory()

    def _try_postgres(self) -> Optional[BaseChatMessageHistory]:
        """Try to create PostgreSQL chat history."""

        # Check for PostgreSQL environment variables
        postgres_url = os.getenv("POSTGRES_CHAT_URL", None)
        logger.debug(f"Checking PostgreSQL chat memory with URL: {postgres_url and 'configured' or 'not configured'}")

        if not postgres_url:
            logger.info("❌ PostgreSQL chat memory: POSTGRES_CHAT_URL not found, falling back to file/memory storage")
            return None

        try:
            logger.debug(f"Attempting PostgreSQL connection for session: {self.session_id}")
            # Log the actual URL being used (mask password)
            if postgres_url and '@' in postgres_url:
                url_parts = postgres_url.split('@')
                masked_url = url_parts[0].split(':')[:-1] + ['***'] + ['@'] + url_parts[1:]
                logger.debug(f"Connection URL: {''.join(masked_url)}")

            # Test the connection and create chat history
            # Create connection from URL and pass to PostgresChatMessageHistory
            import psycopg
            logger.debug("Creating psycopg connection...")

            # Try to force IPv4 by replacing localhost with 127.0.0.1
            if 'localhost' in postgres_url:
                ipv4_url = postgres_url.replace('localhost', '127.0.0.1')
                logger.debug("Trying IPv4 connection (127.0.0.1 instead of localhost)")
                sync_connection = psycopg.connect(ipv4_url)
            else:
                sync_connection = psycopg.connect(postgres_url)

            # Create table if it doesn't exist
            PostgresChatMessageHistory.create_tables(sync_connection, "da_code_chat_history")

            chat_history = PostgresChatMessageHistory(
                "da_code_chat_history",  # table_name (positional)
                self.session_id,         # session_id (positional)
                sync_connection=sync_connection  # connection (keyword)
            )

            # Try to access the chat history to test connection
            logger.debug("Testing PostgreSQL connection by accessing messages...")
            messages = chat_history.messages  # This will trigger connection test
            logger.debug(f"PostgreSQL connection successful, found {len(messages)} existing messages")

            logger.info(f"✅ PostgreSQL chat memory connected (session: {self.session_id})")
            self.memory_type = "postgres"
            return chat_history

        except Exception as e:
            logger.warning(f"❌ PostgreSQL chat memory connection failed: {type(e).__name__}: {e}")
            logger.debug(f"PostgreSQL connection error details: {e}", exc_info=True)
            return None

    def _try_file(self) -> Optional[BaseChatMessageHistory]:
        """Try to create file-based chat history."""
        try:
            # Create chat memory directory
            chat_memory_dir_path = os.getenv("DA_CODE_CHAT_MEMORY_DIR", "./da_code_chat_memory")
            logger.debug(f"Attempting file-based chat history in directory: {chat_memory_dir_path}")

            chat_memory_dir = Path(chat_memory_dir_path)
            chat_memory_dir.mkdir(exist_ok=True, parents=True)
            logger.debug(f"Chat memory directory created/verified: {chat_memory_dir.absolute()}")

            session_file = chat_memory_dir / f"{self.session_id}.json"
            logger.debug(f"Session file path: {session_file.absolute()}")

            chat_history = FileChatMessageHistory(str(session_file))

            # Test if file exists and check message count
            if session_file.exists():
                try:
                    existing_messages = chat_history.messages
                    logger.debug(f"Found existing file with {len(existing_messages)} messages")
                except Exception as e:
                    logger.debug(f"Error reading existing file: {e}")

            logger.info(f"✅ Using file-based chat history: {session_file}")
            self.memory_type = "file"
            return chat_history

        except Exception as e:
            logger.warning(f"❌ File-based chat history failed: {type(e).__name__}: {e}")
            logger.debug(f"File-based chat history error details: {e}", exc_info=True)
            return None

    def get_memory_info(self) -> dict:
        """Get information about the current memory configuration."""
        return {
            "memory_type": self.memory_type,
            "session_id": self.session_id,
            "persistent": self.memory_type in ["postgres", "redis", "file"],
            "shared": self.memory_type in ["postgres", "redis"],
            "message_count": len(self.chat_history.messages) if self.chat_history else 0
        }

    def clear_history(self) -> bool:
        """Clear chat history for current session."""
        try:
            if self.chat_history:
                self.chat_history.clear()
                logger.info(f"Cleared chat history for session {self.session_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to clear chat history: {e}")
            return False

    def export_history(self) -> list:
        """Export chat history as a list of message dicts."""
        try:
            if self.chat_history:
                messages = []
                for msg in self.chat_history.messages:
                    messages.append({
                        "type": msg.__class__.__name__,
                        "content": msg.content,
                        "timestamp": getattr(msg, "timestamp", None)
                    })
                return messages
            return []
        except Exception as e:
            logger.error(f"Failed to export chat history: {e}")
            return []


def create_chat_memory_manager(session_id: str) -> ChatMemoryManager:
    """Factory function to create a chat memory manager."""
    return ChatMemoryManager(session_id)

