"""Configuration management for ToolSession MCP Server."""

import json
import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field


class ServerConfig(BaseModel):
    """Server configuration."""
    host: str = Field(default="0.0.0.0", description="Host to bind to")
    port: int = Field(default=8002, description="Port to bind to")


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = Field(default="INFO", description="Log level")
    file: str = Field(default="/tmp/toolsession_mcp.log", description="Log file path")


class SessionConfig(BaseModel):
    """Session configuration."""
    command: str = Field(default="python -i -u", description="Command to run in session")
    working_directory: str = Field(default="/tmp", description="Working directory")
    prompt_string: str = Field(default=">>> ", description="Prompt string to detect")
    timeout: int = Field(default=30, description="Command timeout in seconds")


class ToolSessionConfig(BaseModel):
    """Complete ToolSession configuration."""
    server: ServerConfig = Field(default_factory=ServerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> 'ToolSessionConfig':
        """Load configuration from file and environment variables."""
        config_data = {}

        # Load from file if provided
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config_data = json.load(f)

        # Override with environment variables
        config_data = cls._apply_env_overrides(config_data)

        return cls(**config_data)

    @classmethod
    def _apply_env_overrides(cls, config_data: dict) -> dict:
        """Apply environment variable overrides."""
        # Server config
        if "server" not in config_data:
            config_data["server"] = {}

        if os.getenv("TOOLSESSION_HOST"):
            config_data["server"]["host"] = os.getenv("TOOLSESSION_HOST")
        if os.getenv("TOOLSESSION_PORT"):
            config_data["server"]["port"] = int(os.getenv("TOOLSESSION_PORT"))

        # Logging config
        if "logging" not in config_data:
            config_data["logging"] = {}

        if os.getenv("TOOLSESSION_LOG_LEVEL"):
            config_data["logging"]["level"] = os.getenv("TOOLSESSION_LOG_LEVEL", "INFO")
        if os.getenv("TOOLSESSION_LOG_FILE"):
            config_data["logging"]["file"] = os.getenv("TOOLSESSION_LOG_FILE", "toolsession_mcp.log")

        # Session config
        if "session" not in config_data:
            config_data["session"] = {}

        if os.getenv("TOOLSESSION_COMMAND"):
            config_data["session"]["command"] = os.getenv("TOOLSESSION_COMMAND")
        if os.getenv("TOOLSESSION_WORKING_DIR"):
            config_data["session"]["working_directory"] = os.getenv("TOOLSESSION_WORKING_DIR")
        if os.getenv("TOOLSESSION_PROMPT"):
            config_data["session"]["prompt_string"] = os.getenv("TOOLSESSION_PROMPT")
        if os.getenv("TOOLSESSION_TIMEOUT"):
            config_data["session"]["timeout"] = int(os.getenv("TOOLSESSION_TIMEOUT"))

        return config_data