"""Configuration management for FileIO MCP Server."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class SecurityConfig:
    """Security configuration settings."""

    enable_write: bool
    enable_delete: bool
    sandbox_mode: bool


@dataclass
class LoggingConfig:
    """Logging configuration settings."""

    level: str
    file: str


@dataclass
class ServerConfig:
    """Server configuration settings."""

    host: str
    port: int


@dataclass
class FileIOConfig:
    """Main configuration for FileIO MCP Server."""

    name: str
    version: str
    base_path: Path
    allowed_directories: List[str]
    max_file_size: int
    allowed_extensions: List[str]
    security: SecurityConfig
    logging: LoggingConfig
    server: ServerConfig

    @classmethod
    def load(cls, config_path: str = "config.json"):
        """Load configuration from JSON file with environment variable overrides."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_file) as f:
            data = json.load(f)

        # Override with environment variables if present
        base_path = os.getenv("FILEIO_BASE_PATH", data["base_path"])
        allowed_directories = os.getenv(
            "FILEIO_ALLOWED_DIRS", ",".join(data["allowed_directories"])
        ).split(",")
        max_file_size = int(os.getenv("FILEIO_MAX_FILE_SIZE", data["max_file_size"]))

        # Server configuration from environment
        server_host = os.getenv("FILEIO_HOST", data["server"]["host"])
        server_port = int(os.getenv("FILEIO_PORT", data["server"]["port"]))

        return cls(
            name=data["name"],
            version=data["version"],
            base_path=Path(base_path),
            allowed_directories=[d.strip() for d in allowed_directories],
            max_file_size=max_file_size,
            allowed_extensions=data["allowed_extensions"],
            security=SecurityConfig(**data["security"]),
            logging=LoggingConfig(**data["logging"]),
            server=ServerConfig(host=server_host, port=server_port),
        )

    def validate(self) -> bool:
        """Validate configuration settings."""
        if not self.base_path.exists():
            raise ValueError(f"Base path does not exist: {self.base_path}")

        for directory in self.allowed_directories:
            dir_path = self.base_path / directory
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)

        return True
