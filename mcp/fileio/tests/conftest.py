"""Pytest configuration and shared fixtures for FileIO MCP Server tests."""

import asyncio
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest

from config import FileIOConfig
from directory_ops import DirectoryOperations
from file_ops import FileOperations
from mcp.fileio.server import MCPFileIOServer

# Removed event_loop fixture as it conflicts with pytest-asyncio auto mode


@pytest.fixture
def temp_base_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        base_path = Path(temp_dir)

        # Create standard workflow directories
        (base_path / "ingress").mkdir()
        (base_path / "wip").mkdir()
        (base_path / "completed").mkdir()

        yield base_path


@pytest.fixture
def test_config(temp_base_dir: Path) -> FileIOConfig:
    """Create a test configuration."""
    config_data = {
        "name": "fileio-test",
        "version": "1.0.0",
        "base_path": str(temp_base_dir),
        "allowed_directories": ["ingress", "wip", "completed"],
        "max_file_size": 1048576,  # 1MB for tests
        "allowed_extensions": [
            ".txt",
            ".json",
            ".csv",
            ".md",
            ".log",
            ".xml",
            ".yaml",
            ".yml",
            ".py",
            ".js",
            ".html",
            ".css",
            ".zip",
        ],
        "security": {"enable_write": True, "enable_delete": True, "sandbox_mode": True},
        "logging": {"level": "DEBUG", "file": "/tmp/fileio-test.log"},
        "server": {"host": "127.0.0.1", "port": 18000},
    }

    # Write temporary config file
    config_file = temp_base_dir / "test_config.json"
    with open(config_file, "w") as f:
        json.dump(config_data, f, indent=2)

    return FileIOConfig.load(str(config_file))


@pytest.fixture
def file_ops(test_config: FileIOConfig) -> FileOperations:
    """Create FileOperations instance with test config."""
    return FileOperations(test_config)


@pytest.fixture
def dir_ops(test_config: FileIOConfig) -> DirectoryOperations:
    """Create DirectoryOperations instance with test config."""
    return DirectoryOperations(test_config)


@pytest.fixture
async def mcp_server(
    test_config: FileIOConfig,
) -> AsyncGenerator[MCPFileIOServer, None]:
    """Create MCP server instance for testing."""
    # Mock MongoDB for tests
    server = MCPFileIOServer.__new__(MCPFileIOServer)
    server.config = test_config
    server.logger = MagicMock()
    server.mongodb_client = None
    server.database = None
    server.file_ops = FileOperations(test_config)
    server.dir_ops = DirectoryOperations(test_config)
    server.initialized = False
    server.client_capabilities = {}
    server.protocol_version = "2024-11-05"
    server._file_locks = {}

    # Mock log_execution to avoid MongoDB dependency
    server.log_execution = AsyncMock()

    yield server


@pytest.fixture
def sample_files(temp_base_dir: Path) -> dict:
    """Create sample test files."""
    files = {}

    # Text file
    text_file = temp_base_dir / "ingress" / "sample.txt"
    text_content = "Hello, World!\nThis is a test file.\nLine 3 of content.\n"
    text_file.write_text(text_content)
    files["text"] = {"path": text_file, "content": text_content}

    # JSON file
    json_file = temp_base_dir / "ingress" / "data.json"
    json_content = {"name": "test", "value": 42, "items": [1, 2, 3]}
    with open(json_file, "w") as f:
        json.dump(json_content, f, indent=2)
    files["json"] = {"path": json_file, "content": json_content}

    # Binary-like file (CSV)
    csv_file = temp_base_dir / "ingress" / "data.csv"
    csv_content = "name,age,city\nJohn,30,NYC\nJane,25,LA\n"
    csv_file.write_text(csv_content)
    files["csv"] = {"path": csv_file, "content": csv_content}

    # Large file for size testing
    large_file = temp_base_dir / "ingress" / "large.txt"
    large_content = "x" * 1000  # 1KB file
    large_file.write_text(large_content)
    files["large"] = {"path": large_file, "content": large_content}

    # Empty file
    empty_file = temp_base_dir / "ingress" / "empty.txt"
    empty_file.touch()
    files["empty"] = {"path": empty_file, "content": ""}

    return files


@pytest.fixture
def sample_archive(temp_base_dir: Path, sample_files: dict) -> Path:
    """Create a sample ZIP archive for testing."""
    archive_path = temp_base_dir / "completed" / "test_archive.zip"

    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(sample_files["text"]["path"], "sample.txt")
        zipf.write(sample_files["json"]["path"], "data.json")

    return archive_path


@pytest.fixture
def mock_mongodb():
    """Mock MongoDB for testing."""
    mock_client = AsyncMock()
    mock_db = AsyncMock()
    mock_collection = AsyncMock()

    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_db.command = AsyncMock(return_value={"ok": 1})
    mock_db.mcp_executions = mock_collection
    mock_collection.insert_one = AsyncMock()

    return {"client": mock_client, "database": mock_db, "collection": mock_collection}


@pytest.fixture(autouse=True)
def clean_environment():
    """Clean environment variables before each test."""
    original_env = os.environ.copy()
    yield
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def file_lock_timeout():
    """Short timeout for file lock tests."""
    return 1.0  # 1 second timeout for faster tests


# Helper functions for test data generation
def generate_test_content(size_kb: int = 1) -> str:
    """Generate test content of specified size."""
    content = "Test data line {}\n"
    line_size = len(content.format(0))
    lines_needed = (size_kb * 1024) // line_size

    return "".join(content.format(i) for i in range(lines_needed))


def create_nested_structure(base_path: Path, structure: dict):
    """Create nested directory structure for testing."""
    for name, content in structure.items():
        path = base_path / name

        if isinstance(content, dict):
            path.mkdir(exist_ok=True)
            create_nested_structure(path, content)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, str):
                path.write_text(content)
            else:
                path.write_bytes(content)
