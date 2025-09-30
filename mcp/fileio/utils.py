"""Utility functions for FileIO MCP Server."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Union


def human_size(bytes_size: int) -> str:
    """Convert bytes to human readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_size < 1024:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f} PB"


def format_timestamp(timestamp: float) -> str:
    """Format Unix timestamp to readable string."""
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def safe_json_dumps(obj: Any, indent: int = 2) -> str:
    """Safely serialize object to JSON string."""
    try:
        return json.dumps(obj, indent=indent, default=str)
    except Exception as e:
        return f"Error serializing to JSON: {str(e)}"


def validate_file_extension(file_path: Path, allowed_extensions: list) -> bool:
    """Check if file extension is allowed."""
    if not allowed_extensions:
        return True
    return file_path.suffix.lower() in [ext.lower() for ext in allowed_extensions]


def get_mime_type(file_path: Path) -> str:
    """Get MIME type for file."""
    import mimetypes

    mime_type, _ = mimetypes.guess_type(str(file_path))
    return mime_type or "application/octet-stream"


def create_file_info(file_path: Path, base_path: Path) -> Dict[str, Any]:
    """Create file information dictionary."""
    try:
        stat = file_path.stat()
        return {
            "name": file_path.name,
            "path": str(file_path.relative_to(base_path)),
            "size": stat.st_size,
            "size_human": human_size(stat.st_size),
            "modified": format_timestamp(stat.st_mtime),
            "created": format_timestamp(stat.st_ctime),
            "is_file": file_path.is_file(),
            "is_directory": file_path.is_dir(),
            "extension": file_path.suffix,
            "mime_type": get_mime_type(file_path) if file_path.is_file() else None,
        }
    except Exception as e:
        return {"name": file_path.name, "error": str(e)}
