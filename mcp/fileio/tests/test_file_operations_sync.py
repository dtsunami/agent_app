"""Synchronous unit tests for FileOperations class."""

from pathlib import Path

import pytest

from config import FileIOConfig
from file_ops import FileOperations


def test_get_tools_returns_all_operations(file_ops: FileOperations):
    """Test that get_tools returns all expected operations."""
    tools = file_ops.get_tools()
    tool_names = [tool.name for tool in tools]

    expected_tools = [
        "read_file",
        "get_file_info",
        "check_file_exists",
        "write_file",
        "append_to_file",
        "delete_file",
        "copy_file",
        "move_file",
        "compress_file",
        "extract_file",
        "file_lock",
    ]

    for expected_tool in expected_tools:
        assert expected_tool in tool_names


def test_tools_have_valid_schemas(file_ops: FileOperations):
    """Test that all tools have valid JSON schemas."""
    tools = file_ops.get_tools()

    for tool in tools:
        assert hasattr(tool, "name")
        assert hasattr(tool, "description")
        assert hasattr(tool, "inputSchema")

        # Verify schema has required structure
        schema = tool.inputSchema
        assert isinstance(schema, dict)
        assert "type" in schema
        assert schema["type"] == "object"
        assert "properties" in schema


def test_conditional_tools_based_on_config(
    test_config: FileIOConfig, temp_base_dir: Path
):
    """Test that tools are conditionally included based on config."""
    # Test with write disabled
    test_config.security.enable_write = False
    file_ops_no_write = FileOperations(test_config)
    tools_no_write = [tool.name for tool in file_ops_no_write.get_tools()]

    assert "write_file" not in tools_no_write
    assert "append_to_file" not in tools_no_write

    # Test with delete disabled
    test_config.security.enable_write = True
    test_config.security.enable_delete = False
    file_ops_no_delete = FileOperations(test_config)
    tools_no_delete = [tool.name for tool in file_ops_no_delete.get_tools()]

    assert "delete_file" not in tools_no_delete

    # Advanced operations should always be available
    assert "copy_file" in tools_no_delete
    assert "move_file" in tools_no_delete
