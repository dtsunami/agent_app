"""Asynchronous unit tests for FileOperations class."""

import asyncio
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from mcp import types

from config import FileIOConfig
from file_ops import FileOperations

# All tests in this file are async - using pytest-asyncio auto mode
pytestmark = pytest.mark.asyncio


async def test_read_file_success(file_ops: FileOperations, sample_files: dict):
    """Test successful file reading."""
    result = await file_ops._read_file({"directory": "ingress", "path": "sample.txt"})

    assert len(result) == 1
    assert isinstance(result[0], types.TextContent)
    assert result[0].text == sample_files["text"]["content"]


async def test_read_file_not_found(file_ops: FileOperations):
    """Test reading non-existent file."""
    result = await file_ops._read_file(
        {"directory": "ingress", "path": "nonexistent.txt"}
    )

    assert len(result) == 1
    assert "File not found" in result[0].text


async def test_read_file_directory_not_file(
    file_ops: FileOperations, temp_base_dir: Path
):
    """Test reading a directory instead of file."""
    (temp_base_dir / "ingress" / "testdir").mkdir()

    result = await file_ops._read_file({"directory": "ingress", "path": "testdir"})

    assert len(result) == 1
    assert "Path is not a file" in result[0].text


async def test_read_file_encoding(file_ops: FileOperations, temp_base_dir: Path):
    """Test reading file with different encoding."""
    test_file = temp_base_dir / "ingress" / "utf8.txt"
    test_content = "Hello 世界! 🌍"
    test_file.write_text(test_content, encoding="utf-8")

    result = await file_ops._read_file(
        {"directory": "ingress", "path": "utf8.txt", "encoding": "utf-8"}
    )

    assert len(result) == 1
    assert result[0].text == test_content


async def test_write_file_success(file_ops: FileOperations, temp_base_dir: Path):
    """Test successful file writing."""
    test_content = "New file content"

    result = await file_ops._write_file(
        {"directory": "wip", "path": "new_file.txt", "content": test_content}
    )

    assert len(result) == 1
    assert "File written successfully" in result[0].text

    # Verify file was created
    written_file = temp_base_dir / "wip" / "new_file.txt"
    assert written_file.exists()
    assert written_file.read_text() == test_content


async def test_write_file_create_directories(
    file_ops: FileOperations, temp_base_dir: Path
):
    """Test writing file with directory creation."""
    test_content = "Content in nested dir"

    result = await file_ops._write_file(
        {
            "directory": "wip",
            "path": "nested/dir/file.txt",
            "content": test_content,
            "create_dirs": True,
        }
    )

    assert len(result) == 1
    assert "File written successfully" in result[0].text

    # Verify nested structure was created
    written_file = temp_base_dir / "wip" / "nested" / "dir" / "file.txt"
    assert written_file.exists()
    assert written_file.read_text() == test_content


async def test_append_to_file_success(file_ops: FileOperations, sample_files: dict):
    """Test successful file appending."""
    append_content = "\nAppended line"

    result = await file_ops._append_to_file(
        {"directory": "ingress", "path": "sample.txt", "content": append_content}
    )

    assert len(result) == 1
    assert "Content appended" in result[0].text

    # Verify content was appended
    file_content = sample_files["text"]["path"].read_text()
    assert file_content.endswith(append_content)


async def test_append_to_nonexistent_file(file_ops: FileOperations):
    """Test appending to non-existent file."""
    result = await file_ops._append_to_file(
        {"directory": "wip", "path": "nonexistent.txt", "content": "test"}
    )

    assert len(result) == 1
    assert "File not found" in result[0].text


async def test_delete_file_success(file_ops: FileOperations, sample_files: dict):
    """Test successful file deletion."""
    result = await file_ops._delete_file(
        {"directory": "ingress", "path": "empty.txt", "confirm": True}
    )

    assert len(result) == 1
    assert "File deleted successfully" in result[0].text
    assert not sample_files["empty"]["path"].exists()


async def test_delete_file_requires_confirmation(file_ops: FileOperations):
    """Test deletion requires confirmation."""
    result = await file_ops._delete_file(
        {"directory": "ingress", "path": "sample.txt", "confirm": False}
    )

    assert len(result) == 1
    assert "confirmation flag" in result[0].text


async def test_get_file_info(file_ops: FileOperations, sample_files: dict):
    """Test getting file information."""
    result = await file_ops._get_file_info(
        {"directory": "ingress", "path": "sample.txt"}
    )

    assert len(result) == 1
    assert "File Information:" in result[0].text
    # Should contain JSON with file metadata
    assert '"size"' in result[0].text
    assert '"modified"' in result[0].text


async def test_check_file_exists_true(file_ops: FileOperations, sample_files: dict):
    """Test checking existing file."""
    result = await file_ops._check_file_exists(
        {"directory": "ingress", "path": "sample.txt"}
    )

    assert len(result) == 1
    assert '"exists": true' in result[0].text
    assert '"is_file": true' in result[0].text


async def test_check_file_exists_false(file_ops: FileOperations):
    """Test checking non-existent file."""
    result = await file_ops._check_file_exists(
        {"directory": "ingress", "path": "nonexistent.txt"}
    )

    assert len(result) == 1
    assert '"exists": false' in result[0].text
    assert '"is_file": false' in result[0].text


async def test_copy_file_success(
    file_ops: FileOperations, sample_files: dict, temp_base_dir: Path
):
    """Test successful file copying."""
    result = await file_ops._copy_file(
        {
            "source_directory": "ingress",
            "source_path": "sample.txt",
            "target_directory": "wip",
            "target_path": "copied_sample.txt",
        }
    )

    assert len(result) == 1
    assert "File copied successfully" in result[0].text

    # Verify file was copied
    target_file = temp_base_dir / "wip" / "copied_sample.txt"
    assert target_file.exists()
    assert target_file.read_text() == sample_files["text"]["content"]

    # Verify original still exists
    assert sample_files["text"]["path"].exists()


async def test_copy_file_overwrite_protection(
    file_ops: FileOperations, sample_files: dict, temp_base_dir: Path
):
    """Test copy operation overwrite protection."""
    # Create target file first
    target_file = temp_base_dir / "wip" / "existing.txt"
    target_file.write_text("existing content")

    result = await file_ops._copy_file(
        {
            "source_directory": "ingress",
            "source_path": "sample.txt",
            "target_directory": "wip",
            "target_path": "existing.txt",
            "overwrite": False,
        }
    )

    assert len(result) == 1
    assert "Target file exists and overwrite=false" in result[0].text

    # Verify original content preserved
    assert target_file.read_text() == "existing content"


async def test_copy_file_with_overwrite(
    file_ops: FileOperations, sample_files: dict, temp_base_dir: Path
):
    """Test copy operation with overwrite enabled."""
    # Create target file first
    target_file = temp_base_dir / "wip" / "existing.txt"
    target_file.write_text("existing content")

    result = await file_ops._copy_file(
        {
            "source_directory": "ingress",
            "source_path": "sample.txt",
            "target_directory": "wip",
            "target_path": "existing.txt",
            "overwrite": True,
        }
    )

    assert len(result) == 1
    assert "File copied successfully" in result[0].text

    # Verify content was overwritten
    assert target_file.read_text() == sample_files["text"]["content"]


async def test_move_file_success(
    file_ops: FileOperations, sample_files: dict, temp_base_dir: Path
):
    """Test successful file moving."""
    # Create a file to move
    source_file = temp_base_dir / "ingress" / "to_move.txt"
    source_content = "Content to move"
    source_file.write_text(source_content)

    result = await file_ops._move_file(
        {
            "source_directory": "ingress",
            "source_path": "to_move.txt",
            "target_directory": "completed",
            "target_path": "moved_file.txt",
        }
    )

    assert len(result) == 1
    assert "File moved successfully" in result[0].text

    # Verify file was moved
    target_file = temp_base_dir / "completed" / "moved_file.txt"
    assert target_file.exists()
    assert target_file.read_text() == source_content

    # Verify source no longer exists
    assert not source_file.exists()


async def test_compress_file_success(
    file_ops: FileOperations, sample_files: dict, temp_base_dir: Path
):
    """Test successful file compression."""
    result = await file_ops._compress_file(
        {
            "files": [
                {"directory": "ingress", "path": "sample.txt"},
                {"directory": "ingress", "path": "data.json"},
            ],
            "archive_directory": "completed",
            "archive_path": "test_archive.zip",
        }
    )

    assert len(result) == 1
    assert "Archive created successfully" in result[0].text
    assert "(2 files)" in result[0].text

    # Verify archive was created
    archive_file = temp_base_dir / "completed" / "test_archive.zip"
    assert archive_file.exists()

    # Verify archive contents
    with zipfile.ZipFile(archive_file, "r") as zipf:
        files_in_archive = zipf.namelist()
        assert "sample.txt" in files_in_archive
        assert "data.json" in files_in_archive


async def test_compress_file_nonexistent_source(file_ops: FileOperations):
    """Test compression with non-existent source file."""
    result = await file_ops._compress_file(
        {
            "files": [{"directory": "ingress", "path": "nonexistent.txt"}],
            "archive_directory": "completed",
            "archive_path": "test_archive.zip",
        }
    )

    assert len(result) == 1
    assert "Source file not found" in result[0].text


async def test_extract_file_success(
    file_ops: FileOperations, sample_archive: Path, temp_base_dir: Path
):
    """Test successful file extraction."""
    result = await file_ops._extract_file(
        {
            "archive_directory": "completed",
            "archive_path": "test_archive.zip",
            "extract_directory": "wip",
            "extract_path": "extracted",
        }
    )

    assert len(result) == 1
    assert "Archive extracted successfully" in result[0].text
    assert "2 files" in result[0].text

    # Verify extracted files
    extract_dir = temp_base_dir / "wip" / "extracted"
    assert (extract_dir / "sample.txt").exists()
    assert (extract_dir / "data.json").exists()


async def test_extract_file_security_check(
    file_ops: FileOperations, temp_base_dir: Path
):
    """Test extraction security against path traversal."""
    # Create malicious archive
    malicious_archive = temp_base_dir / "completed" / "malicious.zip"
    with zipfile.ZipFile(malicious_archive, "w") as zipf:
        zipf.writestr("../../../etc/passwd", "malicious content")

    result = await file_ops._extract_file(
        {
            "archive_directory": "completed",
            "archive_path": "malicious.zip",
            "extract_directory": "wip",
            "extract_path": "extracted",
        }
    )

    assert len(result) == 1
    assert "Unsafe path in archive" in result[0].text


async def test_file_lock_list_empty(file_ops: FileOperations):
    """Test listing active locks when none exist."""
    result = await file_ops._file_lock({"action": "list_active"})

    assert len(result) == 1
    assert "No active file locks" in result[0].text


async def test_file_lock_status_unlocked(file_ops: FileOperations, sample_files: dict):
    """Test checking status of unlocked file."""
    result = await file_ops._file_lock(
        {"action": "status", "directory": "ingress", "path": "sample.txt"}
    )

    assert len(result) == 1
    assert "is unlocked" in result[0].text


async def test_file_locking_context_manager(
    file_ops: FileOperations, sample_files: dict
):
    """Test file locking context manager."""
    file_path = sample_files["text"]["path"]

    # Test that lock is acquired and released
    async def test_lock():
        with file_ops._acquire_file_lock(file_path, timeout=1.0):
            # Check lock is active
            assert str(file_path) in file_ops._file_locks
            # Simulate some work
            await asyncio.sleep(0.1)

        # After context, lock should be released
        assert str(file_path) not in file_ops._file_locks

    await test_lock()


async def test_file_lock_timeout(file_ops: FileOperations, sample_files: dict):
    """Test file lock timeout behavior."""
    file_path = sample_files["text"]["path"]

    # First acquire a lock
    try:
        with file_ops._acquire_file_lock(file_path, timeout=1.0):
            # Try to acquire the same lock from another context (should timeout)
            with pytest.raises((TimeoutError, FileExistsError)):
                with file_ops._acquire_file_lock(file_path, timeout=0.1):
                    pass
    except FileExistsError:
        # If file already locked, test that timeout occurs for subsequent locks
        with pytest.raises((TimeoutError, FileExistsError)):
            with file_ops._acquire_file_lock(file_path, timeout=0.1):
                pass


async def test_file_lock_concurrent_operations(
    file_ops: FileOperations, sample_files: dict, temp_base_dir: Path
):
    """Test that copy operations use file locking properly."""
    # This test verifies that copy operations acquire locks
    # We can't easily test race conditions in unit tests, but we can verify the lock mechanism is called

    with patch.object(file_ops, "_acquire_file_lock") as mock_lock:
        mock_lock.return_value.__enter__ = MagicMock()
        mock_lock.return_value.__exit__ = MagicMock()

        await file_ops._copy_file(
            {
                "source_directory": "ingress",
                "source_path": "sample.txt",
                "target_directory": "wip",
                "target_path": "copied.txt",
            }
        )

        # Verify locks were acquired for both source and target
        assert mock_lock.call_count == 2


async def test_invalid_directory(file_ops: FileOperations):
    """Test operations with invalid directory."""
    with pytest.raises(ValueError, match="Directory not allowed"):
        await file_ops._read_file({"directory": "invalid_dir", "path": "test.txt"})


async def test_path_traversal_prevention(file_ops: FileOperations):
    """Test path traversal attack prevention."""
    with pytest.raises(ValueError, match="Path escape attempt"):
        await file_ops._read_file(
            {"directory": "ingress", "path": "../../../etc/passwd"}
        )


async def test_file_size_limit(file_ops: FileOperations, temp_base_dir: Path):
    """Test file size limit enforcement."""
    # Create a large file that exceeds the limit
    large_file = temp_base_dir / "ingress" / "too_large.txt"
    large_content = "x" * (file_ops.config.max_file_size + 1)
    large_file.write_text(large_content)

    result = await file_ops._read_file(
        {"directory": "ingress", "path": "too_large.txt"}
    )

    assert len(result) == 1
    assert "File too large" in result[0].text


async def test_binary_file_handling(file_ops: FileOperations, temp_base_dir: Path):
    """Test handling of binary files."""
    # Create a binary file
    binary_file = temp_base_dir / "ingress" / "binary.bin"
    binary_file.write_bytes(bytes(range(256)))

    result = await file_ops._read_file({"directory": "ingress", "path": "binary.bin"})

    assert len(result) == 1
    assert (
        "Cannot decode file" in result[0].text or "encoding" in result[0].text.lower()
    )
