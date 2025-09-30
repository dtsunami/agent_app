"""File operations tools for FileIO MCP Server."""

import fcntl
import json
import os
import shutil
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

from mcp import types

from models import FileIOConfig
from utils import create_file_info, safe_json_dumps, validate_file_extension


class FileOperations:
    """Handle file-level operations."""

    def __init__(self, config: FileIOConfig):
        self.config = config
        self._file_locks = {}  # Track active file locks

    @contextmanager
    def _acquire_file_lock(self, file_path: Path, timeout: float = 30.0):
        """Context manager for file locking."""
        lock_file = file_path.with_suffix(file_path.suffix + ".lock")
        lock_fd = None

        try:
            # Create lock file
            lock_fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)

            # Try to acquire lock with timeout
            start_time = time.time()
            while True:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.time() - start_time > timeout:
                        raise TimeoutError(
                            f"Could not acquire lock for {file_path} within {timeout}s"
                        )
                    time.sleep(0.1)

            # Store lock info
            self._file_locks[str(file_path)] = {
                "lock_file": lock_file,
                "lock_fd": lock_fd,
                "timestamp": time.time(),
            }

            yield

        finally:
            # Release lock
            if lock_fd is not None:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    os.close(lock_fd)
                except:
                    pass

            # Remove lock file
            try:
                lock_file.unlink(missing_ok=True)
            except:
                pass

            # Remove from tracking
            self._file_locks.pop(str(file_path), None)

    def get_tools(self) -> List[types.Tool]:
        """Get list of available file operation tools."""
        tools = [
            types.Tool(
                name="read_file",
                description="Read file content from workflow directories (ingress, wip, completed)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path relative to directory",
                        },
                        "directory": {
                            "type": "string",
                            "enum": self.config.allowed_directories,
                            "description": "Target directory (ingress, wip, or completed)",
                        },
                        "encoding": {
                            "type": "string",
                            "default": "utf-8",
                            "description": "Text encoding to use",
                        },
                    },
                    "required": ["path", "directory"],
                },
            ),
            types.Tool(
                name="get_file_info",
                description="Get detailed file metadata and information",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path relative to directory",
                        },
                        "directory": {
                            "type": "string",
                            "enum": self.config.allowed_directories,
                            "description": "Target directory",
                        },
                    },
                    "required": ["path", "directory"],
                },
            ),
            types.Tool(
                name="check_file_exists",
                description="Check if a file exists in the specified directory",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path relative to directory",
                        },
                        "directory": {
                            "type": "string",
                            "enum": self.config.allowed_directories,
                            "description": "Target directory",
                        },
                    },
                    "required": ["path", "directory"],
                },
            ),
        ]

        # Add write operations if enabled
        if self.config.security.enable_write:
            tools.extend(
                [
                    types.Tool(
                        name="write_file",
                        description="Write content to file in workflow directories",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": "File path relative to directory",
                                },
                                "content": {
                                    "type": "string",
                                    "description": "Content to write to file",
                                },
                                "directory": {
                                    "type": "string",
                                    "enum": self.config.allowed_directories,
                                    "description": "Target directory",
                                },
                                "encoding": {
                                    "type": "string",
                                    "default": "utf-8",
                                    "description": "Text encoding to use",
                                },
                                "create_dirs": {
                                    "type": "boolean",
                                    "default": True,
                                    "description": "Create parent directories if needed",
                                },
                            },
                            "required": ["path", "content", "directory"],
                        },
                    ),
                    types.Tool(
                        name="append_to_file",
                        description="Append content to existing file",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": "File path relative to directory",
                                },
                                "content": {
                                    "type": "string",
                                    "description": "Content to append to file",
                                },
                                "directory": {
                                    "type": "string",
                                    "enum": self.config.allowed_directories,
                                    "description": "Target directory",
                                },
                                "encoding": {
                                    "type": "string",
                                    "default": "utf-8",
                                    "description": "Text encoding to use",
                                },
                            },
                            "required": ["path", "content", "directory"],
                        },
                    ),
                ]
            )

        # Add delete operations if enabled
        if self.config.security.enable_delete:
            tools.append(
                types.Tool(
                    name="delete_file",
                    description="Delete a file from workflow directories (ingress, wip, completed)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "File path relative to directory",
                            },
                            "directory": {
                                "type": "string",
                                "enum": self.config.allowed_directories,
                                "description": "Target directory",
                            },
                            "confirm": {
                                "type": "boolean",
                                "default": False,
                                "description": "Confirmation flag required for deletion",
                            },
                        },
                        "required": ["path", "directory", "confirm"],
                    },
                )
            )

        # Add advanced file operations (always available)
        tools.extend(
            [
                types.Tool(
                    name="copy_file",
                    description="Copy file between workflow directories with locking",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "source_path": {
                                "type": "string",
                                "description": "Source file path relative to source directory",
                            },
                            "source_directory": {
                                "type": "string",
                                "enum": self.config.allowed_directories,
                                "description": "Source directory",
                            },
                            "target_path": {
                                "type": "string",
                                "description": "Target file path relative to target directory",
                            },
                            "target_directory": {
                                "type": "string",
                                "enum": self.config.allowed_directories,
                                "description": "Target directory",
                            },
                            "overwrite": {
                                "type": "boolean",
                                "default": False,
                                "description": "Overwrite target file if it exists",
                            },
                        },
                        "required": [
                            "source_path",
                            "source_directory",
                            "target_path",
                            "target_directory",
                        ],
                    },
                ),
                types.Tool(
                    name="move_file",
                    description="Move/rename file between workflow directories with locking",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "source_path": {
                                "type": "string",
                                "description": "Source file path relative to source directory",
                            },
                            "source_directory": {
                                "type": "string",
                                "enum": self.config.allowed_directories,
                                "description": "Source directory",
                            },
                            "target_path": {
                                "type": "string",
                                "description": "Target file path relative to target directory",
                            },
                            "target_directory": {
                                "type": "string",
                                "enum": self.config.allowed_directories,
                                "description": "Target directory",
                            },
                            "overwrite": {
                                "type": "boolean",
                                "default": False,
                                "description": "Overwrite target file if it exists",
                            },
                        },
                        "required": [
                            "source_path",
                            "source_directory",
                            "target_path",
                            "target_directory",
                        ],
                    },
                ),
                types.Tool(
                    name="compress_file",
                    description="Create ZIP archive from files",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "files": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "path": {"type": "string"},
                                        "directory": {
                                            "type": "string",
                                            "enum": self.config.allowed_directories,
                                        },
                                    },
                                    "required": ["path", "directory"],
                                },
                                "description": "List of files to compress",
                            },
                            "archive_path": {
                                "type": "string",
                                "description": "Output archive path relative to directory",
                            },
                            "archive_directory": {
                                "type": "string",
                                "enum": self.config.allowed_directories,
                                "description": "Directory for output archive",
                            },
                        },
                        "required": ["files", "archive_path", "archive_directory"],
                    },
                ),
                types.Tool(
                    name="extract_file",
                    description="Extract ZIP archive to directory",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "archive_path": {
                                "type": "string",
                                "description": "Archive file path relative to directory",
                            },
                            "archive_directory": {
                                "type": "string",
                                "enum": self.config.allowed_directories,
                                "description": "Directory containing archive",
                            },
                            "extract_directory": {
                                "type": "string",
                                "enum": self.config.allowed_directories,
                                "description": "Directory to extract files to",
                            },
                            "extract_path": {
                                "type": "string",
                                "default": ".",
                                "description": "Subdirectory path within extract_directory",
                            },
                        },
                        "required": [
                            "archive_path",
                            "archive_directory",
                            "extract_directory",
                        ],
                    },
                ),
                types.Tool(
                    name="file_lock",
                    description="Manage file locks for preventing race conditions",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "File path relative to directory",
                            },
                            "directory": {
                                "type": "string",
                                "enum": self.config.allowed_directories,
                                "description": "Target directory",
                            },
                            "action": {
                                "type": "string",
                                "enum": ["status", "list_active"],
                                "description": "Lock action: status (check if locked), list_active (list all active locks)",
                            },
                        },
                        "required": ["action"],
                    },
                ),
            ]
        )

        return tools

    async def execute(self, name: str, arguments: dict) -> List[types.TextContent]:
        """Execute file operation tool."""
        try:
            if name == "read_file":
                return await self._read_file(arguments)
            elif name == "get_file_info":
                return await self._get_file_info(arguments)
            elif name == "check_file_exists":
                return await self._check_file_exists(arguments)
            elif name == "write_file" and self.config.security.enable_write:
                return await self._write_file(arguments)
            elif name == "append_to_file" and self.config.security.enable_write:
                return await self._append_to_file(arguments)
            elif name == "delete_file" and self.config.security.enable_delete:
                return await self._delete_file(arguments)
            elif name == "copy_file":
                return await self._copy_file(arguments)
            elif name == "move_file":
                return await self._move_file(arguments)
            elif name == "compress_file":
                return await self._compress_file(arguments)
            elif name == "extract_file":
                return await self._extract_file(arguments)
            elif name == "file_lock":
                return await self._file_lock(arguments)
            else:
                return [
                    types.TextContent(
                        type="text", text=f"Unknown or disabled operation: {name}"
                    )
                ]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error in {name}: {str(e)}")]

    async def _read_file(self, args: dict) -> List[types.TextContent]:
        """Read file content."""
        file_path = self._get_safe_path(args["directory"], args["path"])

        if not file_path.exists():
            return [
                types.TextContent(
                    type="text",
                    text=f"File not found: {file_path.relative_to(self.config.base_path)}",
                )
            ]

        if not file_path.is_file():
            return [
                types.TextContent(
                    type="text",
                    text=f"Path is not a file: {file_path.relative_to(self.config.base_path)}",
                )
            ]

        # Check file size
        if file_path.stat().st_size > self.config.max_file_size:
            return [
                types.TextContent(
                    type="text",
                    text=f"File too large (max: {self.config.max_file_size} bytes)",
                )
            ]

        try:
            content = file_path.read_text(encoding=args.get("encoding", "utf-8"))
            return [types.TextContent(type="text", text=content)]
        except UnicodeDecodeError:
            return [
                types.TextContent(
                    type="text",
                    text=f"Cannot decode file with {args.get('encoding', 'utf-8')} encoding. File may be binary.",
                )
            ]

    async def _get_file_info(self, args: dict) -> List[types.TextContent]:
        """Get file information."""
        file_path = self._get_safe_path(args["directory"], args["path"])

        if not file_path.exists():
            return [
                types.TextContent(
                    type="text",
                    text=f"File not found: {file_path.relative_to(self.config.base_path)}",
                )
            ]

        info = create_file_info(file_path, self.config.base_path)
        return [
            types.TextContent(
                type="text", text=f"File Information:\n{safe_json_dumps(info)}"
            )
        ]

    async def _check_file_exists(self, args: dict) -> List[types.TextContent]:
        """Check if file exists."""
        file_path = self._get_safe_path(args["directory"], args["path"])

        exists = file_path.exists()
        is_file = file_path.is_file() if exists else False

        result = {
            "path": str(file_path.relative_to(self.config.base_path)),
            "exists": exists,
            "is_file": is_file,
            "is_directory": file_path.is_dir() if exists else False,
        }

        return [types.TextContent(type="text", text=safe_json_dumps(result))]

    async def _write_file(self, args: dict) -> List[types.TextContent]:
        """Write content to file."""
        file_path = self._get_safe_path(args["directory"], args["path"])

        # Check file extension
        if not validate_file_extension(file_path, self.config.allowed_extensions):
            return [
                types.TextContent(
                    type="text",
                    text=f"File extension not allowed: {file_path.suffix}. Allowed: {self.config.allowed_extensions}",
                )
            ]

        # Create parent directories if needed
        if args.get("create_dirs", True):
            file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            file_path.write_text(
                args["content"], encoding=args.get("encoding", "utf-8")
            )
            return [
                types.TextContent(
                    type="text",
                    text=f"File written successfully: {file_path.relative_to(self.config.base_path)}",
                )
            ]
        except Exception as e:
            return [
                types.TextContent(type="text", text=f"Error writing file: {str(e)}")
            ]

    async def _append_to_file(self, args: dict) -> List[types.TextContent]:
        """Append content to file."""
        file_path = self._get_safe_path(args["directory"], args["path"])

        if not file_path.exists():
            return [
                types.TextContent(
                    type="text",
                    text=f"File not found: {file_path.relative_to(self.config.base_path)}",
                )
            ]

        try:
            with open(file_path, "a", encoding=args.get("encoding", "utf-8")) as f:
                f.write(args["content"])

            return [
                types.TextContent(
                    type="text",
                    text=f"Content appended to: {file_path.relative_to(self.config.base_path)}",
                )
            ]
        except Exception as e:
            return [
                types.TextContent(
                    type="text", text=f"Error appending to file: {str(e)}"
                )
            ]

    async def _delete_file(self, args: dict) -> List[types.TextContent]:
        """Delete a file."""
        # Check confirmation flag
        if not args.get("confirm", False):
            return [
                types.TextContent(
                    type="text",
                    text="File deletion requires confirmation flag to be set to true",
                )
            ]

        file_path = self._get_safe_path(args["directory"], args["path"])

        if not file_path.exists():
            return [
                types.TextContent(
                    type="text",
                    text=f"File not found: {file_path.relative_to(self.config.base_path)}",
                )
            ]

        if not file_path.is_file():
            return [
                types.TextContent(
                    type="text",
                    text=f"Path is not a file: {file_path.relative_to(self.config.base_path)}",
                )
            ]

        try:
            # Since we're in sandbox mode, no need to check file extensions
            file_path.unlink()
            return [
                types.TextContent(
                    type="text",
                    text=f"File deleted successfully: {file_path.relative_to(self.config.base_path)}",
                )
            ]
        except Exception as e:
            return [
                types.TextContent(type="text", text=f"Error deleting file: {str(e)}")
            ]

    async def _copy_file(self, args: dict) -> List[types.TextContent]:
        """Copy file between directories with locking."""
        source_path = self._get_safe_path(args["source_directory"], args["source_path"])
        target_path = self._get_safe_path(args["target_directory"], args["target_path"])

        if not source_path.exists():
            return [
                types.TextContent(
                    type="text",
                    text=f"Source file not found: {source_path.relative_to(self.config.base_path)}",
                )
            ]

        if not source_path.is_file():
            return [
                types.TextContent(
                    type="text",
                    text=f"Source path is not a file: {source_path.relative_to(self.config.base_path)}",
                )
            ]

        if target_path.exists() and not args.get("overwrite", False):
            return [
                types.TextContent(
                    type="text",
                    text=f"Target file exists and overwrite=false: {target_path.relative_to(self.config.base_path)}",
                )
            ]

        try:
            # Use file locking for both source and target
            with (
                self._acquire_file_lock(source_path),
                self._acquire_file_lock(target_path),
            ):
                # Create target directory if needed
                target_path.parent.mkdir(parents=True, exist_ok=True)

                # Copy file
                shutil.copy2(source_path, target_path)

            return [
                types.TextContent(
                    type="text",
                    text=f"File copied successfully: {source_path.relative_to(self.config.base_path)} -> {target_path.relative_to(self.config.base_path)}",
                )
            ]
        except Exception as e:
            return [
                types.TextContent(type="text", text=f"Error copying file: {str(e)}")
            ]

    async def _move_file(self, args: dict) -> List[types.TextContent]:
        """Move/rename file between directories with locking."""
        source_path = self._get_safe_path(args["source_directory"], args["source_path"])
        target_path = self._get_safe_path(args["target_directory"], args["target_path"])

        if not source_path.exists():
            return [
                types.TextContent(
                    type="text",
                    text=f"Source file not found: {source_path.relative_to(self.config.base_path)}",
                )
            ]

        if not source_path.is_file():
            return [
                types.TextContent(
                    type="text",
                    text=f"Source path is not a file: {source_path.relative_to(self.config.base_path)}",
                )
            ]

        if target_path.exists() and not args.get("overwrite", False):
            return [
                types.TextContent(
                    type="text",
                    text=f"Target file exists and overwrite=false: {target_path.relative_to(self.config.base_path)}",
                )
            ]

        try:
            # Use file locking for both source and target
            with (
                self._acquire_file_lock(source_path),
                self._acquire_file_lock(target_path),
            ):
                # Create target directory if needed
                target_path.parent.mkdir(parents=True, exist_ok=True)

                # Move file
                shutil.move(str(source_path), str(target_path))

            return [
                types.TextContent(
                    type="text",
                    text=f"File moved successfully: {source_path.relative_to(self.config.base_path)} -> {target_path.relative_to(self.config.base_path)}",
                )
            ]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error moving file: {str(e)}")]

    async def _compress_file(self, args: dict) -> List[types.TextContent]:
        """Create ZIP archive from files."""
        archive_path = self._get_safe_path(
            args["archive_directory"], args["archive_path"]
        )

        if archive_path.exists():
            return [
                types.TextContent(
                    type="text",
                    text=f"Archive file already exists: {archive_path.relative_to(self.config.base_path)}",
                )
            ]

        files = args["files"]
        if not files:
            return [
                types.TextContent(
                    type="text", text="No files specified for compression"
                )
            ]

        try:
            # Validate all source files first
            source_paths = []
            for file_info in files:
                file_path = self._get_safe_path(
                    file_info["directory"], file_info["path"]
                )
                if not file_path.exists():
                    return [
                        types.TextContent(
                            type="text",
                            text=f"Source file not found: {file_path.relative_to(self.config.base_path)}",
                        )
                    ]
                if not file_path.is_file():
                    return [
                        types.TextContent(
                            type="text",
                            text=f"Source path is not a file: {file_path.relative_to(self.config.base_path)}",
                        )
                    ]
                source_paths.append(file_path)

            # Create archive directory if needed
            archive_path.parent.mkdir(parents=True, exist_ok=True)

            # Create ZIP archive with file locking
            with self._acquire_file_lock(archive_path):
                with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                    for i, file_path in enumerate(source_paths):
                        with self._acquire_file_lock(file_path):
                            # Use relative path within archive
                            arcname = files[i]["path"]
                            zipf.write(file_path, arcname)

            return [
                types.TextContent(
                    type="text",
                    text=f"Archive created successfully: {archive_path.relative_to(self.config.base_path)} ({len(files)} files)",
                )
            ]
        except Exception as e:
            return [
                types.TextContent(type="text", text=f"Error creating archive: {str(e)}")
            ]

    async def _extract_file(self, args: dict) -> List[types.TextContent]:
        """Extract ZIP archive to directory."""
        archive_path = self._get_safe_path(
            args["archive_directory"], args["archive_path"]
        )
        extract_dir = self._get_safe_path(
            args["extract_directory"], args.get("extract_path", ".")
        )

        if not archive_path.exists():
            return [
                types.TextContent(
                    type="text",
                    text=f"Archive file not found: {archive_path.relative_to(self.config.base_path)}",
                )
            ]

        if not archive_path.is_file():
            return [
                types.TextContent(
                    type="text",
                    text=f"Archive path is not a file: {archive_path.relative_to(self.config.base_path)}",
                )
            ]

        try:
            # Create extraction directory if needed
            extract_dir.mkdir(parents=True, exist_ok=True)

            extracted_files = []
            with self._acquire_file_lock(archive_path):
                with zipfile.ZipFile(archive_path, "r") as zipf:
                    # Validate zip file
                    zipf.testzip()

                    # Extract all files
                    for member in zipf.namelist():
                        # Security check: prevent path traversal
                        if ".." in member or member.startswith("/"):
                            return [
                                types.TextContent(
                                    type="text",
                                    text=f"Unsafe path in archive: {member}",
                                )
                            ]

                        # Extract file
                        zipf.extract(member, extract_dir)
                        extracted_files.append(member)

            return [
                types.TextContent(
                    type="text",
                    text=f"Archive extracted successfully to {extract_dir.relative_to(self.config.base_path)}: {len(extracted_files)} files",
                )
            ]
        except zipfile.BadZipFile:
            return [
                types.TextContent(type="text", text="Invalid or corrupted ZIP archive")
            ]
        except Exception as e:
            return [
                types.TextContent(
                    type="text", text=f"Error extracting archive: {str(e)}"
                )
            ]

    async def _file_lock(self, args: dict) -> List[types.TextContent]:
        """Manage file locks."""
        action = args["action"]

        if action == "list_active":
            if not self._file_locks:
                return [types.TextContent(type="text", text="No active file locks")]

            lock_info = []
            current_time = time.time()
            for file_path, lock_data in self._file_locks.items():
                age = current_time - lock_data["timestamp"]
                lock_info.append(f"  {file_path} (locked for {age:.1f}s)")

            return [
                types.TextContent(
                    type="text",
                    text=f"Active file locks ({len(self._file_locks)}):\n"
                    + "\n".join(lock_info),
                )
            ]

        elif action == "status":
            if "path" not in args or "directory" not in args:
                return [
                    types.TextContent(
                        type="text",
                        text="path and directory required for status action",
                    )
                ]

            file_path = self._get_safe_path(args["directory"], args["path"])
            lock_file = file_path.with_suffix(file_path.suffix + ".lock")

            is_locked = str(file_path) in self._file_locks or lock_file.exists()
            status = "locked" if is_locked else "unlocked"

            return [
                types.TextContent(
                    type="text",
                    text=f"File {file_path.relative_to(self.config.base_path)} is {status}",
                )
            ]

        else:
            return [
                types.TextContent(type="text", text=f"Unknown lock action: {action}")
            ]

    def _get_safe_path(self, directory: str, path: str) -> Path:
        """Get safe file path within allowed directory."""
        if directory not in self.config.allowed_directories:
            raise ValueError(f"Directory not allowed: {directory}")

        # Construct full path
        full_path = self.config.base_path / directory / path

        # Resolve and verify it's within allowed directory
        try:
            resolved = full_path.resolve()
            base_resolved = (self.config.base_path / directory).resolve()

            if not str(resolved).startswith(str(base_resolved)):
                raise ValueError(f"Path escape attempt detected: {path}")

            return resolved
        except Exception as e:
            raise ValueError(f"Invalid path: {path} - {str(e)}")
