"""Directory operations tools for FileIO MCP Server."""

import json
from pathlib import Path
from typing import Any, Dict, List

from mcp import types

from models import FileIOConfig
from utils import create_file_info, human_size, safe_json_dumps


class DirectoryOperations:
    """Handle directory-level operations."""

    def __init__(self, config: FileIOConfig):
        self.config = config

    def get_tools(self) -> List[types.Tool]:
        """Get list of available directory operation tools."""
        return [
            types.Tool(
                name="list_files",
                description="List files and directories in workflow directories",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "directory": {
                            "type": "string",
                            "enum": self.config.allowed_directories,
                            "description": "Target directory (ingress, wip, or completed)",
                        },
                        "pattern": {
                            "type": "string",
                            "default": "*",
                            "description": "Glob pattern to filter files",
                        },
                        "recursive": {
                            "type": "boolean",
                            "default": False,
                            "description": "Search recursively in subdirectories",
                        },
                        "include_hidden": {
                            "type": "boolean",
                            "default": False,
                            "description": "Include hidden files and directories",
                        },
                        "details": {
                            "type": "boolean",
                            "default": False,
                            "description": "Include detailed file information",
                        },
                    },
                    "required": ["directory"],
                },
            ),
            types.Tool(
                name="get_directory_tree",
                description="Get directory structure as a tree view",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "directory": {
                            "type": "string",
                            "enum": self.config.allowed_directories,
                            "description": "Target directory",
                        },
                        "max_depth": {
                            "type": "integer",
                            "default": 3,
                            "minimum": 1,
                            "maximum": 10,
                            "description": "Maximum depth to traverse",
                        },
                        "include_hidden": {
                            "type": "boolean",
                            "default": False,
                            "description": "Include hidden files and directories",
                        },
                    },
                    "required": ["directory"],
                },
            ),
            types.Tool(
                name="get_directory_stats",
                description="Get directory statistics (file count, sizes, types)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "directory": {
                            "type": "string",
                            "enum": self.config.allowed_directories,
                            "description": "Target directory",
                        },
                        "recursive": {
                            "type": "boolean",
                            "default": True,
                            "description": "Include subdirectories in statistics",
                        },
                    },
                    "required": ["directory"],
                },
            ),
            types.Tool(
                name="search_files",
                description="Search for files by name or content pattern",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "directory": {
                            "type": "string",
                            "enum": self.config.allowed_directories,
                            "description": "Target directory",
                        },
                        "name_pattern": {
                            "type": "string",
                            "description": "Filename pattern to search for",
                        },
                        "content_pattern": {
                            "type": "string",
                            "description": "Content pattern to search for (text files only)",
                        },
                        "case_sensitive": {
                            "type": "boolean",
                            "default": False,
                            "description": "Case sensitive search",
                        },
                        "max_results": {
                            "type": "integer",
                            "default": 50,
                            "maximum": 200,
                            "description": "Maximum number of results",
                        },
                    },
                    "required": ["directory"],
                    "anyOf": [
                        {"required": ["name_pattern"]},
                        {"required": ["content_pattern"]},
                    ],
                },
            ),
        ]

    async def execute(self, name: str, arguments: dict) -> List[types.TextContent]:
        """Execute directory operation tool."""
        try:
            if name == "list_files":
                return await self._list_files(arguments)
            elif name == "get_directory_tree":
                return await self._get_directory_tree(arguments)
            elif name == "get_directory_stats":
                return await self._get_directory_stats(arguments)
            elif name == "search_files":
                return await self._search_files(arguments)
            else:
                return [
                    types.TextContent(type="text", text=f"Unknown operation: {name}")
                ]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error in {name}: {str(e)}")]

    async def _list_files(self, args: dict) -> List[types.TextContent]:
        """List files in directory."""
        dir_path = self.config.base_path / args["directory"]
        pattern = args.get("pattern", "*")
        recursive = args.get("recursive", False)
        include_hidden = args.get("include_hidden", False)
        details = args.get("details", False)

        if recursive:
            files = dir_path.rglob(pattern)
        else:
            files = dir_path.glob(pattern)

        file_list = []
        for file_path in sorted(files):
            # Skip hidden files unless requested
            if not include_hidden and file_path.name.startswith("."):
                continue

            if details:
                info = create_file_info(file_path, self.config.base_path)
                file_list.append(info)
            else:
                rel_path = file_path.relative_to(dir_path)
                file_type = "📁" if file_path.is_dir() else "📄"
                size = (
                    human_size(file_path.stat().st_size) if file_path.is_file() else ""
                )
                file_list.append(f"{file_type} {rel_path} {size}".strip())

        if not file_list:
            return [
                types.TextContent(
                    type="text", text=f"No files found matching pattern: {pattern}"
                )
            ]

        if details:
            return [
                types.TextContent(
                    type="text",
                    text=f"Files in {args['directory']}:\n{safe_json_dumps(file_list)}",
                )
            ]
        else:
            return [
                types.TextContent(
                    type="text",
                    text=f"Files in {args['directory']}:\n" + "\n".join(file_list),
                )
            ]

    async def _get_directory_tree(self, args: dict) -> List[types.TextContent]:
        """Get directory tree structure."""
        dir_path = self.config.base_path / args["directory"]
        max_depth = args.get("max_depth", 3)
        include_hidden = args.get("include_hidden", False)

        def build_tree(path: Path, depth: int = 0, prefix: str = "") -> List[str]:
            if depth > max_depth:
                return []

            items = []
            try:
                children = sorted(path.iterdir())
                if not include_hidden:
                    children = [c for c in children if not c.name.startswith(".")]

                for i, child in enumerate(children):
                    is_last = i == len(children) - 1
                    current_prefix = "└── " if is_last else "├── "
                    next_prefix = "    " if is_last else "│   "

                    icon = "📁" if child.is_dir() else "📄"
                    size_info = ""
                    if child.is_file():
                        try:
                            size_info = f" ({human_size(child.stat().st_size)})"
                        except:
                            pass

                    items.append(
                        f"{prefix}{current_prefix}{icon} {child.name}{size_info}"
                    )

                    if child.is_dir() and depth < max_depth:
                        items.extend(build_tree(child, depth + 1, prefix + next_prefix))

            except PermissionError:
                items.append(f"{prefix}└── [Permission Denied]")

            return items

        tree_lines = [f"📁 {args['directory']}/"] + build_tree(dir_path)
        return [types.TextContent(type="text", text="\n".join(tree_lines))]

    async def _get_directory_stats(self, args: dict) -> List[types.TextContent]:
        """Get directory statistics."""
        dir_path = self.config.base_path / args["directory"]
        recursive = args.get("recursive", True)

        stats = {
            "directory": args["directory"],
            "total_files": 0,
            "total_directories": 0,
            "total_size": 0,
            "file_types": {},
            "largest_files": [],
            "newest_files": [],
        }

        files_info = []

        if recursive:
            iterator = dir_path.rglob("*")
        else:
            iterator = dir_path.glob("*")

        for item in iterator:
            try:
                if item.is_file():
                    stats["total_files"] += 1
                    size = item.stat().st_size
                    stats["total_size"] += size

                    # Track file types
                    ext = item.suffix.lower() or "no_extension"
                    stats["file_types"][ext] = stats["file_types"].get(ext, 0) + 1

                    # Track for largest/newest files
                    files_info.append(
                        {
                            "path": str(item.relative_to(self.config.base_path)),
                            "size": size,
                            "modified": item.stat().st_mtime,
                        }
                    )

                elif item.is_dir():
                    stats["total_directories"] += 1

            except (PermissionError, OSError):
                continue

        # Get largest files (top 5)
        stats["largest_files"] = sorted(
            files_info, key=lambda x: x["size"], reverse=True
        )[:5]

        # Get newest files (top 5)
        stats["newest_files"] = sorted(
            files_info, key=lambda x: x["modified"], reverse=True
        )[:5]

        # Convert size to human readable
        stats["total_size_human"] = human_size(stats["total_size"])

        return [
            types.TextContent(
                type="text", text=f"Directory Statistics:\n{safe_json_dumps(stats)}"
            )
        ]

    async def _search_files(self, args: dict) -> List[types.TextContent]:
        """Search for files by name or content."""
        dir_path = self.config.base_path / args["directory"]
        name_pattern = args.get("name_pattern")
        content_pattern = args.get("content_pattern")
        case_sensitive = args.get("case_sensitive", False)
        max_results = args.get("max_results", 50)

        results = []

        # Search by filename
        if name_pattern:
            if not case_sensitive:
                name_pattern = name_pattern.lower()

            for file_path in dir_path.rglob("*"):
                if len(results) >= max_results:
                    break

                filename = file_path.name
                if not case_sensitive:
                    filename = filename.lower()

                if name_pattern in filename:
                    results.append(
                        {
                            "type": "name_match",
                            "path": str(file_path.relative_to(self.config.base_path)),
                            "match": name_pattern,
                            "is_file": file_path.is_file(),
                        }
                    )

        # Search by content
        if content_pattern and len(results) < max_results:
            search_pattern = (
                content_pattern if case_sensitive else content_pattern.lower()
            )

            for file_path in dir_path.rglob("*"):
                if len(results) >= max_results:
                    break

                if not file_path.is_file():
                    continue

                # Only search text files
                if file_path.suffix.lower() not in [
                    ".txt",
                    ".md",
                    ".json",
                    ".yaml",
                    ".yml",
                    ".log",
                    ".csv",
                ]:
                    continue

                try:
                    # Check file size
                    if file_path.stat().st_size > self.config.max_file_size:
                        continue

                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    search_content = content if case_sensitive else content.lower()

                    if search_pattern in search_content:
                        # Find line number of first match
                        lines = search_content.split("\n")
                        line_num = next(
                            (
                                i + 1
                                for i, line in enumerate(lines)
                                if search_pattern in line
                            ),
                            1,
                        )

                        results.append(
                            {
                                "type": "content_match",
                                "path": str(
                                    file_path.relative_to(self.config.base_path)
                                ),
                                "match": content_pattern,
                                "line": line_num,
                            }
                        )

                except (UnicodeDecodeError, PermissionError, OSError):
                    continue

        if not results:
            return [
                types.TextContent(
                    type="text", text="No files found matching the search criteria."
                )
            ]

        return [
            types.TextContent(
                type="text",
                text=f"Search Results ({len(results)} found):\n{safe_json_dumps(results)}",
            )
        ]
