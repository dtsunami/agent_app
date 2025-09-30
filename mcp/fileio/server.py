"""MCP JSON-RPC Server implementation for FileIO operations using BaseMCPServer foundation."""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv("../../.env")

from basemcp.server import BaseMCPServer
from models import FileIOConfig
from directory_ops import DirectoryOperations
from file_ops import FileOperations


class FileIOMCPServer(BaseMCPServer):
    """MCP JSON-RPC Server for FileIO operations using BaseMCPServer foundation."""

    def __init__(self, config_path: str = "config.json"):
        # Load configuration
        self.config = FileIOConfig.load(config_path)

        # Initialize the base MCP server
        super().__init__("FileIO", "1.0.0")

        # Setup file operations
        self.file_ops = None
        self.dir_ops = None
        self.setup_operations()

    def setup_operations(self):
        """Setup file and directory operations."""
        self.file_ops = FileOperations(self.config)
        self.dir_ops = DirectoryOperations(self.config)

    async def get_health_status(self) -> Dict[str, Any]:
        """Override health status to include FileIO-specific information."""
        base_status = await super().get_health_status()
        base_status.update({
            "base_path_exists": self.config.base_path.exists(),
            "allowed_directories": [
                {"name": d, "exists": (self.config.base_path / d).exists()}
                for d in self.config.allowed_directories
            ],
        })
        return base_status

    async def handle_tools_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/list request."""
        tools = []

        # Get tools from file operations
        for tool in self.file_ops.get_tools():
            tools.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.inputSchema,
                }
            )

        # Get tools from directory operations
        for tool in self.dir_ops.get_tools():
            tools.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.inputSchema,
                }
            )

        return {"tools": tools}

    async def handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if not tool_name:
            raise ValueError("Tool name is required")

        self.logger.info(f"Executing tool: {tool_name} with args: {arguments}")

        # Route to appropriate handler
        file_tool_names = [tool.name for tool in self.file_ops.get_tools()]
        dir_tool_names = [tool.name for tool in self.dir_ops.get_tools()]

        if tool_name in file_tool_names:
            result = await self.file_ops.execute(tool_name, arguments)
        elif tool_name in dir_tool_names:
            result = await self.dir_ops.execute(tool_name, arguments)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

        # Convert MCP TextContent to proper format
        content = []
        for item in result:
            content.append({"type": item.type, "text": item.text})

        return {"content": content, "isError": False}

    def run(self):
        """Run the MCP server."""
        self.logger.info(f"Monitoring directories: {self.config.allowed_directories}")
        self.logger.info(f"Base path: {self.config.base_path}")

        # Use the base server's run method with config values
        super().run(
            host=self.config.server.host,
            port=self.config.server.port
        )


def main():
    """Main entry point for the MCP server."""
    import argparse

    parser = argparse.ArgumentParser(description="FileIO MCP JSON-RPC Server")
    parser.add_argument(
        "--config", default="config.json", help="Path to configuration file"
    )

    args = parser.parse_args()

    # Run the server
    server = FileIOMCPServer(args.config)
    server.run()


if __name__ == "__main__":
    main()
