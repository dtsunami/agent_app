"""Context loading for DA.md and DA.json files."""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from .models import MCPServerInfo, ProjectContext

logger = logging.getLogger(__name__)


class ContextLoader:
    """Loads project context from DA.md and MCP server info from DA.json."""

    def __init__(self, project_root: Optional[str] = None):
        """Initialize context loader with project root directory."""
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.da_md_path = self.project_root / "DA.md"
        self.da_json_path = self.project_root / "DA.json"

    def load_project_context(self) -> Optional[ProjectContext]:
        """Load project context from DA.md file."""
        try:
            if not self.da_md_path.exists():
                logger.warning(f"DA.md not found at {self.da_md_path}")
                return None

            with open(self.da_md_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if not content.strip():
                logger.warning("DA.md is empty")
                return None

            # Extract project name and description from markdown
            project_name = self._extract_project_name(content)
            description = self._extract_description(content)
            instructions = self._extract_instructions(content)

            context = ProjectContext(
                project_name=project_name,
                description=description,
                instructions=instructions,
                file_content=content
            )

            logger.info(f"Loaded project context from {self.da_md_path}")
            return context

        except Exception as e:
            logger.error(f"Failed to load DA.md: {e}")
            return None

    def load_mcp_servers(self) -> List[MCPServerInfo]:
        """Load MCP server information from DA.json file and add built-in servers."""
        servers = []

        # Load external MCP servers from DA.json
        try:
            if not self.da_json_path.exists():
                logger.warning(f"DA.json not found at {self.da_json_path}")
                return servers

            with open(self.da_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            mcp_servers = data.get('mcp_servers', [])

            for server_data in mcp_servers:
                try:
                    server = MCPServerInfo(**server_data)
                    servers.append(server)
                except Exception as e:
                    logger.error(f"Invalid MCP server data: {server_data}, error: {e}")
                    continue

            logger.info(f"Loaded {len(servers)} total MCP servers ({len(mcp_servers)} from DA.json + 1 built-in)")
            return servers

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in DA.json: {e}")
            return servers
        except Exception as e:
            logger.error(f"Failed to load DA.json: {e}")
            return servers

    def _extract_project_name(self, content: str) -> Optional[str]:
        """Extract project name from markdown content."""
        lines = content.split('\n')

        for line in lines:
            line = line.strip()
            # Look for first H1 heading
            if line.startswith('# '):
                return line[2:].strip()

        return None

    def _extract_description(self, content: str) -> Optional[str]:
        """Extract project description from markdown content."""
        lines = content.split('\n')
        description_lines = []
        found_title = False

        for line in lines:
            line = line.strip()

            # Skip empty lines before finding title
            if not found_title and not line:
                continue

            # Found the title (H1)
            if line.startswith('# '):
                found_title = True
                continue

            # Stop at next heading or section
            if found_title and (line.startswith('#') or line.startswith('##')):
                break

            # Collect description lines
            if found_title:
                description_lines.append(line)

        description = '\n'.join(description_lines).strip()
        return description if description else None

    def _extract_instructions(self, content: str) -> Optional[str]:
        """Extract instructions from markdown content."""
        # Look for sections with 'instruction' in the heading
        lines = content.split('\n')
        instructions_lines = []
        in_instructions = False

        for line in lines:
            line_lower = line.lower().strip()

            # Check if this is an instructions heading
            if line_lower.startswith('#') and 'instruction' in line_lower:
                in_instructions = True
                continue

            # Stop at next heading
            if in_instructions and line.strip().startswith('#'):
                break

            # Collect instruction lines
            if in_instructions:
                instructions_lines.append(line)

        instructions = '\n'.join(instructions_lines).strip()
        return instructions if instructions else None

    def create_sample_da_json(self) -> None:
        """Create a sample DA.json file with common MCP servers."""
        sample_data = {
            "mcp_servers": [
                {
                    "name": "search",
                    "url": "http://localhost:8080/search",
                    "port": 8003,
                    "description": "Web search MCP server",
                    "tools": ["web_search", "extract_content"]
                }
            ],
            "default_working_directory": ".",
            "agent_settings": {
                "model": "gpt-40",
                "temperature": 0.7,
                "max_tokens": None,
                "require_confirmation": True
            }
        }

        try:
            with open(self.da_json_path, 'w', encoding='utf-8') as f:
                json.dump(sample_data, f, indent=2)

            logger.info(f"Created sample DA.json at {self.da_json_path}")
        except Exception as e:
            logger.error(f"Failed to create sample DA.json: {e}")

    def create_sample_da_md(self) -> None:
        """Create a sample DA.md file."""
        sample_content = """# Project Name

Brief description of your project goes here.

## Instructions

Instructions for the AI agent on how to work with this project:

- Key guidelines
- Important files and directories
- Coding standards
- Testing procedures
- Deployment notes

## Architecture

Description of the project architecture, key components, and how they interact.

## Development Guidelines

- Code style preferences
- File organization
- Dependencies management
- Environment setup

## Important Files

- List key files and their purposes
- Configuration files
- Entry points
- Documentation
"""

        try:
            with open(self.da_md_path, 'w', encoding='utf-8') as f:
                f.write(sample_content)

            logger.info(f"Created sample DA.md at {self.da_md_path}")
        except Exception as e:
            logger.error(f"Failed to create sample DA.md: {e}")


async def check_mcp_server_health(server: MCPServerInfo) -> bool:
    """Check if an MCP server is healthy and responsive."""
    try:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            health_url = f"{server.url.rstrip('/')}/health"
            async with session.get(health_url, timeout=5) as response:
                return response.status == 200

    except Exception as e:
        logger.error(f"Health check failed for {server.name}: {e}")
        return False


async def discover_mcp_tools(server: MCPServerInfo) -> List[str]:
    """Discover available tools from an MCP server."""
    try:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            tools_url = f"{server.url.rstrip('/')}/mcp"
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list"
            }

            async with session.post(tools_url, json=payload, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    result = data.get('result', {})
                    tools = result.get('tools', [])
                    return [tool.get('name', '') for tool in tools if tool.get('name')]

    except Exception as e:
        logger.error(f"Tool discovery failed for {server.name}: {e}")

    return []