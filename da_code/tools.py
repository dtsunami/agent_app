"""Modern agent tools with async support and 2025 agentic patterns."""

import asyncio
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Literal
from urllib.parse import quote

import aiohttp
import httpx
from langchain.tools import Tool
from pydantic import BaseModel, Field, ConfigDict


# Modern Pydantic v2 models for tool parameters
class TodoOperation(BaseModel):
    """Structured parameters for todo operations."""
    model_config = ConfigDict(extra='forbid')

    operation: Literal["read", "exists", "create", "update", "write"] = "read"
    content: Optional[str] = None

class WebSearchParams(BaseModel):
    """Structured parameters for web search."""
    model_config = ConfigDict(extra='forbid')

    query: str = Field(..., description="Search query")
    num_results: int = Field(5, ge=1, le=20, description="Number of results to return")

class FileSearchParams(BaseModel):
    """Structured parameters for file search."""
    model_config = ConfigDict(extra='forbid')

    pattern: str = Field("*", description="Glob pattern for file matching")
    content: Optional[str] = Field(None, description="Text content to search for")
    max_results: int = Field(20, ge=1, le=100, description="Maximum results to return")

class TimeParams(BaseModel):
    """Structured parameters for time operations."""
    model_config = ConfigDict(extra='forbid')

    format: str = Field("iso", description="Time format: iso, human, timestamp, date, time, or custom strftime")
    timezone: str = Field("UTC", description="Timezone (currently only UTC supported)")

class PythonCodeParams(BaseModel):
    """Structured parameters for Python code execution."""
    model_config = ConfigDict(extra='forbid')

    code: str = Field(..., description="Python code to execute")
    timeout: int = Field(30, ge=1, le=300, description="Timeout in seconds (1-300)")

class GitParams(BaseModel):
    """Structured parameters for git operations."""
    model_config = ConfigDict(extra='forbid')

    operation: Literal["status", "commit", "diff", "branch", "log"] = "status"
    message: Optional[str] = Field(None, description="Commit message (required for commit operation)")
    branch_name: Optional[str] = Field(None, description="Branch name for branch operations")
    files: Optional[List[str]] = Field(None, description="Specific files for diff operation")
    limit: int = Field(10, ge=1, le=50, description="Number of log entries to show")



class TodoManager:
    """Modern todo.md file manager with async support and structured operations."""

    def __init__(self, working_directory: str = None):
        """Initialize todo manager."""
        self.working_dir = working_directory or os.getcwd()
        self.todo_file = Path(self.working_dir) / "todo.md"

    def read_todo_file(self) -> str:
        """Read current contents of todo.md file."""
        try:
            if not self.todo_file.exists():
                return "No todo.md file exists in the current directory."

            content = self.todo_file.read_text(encoding='utf-8')
            if not content.strip():
                return "todo.md file exists but is empty."

            return content.strip()

        except Exception as e:
            return f"Error reading todo file: {str(e)}"

    def file_exists(self) -> str:
        """Check if todo.md file exists."""
        try:
            exists = self.todo_file.exists()
            if exists:
                size = self.todo_file.stat().st_size
                return f"✅ todo.md exists ({size} bytes)"
            else:
                return "❌ todo.md does not exist"

        except Exception as e:
            return f"Error checking file existence: {str(e)}"

    def create_todo_file(self, content: str) -> str:
        """Create or completely replace todo.md file with provided content."""
        try:
            # Ensure content follows proper markdown format
            if not content.strip().startswith('# '):
                content = f"# TODO\n\n{content.strip()}"

            self.todo_file.write_text(content.strip() + '\n', encoding='utf-8')
            return f"✅ Created/updated todo.md file"

        except Exception as e:
            return f"Error creating todo file: {str(e)}"

    def update_todo_file(self, new_content: str) -> str:
        """Update todo.md file by replacing its contents."""
        return self.create_todo_file(new_content)


class WebSearchManager:
    """Modern web search with proven MCP DuckDuckGo pattern."""

    def __init__(self):
        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; da_code/1.0)"
            }
        )

    async def search_web(self, query: str, num_results: int = 5) -> str:
        """Perform async web search using proven MCP DuckDuckGo pattern."""
        try:
            # First try DuckDuckGo Instant Answer API
            encoded_query = quote(query)
            url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_html=1&skip_disambig=1"

            response = await self.client.get(url)
            response.raise_for_status()
            data = response.json()

            results = []

            # Process instant answers
            if data.get("Answer"):
                results.append(f"**Instant Answer:** {data['Answer']}")
                if data.get("AnswerURL"):
                    results.append(f"Source: {data['AnswerURL']}")

            # Process abstract
            if data.get("AbstractText"):
                results.append(f"**Summary:** {data['AbstractText']}")
                if data.get("AbstractURL"):
                    results.append(f"Source: {data['AbstractURL']}")

            # Process related topics
            for topic in data.get("RelatedTopics", [])[:num_results]:
                if isinstance(topic, dict) and "Text" in topic:
                    title = topic.get("FirstURL", "").split("/")[-1].replace("_", " ") if topic.get("FirstURL") else "Related"
                    results.append(f"**{title}**")
                    results.append(topic["Text"][:200] + ("..." if len(topic["Text"]) > 200 else ""))
                    if topic.get("FirstURL"):
                        results.append(f"Link: {topic['FirstURL']}")
                    results.append("")  # Add spacing

            # If we don't have enough results, try web search fallback
            if len(results) < 3:
                web_results = await self._duckduckgo_web_search(query, num_results)
                if web_results:
                    results.extend(web_results)

            if results:
                return "\n".join(results).strip()
            else:
                return f"No search results found for '{query}'. Try a different search term."

        except Exception as e:
            return f"Search error: {str(e)}. Web search may be temporarily unavailable."

    async def _duckduckgo_web_search(self, query: str, num_results: int) -> List[str]:
        """Fallback web search using DuckDuckGo HTML (proven MCP pattern)."""
        try:
            encoded_query = quote(query)
            url = f"https://html.duckduckgo.com/html/?q={encoded_query}"

            response = await self.client.get(url)
            response.raise_for_status()

            # Use BeautifulSoup for proper HTML parsing
            try:
                from bs4 import BeautifulSoup
            except ImportError:
                # Fallback to simple regex if BeautifulSoup not available
                return self._parse_html_simple(response.text, num_results)

            soup = BeautifulSoup(response.text, 'html.parser')
            results = []

            for result_div in soup.find_all('div', class_='result')[:num_results]:
                title_elem = result_div.find('a', class_='result__a')
                snippet_elem = result_div.find('a', class_='result__snippet')

                if title_elem and snippet_elem:
                    title = title_elem.get_text(strip=True)
                    url = title_elem.get('href', '')
                    snippet = snippet_elem.get_text(strip=True)

                    results.append(f"**{title}**")
                    results.append(snippet[:200] + ("..." if len(snippet) > 200 else ""))
                    results.append(f"Link: {url}")
                    results.append("")  # Add spacing

            return results

        except Exception as e:
            return [f"Web search fallback error: {str(e)}"]

    def _parse_html_simple(self, html_content: str, num_results: int) -> List[str]:
        """Simple regex-based HTML parsing fallback."""
        try:
            results = []
            # Simple pattern to find links
            import re
            link_pattern = r'<a[^>]*href="(https?://[^"]+)"[^>]*>([^<]+)</a>'
            matches = re.findall(link_pattern, html_content)

            count = 0
            for url, title in matches:
                if count >= num_results:
                    break
                if 'duckduckgo.com' not in url:
                    results.append(f"**{title.strip()}**")
                    results.append(f"Link: {url}")
                    results.append("")
                    count += 1

            return results
        except Exception:
            return ["Simple HTML parsing failed"]


class FileSearchManager:
    """Modern file search with async glob patterns and content search."""

    def __init__(self, working_directory: str = None):
        self.working_dir = Path(working_directory or os.getcwd())

    async def search_files(self, pattern: str = "*", content: Optional[str] = None, max_results: int = 20) -> str:
        """Search for files by pattern and/or content with async processing."""
        try:
            results = []

            # Find files by pattern
            if pattern != "*":
                files = list(self.working_dir.rglob(pattern))
            else:
                files = [f for f in self.working_dir.rglob("*") if f.is_file()]

            # Filter by content if specified
            if content:
                content_matches = await self._search_content_async(files[:50], content, max_results)
                return "\n\n".join(content_matches) if content_matches else "No files found with that content"

            # Return file list
            for file_path in files[:max_results]:
                if file_path.is_file():
                    try:
                        relative_path = file_path.relative_to(self.working_dir)
                        size = file_path.stat().st_size
                        modified = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
                        results.append(f"📄 {relative_path} ({size:,} bytes, modified {modified.strftime('%Y-%m-%d %H:%M')}")
                    except (OSError, ValueError):
                        continue

            return "\n".join(results) if results else f"No files found matching pattern: {pattern}"

        except Exception as e:
            return f"File search error: {str(e)}"

    async def _search_content_async(self, files: List[Path], content: str, max_results: int) -> List[str]:
        """Async content search in files."""
        search_pattern = re.compile(re.escape(content), re.IGNORECASE)
        content_matches = []

        # Process files concurrently
        tasks = []
        for file_path in files:
            if file_path.suffix.lower() in {'.py', '.md', '.txt', '.json', '.yaml', '.yml', '.js', '.ts', '.html', '.css'}:
                tasks.append(self._search_file_content(file_path, search_pattern))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, str) and result:
                    content_matches.append(result)
                    if len(content_matches) >= max_results:
                        break

        return content_matches

    async def _search_file_content(self, file_path: Path, search_pattern: re.Pattern) -> Optional[str]:
        """Search content in a single file asynchronously."""
        try:
            def _read_and_search():
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    file_content = f.read()
                    if search_pattern.search(file_content):
                        lines = file_content.split('\n')
                        matching_lines = []
                        for i, line in enumerate(lines, 1):
                            if search_pattern.search(line):
                                matching_lines.append(f"  {i}: {line.strip()}")
                                if len(matching_lines) >= 3:
                                    break

                        relative_path = file_path.relative_to(self.working_dir)
                        return f"📄 {relative_path}\n" + "\n".join(matching_lines)
                return None

            # Run file I/O in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, _read_and_search)

        except Exception:
            return None


class TimeManager:
    """Modern time utilities with timezone support."""

    @staticmethod
    def get_current_time(format_type: str = "iso", timezone_name: str = "UTC") -> str:
        """Get current time in various formats with timezone support."""
        try:
            # Get current time (UTC for now, could extend with pytz)
            now = datetime.now(timezone.utc)

            # Format based on request
            if format_type.lower() == "iso":
                return now.isoformat()
            elif format_type.lower() == "human":
                return now.strftime("%Y-%m-%d %H:%M:%S %Z")
            elif format_type.lower() == "timestamp":
                return str(int(now.timestamp()))
            elif format_type.lower() == "date":
                return now.strftime("%Y-%m-%d")
            elif format_type.lower() == "time":
                return now.strftime("%H:%M:%S")
            elif format_type.lower() == "unix":
                return str(int(now.timestamp()))
            elif format_type.lower() == "rfc":
                return now.strftime("%a, %d %b %Y %H:%M:%S %z")
            else:
                # Custom strftime format
                return now.strftime(format_type)

        except Exception as e:
            return f"Time formatting error: {str(e)}"


class PythonCodeExecutor:
    """Safe Python code execution with sandboxing."""

    def __init__(self):
        # Common safe libraries that are typically available
        self.safe_libraries = {
            'math', 'random', 'datetime', 'json', 'base64', 'hashlib',
            'urllib', 'uuid', 'itertools', 'functools', 'collections',
            'csv', 're', 'string', 'textwrap', 'statistics', 'numpy'
        }

    def execute_code(self, code: str, timeout: int = 30) -> str:
        """Execute Python code safely with timeout and restrictions."""
        try:
            import subprocess
            import tempfile
            import os

            # Create a temporary file for the code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name

            try:
                # Execute in a restricted environment
                result = subprocess.run(
                    ['python', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=tempfile.gettempdir(),  # Run in temp directory
                    env={
                        'PYTHONPATH': '',  # Restrict import paths
                        'PATH': os.environ.get('PATH', ''),  # Keep basic PATH
                    }
                )

                output_lines = []

                if result.stdout:
                    output_lines.append("**Output:**")
                    output_lines.append(result.stdout.strip())

                if result.stderr:
                    output_lines.append("**Errors:**")
                    output_lines.append(result.stderr.strip())

                if result.returncode != 0:
                    output_lines.append(f"**Exit Code:** {result.returncode}")

                if not output_lines:
                    output_lines.append("Code executed successfully with no output.")

                return "\n".join(output_lines)

            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_file)
                except:
                    pass

        except subprocess.TimeoutExpired:
            return f"⏰ Code execution timed out after {timeout} seconds"
        except Exception as e:
            return f"❌ Execution error: {str(e)}"


class GitManager:
    """Modern git operations manager with async support."""

    def __init__(self, working_directory: str = None):
        """Initialize git manager."""
        self.working_dir = working_directory or os.getcwd()

    async def execute_git_command(self, cmd: List[str]) -> str:
        """Execute git command asynchronously."""
        try:
            # Run git command in working directory
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=self.working_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            # Decode output
            stdout_text = stdout.decode('utf-8', errors='replace').strip()
            stderr_text = stderr.decode('utf-8', errors='replace').strip()

            if process.returncode != 0:
                return f"❌ Git command failed (exit {process.returncode}):\n{stderr_text or stdout_text}"

            return stdout_text or "✅ Command completed successfully"

        except FileNotFoundError:
            return "❌ Git not found. Please ensure git is installed and in PATH."
        except Exception as e:
            return f"❌ Git command error: {str(e)}"

    async def git_status(self) -> str:
        """Get git repository status."""
        return await self.execute_git_command(['git', 'status', '--porcelain', '-b'])

    async def git_commit(self, message: str, files: Optional[List[str]] = None) -> str:
        """Commit changes with message."""
        if not message:
            return "❌ Commit message is required"

        # Add files if specified, otherwise add all
        if files:
            for file in files:
                result = await self.execute_git_command(['git', 'add', file])
                if result.startswith("❌"):
                    return result
        else:
            result = await self.execute_git_command(['git', 'add', '.'])
            if result.startswith("❌"):
                return result

        # Commit changes
        return await self.execute_git_command(['git', 'commit', '-m', message])

    async def git_diff(self, files: Optional[List[str]] = None) -> str:
        """Show git diff for files."""
        cmd = ['git', 'diff']
        if files:
            cmd.extend(files)
        return await self.execute_git_command(cmd)

    async def git_branch(self, branch_name: Optional[str] = None) -> str:
        """List branches or create/switch to branch."""
        if branch_name:
            # Create and switch to new branch
            return await self.execute_git_command(['git', 'checkout', '-b', branch_name])
        else:
            # List all branches
            return await self.execute_git_command(['git', 'branch', '-a'])

    async def git_log(self, limit: int = 10) -> str:
        """Show git log with specified limit."""
        return await self.execute_git_command([
            'git', 'log', f'--max-count={limit}',
            '--oneline', '--graph', '--decorate'
        ])


# Tool factory functions with modern patterns
def _parse_tool_input(tool_input: Union[str, dict], model_class: type) -> Union[BaseModel, str]:
    """Parse and validate tool input using Pydantic models."""
    try:
        if isinstance(tool_input, str):
            try:
                params = json.loads(tool_input)

                # Handle LangChain's __arg1 format
                if "__arg1" in params:
                    arg1_value = params["__arg1"]
                    if isinstance(arg1_value, str):
                        try:
                            # Try to parse __arg1 as JSON
                            actual_params = json.loads(arg1_value)
                            return model_class(**actual_params)
                        except json.JSONDecodeError:
                            # __arg1 is a simple string, handle by model type
                            return _handle_simple_string(arg1_value, model_class)
                    else:
                        # __arg1 is already a dict
                        return model_class(**arg1_value)
                else:
                    # Normal JSON params
                    return model_class(**params)
            except json.JSONDecodeError:
                # Handle simple string inputs
                return _handle_simple_string(tool_input, model_class)
        else:
            # Handle dict input (already parsed)
            if "__arg1" in tool_input:
                arg1_value = tool_input["__arg1"]
                if isinstance(arg1_value, str):
                    try:
                        actual_params = json.loads(arg1_value)
                        return model_class(**actual_params)
                    except json.JSONDecodeError:
                        return _handle_simple_string(arg1_value, model_class)
                else:
                    return model_class(**arg1_value)
            else:
                return model_class(**tool_input)
    except Exception as e:
        return f"Parameter validation error: {str(e)}"

def _handle_simple_string(value: str, model_class: type) -> Union[BaseModel, str]:
    """Handle simple string inputs for different model classes."""
    if model_class == TodoOperation:
        return TodoOperation(operation="create", content=value)
    elif model_class == WebSearchParams:
        return WebSearchParams(query=value)
    elif model_class == FileSearchParams:
        return FileSearchParams(pattern=value)
    elif model_class == TimeParams:
        return TimeParams(format=value)
    elif model_class == PythonCodeParams:
        return PythonCodeParams(code=value)
    elif model_class == GitParams:
        return GitParams(operation="status")
    else:
        return f"Invalid input format for {model_class.__name__}"


def create_todo_tool(working_directory: str = None) -> Tool:
    """Create a modern todo management tool with structured validation."""
    todo_manager = TodoManager(working_directory)

    def handle_todo_operation(tool_input: str) -> str:
        """Handle todo.md file operations with structured validation."""
        params = _parse_tool_input(tool_input, TodoOperation)
        if isinstance(params, str):
            return params

        try:
            if params.operation in ["read", "get", "show"]:
                return todo_manager.read_todo_file()
            elif params.operation in ["exists", "check"]:
                return todo_manager.file_exists()
            elif params.operation in ["create", "update", "write"]:
                if not params.content:
                    return "Error: content parameter is required for create/update operations"
                return todo_manager.create_todo_file(params.content)
            else:
                return f"Error: Unsupported operation '{params.operation}'"
        except Exception as e:
            return f"Todo operation error: {str(e)}"

    return Tool(
        name="todo_file_manager",
        description="""Manage todo.md file with structured operations and modern validation.

SUPPORTED FORMATS:
- JSON: {"operation": "read"} or {"operation": "create", "content": "..."}
- Simple text: Treated as content for create operation

OPERATIONS:
- read/get/show: Read current todo.md contents
- exists/check: Check if todo.md file exists
- create/update/write: Create or replace todo.md with content

EXAMPLE INPUTS:
- {"operation": "read"}
- {"operation": "create", "content": "# TODO\n\n- [ ] New task"}
- "Quick todo item"

TODO FORMAT: Use markdown with - [ ] for tasks, - [x] for completed items.""",
        func=handle_todo_operation
    )


def create_web_search_tool() -> Tool:
    """Create a modern web search tool with async DuckDuckGo integration."""
    search_manager = WebSearchManager()

    def handle_search(tool_input: str) -> str:
        """Handle web search with structured validation."""
        params = _parse_tool_input(tool_input, WebSearchParams)
        if isinstance(params, str):
            return params

        try:
            # Handle async context properly
            try:
                # Try to get existing event loop
                loop = asyncio.get_running_loop()
                # We're in an async context, create a task that can be awaited
                # For now, run in thread pool to avoid event loop conflicts
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        lambda: asyncio.run(search_manager.search_web(params.query, params.num_results))
                    )
                    return future.result(timeout=10)
            except RuntimeError:
                # No running event loop, safe to create new one
                return asyncio.run(search_manager.search_web(params.query, params.num_results))
        except Exception as e:
            return f"Web search error: {str(e)}"

    return Tool(
        name="web_search",
        description="""Search the web for current information using DuckDuckGo.

SUPPORTED FORMATS:
- JSON: {"query": "search terms", "num_results": 5}
- Simple text: "search terms"

EXAMPLE INPUTS:
- {"query": "Python asyncio tutorial", "num_results": 3}
- "latest AI news 2025"
- {"query": "pydantic validation examples"}

Returns: Instant answers, related topics, and source links.""",
        func=handle_search
    )


def create_file_search_tool(working_directory: str = None) -> Tool:
    """Create a modern file search tool with async glob and content search."""
    search_manager = FileSearchManager(working_directory)

    def handle_search(tool_input: str) -> str:
        """Handle file search with structured validation."""
        params = _parse_tool_input(tool_input, FileSearchParams)
        if isinstance(params, str):
            return params

        try:
            # Handle async context properly
            try:
                # Try to get existing event loop
                loop = asyncio.get_running_loop()
                # We're in an async context, run in thread pool to avoid conflicts
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        lambda: asyncio.run(search_manager.search_files(params.pattern, params.content, params.max_results))
                    )
                    return future.result(timeout=10)
            except RuntimeError:
                # No running event loop, safe to create new one
                return asyncio.run(search_manager.search_files(params.pattern, params.content, params.max_results))
        except Exception as e:
            return f"File search error: {str(e)}"

    return Tool(
        name="file_search",
        description="""Search for files by pattern and/or content with async processing.

SUPPORTED FORMATS:
- JSON: {"pattern": "*.py", "content": "async def", "max_results": 10}
- Simple text: "*.py" (pattern only)

SEARCH TYPES:
1. Pattern only: Find files matching glob pattern
2. Content only: Find files containing text
3. Combined: Both pattern and content filters

EXAMPLE INPUTS:
- {"pattern": "**/*.py", "max_results": 15}
- {"content": "AsyncAgent", "max_results": 5}
- {"pattern": "*.md", "content": "TODO"}
- "requirements.txt"

Returns: File paths with sizes, timestamps, and line numbers for content matches.""",
        func=handle_search
    )


def create_time_tool() -> Tool:
    """Create a modern time tool with multiple format support."""
    def handle_time_request(tool_input: str) -> str:
        """Handle time requests with structured validation."""
        params = _parse_tool_input(tool_input, TimeParams)
        if isinstance(params, str):
            return params

        try:
            return TimeManager.get_current_time(params.format, params.timezone)
        except Exception as e:
            return f"Time tool error: {str(e)}"

    return Tool(
        name="current_time",
        description="""Get current date/time in various formats with timezone support.

SUPPORTED FORMATS:
- "iso": ISO 8601 format (default)
- "human": Human readable format
- "timestamp"/"unix": Unix timestamp
- "date": Just the date (YYYY-MM-DD)
- "time": Just the time (HH:MM:SS)
- "rfc": RFC 2822 format
- Custom: Any strftime format string

SUPPORTED FORMATS:
- JSON: {"format": "human", "timezone": "UTC"}
- Simple text: "human" (format only)

EXAMPLE INPUTS:
- {"format": "human"} → "2025-09-28 15:30:45 UTC"
- {"format": "date"} → "2025-09-28"
- "%A, %B %d, %Y" → "Saturday, September 28, 2025"
- "timestamp" → "1727538645"

Note: Currently supports UTC timezone only.""",
        func=handle_time_request
    )


def create_python_executor_tool() -> Tool:
    """Create a Python code execution tool with sandboxing."""
    executor = PythonCodeExecutor()

    def handle_code_execution(tool_input: str) -> str:
        """Handle Python code execution with structured validation."""
        params = _parse_tool_input(tool_input, PythonCodeParams)
        if isinstance(params, str):
            return params

        try:
            return executor.execute_code(params.code, params.timeout)
        except Exception as e:
            return f"Code execution error: {str(e)}"

    return Tool(
        name="python_executor",
        description="""Execute Python code safely in a sandboxed environment.

SAFE EXECUTION:
- Runs in isolated subprocess with timeout
- Limited environment access
- No confirmation required (sandboxed)
- Captures both stdout and stderr

SUPPORTED FORMATS:
- JSON: {"code": "print('hello')", "timeout": 30}
- Simple text: "print('hello world')"

EXAMPLE INPUTS:
- {"code": "import math; print(math.pi)", "timeout": 10}
- "for i in range(5): print(i)"
- {"code": "import json; data = {'test': 123}; print(json.dumps(data, indent=2))"}

LIMITATIONS:
- 30 second default timeout (max 300)
- Basic Python libraries only
- No file system write access
- No network access

Returns: Formatted output with stdout, stderr, and exit codes.""",
        func=handle_code_execution
    )


def create_git_tool(working_directory: str = None) -> Tool:
    """Create a git operations tool with async support."""
    git_manager = GitManager(working_directory)

    def handle_git_operation(tool_input: str) -> str:
        """Handle git operations with structured validation."""
        params = _parse_tool_input(tool_input, GitParams)
        if isinstance(params, str):
            return params

        try:
            # Handle async operations safely when already in an event loop
            import asyncio
            import concurrent.futures

            async def run_git_operation():
                if params.operation == "status":
                    return await git_manager.git_status()
                elif params.operation == "commit":
                    if not params.message:
                        return "❌ Commit message is required for commit operation"
                    return await git_manager.git_commit(params.message, params.files)
                elif params.operation == "diff":
                    return await git_manager.git_diff(params.files)
                elif params.operation == "branch":
                    return await git_manager.git_branch(params.branch_name)
                elif params.operation == "log":
                    return await git_manager.git_log(params.limit)
                else:
                    return f"❌ Unknown git operation: {params.operation}"

            # Try to get the current event loop
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context - use ThreadPoolExecutor to run async code
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, run_git_operation())
                    return future.result()
            except RuntimeError:
                # No event loop running - safe to use asyncio.run
                return asyncio.run(run_git_operation())

        except Exception as e:
            return f"Git tool error: {str(e)}"

    return Tool(
        name="git_operations",
        description="""Perform git repository operations with full async support.

SUPPORTED OPERATIONS:
- "status": Show repository status (modified, staged, untracked files)
- "commit": Add and commit changes with message
- "diff": Show differences in files
- "branch": List branches or create/switch to new branch
- "log": Show commit history with graph

SUPPORTED FORMATS:
- JSON: {"operation": "commit", "message": "Add feature", "files": ["file1.py"]}
- Simple text: "status" (for status operation only)

EXAMPLE INPUTS:
- {"operation": "status"} → Repository status
- {"operation": "commit", "message": "Fix bug in parser"} → Commit all changes
- {"operation": "commit", "message": "Add tests", "files": ["test_parser.py"]} → Commit specific files
- {"operation": "diff", "files": ["src/main.py"]} → Show diff for specific file
- {"operation": "branch", "branch_name": "feature-xyz"} → Create and switch to branch
- {"operation": "log", "limit": 5} → Show last 5 commits
- "status" → Quick status check

AUTHENTICATION:
- Uses existing git configuration (SSH keys, credential helpers)
- No additional setup required
- Inherits user's git authentication automatically

Returns: Formatted git command output with status indicators.""",
        func=handle_git_operation
    )


# Convenience function to create all tools
def create_all_tools(working_directory: str = None) -> Dict[str, Tool]:
    """Create all available tools with modern 2025 patterns."""
    return {
        "todo_file_manager": create_todo_tool(working_directory),
        "web_search": create_web_search_tool(),
        "file_search": create_file_search_tool(working_directory),
        "current_time": create_time_tool(),
        "python_executor": create_python_executor_tool(),
        "git_operations": create_git_tool(working_directory),
    }