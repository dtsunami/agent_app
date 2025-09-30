"""ToolSession MCP Server - Persistent interactive tool sessions using FastAPI."""

import asyncio
import os
import subprocess
import sys
import threading
import time
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from basemcp.server import BaseMCPServer
from models import ToolSessionConfig


class ToolSessionOperations:
    """Tool session operations handler."""

    def __init__(self, config: ToolSessionConfig):
        self.config = config
        self.process = None
        self.session_active = False
        self.output_buffer = ""
        self.output_lock = threading.Lock()
        self.prompt_detected = threading.Event()

        # Configuration from config object
        self.working_dir = self.config.session.working_directory
        self.command = self.config.session.command
        self.prompt_string = self.config.session.prompt_string
        self.timeout = self.config.session.timeout

    def get_tools(self):
        """Get available tools."""
        return [
            {
                "name": "execute_command",
                "description": "Execute a command in the persistent session",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Command to execute"}
                    },
                    "required": ["command"]
                }
            },
            {
                "name": "get_output",
                "description": "Get output from the session",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "lines": {"type": "integer", "description": "Number of recent lines (0 = all)"}
                    }
                }
            },
            {
                "name": "execute_script",
                "description": "Execute a script in the session",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "script_content": {"type": "string", "description": "Script content"},
                        "language": {"type": "string", "description": "Script language (python, bash, etc.)"}
                    },
                    "required": ["script_content"]
                }
            },
            {
                "name": "get_status",
                "description": "Get current session status",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "clear_output",
                "description": "Clear the output buffer",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]

    def monitor_output(self):
        """Monitor process output in a separate thread."""
        try:
            while self.process and self.process.poll() is None and self.session_active:
                if self.process.stdout:
                    char = self.process.stdout.read(1)
                    if not char:
                        break

                    with self.output_lock:
                        self.output_buffer += char

                    # Check for prompt detection
                    if self.output_buffer.endswith(self.prompt_string):
                        self.prompt_detected.set()

        except Exception as e:
            print(f"Output monitoring error: {e}")

    def wait_for_prompt(self, timeout: int = None) -> bool:
        """Wait for the prompt to appear."""
        if timeout is None:
            timeout = self.timeout
        self.prompt_detected.clear()
        return self.prompt_detected.wait(timeout)

    def start_session(self) -> bool:
        """Start the persistent session."""
        if self.session_active and self.process and self.process.poll() is None:
            return True

        try:
            # Change to working directory
            os.chdir(self.working_dir)

            # Start the process
            self.process = subprocess.Popen(
                self.command,
                shell=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            self.session_active = True
            self.output_buffer = ""

            # Start output monitoring thread
            monitor_thread = threading.Thread(target=self.monitor_output, daemon=True)
            monitor_thread.start()

            # Wait for initial prompt
            if self.wait_for_prompt(30):
                return True
            else:
                self.stop_session()
                return False

        except Exception as e:
            self.session_active = False
            return False

    def stop_session(self):
        """Stop the session."""
        self.session_active = False

        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
            except Exception:
                pass

            self.process = None

    async def execute(self, tool_name: str, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute a tool and return MCP-compatible content."""
        if tool_name == "execute_command":
            result = await self.execute_command_tool(arguments)
        elif tool_name == "get_output":
            result = await self.get_output_tool(arguments)
        elif tool_name == "execute_script":
            result = await self.execute_script_tool(arguments)
        elif tool_name == "get_status":
            result = await self.get_status_tool(arguments)
        elif tool_name == "clear_output":
            result = await self.clear_output_tool(arguments)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

        return [{"type": "text", "text": result}]

    async def execute_command_tool(self, args: Dict[str, Any]) -> str:
        """Execute a command in the persistent session."""
        if not self.session_active or not self.process:
            return "Persistent session not available."

        if self.process.poll() is not None:
            # Try to restart the session
            if not self.start_session():
                return "Session terminated and failed to restart."

        command = args.get("command")
        if not command:
            return "Command is required"

        try:
            # Clear previous output buffer content after the last prompt
            with self.output_lock:
                if self.prompt_string in self.output_buffer:
                    last_prompt_idx = self.output_buffer.rfind(self.prompt_string)
                    if last_prompt_idx != -1:
                        self.output_buffer = self.output_buffer[last_prompt_idx + len(self.prompt_string):]

            # Send command
            self.process.stdin.write(command + '\n')
            self.process.stdin.flush()

            # Wait for prompt to return
            if self.wait_for_prompt():
                with self.output_lock:
                    result = self.output_buffer.strip()
                    # Remove the prompt from the end if present
                    if result.endswith(self.prompt_string):
                        result = result[:-len(self.prompt_string)].strip()
                    # Remove the echoed command from the beginning if present
                    if result.startswith(command):
                        result = result[len(command):].strip()

                return f"Command executed successfully.\nOutput:\n{result}" if result else "Command executed successfully (no output)"
            else:
                return f"Command may still be running (timeout after {self.timeout}s). Use get_output() to check for results."

        except Exception as e:
            return f"Failed to execute command: {e}"

    async def get_output_tool(self, args: Dict[str, Any]) -> str:
        """Get the current output buffer."""
        if not self.session_active:
            return "No active session"

        lines = args.get("lines", 0)

        with self.output_lock:
            content = self.output_buffer.strip()

            if lines > 0 and content:
                content_lines = content.split('\n')
                content = '\n'.join(content_lines[-lines:])

            return content if content else "No output available"

    async def execute_script_tool(self, args: Dict[str, Any]) -> str:
        """Execute a script in the persistent session."""
        if not self.session_active or not self.process:
            return "Persistent session not available."

        script_content = args.get("script_content")
        language = args.get("language", "python")

        if not script_content:
            return "Script content is required"

        try:
            # Create temporary script file
            suffix = {
                "python": ".py",
                "bash": ".sh",
                "javascript": ".js",
                "r": ".r"
            }.get(language.lower(), ".txt")

            with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
                f.write(script_content)
                script_path = f.name

            try:
                # Execute script based on language
                if language.lower() == "python":
                    #command = f"exec(open('{script_path}').read())"
                    command = f"import {script_path.replace('.py', '').replace('/tmp/', '')}"
                elif language.lower() == "bash":
                    command = f"import subprocess; subprocess.run(['bash', '{script_path}'])"
                else:
                    command = f"# Script execution for {language} not implemented"

                result = await self.execute_command_tool({"command": command})
                return result

            finally:
                # Clean up script file
                try:
                    os.unlink(script_path)
                except:
                    pass

        except Exception as e:
            return f"Failed to execute script: {e}"

    async def get_status_tool(self, args: Dict[str, Any]) -> str:
        """Get current session status."""
        if not self.session_active:
            return "Persistent session not active"

        if self.process:
            if self.process.poll() is None:
                status = f"Persistent session active (PID: {self.process.pid})"
            else:
                status = f"Persistent session process terminated (exit code: {self.process.returncode})"
        else:
            status = "Persistent session state unknown"

        return f"{status}\nWorking directory: {self.working_dir}\nPrompt string: {self.prompt_string}\nCommand: {self.command}"

    async def clear_output_tool(self, args: Dict[str, Any]) -> str:
        """Clear the output buffer."""
        with self.output_lock:
            self.output_buffer = ""

        return "Output buffer cleared"


class ToolSessionMCPServer(BaseMCPServer):
    """MCP Server for ToolSession operations."""

    def __init__(self, config: ToolSessionConfig):
        super().__init__("ToolSession", "1.0.0")
        self.config = config
        self.tool_ops = ToolSessionOperations(config)

    async def on_startup(self):
        """Start persistent session on startup."""
        await super().on_startup()

        # Start the persistent session immediately
        self.logger.info(f"Starting persistent session with command: {self.tool_ops.command}")
        if self.tool_ops.start_session():
            self.logger.info(f"Persistent session started successfully (PID: {self.tool_ops.process.pid})")
        else:
            self.logger.error("Failed to start persistent session")

    async def on_shutdown(self):
        """Stop persistent session on shutdown."""
        if self.tool_ops.session_active:
            self.tool_ops.stop_session()
            self.logger.info("Persistent session stopped")
        await super().on_shutdown()

    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status including session info."""
        base_status = await super().get_health_status()
        base_status.update({
            "session_active": self.tool_ops.session_active,
            "session_pid": self.tool_ops.process.pid if self.tool_ops.process else None,
            "working_directory": self.tool_ops.working_dir,
            "session_command": self.tool_ops.command,
        })
        return base_status

    async def handle_tools_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/list request."""
        tools = []
        for tool in self.tool_ops.get_tools():
            tools.append({
                "name": tool["name"],
                "description": tool["description"],
                "inputSchema": tool["inputSchema"],
            })
        return {"tools": tools}

    async def handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if not tool_name:
            raise ValueError("Tool name is required")

        self.logger.info(f"Executing tool: {tool_name} with args: {arguments}")

        # Execute the tool
        result = await self.tool_ops.execute(tool_name, arguments)

        return {"content": result, "isError": False}


def main():
    """Main entry point for the MCP server."""
    import argparse

    parser = argparse.ArgumentParser(description="ToolSession MCP Server")
    parser.add_argument("--config", help="Path to configuration file", default="config.json")

    args = parser.parse_args()

    # Load configuration (from file and environment variables)
    config = ToolSessionConfig.load(args.config)

    # Run the server
    server = ToolSessionMCPServer(config)
    server.run(host=config.server.host, port=config.server.port)


if __name__ == "__main__":
    main()