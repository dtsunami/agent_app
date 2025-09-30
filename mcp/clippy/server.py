#!/usr/bin/env python3
"""
Clippy - CLIPboard PYthon MCP Server
Lightweight MCP server for Windows clipboard access

Usage:
    clippy [--port 8000]

On startup, copies connection prompt to clipboard for pasting into da_code.
"""

import argparse
import asyncio
import base64
import io
import socket
import sys
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Clipboard dependencies
try:
    import pyperclip
    from PIL import ImageGrab
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install clippy with: pip install -e .")
    sys.exit(1)


class MCPRequest(BaseModel):
    """MCP tool call request."""
    arguments: Dict[str, Any] = {}


class MCPResponse(BaseModel):
    """MCP tool call response."""
    content: list
    isError: bool = False


class ClippyServer:
    """Clippy - Windows clipboard MCP server."""

    def __init__(self, port: int = 8000):
        self.port = port
        self.app = FastAPI(
            title="Clippy - Windows Clipboard MCP Server",
            description="Remote clipboard access for da_code agents",
            version="1.0.0"
        )

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self.setup_routes()
        self.tools = {
            "read_text": {
                "name": "read_text",
                "description": "Read text content from Windows clipboard",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                }
            },
            "read_image": {
                "name": "read_image",
                "description": "Read image from Windows clipboard and return as base64",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "format": {
                            "type": "string",
                            "enum": ["PNG", "JPEG"],
                            "default": "PNG"
                        }
                    },
                    "additionalProperties": False
                }
            },
            "write_text": {
                "name": "write_text",
                "description": "Write text content to Windows clipboard",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to write to clipboard"
                        }
                    },
                    "required": ["text"],
                    "additionalProperties": False
                }
            },
            "write_image": {
                "name": "write_image",
                "description": "Write base64 image to Windows clipboard",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "image_data": {
                            "type": "string",
                            "description": "Base64 encoded image data"
                        },
                        "format": {
                            "type": "string",
                            "enum": ["PNG", "JPEG"],
                            "default": "PNG"
                        }
                    },
                    "required": ["image_data"],
                    "additionalProperties": False
                }
            }
        }

    def setup_routes(self):
        """Setup FastAPI routes for MCP protocol."""

        @self.app.get("/")
        async def root():
            return {
                "name": "Clippy - Windows Clipboard MCP Server",
                "version": "1.0.0",
                "status": "running",
                "tools_available": len(self.tools),
                "connection_prompt": self.generate_connection_prompt()
            }

        @self.app.get("/mcp/tools")
        async def list_tools():
            """List available MCP tools."""
            return {"tools": list(self.tools.values())}

        @self.app.post("/mcp/call/{tool_name}")
        async def call_tool(tool_name: str, request: MCPRequest):
            """Execute an MCP tool."""
            if tool_name not in self.tools:
                raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")

            try:
                if tool_name == "read_text":
                    result = self._read_clipboard_text()
                elif tool_name == "read_image":
                    image_format = request.arguments.get("format", "PNG")
                    result = self._read_clipboard_image(image_format)
                elif tool_name == "write_text":
                    text = request.arguments.get("text")
                    if not text:
                        result = "❌ Error: 'text' parameter is required"
                    else:
                        result = self._write_clipboard_text(text)
                elif tool_name == "write_image":
                    image_data = request.arguments.get("image_data")
                    image_format = request.arguments.get("format", "PNG")
                    if not image_data:
                        result = "❌ Error: 'image_data' parameter is required"
                    else:
                        result = self._write_clipboard_image(image_data, image_format)
                else:
                    raise HTTPException(status_code=400, detail=f"Unknown tool: {tool_name}")

                return MCPResponse(content=[{"type": "text", "text": result}])

            except Exception as e:
                return MCPResponse(
                    content=[{"type": "text", "text": f"Error: {str(e)}"}],
                    isError=True
                )

        @self.app.get("/mcp/connect")
        async def get_connection_prompt():
            """Get connection prompt for da_code."""
            return {"prompt": self.generate_connection_prompt()}

    def get_local_ip(self) -> str:
        """Get local network IP address."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "localhost"

    def generate_connection_prompt(self) -> str:
        """Generate compact command for da_code agent."""
        import json
        ip = self.get_local_ip()
        config = {
            "name": "clipboard",
            "url": f"http://{ip}:{self.port}",
            "port": self.port,
            "description": f"Windows clipboard MCP server at {ip}",
            "tools": list(self.tools.keys())
        }
        # Return compact JSON as complete command
        return f"add_mcp {json.dumps(config, separators=(',', ':'))}"

    def _read_clipboard_text(self) -> str:
        """Read text from Windows clipboard."""
        try:
            text = pyperclip.paste()

            if not text:
                return "📋 Clipboard is empty or contains no text"

            if len(text) > 10000:
                return f"📋 **Clipboard Text** (truncated from {len(text):,} chars):\n\n{text[:10000]}...\n\n*(truncated)*"

            return f"📋 **Clipboard Text:**\n\n{text}"

        except Exception as e:
            return f"❌ Error reading clipboard text: {str(e)}"

    def _read_clipboard_image(self, image_format: str = "PNG") -> str:
        """Read image from Windows clipboard."""
        try:
            image = ImageGrab.grabclipboard()

            if image is None:
                return "📋 No image found in clipboard. Copy an image first."

            width, height = image.size
            mode = image.mode

            buffer = io.BytesIO()
            image.save(buffer, format=image_format)
            image_bytes = buffer.getvalue()

            image_b64 = base64.b64encode(image_bytes).decode()

            result = f"🖼️ **Image from Clipboard:**\n"
            result += f"**Dimensions:** {width} x {height} pixels\n"
            result += f"**Mode:** {mode}\n"
            result += f"**Format:** {image_format}\n"
            result += f"**Size:** {len(image_bytes):,} bytes\n"
            result += f"**Base64 Length:** {len(image_b64):,} characters\n\n"
            result += f"**Base64 Data:**\n{image_b64}\n\n"
            result += "✅ Image successfully read from clipboard"

            return result

        except Exception as e:
            return f"❌ Error reading clipboard image: {str(e)}"

    def _write_clipboard_text(self, text: str) -> str:
        """Write text to Windows clipboard."""
        try:
            pyperclip.copy(text)
            char_count = len(text)
            return f"✅ **Text Written to Clipboard:**\n\n{char_count:,} characters written successfully"
        except Exception as e:
            return f"❌ Error writing clipboard text: {str(e)}"

    def _write_clipboard_image(self, image_data: str, image_format: str = "PNG") -> str:
        """Write base64 image to Windows clipboard."""
        try:
            from PIL import Image
            import io

            # Decode base64 image
            try:
                image_bytes = base64.b64decode(image_data)
                image_buffer = io.BytesIO(image_bytes)
                image = Image.open(image_buffer)
            except Exception as e:
                return f"❌ Invalid base64 image data: {str(e)}"

            # Copy to clipboard using PIL
            # Note: This requires PIL to be able to write to clipboard
            # On Windows, we need to use a different approach
            try:
                # Save to temporary location and use system clipboard
                import tempfile
                import os
                with tempfile.NamedTemporaryFile(suffix=f".{image_format.lower()}", delete=False) as tmp:
                    image.save(tmp.name, format=image_format)
                    tmp_path = tmp.name

                # Use PowerShell to copy image to clipboard
                import subprocess
                ps_command = f'Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.Clipboard]::SetImage([System.Drawing.Image]::FromFile("{tmp_path}"))'
                subprocess.run(['powershell', '-Command', ps_command], check=True)

                # Clean up temp file
                os.unlink(tmp_path)

                width, height = image.size
                return f"✅ **Image Written to Clipboard:**\n\n{width}x{height} pixels ({image_format} format) written successfully"

            except subprocess.CalledProcessError as e:
                return f"❌ Error copying image to clipboard: {str(e)}"
            except Exception as e:
                return f"❌ Error processing image: {str(e)}"

        except Exception as e:
            return f"❌ Error writing clipboard image: {str(e)}"

    def copy_connection_prompt_to_clipboard(self):
        """Copy complete add_mcp command to clipboard for easy pasting."""
        try:
            command = self.generate_connection_prompt()
            pyperclip.copy(command)
            print(f"✅ Complete command copied to clipboard:")
            print(f"   {command}")
        except Exception as e:
            print(f"❌ Could not copy to clipboard: {e}")

    async def start_server(self):
        """Start the Clippy MCP server."""
        print(f"\n📎 Starting Clippy (CLIPboard PYthon) MCP Server on port {self.port}")
        print(f"📋 Local access: http://localhost:{self.port}")
        print(f"🌐 Network access: http://{self.get_local_ip()}:{self.port}")
        print(f"🔧 Tools: {', '.join(self.tools.keys())}")

        self.copy_connection_prompt_to_clipboard()

        print(f"\n📝 Paste the command above into your da_code agent to enable remote clipboard access.")
        print(f"⏹️  Press Ctrl+C to stop the server\n")

        config = uvicorn.Config(
            self.app,
            host="0.0.0.0",
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Clippy - Windows Clipboard MCP Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run server on (default: 8000)")
    args = parser.parse_args()

    server = ClippyServer(port=args.port)

    try:
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        print("\n👋 Clippy stopped")


if __name__ == "__main__":
    main()