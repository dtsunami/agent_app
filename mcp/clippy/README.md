# Clippy - CLIPboard PYthon MCP Server

📎 Lightweight MCP server for Windows clipboard access, enabling remote clipboard reading and writing for da_code agents running on Linux.

## Quick Start

1. **Install on Windows machine:**
   ```bash
   git clone <repo>
   cd mcp/clippy
   pip install -e .
   ```

2. **Launch Clippy:**
   ```bash
   clippy
   # or with custom port:
   clippy --port 8080
   ```

3. **Connect from da_code:**
   - Clippy automatically copies JSON connection config to clipboard
   - Use `add_mcp <JSON_CONFIG>` in your da_code agent running on Linux
   - Example: `add_mcp {"name": "clipboard", "url": "http://192.168.1.100:8000", "port": 8000, "description": "Windows clipboard", "tools": ["read_text", "read_image", "write_text", "write_image"]}`

## Features

- 📋 **Read text from clipboard** - Access copied text remotely
- 🖼️ **Read images from clipboard** - Get screenshots as base64 data
- ✍️ **Write text to clipboard** - Send text to Windows clipboard
- 🖼️ **Write images to clipboard** - Send base64 images to Windows clipboard
- 🌐 **Network accessible** - Works across machines on same network
- 📎 **Auto-connect config** - Copies JSON configuration to clipboard on startup

## Tools Available

### `read_text`
Reads text content from Windows clipboard.

### `read_image`
Reads image from Windows clipboard and returns as base64 data.
- Supports PNG and JPEG formats
- Includes image metadata (dimensions, size, etc.)

### `write_text`
Writes text content to Windows clipboard.
- Parameter: `text` (string)

### `write_image`
Writes base64 image data to Windows clipboard.
- Parameters: `image_data` (base64 string), `format` (PNG/JPEG)
- Uses PowerShell for clipboard image operations

## Requirements

- Windows (for clipboard access)
- Python 3.9+
- Network connectivity between Windows and Linux machines

## Usage Example

```bash
# Start Clippy on Windows
clippy --port 8000

# JSON config automatically copied to clipboard:
# {
#   "name": "clipboard",
#   "url": "http://192.168.1.100:8000",
#   "port": 8000,
#   "description": "Windows clipboard MCP server at 192.168.1.100",
#   "tools": ["read_text", "read_image", "write_text", "write_image"]
# }

# On Linux with da_code, use:
# add_mcp <paste_JSON_config>
```

## Architecture

Clippy implements the MCP (Model Context Protocol) over HTTP using FastAPI, providing:
- Standard MCP tool discovery (`/mcp/tools`)
- Tool execution endpoints (`/mcp/call/{tool_name}`)
- JSON connection config generation (`/mcp/connect`)

Perfect for enabling cross-platform clipboard access in agent workflows!