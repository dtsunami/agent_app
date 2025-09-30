# ClipJS - Node.js MCP Clipboard Server

📎 Lightweight Node.js MCP server for Windows clipboard access, enabling remote clipboard reading for da_code agents running on Linux.

## Quick Start

1. **Install on Windows machine:**
   ```bash
   git clone <repo>
   cd mcp/clipjs
   npm install
   npm link  # Makes 'clipjs' command globally available
   ```

2. **Launch ClipJS:**
   ```bash
   clipjs
   # or with custom port:
   clipjs --port 8080
   ```

3. **Connect from da_code:**
   - ClipJS automatically copies connection prompt to clipboard
   - Paste the prompt into your da_code agent running on Linux
   - Example: `Connect to my Windows clipboard server at http://192.168.1.100:3000 with tools read_clipboard_text and read_clipboard_image for clipboard access`

## Features

- 📋 **Read text from clipboard** - Access copied text remotely
- 🖼️ **Read images from clipboard** - Get screenshots as base64 data (Windows PowerShell)
- 🌐 **Network accessible** - Works across machines on same network
- 🔒 **Read-only access** - Security-focused, no clipboard writing
- 📎 **Auto-connect prompt** - Copies connection string to clipboard on startup
- ⚡ **Lightweight** - Fast Node.js runtime, quick startup

## Tools Available

### `read_clipboard_text`
Reads text content from Windows clipboard using clipboardy.

### `read_clipboard_image`
Reads image from Windows clipboard using PowerShell and returns as base64 data.
- Supports PNG and JPEG formats
- Windows-specific implementation

## Requirements

- Windows (for clipboard access)
- Node.js 18+
- Network connectivity between Windows and Linux machines

## Commands

```bash
# Install dependencies
npm install

# Start server (development)
npm run dev

# Start server (production)
npm start

# Install globally
npm link
clipjs --help
```

## Usage Example

```bash
# Start ClipJS on Windows
clipjs --port 3000

# On Linux with da_code, paste the connection prompt:
# "Connect to my Windows clipboard server at http://192.168.1.100:3000..."
```

## Architecture

ClipJS implements the MCP (Model Context Protocol) over HTTP using Express.js:
- Standard MCP tool discovery (`/mcp/tools`)
- Tool execution endpoints (`/mcp/call/{tool_name}`)
- Connection prompt generation (`/mcp/connect`)

Lightweight alternative to the Python Clippy server, perfect for minimal Windows environments!