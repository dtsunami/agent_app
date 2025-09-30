# ToolSession MCP Server

Persistent interactive tool session management using FastAPI and MCP protocol.

## Features

- **Persistent Sessions**: Session starts automatically with server and runs until killed
- **FastAPI-based**: HTTP MCP server using the proven FileIO MCP pattern
- **Real-time I/O**: Captures command output with proper prompt detection
- **Universal Tool Support**: Works with any command-line tool (Python, bash, etc.)
- **Configuration Driven**: JSON config with environment variable overrides
- **Thread-safe Output**: Concurrent output monitoring and command execution

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MCP Client    │◄──►│  ToolSession    │◄──►│ Persistent Tool │
│   (Claude)      │    │   FastAPI       │    │   Session       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │  Output Monitor │
                       │    (Thread)     │
                       └─────────────────┘
```

## MCP Tools

| Tool | Description | Arguments |
|------|-------------|-----------|
| `execute_command` | Execute command in persistent session | `command` (string) |
| `get_output` | Get session output | `lines` (int, optional) |
| `execute_script` | Execute script in session | `script_content` (string), `language` (string, optional) |
| `get_status` | Get current session status | none |
| `clear_output` | Clear output buffer | none |

## Configuration

### Command Line Usage

```bash
# Use default config.json
python server.py

# Specify custom config
python server.py --config my_config.json
```

### Configuration File (config.json)

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8034
  },
  "logging": {
    "level": "INFO",
    "file": "toolsession_mcp.log"
  },
  "session": {
    "command": "python -i -u",
    "working_directory": "/tmp",
    "prompt_string": ">>> ",
    "timeout": 30
  }
}
```

### Environment Variables

```bash
# Server config
TOOLSESSION_HOST=0.0.0.0
TOOLSESSION_PORT=8034

# Logging config
TOOLSESSION_LOG_LEVEL=DEBUG
TOOLSESSION_LOG_FILE=/var/log/toolsession.log

# Session config
TOOLSESSION_COMMAND="python -i -u"
TOOLSESSION_WORKING_DIR=/tmp
TOOLSESSION_PROMPT=">>> "
TOOLSESSION_TIMEOUT=60
```

## Usage Examples

### 1. Execute Python Commands

```python
# Execute Python code
await mcp_call("execute_command", {"command": "print('Hello World!')"})

# Import libraries
await mcp_call("execute_command", {"command": "import numpy as np"})

# Get output
output = await mcp_call("get_output", {"lines": 10})
```

### 2. Execute Scripts

```python
# Python script
script_content = '''
import math
result = math.sqrt(16)
print(f"Square root of 16 is {result}")
'''

await mcp_call("script", {
    "text": script_content,
    "command": "source {script}"
})
```

## Configuration

Edit `config.json` to add/modify tools:

```json
{
  "tools": {
    "my_tool": {
      "environment_command": "source /path/to/setup.sh",
      "launch_command": "my_tool -interactive",
      "prompt_string": "mytool> ",
      "working_directory": "/project/work",
      "timeout": 300
    }
  }
}
```

### Configuration Fields

- **environment_command**: Optional setup command (e.g., sourcing environment)
- **launch_command**: Command to start the tool
- **prompt_string**: String that indicates tool is ready for input
- **working_directory**: Directory to run tool from
- **timeout**: Maximum seconds to wait for prompt

## Installation

```bash
cd mcp/toolsession
pip install -r requirements.txt
python toolsession_server.py --config config.json
```

## Server Endpoints

- **MCP**: `http://localhost:8002/mcp` (JSON-RPC)
- **Health**: `http://localhost:8002/health`

## Example Workflow

```python
# 1. Start session
await start_session(tool="fusion_compiler")

# 2. Load libraries and design
await input(command="read_lib /libs/tech.lib")
await input(command="read_design /designs/cpu.v")

# 3. Execute synthesis script
script = "synthesize -effort high"
await script(text=script, command="source {script}")

# 4. Get results
output = await output(lines=50)

# 5. Clean up
await stop_session()
```

## Error Handling

The server provides detailed error messages for:

- **Session Errors**: Failed to start tool, environment issues
- **Command Errors**: Invalid commands, timeout waiting for prompt
- **Script Errors**: File creation issues, script execution failures
- **Configuration Errors**: Missing tools, invalid settings

## Integration with n8n Stack

The ToolSession server integrates with the existing MCP architecture:

```
mcp/
├── fileio/              # File operations
├── toolsession/         # Interactive tool sessions ←
├── gateway/             # HTTP gateway
└── docker-compose.yml   # Container orchestration
```

Add to `docker-compose.yml`:

```yaml
toolsession-mcp:
  build: ./toolsession
  ports:
    - "8002:8002"
  volumes:
    - ../work:/project/work
    - ./logs:/app/logs
```