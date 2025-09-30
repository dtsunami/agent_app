# MCP Servers for n8n Stack

Model Context Protocol (MCP) servers providing AI agents with access to n8n workflow data and operations.

## Overview

This directory contains containerized MCP servers that extend AI capabilities with:
- **File Operations**: Access to n8n workflow directories (ingress, wip, completed)
- **Database Operations**: Direct queries to PostgreSQL and MongoDB instances
- **Workflow Control**: n8n workflow management and monitoring
- **System Monitoring**: Health checks and performance metrics

## Architecture

```
mcp/
├── fileio/              # File operations MCP server
├── n8n-control/         # n8n workflow management (future)
├── database/            # Database operations (future)
├── monitoring/          # System monitoring (future)
├── gateway/             # Nginx gateway for HTTP access
├── logs/                # Shared logging directory
└── docker-compose.yml   # Container orchestration
```

## Quick Start

### 1. Start MCP Services

```bash
cd mcp
docker compose up -d
```

### 2. Check Service Status

```bash
docker compose ps
```

### 3. View Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f fileio-mcp
```

## Services

### FileIO MCP Server

**Purpose**: Provides AI agents with secure file operations for n8n workflow directories.

**Capabilities**:
- Read files from ingress, wip, completed directories
- Write files to workflow directories (configurable)
- List directory contents with filtering
- Search files by name or content
- Get file metadata and statistics
- Directory tree visualization

**Access**:
- **Container**: `fileio-mcp:8000`
- **Gateway**: `http://localhost:8080/fileio/`
- **Direct**: `http://localhost:8001`

**Configuration**: `fileio/config/fileio_config.json`

### MCP Gateway

**Purpose**: HTTP gateway providing unified access to all MCP servers.

**Features**:
- Load balancing across MCP servers
- Health monitoring
- Request routing
- Centralized logging

**Access**: `http://localhost:8080`

## Configuration

### FileIO Configuration

Edit `fileio/config/fileio_config.json`:

```json
{
  "name": "fileio",
  "base_path": "/mnt/blk/lostboy/work",
  "allowed_directories": ["ingress", "wip", "completed"],
  "max_file_size": 10485760,
  "allowed_extensions": [".txt", ".json", ".csv", ".md", ".log"],
  "security": {
    "enable_write": true,
    "enable_delete": false,
    "sandbox_mode": true
  }
}
```

**Key Settings**:
- `base_path`: Root directory for file operations
- `allowed_directories`: Permitted subdirectories
- `max_file_size`: Maximum file size for read operations (bytes)
- `allowed_extensions`: Permitted file types for write operations
- `security.enable_write`: Allow file write operations
- `security.sandbox_mode`: Restrict access to allowed directories only

## Usage with AI Agents

### Claude Desktop Integration

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "fileio": {
      "command": "docker",
      "args": [
        "exec", "-i", "fileio_mcp",
        "python", "-m", "fileio_mcp.server"
      ]
    }
  }
}
```

### Direct HTTP Access

```bash
# List available tools
curl http://localhost:8080/fileio/tools

# Execute file operation
curl -X POST http://localhost:8080/fileio/execute \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "list_files",
    "arguments": {
      "directory": "ingress",
      "pattern": "*.json"
    }
  }'
```

### Example Operations

**Read a workflow file**:
```json
{
  "tool": "read_file",
  "arguments": {
    "directory": "completed",
    "path": "2024-01-15/processed_data.json"
  }
}
```

**Search for specific content**:
```json
{
  "tool": "search_files",
  "arguments": {
    "directory": "wip",
    "content_pattern": "error",
    "case_sensitive": false
  }
}
```

**Get directory statistics**:
```json
{
  "tool": "get_directory_stats",
  "arguments": {
    "directory": "completed",
    "recursive": true
  }
}
```

## Security

### File Access Control

- **Sandboxed**: Access restricted to configured directories only
- **Extension Filtering**: Write operations limited to safe file types
- **Size Limits**: Large file protection
- **Path Validation**: Prevents directory traversal attacks

### Container Security

- **Non-root User**: Services run as unprivileged user
- **Read-only Volumes**: n8n data mounted read-only where possible
- **Network Isolation**: Services isolated in dedicated Docker network
- **Resource Limits**: CPU and memory constraints

### Logging and Monitoring

- **Comprehensive Logging**: All operations logged with timestamps
- **Health Checks**: Automated service health monitoring
- **Audit Trail**: File operations tracked for security review

## Development

### Building MCP Servers

All MCP servers use a shared foundation located in `mcp/basemcp/`. The build system uses a shared context from the MCP root directory to access these foundation files:

```bash
# Build from project root using docker-compose (recommended)
docker-compose build --no-cache fileio python search mcpmongo

# Or build individual services with proper context
cd /path/to/project
docker-compose build fileio  # Builds fileio:v2
docker-compose build python  # Builds toolsession:v2
docker-compose build search  # Builds search:v2
docker-compose build mcpmongo # Builds mongodb:v2
```

**Current Image Tags**: All MCP services use `v2` tags for the latest stable build.

### Foundation Module Structure

The `mcp/basemcp/` directory contains shared resources:
- `server.py`: Base MCP server class with FastAPI integration (formerly `mcp_foundation.py`)
- `__init__.py`: Python package initialization
- Additional utility modules as needed

**Import Pattern**: All MCP servers import the base class using:
```python
from basemcp.server import BaseMCPServer
```

### Adding New MCP Servers

1. **Create Server Directory**:
   ```bash
   mkdir mcp/new-server
   cd mcp/new-server
   ```

2. **Follow Standard Structure**:
   ```
   new-server/
   ├── server.py           # Main server file
   ├── models.py           # Pydantic models
   ├── Dockerfile          # Docker build file
   └── pyproject.toml      # Python package config
   ```

3. **Dockerfile Template**:
   ```dockerfile
   FROM python:3.12-slim

   # Install system dependencies
   RUN apt-get update && apt-get install -y gcc curl && rm -rf /var/lib/apt/lists/*

   WORKDIR /app

   # Copy base MCP foundation files first
   COPY basemcp/ ./basemcp/

   # Copy application code
   COPY new-server/ .

   # Copy pyproject.toml for package installation
   COPY new-server/pyproject.toml .

   # Install Python dependencies
   RUN pip install --no-cache-dir -e .

   # Create non-root user
   RUN useradd -m -u 1000 newserver && chown -R newserver:newserver /app

   # Set environment variables
   ENV PYTHONPATH=/app
   ENV PYTHONUNBUFFERED=1

   USER newserver

   # Health check
   HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
       CMD curl -f http://localhost:PORT/health || exit 1

   CMD ["python", "server.py"]
   ```

4. **Add to docker-compose.yml**:
   ```yaml
   new-server:
     build:
       context: ./mcp
       dockerfile: new-server/Dockerfile
     image: new-server:v2
     container_name: new_server_mcp
     ports:
       - "PORT:PORT"
   ```

5. **Build the Service**:
   ```bash
   docker-compose build new-server
   ```

6. **Import the Base Server**:
   ```python
   # In your new-server/server.py file
   from basemcp.server import BaseMCPServer

   class NewMCPServer(BaseMCPServer):
       def __init__(self):
           super().__init__(name="new-server")
           # Your server initialization
   ```

6. **Update Gateway**:
   Add routing rules to `gateway/nginx.conf`

### Testing

```bash
# Validate configuration
docker exec fileio_mcp python -m fileio_mcp.server --validate-only

# Run tests
docker exec fileio_mcp python -m pytest

# Check logs
docker compose logs fileio-mcp | tail -100
```

## Troubleshooting

### Common Issues

**Permission Errors**:
```bash
# Fix volume permissions
sudo chown -R 1000:1000 ../work/
```

**Configuration Errors**:
```bash
# Validate config
docker exec fileio_mcp python -c "from fileio_mcp.config import FileIOConfig; FileIOConfig.load().validate()"
```

**Connection Issues**:
```bash
# Check network connectivity
docker exec fileio_mcp ping n8ngui
```

### Log Analysis

```bash
# View real-time logs
docker compose logs -f fileio-mcp

# Check error logs
grep -i error logs/fileio.log

# Monitor file operations
grep "Tool called" logs/fileio.log | tail -20
```

## Roadmap

### Planned MCP Servers

- **n8n-control**: Workflow management and execution control
- **database**: Direct database queries for chat memory and vectors
- **monitoring**: System health and performance metrics
- **analytics**: Usage analytics and workflow optimization

### Future Enhancements

- **Authentication**: JWT-based authentication for HTTP access
- **Rate Limiting**: Request throttling and quota management
- **Caching**: Response caching for improved performance
- **Webhooks**: Event-driven notifications for file changes

## Support

### Logs Location
- **Container Logs**: `docker compose logs [service]`
- **Application Logs**: `logs/fileio.log`
- **Gateway Logs**: `logs/nginx/access.log`, `logs/nginx/error.log`

### Configuration Files
- **FileIO**: `fileio/config/fileio_config.json`
- **Gateway**: `gateway/nginx.conf`
- **Docker**: `docker-compose.yml`

### Health Checks
- **FileIO**: `http://localhost:8001/health`
- **Gateway**: `http://localhost:8080/health`
- **All Services**: `docker compose ps`