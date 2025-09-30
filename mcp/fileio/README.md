# FileIO MCP Server

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![MCP Protocol](https://img.shields.io/badge/MCP-2024--11--05-green.svg)](https://github.com/anthropics/mcp)
[![Docker](https://img.shields.io/badge/docker-supported-blue.svg)](https://docker.com)
[![Tests](https://img.shields.io/badge/tests-passing-green.svg)](#testing)

A state-of-the-art Model Context Protocol (MCP) server providing AI agents with secure file operations for n8n workflow directories. Features advanced file operations, race condition prevention, and comprehensive testing.

## 🚀 Features

### Core File Operations
- **📖 Read Files**: Access file contents with encoding support
- **✏️  Write Files**: Create/modify files with extension filtering
- **➕ Append Content**: Add content to existing files
- **❌ Delete Files**: Secure file deletion with confirmation
- **📄 File Info**: Get detailed metadata (size, timestamps, type)
- **✅ Existence Checks**: Verify file presence and type

### Advanced Operations (NEW!)
- **📋 Copy Files**: Copy between workflow stages with locking
- **🔄 Move Files**: Move/rename with race condition prevention
- **🗜️  Compress Files**: Create ZIP archives from multiple files
- **📦 Extract Files**: Extract ZIP archives with security checks
- **🔒 File Locking**: Prevent race conditions in concurrent operations

### Directory Operations
- **📁 List Files**: Browse with pattern filtering and recursion
- **🌳 Directory Tree**: Visualize nested directory structure
- **📊 Statistics**: Comprehensive directory metrics
- **🔍 Search**: Find files by name or content patterns

### Security & Performance
- **🛡️  Sandboxed Access**: Restricted to configured directories
- **🔐 Path Validation**: Blocks directory traversal attacks
- **⚡ File Locking**: Prevents race conditions
- **📏 Size Limits**: Configurable file size restrictions
- **🎯 Extension Filtering**: Control writable file types
- **📝 Audit Logging**: Complete operation history with MongoDB

## 🏗️ Architecture

### MCP Protocol Implementation
This server implements the **Model Context Protocol (MCP) v2024-11-05** specification:

- ✅ **JSON-RPC 2.0** compliant communication
- ✅ **Tool Discovery** and execution
- ✅ **Error Handling** with standard codes
- ✅ **Multiple Transports** (HTTP, WebSocket, Stdio)
- ✅ **Capability Negotiation** between client/server
- ✅ **Async Operations** for high performance

### Workflow Integration
Perfect for **n8n workflow automation** with three-stage processing:

```
📥 ingress/ → 🔄 wip/ → ✅ completed/
```

- **Ingress**: Raw data input and initial processing
- **WIP**: Active workflow processing and transformation
- **Completed**: Final outputs and archived results

## 🛠️ Installation & Setup

### Quick Start with Docker (Recommended)

```bash
# 1. Clone and navigate
cd /mnt/blk/lostboy/mcp/fileio

# 2. Setup development environment
./dev-setup.sh

# 3. Run local CI pipeline
./ci-local.sh

# 4. Build and deploy
docker build -t fileio:advanced0 .
docker-compose up -d fileio
```

### Manual Installation

```bash
# Install with all dependencies
pip install -e .[dev]

# Run tests
pytest

# Start server
python mcp_server.py
```

## 🧪 Testing (State-of-the-Art)

### Comprehensive Test Suite

```bash
# Full CI pipeline (local)
./ci-local.sh

# Individual test categories
pytest tests/unit/         # Unit tests
pytest tests/integration/  # Integration tests
pytest tests/performance/  # Performance benchmarks

# Test with coverage
pytest --cov=. --cov-report=html

# Security scanning
bandit -r .
safety check
```

### Test Categories
- **Unit Tests**: Fast, isolated component testing
- **Integration Tests**: Real MCP protocol workflows
- **Performance Tests**: Benchmarking and load testing
- **Security Tests**: Path traversal, input validation
- **Race Condition Tests**: Concurrent operation safety

### Coverage Targets
- 📊 **90%+ Overall Coverage**
- 🎯 **100% Critical Path Coverage**
- ⚡ **Sub-second Test Execution**
- 🔄 **CI/CD Integration Ready**

## 📡 MCP Protocol Usage

### JSON-RPC 2.0 Interface

**Initialize Connection:**
```json
POST /mcp
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2024-11-05",
    "capabilities": {"tools": {}},
    "clientInfo": {"name": "n8n", "version": "1.0.0"}
  }
}
```

**List Available Tools:**
```json
POST /mcp
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/list",
  "params": {}
}
```

**Execute Tool:**
```json
POST /mcp
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/call",
  "params": {
    "name": "copy_file",
    "arguments": {
      "source_directory": "ingress",
      "source_path": "data.csv",
      "target_directory": "wip",
      "target_path": "processing/data.csv"
    }
  }
}
```

## 🔧 Configuration

### Main Config (`config.json`)
```json
{
  "name": "fileio",
  "version": "1.0.0",
  "base_path": "/cwd",
  "allowed_directories": ["ingress", "wip", "completed"],
  "max_file_size": 10485760,
  "allowed_extensions": [".txt", ".json", ".csv", ".md", ".log", ".xml", ".yaml", ".yml", ".py", ".js", ".html", ".css", ".zip"],
  "security": {
    "enable_write": true,
    "enable_delete": true,
    "sandbox_mode": true
  },
  "logging": {
    "level": "INFO",
    "file": "/app/logs/fileio.log"
  },
  "server": {
    "host": "0.0.0.0",
    "port": 8000
  }
}
```

### Environment Variables
```bash
# MongoDB connection
MONGO_URI=mongodb://user:pass@mongo:27017/db?authSource=admin

# File operations
FILEIO_BASE_PATH=/cwd
FILEIO_ALLOWED_DIRS=ingress,wip,completed
FILEIO_MAX_FILE_SIZE=10485760

# Server configuration
FILEIO_HOST=0.0.0.0
FILEIO_PORT=8000
```

## 🎯 Available Tools

### File Operations

| Tool | Description | Key Features |
|------|-------------|--------------|
| `read_file` | Read file content | Encoding support, size limits |
| `write_file` | Create/modify files | Directory creation, validation |
| `append_to_file` | Append to existing files | Encoding support |
| `delete_file` | Delete files securely | Confirmation required |
| `get_file_info` | File metadata | Size, timestamps, type |
| `check_file_exists` | File existence check | Fast validation |

### Advanced Operations

| Tool | Description | Race-Safe |
|------|-------------|-----------|
| `copy_file` | Copy between directories | ✅ File locking |
| `move_file` | Move/rename files | ✅ Atomic operations |
| `compress_file` | Create ZIP archives | ✅ Multi-file support |
| `extract_file` | Extract ZIP archives | ✅ Security validation |
| `file_lock` | Lock management | ✅ Timeout handling |

### Directory Operations

| Tool | Description | Features |
|------|-------------|----------|
| `list_files` | Browse directories | Filters, recursion, details |
| `get_directory_tree` | Tree visualization | Depth control, ASCII art |
| `get_directory_stats` | Directory metrics | Size, counts, types |
| `search_files` | Content/name search | Regex, case sensitivity |

## 🔐 Security Features

### File System Security
- **🛡️  Sandboxing**: Operations restricted to configured base path
- **🔍 Path Validation**: Prevents directory traversal (`../` attacks)
- **📝 Extension Filtering**: Controls writable file types
- **📏 Size Limits**: Prevents resource exhaustion
- **🔒 File Locking**: Race condition prevention

### Container Security
- **👤 Non-root User**: Server runs as unprivileged `mcpuser`
- **📖 Read-only Mounts**: Base system mounted read-only
- **🌐 Network Isolation**: Limited network access
- **🔒 Resource Limits**: CPU and memory constraints

### Audit & Compliance
- **📊 MongoDB Logging**: All operations logged with timing
- **⚙️  Configuration Tracking**: Settings changes recorded
- **📈 Usage Analytics**: Pattern analysis for security review
- **🚨 Error Monitoring**: Failed operations flagged

## 🚀 Performance Optimizations

### File Operations
- **⚡ Async I/O**: All operations non-blocking
- **🔒 Smart Locking**: Minimal lock duration
- **📦 Streaming**: Large file handling without memory loading
- **🗜️  Compression**: ZIP operations with progress tracking

### Database Integration
- **🔗 Connection Pooling**: Efficient MongoDB connections
- **📊 Batch Logging**: Reduced database overhead
- **💾 Caching**: Optional response caching for reads
- **⏱️  Timeout Management**: Graceful operation timeouts

## 📊 Monitoring & Observability

### Health Endpoints
```bash
# Application health
GET /health

# MongoDB connectivity
GET /health/mongo

# File system access
GET /health/filesystem
```

### Metrics Collection
- **📈 Operation Latency**: P95, P99 response times
- **📊 Throughput**: Operations per second
- **❌ Error Rates**: Failed operation percentages
- **💾 Resource Usage**: Memory, disk, CPU metrics

### Logging Integration
```bash
# View real-time logs
docker-compose logs -f fileio

# MongoDB execution logs
db.mcp_executions.find().sort({timestamp: -1}).limit(10)

# Application logs
tail -f /app/logs/fileio.log
```

## 🔗 Integration Examples

### n8n Workflow Node
```javascript
// Use HTTP Request node with MCP endpoint
{
  "method": "POST",
  "url": "http://fileio_mcp:8000/mcp",
  "body": {
    "jsonrpc": "2.0",
    "id": "{{ $runIndex }}",
    "method": "tools/call",
    "params": {
      "name": "move_file",
      "arguments": {
        "source_directory": "wip",
        "source_path": "{{ $json.filename }}",
        "target_directory": "completed",
        "target_path": "processed/{{ $json.filename }}"
      }
    }
  }
}
```

### Claude Desktop Integration
```json
{
  "mcpServers": {
    "fileio": {
      "command": "docker",
      "args": [
        "exec", "-i", "fileio_mcp",
        "python", "mcp_server.py"
      ]
    }
  }
}
```

### Direct HTTP Integration
```python
import httpx

async def call_mcp_tool(tool_name: str, arguments: dict):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:3456/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }
        )
        return response.json()

# Example usage
result = await call_mcp_tool("compress_file", {
    "files": [
        {"directory": "completed", "path": "data1.csv"},
        {"directory": "completed", "path": "data2.csv"}
    ],
    "archive_directory": "completed",
    "archive_path": "backup.zip"
})
```

## 🐛 Troubleshooting

### Common Issues

**Container won't start:**
```bash
docker-compose logs fileio
docker-compose ps
# Check for port conflicts, volume permissions
```

**MongoDB connection failed:**
```bash
# Verify MongoDB container
docker-compose logs mongo

# Test connection
docker-compose exec fileio python -c "
import os; from motor.motor_asyncio import AsyncIOMotorClient;
print('URI:', os.getenv('MONGO_URI'))
"
```

**File operation denied:**
```bash
# Check directory permissions
ls -la /mnt/blk/lostboy/work/

# Verify configuration
docker-compose exec fileio cat config.json | jq .allowed_directories
```

**Tool not found:**
```bash
# List available tools
curl -X POST http://localhost:3456/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
```

### Debug Mode
```bash
# Enable debug logging
export MCP_LOG_LEVEL=DEBUG

# Run server locally
python mcp_server.py --config config.json

# Test with verbose output
pytest -v -s tests/
```

## 🏆 Development

### Code Quality Standards
- **🖤 Black**: Code formatting (88 char line length)
- **📦 isort**: Import sorting with Black profile
- **🔍 Ruff**: Modern, fast Python linting
- **🔍 MyPy**: Static type checking (strict mode)
- **🔒 Bandit**: Security vulnerability scanning
- **⚗️  Safety**: Dependency vulnerability checking

### Pre-commit Hooks
```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### Local CI/CD Pipeline
```bash
# Complete pipeline
./ci-local.sh --full

# Quick checks
./ci-local.sh

# Performance testing only
./ci-local.sh --performance

# Docker integration tests
./ci-local.sh --docker
```

## 📦 Dependencies

### Core Runtime
```toml
dependencies = [
    "mcp>=1.0.0",           # Model Context Protocol
    "pydantic>=2.0.0",      # Data validation
    "fastapi>=0.104.0",     # Web framework
    "uvicorn>=0.24.0",      # ASGI server
    "motor>=3.3.0",         # Async MongoDB driver
    "pymongo>=4.6.0"        # MongoDB Python driver
]
```

### Development Tools
```toml
dev = [
    # Testing framework
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
    "pytest-benchmark>=4.0.0",

    # Code quality
    "black>=23.0.0",
    "isort>=5.12.0",
    "ruff>=0.1.0",
    "mypy>=1.5.0",

    # Security scanning
    "bandit>=1.7.0",
    "safety>=2.3.0",

    # Development utilities
    "pre-commit>=3.3.0"
]
```

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. **Setup Development Environment**: `./dev-setup.sh`
2. **Run Tests**: `./ci-local.sh`
3. **Follow Code Standards**: Pre-commit hooks enforce quality
4. **Write Tests**: Maintain 90%+ coverage
5. **Update Documentation**: Keep README current

## 🔗 Related Projects

- **[MCP Specification](https://github.com/anthropics/mcp)**: Official MCP protocol
- **[n8n](https://n8n.io)**: Workflow automation platform
- **[Claude Desktop](https://claude.ai)**: AI assistant with MCP support
- **[FastAPI](https://fastapi.tiangolo.com)**: Modern Python web framework

---

**Built with ❤️ for the n8n workflow automation community**

*Supports Python 3.9+ • MCP Protocol 2024-11-05 • Docker Ready • Production Tested*