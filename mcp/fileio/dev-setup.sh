#!/bin/bash
# FileIO MCP Server - Development Environment Setup
# Modern Python development setup script

set -e

echo "🚀 Setting up FileIO MCP Server development environment..."
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python $required_version or higher is required. Found: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "../../.venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv ../../.venv
else
    echo "📦 Using existing virtual environment..."
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source ../../.venv/bin/activate

# Set PYTHONPATH to current directory for module imports
export PYTHONPATH="$PWD"

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install the package in development mode with all dependencies
echo "📥 Installing FileIO MCP Server in development mode..."
pip install -e .[dev]


# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p reports htmlcov test-data/{ingress,wip,completed}

# Run initial tests to verify setup
echo "🧪 Running initial tests to verify setup..."
pytest tests/unit/ -v --tb=short -x

echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "📚 Usage:"
echo "  # Activate environment:"
echo "  source ../../.venv/bin/activate"
echo "  export PYTHONPATH=\"\$PWD\""
echo ""
echo "  # Run tests:"
echo "  pytest                    # All tests"
echo "  pytest tests/unit/        # Unit tests only"
echo "  pytest tests/integration/ # Integration tests only"
echo "  pytest tests/performance/ # Performance tests only"
echo ""
echo "  # Code quality:"
echo "  black .                   # Format code"
echo "  isort .                   # Sort imports"
echo "  ruff check .              # Lint code"
echo "  mypy .                    # Type check"
echo ""
echo "  # Security:"
echo "  bandit -r .               # Security scan"
echo "  safety check              # Dependency vulnerabilities"
echo ""
echo "  # Coverage:"
echo "  pytest --cov=. --cov-report=html"
echo ""
echo "Happy coding! 🎯"