#!/bin/bash
# FileIO MCP Server - Local CI/CD Pipeline
# State-of-the-art DevOps practices for local development

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_step() {
    echo -e "${BLUE}🔄 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Create reports directory
mkdir -p reports htmlcov

echo -e "${BLUE}"
echo "===================="
echo "FileIO MCP CI Pipeline"
echo "===================="
echo -e "${NC}"

# Step 1: Environment Check
print_step "Checking environment..."
if [ ! -f "../../.venv/bin/activate" ]; then
    print_warning "Virtual environment not found. Run ./dev-setup.sh first"
    exit 1
fi

source ../../.venv/bin/activate
export PYTHONPATH="$PWD"
print_success "Environment activated"

# Step 2: Code Formatting Check
print_step "Checking code formatting..."
if black --check --diff .; then
    print_success "Code formatting is correct"
else
    print_error "Code formatting issues found. Run 'black .' to fix"
    exit 1
fi

# Step 3: Import Sorting Check
print_step "Checking import sorting..."
if isort --check-only --diff .; then
    print_success "Import sorting is correct"
else
    print_error "Import sorting issues found. Run 'isort .' to fix"
    exit 1
fi

# Step 4: Linting
print_step "Running linter..."
if ruff check . --show-files; then
    print_success "Linting passed"
else
    print_error "Linting issues found"
    exit 1
fi

# Step 5: Type Checking
print_step "Running type checker..."
if mypy . --ignore-missing-imports; then
    print_success "Type checking passed"
else
    print_warning "Type checking issues found (continuing...)"
fi

# Step 6: Unit Tests
print_step "Running unit tests..."
if pytest tests/test_file_operations_sync.py tests/test_file_operations_async.py \
    -v \
    --cov=. \
    --cov-report=term-missing \
    --cov-report=html:htmlcov/unit \
    --cov-report=xml:reports/coverage-unit.xml \
    --junit-xml=reports/junit-unit.xml \
    --maxfail=5; then
    print_success "Unit tests passed"
else
    print_error "Unit tests failed"
    exit 1
fi

# Final Summary
echo -e "${GREEN}"
echo "======================="
echo "CI Pipeline Complete! 🎉"
echo "======================="
echo -e "${NC}"
echo "📊 Reports generated in reports/ directory"
echo "📈 Coverage reports in htmlcov/ directory"
echo "🚀 Ready for deployment!"
