"""Pydantic models for Search MCP Server."""

from typing import Optional
from pydantic import BaseModel, Field


class SearchConfig(BaseModel):
    """Configuration for Search MCP Server."""

    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8003, description="Server port")
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: str = Field(default="search_mcp.log", description="Log file path")

    # Search configuration
    max_results: int = Field(default=10, description="Maximum search results")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    user_agent: str = Field(default="Mozilla/5.0 (compatible; SearchMCP/1.0)", description="User agent string")


class SearchResult(BaseModel):
    """Model for search results."""

    title: str
    url: str
    snippet: str
    source: str
    published_date: Optional[str] = None


class ContentExtraction(BaseModel):
    """Model for extracted content."""

    title: str
    content: str
    url: str
    word_count: int
    extracted_at: str


class NewsResult(BaseModel):
    """Model for news search results."""

    title: str
    url: str
    summary: str
    published_date: str
    source: str
    category: Optional[str] = None