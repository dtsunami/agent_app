"""Pydantic models for MongoDB MCP Server."""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from bson import ObjectId


class MongoConfig(BaseModel):
    """Configuration for MongoDB MCP Server."""

    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8004, description="Server port")
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: str = Field(default="mongo_mcp.log", description="Log file path")

    # MongoDB connection settings
    mongo_uri: Optional[str] = Field(default=None, description="MongoDB connection URI")
    mongo_host: str = Field(default="localhost", description="MongoDB host")
    mongo_port: int = Field(default=27017, description="MongoDB port")
    mongo_user: Optional[str] = Field(default=None, description="MongoDB username")
    mongo_password: Optional[str] = Field(default=None, description="MongoDB password")
    mongo_database: str = Field(default="test", description="Default MongoDB database")


class FindParams(BaseModel):
    """Parameters for find operation."""

    collection: str = Field(description="Collection name")
    filter: Optional[Dict[str, Any]] = Field(default={}, description="Query filter")
    projection: Optional[Dict[str, Any]] = Field(default=None, description="Fields to include/exclude")
    limit: Optional[int] = Field(default=None, description="Maximum number of documents")
    skip: Optional[int] = Field(default=None, description="Number of documents to skip")
    sort: Optional[Dict[str, int]] = Field(default=None, description="Sort specification")


class InsertParams(BaseModel):
    """Parameters for insert operation."""

    collection: str = Field(description="Collection name")
    document: Optional[Dict[str, Any]] = Field(default=None, description="Single document to insert")
    documents: Optional[List[Dict[str, Any]]] = Field(default=None, description="Multiple documents to insert")


class UpdateParams(BaseModel):
    """Parameters for update operation."""

    collection: str = Field(description="Collection name")
    filter: Dict[str, Any] = Field(description="Filter for documents to update")
    update: Dict[str, Any] = Field(description="Update operations")
    upsert: bool = Field(default=False, description="Create document if not found")
    multi: bool = Field(default=False, description="Update multiple documents")


class DeleteParams(BaseModel):
    """Parameters for delete operation."""

    collection: str = Field(description="Collection name")
    filter: Dict[str, Any] = Field(description="Filter for documents to delete")
    multi: bool = Field(default=False, description="Delete multiple documents")


class AggregateParams(BaseModel):
    """Parameters for aggregation operation."""

    collection: str = Field(description="Collection name")
    pipeline: List[Dict[str, Any]] = Field(description="Aggregation pipeline")
    options: Optional[Dict[str, Any]] = Field(default=None, description="Aggregation options")


class IndexParams(BaseModel):
    """Parameters for index operations."""

    collection: str = Field(description="Collection name")
    keys: Optional[Dict[str, int]] = Field(default=None, description="Index keys specification")
    options: Optional[Dict[str, Any]] = Field(default=None, description="Index options")
    index_name: Optional[str] = Field(default=None, description="Index name for operations")


class DatabaseStats(BaseModel):
    """Database statistics model."""

    database: str
    collections: int
    avg_obj_size: float
    data_size: int
    storage_size: int
    indexes: int
    index_size: int
    objects: int


class CollectionStats(BaseModel):
    """Collection statistics model."""

    namespace: str
    count: int
    size: int
    avg_obj_size: float
    storage_size: int
    total_index_size: int
    total_size: int