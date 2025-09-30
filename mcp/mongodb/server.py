"""MongoDB MCP Server - Database operations for MongoDB using common foundation."""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from bson import ObjectId, json_util
from bson.errors import InvalidId
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import PyMongoError, DuplicateKeyError, OperationFailure

# Load environment variables
load_dotenv("../../.env")

from basemcp.server import BaseMCPServer
from models import (
    MongoConfig, FindParams, InsertParams, UpdateParams,
    DeleteParams, AggregateParams, IndexParams,
    DatabaseStats, CollectionStats
)


class MongoOperations:
    """Handle MongoDB operations."""

    def __init__(self, config: MongoConfig):
        self.config = config
        self.client: Optional[AsyncIOMotorClient] = None
        self.database: Optional[AsyncIOMotorDatabase] = None

    async def connect(self):
        """Connect to MongoDB."""
        try:
            if self.config.mongo_uri:
                self.client = AsyncIOMotorClient(self.config.mongo_uri)
            else:
                auth_source = ""
                if self.config.mongo_user and self.config.mongo_password:
                    auth_source = f"{self.config.mongo_user}:{self.config.mongo_password}@"

                mongo_url = f"mongodb://{auth_source}{self.config.mongo_host}:{self.config.mongo_port}/{self.config.mongo_database}"
                self.client = AsyncIOMotorClient(mongo_url)

            self.database = self.client[self.config.mongo_database]

            # Test connection
            await self.database.command("ping")
            logging.info("Successfully connected to MongoDB")
            return True
        except Exception as e:
            logging.error(f"Failed to connect to MongoDB: {e}")
            return False

    async def disconnect(self):
        """Disconnect from MongoDB."""
        if self.client:
            self.client.close()

    def serialize_mongo_doc(self, doc: Any) -> Any:
        """Convert MongoDB document to JSON serializable format."""
        return json.loads(json_util.dumps(doc))

    def get_tools(self):
        """Get available MongoDB tools."""
        return [
            {
                "name": "mongo_find",
                "description": "Find documents in a MongoDB collection",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "collection": {"type": "string", "description": "Collection name"},
                        "filter": {"type": "object", "description": "Query filter", "default": {}},
                        "projection": {"type": "object", "description": "Fields to include/exclude"},
                        "limit": {"type": "integer", "description": "Maximum number of documents"},
                        "skip": {"type": "integer", "description": "Number of documents to skip"},
                        "sort": {"type": "object", "description": "Sort specification"}
                    },
                    "required": ["collection"]
                }
            },
            {
                "name": "mongo_insert",
                "description": "Insert document(s) into a MongoDB collection",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "collection": {"type": "string", "description": "Collection name"},
                        "document": {"type": "object", "description": "Single document to insert"},
                        "documents": {"type": "array", "description": "Multiple documents to insert"}
                    },
                    "required": ["collection"]
                }
            },
            {
                "name": "mongo_update",
                "description": "Update document(s) in a MongoDB collection",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "collection": {"type": "string", "description": "Collection name"},
                        "filter": {"type": "object", "description": "Filter for documents to update"},
                        "update": {"type": "object", "description": "Update operations"},
                        "upsert": {"type": "boolean", "description": "Create document if not found", "default": False},
                        "multi": {"type": "boolean", "description": "Update multiple documents", "default": False}
                    },
                    "required": ["collection", "filter", "update"]
                }
            },
            {
                "name": "mongo_delete",
                "description": "Delete document(s) from a MongoDB collection",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "collection": {"type": "string", "description": "Collection name"},
                        "filter": {"type": "object", "description": "Filter for documents to delete"},
                        "multi": {"type": "boolean", "description": "Delete multiple documents", "default": False}
                    },
                    "required": ["collection", "filter"]
                }
            },
            {
                "name": "mongo_aggregate",
                "description": "Run aggregation pipeline on a MongoDB collection",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "collection": {"type": "string", "description": "Collection name"},
                        "pipeline": {"type": "array", "description": "Aggregation pipeline"},
                        "options": {"type": "object", "description": "Aggregation options"}
                    },
                    "required": ["collection", "pipeline"]
                }
            },
            {
                "name": "mongo_stats",
                "description": "Get database or collection statistics",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "collection": {"type": "string", "description": "Collection name (optional for db stats)"},
                        "type": {"type": "string", "enum": ["database", "collection"], "default": "database"}
                    }
                }
            }
        ]

    async def execute(self, tool_name: str, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute a MongoDB tool and return MCP-compatible content."""
        if not self.database:
            raise ValueError("MongoDB connection not established")

        try:
            if tool_name == "mongo_find":
                collection_name = arguments.get("collection")
                filter_doc = arguments.get("filter", {})
                projection = arguments.get("projection")
                limit = arguments.get("limit")
                skip = arguments.get("skip")
                sort = arguments.get("sort")

                collection = self.database[collection_name]
                cursor = collection.find(filter_doc, projection)

                if sort:
                    cursor = cursor.sort(list(sort.items()))
                if skip:
                    cursor = cursor.skip(skip)
                if limit:
                    cursor = cursor.limit(limit)

                documents = await cursor.to_list(length=limit or 1000)
                serialized_docs = [self.serialize_mongo_doc(doc) for doc in documents]

                content = f"Found {len(documents)} documents in collection '{collection_name}'"
                if documents:
                    content += f":\n\n{json.dumps(serialized_docs, indent=2)}"

                return [{"type": "text", "text": content}]

            elif tool_name == "mongo_insert":
                collection_name = arguments.get("collection")
                document = arguments.get("document")
                documents = arguments.get("documents")

                collection = self.database[collection_name]

                if documents:
                    result = await collection.insert_many(documents)
                    content = f"Inserted {len(result.inserted_ids)} documents into '{collection_name}'"
                    content += f"\nInserted IDs: {[str(id) for id in result.inserted_ids]}"
                elif document:
                    result = await collection.insert_one(document)
                    content = f"Inserted document into '{collection_name}'"
                    content += f"\nInserted ID: {str(result.inserted_id)}"
                else:
                    raise ValueError("Either 'document' or 'documents' must be provided")

                return [{"type": "text", "text": content}]

            elif tool_name == "mongo_update":
                collection_name = arguments.get("collection")
                filter_doc = arguments.get("filter")
                update_doc = arguments.get("update")
                upsert = arguments.get("upsert", False)
                multi = arguments.get("multi", False)

                collection = self.database[collection_name]

                if multi:
                    result = await collection.update_many(filter_doc, update_doc, upsert=upsert)
                    content = f"Updated {result.modified_count} documents in '{collection_name}'"
                    if result.upserted_id:
                        content += f"\nUpserted ID: {str(result.upserted_id)}"
                else:
                    result = await collection.update_one(filter_doc, update_doc, upsert=upsert)
                    content = f"Updated {result.modified_count} document in '{collection_name}'"
                    if result.upserted_id:
                        content += f"\nUpserted ID: {str(result.upserted_id)}"

                return [{"type": "text", "text": content}]

            elif tool_name == "mongo_delete":
                collection_name = arguments.get("collection")
                filter_doc = arguments.get("filter")
                multi = arguments.get("multi", False)

                collection = self.database[collection_name]

                if multi:
                    result = await collection.delete_many(filter_doc)
                    content = f"Deleted {result.deleted_count} documents from '{collection_name}'"
                else:
                    result = await collection.delete_one(filter_doc)
                    content = f"Deleted {result.deleted_count} document from '{collection_name}'"

                return [{"type": "text", "text": content}]

            elif tool_name == "mongo_aggregate":
                collection_name = arguments.get("collection")
                pipeline = arguments.get("pipeline")
                options = arguments.get("options", {})

                collection = self.database[collection_name]
                cursor = collection.aggregate(pipeline, **options)
                results = await cursor.to_list(length=None)
                serialized_results = [self.serialize_mongo_doc(doc) for doc in results]

                content = f"Aggregation completed on '{collection_name}'"
                content += f"\nResults ({len(results)} documents):\n\n{json.dumps(serialized_results, indent=2)}"

                return [{"type": "text", "text": content}]

            elif tool_name == "mongo_stats":
                stats_type = arguments.get("type", "database")
                collection_name = arguments.get("collection")

                if stats_type == "collection" and collection_name:
                    stats = await self.database.command("collStats", collection_name)
                    content = f"Collection '{collection_name}' statistics:\n\n{json.dumps(self.serialize_mongo_doc(stats), indent=2)}"
                else:
                    stats = await self.database.command("dbStats")
                    content = f"Database '{self.database.name}' statistics:\n\n{json.dumps(self.serialize_mongo_doc(stats), indent=2)}"

                return [{"type": "text", "text": content}]

            else:
                raise ValueError(f"Unknown tool: {tool_name}")

        except Exception as e:
            logging.error(f"MongoDB operation error: {e}")
            raise ValueError(f"MongoDB operation failed: {str(e)}")


class MongoMCPServer(BaseMCPServer):
    """MCP Server for MongoDB operations."""

    def __init__(self):
        super().__init__("MongoDB", "1.0.0")

        # Load configuration
        self.config = MongoConfig(
            host=os.getenv("MONGO_MCP_HOST", "0.0.0.0"),
            port=int(os.getenv("MONGO_MCP_PORT", "8004")),
            log_level=os.getenv("MONGO_LOG_LEVEL", "INFO"),
            log_file=os.getenv("MONGO_LOG_FILE", "mongo_mcp.log"),
            mongo_uri=os.getenv("MONGO_URI"),
            mongo_host=os.getenv("MONGO_HOST", "localhost"),
            mongo_port=int(os.getenv("MONGO_PORT", "27017")),
            mongo_user=os.getenv("MONGO_USER"),
            mongo_password=os.getenv("MONGO_PASSWORD"),
            mongo_database=os.getenv("MONGO_DATABASE", "test")
        )

        self.mongo_ops = MongoOperations(self.config)

    async def on_startup(self):
        """Initialize MongoDB connection on startup."""
        await super().on_startup()

        # Connect to MongoDB
        if await self.mongo_ops.connect():
            self.logger.info("MongoDB MCP server started successfully")
        else:
            self.logger.error("Failed to connect to MongoDB")

    async def on_shutdown(self):
        """Clean up MongoDB connection on shutdown."""
        await self.mongo_ops.disconnect()
        await super().on_shutdown()

    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status including MongoDB connection info."""
        base_status = await super().get_health_status()
        base_status.update({
            "mongodb_connected": self.mongo_ops.database is not None,
            "mongodb_database": self.config.mongo_database
        })
        return base_status

    async def handle_tools_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/list request."""
        tools = []
        for tool in self.mongo_ops.get_tools():
            tools.append({
                "name": tool["name"],
                "description": tool["description"],
                "inputSchema": tool["inputSchema"],
            })
        return {"tools": tools}

    async def handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if not tool_name:
            raise ValueError("Tool name is required")

        self.logger.info(f"Executing MongoDB tool: {tool_name} with args: {arguments}")

        # Execute the tool
        result = await self.mongo_ops.execute(tool_name, arguments)

        return {"content": result, "isError": False}


def main():
    """Main entry point for the MongoDB MCP server."""
    server = MongoMCPServer()
    server.run(host=server.config.host, port=server.config.port)


if __name__ == "__main__":
    main()