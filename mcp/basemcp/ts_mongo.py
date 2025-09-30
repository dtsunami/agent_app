"""MongoDB async operations for ToolSession and ToolConfig models."""

import os
import logging
from typing import List, Optional, Dict, Any
import motor

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from pymongo.errors import ConnectionFailure, OperationFailure
from bson import ObjectId
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv("../../.env")
from tools import ToolSession, ToolConfig

# Global MongoDB connection state
client: Optional[AsyncIOMotorClient] = None
tool_mongo: Optional[AsyncIOMotorDatabase] = None
sessions_collection: Optional[AsyncIOMotorCollection] = None
configs_collection: Optional[AsyncIOMotorCollection] = None
logger = logging.getLogger("toolsession.mongo")


# database info with defaults
MONGO_PORT = int(os.environ.get("MONGO_PORT", "27017"))
MONGO_DBNAME = os.environ.get("MONGO_DBNAME", "orenco_toolsessions")
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://lostboy:OrencoMongo2024*@localhost:27017")

# Initialize connection only if environment variables are properly set
try:
    kwargs = {'tls': True, 'tlsAllowInvalidCertificates': True} if MONGO_URI.startswith('mongodb+srv') else {}
    client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI, **kwargs)
    tool_mongo = client[MONGO_DBNAME]
    sessions_collection = tool_mongo.tool_sessions
    configs_collection = tool_mongo.tool_configs
    logger.info(f"MongoDB connection initialized for {MONGO_DBNAME}")
except Exception as e:
    logger.warning(f"MongoDB connection failed, running without persistence: {e}")
    client = None
    tool_mongo = None
    sessions_collection = None
    configs_collection = None

async def mongo_connect() -> bool:
    """Ping to MongoDB """

    try:
        
        # Test the connection
        await tool_mongo.command("ping")
        logger.info(f"Successfully connected to MongoDB: {MONGO_DBNAME}")

        # Skip index creation for now to avoid compatibility issues
        logger.info("Skipping index creation - MongoDB connection established successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        return False


async def create_indexes():
    """Create database indexes for better performance."""
    try:
        if sessions_collection is None or configs_collection is None:
            logger.warning("Collections not initialized, skipping index creation")
            return

        # Indexes for tool_sessions collection
        try:
            await sessions_collection.create_index("session_id", unique=True)
        except Exception:
            pass  # Index might already exist

        try:
            await sessions_collection.create_index([
                ("tool_config.name", 1),
                ("status", 1)
            ])
        except Exception:
            pass  # Index might already exist

        try:
            await sessions_collection.create_index([
                ("created_at", -1),
                ("updated_at", -1)
            ])
        except Exception:
            pass  # Index might already exist

        # Indexes for tool_configs collection
        try:
            await configs_collection.create_index("name", unique=True)
        except Exception:
            pass  # Index might already exist

        try:
            await configs_collection.create_index("created_at", -1)
        except Exception:
            pass  # Index might already exist

        logger.info("Database indexes created successfully")

    except Exception as e:
        logger.warning(f"Failed to create indexes: {e}")


async def disconnect():
    """Disconnect from MongoDB."""
    global client
    if client:
        client.close()
        logger.info("Disconnected from MongoDB")


# ToolSession operations
async def create_session(session: ToolSession) -> bool:
    """Create a new session in the database."""
    try:
        if sessions_collection is None:
            raise RuntimeError("Not connected to database")

        session_data = session.model_dump(by_alias=True)
        result = await sessions_collection.insert_one(session_data)

        if result.inserted_id:
            logger.debug(f"Session created: {session.session_id}")
            return True
        else:
            logger.warning(f"Failed to create session: {session.session_id}")
            return False

    except Exception as e:
        logger.error(f"Failed to create session {session.session_id}: {e}")
        return False


async def read_session(session_id: str) -> Optional[ToolSession]:
    """Read a session from the database by session_id."""
    try:
        if sessions_collection is None:
            raise RuntimeError("Not connected to database")

        session_data = await sessions_collection.find_one({"session_id": session_id})

        if session_data:
            return ToolSession(**session_data)
        else:
            logger.debug(f"Session not found: {session_id}")
            return None

    except Exception as e:
        logger.error(f"Failed to read session {session_id}: {e}")
        return None


async def read_session_by_id(object_id: str) -> Optional[ToolSession]:
    """Read a session from the database by ObjectId."""
    try:
        if sessions_collection is None:
            raise RuntimeError("Not connected to database")

        session_data = await sessions_collection.find_one({"_id": ObjectId(object_id)})

        if session_data:
            return ToolSession(**session_data)
        else:
            logger.debug(f"Session not found by ID: {object_id}")
            return None

    except Exception as e:
        logger.error(f"Failed to read session by ID {object_id}: {e}")
        return None


async def update_session(session: ToolSession) -> bool:
    """Update an existing session in the database."""
    try:
        if sessions_collection is None:
            raise RuntimeError("Not connected to database")

        session_data = session.model_dump(by_alias=True)
        result = await sessions_collection.replace_one(
            {"session_id": session.session_id},
            session_data
        )

        if result.modified_count > 0:
            logger.debug(f"Session updated: {session.session_id}")
            return True
        elif result.matched_count > 0:
            logger.debug(f"Session unchanged: {session.session_id}")
            return True
        else:
            logger.warning(f"Session not found for update: {session.session_id}")
            return False

    except Exception as e:
        logger.error(f"Failed to update session {session.session_id}: {e}")
        return False


async def upsert_session(session: ToolSession) -> bool:
    """Create or update a session (upsert operation)."""
    try:
        if sessions_collection is None:
            logger.debug("MongoDB not available, skipping session upsert")
            return False

        session_data = session.model_dump(by_alias=True)
        result = await sessions_collection.replace_one(
            {"session_id": session.session_id},
            session_data,
            upsert=True
        )

        if result.upserted_id or result.modified_count > 0:
            logger.debug(f"Session upserted: {session.session_id}")
            return True
        else:
            logger.warning(f"Failed to upsert session: {session.session_id}")
            return False

    except Exception as e:
        logger.error(f"Failed to upsert session {session.session_id}: {e}")
        return False


# ToolConfig operations
async def create_config(config: ToolConfig) -> bool:
    """Create a new tool config in the database."""
    try:
        if configs_collection is None:
            raise RuntimeError("Not connected to database")

        config_data = config.model_dump(by_alias=True)
        result = await configs_collection.insert_one(config_data)

        if result.inserted_id:
            logger.debug(f"Config created: {config.name}")
            return True
        else:
            logger.warning(f"Failed to create config: {config.name}")
            return False

    except Exception as e:
        logger.error(f"Failed to create config {config.name}: {e}")
        return False


async def read_config(name: str) -> Optional[ToolConfig]:
    """Read a tool config from the database by name."""
    try:
        if configs_collection is None:
            raise RuntimeError("Not connected to database")

        config_data = await configs_collection.find_one({"name": name})

        if config_data:
            return ToolConfig(**config_data)
        else:
            logger.debug(f"Config not found: {name}")
            return None

    except Exception as e:
        logger.error(f"Failed to read config {name}: {e}")
        return None


async def read_config_by_id(object_id: str) -> Optional[ToolConfig]:
    """Read a tool config from the database by ObjectId."""
    try:
        if configs_collection is None:
            raise RuntimeError("Not connected to database")

        config_data = await configs_collection.find_one({"_id": ObjectId(object_id)})

        if config_data:
            return ToolConfig(**config_data)
        else:
            logger.debug(f"Config not found by ID: {object_id}")
            return None

    except Exception as e:
        logger.error(f"Failed to read config by ID {object_id}: {e}")
        return None


async def update_config(config: ToolConfig) -> bool:
    """Update an existing tool config in the database."""
    try:
        if configs_collection is None:
            raise RuntimeError("Not connected to database")

        config_data = config.model_dump(by_alias=True)
        result = await configs_collection.replace_one(
            {"name": config.name},
            config_data
        )

        if result.modified_count > 0:
            logger.debug(f"Config updated: {config.name}")
            return True
        elif result.matched_count > 0:
            logger.debug(f"Config unchanged: {config.name}")
            return True
        else:
            logger.warning(f"Config not found for update: {config.name}")
            return False

    except Exception as e:
        logger.error(f"Failed to update config {config.name}: {e}")
        return False


async def upsert_config(config: ToolConfig) -> bool:
    """Create or update a tool config (upsert operation)."""
    try:
        if configs_collection is None:
            raise RuntimeError("Not connected to database")

        config_data = config.model_dump(by_alias=True)
        result = await configs_collection.replace_one(
            {"name": config.name},
            config_data,
            upsert=True
        )

        if result.upserted_id or result.modified_count > 0:
            logger.debug(f"Config upserted: {config.name}")
            return True
        else:
            logger.warning(f"Failed to upsert config: {config.name}")
            return False

    except Exception as e:
        logger.error(f"Failed to upsert config {config.name}: {e}")
        return False


# Query operations
async def list_sessions(
    tool_name: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    skip: int = 0
) -> List[Dict[str, Any]]:
    """List sessions with optional filtering."""
    try:
        if sessions_collection is None:
            raise RuntimeError("Not connected to database")

        # Build query filter
        query = {}
        if tool_name:
            query["tool_config.name"] = tool_name
        if status:
            query["status"] = status

        # Execute query
        cursor = sessions_collection.find(query) \
            .sort("updated_at", -1) \
            .skip(skip) \
            .limit(limit)

        sessions = []
        async for session_data in cursor:
            sessions.append({
                "id": str(session_data["_id"]),
                "session_id": session_data["session_id"],
                "tool_name": session_data["tool_config"]["name"],
                "status": session_data["status"],
                "created_at": session_data["created_at"],
                "updated_at": session_data["updated_at"],
                "total_commands": session_data.get("total_commands", 0),
                "total_scripts": session_data.get("total_scripts", 0),
                "total_errors": session_data.get("total_errors", 0)
            })

        return sessions

    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        return []


async def list_configs(limit: int = 50, skip: int = 0) -> List[Dict[str, Any]]:
    """List all tool configs."""
    try:
        if configs_collection is None:
            raise RuntimeError("Not connected to database")

        cursor = configs_collection.find({}) \
            .sort("created_at", -1) \
            .skip(skip) \
            .limit(limit)

        configs = []
        async for config_data in cursor:
            configs.append({
                "id": str(config_data["_id"]),
                "name": config_data["name"],
                "launch_command": config_data["launch_command"],
                "working_directory": config_data["working_directory"],
                "created_at": config_data["created_at"]
            })

        return configs

    except Exception as e:
        logger.error(f"Failed to list configs: {e}")
        return []


async def delete_session(session_id: str) -> bool:
    """Delete a session from the database."""
    try:
        if sessions_collection is None:
            raise RuntimeError("Not connected to database")

        result = await sessions_collection.delete_one({"session_id": session_id})

        if result.deleted_count > 0:
            logger.info(f"Session deleted: {session_id}")
            return True
        else:
            logger.warning(f"Session not found for deletion: {session_id}")
            return False

    except Exception as e:
        logger.error(f"Failed to delete session {session_id}: {e}")
        return False


async def delete_config(name: str) -> bool:
    """Delete a tool config from the database."""
    try:
        if configs_collection is None:
            raise RuntimeError("Not connected to database")

        result = await configs_collection.delete_one({"name": name})

        if result.deleted_count > 0:
            logger.info(f"Config deleted: {name}")
            return True
        else:
            logger.warning(f"Config not found for deletion: {name}")
            return False

    except Exception as e:
        logger.error(f"Failed to delete config {name}: {e}")
        return False


async def get_database_stats() -> Dict[str, Any]:
    """Get statistics about the database."""
    try:
        if not tool_mongo:
            raise RuntimeError("Not connected to database")

        # Collection stats
        sessions_count = await sessions_collection.count_documents({})
        configs_count = await configs_collection.count_documents({})

        # Session status breakdown
        status_pipeline = [
            {"$group": {"_id": "$status", "count": {"$sum": 1}}}
        ]
        status_counts = {}
        async for result in sessions_collection.aggregate(status_pipeline):
            status_counts[result["_id"]] = result["count"]


        return {
            "database_name": MONGO_DBNAME,
            "total_sessions": sessions_count,
            "total_configs": configs_count,
            "session_status_breakdown": status_counts
        }

    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        return {}