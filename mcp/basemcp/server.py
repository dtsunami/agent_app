"""Common MCP server foundation for building FastAPI-based MCP servers."""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import uvicorn
from bson import ObjectId
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, ConfigDict, Field, field_serializer


# MCP Protocol Models
@dataclass
class MCPRequest:
    jsonrpc: str
    id: Union[str, int, None]
    method: str
    params: Optional[Dict[str, Any]] = None


@dataclass
class MCPResponse:
    jsonrpc: str
    id: Union[str, int, None]
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None


@dataclass
class MCPError:
    code: int
    message: str
    data: Optional[Dict[str, Any]] = None


# MongoDB Models for logging
class PyObjectId(ObjectId):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        from pydantic_core import core_schema
        return core_schema.no_info_plain_validator_function(cls.validate)

    @classmethod
    def validate(cls, v):
        if isinstance(v, ObjectId):
            return v
        if isinstance(v, str) and ObjectId.is_valid(v):
            return ObjectId(v)
        raise ValueError("Invalid ObjectId")


class MCPExecution(BaseModel):
    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    method: str
    params: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    execution_time: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    client_info: Optional[Dict[str, Any]] = None

    @field_serializer("id")
    def serialize_object_id(self, value: Optional[PyObjectId]) -> Optional[str]:
        """Serialize ObjectId to string for JSON output."""
        if value is None:
            return None
        return str(value)


class BaseMCPServer:
    """Base class for FastAPI-based MCP servers."""

    def __init__(self, server_name: str, server_version: str = "1.0.0"):
        self.server_name = server_name
        self.server_version = server_version
        self.logger = None
        self.mongodb_client = None
        self.database = None
        self.initialized = False
        self.client_capabilities = {}

        # MCP Protocol version
        self.protocol_version = "2024-11-05"

        # Setup FastAPI app
        self.app = FastAPI(
            title=f"{server_name} MCP Server",
            description=f"MCP JSON-RPC server for {server_name.lower()} operations",
            version=server_version,
        )

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Setup routes
        self.setup_routes()

        # Setup logging
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration."""
        log_file = os.getenv("LOG_FILE", f"/tmp/{self.server_name.lower()}_mcp.log")
        log_level = os.getenv("LOG_LEVEL", "INFO")

        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=getattr(logging, log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(f"{self.server_name.lower()}-mcp")

    def setup_routes(self):
        """Setup FastAPI routes."""

        @self.app.on_event("startup")
        async def startup_event():
            """Initialize connections and services on startup."""
            await self.on_startup()

        @self.app.on_event("shutdown")
        async def shutdown_event():
            """Clean up connections and services on shutdown."""
            await self.on_shutdown()

        @self.app.post("/mcp")
        async def mcp_endpoint(request: Request):
            """Main MCP JSON-RPC endpoint."""
            try:
                body = await request.json()
                response = await self.handle_mcp_request(body)
                return JSONResponse(content=response)
            except json.JSONDecodeError:
                return JSONResponse(content=self.create_error_response(None, -32700, "Parse error"))
            except Exception as e:
                self.logger.error(f"Unexpected error: {e}", exc_info=True)
                return JSONResponse(content=self.create_error_response(None, -32603, "Internal error"))

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return await self.get_health_status()

    async def on_startup(self):
        """Override this method for custom startup logic."""
        await self.connect_mongodb()

    async def on_shutdown(self):
        """Override this method for custom shutdown logic."""
        if self.mongodb_client:
            self.mongodb_client.close()

    async def get_health_status(self) -> Dict[str, Any]:
        """Override this method for custom health status."""
        return {
            "status": "healthy",
            "server_name": self.server_name,
            "protocol_version": self.protocol_version,
            "initialized": self.initialized,
            "mongodb_connected": self.database is not None,
        }

    async def connect_mongodb(self):
        """Connect to MongoDB."""
        try:
            mongo_url = os.getenv("MONGO_URI")
            if mongo_url is None:
                self.logger.warning("MONGO_URI environment variable not set - running without MongoDB")
                return False

            self.mongodb_client = AsyncIOMotorClient(mongo_url)
            self.database = self.mongodb_client.orenco_pydantic

            # Test the connection
            await self.database.command("ping")
            self.logger.info("Successfully connected to MongoDB")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to MongoDB: {e}")
            return False

    async def log_execution(
        self,
        method: str,
        params: Dict[str, Any],
        result: Dict[str, Any] = None,
        error: Dict[str, Any] = None,
        execution_time: float = 0.0,
        client_info: Dict[str, Any] = None,
    ):
        """Log MCP execution to MongoDB."""
        if self.database is None:
            return

        try:
            execution = MCPExecution(
                method=method,
                params=params,
                result=result,
                error=error,
                execution_time=execution_time,
                client_info=client_info,
            )

            await self.database.mcp_executions.insert_one(execution.dict(by_alias=True))
        except Exception as e:
            self.logger.error(f"Failed to log execution: {e}")

    def create_response(
        self, request_id: Union[str, int, None], result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a successful MCP response."""
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def create_error_response(
        self,
        request_id: Union[str, int, None],
        code: int,
        message: str,
        data: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Create an MCP error response."""
        error = {"code": code, "message": message}
        if data:
            error["data"] = data

        return {"jsonrpc": "2.0", "id": request_id, "error": error}

    async def handle_mcp_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming MCP JSON-RPC request."""
        start_time = time.time()

        # Validate JSON-RPC structure
        if not isinstance(request_data, dict):
            return self.create_error_response(None, -32600, "Invalid Request")

        jsonrpc = request_data.get("jsonrpc")
        if jsonrpc != "2.0":
            return self.create_error_response(None, -32600, "Invalid JSON-RPC version")

        request_id = request_data.get("id")
        method = request_data.get("method")
        params = request_data.get("params", {})

        if not method:
            return self.create_error_response(request_id, -32600, "Missing method")

        self.logger.info(f"MCP request: {method} with params: {params}")

        try:
            # Route to appropriate handler
            if method == "initialize":
                result = await self.handle_initialize(params)
            elif method == "tools/list":
                result = await self.handle_tools_list(params)
            elif method == "tools/call":
                result = await self.handle_tools_call(params)
            else:
                return self.create_error_response(
                    request_id, -32601, f"Method not found: {method}"
                )

            execution_time = time.time() - start_time

            # Log successful execution
            await self.log_execution(method, params, result, None, execution_time)

            return self.create_response(request_id, result)

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            self.logger.error(f"Error handling {method}: {error_msg}", exc_info=True)

            # Log failed execution
            error_data = {"code": -32603, "message": error_msg}
            await self.log_execution(method, params, None, error_data, execution_time)

            return self.create_error_response(request_id, -32603, error_msg)

    async def handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP initialize request."""
        protocol_version = params.get("protocolVersion", "2024-11-05")
        client_capabilities = params.get("capabilities", {})
        client_info = params.get("clientInfo", {})

        # Store client info
        self.client_capabilities = client_capabilities

        # Mark as initialized
        self.initialized = True

        self.logger.info(f"Initialized MCP server for client: {client_info}")

        return {
            "protocolVersion": self.protocol_version,
            "capabilities": {"tools": {"listChanged": True}, "logging": {}},
            "serverInfo": {"name": f"{self.server_name.lower()}-mcp", "version": self.server_version},
        }

    async def handle_tools_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/list request. Override this method."""
        raise NotImplementedError("Subclasses must implement handle_tools_list")

    async def handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request. Override this method."""
        raise NotImplementedError("Subclasses must implement handle_tools_call")

    def run(self, host: str = "0.0.0.0", port: int = 8002):
        """Run the MCP server."""
        self.logger.info(f"Starting {self.server_name} MCP Server v{self.server_version}")
        self.logger.info(f"Protocol version: {self.protocol_version}")

        try:
            self.logger.info(f"MCP Server running on http://{host}:{port}")
            self.logger.info(f"MCP endpoint: http://{host}:{port}/mcp")

            # Run the FastAPI server
            uvicorn.run(
                self.app,
                host=host,
                port=port,
                log_level="info",
            )
        except KeyboardInterrupt:
            self.logger.info("Server stopped by user")
        except Exception as e:
            self.logger.error(f"Server error: {e}", exc_info=True)
            raise
        finally:
            self.logger.info(f"{self.server_name} MCP Server shutdown")