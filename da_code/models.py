"""Pydantic models and session tracking for da_code CLI tool."""
from pathlib import Path
env_file = Path('.env')
if env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(env_file, override=False)

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import aiohttp
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field, ConfigDict

logger = logging.getLogger(__name__)


# Removed CommandConfirmationNeeded - using pure generator pattern now


# Agent framework removed - da_code now uses LangGraph exclusively


class CommandStatus(str, Enum):
    """Status of command execution."""
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXECUTING = "executing"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


class UserResponse(str, Enum):
    """User response to command confirmation."""
    YES = "yes"
    NO = "no"
    MODIFY = "modify"
    EXPLAIN = "explain"


class LLMCallStatus(str, Enum):
    """Status of LLM call."""
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


class ToolCallStatus(str, Enum):
    """Status of tool/MCP call."""
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


class StatusType(str, Enum):
    """Status message types for live interface."""
    INFO = "info"
    WORKING = "working"
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"


class CommandExecution(BaseModel):
    """Model for individual command execution tracking."""
    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid',
        str_strip_whitespace=True,
        populate_by_name=True
    )

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Command details
    command: str = Field(..., description="Shell command to execute")
    explanation: Optional[str] = Field(None, description="Agent explanation of what command does")
    working_directory: str = Field("/tmp", description="Working directory for command execution")

    # User interaction
    user_prompt: Optional[str] = Field(None, description="Prompt shown to user for confirmation")
    user_response: Optional[UserResponse] = Field(None, description="User's response to confirmation")
    user_modifications: Optional[str] = Field(None, description="User modifications to command")

    # Execution tracking
    status: CommandStatus = Field(CommandStatus.PENDING, description="Current status of command")
    exit_code: Optional[int] = Field(None, description="Command exit code")
    stdout: Optional[str] = Field(None, description="Command standard output")
    stderr: Optional[str] = Field(None, description="Command standard error")
    execution_time: Optional[float] = Field(None, description="Execution time in seconds")
    timeout_seconds: int = Field(300, description="Command timeout in seconds")

    # Agent context
    agent_reasoning: Optional[str] = Field(None, description="Agent's reasoning for this command")
    related_files: List[str] = Field(default_factory=list, description="Files related to this command")
    explanation_requested: bool = Field(False, description="Whether user requested explanation for this command")

    def update_status(self, status: CommandStatus) -> None:
        """Update command status and timestamp."""
        self.status = status
        self.updated_at = datetime.now(timezone.utc)

    def set_result(self, exit_code: int, stdout: str, stderr: str, execution_time: float) -> None:
        """Set command execution result."""
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.execution_time = execution_time
        self.status = CommandStatus.SUCCESS if exit_code == 0 else CommandStatus.FAILED
        self.updated_at = datetime.now(timezone.utc)


class MCPServerInfo(BaseModel):
    """Information about an MCP server."""
    model_config = ConfigDict(
        validate_assignment=True,
        extra='allow',
        str_strip_whitespace=True
    )

    name: str = Field(..., description="MCP server name")
    url: str = Field(..., description="MCP server URL")
    port: Optional[int] = Field(None, description="MCP server port")
    description: Optional[str] = Field(None, description="Server description")
    status: str = Field("unknown", description="Server status")
    tools: List[str] = Field(default_factory=list, description="Available tools")


class LLMCall(BaseModel):
    """Model for tracking LLM API calls."""
    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid',
        str_strip_whitespace=True
    )

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # LLM call details
    model_name: str = Field(..., description="Model name (e.g., gpt-4)")
    provider: str = Field("azure_openai", description="LLM provider")
    prompt: str = Field(..., description="Input prompt sent to LLM")
    response: Optional[str] = Field(None, description="LLM response content")

    # Execution details
    status: LLMCallStatus = Field(LLMCallStatus.PENDING)
    response_time_ms: Optional[float] = Field(None, description="Response time in milliseconds")
    error_message: Optional[str] = Field(None, description="Error message if failed")

    # Usage tracking
    prompt_tokens: Optional[int] = Field(None, description="Input tokens used")
    completion_tokens: Optional[int] = Field(None, description="Output tokens generated")
    total_tokens: Optional[int] = Field(None, description="Total tokens used")
    estimated_cost: Optional[float] = Field(None, description="Estimated cost in USD")


class ToolCall(BaseModel):
    """Model for tracking tool/MCP calls."""
    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid',
        str_strip_whitespace=True
    )

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Tool call details
    server_name: str = Field(..., description="MCP server name")
    tool_name: str = Field(..., description="Tool name called")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
    result: Optional[Dict[str, Any]] = Field(None, description="Tool execution result")

    # Execution details
    status: ToolCallStatus = Field(ToolCallStatus.PENDING)
    response_time_ms: Optional[float] = Field(None, description="Response time in milliseconds")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class ProjectContext(BaseModel):
    """Project context loaded from DA.md."""
    model_config = ConfigDict(
        validate_assignment=True,
        extra='allow',
        str_strip_whitespace=True
    )

    project_name: Optional[str] = Field(None, description="Project name")
    description: Optional[str] = Field(None, description="Project description")
    instructions: Optional[str] = Field(None, description="Project instructions")
    file_content: str = Field(..., description="Full DA.md content")
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CodeSession(BaseModel):
    """Main session model containing all command executions and context."""
    model_config = ConfigDict(
        validate_assignment=True,
        extra='allow',
        populate_by_name=True
    )

    # Session identification
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Session context
    working_directory: str = Field(..., description="Base working directory for session")
    project_context: Optional[ProjectContext] = Field(None, description="Loaded project context")
    mcp_servers: List[MCPServerInfo] = Field(default_factory=list, description="Available MCP servers")

    # Execution tracking
    executions: List[CommandExecution] = Field(default_factory=list, description="All command executions")
    llm_calls: List[LLMCall] = Field(default_factory=list, description="All LLM API calls")
    tool_calls: List[ToolCall] = Field(default_factory=list, description="All tool/MCP calls")

    # Statistics
    total_commands: int = Field(0, description="Total number of commands executed")
    successful_commands: int = Field(0, description="Number of successful commands")
    failed_commands: int = Field(0, description="Number of failed commands")
    total_llm_calls: int = Field(0, description="Total number of LLM calls")
    total_tool_calls: int = Field(0, description="Total number of tool calls")
    total_tokens: int = Field(0, description="Total tokens used across all LLM calls")
    estimated_cost: float = Field(0.0, description="Total estimated cost in USD")

    # Agent configuration
    agent_model: str = Field("gpt-4", description="Azure OpenAI model being used")
    agent_temperature: float = Field(0.7, description="Agent temperature setting")

    def add_execution(self, execution: CommandExecution) -> None:
        """Add a command execution to the session."""
        self.executions.append(execution)
        self.updated_at = datetime.now(timezone.utc)

        if execution.status == CommandStatus.SUCCESS:
            self.successful_commands += 1
            self.total_commands += 1
        elif execution.status == CommandStatus.FAILED:
            self.failed_commands += 1
            self.total_commands += 1

    def get_recent_executions(self, count: int = 10) -> List[CommandExecution]:
        """Get the most recent command executions."""
        return self.executions[-count:] if self.executions else []

    def add_llm_call(self, llm_call: LLMCall) -> None:
        """Add an LLM call to the session."""
        self.llm_calls.append(llm_call)
        self.updated_at = datetime.now(timezone.utc)
        self.total_llm_calls += 1

        if llm_call.total_tokens:
            self.total_tokens += llm_call.total_tokens
        if llm_call.estimated_cost:
            self.estimated_cost += llm_call.estimated_cost

    def add_tool_call(self, tool_call: ToolCall) -> None:
        """Add a tool call to the session."""
        self.tool_calls.append(tool_call)
        self.updated_at = datetime.now(timezone.utc)
        self.total_tool_calls += 1

    def get_session_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of the session."""
        duration = (self.updated_at - self.created_at).total_seconds()

        return {
            "session_id": self.session_id,
            "duration_seconds": duration,
            "total_commands": self.total_commands,
            "successful_commands": self.successful_commands,
            "failed_commands": self.failed_commands,
            "total_llm_calls": self.total_llm_calls,
            "total_tool_calls": self.total_tool_calls,
            "total_tokens": self.total_tokens,
            "estimated_cost": self.estimated_cost,
            "working_directory": self.working_directory,
            "agent_model": self.agent_model,
            "mcp_servers_count": len(self.mcp_servers),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


class AgentConfig(BaseModel):
    """Configuration for multi-framework agents."""
    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid',
        str_strip_whitespace=True
    )

    # Azure OpenAI configuration
    azure_endpoint: str = Field(..., description="Azure OpenAI endpoint")
    api_key: str = Field(..., description="Azure OpenAI API key")
    api_version: str = Field("2023-12-01-preview", description="Azure OpenAI API version")
    deployment_name: str = Field("gpt-4", description="Azure OpenAI deployment name")

    # Agent behavior
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Model temperature")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens per response")
    agent_timeout: Optional[int] = Field(60, description="Request timeout in seconds")
    max_retries: int = Field(2, description="Maximum number of retries")

    # Tool configuration
    command_timeout: int = Field(300, description="Default command timeout in seconds")
    require_confirmation: bool = Field(True, description="Require user confirmation for commands")

    # Framework configuration (LangGraph only)
    # Note: da_code now uses LangGraph exclusively for simplicity and reliability
    # CLI configuration
    history_file_path: str = Field(..., description="Path to command history file")


class DaMongoTracker:
    """Async MongoDB tracker using Motor."""

    def __init__(self):
        self.mongo_enabled = False
        self.client: Optional[AsyncIOMotorClient] = None
        self.database = "da_code"
        self._init_mongo_client()

    def _init_mongo_client(self) -> None:
        """Initialize MongoDB client."""
        try:
            mongo_host = os.getenv('MONGO_HOST', 'localhost')
            mongo_port = int(os.getenv('MONGO_PORT', '8004'))  # MongoDB port for da_code telemetry
            mongo_uri = f"mongodb://{mongo_host}:{mongo_port}"

            self.client = AsyncIOMotorClient(mongo_uri, serverSelectionTimeoutMS=3000)
            self.mongo_enabled = True
            logger.debug(f"MongoDB client initialized: {mongo_uri}")
        except Exception as e:
            logger.debug(f"MongoDB not available: {e}")
            self.mongo_enabled = False

    async def _save_to_mongo(self, collection: str, document: Dict[str, Any]) -> bool:
        """Save document directly to MongoDB."""
        if not self.mongo_enabled or not self.client:
            return False

        try:
            db = self.client[self.database]
            coll = db[collection]
            await coll.insert_one(document)
            return True
        except Exception:
            self.mongo_enabled = False
            return False

    def _save_to_file(self, filename: str, data: Dict[str, Any]) -> None:
        """Fallback: save to local file."""
        try:
            Path("da_sessions").mkdir(exist_ok=True)
            with open(f"da_sessions/{filename}", 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception:
            pass

    async def save_session(self, session: CodeSession) -> None:
        """Save session to MongoDB or file."""
        session_dict = session.dict()

        success = await self._save_to_mongo("sessions", session_dict)
        if not success:
            self._save_to_file(f"{session.session_id}.json", session_dict)

    async def save_llm_call(self, session_id: str, llm_call: LLMCall) -> None:
        """Save LLM call to MongoDB or file."""
        call_dict = llm_call.dict()
        call_dict["session_id"] = session_id

        success = await self._save_to_mongo("llm_calls", call_dict)
        if not success:
            self._save_to_file(f"llm_{llm_call.id}.json", call_dict)

    async def save_tool_call(self, session_id: str, tool_call: ToolCall) -> None:
        """Save tool call to MongoDB or file."""
        call_dict = tool_call.dict()
        call_dict["session_id"] = session_id

        success = await self._save_to_mongo("tool_calls", call_dict)
        if not success:
            self._save_to_file(f"tool_{tool_call.id}.json", call_dict)

    async def close(self) -> None:
        """Close MongoDB connection."""
        if self.client:
            self.client.close()


class StatusMessage(BaseModel):
    """Status message for live interface display."""
    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid',
        str_strip_whitespace=True
    )

    message: str = Field(..., description="Status message text")
    status_type: StatusType = Field(StatusType.INFO, description="Type of status message")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    details: Optional[str] = Field(None, description="Additional details")
    session_id: Optional[str] = Field(None, description="Associated session ID")


class InterfaceState(BaseModel):
    """State tracking for live interface."""
    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid'
    )

    is_executing: bool = Field(False, description="Whether agent is currently executing")
    execution_start_time: Optional[float] = Field(None, description="Execution start timestamp")
    current_status: str = Field("Ready", description="Current status description")
    timeout_seconds: int = Field(300, description="Execution timeout in seconds")

    # Events for async coordination
    interrupt_requested: bool = Field(False, description="Whether interrupt was requested")
    confirmation_pending: bool = Field(False, description="Whether confirmation is pending")
    confirmation_result: Optional[bool] = Field(None, description="Result of confirmation")

    def start_execution(self, description: str) -> None:
        """Start execution tracking."""
        self.is_executing = True
        self.execution_start_time = datetime.now(timezone.utc).timestamp()
        self.current_status = description
        self.interrupt_requested = False

    def stop_execution(self) -> None:
        """Stop execution tracking."""
        self.is_executing = False
        self.execution_start_time = None
        self.current_status = "Ready"

    def get_elapsed_time(self) -> float:
        """Get elapsed execution time in seconds."""
        if not self.execution_start_time:
            return 0.0
        return datetime.now(timezone.utc).timestamp() - self.execution_start_time

    def get_remaining_time(self) -> float:
        """Get remaining time before timeout."""
        elapsed = self.get_elapsed_time()
        return max(0.0, self.timeout_seconds - elapsed)


# Global session tracker
da_mongo = DaMongoTracker()


def get_mongo_status() -> bool:
    """Get current MongoDB connection status."""
    try:
        return da_mongo.mongo_enabled and da_mongo.client is not None
    except:
        return False