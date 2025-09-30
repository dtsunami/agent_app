"""Unified execution event system for multi-framework agent streaming."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional, Union, AsyncGenerator
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from .models import CommandExecution


class EventType(str, Enum):
    """Types of execution events."""
    EXECUTION_START = "execution_start"
    EXECUTION_END = "execution_end"
    LLM_START = "llm_start"
    LLM_END = "llm_end"
    LLM_STREAM = "llm_stream"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    COMMAND_CONFIRMATION_NEEDED = "command_confirmation_needed"
    COMMAND_EXECUTED = "command_executed"
    AGENT_THOUGHT = "agent_thought"
    ERROR = "error"
    FINAL_RESPONSE = "final_response"


class ExecutionEvent(BaseModel):
    """Unified execution event that any framework can emit."""

    event_type: EventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    framework: str = Field(..., description="Framework that generated this event")

    # Event data - different types use different fields
    content: Optional[str] = None
    execution: Optional[CommandExecution] = None
    tool_name: Optional[str] = None
    tokens_used: Optional[int] = None
    execution_time_ms: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class ConfirmationRequest(BaseModel):
    """Request for user confirmation during execution."""

    execution: CommandExecution
    choices: list[str] = Field(default=["yes", "no", "modify", "explain"])
    default_choice: str = "no"


class ConfirmationResponse(BaseModel):
    """User response to confirmation request."""

    choice: str
    modified_command: Optional[str] = None


class StreamingExecutor(ABC):
    """Abstract base for streaming execution across frameworks."""

    @abstractmethod
    async def execute_with_stream(self, task: str) -> AsyncGenerator[ExecutionEvent, ConfirmationResponse]:
        """
        Execute task and yield events. Can receive confirmation responses.

        Usage:
            executor = SomeStreamingExecutor()
            confirmation_response = None

            async for event in executor.execute_with_stream(task):
                if event.event_type == EventType.COMMAND_CONFIRMATION_NEEDED:
                    # Get user input
                    user_choice = await prompt_user(event.execution)
                    confirmation_response = ConfirmationResponse(choice=user_choice)

                    # Send response back to generator
                    try:
                        event = await executor.asend(confirmation_response)
                    except StopAsyncIteration:
                        break

                elif event.event_type == EventType.FINAL_RESPONSE:
                    return event.content
        """
        pass

    @abstractmethod
    def get_framework_name(self) -> str:
        """Get the framework identifier."""
        pass


# Legacy event conversion removed - da_code now uses LangGraph exclusively
# Event streaming is handled directly in the LangGraph agent implementation