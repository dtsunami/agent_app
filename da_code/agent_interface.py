"""Agent interface abstraction - now LangGraph only for simplicity."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, AsyncGenerator

from .models import CodeSession, AgentConfig
from .execution_events import ExecutionEvent, ConfirmationResponse


class AgentInterface(ABC):
    """Abstract interface for all agent frameworks."""

    def __init__(self, config: AgentConfig, session: CodeSession):
        """Initialize agent with configuration and session."""
        self.config = config
        self.session = session

    @abstractmethod
    async def execute_task(self, task: str) -> str:
        """Execute a task and return the response (legacy method)."""
        pass

    @abstractmethod
    async def execute_task_stream(self, task: str) -> AsyncGenerator[ExecutionEvent, ConfirmationResponse]:
        """
        Execute a task with streaming events and confirmation support.

        This is the new preferred method that supports:
        - Real-time progress updates
        - Command confirmations
        - Framework-agnostic event streaming

        Usage:
            async for event in agent.execute_task_stream(task):
                if event.event_type == EventType.COMMAND_CONFIRMATION_NEEDED:
                    user_choice = await prompt_user()
                    response = ConfirmationResponse(choice=user_choice)
                    # Send response back to generator
                    event = await agent.asend(response)
                elif event.event_type == EventType.FINAL_RESPONSE:
                    return event.content
        """
        pass

    @abstractmethod
    def get_framework_name(self) -> str:
        """Return the framework identifier."""
        pass

    @abstractmethod
    def get_session_info(self) -> Dict[str, Any]:
        """Get current session information."""
        pass

    @abstractmethod
    def clear_memory(self) -> None:
        """Clear agent conversation memory."""
        pass

    async def get_metrics(self) -> Dict[str, Any]:
        """Get framework-specific metrics."""
        return {
            "framework": self.get_framework_name(),
            "session_id": self.session.session_id,
            "total_commands": len(self.session.executions),
            "successful_commands": self.session.successful_commands,
            "failed_commands": self.session.failed_commands,
        }