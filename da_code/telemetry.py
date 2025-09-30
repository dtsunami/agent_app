"""Centralized telemetry tracking for all agent frameworks."""

import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from .models import CodeSession, LLMCall, LLMCallStatus, ToolCall, ToolCallStatus, da_mongo

logger = logging.getLogger(__name__)


class TelemetryManager:
    """Manages telemetry tracking across different agent frameworks."""

    def __init__(self, session: CodeSession):
        """Initialize telemetry manager."""
        self.session = session
        self.agent_metrics: Dict[str, Any] = {
            'calls': 0, 'tokens': 0, 'total_time_ms': 0
        }

    async def track_agent_call(
        self,
        prompt: str,
        response: str,
        tokens_used: int = 0,
        execution_time_ms: float = 0,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> str:
        """Track an agent call with unified metrics."""

        # Create LLM call record
        llm_call = LLMCall(
            model_name=self.session.agent_model,
            provider="langgraph_provider",
            prompt=prompt,
            response=response if success else None,
            status=LLMCallStatus.SUCCESS if success else LLMCallStatus.FAILED,
            response_time_ms=execution_time_ms,
            error_message=error_message,
            total_tokens=tokens_used
        )

        # Add to session
        self.session.add_llm_call(llm_call)

        # Update agent metrics
        self.agent_metrics['calls'] += 1
        self.agent_metrics['tokens'] += tokens_used
        self.agent_metrics['total_time_ms'] += execution_time_ms

        # Save to MongoDB asynchronously
        try:
            await da_mongo.save_session(self.session)
            await da_mongo.save_llm_call(self.session.session_id, llm_call)
        except Exception as e:
            logger.debug(f"Failed to save telemetry to MongoDB: {e}")

        logger.debug(f"Tracked agent call: {tokens_used} tokens, {execution_time_ms}ms")
        return llm_call.id

    async def track_tool_call(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Optional[Dict[str, Any]] = None,
        execution_time_ms: float = 0,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> str:
        """Track a tool/MCP call."""

        # Create tool call record
        tool_call = ToolCall(
            server_name=server_name,
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            status=ToolCallStatus.SUCCESS if success else ToolCallStatus.FAILED,
            response_time_ms=execution_time_ms,
            error_message=error_message
        )

        # Add to session
        self.session.add_tool_call(tool_call)

        # Save to MongoDB asynchronously
        try:
            await da_mongo.save_session(self.session)
            await da_mongo.save_tool_call(self.session.session_id, tool_call)
        except Exception as e:
            logger.debug(f"Failed to save tool call to MongoDB: {e}")

        logger.debug(f"Tracked tool call: {server_name}.{tool_name}, {execution_time_ms}ms")
        return tool_call.id

    def get_framework_metrics(self, framework: str) -> Dict[str, Any]:
        """Get metrics for the agent (framework parameter kept for compatibility)."""
        return self.agent_metrics

    def get_all_framework_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for the agent (simplified from multi-framework)."""
        return {"langgraph": self.agent_metrics}

    def get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive session summary with agent metrics."""
        summary = self.session.get_session_summary()
        summary['framework_breakdown'] = self.get_all_framework_metrics()

        # Calculate agent efficiency metrics
        if self.agent_metrics['calls'] > 0:
            avg_time = self.agent_metrics['total_time_ms'] / self.agent_metrics['calls']
            avg_tokens = self.agent_metrics['tokens'] / self.agent_metrics['calls'] if self.agent_metrics['tokens'] > 0 else 0
            summary['framework_breakdown']['langgraph'].update({
                'avg_time_ms': avg_time,
                'avg_tokens_per_call': avg_tokens
            })

        return summary

    def reset_framework_metrics(self, framework: Optional[str] = None) -> None:
        """Reset agent metrics (framework parameter kept for compatibility)."""
        self.agent_metrics = {'calls': 0, 'tokens': 0, 'total_time_ms': 0}
        logger.debug(f"Reset agent metrics")


class PerformanceTracker:
    """Context manager for tracking execution performance."""

    def __init__(self, telemetry: TelemetryManager, framework: str, operation: str):
        """Initialize performance tracker."""
        self.telemetry = telemetry
        self.framework = framework
        self.operation = operation
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log performance."""
        self.end_time = time.time()
        duration_ms = (self.end_time - self.start_time) * 1000

        success = exc_type is None
        error_message = str(exc_val) if exc_val else None

        logger.debug(f"{self.framework} {self.operation}: {duration_ms:.2f}ms, success: {success}")

    def get_duration_ms(self) -> float:
        """Get duration in milliseconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0