"""Simple agent factory - LangGraph only for reliability and simplicity."""

import logging

from .agent_interface import AgentInterface
from .langgraph_agent_correct import CorrectLangGraphAgent
from .models import AgentConfig, CodeSession

logger = logging.getLogger(__name__)


class AgentFactory:
    """Simple factory for creating LangGraph agents."""

    @staticmethod
    def create_agent(
        config: AgentConfig,
        session: CodeSession
    ) -> AgentInterface:
        """Create a LangGraph agent instance."""

        logger.debug("Creating LangGraph agent")

        try:
            return CorrectLangGraphAgent(config, session)
        except Exception as e:
            logger.error(f"Failed to create LangGraph agent: {e}")
            raise

    @staticmethod
    def get_framework_name() -> str:
        """Get the framework name."""
        return "langgraph"

    @staticmethod
    def is_available() -> bool:
        """Check if LangGraph framework is available."""
        try:
            # Simple check - try to import the main class
            from .langgraph_agent_correct import CorrectLangGraphAgent
            return True
        except ImportError:
            return False