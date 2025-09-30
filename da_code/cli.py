
"""Main CLI entry point for da_code tool."""

import asyncio
import logging
import os
import sys
import time

import random
from typing import List, Optional
from pathlib import Path
import termios
import tty

from rich.console import Console
from rich.status import Status
from rich.text import Text
from rich.panel import Panel

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.formatted_text import HTML

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult

from .agent_factory import AgentFactory
from .agent_interface import AgentInterface
from .config import ConfigManager, setup_logging
from .context import ContextLoader
from .models import CodeSession
from .execution_events import EventType, ConfirmationResponse
from .models import CommandExecution, UserResponse

logger = logging.getLogger(__name__)

# Global console for clean interaction
console = Console()


async def async_prompt_user_silent(choices: List[str], default: str = None, command: str = None) -> str:
    """Interactive prompt with visual arrow indicator for selected option."""


    # Choice configuration with colors
    choice_config = {
        "yes": {"label": "✅ Yes", "desc": "Execute the command as shown", "color": "green"},
        "no": {"label": "❌ No", "desc": "Cancel command execution", "color": "red"},
        "modify": {"label": "✏️  Modify", "desc": "Edit the command before execution", "color": "yellow"},
        "explain": {"label": "❓ Explain", "desc": "Ask agent to explain the command", "color": "blue"}
    }

    def display_static_confirmation():
        """Display confirmation panel once - no updates until selection made."""
        content_lines = Text()

        # Add command info if provided
        if command:
            command_text = Text()
            command_text.append("Command: ", style="bold cyan")
            command_text.append(f"`{command}`", style="bold yellow")
            content_lines.append(command_text)
            content_lines.append("\n")

        # Add choices (no arrow initially)
        for i, choice in enumerate(choices):
            config = choice_config.get(choice.lower(), {"label": choice, "desc": "", "color": "white"})
            line = Text()
            line.append("    ", style="white")
            line.append(f"{i+1}. {config['label']}", style=config['color'])
            if config['desc']:
                line.append(f" - {config['desc']}", style="white dim")
            line.append("\n")
            content_lines.append(line) 


        # Instructions
        instructions = Text()
        instructions.append("Press 1-4", style="cyan bold")
        instructions.append(" or use ", style="white dim")
        instructions.append("↑/↓ arrows", style="cyan bold")
        instructions.append(" and ", style="white dim")
        instructions.append("Enter", style="red bold")
        instructions.append(" to select\n", style="white dim")
        content_lines.append(instructions)


        # Display the panel once
        unified_panel = Panel(
            content_lines,
            title="🤖 Confirm Agent Command",
            title_align="left",
            border_style="cyan"
        )
        console.print(unified_panel)

    def get_keypress_choice() -> str:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        selected_index = 0  # Initialize locally

        try:
            tty.setraw(sys.stdin.fileno())

            # Display confirmation panel once
            display_static_confirmation()
            
            # display the default choice
            config = choice_config.get(choices[selected_index].lower(), {"label": choices[selected_index], "desc": "", "color": "white"})
            print(f"\r\033[K▶ {selected_index + 1}. {config['label']}", end='', flush=True)
            
            while True:
                key = sys.stdin.read(1)

                # Number key shortcuts
                if key in ['1', '2', '3', '4']:
                    idx = int(key) - 1
                    if idx < len(choices):
                        return choices[idx]

                # Arrow keys - track selection and show simple feedback
                elif key == '\x1b':
                    key += sys.stdin.read(2)
                    if key == '\x1b[A':  # Up
                        selected_index = (selected_index - 1) % len(choices)
                        # Show simple selection feedback
                        config = choice_config.get(choices[selected_index].lower(), {"label": choices[selected_index], "color": "white"})
                        print(f"\r\033[K▶ {selected_index + 1}. {config['label']}", end='', flush=True)
                    elif key == '\x1b[B':  # Down
                        selected_index = (selected_index + 1) % len(choices)
                        # Show simple selection feedback
                        config = choice_config.get(choices[selected_index].lower(), {"label": choices[selected_index], "color": "white"})
                        print(f"\r\033[K▶ {selected_index + 1}. {config['label']}", end='', flush=True)

                # Enter key
                elif key in ['\r', '\n']:
                    # Clear any status line before returning
                    print('\r\033[K', end='', flush=True)
                    return choices[selected_index]

                # Ctrl+C
                elif key == '\x03':
                    raise KeyboardInterrupt()

        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    # Get choice asynchronously
    loop = asyncio.get_event_loop()
    selected_choice = await loop.run_in_executor(None, get_keypress_choice)

    # Show clean selection result in console history
    config = choice_config.get(selected_choice.lower(), {"label": selected_choice, "color": "white"})
    console.print(f"[green bold]✅ Selected: {config['label']}[/green bold]")

    return selected_choice


async def async_prompt_text(message: str, default: str = None) -> str:
    """Async wrapper for Rich text prompts."""
    import asyncio
    from rich.prompt import Prompt

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: Prompt.ask(message, default=default)
    )


def display_simple_confirmation(execution) -> None:
    """Display simplified command confirmation panel using UserResponse enum."""
    from rich.panel import Panel
    from rich.text import Text

    # Build content
    content = []

    # Command info section
    command_text = Text()
    command_text.append("Command: ", style="bold cyan")
    command_text.append(execution.command, style="bold white")
    content.append(str(command_text))

    # Add separator
    content.append("")

    # Complete choice display using all UserResponse enum values
    choices = [
        ("✅ Yes", "Execute the command as shown"),
        ("❌ No", "Cancel command execution"),
        ("✏️ Modify", "Edit the command before execution"),
        ("❓ Explain", "Ask agent to explain the command")
    ]

    for i, (label, desc) in enumerate(choices):
        choice_line = Text()
        choice_line.append(f"  {i+1}. {label}", style="bold")
        choice_line.append(f" - {desc}", style="white dim")
        content.append(str(choice_line))

    # Add instructions
    content.append("")
    instructions = Text()
    instructions.append("Press ", style="white dim")
    instructions.append("1-4", style="cyan bold")
    instructions.append(" or use ", style="white dim")
    instructions.append("↑/↓ arrows", style="cyan bold")
    instructions.append(" + ", style="white dim")
    instructions.append("Enter", style="green bold")
    content.append(str(instructions))

    # Create panel
    panel = Panel(
        "\n".join(content),
        title="🤖 Command Confirmation",
        title_align="left",
        border_style="cyan",
        padding=(1, 2)
    )

    console.print(panel)



async def execute_with_streaming_confirmations(
    agent: AgentInterface,
    user_input: str,
    status_interface
) -> Optional[str]:
    """Execute task with beautiful unified streaming and confirmation handling."""

    async def confirmation_handler(execution: CommandExecution) -> ConfirmationResponse:
        """Handle command confirmation directly."""
        # Stop status to show confirmation dialog cleanly
        status_interface.stop_execution()

        # Get user choice using enum values with command display
        response = await async_prompt_user_silent([
            UserResponse.YES.value.title(),    # "Yes"
            UserResponse.NO.value.title(),     # "No"
            UserResponse.MODIFY.value.title(), # "Modify"
            UserResponse.EXPLAIN.value.title() # "Explain"
        ], default=UserResponse.YES.value.title(), command=execution.command)

        modified_command = None
        if response.lower() == UserResponse.MODIFY.value:
            # Get modified command from user
            modified_command = await async_prompt_text("Enter modified command", default=execution.command)
            if not modified_command:
                response = UserResponse.NO.value.title()

        # Response selection is already shown in the arrow interface, no need to print again

        # Restart status interface for continued execution
        status_interface.start_execution("Processing...")

        # Convert response back to enum value for consistency
        response_lower = response.lower()
        if response_lower == "yes":
            enum_choice = UserResponse.YES.value
        elif response_lower == "no":
            enum_choice = UserResponse.NO.value
        elif response_lower == "modify":
            enum_choice = UserResponse.MODIFY.value
        elif response_lower == "explain":
            enum_choice = UserResponse.EXPLAIN.value
        else:
            enum_choice = UserResponse.NO.value  # Default fallback

        return ConfirmationResponse(
            choice=enum_choice,
            modified_command=modified_command
        )

    final_response = None
    event_generator = agent.execute_task_stream(user_input, confirmation_handler)

    try:
        async for event in event_generator:
            # Skip None events (these happen during generator pauses for user input)
            if event is None:
                continue

            # Simplified event handling - only process events that are actually emitted
            if event.event_type == EventType.EXECUTION_START:
                status_interface.update_status("Starting execution...")

            elif event.event_type == EventType.COMMAND_CONFIRMATION_NEEDED:
                # This event is handled by the confirmation_handler, just update status
                status_interface.update_status("Waiting for user confirmation...")

            elif event.event_type == EventType.FINAL_RESPONSE:
                final_response = event.content
                status_interface.update_status("Task completed")

            elif event.event_type == EventType.ERROR:
                status_interface.update_status(f"Error: {event.error_message}")
                return f"❌ Execution failed: {event.error_message}"

        return final_response

    except StopAsyncIteration:
        return final_response
    except KeyboardInterrupt:
        console.print("❌ Execution interrupted by user.")
        return "❌ Execution cancelled."


class RealTimeAgentCallbackHandler(AsyncCallbackHandler):
    """Async callback handler for real-time agent monitoring."""

    def __init__(self, status_interface):
        self.status_interface = status_interface
        self.interrupt_requested = False

    async def on_llm_start(self, serialized, prompts, **kwargs):
        """Called when LLM starts."""
        self.status_interface.llm_calls += 1
        self.status_interface.update_status("Calling Azure OpenAI...")

    async def on_llm_end(self, response: LLMResult, **kwargs):
        """Called when LLM ends - track token usage."""
        if hasattr(response, 'llm_output') and response.llm_output:
            token_usage = response.llm_output.get('token_usage', {})
            tokens = token_usage.get('total_tokens', 0)
            if tokens > 0:
                self.status_interface.total_tokens += tokens
        self.status_interface.update_status("Processing response...")

    async def on_tool_start(self, serialized, input_str, **kwargs):
        """Called when tool starts."""
        tool_name = serialized.get('name', 'Unknown')
        self.status_interface.tool_calls += 1
        self.status_interface.update_status(f"Using tool: {tool_name}")

    async def on_tool_end(self, output, **kwargs):
        """Called when tool ends."""
        self.status_interface.update_status("Tool completed")

    async def on_agent_action(self, action, **kwargs):
        """Called when agent takes an action."""
        self.status_interface.update_status(f"Agent action: {action.tool}")

    async def on_agent_finish(self, finish, **kwargs):
        """Called when agent finishes."""
        self.status_interface.update_status("Finalizing response...")

    async def on_text(self, text: str, **kwargs):
        """Called on arbitrary text - can be used for streaming."""
        # This could be used for token-by-token streaming
        pass

    def request_interrupt(self):
        """Request interruption of agent execution."""
        self.interrupt_requested = True


class SimpleStatusInterface:
    """Simple status interface with Rich spinner and agent insights."""

    def __init__(self):
        self.start_time = None
        self.current_status = None
        self.llm_calls = 0
        self.tool_calls = 0
        self.total_tokens = 0
        self.callback_handler = None
        # Agent metrics (LangGraph only)
        self.agent_metrics = {'calls': 0, 'tokens': 0}

    def start_execution(self, message: str):
        """Start execution with status message."""
        self.start_time = time.time()
        self.llm_calls = 0
        self.tool_calls = 0
        self.total_tokens = 0
        # Reset agent metrics
        self.agent_metrics = {'calls': 0, 'tokens': 0}
        self.current_status = Status(f"🤖 {message}", spinner="dots")
        self.current_status.start()
        self.callback_handler = RealTimeAgentCallbackHandler(self)

    def update_status(self, message: str):
        """Update the current status message."""
        if self.current_status:
            elapsed = time.time() - self.start_time if self.start_time else 0
            status_text = f"🤖 {message} | {elapsed:.1f}s"
            if self.llm_calls > 0:
                status_text += f" | LLM: {self.llm_calls}"
            if self.tool_calls > 0:
                status_text += f" | Tools: {self.tool_calls}"
            if self.total_tokens > 0:
                status_text += f" | Tokens: {self.total_tokens}"
            self.current_status.update(status_text)

    def log_llm_call(self, tokens_used: int = 0):
        """Log an LLM call."""
        self.llm_calls += 1
        if tokens_used > 0:
            self.total_tokens += tokens_used
        self.update_status("Processing...")

    def log_tool_call(self, tool_name: str = ""):
        """Log a tool call."""
        self.tool_calls += 1
        self.update_status(f"Using tool: {tool_name}" if tool_name else "Using tool...")

    def track_agent_call(self, tokens: int = 0):
        """Track agent calls and token usage."""
        self.agent_metrics['calls'] += 1
        self.agent_metrics['tokens'] += tokens

    def stop_execution(self, success: bool = True, final_message: str = None):
        """Stop execution and show final result."""
        if self.current_status:
            self.current_status.stop()

        elapsed = time.time() - self.start_time if self.start_time else 0

        if success:
            result_text = "✅ Complete"
        else:
            result_text = "❌ Failed"

        result_text += f" {elapsed:.1f}s"

        # Add current directory
        current_dir = os.path.basename(os.getcwd()) or "/"
        result_text += f" | 📂 {current_dir}"

        if self.llm_calls > 0:
            result_text += f" | LLM: {self.llm_calls}"
        if self.tool_calls > 0:
            result_text += f" | Tools: {self.tool_calls}"
        if self.total_tokens > 0:
            result_text += f" | Tokens: {self.total_tokens}"

        if final_message:
            result_text += f" | {final_message}"

        console.print(result_text)
        self.current_status = None
        self.callback_handler = None


"""ASCII art splash screen for da_code."""



def get_splash_screen() -> str:
    """Get the da_code ASCII splash screen."""

    # Main da_code ASCII art
    ascii_art = r"""
██████╗  █████╗      ██████╗ ██████╗ ██████╗ ███████╗
██╔══██╗██╔══██╗    ██╔════╝██╔═══██╗██╔══██╗██╔════╝
██║  ██║███████║    ██║     ██║   ██║██║  ██║█████╗
██║  ██║██╔══██║    ██║     ██║   ██║██║  ██║██╔══╝
██████╔╝██║  ██║    ╚██████╗╚██████╔╝██████╔╝███████╗
╚═════╝ ╚═╝  ╚═╝     ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝

    🤖 Agentic CLI with LangChain & Azure OpenAI 🚀
    """

    return ascii_art


def get_animated_splash() -> List[str]:
    """Get animated frames for splash screen."""

    frames = [
        # Frame 1
        r"""
██████╗  █████╗      ██████╗ ██████╗ ██████╗ ███████╗
██╔══██╗██╔══██╗    ██╔════╝██╔═══██╗██╔══██╗██╔════╝
██║  ██║███████║    ██║     ██║   ██║██║  ██║█████╗
██║  ██║██╔══██║    ██║     ██║   ██║██║  ██║██╔══╝
██████╔╝██║  ██║    ╚██████╗╚██████╔╝██████╔╝███████╗
╚═════╝ ╚═╝  ╚═╝     ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝

    🤖 Initializing AI Agent...
""",
        # Frame 2
        r"""
██████╗  █████╗      ██████╗ ██████╗ ██████╗ ███████╗
██╔══██╗██╔══██╗    ██╔════╝██╔═══██╗██╔══██╗██╔════╝
██║  ██║███████║    ██║     ██║   ██║██║  ██║█████╗
██║  ██║██╔══██║    ██║     ██║   ██║██║  ██║██╔══╝
██████╔╝██║  ██║    ╚██████╗╚██████╔╝██████╔╝███████╗
╚═════╝ ╚═╝  ╚═╝     ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝

    🧠 Loading LangChain...
""",
        # Frame 3
        r"""
██████╗  █████╗      ██████╗ ██████╗ ██████╗ ███████╗
██╔══██╗██╔══██╗    ██╔════╝██╔═══██╗██╔══██╗██╔════╝
██║  ██║███████║    ██║     ██║   ██║██║  ██║█████╗
██║  ██║██╔══██║    ██║     ██║   ██║██║  ██║██╔══╝
██████╔╝██║  ██║    ╚██████╗╚██████╔╝██████╔╝███████╗
╚═════╝ ╚═╝  ╚═╝     ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝

    ☁️  Connecting to Azure OpenAI...
""",
        # Frame 4
        r"""
██████╗  █████╗      ██████╗ ██████╗ ██████╗ ███████╗
██╔══██╗██╔══██╗    ██╔════╝██╔═══██╗██╔══██╗██╔════╝
██║  ██║███████║    ██║     ██║   ██║██║  ██║█████╗
██║  ██║██╔══██║    ██║     ██║   ██║██║  ██║██╔══╝
██████╔╝██║  ██║    ╚██████╗╚██████╔╝██████╔╝███████╗
╚═════╝ ╚═╝  ╚═╝     ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝

    🔧 Loading MCP Servers...
""",
        # Final frame
        r"""
██████╗  █████╗      ██████╗ ██████╗ ██████╗ ███████╗
██╔══██╗██╔══██╗    ██╔════╝██╔═══██╗██╔══██╗██╔════╝
██║  ██║███████║    ██║     ██║   ██║██║  ██║█████╗
██║  ██║██╔══██║    ██║     ██║   ██║██║  ██║██╔══╝
██████╔╝██║  ██║    ╚██████╗╚██████╔╝██████╔╝███████╗
╚═════╝ ╚═╝  ╚═╝     ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝

    🚀 Ready! Your AI coding assistant is online.
"""
    ]

    return frames


def get_random_taglines() -> List[str]:
    """Get random taglines for variety."""
    return [
        "🤖 Agentic CLI with LangChain & Azure OpenAI 🚀",
        "🧠 AI-Powered Command Line Assistant 🔧",
        "⚡ Smart Automation with Human Oversight 🛡️",
        "🎯 Precision Coding with AI Intelligence 💡",
        "🔬 Advanced AI Tooling for Developers 🚀",
        "🌟 Next-Gen CLI Experience 🤖",
        "⚙️  Intelligent Command Execution 🎪",
        "🎨 Where AI Meets Development Workflow 🔥"
    ]


def get_mini_splash() -> str:
    """Get a smaller splash for quick starts."""
    return r"""
██████╗  █████╗ ██████╗
██╔══██╗██╔══██╗██╔════╝
██║  ██║███████║██║     
██║  ██║██╔══██║██║     
██████╔╝██║  ██║╚██████╗
╚═════╝ ╚═╝  ╚═╝ ╚═════╝
🤖 Your AI Coding Assistant 🚀
"""


def get_status_splash() -> str:
    """Get splash screen for status/setup commands."""
    return r"""
    ╔═════════════════════════════════╗
    ║     ██████╗  █████╗ ██████╗     ║
    ║     ██╔══██╗██╔══██╗██╔════╝    ║
    ║     ██║  ██║███████║██║         ║
    ║     ██║  ██║██╔══██║██║         ║
    ║     ██████╔╝██║  ██╚██████╗     ║
    ║     ╚═════╝ ╚═╝  ╚═╝╚═════╝     ║
    ╚═════════════════════════════════╝
    """


def print_with_colors(text: str, color_code: str = "94") -> None:
    """Print text with ANSI colors."""
    print(f"\033[{color_code}m{text}\033[0m")


def print_gradient_splash(text: str) -> None:
    """Print splash with gradient effect."""
    lines = text.split('\n')
    # True blue gradient: dark blue -> bright blue -> cyan
    colors = ["34", "94", "94", "96", "96", "36", "36", "96"]

    for i, line in enumerate(lines):
        color = colors[i % len(colors)]
        print(f"\033[{color}m{line}\033[0m")


def print_rainbow_splash(text: str) -> None:
    """Print splash with rainbow colors."""
    lines = text.split('\n')
    rainbow_colors = ["91", "93", "92", "96", "94", "95"]

    for i, line in enumerate(lines):
        if line.strip():  # Only color non-empty lines
            color = rainbow_colors[i % len(rainbow_colors)]
            print(f"\033[{color}m{line}\033[0m")
        else:
            print(line)


def show_splash(style: str = "default", mini: bool = False) -> None:
    """Show splash screen with specified style."""

    if mini:
        splash = get_mini_splash()
    else:
        splash = get_splash_screen()
        # Add random tagline
        taglines = get_random_taglines()
        random_tagline = random.choice(taglines)
        splash = splash.replace("🤖 Agentic CLI with LangChain & Azure OpenAI 🚀", random_tagline)

    # Apply styling
    if style == "gradient":
        print_gradient_splash(splash)
    elif style == "rainbow":
        print_rainbow_splash(splash)
    elif style == "blue":
        print_with_colors(splash, "94")
    elif style == "cyan":
        print_with_colors(splash, "96")
    elif style == "green":
        print_with_colors(splash, "92")
    elif style == "yellow":
        print_with_colors(splash, "93")
    elif style == "purple":
        print_with_colors(splash, "95")
    else:
        # Default - no colors
        print(splash)


def show_status_splash() -> None:
    """Show splash for status/configuration commands."""
    print_with_colors(get_status_splash(), "96")


def create_session(context_ldr: ContextLoader):
    # Load project context
    project_context = context_ldr.load_project_context()

    # Load MCP servers
    mcp_servers = context_ldr.load_mcp_servers()

    # Determine working directory
    working_dir = os.getcwd()

    # Create session
    code_session = CodeSession(
        working_directory=working_dir,
        project_context=project_context,
        mcp_servers=mcp_servers,
    )
    return code_session


def initialize_agent(config_mgr: ConfigManager, code_session: CodeSession) -> Optional[AgentInterface]:
    """Initialize agent after configuration is validated."""
    try:
        # Validate configuration
        if not config_mgr.validate_config():
            return None

        # Create agent using the factory
        agent_config = config_mgr.create_agent_config()
        new_agent = AgentFactory.create_agent(agent_config, code_session)

        logger.info("Agent initialized successfully")
        return new_agent

    except Exception as e:
        logger.error(f"Agent initialization failed: {e}")
        print(f"❌ Failed to initialize agent: {e}")
        return None


def create_example_configuration(config_mgr: ConfigManager, context_ldr: ContextLoader) -> int:
    """Setup configuration files."""
    show_status_splash()
    print("🔧 Setting up da_code configuration...")

    try:
        # Create sample environment file
        config_mgr.create_sample_env()

        # Create sample DA files if they don't exist
        if not Path('DA.md').exists():
            context_ldr.create_sample_da_md()

        if not Path('DA.json').exists():
            context_ldr.create_sample_da_json()

        print("\n✅ Setup complete!")
        print("\nNext steps:")
        print("1. Edit .env with your Azure OpenAI credentials")
        print("2. Edit DA.md with your project information")
        print("3. Edit DA.json with your MCP server configuration")
        print("4. Run 'da_code status' to verify configuration")
        print("5. Run 'da_code' to start interactive session")

        return 0

    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return 1



def show_status(config_mgr: ConfigManager, context_ldr: ContextLoader) -> int:
    """Show configuration and system status."""
    try:
        config_mgr.print_config_status()

        # Check project context
        print("\n=== Project Context ===")
        project_context = context_ldr.load_project_context()
        if project_context:
            print(f"✓ DA.md loaded: {project_context.project_name or 'Unnamed project'}")
        else:
            print("✗ DA.md not found or empty")

        # Check MCP servers
        mcp_servers = context_ldr.load_mcp_servers()
        print(f"\n=== MCP Servers ===")
        if mcp_servers:
            print(f"✓ Found {len(mcp_servers)} MCP servers:")
            for server in mcp_servers:
                print(f"  - {server.name}: {server.url}")
        else:
            print("✗ No MCP servers configured")

        return 0

    except Exception as e:
        print(f"❌ Status check failed: {e}")
        return 1

#---------------------------------------------------------------
# Setup
#---------------------------------------------------------------

# Agent will be initialized in main() after config validation


# Available commands
commands = ['help', 'setup', 'status', 'test', 'reload', 'add_mcp', 'exit', 'quit', 'q']

async def async_main():
    """Async main with simple status interface."""
    status_interface = SimpleStatusInterface()

    show_splash("gradient")

    # Check if we need to run setup first
    if not ConfigManager().validate_config():
        console.print("[yellow]Configuration not found. Run 'setup' to create configuration files.[/yellow]")
        console.print(f"Available commands: {', '.join(commands)}")
    else:
        # Initialize agent if config is valid
        status_interface.start_execution("Initializing session...")
        code_session = create_session(context_ldr=ContextLoader())
        if code_session is None:
            status_interface.stop_execution(False, "Session creation failed")
            raise ValueError("Failed to create code session!")

        status_interface.update_status("Initializing agent...")
        agent = initialize_agent(config_mgr=ConfigManager(), code_session=code_session)
        if agent is None:
            status_interface.stop_execution(False, "Agent initialization failed")
            console.print("[red]Agent initialization failed. Run 'setup' to regenerate.[/red]")
        else:
            # Get deployment name
            deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-4')

            # Get chat memory status from agent
            try:
                session_info = agent.get_session_info()
                memory_info = session_info.get('memory_info', {})
                memory_type = memory_info.get('memory_type', 'unknown')
                message_count = memory_info.get('message_count', 0)

                if memory_type == 'postgres':
                    memory_status = "[green]PostgreSQL[/green]"
                elif memory_type == 'file':
                    memory_status = "[yellow]File[/yellow]"
                else:
                    memory_status = "[red]Memory[/red]"

                # Add message count if exists
                if message_count > 0:
                    memory_status += f" ({message_count})"

            except Exception as e:
                memory_status = "[red]Unknown[/red]"

            # Get MongoDB status
            try:
                from .models import get_mongo_status
                mongo_status = get_mongo_status()
                if mongo_status:
                    mongo_status_str = "[green]Connected[/green]"
                else:
                    mongo_status_str = "[yellow]Disconnected[/yellow]"
            except:
                mongo_status_str = "[red]Unknown[/red]"

            # Combined status line
            status_interface.stop_execution(True, f"🤖 {deployment_name} | 💾 {memory_status} | 🍃 {mongo_status_str}")

    history = FileHistory(agent.config.history_file_path)

    # Create PromptSession for async usage
    session = PromptSession(
        history=history,
        multiline=False,
        complete_style='column',
    )

    async def get_user_input_with_history():
        """Get user input with file-based history and arrow key support."""
        try:
            return await session.prompt_async(HTML('<cyan>></cyan> '))
        except (EOFError, KeyboardInterrupt):
            return None

    while True:
        try:
            user_input = await get_user_input_with_history()

            # Handle EOF or interrupt
            if user_input is None:
                break

            if user_input.lower() in ['exit', 'quit', 'q']:
                console.print("👋 Goodbye!")
                break
            elif user_input.lower() == 'help':
                console.print("[bold]Available commands:[/bold]")
                for cmd in commands:
                    console.print(f"  • {cmd}")
            elif user_input.lower() == 'setup':
                create_example_configuration(config_mgr=ConfigManager(), context_ldr=ContextLoader())
                console.print("[green]Edit files and reload to update agent context[/green]")
            elif user_input.lower() == 'status':
                show_status(config_mgr=ConfigManager(), context_ldr=ContextLoader())
            elif user_input.lower() == 'test':
                console.print("🧪 Running Azure OpenAI connection test...")
                try:
                    from .test_connection import main as test_main
                    test_main()
                except Exception as e:
                    console.print(f"[red]❌ Test failed: {e}[/red]")
            elif user_input.lower() == 'reload':
                status_interface.start_execution("Reloading configuration...")
                try:
                    status_interface.update_status("Creating new session...")
                    code_session = create_session(context_ldr=ContextLoader())
                    if code_session is None:
                        raise ValueError("Failed to create code session!")

                    status_interface.update_status("Reinitializing agent...")
                    agent = initialize_agent(config_mgr=ConfigManager(), code_session=code_session)
                    if agent is None:
                        status_interface.stop_execution(False, "Agent initialization failed")
                        console.print("[red]Agent initialization failed. Run 'setup' to regenerate.[/red]")
                    else:
                        status_interface.stop_execution(True, "Configuration reloaded")
                except Exception as e:
                    status_interface.stop_execution(False, f"Reload error: {str(e)}")
                    console.print(f"[red]❌ Reload failed: {e}[/red]")
            elif user_input.lower().startswith('add_mcp '):
                mcp_config_json = user_input[8:].strip()  # Remove 'add_mcp ' prefix
                if not mcp_config_json:
                    console.print("[yellow]Usage: add_mcp <JSON_CONFIG>[/yellow]")
                    console.print('[yellow]Example: add_mcp {"name": "clipboard", "url": "http://192.168.1.100:8000", "port": 8000, "description": "Windows clipboard", "tools": ["read_text", "write_text"]}[/yellow]')
                    continue

                status_interface.start_execution("Adding MCP server...")
                try:
                    import json
                    from pathlib import Path

                    # Parse JSON config
                    status_interface.update_status("Parsing MCP configuration...")
                    try:
                        mcp_config = json.loads(mcp_config_json)
                    except json.JSONDecodeError as e:
                        status_interface.stop_execution(False, "Invalid JSON format")
                        console.print(f"[red]❌ Invalid JSON: {str(e)}[/red]")
                        continue

                    # Validate required fields
                    required_fields = ["name", "url", "port", "description", "tools"]
                    missing_fields = [field for field in required_fields if field not in mcp_config]
                    if missing_fields:
                        status_interface.stop_execution(False, f"Missing required fields: {missing_fields}")
                        console.print(f"[red]❌ Missing required fields: {', '.join(missing_fields)}[/red]")
                        continue

                    # Add to current session dynamically (session-only, not persistent)
                    status_interface.update_status("Adding MCP server to current session...")

                    # Create MCP server object for the session
                    from .models import MCPServerInfo
                    mcp_server = MCPServerInfo(
                        name=mcp_config["name"],
                        url=mcp_config["url"],
                        port=mcp_config["port"],
                        description=mcp_config["description"],
                        tools=mcp_config["tools"]
                    )

                    # Add to current code_session (session-only)
                    if code_session and hasattr(code_session, 'mcp_servers'):
                        # Remove existing server with same name if it exists
                        code_session.mcp_servers = [s for s in code_session.mcp_servers if s.name != mcp_server.name]
                        # Add new server
                        code_session.mcp_servers.append(mcp_server)

                        # Reinitialize agent with updated session
                        status_interface.update_status("Reinitializing agent with new MCP server...")
                        agent = initialize_agent(config_mgr=ConfigManager(), code_session=code_session)
                        if agent is None:
                            status_interface.stop_execution(False, "Agent initialization failed")
                            console.print("[red]Agent initialization failed with new MCP server.[/red]")
                        else:
                            status_interface.stop_execution(True, f"MCP server '{mcp_server.name}' added to session")
                            console.print(f"[green]✅ MCP server '{mcp_server.name}' added successfully![/green]")
                            console.print(f"[green]📋 Available tools: {', '.join(mcp_server.tools)}[/green]")
                            console.print(f"[yellow]⚠️  Session-only: Server will be removed when da_code restarts[/yellow]")
                    else:
                        status_interface.stop_execution(False, "No active session")
                        console.print("[red]❌ No active code session available[/red]")

                except Exception as e:
                    status_interface.stop_execution(False, f"Add MCP error: {str(e)}")
                    console.print(f"[red]❌ Failed to add MCP server: {e}[/red]")
            elif user_input.strip() == '':
                continue
            else:
                if agent is None:
                    console.print("[yellow]Agent not initialized. Run 'setup' first.[/yellow]")
                    continue

                # Beautiful unified streaming execution 🚀
                status_interface.start_execution(f"Processing: {user_input[:40]}...")
                try:
                    final_response = await execute_with_streaming_confirmations(
                        agent, user_input, status_interface
                    )

                    status_interface.stop_execution(True)
                    console.print()  # Empty line for spacing

                    # Display the final response
                    if final_response:
                        console.print(final_response)
                    else:
                        console.print("[yellow]No response received from agent[/yellow]")

                except Exception as e:
                    status_interface.stop_execution(False, str(e))
                    console.print(f"[red]Sorry, I encountered an error: {str(e)}[/red]")
                    logger.error(f"Agent chat error: {type(e).__name__}: {str(e)}", exc_info=True)

        except KeyboardInterrupt:
            if status_interface.current_status:
                status_interface.stop_execution(False, "Interrupted by user")
            console.print("[yellow]⚠️ Interrupted[/yellow]")
            continue
        except EOFError:
            break


def main():
    """Entry point with argument parsing."""
    import argparse

    parser = argparse.ArgumentParser(description="da_code - AI Coding Assistant")
    parser.add_argument('command', nargs='?', choices=['setup', 'status', 'test'],
                       help='Command to run (setup creates config files and exits, test checks connection)')
    parser.add_argument('--working-dir', type=str, help='Working directory')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')

    args = parser.parse_args()

    # Environment is automatically loaded when config module is imported

    # Setup logging with command line arg overriding environment LOG_LEVEL
    log_level = args.log_level if args.log_level != 'INFO' else os.getenv('LOG_LEVEL', 'INFO')
    setup_logging(log_level)

    # Handle command-line commands that exit immediately
    if args.command == 'setup':
        show_status_splash()
        result = create_example_configuration(config_mgr=ConfigManager(), context_ldr=ContextLoader())
        sys.exit(result)
    elif args.command == 'status':
        show_status_splash()
        result = show_status(config_mgr=ConfigManager(), context_ldr=ContextLoader())
        sys.exit(result)
    elif args.command == 'test':
        show_status_splash()
        print("🧪 Running Azure OpenAI connection test...")
        try:
            # Import and run the connection test
            from .test_connection import main as test_main
            result = test_main()
            sys.exit(result)
        except Exception as e:
            print(f"❌ Test failed: {e}")
            sys.exit(1)

    # Interactive mode
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"Main execution error: {e}")

if __name__ == '__main__':
    main()
