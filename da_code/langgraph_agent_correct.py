"""Correct LangGraph agent with proper async HIL pattern."""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, AsyncGenerator

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import Tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import interrupt, Command

from .agent_interface import AgentInterface
from .models import (
    AgentConfig, CodeSession, CommandExecution, CommandStatus,
    LLMCall, LLMCallStatus, ToolCall, ToolCallStatus, UserResponse, da_mongo
)
from .execution_events import ExecutionEvent, EventType, ConfirmationResponse
from .telemetry import TelemetryManager, PerformanceTracker
from .tools import create_all_tools

logger = logging.getLogger(__name__)


class CorrectLangGraphAgent(AgentInterface):
    """LangGraph agent with correct async HIL pattern."""

    def __init__(self, config: AgentConfig, session: CodeSession):
        """Initialize the correct LangGraph agent."""
        super().__init__(config, session)

        # Initialize connection pool for PostgreSQL (if used)
        self.connection_pool = None
        self.checkpointer_setup_complete = False

        # Initialize telemetry manager
        self.telemetry = TelemetryManager(session)

        # Initialize Azure OpenAI
        self.llm = AzureChatOpenAI(
            azure_endpoint=config.azure_endpoint,
            api_key=config.api_key,
            api_version=config.api_version,
            azure_deployment=config.deployment_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.agent_timeout,
            max_retries=config.max_retries
        )


        # Confirmation handler callback
        self.confirmation_handler = None

        # Initialize tools
        self.tools = self._create_tools()

        # Create checkpointer with fallback strategy
        self.checkpointer = self._create_checkpointer()

        # Create the graph with correct HIL pattern
        self.graph = self._create_graph()

        logger.info("Correct LangGraph agent initialized successfully")


    def _create_tools(self) -> Dict[str, Tool]:
        """Create tools for the agent."""
        agent_tools = {}

        # Get standard tools
        modern_tools = create_all_tools(self.session.working_directory)
        for name, tool in modern_tools.items():
            agent_tools[name] = tool

        # Add execute_command tool
        execute_command_tool = Tool(
            name="execute_command",
            description="""Execute shell/bash commands with automatic user confirmation.

Input should be a JSON string with:
- command: The command to execute (required)
- explanation: Brief explanation of what the command does (optional)
- reasoning: Why this command is needed (optional)
- working_directory: Directory to run command in (optional)
- related_files: List of files this command affects (optional)

Example: {"command": "ls -la", "explanation": "List directory contents", "reasoning": "User wants to see files"}

The tool will request user confirmation before executing any command.""",
            func=self._execute_command_sync_wrapper
        )
        agent_tools["execute_command"] = execute_command_tool

        # Add MCP server tools
        mcp_tools = self._create_mcps()
        agent_tools.update(mcp_tools)

        return agent_tools

    def _create_checkpointer(self):
        """Create checkpointer with fallback strategy: PostgreSQL -> Memory."""
        import os

        postgres_url = os.getenv("POSTGRES_CHAT_URL", None)
        logger.info(f"🔍 CHECKPOINTER: POSTGRES_CHAT_URL configured: {postgres_url is not None}")

        if postgres_url:
            try:
                logger.info("🔍 CHECKPOINTER: Attempting to create PostgreSQL checkpointer...")

                # Import PostgreSQL dependencies
                from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
                from psycopg_pool import AsyncConnectionPool

                # Create connection pool directly - this works better for non-context scenarios
                logger.info("🔍 CHECKPOINTER: Creating AsyncConnectionPool...")
                pool = AsyncConnectionPool(
                    conninfo=postgres_url,
                    max_size=10,
                    min_size=1,
                    open=False,  # We'll open it later
                    kwargs={
                        "autocommit": True  # Required for CREATE INDEX CONCURRENTLY
                    }
                )

                # Create checkpointer with the pool
                checkpointer = AsyncPostgresSaver(pool)

                # Store pool for lifecycle management
                self.connection_pool = pool

                logger.info("✅ CHECKPOINTER: PostgreSQL checkpointer created successfully")
                return checkpointer

            except ImportError as e:
                logger.warning(f"🔍 CHECKPOINTER: PostgreSQL dependencies not installed: {e}")
                logger.info("🔍 CHECKPOINTER: Install with: pip install langgraph-checkpoint-postgres")
            except Exception as e:
                logger.warning(f"❌ CHECKPOINTER: PostgreSQL setup failed: {type(e).__name__}: {e}")

        logger.info("🔍 CHECKPOINTER: Using in-memory checkpointer (no persistence)")
        return MemorySaver()

    async def _setup_checkpointer(self):
        """Setup the checkpointer (open pool and create tables if using PostgreSQL)."""
        # Skip if already set up
        if self.checkpointer_setup_complete:
            return

        # Open connection pool if we have one
        if self.connection_pool:
            try:
                # Try to open the pool - this is idempotent, won't fail if already open
                logger.info("🔍 CHECKPOINTER: Opening connection pool...")
                await self.connection_pool.open(wait=True)
                logger.info("✅ CHECKPOINTER: Connection pool opened successfully")
            except Exception as e:
                logger.warning(f"❌ CHECKPOINTER: Failed to open connection pool: {e}")
                # Don't return here - still try to setup in case it's just a duplicate open

        # Setup tables if checkpointer supports it
        if hasattr(self.checkpointer, 'setup'):
            try:
                logger.info("🔍 CHECKPOINTER: Setting up PostgreSQL tables...")
                await self.checkpointer.setup()
                logger.info("✅ CHECKPOINTER: PostgreSQL tables created successfully")
                self.checkpointer_setup_complete = True
            except Exception as e:
                logger.warning(f"❌ CHECKPOINTER: Setup failed: {e}")
                logger.info("🔍 CHECKPOINTER: Continuing with checkpointer anyway")
                # Don't mark as complete if it failed
        else:
            # If no setup method, mark as complete anyway
            self.checkpointer_setup_complete = True

    def _execute_command_sync_wrapper(self, tool_input: str) -> str:
        """Sync wrapper for execute_command tool - used by LangGraph's sync tool interface."""
        try:
            # Parse tool input
            if tool_input.strip().startswith('{'):
                try:
                    params = json.loads(tool_input)
                except json.JSONDecodeError:
                    return f"Error: Invalid JSON format: {tool_input[:100]}"
            else:
                params = {"command": tool_input}

            command = params.get("command", "").strip()
            if not command:
                return "Error: No command provided"

            # Create a fake tool_call dict for compatibility with _execute_command_tool
            tool_call = {
                "args": params,
                "id": "sync_wrapper"
            }

            # Run the async method in the current event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in an async context, create a task
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, self._execute_command_tool(tool_call))
                        return future.result()
                else:
                    return loop.run_until_complete(self._execute_command_tool(tool_call))
            except RuntimeError:
                # No event loop, create one
                return asyncio.run(self._execute_command_tool(tool_call))

        except Exception as e:
            logger.error(f"Error in command sync wrapper: {e}")
            return f"Error executing command: {str(e)}"

    def _create_mcps(self) -> Dict[str, Tool]:
        """Create MCP server tools from session configuration."""
        mcp_tools = {}

        if not self.session.mcp_servers:
            return mcp_tools

        for mcp_server in self.session.mcp_servers:
            try:
                for tool_name in mcp_server.tools:
                    mcp_tool_name = f"{mcp_server.name}_{tool_name}"
                    mcp_tools[mcp_tool_name] = self._create_mcp_tool(mcp_server, tool_name)
                    logger.info(f"Added MCP tool: {mcp_tool_name} from {mcp_server.name}")
            except Exception as e:
                logger.error(f"Failed to load MCP server {mcp_server.name}: {e}")
                continue

        return mcp_tools

    def _create_mcp_tool(self, mcp_server, tool_name: str) -> Tool:
        """Create a tool that calls an MCP server."""
        import httpx

        def sync_mcp_call(tool_input: str) -> str:
            try:
                try:
                    parsed_input = json.loads(tool_input) if tool_input.strip().startswith('{') else {"input": tool_input}

                    # Handle LangChain's __arg1 format when called via bind_tools()
                    if "__arg1" in parsed_input:
                        # For clipboard tools, map __arg1 to the expected argument name
                        if tool_name == "write_text":
                            arguments = {"text": parsed_input["__arg1"]}
                        elif tool_name == "read_text":
                            arguments = {}  # read_text takes no arguments
                        elif tool_name == "write_image":
                            arguments = {
                                "image_data": parsed_input["__arg1"],
                                "format": parsed_input.get("format", "PNG")
                            }
                        elif tool_name == "read_image":
                            arguments = {"format": parsed_input.get("format", "PNG")}
                        else:
                            # For other tools, use generic mapping
                            arguments = {"input": parsed_input["__arg1"]}
                    else:
                        arguments = parsed_input
                except json.JSONDecodeError:
                    arguments = {"input": tool_input}

                logger.debug(f"MCP tool {tool_name} called with arguments: {arguments}")

                # Additional debug for image tools
                if tool_name in ["write_image", "read_image"]:
                    if "image_data" in arguments:
                        logger.debug(f"Image data length: {len(arguments['image_data'])} characters")
                    logger.debug(f"Full MCP call URL: {mcp_server.url}/mcp/call/{tool_name}")

                async def call_mcp():
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        payload = {"arguments": arguments}
                        logger.debug(f"MCP payload for {tool_name}: {payload if tool_name != 'write_image' else {k: f'<{len(v)} chars>' if k == 'arguments' and 'image_data' in str(v) else v for k, v in payload.items()}}")

                        response = await client.post(
                            f"{mcp_server.url}/mcp/call/{tool_name}",
                            json=payload
                        )
                        response.raise_for_status()
                        result = response.json()

                        logger.debug(f"MCP response for {tool_name}: {result}")

                        if result.get("isError", False):
                            error_content = result.get('content', [{}])
                            error_text = error_content[0].get('text', 'Unknown error') if error_content else 'Unknown error'
                            return f"❌ MCP Error: {error_text}"
                        content = result.get("content", [])
                        return content[0].get("text", "No response") if content else "No response from MCP server"

                try:
                    # Try to get the current event loop
                    try:
                        loop = asyncio.get_running_loop()
                        # We're in an async context - use ThreadPoolExecutor to run async code
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(asyncio.run, call_mcp())
                            result = future.result()
                            logger.debug(f"MCP tool {tool_name} result: {result}")
                            return result
                    except RuntimeError:
                        # No running loop - safe to use asyncio.run
                        result = asyncio.run(call_mcp())
                        logger.debug(f"MCP tool {tool_name} result: {result}")
                        return result
                except Exception as async_error:
                    logger.error(f"Async execution error for MCP tool {tool_name}: {async_error}")
                    return f"❌ MCP async error: {str(async_error)}"

            except Exception as e:
                return f"❌ MCP tool error: {str(e)}"

        # Create more descriptive tool descriptions based on tool name and server
        tool_description = f"{tool_name} tool from {mcp_server.name} MCP server"

        # Enhanced descriptions for clipboard tools
        if mcp_server.name == "clipboard":
            if tool_name == "write_text":
                tool_description = "Write text to Windows clipboard. Use this for copying text to clipboard instead of shell commands."
            elif tool_name == "read_text":
                tool_description = "Read text from Windows clipboard. Use this to get clipboard contents."
            elif tool_name == "write_image":
                tool_description = "Write base64 image to Windows clipboard. Use this for copying images to clipboard."
            elif tool_name == "read_image":
                tool_description = "Read image from Windows clipboard as base64. Use this to get clipboard images."

        return Tool(
            name=f"{mcp_server.name}_{tool_name}",
            description=tool_description,
            func=sync_mcp_call
        )

    def _create_graph(self):
        """Create the LangGraph workflow with correct HIL pattern."""
        workflow = StateGraph(MessagesState)

        # Add nodes
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", self._tools_node)

        # Add edges
        workflow.add_edge(START, "agent")

        # Conditional edge: continue to tools or end
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END
            }
        )

        # After tools, go back to agent
        workflow.add_edge("tools", "agent")

        # Compile without interrupt_before since we handle interrupts in the tools node
        return workflow.compile(
            checkpointer=self.checkpointer
        )

    async def _agent_node(self, state: MessagesState):
        """The main agent reasoning node."""
        system_prompt = self._build_system_prompt()
        messages = [SystemMessage(content=system_prompt)] + state["messages"]

        # Debug: Log message sequence
        logger.info(f"🔍 AGENT: Processing {len(messages)} messages")
        for i, msg in enumerate(messages[-5:]):  # Log last 5 messages
            msg_type = type(msg).__name__
            has_tool_calls = hasattr(msg, 'tool_calls') and msg.tool_calls
            tool_call_id = getattr(msg, 'tool_call_id', None)
            content_preview = str(msg.content)[:100] if hasattr(msg, 'content') and msg.content else 'No content'
            logger.info(f"🔍 AGENT: Msg {i}: {msg_type} | Tool calls: {has_tool_calls} | Tool call ID: {tool_call_id} | Content: {content_preview}")

        # Bind tools to LLM
        logger.info(f"🔍 AGENT: Available tools: {list(self.tools.keys())}")
        logger.info(f"🔍 AGENT: Binding {len(self.tools)} tools to LLM")

        tools_list = list(self.tools.values())
        for i, tool in enumerate(tools_list[:3]):  # Log first 3 tools
            logger.info(f"🔍 AGENT: Tool {i}: {tool.name} - {type(tool).__name__}")

        llm_with_tools = self.llm.bind_tools(tools_list)

        # Get response from LLM
        start_time = time.time()
        try:
            response = await llm_with_tools.ainvoke(messages)

            # Debug: Log response details
            logger.info(f"🔍 AGENT: LLM response type: {type(response).__name__}")
            logger.info(f"🔍 AGENT: Response content: {response.content[:200] if hasattr(response, 'content') and response.content else 'No content'}...")
            logger.info(f"🔍 AGENT: Has tool_calls: {hasattr(response, 'tool_calls')}")
            if hasattr(response, 'tool_calls') and response.tool_calls:
                logger.info(f"🔍 AGENT: Number of tool calls: {len(response.tool_calls)}")
                for i, tool_call in enumerate(response.tool_calls[:2]):  # Log first 2
                    logger.info(f"🔍 AGENT: Tool call {i}: {tool_call.get('name', 'Unknown')} (ID: {tool_call.get('id', 'No ID')})")

            # Track LLM call
            llm_call = LLMCall(
                model_name=self.config.deployment_name,
                provider="azure_openai",
                prompt=str(messages[-1].content) if messages else "",
                response=response.content if response.content else "[Tool calls requested]",
                status=LLMCallStatus.SUCCESS,
                response_time_ms=(time.time() - start_time) * 1000
            )
            self.session.add_llm_call(llm_call)

            return {"messages": [response]}

        except Exception as e:
            llm_call = LLMCall(
                model_name=self.config.deployment_name,
                provider="azure_openai",
                prompt=str(messages[-1].content) if messages else "",
                response="",
                status=LLMCallStatus.FAILED,
                error_message=str(e),
                response_time_ms=(time.time() - start_time) * 1000
            )
            self.session.add_llm_call(llm_call)
            raise

    async def _tools_node(self, state: MessagesState):
        """Execute tools with proper interrupt pattern for dangerous tools."""
        logger.info("🔍 TOOLS: Starting tools node execution")
        messages = state["messages"]
        last_message = messages[-1]

        logger.info(f"🔍 TOOLS: Processing state with {len(messages)} messages")
        logger.info(f"🔍 TOOLS: Last message type: {type(last_message).__name__}")

        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            logger.info("🔍 TOOLS: No tool calls found, returning empty")
            return {"messages": []}

        logger.info(f"🔍 TOOLS: Found {len(last_message.tool_calls)} tool calls to execute")

        # Define dangerous tools that require confirmation
        dangerous_tools = {"execute_command"}
        tool_messages = []
        dangerous_tool_calls = []

        # Separate safe and dangerous tool calls
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            if tool_name in dangerous_tools:
                dangerous_tool_calls.append(tool_call)
            else:
                # Execute safe tools immediately
                logger.info(f"🔍 TOOLS: Auto-executing safe tool: {tool_name}")
                result = await self._execute_single_tool(tool_call)
                tool_messages.append(result)

        # If we have dangerous tools, interrupt for confirmation
        if dangerous_tool_calls:
            logger.info(f"🔍 TOOLS: Found {len(dangerous_tool_calls)} dangerous tools requiring confirmation")

            # Prepare confirmation data
            confirmations_needed = []
            for tool_call in dangerous_tool_calls:
                args = tool_call["args"]
                # Handle LangChain's tool argument format
                if "__arg1" in args:
                    try:
                        args = json.loads(args["__arg1"])
                    except json.JSONDecodeError:
                        pass

                confirmations_needed.append({
                    "tool_call": tool_call,
                    "tool_name": tool_call["name"],
                    "args": args
                })

            # Use interrupt to pause and get user confirmation
            user_response = interrupt({
                "type": "tool_confirmation",
                "tools_needing_confirmation": confirmations_needed,
                "message": f"The following {len(dangerous_tool_calls)} dangerous tool(s) require your confirmation:"
            })

            logger.info(f"🔍 TOOLS: Received user response: {user_response}")

            # Process user response and execute confirmed tools
            if isinstance(user_response, dict) and user_response.get("confirmed_tools"):
                for tool_call in user_response["confirmed_tools"]:
                    logger.info(f"🔍 TOOLS: Executing confirmed dangerous tool: {tool_call['name']}")
                    result = await self._execute_single_tool(tool_call)
                    tool_messages.append(result)
            else:
                # User cancelled - create cancellation messages
                for tool_call in dangerous_tool_calls:
                    tool_messages.append(ToolMessage(
                        content="Command execution was cancelled by user.",
                        tool_call_id=tool_call["id"]
                    ))

        return {"messages": tool_messages}

    async def _execute_single_tool(self, tool_call: dict):
        """Execute a single tool call and return the result message."""
        tool_name = tool_call["name"]
        tool_id = tool_call["id"]
        logger.info(f"🔍 TOOLS: Executing tool: {tool_name} (ID: {tool_id})")

        if tool_name in self.tools:
            tool = self.tools[tool_name]

            # Track tool call
            tool_call_record = ToolCall(
                server_name="internal",
                tool_name=tool_name,
                arguments=tool_call["args"],
                status=ToolCallStatus.PENDING
            )

            try:
                logger.info(f"🔍 TOOLS: Starting execution of {tool_name}")
                # Handle execute_command specially
                if tool_name == "execute_command":
                    result = await self._execute_command_tool(tool_call)
                else:
                    # Execute other tools normally
                    if hasattr(tool, 'func'):
                        tool_input = json.dumps(tool_call["args"]) if isinstance(tool_call["args"], dict) else str(tool_call["args"])
                        logger.info(f"🔍 TOOLS: Calling {tool_name}.func() with input: {tool_input[:100]}...")
                        result = tool.func(tool_input)
                    elif hasattr(tool, 'execute'):
                        result = await tool.execute(json.dumps(tool_call["args"]))
                    else:
                        result = f"Error: Tool {tool_name} has no callable method"

                logger.info(f"🔍 TOOLS: Tool {tool_name} completed successfully")
                logger.info(f"🔍 TOOLS: Tool {tool_name} result preview: {str(result)[:200]}...")

                tool_call_record.result = {"output": result}
                tool_call_record.status = ToolCallStatus.SUCCESS

            except Exception as e:
                logger.info(f"🔍 TOOLS: Tool {tool_name} failed with error: {str(e)}")
                result = f"Error executing {tool_name}: {str(e)}"
                tool_call_record.status = ToolCallStatus.FAILED
                tool_call_record.result = {"error": str(e)}

            finally:
                self.session.add_tool_call(tool_call_record)

            return ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            )
        else:
            logger.info(f"🔍 TOOLS: Unknown tool: {tool_name}")
            return ToolMessage(
                content=f"Error: Unknown tool {tool_name}",
                tool_call_id=tool_call["id"]
            )

    async def _execute_command_tool(self, tool_call: Dict[str, Any]) -> str:
        """Execute command with confirmation handling."""
        try:
            args = tool_call["args"]

            # Handle LangChain's tool argument format
            if isinstance(args, dict) and "__arg1" in args:
                try:
                    args = json.loads(args["__arg1"])
                except json.JSONDecodeError:
                    args = {"command": args["__arg1"]}

            # Parse command input
            if isinstance(args, dict):
                command = args.get("command", "")
                explanation = args.get("explanation", "")
                reasoning = args.get("reasoning", "")
                working_directory = args.get("working_directory", self.session.working_directory)
                related_files = args.get("related_files", [])
            else:
                # Fallback if args is a string
                try:
                    parsed = json.loads(str(args))
                    command = parsed.get("command", str(args))
                    explanation = parsed.get("explanation", "")
                    reasoning = parsed.get("reasoning", "")
                    working_directory = parsed.get("working_directory", self.session.working_directory)
                    related_files = parsed.get("related_files", [])
                except:
                    command = str(args)
                    explanation = ""
                    reasoning = ""
                    working_directory = self.session.working_directory
                    related_files = []

            if not command.strip():
                return "Error: No command provided"

            # Create command execution record
            execution = CommandExecution(
                command=command,
                explanation=explanation,
                working_directory=working_directory,
                agent_reasoning=reasoning,
                related_files=related_files
            )

            # The HIL confirmation happens BEFORE this node is called via LangGraph interrupt
            # So if we get here, the command was already approved - just execute it

            # Execute the approved command
            return await self._execute_approved_command(execution)

        except Exception as e:
            logger.error(f"Error in command execution: {e}")
            return f"Error executing command: {str(e)}"

    async def _execute_approved_command(self, execution: CommandExecution) -> str:
        """Execute an approved command."""
        import subprocess

        try:
            execution.update_status(CommandStatus.APPROVED)
            start_time = time.time()

            result = subprocess.run(
                execution.command,
                shell=True,
                cwd=execution.working_directory,
                capture_output=True,
                text=True,
                timeout=300
            )

            execution.exit_code = result.returncode
            execution.stdout = result.stdout
            execution.stderr = result.stderr
            execution.execution_time = time.time() - start_time

            if result.returncode == 0:
                execution.update_status(CommandStatus.SUCCESS)
                self.session.add_execution(execution)

                output = f"✅ Command executed successfully\n"
                if result.stdout:
                    stdout = result.stdout.strip()
                    output += f"Output:\n{stdout[:2000]}" + ("...\n(truncated)" if len(stdout) > 2000 else "")
                else:
                    output += "No output"
                return output
            else:
                execution.update_status(CommandStatus.FAILED)
                self.session.add_execution(execution)

                output = f"❌ Command failed (exit code: {result.returncode})\n"
                if result.stderr:
                    stderr = result.stderr.strip()
                    output += f"Error:\n{stderr[:1000]}" + ("...\n(truncated)" if len(stderr) > 1000 else "")
                return output

        except subprocess.TimeoutExpired:
            execution.update_status(CommandStatus.TIMEOUT)
            self.session.add_execution(execution)
            return "⏰ Command timed out after 5 minutes"
        except Exception as e:
            execution.update_status(CommandStatus.FAILED)
            execution.stderr = str(e)
            self.session.add_execution(execution)
            return f"❌ Command execution failed: {str(e)}"

    def _should_continue(self, state: MessagesState) -> str:
        """Decide whether to continue to tools or end."""
        messages = state["messages"]
        last_message = messages[-1]

        # Check if there are tool calls
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "continue"

        # No tool calls, end the conversation
        return "end"

    async def execute_task(self, task: str) -> str:
        """Execute a task and return the response (legacy method)."""
        final_response = None
        async for event in self.execute_task_stream(task):
            if event.event_type == EventType.FINAL_RESPONSE:
                final_response = event.content
                break
        return final_response or "No response generated"

    async def execute_task_stream(self, task: str, confirmation_handler=None) -> AsyncGenerator[ExecutionEvent, ConfirmationResponse]:
        """Execute a task with streaming events and confirmation support."""

        self.confirmation_handler = confirmation_handler
        logger.info(f"🔍 EXEC: Starting task execution: '{task[:50]}...'")

        # Setup checkpointer if needed (first time only)
        await self._setup_checkpointer()

        with PerformanceTracker(self.telemetry, "langgraph", "execute_task_stream") as tracker:
            yield ExecutionEvent(
                event_type=EventType.EXECUTION_START,
                framework="langgraph",
                content=f"Starting task: {task[:50]}..."
            )

            try:
                # Create config with thread ID for checkpointing
                config = {"configurable": {"thread_id": self.session.session_id}}
                logger.info(f"🔍 EXEC: Created config with thread_id: {self.session.session_id}")

                # Initial state with user message
                initial_state = {"messages": [HumanMessage(content=task)]}
                logger.info(f"🔍 EXEC: Created initial state with {len(initial_state['messages'])} messages")

                # Execute the graph using proper interrupt/resume pattern
                logger.info("🔍 EXEC: Starting graph execution with proper interrupt/resume pattern...")

                # Start execution using async API since our nodes are async
                result = await self.graph.ainvoke(initial_state, config=config)

                # Check if we hit an interrupt
                while "__interrupt__" in result:
                    logger.info("🔍 EXEC: Hit interrupt, checking for tool confirmation needs...")

                    # Extract interrupt data
                    interrupt_data = result["__interrupt__"][0]
                    interrupt_value = interrupt_data.value if hasattr(interrupt_data, 'value') else interrupt_data.get('value', {})

                    logger.info(f"🔍 EXEC: Interrupt data: {interrupt_value}")

                    if interrupt_value.get("type") == "tool_confirmation":
                        # This is a tool confirmation interrupt
                        tools_needing_confirmation = interrupt_value.get("tools_needing_confirmation", [])

                        # Use the existing confirmation handler pattern instead of generator yield
                        if self.confirmation_handler:
                            confirmed_tools = []

                            for tool_data in tools_needing_confirmation:
                                # Create CommandExecution for the confirmation handler
                                args = tool_data["args"]
                                execution = CommandExecution(
                                    command=args.get("command", ""),
                                    explanation=args.get("explanation", ""),
                                    working_directory=args.get("working_directory", self.session.working_directory),
                                    agent_reasoning=args.get("reasoning", ""),
                                    related_files=args.get("related_files", [])
                                )

                                # Get user confirmation using the callback
                                logger.info(f"🔍 EXEC: Requesting confirmation for: {execution.command}")
                                confirmation_response = await self.confirmation_handler(execution)

                                if confirmation_response and confirmation_response.choice.lower() == "yes":
                                    confirmed_tools.append(tool_data["tool_call"])
                                    logger.info(f"🔍 EXEC: User confirmed: {execution.command}")
                                else:
                                    logger.info(f"🔍 EXEC: User declined: {execution.command}")

                            resume_data = {
                                "confirmed_tools": confirmed_tools
                            }
                        else:
                            # No confirmation handler - reject all dangerous tools
                            logger.warning("🔍 EXEC: No confirmation handler available, rejecting dangerous tools")
                            resume_data = {
                                "confirmed_tools": []
                            }

                        # Resume execution with user response
                        logger.info(f"🔍 EXEC: Resuming with data: {resume_data}")
                        result = await self.graph.ainvoke(Command(resume=resume_data), config=config)
                    else:
                        # Unknown interrupt type - break out
                        logger.warning(f"🔍 EXEC: Unknown interrupt type: {interrupt_value.get('type')}")
                        break

                # Extract final response from the completed execution
                logger.info("🔍 EXEC: Extracting final response...")
                final_response = "Task completed"

                # Extract from the graph state which contains all messages
                # Try async first (for AsyncPostgresSaver), fall back to sync (for MemorySaver compatibility)
                try:
                    graph_state = await self.graph.aget_state(config)
                except AttributeError:
                    # Fallback for checkpointers that don't support async state access
                    graph_state = self.graph.get_state(config)
                if hasattr(graph_state, 'values') and 'messages' in graph_state.values:
                    messages = graph_state.values['messages']
                    logger.info(f"🔍 EXEC: Found {len(messages)} messages in final graph state")

                    # Get the last AI message that has content
                    for i, msg in enumerate(reversed(messages)):
                        logger.debug(f"Message {i}: {type(msg).__name__}: {msg.content[:100] if hasattr(msg, 'content') and msg.content else 'No content'}...")
                        if isinstance(msg, AIMessage) and msg.content and not hasattr(msg, 'tool_calls'):
                            # This is a final AI response without tool calls
                            final_response = msg.content
                            logger.info(f"🔍 EXEC: Found final AI response: {final_response[:100]}...")
                            break
                        elif isinstance(msg, AIMessage) and msg.content:
                            # This might be an AI message with tool calls, but if it has content, use it as fallback
                            if not final_response or final_response == "Task completed":
                                final_response = msg.content
                                logger.info(f"🔍 EXEC: Using AI message with tool calls as response: {final_response[:100]}...")
                else:
                    logger.warning("🔍 EXEC: No messages found in graph state")

                logger.info(f"🔍 EXEC: Final response length: {len(final_response)} characters")


                # Track successful execution
                await self.telemetry.track_agent_call(
                    prompt=task,
                    response=final_response,
                    tokens_used=0,
                    execution_time_ms=tracker.get_duration_ms(),
                    success=True
                )

                await da_mongo.save_session(self.session)

                yield ExecutionEvent(
                    event_type=EventType.FINAL_RESPONSE,
                    framework="langgraph",
                    content=final_response
                )

            except Exception as e:
                logger.info(f"❌ EXEC: Exception caught: {type(e).__name__}: {str(e)}")
                logger.info(f"🔍 EXEC: Full exception details: {e}", exc_info=True)

                await self.telemetry.track_agent_call(
                    prompt=task,
                    response="",
                    tokens_used=0,
                    execution_time_ms=tracker.get_duration_ms(),
                    success=False,
                    error_message=str(e)
                )

                yield ExecutionEvent(
                    event_type=EventType.ERROR,
                    framework="langgraph",
                    error_message=str(e),
                    content=f"Execution failed: {str(e)}"
                )

                logger.error(f"Correct LangGraph agent execution failed: {e}")

            finally:
                self.confirmation_handler = None

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the agent."""
        context_parts = []

        if self.session.project_context:
            context_parts.append(f"Project: {self.session.project_context.project_name}")
            if self.session.project_context.description:
                context_parts.append(f"Description: {self.session.project_context.description}")

        context_parts.append(f"Working Directory: {self.session.working_directory}")
        context = "\n".join(context_parts) if context_parts else "No additional context available."

        tools_desc = "\n".join([f"- {name}: {tool.description}" for name, tool in self.tools.items()])

        # Check if clipboard tools are available
        has_clipboard_tools = any("clipboard_" in name for name in self.tools.keys())
        clipboard_instruction = ""
        if has_clipboard_tools:
            clipboard_instruction = "\n3. CLIPBOARD OPERATIONS: When user asks to copy text, data, or files to clipboard, ALWAYS use clipboard_write_text tool instead of echo, cat, or shell commands"

        return f"""You are da_code, an AI coding assistant with access to tools for command execution and file operations.

CONTEXT:
{context}

AVAILABLE TOOLS:
{tools_desc}

INSTRUCTIONS:
1. Use the available tools to help users with their coding tasks
2. For command execution, use the execute_command tool - user confirmation is handled automatically{clipboard_instruction}
4. Be thorough but efficient in your approach
5. Provide clear explanations for your actions
6. Always use proper tool arguments as specified in the tool descriptions

TODO MANAGEMENT:
- Use the todo_file_manager tool to track work items for complex multi-step tasks
- Read existing todos at the start: {{"operation": "read"}}
- Create/update todos when planning work: {{"operation": "create", "content": "markdown todo list"}}
- Use proper markdown format with checkboxes: `- [ ] Task description`
- Mark completed items: `- [x] Completed task`

You have access to advanced tools for file operations, web search, and system commands. Use them wisely to complete user requests."""

    def get_framework_name(self) -> str:
        return "langgraph"

    def get_session_info(self) -> Dict[str, Any]:
        info = {
            "framework": self.get_framework_name(),
            "session_id": self.session.session_id,
            "agent_version": "1.0.0",
            "model_name": self.config.deployment_name,
            "temperature": self.config.temperature,
            "total_commands": len(self.session.executions),
            "successful_commands": self.session.successful_commands,
            "failed_commands": self.session.failed_commands,
        }

        # Map checkpointer type to memory_type for CLI status display
        checkpointer_name = type(self.checkpointer).__name__
        if "PostgresSaver" in checkpointer_name or "AsyncPostgresSaver" in checkpointer_name:
            memory_type = "postgres"
        elif "MemorySaver" in checkpointer_name:
            memory_type = "memory"
        else:
            memory_type = "unknown"

        info["memory_info"] = {
            "memory_type": memory_type,
            "persistent": memory_type == "postgres",
            "message_count": 0
        }

        framework_metrics = self.telemetry.get_framework_metrics("langgraph")
        if framework_metrics:
            info["framework_metrics"] = framework_metrics

        return info

    def clear_memory(self) -> None:
        try:
            # Clear checkpointer state for this session's thread
            config = {"configurable": {"thread_id": self.session.session_id}}

            # Delete the thread (async for AsyncPostgresSaver)
            if hasattr(self.checkpointer, 'adelete_thread'):
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # We're in an async context, create a task
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(asyncio.run, self.checkpointer.adelete_thread(config))
                            future.result()
                    else:
                        loop.run_until_complete(self.checkpointer.adelete_thread(config))
                    logger.info(f"Checkpointer thread {self.session.session_id} cleared")
                except Exception as async_error:
                    logger.error(f"Failed to delete thread asynchronously: {async_error}")
            elif hasattr(self.checkpointer, 'delete_thread'):
                self.checkpointer.delete_thread(config)
                logger.info(f"Checkpointer thread {self.session.session_id} cleared")
            else:
                logger.warning("Checkpointer doesn't support thread deletion - memory not cleared")
        except Exception as e:
            logger.error(f"Failed to clear checkpointer state: {e}")
            raise

    async def get_metrics(self) -> Dict[str, Any]:
        base_metrics = await super().get_metrics()
        langgraph_metrics = self.telemetry.get_framework_metrics("langgraph")
        base_metrics.update({
            "langgraph_metrics": langgraph_metrics,
            "checkpointer_type": type(self.checkpointer).__name__,
            "model_name": self.config.deployment_name,
            "temperature": self.config.temperature
        })
        return base_metrics