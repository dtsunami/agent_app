"""Pydantic models for ToolSession tracking."""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict, field_serializer
from bson import ObjectId


class PyObjectId(ObjectId):
    """Custom ObjectId for Pydantic compatibility."""
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


class ToolConfig(BaseModel):
    """Pydantic V2 model for tool configuration."""
    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid',
        str_strip_whitespace=True,
        populate_by_name=True,
        arbitrary_types_allowed=True
    )

    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    name: str = Field(..., description="Tool name identifier")
    environment_command: Optional[str] = Field(None, description="Optional environment setup command")
    launch_command: str = Field(..., description="Command to launch the tool")
    prompt_string: str = Field("$ ", description="String that indicates tool is ready for input")
    working_directory: str = Field("/tmp", description="Working directory for the tool")
    timeout: int = Field(300, ge=1, le=3600, description="Timeout in seconds")

    @field_serializer("id")
    def serialize_object_id(self, value: PyObjectId) -> str:
        """Serialize ObjectId to string for JSON output."""
        return str(value)


class SessionError(BaseModel):
    """Model for tracking session errors."""
    model_config = ConfigDict(
        validate_assignment=True,
        populate_by_name=True,
        arbitrary_types_allowed=True
    )

    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error_type: str = Field(..., description="Type of error (startup, command, timeout, etc.)")
    message: str = Field(..., description="Error message")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional error context")

    @field_serializer("id")
    def serialize_object_id(self, value: PyObjectId) -> str:
        """Serialize ObjectId to string for JSON output."""
        return str(value)


class SessionInput(BaseModel):
    """Model for tracking session inputs."""
    model_config = ConfigDict(
        validate_assignment=True,
        populate_by_name=True,
        arbitrary_types_allowed=True
    )

    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    command: str = Field(..., description="Command sent to session")
    input_type: str = Field("command", description="Type of input (command, script)")
    script_file: Optional[str] = Field(None, description="Path to script file if applicable")

    @field_serializer("id")
    def serialize_object_id(self, value: PyObjectId) -> str:
        """Serialize ObjectId to string for JSON output."""
        return str(value)


class SessionOutput(BaseModel):
    """Model for tracking session outputs between prompts."""
    model_config = ConfigDict(
        validate_assignment=True,
        populate_by_name=True,
        arbitrary_types_allowed=True
    )

    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    text: str = Field(..., description="Output text between prompts")
    lines_count: int = Field(..., ge=0, description="Number of lines in output")
    related_input_id: Optional[str] = Field(None, description="ID of input that generated this output")

    @field_serializer("id")
    def serialize_object_id(self, value: PyObjectId) -> str:
        """Serialize ObjectId to string for JSON output."""
        return str(value)


class SessionScript(BaseModel):
    """Model for tracking script executions."""
    model_config = ConfigDict(
        validate_assignment=True,
        populate_by_name=True,
        arbitrary_types_allowed=True
    )

    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    script_content: str = Field(..., description="Content of the executed script")
    command_template: str = Field(..., description="Command template used to execute script")
    script_file: str = Field(..., description="Path to temporary script file")
    execution_command: str = Field(..., description="Final command executed")

    @field_serializer("id")
    def serialize_object_id(self, value: PyObjectId) -> str:
        """Serialize ObjectId to string for JSON output."""
        return str(value)


class ToolSession(BaseModel):
    """Pydantic V2 model for tracking tool session state and history."""
    model_config = ConfigDict(
        validate_assignment=True,
        extra='allow',  # Allow additional fields for extensibility
        populate_by_name=True,
        arbitrary_types_allowed=True
    )

    # Session identification
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_serializer("id")
    def serialize_object_id(self, value: PyObjectId) -> str:
        """Serialize ObjectId to string for JSON output."""
        return str(value)

    # Configuration reference
    tool_config: ToolConfig = Field(..., description="Tool configuration used to create session")

    # Session state
    status: str = Field("created", description="Session status (created, starting, active, stopped, error)")
    pid: Optional[int] = Field(None, description="Process ID of the tool session")
    output_file: str = Field(..., description="Path to session output file")

    # Session history tracking
    errors: List[SessionError] = Field(default_factory=list, description="List of session errors")
    inputs: List[SessionInput] = Field(default_factory=list, description="List of session inputs")
    outputs: List[SessionOutput] = Field(default_factory=list, description="List of session outputs")
    scripts: List[SessionScript] = Field(default_factory=list, description="List of script executions")

    # Statistics
    total_commands: int = Field(0, ge=0, description="Total number of commands executed")
    total_scripts: int = Field(0, ge=0, description="Total number of scripts executed")
    total_errors: int = Field(0, ge=0, description="Total number of errors encountered")

    def add_error(self, error_type: str, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Add an error to the session history."""
        error = SessionError(
            error_type=error_type,
            message=message,
            context=context
        )
        self.errors.append(error)
        self.total_errors += 1
        self.status = "error"
        self.updated_at = datetime.now(timezone.utc)

    def add_input(self, command: str, input_type: str = "command", script_file: Optional[str] = None) -> str:
        """Add an input to the session history. Returns input ID."""
        input_record = SessionInput(
            command=command,
            input_type=input_type,
            script_file=script_file
        )
        self.inputs.append(input_record)
        if input_type == "command":
            self.total_commands += 1
        self.updated_at = datetime.now(timezone.utc)
        return str(len(self.inputs) - 1)  # Return index as ID

    def add_output(self, text: str, related_input_id: Optional[str] = None) -> None:
        """Add output to the session history."""
        lines_count = len(text.split('\n')) if text else 0
        output = SessionOutput(
            text=text,
            lines_count=lines_count,
            related_input_id=related_input_id
        )
        self.outputs.append(output)
        self.updated_at = datetime.now(timezone.utc)

    def add_script(self, script_content: str, command_template: str, script_file: str, execution_command: str) -> None:
        """Add script execution to the session history."""
        script = SessionScript(
            script_content=script_content,
            command_template=command_template,
            script_file=script_file,
            execution_command=execution_command
        )
        self.scripts.append(script)
        self.total_scripts += 1
        self.updated_at = datetime.now(timezone.utc)

    def update_status(self, status: str) -> None:
        """Update session status and timestamp."""
        self.status = status
        self.updated_at = datetime.now(timezone.utc)

    def get_recent_outputs(self, count: int = 10) -> List[SessionOutput]:
        """Get the most recent outputs."""
        return self.outputs[-count:] if self.outputs else []

    def get_recent_errors(self, count: int = 5) -> List[SessionError]:
        """Get the most recent errors."""
        return self.errors[-count:] if self.errors else []

    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the session."""
        duration = None
        if self.status in ["stopped", "error"]:
            duration = (self.updated_at - self.created_at).total_seconds()

        return {
            "session_id": self.session_id,
            "tool_name": self.tool_config.name,
            "status": self.status,
            "duration_seconds": duration,
            "total_commands": self.total_commands,
            "total_scripts": self.total_scripts,
            "total_errors": self.total_errors,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }