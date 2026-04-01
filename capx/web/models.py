"""Pydantic models for WebSocket messages and REST API."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ============================================================================
# Enums
# ============================================================================


class SessionState(str, Enum):
    """Possible states of a trial session."""

    IDLE = "idle"
    LOADING_CONFIG = "loading_config"
    RUNNING = "running"
    AWAITING_USER_INPUT = "awaiting_user_input"
    COMPLETE = "complete"
    ERROR = "error"


class DecisionType(str, Enum):
    """Model decision types in multi-turn execution."""

    INITIAL = "initial"
    REGENERATE = "regenerate"
    FINISH = "finish"


class ThinkingPhase(str, Enum):
    """Phase of model thinking."""

    INITIAL = "initial"
    MULTI_TURN = "multi_turn_decision"


# ============================================================================
# WebSocket Event Models (Server -> Client)
# ============================================================================


class WSEventBase(BaseModel):
    """Base model for all WebSocket events."""

    type: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    session_id: str


class EnvironmentInitEvent(WSEventBase):
    """Emitted when environment initialization starts/completes."""

    type: str = "environment_init"
    status: str  # "starting", "complete", "building_description", "description_complete"
    message: str | None = None
    description_content: str | None = None  # The actual description text (for description_complete)


class ModelThinkingEvent(WSEventBase):
    """Emitted when the model starts generating (non-streaming)."""

    type: str = "model_thinking"
    phase: ThinkingPhase


class ModelStreamingStartEvent(WSEventBase):
    """Emitted when streaming model response starts."""

    type: str = "model_streaming_start"
    phase: ThinkingPhase
    turn_number: int = 0  # 0 for initial, 1+ for multi-turn
    model_name: str | None = None


class ModelStreamingDeltaEvent(WSEventBase):
    """Emitted for each chunk of streaming content."""

    type: str = "model_streaming_delta"
    content_delta: str | None = None
    reasoning_delta: str | None = None


class ModelStreamingEndEvent(WSEventBase):
    """Emitted when streaming completes."""

    type: str = "model_streaming_end"
    content: str
    reasoning: str | None = None
    code_blocks: list[str] = Field(default_factory=list)
    decision: DecisionType


class ModelResponseEvent(WSEventBase):
    """Emitted when the model responds with code."""

    type: str = "model_response"
    content: str
    reasoning: str | None = None
    code_blocks: list[str] = Field(default_factory=list)
    decision: DecisionType


class CodeExecutionStartEvent(WSEventBase):
    """Emitted when code block execution begins."""

    type: str = "code_execution_start"
    block_index: int
    code: str


class CodeExecutionResultEvent(WSEventBase):
    """Emitted when code block execution completes."""

    type: str = "code_execution_result"
    block_index: int
    success: bool
    stdout: str
    stderr: str
    reward: float
    task_completed: bool | None = None


class ExecutionStepEvent(WSEventBase):
    """Emitted for each detailed execution step during code execution.

    Used to show real-time feedback about API calls, IK planning,
    camera captures, etc. during code execution.
    """

    type: str = "execution_step"
    block_index: int
    step_index: int
    tool_name: str  # e.g., "SAM3 Segmentation", "IK Planning", "Camera Capture"
    text: str  # Description text (supports markdown)
    images: list[str] = Field(default_factory=list)  # Base64 encoded images
    highlight: bool = False  # If True, display with highlighted color scheme


class VisualFeedbackEvent(WSEventBase):
    """Emitted when visual feedback is captured."""

    type: str = "visual_feedback"
    image_base64: str
    description: str | None = None


class ImageAnalysisEvent(WSEventBase):
    """Emitted when image differencing/analysis is performed."""

    type: str = "image_analysis"
    analysis_type: str  # "initial_description" or "state_comparison"
    content: str
    model_used: str | None = None


class UserPromptRequestEvent(WSEventBase):
    """Emitted when waiting for user input."""

    type: str = "user_prompt_request"
    current_state_summary: str
    executed_code_blocks: int


class TrialCompleteEvent(WSEventBase):
    """Emitted when trial finishes."""

    type: str = "trial_complete"
    success: bool
    total_reward: float
    task_completed: bool | None = None
    num_regenerations: int
    num_code_blocks: int
    summary: str


class StateUpdateEvent(WSEventBase):
    """Emitted when session state changes."""

    type: str = "state_update"
    state: SessionState


class ErrorEvent(WSEventBase):
    """Emitted when an error occurs."""

    type: str = "error"
    message: str
    recoverable: bool = True


# ============================================================================
# WebSocket Command Models (Client -> Server)
# ============================================================================


class WSCommandBase(BaseModel):
    """Base model for all WebSocket commands."""

    type: str


class InjectPromptCommand(WSCommandBase):
    """User injects text into the next prompt."""

    type: str = "inject_prompt"
    text: str


class StopCommand(WSCommandBase):
    """User requests to stop the trial."""

    type: str = "stop"


class ResumeCommand(WSCommandBase):
    """User resumes after input."""

    type: str = "resume"


# ============================================================================
# REST API Models
# ============================================================================


class ConfigListResponse(BaseModel):
    """Response for listing available configs."""

    configs: list[str]


class LoadConfigRequest(BaseModel):
    """Request to load a YAML config."""

    config_path: str


class LoadConfigResponse(BaseModel):
    """Response after loading config."""

    status: str
    config_summary: dict[str, Any] = Field(default_factory=dict)
    task_prompt: str | None = None


class StartTrialRequest(BaseModel):
    """Request to start a trial."""

    config_path: str
    model: str = "google/gemini-3.1-pro-preview"
    server_url: str = "http://127.0.0.1:8110/chat/completions"
    temperature: float = 1.0
    max_tokens: int = 20480
    use_visual_feedback: bool | None = None
    use_img_differencing: bool | None = None
    visual_differencing_model: str | None = "google/gemini-3.1-pro-preview"
    visual_differencing_model_server_url: str | None = "http://127.0.0.1:8110/chat/completions"
    await_user_input_each_turn: bool = False
    execution_timeout: int = 180  # seconds per code block execution


class StartTrialResponse(BaseModel):
    """Response after starting trial."""

    session_id: str
    status: str


class StopTrialRequest(BaseModel):
    """Request to stop a trial."""

    session_id: str


class StopTrialResponse(BaseModel):
    """Response after stopping trial."""

    status: str


class SessionStatusResponse(BaseModel):
    """Response for session status query."""

    session_id: str
    state: SessionState
    current_block_index: int = 0
    total_code_blocks: int = 0
    num_regenerations: int = 0
