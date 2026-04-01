"""Session manager for tracking active trial sessions."""

from __future__ import annotations

import asyncio
import ctypes
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Awaitable

from fastapi import WebSocket

from capx.web.models import SessionState, WSEventBase

if TYPE_CHECKING:
    from capx.web.async_trial_runner import TrialContext

logger = logging.getLogger(__name__)


async def run_blocking_with_interrupt(
    session: "Session",
    func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run a blocking function in a thread while tracking the thread ID for interruption.

    This allows the stop_session method to interrupt long-running code execution.
    """
    def wrapper():
        # Store the current thread ID so it can be interrupted
        session.execution_thread_id = threading.get_ident()
        try:
            return func(*args, **kwargs)
        finally:
            session.execution_thread_id = None

    return await asyncio.to_thread(wrapper)


def _raise_exception_in_thread(thread_id: int, exception_type: type) -> bool:
    """Raise an exception in another thread.

    This is a safety mechanism to interrupt code execution.
    Returns True if successful, False otherwise.
    """
    try:
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(thread_id),
            ctypes.py_object(exception_type)
        )
        if res == 0:
            logger.warning(f"Thread {thread_id} not found")
            return False
        elif res > 1:
            # If more than one thread was affected, reset
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(thread_id), None)
            logger.error(f"Multiple threads affected when interrupting {thread_id}")
            return False
        return True
    except Exception as e:
        logger.error(f"Failed to raise exception in thread: {e}")
        return False


@dataclass
class Session:
    """Represents an active trial session."""

    session_id: str
    state: SessionState = SessionState.IDLE
    config_path: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    env_factory: dict[str, Any] | None = None

    # Settings that can be changed during a trial
    await_user_input_each_turn: bool = False
    execution_timeout: int = 180  # seconds per code block

    # Event history for replay on reconnect
    event_history: list[str] = field(default_factory=list)

    # Async coordination
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    user_injection_queue: asyncio.Queue[str] = field(default_factory=asyncio.Queue)

    # Running task reference
    task: asyncio.Task | None = None

    # Environment reference for forced shutdown
    env: Any = None

    # Thread tracking for interruption
    execution_thread_id: int | None = None

    # Connected WebSocket clients
    websockets: list[WebSocket] = field(default_factory=list)

    # Execution state
    current_block_index: int = 0
    total_code_blocks: int = 0
    num_regenerations: int = 0

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    async def emit(self, event: WSEventBase) -> None:
        """Broadcast event to all connected WebSocket clients and store for replay."""
        message = event.model_dump_json()
        # Store for replay on reconnect (skip high-frequency streaming deltas)
        if event.type != "model_streaming_delta":
            self.event_history.append(message)
        disconnected = []

        for ws in self.websockets:
            try:
                await ws.send_text(message)
            except Exception as e:
                logger.warning(f"Failed to send to WebSocket: {e}")
                disconnected.append(ws)

        # Clean up disconnected clients
        for ws in disconnected:
            self.websockets.remove(ws)

    def reset(self) -> None:
        """Reset session state for a new trial."""
        self.state = SessionState.IDLE
        self.cancel_event = asyncio.Event()
        self.user_injection_queue = asyncio.Queue()
        self.event_history = []
        self.task = None
        self.env = None
        self.execution_thread_id = None
        self.current_block_index = 0
        self.total_code_blocks = 0
        self.num_regenerations = 0
        self.started_at = None
        self.completed_at = None


class SessionManager:
    """Manages all active trial sessions.

    Only one session can be active at a time. Creating a new session
    will automatically stop and clean up any existing sessions.
    """

    def __init__(self):
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()

    async def create_session(self) -> Session:
        """Create a new session, stopping any existing sessions first."""
        async with self._lock:
            # Stop and clean up ALL existing sessions first
            for session_id in list(self._sessions.keys()):
                await self._cleanup_session_unlocked(session_id)

            session_id = str(uuid.uuid4())
            session = Session(session_id=session_id)
            self._sessions[session_id] = session
            logger.info(f"Created session: {session_id}")
            return session

    async def _cleanup_session_unlocked(self, session_id: str) -> None:
        """Clean up a session (must be called with lock held)."""
        if session_id not in self._sessions:
            return

        session = self._sessions[session_id]
        logger.info(f"Cleaning up session: {session_id}")

        # Cancel any running task
        if session.task and not session.task.done():
            session.cancel_event.set()
            session.task.cancel()
            try:
                await asyncio.wait_for(session.task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

        # Close all WebSocket connections
        for ws in session.websockets:
            try:
                await ws.close(code=4001, reason="Session replaced")
            except Exception:
                pass

        del self._sessions[session_id]
        logger.info(f"Session cleaned up: {session_id}")

    async def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    async def remove_session(self, session_id: str) -> None:
        """Remove a session."""
        async with self._lock:
            await self._cleanup_session_unlocked(session_id)

    async def stop_session(self, session_id: str) -> bool:
        """Stop a running session immediately.

        This is a safety-critical operation that should interrupt code execution
        as quickly as possible.
        """
        session = await self.get_session(session_id)
        if not session:
            return False

        if session.task and not session.task.done():
            logger.info(f"STOPPING session (safety interrupt): {session_id}")
            session.cancel_event.set()

            # Interrupt the execution thread if code is running
            if session.execution_thread_id is not None:
                logger.info(f"Interrupting execution thread {session.execution_thread_id}")
                if _raise_exception_in_thread(session.execution_thread_id, KeyboardInterrupt):
                    logger.info("Thread interrupt sent successfully")
                else:
                    logger.warning("Thread interrupt failed")

            # Try to close the environment immediately to interrupt any running code
            if session.env is not None:
                try:
                    logger.info(f"Attempting to close environment for session {session_id}")
                    if hasattr(session.env, 'close'):
                        session.env.close()
                    elif hasattr(session.env, 'shutdown'):
                        session.env.shutdown()
                except Exception as e:
                    logger.warning(f"Error closing environment: {e}")

            # Cancel the task immediately (don't wait for graceful shutdown)
            session.task.cancel()
            try:
                await asyncio.wait_for(session.task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

            session.state = SessionState.IDLE
            session.env = None  # Clear env reference
            session.execution_thread_id = None  # Clear thread reference
            logger.info(f"Session {session_id} stopped")
            return True

        return False

    async def inject_prompt(self, session_id: str, text: str) -> bool:
        """Inject user prompt text into a session."""
        session = await self.get_session(session_id)
        if not session:
            return False

        if session.state == SessionState.AWAITING_USER_INPUT:
            await session.user_injection_queue.put(text)
            logger.info(f"Injected prompt into session {session_id}: {text[:50]}...")
            return True

        return False

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all sessions with their status."""
        return [
            {
                "session_id": s.session_id,
                "state": s.state.value,
                "config_path": s.config_path,
                "created_at": s.created_at.isoformat(),
            }
            for s in self._sessions.values()
        ]

    def get_active_session(self) -> Session | None:
        """Get the currently active session (if any).

        Since only one session is allowed at a time, this returns
        the single session if it exists and is still running.
        """
        for session in self._sessions.values():
            if session.state in (SessionState.RUNNING, SessionState.AWAITING_USER_INPUT, SessionState.LOADING_CONFIG):
                return session
        return None

    async def on_websocket_disconnect(self, session_id: str) -> None:
        """Handle WebSocket disconnection.

        If the session has no more connected WebSockets and is still running,
        we'll keep it alive briefly in case of reconnection. If it's complete
        or errored, clean it up.
        """
        session = await self.get_session(session_id)
        if not session:
            return

        # If session is complete/error and no WebSockets, clean up
        if session.state in (SessionState.COMPLETE, SessionState.ERROR, SessionState.IDLE):
            if not session.websockets:
                logger.info(f"Session {session_id} has no connections and is {session.state}, cleaning up")
                await self.remove_session(session_id)


# Global singleton
_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    global _manager
    if _manager is None:
        _manager = SessionManager()
    return _manager
