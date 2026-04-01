"""Async trial runner for interactive web UI."""

from __future__ import annotations

import asyncio
import copy
import gc
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

from capx.envs.configs.instantiate import instantiate
from capx.llm.client import (
    VLM_MODELS,
    ModelQueryArgs,
    query_model as _query_model,
    query_model_streaming as _query_model_streaming,
)
from capx.utils.launch_utils import (
    TrialSummary,
    _build_multi_turn_decision_prompt,
    _extract_code,
    _get_visual_feedback,
    _parse_multi_turn_decision,
    _save_trial_artifacts,
)
from capx.utils.video_utils import _write_video
from capx.web.models import (
    CodeExecutionResultEvent,
    CodeExecutionStartEvent,
    DecisionType,
    EnvironmentInitEvent,
    ErrorEvent,
    ExecutionStepEvent,
    ImageAnalysisEvent,
    ModelResponseEvent,
    ModelStreamingDeltaEvent,
    ModelStreamingEndEvent,
    ModelStreamingStartEvent,
    ModelThinkingEvent,
    SessionState,
    StateUpdateEvent,
    ThinkingPhase,
    TrialCompleteEvent,
    UserPromptRequestEvent,
    VisualFeedbackEvent,
    WSEventBase,
)
from capx.utils import execution_logger
from capx.web.session_manager import Session, run_blocking_with_interrupt

logger = logging.getLogger(__name__)

MULTITURN_LIMIT = 30


@dataclass
class LaunchArgsCompat:
    """Compatible args structure for _query_model."""

    model: str
    server_url: str
    api_key: str | None
    max_tokens: int
    temperature: float
    reasoning_effort: str
    debug: bool

    # Image differencing
    visual_differencing_model: str | None
    visual_differencing_model_server_url: str | None
    visual_differencing_model_api_key: str | None


async def run_trial_async(
    session: Session,
    args: LaunchArgsCompat,
) -> TrialSummary | None:
    """Run a single trial asynchronously with WebSocket event emission.

    Args:
        session: The session containing config, env_factory, and WebSocket connections.
            The session.await_user_input_each_turn setting can be toggled during execution.
        args: Model query arguments.

    Returns:
        TrialSummary on completion, None if cancelled.
    """
    trial_start_time = time.time()
    trial = 1  # Interactive mode runs one trial at a time

    # Helper to emit events
    async def emit(event: WSEventBase) -> None:
        await session.emit(event)

    # Helper to check cancellation
    def is_cancelled() -> bool:
        return session.cancel_event.is_set()

    try:
        # Clear any previous execution histories at trial start
        execution_logger.clear_all_histories()

        # Wait for WebSocket connection before proceeding
        # This ensures early events aren't lost
        logger.info(f"Waiting for WebSocket connection for session {session.session_id}")
        for _ in range(50):  # Wait up to 5 seconds
            if session.websockets:
                logger.info(f"WebSocket connected for session {session.session_id}")
                break
            await asyncio.sleep(0.1)
        else:
            logger.warning(f"No WebSocket connected after 5s for session {session.session_id}")

        # Update state
        session.state = SessionState.RUNNING
        await emit(StateUpdateEvent(
            session_id=session.session_id,
            state=SessionState.RUNNING,
        ))

        # Instantiate environment
        await emit(EnvironmentInitEvent(
            session_id=session.session_id,
            status="starting",
            message="Initializing environment...",
        ))
        # Force enable_render and viser for the web UI so users get 3D visualization
        if "cfg" in session.env_factory:
            session.env_factory["cfg"]["enable_render"] = True
            session.env_factory["cfg"]["viser_debug"] = True

        # Use a single-worker thread pool for ALL env operations so that MuJoCo's
        # thread-local osmesa GL context is always available for rendering.
        import concurrent.futures
        env_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="env")
        loop = asyncio.get_running_loop()

        async def run_in_env_thread(func, *args):
            return await loop.run_in_executor(env_executor, func, *args)

        logger.info(f"Instantiating environment for session {session.session_id}")
        env = await run_in_env_thread(instantiate, session.env_factory)

        # Store env reference in session for safety interrupt
        session.env = env

        # Enable web UI logging on all API instances
        if hasattr(env, "_apis"):
            for api in env._apis.values():
                api.enable_webui(True)
            logger.info(f"Enabled web UI logging on {len(env._apis)} API(s)")

        # Get prompts from config
        multi_turn_prompt = session.env_factory["cfg"].get("multi_turn_prompt", None)
        task_only_prompt = session.env_factory["cfg"].get("task_only_prompt", None)

        # Reset environment
        await emit(EnvironmentInitEvent(
            session_id=session.session_id,
            status="resetting",
            message="Resetting environment...",
        ))
        # Reset and capture initial frame in the same thread to avoid
        # MuJoCo OpenGL context issues (osmesa contexts are thread-local).
        def _reset_and_render():
            obs, info = env.reset()
            frame = env.render() if hasattr(env, "render") else None
            return obs, info, frame

        obs, _, initial_frame = await run_in_env_thread(_reset_and_render)
        obs["full_prompt"] = copy.deepcopy(obs["full_prompt"])

        # Patch LIBERO task language into prompt template
        from capx.envs.trial import _patch_libero_goal
        _patch_libero_goal(env, obs)

        # Extract the actual (substituted) task prompt for the UI
        actual_task_prompt = None
        try:
            prompt_content = obs["full_prompt"][-1]["content"]
            if isinstance(prompt_content, list):
                actual_task_prompt = prompt_content[0].get("text", "")
            elif isinstance(prompt_content, str):
                actual_task_prompt = prompt_content
        except (KeyError, IndexError):
            pass

        await emit(EnvironmentInitEvent(
            session_id=session.session_id,
            status="complete",
            message="Environment ready",
            description_content=actual_task_prompt,
        ))

        # Enable video capture if configured
        if session.config.get("record_video") and hasattr(env, "enable_video_capture"):
            env.enable_video_capture(True, clear=True)

        # Initialize tracking variables
        raw_code = None
        code_blocks: list[str] = []
        code_block_metadata: list[dict] = []
        code_block_idx = 0
        num_regenerations = 0
        num_finishes = 0
        all_responses: list[dict] = []
        visual_feedback_imgs = []
        visual_feedback_base64_history: list[str] = []
        stderr_history: list[str] = []

        info_step = {"sandbox_rc": -1, "stdout": "", "stderr": "", "task_completed": False}
        reward = 0.0
        terminated = truncated = False

        # Build image differencing args if needed
        visual_differencing_args = ModelQueryArgs(
            model=args.visual_differencing_model,
            server_url=args.visual_differencing_model_server_url,
            api_key=args.visual_differencing_model_api_key,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            reasoning_effort=args.reasoning_effort,
            debug=args.debug,
        )

        use_visual_feedback = session.config.get("use_visual_feedback", False)
        use_img_differencing = session.config.get("use_img_differencing", False)

        # Build initial visual feedback from the frame captured in the reset thread
        initial_visual_feedback_base64 = None
        if (
            (use_visual_feedback and args.model in VLM_MODELS)
            or (use_img_differencing and visual_differencing_args.model in VLM_MODELS)
        ) and initial_frame is not None:
            from PIL import Image
            initial_visual_feedback_img = Image.fromarray(initial_frame)
            visual_feedback_imgs.append(initial_visual_feedback_img)
            import io as _io, base64 as _b64
            buf = _io.BytesIO()
            initial_visual_feedback_img.save(buf, format="png")
            initial_visual_feedback_base64 = f"data:image/png;base64,{_b64.b64encode(buf.getvalue()).decode('utf-8')}"
            if initial_visual_feedback_img:
                visual_feedback_imgs.append(initial_visual_feedback_img)
            if initial_visual_feedback_base64:
                visual_feedback_base64_history.append(initial_visual_feedback_base64)

                # Emit initial visual feedback
                await emit(VisualFeedbackEvent(
                    session_id=session.session_id,
                    image_base64=initial_visual_feedback_base64,
                    description="Initial environment state",
                ))

        # Add visual feedback to prompt if enabled
        if use_visual_feedback and initial_visual_feedback_base64:
            obs["full_prompt"][-1]["content"][0]["text"] += (
                "\n\nIncluded below is an image of the initial state of the environment."
            )
            obs["full_prompt"][-1]["content"].append(
                {"type": "image_url", "image_url": {"url": initial_visual_feedback_base64}}
            )

        # Handle image differencing for initial state
        # Use task_only_prompt if provided, otherwise extract from the full prompt
        # (matches launch.py behaviour which deep-copies the prompt text)
        task_description = task_only_prompt or copy.deepcopy(
            obs["full_prompt"][-1]["content"][0]["text"]
        )
        if use_img_differencing and initial_visual_feedback_base64:

            await emit(EnvironmentInitEvent(
                session_id=session.session_id,
                status="building_description",
                message="Building initial environment description...",
            ))

            initial_env_description_prompt = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that describes the initial state of the environment with the goal of the task in mind. Do *NOT* write any code. Provide ONLY task-relevant information.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": task_description},
                        {"type": "text", "text": "Describe the initial state of the environment with the goal of the task in mind. Do *NOT* write any code. Provide ONLY task-relevant information."},
                        {"type": "image_url", "image_url": {"url": initial_visual_feedback_base64}},
                    ],
                },
            ]

            initial_env_description_out = await asyncio.to_thread(
                _query_model, visual_differencing_args, initial_env_description_prompt
            )
            initial_env_description = initial_env_description_out["content"]
            initial_visual_differencing_feedback = f"The initial state of the environment is described as follows:\n{initial_env_description}"
            obs["full_prompt"][-1]["content"][0]["text"] += f"\n\n{initial_visual_differencing_feedback}"

            await emit(EnvironmentInitEvent(
                session_id=session.session_id,
                status="description_complete",
                message="Environment description ready",
                description_content=initial_env_description,
            ))

        # Display the prompt to the user before generation starts
        prompt_text = ""
        for msg in obs["full_prompt"]:
            content = msg.get("content", "")
            if isinstance(content, list):
                content = "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
            if content:
                prompt_text += f"**[{msg.get('role', 'unknown')}]**\n{content}\n\n"
        if prompt_text:
            await emit(EnvironmentInitEvent(
                session_id=session.session_id,
                status="description_complete",
                message="Task prompt",
                description_content=prompt_text.strip(),
            ))

        if is_cancelled():
            raise asyncio.CancelledError("Cancelled before initial code generation")

        # ========================================================================
        # Initial code generation (with streaming)
        # ========================================================================
        turn_number = 1  # Track multi-turn iterations (starts at 1)
        await emit(ModelStreamingStartEvent(
            session_id=session.session_id,
            phase=ThinkingPhase.INITIAL,
            turn_number=turn_number,
            model_name=args.model,
        ))

        logger.info("Querying model for initial code generation (streaming)")

        # Use queue for true streaming from thread to async
        chunk_queue: asyncio.Queue = asyncio.Queue()

        # Capture event loop BEFORE starting thread (critical!)
        main_loop = asyncio.get_running_loop()

        def stream_to_queue():
            """Run streaming query and put chunks in queue."""
            try:
                for chunk in _query_model_streaming(args, obs["full_prompt"]):
                    # Use the captured loop reference, not get_event_loop()
                    asyncio.run_coroutine_threadsafe(
                        chunk_queue.put(chunk),
                        main_loop
                    )
            except Exception as e:
                asyncio.run_coroutine_threadsafe(
                    chunk_queue.put({"type": "error", "error": str(e)}),
                    main_loop
                )

        # Start streaming in background thread
        stream_task = main_loop.run_in_executor(None, stream_to_queue)

        raw_code = ""
        reasoning = None

        # Process chunks as they arrive
        while True:
            if is_cancelled():
                raise asyncio.CancelledError("Cancelled during initial code generation")

            try:
                chunk = await asyncio.wait_for(chunk_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                # Check if thread is still running
                if stream_task.done():
                    break
                continue

            if chunk["type"] == "content_delta":
                await emit(ModelStreamingDeltaEvent(
                    session_id=session.session_id,
                    content_delta=chunk["content"],
                ))
            elif chunk["type"] == "reasoning_delta":
                await emit(ModelStreamingDeltaEvent(
                    session_id=session.session_id,
                    reasoning_delta=chunk["content"],
                ))
            elif chunk["type"] == "done":
                raw_code = chunk["content"]
                reasoning = chunk.get("reasoning")
                break
            elif chunk["type"] == "error":
                raise RuntimeError(f"Streaming error: {chunk['error']}")

        # Wait for thread to complete
        await stream_task

        if not raw_code:
            raise ValueError("Model returned empty response during streaming")

        initial_blocks = _extract_code(raw_code)
        code_blocks.extend(initial_blocks)
        code_block_metadata.extend([{"generation": 0, "regenerated": False}] * len(initial_blocks))

        session.total_code_blocks = len(code_blocks)

        await emit(ModelStreamingEndEvent(
            session_id=session.session_id,
            content=raw_code,
            reasoning=reasoning,
            code_blocks=initial_blocks,
            decision=DecisionType.INITIAL,
        ))

        all_responses.append({
            "block_idx": [code_block_idx],
            "code_blocks": initial_blocks,
            "decision": "initial",
            "reasoning": reasoning if reasoning else "",
        })

        # ========================================================================
        # Code execution loop
        # ========================================================================
        while code_block_idx < len(code_blocks) and code_block_idx <= MULTITURN_LIMIT:
            if is_cancelled():
                raise asyncio.CancelledError("Cancelled during code execution")

            code = code_blocks[code_block_idx]
            session.current_block_index = code_block_idx

            # Check for cancellation before executing (safety check)
            if is_cancelled():
                raise asyncio.CancelledError("Cancelled before code execution")

            await emit(CodeExecutionStartEvent(
                session_id=session.session_id,
                block_index=code_block_idx,
                code=code,
            ))

            logger.info(f"Executing code block {code_block_idx}")

            # Set up execution logger to capture detailed execution steps
            # The emit callback sends events via WebSocket in real-time
            step_counter = [0]  # Use list to allow mutation in closure

            # Capture event loop BEFORE thread execution (critical!)
            # The callback runs in a thread pool executor, so we need the main loop reference
            exec_main_loop = asyncio.get_running_loop()

            def emit_execution_step(step: execution_logger.ExecutionStep) -> None:
                """Sync callback to emit execution steps during code execution."""
                try:
                    # Create and emit the event by scheduling in the captured main event loop
                    event = ExecutionStepEvent(
                        session_id=session.session_id,
                        block_index=code_block_idx,
                        step_index=step.step_index,
                        tool_name=step.tool_name,
                        text=step.text,
                        images=step.images,
                        highlight=step.highlight,
                    )
                    asyncio.run_coroutine_threadsafe(session.emit(event), exec_main_loop)
                    step_counter[0] += 1
                except Exception as e:
                    logger.warning(f"Failed to emit execution step: {e}")

            # Initialize execution logger for this code block
            execution_logger.init_execution_context(
                code_block_index=code_block_idx,
                emit_callback=emit_execution_step,
            )

            try:
                # Run step + render in the same thread to keep MuJoCo GL context.
                # We capture the frame here because MuJoCo's osmesa GL context is
                # thread-local — rendering must happen on the same thread as the
                # simulation step.
                def _step_and_render(code):
                    result = env.step(code)
                    try:
                        frame = env.render() if hasattr(env, "render") else None
                    except Exception:
                        frame = None
                    return result, frame

                def _step_render_with_interrupt(code):
                    session.execution_thread_id = threading.get_ident()
                    try:
                        return _step_and_render(code)
                    finally:
                        session.execution_thread_id = None

                exec_timeout = getattr(session, 'execution_timeout', 180)
                try:
                    (obs_next, reward, terminated, truncated, info_step), post_step_frame = await asyncio.wait_for(
                        run_in_env_thread(_step_render_with_interrupt, code),
                        timeout=exec_timeout,
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Code block {code_block_idx} timed out after {exec_timeout}s")
                    await emit(CodeExecutionResultEvent(
                        session_id=session.session_id,
                        block_index=code_block_idx,
                        success=False,
                        stdout="",
                        stderr=f"Execution timed out after {exec_timeout} seconds. The code may be stuck in a loop or waiting for an unreachable target.",
                        reward=0.0,
                        task_completed=False,
                    ))
                    # Reset env for next attempt
                    try:
                        obs, _ = await run_in_env_thread(lambda: env.reset())
                    except Exception:
                        pass
                    break  # Exit code block loop, go to multi-turn decision
            finally:
                # Finalize execution logger and get history
                exec_history = execution_logger.finalize_execution_context()
                if exec_history and step_counter[0] > 0:
                    logger.info(f"Code block {code_block_idx} had {len(exec_history.steps)} execution steps")

            # Check for cancellation after executing (might have been stopped during execution)
            if is_cancelled():
                raise asyncio.CancelledError("Cancelled after code execution")

            await emit(CodeExecutionResultEvent(
                session_id=session.session_id,
                block_index=code_block_idx,
                success=info_step["sandbox_rc"] == 0,
                stdout=info_step["stdout"],
                stderr=info_step["stderr"],
                reward=reward,
                task_completed=info_step.get("task_completed"),
            ))

            code_block_idx += 1
            obs = obs_next

            # ====================================================================
            # Multi-turn decision
            # ====================================================================
            if multi_turn_prompt:
                # Compute the executed and remaining code blocks
                # import pdb; pdb.set_trace()
                executed_code = "# Prior executed code blocks:\n"
                for block_idx in range(code_block_idx):
                    if block_idx < code_block_idx - 1:
                        executed_code += f"# Code block {block_idx}\n{code_blocks[block_idx]}\n"
                    else:
                        executed_code += f"\n\n# Last executed code block (Code block {block_idx}):\n{code_blocks[block_idx]}\n"
                # executed_code = "\n".join(code_blocks[:code_block_idx])
                # remaining_code = "\n".join(code_blocks[code_block_idx:])
                # Check for episode termination
                if "terminated episode" in info_step["stderr"]:
                    truncated = True
                    break

                # Build multi-turn prompt
                complete_multi_turn_prompt = multi_turn_prompt.format(
                    executed_code=executed_code,
                    console_stdout=info_step["stdout"],
                    console_stderr=info_step["stderr"],
                )

                if info_step["stderr"]:
                    stderr_history.append(info_step["stderr"])

                # Build visual feedback from the frame captured in the step thread
                visual_feedback_base64 = None
                if (
                    (use_visual_feedback and args.model in VLM_MODELS)
                    or (use_img_differencing and visual_differencing_args.model in VLM_MODELS)
                ) and post_step_frame is not None:
                    from PIL import Image as _Image
                    import io as _io, base64 as _b64
                    visual_feedback_img = _Image.fromarray(post_step_frame)
                    buf = _io.BytesIO()
                    visual_feedback_img.save(buf, format="png")
                    visual_feedback_base64 = f"data:image/png;base64,{_b64.b64encode(buf.getvalue()).decode('utf-8')}"
                    if visual_feedback_img:
                        visual_feedback_imgs.append(visual_feedback_img)
                    if visual_feedback_base64:
                        visual_feedback_base64_history.append(visual_feedback_base64)

                        await emit(VisualFeedbackEvent(
                            session_id=session.session_id,
                            image_base64=visual_feedback_base64,
                        ))

                # Image differencing
                visual_differencing_feedback = None
                if use_img_differencing and len(visual_feedback_base64_history) >= 2:
                    visual_differencing_prompt = [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that describes the difference between the current state of the environment and the previous state of the environment with the goal of the task in mind and whether the task has been completed. Do *NOT* write any code.",
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": task_description},
                                {"type": "text", "text": "Describe the difference between the current state of the environment and the previous state of the environment with the goal of the task in mind and whether the task has been completed. Do *NOT* write any code.."},
                                {"type": "text", "text": "Previous state:"},
                                {"type": "image_url", "image_url": {"url": visual_feedback_base64_history[-2]}},
                                {"type": "text", "text": "Current state:"},
                                {"type": "image_url", "image_url": {"url": visual_feedback_base64_history[-1]}},
                            ],
                        },
                    ]
                    img_diff_response = await asyncio.to_thread(
                        _query_model, visual_differencing_args, visual_differencing_prompt
                    )
                    visual_differencing_feedback = img_diff_response.get("content") if img_diff_response else None

                    # Emit image analysis event for visualization
                    if visual_differencing_feedback:
                        await emit(ImageAnalysisEvent(
                            session_id=session.session_id,
                            analysis_type="state_comparison",
                            content=visual_differencing_feedback,
                            model_used=visual_differencing_args.model,
                        ))

                # Check for user injection - only pause if explicitly requested
                # Read from session to allow dynamic toggling during trial
                if session.await_user_input_each_turn:
                    # Drain any leftover items from previous resumes to prevent auto-continue
                    while not session.user_injection_queue.empty():
                        try:
                            session.user_injection_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break

                    session.state = SessionState.AWAITING_USER_INPUT
                    await emit(StateUpdateEvent(
                        session_id=session.session_id,
                        state=SessionState.AWAITING_USER_INPUT,
                    ))
                    await emit(UserPromptRequestEvent(
                        session_id=session.session_id,
                        current_state_summary=f"Executed {code_block_idx} of {len(code_blocks)} code blocks. Reward: {reward:.3f}",
                        executed_code_blocks=code_block_idx,
                    ))

                    # Wait for user input with timeout
                    try:
                        user_text = await asyncio.wait_for(
                            session.user_injection_queue.get(),
                            timeout=300.0,  # 5 minute timeout
                        )
                        # Append user text to multi-turn prompt if not empty
                        if user_text:
                            complete_multi_turn_prompt += f"\n\nUser feedback: {user_text}"
                            logger.info(f"User injected: {user_text[:100]}...")
                    except asyncio.TimeoutError:
                        logger.info("User input timeout, continuing without injection")

                    session.state = SessionState.RUNNING
                    await emit(StateUpdateEvent(
                        session_id=session.session_id,
                        state=SessionState.RUNNING,
                    ))

                if is_cancelled():
                    raise asyncio.CancelledError("Cancelled during multi-turn")

                # Zero out visual feedback if not using it
                if not use_visual_feedback:
                    visual_feedback_base64 = None

                # Build and query for decision (with streaming)
                turn_number += 1
                await emit(ModelStreamingStartEvent(
                    session_id=session.session_id,
                    phase=ThinkingPhase.MULTI_TURN,
                    turn_number=turn_number,
                    model_name=args.model,
                ))

                multi_turn_decision_prompt = _build_multi_turn_decision_prompt(
                    obs, complete_multi_turn_prompt, visual_feedback_base64, visual_differencing_feedback
                )

                # Stream multi-turn decision
                mt_chunk_queue: asyncio.Queue = asyncio.Queue()

                # Capture event loop BEFORE starting thread (critical!)
                mt_main_loop = asyncio.get_running_loop()

                def stream_multiturn_to_queue():
                    try:
                        for chunk in _query_model_streaming(args, multi_turn_decision_prompt):
                            asyncio.run_coroutine_threadsafe(
                                mt_chunk_queue.put(chunk),
                                mt_main_loop
                            )
                    except Exception as e:
                        asyncio.run_coroutine_threadsafe(
                            mt_chunk_queue.put({"type": "error", "error": str(e)}),
                            mt_main_loop
                        )

                mt_stream_task = mt_main_loop.run_in_executor(None, stream_multiturn_to_queue)

                mt_content = ""
                mt_reasoning = None

                while True:
                    if is_cancelled():
                        raise asyncio.CancelledError("Cancelled during multi-turn decision")

                    try:
                        chunk = await asyncio.wait_for(mt_chunk_queue.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        if mt_stream_task.done():
                            break
                        continue

                    if chunk["type"] == "content_delta":
                        await emit(ModelStreamingDeltaEvent(
                            session_id=session.session_id,
                            content_delta=chunk["content"],
                        ))
                    elif chunk["type"] == "reasoning_delta":
                        await emit(ModelStreamingDeltaEvent(
                            session_id=session.session_id,
                            reasoning_delta=chunk["content"],
                        ))
                    elif chunk["type"] == "done":
                        mt_content = chunk["content"]
                        mt_reasoning = chunk.get("reasoning")
                        break
                    elif chunk["type"] == "error":
                        raise RuntimeError(f"Streaming error: {chunk['error']}")

                await mt_stream_task

                if not mt_content:
                    logger.warning("Model returned empty response for multi-turn decision, defaulting to finish")
                    decision, new_code = "finish", None
                    reasoning = None
                else:
                    reasoning = mt_reasoning
                    decision, new_code = _parse_multi_turn_decision(mt_content)

                if decision == "regenerate":
                    logger.info("Model chose to regenerate code")
                    new_blocks = _extract_code(new_code)

                    await emit(ModelStreamingEndEvent(
                        session_id=session.session_id,
                        content=mt_content,
                        reasoning=reasoning,
                        code_blocks=new_blocks,
                        decision=DecisionType.REGENERATE,
                    ))

                    all_responses.append({
                        "multi_turn_prompt": multi_turn_decision_prompt,
                        "block_idx": [code_block_idx],
                        "code_blocks": new_blocks,
                        "decision": "regenerate",
                        "reasoning": reasoning if reasoning else "",
                    })

                    # Replace remaining blocks
                    del code_blocks[code_block_idx:]
                    del code_block_metadata[code_block_idx:]
                    code_blocks.extend(new_blocks)
                    code_block_metadata.extend([
                        {"generation": num_regenerations + 1, "regenerated": True, "regenerated_at_idx": code_block_idx}
                    ] * len(new_blocks))

                    num_regenerations += 1
                    session.num_regenerations = num_regenerations
                    session.total_code_blocks = len(code_blocks)

                elif decision == "finish":
                    logger.info("Model chose to finish")

                    await emit(ModelStreamingEndEvent(
                        session_id=session.session_id,
                        content=mt_content,
                        reasoning=reasoning,
                        code_blocks=[],
                        decision=DecisionType.FINISH,
                    ))

                    all_responses.append({
                        "multi_turn_prompt": multi_turn_decision_prompt,
                        "decision": "finish",
                        "reasoning": reasoning if reasoning else new_code if new_code else "",
                    })

                    num_finishes += 1
                    break

            logger.info(f"Code block {code_block_idx} done, {len(code_blocks)} total blocks")

        # ========================================================================
        # Trial complete
        # ========================================================================
        logger.info("Trial execution complete")

        # Build final code with annotations
        annotated_blocks = []
        for i, (block, metadata) in enumerate(zip(code_blocks, code_block_metadata, strict=False)):
            header = f"# Code block {i}"
            if metadata.get("regenerated"):
                header += f" (regenerated at step {metadata.get('regenerated_at_idx', '?')})"
            annotated_blocks.append(f"{header}\n{block}")

        final_code = "\n\n".join(annotated_blocks)

        # Handle sandbox_rc override for max steps
        if "executing action in terminated episode" in info_step.get("stderr", ""):
            info_step["sandbox_rc"] = 0

        stderr = "\n\n".join(stderr_history) if stderr_history else info_step.get("stderr", "")

        log_lines = [
            "-" * 100,
            "Generated program:",
            final_code,
            "\n\nEnvironment response:",
            f"  Sandbox failed: {info_step['sandbox_rc']}",
            f"  Stdout: {info_step['stdout']}",
            f"  Stderr: {stderr}",
            f"  Reward: {reward}",
            f"  Task Completed: {info_step.get('task_completed', 'N/A')}",
            f"  Terminated: {terminated}, Truncated: {truncated}",
            f"  Num Regenerations: {num_regenerations}",
            f"  Num Finishes: {num_finishes}",
            f"  Num Code Blocks: {len(code_blocks)}",
            "-" * 100,
        ]

        # Save artifacts if output_dir configured
        code_path = None
        output_dir = session.config.get("output_dir")
        if output_dir:
            logger.info(f"Saving trial artifacts to: {output_dir}")
            code_path = await asyncio.to_thread(
                _save_trial_artifacts,
                session.config,
                trial,
                info_step["sandbox_rc"],
                reward,
                info_step.get("task_completed", False),
                final_code,
                raw_code,
                all_responses,
                log_lines,
                visual_feedback_imgs,
            )
            logger.info(f"Trial artifacts saved to: {code_path}")

            # Save execution histories
            all_exec_histories = execution_logger.get_all_histories()
            if all_exec_histories:
                from pathlib import Path
                exec_history_dir = Path(output_dir) / "execution_history"
                for history in all_exec_histories:
                    await asyncio.to_thread(history.save_to_directory, exec_history_dir)
                logger.info(f"Saved {len(all_exec_histories)} execution histories to: {exec_history_dir}")

            # Save video if configured
            if session.config.get("record_video") and hasattr(env, "get_video_frames"):
                frames = env.get_video_frames(clear=True)
                if frames:
                    video_dir = os.path.join(
                        output_dir,
                        f"trial_{trial:02d}_sandboxrc_{info_step['sandbox_rc']}_reward_{reward:.3f}_taskcompleted_{int(info_step.get('task_completed', False))}",
                    )
                    await asyncio.to_thread(
                        _write_video,
                        frames,
                        video_dir,
                        suffix=f"{reward:.3f}",
                    )
        else:
            logger.info("No output_dir configured, skipping artifact save")

        # Success if the environment reports task completion or termination with positive reward,
        # OR if the model explicitly chose to finish in multi-turn mode.
        task_completed = bool(info_step.get("task_completed", False)) or (terminated and reward > 0)
        success = task_completed or num_finishes > 0

        # Emit completion
        session.state = SessionState.COMPLETE
        await emit(TrialCompleteEvent(
            session_id=session.session_id,
            success=success,
            total_reward=reward,
            task_completed=task_completed,
            num_regenerations=num_regenerations,
            num_code_blocks=len(code_blocks),
            summary="\n".join(log_lines),
        ))
        await emit(StateUpdateEvent(
            session_id=session.session_id,
            state=SessionState.COMPLETE,
        ))

        trial_end_time = time.time()
        logger.info(f"Trial completed in {trial_end_time - trial_start_time:.2f} seconds")

        # Cleanup
        gc.collect()

        return TrialSummary(
            trial=trial,
            success=success,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            sandbox_rc=info_step["sandbox_rc"],
            log="\n".join(log_lines),
            task_completed=task_completed,
            code_path=code_path,
            num_regenerations=num_regenerations,
            num_finishes=num_finishes,
            num_code_blocks=len(code_blocks),
        )

    except asyncio.CancelledError as e:
        logger.info(f"Trial cancelled: {e}")
        session.state = SessionState.IDLE
        await emit(StateUpdateEvent(
            session_id=session.session_id,
            state=SessionState.IDLE,
        ))
        return None

    except KeyboardInterrupt:
        logger.info("Trial interrupted by user (stop button)")
        session.state = SessionState.IDLE
        await emit(StateUpdateEvent(
            session_id=session.session_id,
            state=SessionState.IDLE,
        ))
        return None

    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        logger.exception(f"Trial error: {e}")
        session.state = SessionState.ERROR
        error_msg = str(e) if str(e) else f"{type(e).__name__}: {tb_str.splitlines()[-2].strip()}"
        await emit(ErrorEvent(
            session_id=session.session_id,
            message=error_msg,
            recoverable=True,
        ))
        await emit(StateUpdateEvent(
            session_id=session.session_id,
            state=SessionState.ERROR,
        ))
        # Don't re-raise - error has been communicated via WebSocket
        # This allows clean recovery without task exception handling issues
        return None
