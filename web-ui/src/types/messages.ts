// Session states
export type SessionState =
  | 'idle'
  | 'loading_config'
  | 'running'
  | 'awaiting_user_input'
  | 'complete'
  | 'error';

// Decision types
export type DecisionType = 'initial' | 'regenerate' | 'finish';

// Thinking phases
export type ThinkingPhase = 'initial' | 'multi_turn_decision';

// ============================================================================
// WebSocket Events (Server -> Client)
// ============================================================================

export interface WSEventBase {
  type: string;
  timestamp: string;
  session_id: string;
}

export interface EnvironmentInitEvent extends WSEventBase {
  type: 'environment_init';
  status: string; // "starting", "resetting", "complete", "building_description", "description_complete"
  message: string | null;
  description_content: string | null;
}

export interface ModelThinkingEvent extends WSEventBase {
  type: 'model_thinking';
  phase: ThinkingPhase;
}

export interface ModelStreamingStartEvent extends WSEventBase {
  type: 'model_streaming_start';
  phase: ThinkingPhase;
  turn_number?: number;
  model_name?: string;
}

export interface ModelStreamingDeltaEvent extends WSEventBase {
  type: 'model_streaming_delta';
  content_delta: string | null;
  reasoning_delta: string | null;
}

export interface ModelStreamingEndEvent extends WSEventBase {
  type: 'model_streaming_end';
  content: string;
  reasoning: string | null;
  code_blocks: string[];
  decision: DecisionType;
}

export interface ModelResponseEvent extends WSEventBase {
  type: 'model_response';
  content: string;
  reasoning: string | null;
  code_blocks: string[];
  decision: DecisionType;
}

export interface CodeExecutionStartEvent extends WSEventBase {
  type: 'code_execution_start';
  block_index: number;
  code: string;
}

export interface CodeExecutionResultEvent extends WSEventBase {
  type: 'code_execution_result';
  block_index: number;
  success: boolean;
  stdout: string;
  stderr: string;
  reward: number;
  task_completed: boolean | null;
}

export interface VisualFeedbackEvent extends WSEventBase {
  type: 'visual_feedback';
  image_base64: string;
  description?: string;
}

export interface ImageAnalysisEvent extends WSEventBase {
  type: 'image_analysis';
  analysis_type: 'initial_description' | 'state_comparison';
  content: string;
  model_used?: string;
}

export interface ExecutionStepEvent extends WSEventBase {
  type: 'execution_step';
  block_index: number;
  step_index: number;
  tool_name: string;
  text: string;
  images: string[];  // Base64 encoded images
  highlight?: boolean;  // If true, display with highlighted color scheme
}

export interface UserPromptRequestEvent extends WSEventBase {
  type: 'user_prompt_request';
  current_state_summary: string;
  executed_code_blocks: number;
}

export interface TrialCompleteEvent extends WSEventBase {
  type: 'trial_complete';
  success: boolean;
  total_reward: number;
  task_completed: boolean | null;
  num_regenerations: number;
  num_code_blocks: number;
  summary: string;
}

export interface StateUpdateEvent extends WSEventBase {
  type: 'state_update';
  state: SessionState;
}

export interface ErrorEvent extends WSEventBase {
  type: 'error';
  message: string;
  recoverable: boolean;
}

export type WSEvent =
  | EnvironmentInitEvent
  | ModelThinkingEvent
  | ModelStreamingStartEvent
  | ModelStreamingDeltaEvent
  | ModelStreamingEndEvent
  | ModelResponseEvent
  | CodeExecutionStartEvent
  | CodeExecutionResultEvent
  | VisualFeedbackEvent
  | ExecutionStepEvent
  | ImageAnalysisEvent
  | UserPromptRequestEvent
  | TrialCompleteEvent
  | StateUpdateEvent
  | ErrorEvent;

// ============================================================================
// WebSocket Commands (Client -> Server)
// ============================================================================

export interface InjectPromptCommand {
  type: 'inject_prompt';
  text: string;
}

export interface StopCommand {
  type: 'stop';
}

export interface ResumeCommand {
  type: 'resume';
}

export interface UpdateSettingsCommand {
  type: 'update_settings';
  await_user_input_each_turn?: boolean;
}

export type WSCommand = InjectPromptCommand | StopCommand | ResumeCommand | UpdateSettingsCommand;

// ============================================================================
// REST API Types
// ============================================================================

export interface ConfigListResponse {
  configs: string[];
}

export interface LoadConfigRequest {
  config_path: string;
}

export interface LoadConfigResponse {
  status: string;
  config_summary: Record<string, unknown>;
  task_prompt: string | null;
}

export interface StartTrialRequest {
  config_path: string;
  model?: string;
  server_url?: string;
  temperature?: number;
  max_tokens?: number;
  use_visual_feedback?: boolean | null;
  use_img_differencing?: boolean | null;
  visual_differencing_model?: string | null;
  visual_differencing_model_server_url?: string | null;
  await_user_input_each_turn?: boolean;
}

export interface StartTrialResponse {
  session_id: string;
  status: string;
}

// ============================================================================
// Chat Message Types (for UI rendering)
// ============================================================================

export type ChatMessageType =
  | 'system'
  | 'environment_init'
  | 'model_thinking'
  | 'model_streaming'
  | 'model_response'
  | 'code_execution'
  | 'execution_step'
  | 'visual_feedback'
  | 'image_analysis'
  | 'user_prompt'
  | 'error'
  | 'completion';

// Execution step data structure for execution_step messages
export interface ExecutionStepData {
  toolName: string;
  text: string;
  images: string[];  // Base64 encoded images
  stepIndex: number;
  highlight?: boolean;  // If true, display with highlighted color scheme
}

export interface ChatMessage {
  id: string;
  type: ChatMessageType;
  timestamp: string;
  content?: string;
  reasoning?: string | null;
  codeBlocks?: string[];
  decision?: DecisionType;
  blockIndex?: number;
  success?: boolean;
  stdout?: string;
  stderr?: string;
  reward?: number;
  taskCompleted?: boolean | null;
  imageBase64?: string;
  summary?: string;
  error?: string;
  isExecuting?: boolean;
  analysisType?: 'initial_description' | 'state_comparison';  // for image_analysis messages
  modelUsed?: string;  // for image_analysis and model_streaming messages
  turnNumber?: number;  // for multi-turn tracking
  isStreaming?: boolean;
  thinkingPhase?: ThinkingPhase;
  initStatus?: string;  // for environment_init messages
  descriptionContent?: string;  // for environment description
  startTime?: number;  // timestamp when generation started (for timer)
  endTime?: number;  // timestamp when generation ended
  // Execution step fields
  executionSteps?: ExecutionStepData[];  // Array of steps for the current code block
  toolName?: string;  // For individual execution step (single event)
  stepImages?: string[];  // Images for execution step
}
