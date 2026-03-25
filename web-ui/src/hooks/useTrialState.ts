import { useCallback, useState } from 'react';
import type {
  ChatMessage,
  SessionState,
  WSEvent,
  StartTrialRequest,
  LoadConfigResponse,
} from '../types/messages';
import { useWebSocket } from './useWebSocket';

interface TrialState {
  sessionId: string | null;
  state: SessionState;
  messages: ChatMessage[];
  configPath: string | null;
  taskPrompt: string | null;
  error: string | null;
}

interface ActiveSessionResponse {
  session_id: string | null;
  state?: string;
  config_path?: string;
}

interface UseTrialStateReturn extends TrialState {
  isConnected: boolean;
  loadConfig: (configPath: string) => Promise<LoadConfigResponse | null>;
  startTrial: (request: StartTrialRequest) => Promise<boolean>;
  stopTrial: () => void;
  injectPrompt: (text: string) => void;
  resumeTrial: () => void;
  reset: () => void;
  fullReset: () => void;
  checkActiveSession: () => Promise<ActiveSessionResponse | null>;
  reconnectToSession: (sessionId: string, configPath: string) => void;
  updateSettings: (settings: { await_user_input_each_turn?: boolean }) => void;
}

let messageIdCounter = 0;
function generateMessageId(): string {
  return `msg-${Date.now()}-${++messageIdCounter}`;
}

export function useTrialState(): UseTrialStateReturn {
  const [trialState, setTrialState] = useState<TrialState>({
    sessionId: null,
    state: 'idle',
    messages: [],
    configPath: null,
    taskPrompt: null,
    error: null,
  });

  const addMessage = useCallback((message: Omit<ChatMessage, 'id'>) => {
    const fullMessage: ChatMessage = {
      ...message,
      id: generateMessageId(),
    };
    setTrialState((prev) => ({
      ...prev,
      messages: [...prev.messages, fullMessage],
    }));
    return fullMessage.id;
  }, []);

  const updateMessage = useCallback((id: string, updates: Partial<ChatMessage>) => {
    setTrialState((prev) => ({
      ...prev,
      messages: prev.messages.map((m) =>
        m.id === id ? { ...m, ...updates } : m
      ),
    }));
  }, []);
  void updateMessage; // Keep for future use

  const handleWSMessage = useCallback(
    (event: WSEvent) => {
      switch (event.type) {
        case 'state_update':
          setTrialState((prev) => ({ ...prev, state: event.state }));
          break;

        case 'environment_init':
          // Update or add environment init message
          setTrialState((prev) => {
            const messages = [...prev.messages];

            // Check if this is a completion status
            const isComplete = event.status === 'complete' || event.status === 'description_complete';

            // Update taskPrompt if a substituted prompt was sent with the complete event
            const updatedTaskPrompt = (event.status === 'complete' && event.description_content)
              ? event.description_content
              : prev.taskPrompt;

            if (isComplete) {
              // Mark ALL executing env_init messages as complete
              let foundAny = false;
              for (let i = 0; i < messages.length; i++) {
                if (messages[i].type === 'environment_init' && messages[i].isExecuting) {
                  messages[i] = {
                    ...messages[i],
                    content: event.message || messages[i].content,
                    initStatus: event.status,
                    isExecuting: false,
                    descriptionContent: event.description_content || messages[i].descriptionContent,
                  };
                  foundAny = true;
                }
              }
              if (foundAny) {
                return { ...prev, messages, taskPrompt: updatedTaskPrompt };
              }
            } else {
              // For in-progress statuses, update the last executing message or add new
              const existingIdx = messages.findLastIndex(
                (m: ChatMessage) => m.type === 'environment_init' && m.isExecuting
              );
              if (existingIdx >= 0) {
                // Update existing message with new status
                messages[existingIdx] = {
                  ...messages[existingIdx],
                  content: event.message || messages[existingIdx].content,
                  initStatus: event.status,
                };
                return { ...prev, messages };
              }
            }

            // Add new message if no existing one to update
            messages.push({
              id: `msg-${Date.now()}-${++messageIdCounter}`,
              type: 'environment_init',
              timestamp: event.timestamp,
              content: event.message || 'Initializing...',
              initStatus: event.status,
              isExecuting: !isComplete,
              descriptionContent: event.description_content || undefined,
            });
            return { ...prev, messages };
          });
          break;

        case 'model_thinking':
          addMessage({
            type: 'model_thinking',
            timestamp: event.timestamp,
            thinkingPhase: event.phase,
            isExecuting: true,
          });
          break;

        case 'model_streaming_start':
          // Create a new streaming/generating message
          addMessage({
            type: 'model_streaming',
            timestamp: event.timestamp,
            thinkingPhase: event.phase,
            isStreaming: true,
            content: '',
            reasoning: '',
            startTime: Date.now(),
            turnNumber: event.turn_number,
            modelUsed: event.model_name,
          });
          break;

        case 'model_streaming_delta':
          // Update the streaming message with new content
          setTrialState((prev) => {
            const messages = [...prev.messages];
            const streamingIdx = messages.findLastIndex(
              (m: ChatMessage) => m.type === 'model_streaming' && m.isStreaming
            );
            if (streamingIdx >= 0) {
              const msg = messages[streamingIdx];
              messages[streamingIdx] = {
                ...msg,
                content: (msg.content || '') + (event.content_delta || ''),
                reasoning: (msg.reasoning || '') + (event.reasoning_delta || ''),
              };
            }
            return { ...prev, messages };
          });
          break;

        case 'model_streaming_end':
          // Finalize the streaming message
          setTrialState((prev) => {
            const messages = [...prev.messages];
            const streamingIdx = messages.findLastIndex(
              (m: ChatMessage) => m.type === 'model_streaming' && m.isStreaming
            );
            if (streamingIdx >= 0) {
              messages[streamingIdx] = {
                ...messages[streamingIdx],
                type: 'model_response',
                isStreaming: false,
                content: event.content,
                reasoning: event.reasoning,
                codeBlocks: event.code_blocks,
                decision: event.decision,
                endTime: Date.now(),
              };
            }
            return { ...prev, messages };
          });
          break;

        case 'model_response':
          // Remove the last thinking message (for non-streaming responses)
          setTrialState((prev) => ({
            ...prev,
            messages: prev.messages.filter(
              (m) => !(m.type === 'model_thinking' && m.isExecuting)
            ),
          }));
          addMessage({
            type: 'model_response',
            timestamp: event.timestamp,
            content: event.content,
            reasoning: event.reasoning,
            codeBlocks: event.code_blocks,
            decision: event.decision,
          });
          break;

        case 'code_execution_start':
          addMessage({
            type: 'code_execution',
            timestamp: event.timestamp,
            blockIndex: event.block_index,
            codeBlocks: [event.code],
            isExecuting: true,
            executionSteps: [],  // Initialize empty execution steps array
          });
          break;

        case 'execution_step':
          // Append execution step to the current code_execution message
          setTrialState((prev) => {
            const messages = [...prev.messages];
            const execIdx = messages.findLastIndex(
              (m: ChatMessage) => m.type === 'code_execution' && m.blockIndex === event.block_index
            );
            if (execIdx >= 0) {
              const msg = messages[execIdx];
              const existingSteps = msg.executionSteps || [];
              // Check if this step already exists (update) or is new
              const existingStepIdx = existingSteps.findIndex(
                (s) => s.stepIndex === event.step_index
              );
              let newSteps;
              if (existingStepIdx >= 0) {
                // Update existing step
                newSteps = [...existingSteps];
                newSteps[existingStepIdx] = {
                  toolName: event.tool_name,
                  text: event.text,
                  images: event.images,
                  stepIndex: event.step_index,
                  highlight: event.highlight,
                };
              } else {
                // Add new step
                newSteps = [
                  ...existingSteps,
                  {
                    toolName: event.tool_name,
                    text: event.text,
                    images: event.images,
                    stepIndex: event.step_index,
                    highlight: event.highlight,
                  },
                ];
              }
              messages[execIdx] = {
                ...msg,
                executionSteps: newSteps,
              };
            }
            return { ...prev, messages };
          });
          break;

        case 'code_execution_result':
          // Update the last code execution message
          setTrialState((prev) => {
            const messages = [...prev.messages];
            const lastExecIdx = messages.findLastIndex(
              (m: ChatMessage) => m.type === 'code_execution' && m.blockIndex === event.block_index
            );
            if (lastExecIdx >= 0) {
              messages[lastExecIdx] = {
                ...messages[lastExecIdx],
                isExecuting: false,
                success: event.success,
                stdout: event.stdout,
                stderr: event.stderr,
                reward: event.reward,
                taskCompleted: event.task_completed,
              };
            }
            return { ...prev, messages };
          });
          break;

        case 'visual_feedback':
          addMessage({
            type: 'visual_feedback',
            timestamp: event.timestamp,
            imageBase64: event.image_base64,
            content: event.description,
          });
          break;

        case 'image_analysis':
          addMessage({
            type: 'image_analysis',
            timestamp: event.timestamp,
            content: event.content,
            analysisType: event.analysis_type,
            modelUsed: event.model_used,
          });
          break;

        case 'user_prompt_request':
          addMessage({
            type: 'system',
            timestamp: event.timestamp,
            content: `Executed ${event.executed_code_blocks} block${event.executed_code_blocks !== 1 ? 's' : ''}. Type feedback or click Skip to continue.`,
          });
          break;

        case 'trial_complete':
          addMessage({
            type: 'completion',
            timestamp: event.timestamp,
            success: event.success,
            reward: event.total_reward,
            taskCompleted: event.task_completed,
            summary: event.summary,
          });
          break;

        case 'error':
          addMessage({
            type: 'error',
            timestamp: event.timestamp,
            error: event.message,
          });
          setTrialState((prev) => ({ ...prev, error: event.message }));
          break;
      }
    },
    [addMessage]
  );

  const { isConnected, connect, disconnect, send } = useWebSocket({
    onMessage: handleWSMessage,
  });

  const loadConfig = useCallback(
    async (configPath: string): Promise<LoadConfigResponse | null> => {
      try {
        const response = await fetch('/api/load-config', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ config_path: configPath }),
        });

        if (!response.ok) {
          const error = await response.json();
          throw new Error(error.detail || 'Failed to load config');
        }

        const data: LoadConfigResponse = await response.json();
        setTrialState((prev) => ({
          ...prev,
          configPath,
          taskPrompt: data.task_prompt,
          state: 'idle',
        }));

        return data;
      } catch (e) {
        const errorMsg = e instanceof Error ? e.message : 'Unknown error';
        setTrialState((prev) => ({ ...prev, error: errorMsg }));
        return null;
      }
    },
    []
  );

  const startTrial = useCallback(
    async (request: StartTrialRequest): Promise<boolean> => {
      try {
        setTrialState((prev) => ({
          ...prev,
          state: 'loading_config',
          messages: [],
          error: null,
        }));

        const response = await fetch('/api/start-trial', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(request),
        });

        if (!response.ok) {
          const error = await response.json();
          throw new Error(error.detail || 'Failed to start trial');
        }

        const data = await response.json();
        setTrialState((prev) => ({
          ...prev,
          sessionId: data.session_id,
        }));

        // Connect WebSocket
        connect(data.session_id);

        // Add initial system message
        addMessage({
          type: 'system',
          timestamp: new Date().toISOString(),
          content: `Starting trial with config: ${request.config_path}`,
        });

        return true;
      } catch (e) {
        const errorMsg = e instanceof Error ? e.message : 'Unknown error';
        setTrialState((prev) => ({
          ...prev,
          state: 'error',
          error: errorMsg,
        }));
        return false;
      }
    },
    [connect, addMessage]
  );

  const stopTrial = useCallback(() => {
    send({ type: 'stop' });
  }, [send]);

  const injectPrompt = useCallback(
    (text: string) => {
      // Add user message to chat
      addMessage({
        type: 'user_prompt',
        timestamp: new Date().toISOString(),
        content: text,
      });
      send({ type: 'inject_prompt', text });
    },
    [send, addMessage]
  );

  const resumeTrial = useCallback(() => {
    send({ type: 'resume' });
  }, [send]);

  const updateSettings = useCallback(
    (settings: { await_user_input_each_turn?: boolean }) => {
      send({ type: 'update_settings', ...settings });
    },
    [send]
  );

  const reset = useCallback(() => {
    disconnect();
    // Keep configPath and taskPrompt so user can retry without reloading
    setTrialState((prev) => ({
      sessionId: null,
      state: 'idle',
      messages: [],
      configPath: prev.configPath,
      taskPrompt: prev.taskPrompt,
      error: null,
    }));
  }, [disconnect]);

  const fullReset = useCallback(() => {
    disconnect();
    setTrialState({
      sessionId: null,
      state: 'idle',
      messages: [],
      configPath: null,
      taskPrompt: null,
      error: null,
    });
  }, [disconnect]);

  const checkActiveSession = useCallback(async (): Promise<ActiveSessionResponse | null> => {
    try {
      const response = await fetch('/api/active-session');
      if (!response.ok) {
        return null;
      }
      return await response.json();
    } catch {
      return null;
    }
  }, []);

  const reconnectToSession = useCallback(
    (sessionId: string, configPath: string) => {
      setTrialState((prev) => ({
        ...prev,
        sessionId,
        configPath,
        state: 'running', // Assume running, WebSocket will update
        messages: [
          {
            id: generateMessageId(),
            type: 'system',
            timestamp: new Date().toISOString(),
            content: `Reconnected to existing session`,
          },
        ],
      }));
      connect(sessionId);
    },
    [connect]
  );

  return {
    ...trialState,
    isConnected,
    loadConfig,
    startTrial,
    stopTrial,
    injectPrompt,
    resumeTrial,
    updateSettings,
    reset,
    fullReset,
    checkActiveSession,
    reconnectToSession,
  };
}
