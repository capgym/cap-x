import type { SessionState } from '../types/messages';

interface StatusBarProps {
  state: SessionState;
  sessionId: string | null;
  isConnected: boolean;
  error: string | null;
}

export function StatusBar({ state, sessionId, isConnected, error }: StatusBarProps) {
  const stateColors: Record<SessionState, string> = {
    idle: 'bg-slate-500',
    loading_config: 'bg-blue-500',
    running: 'bg-green-500 animate-pulse',
    awaiting_user_input: 'bg-amber-500',
    complete: 'bg-green-600',
    error: 'bg-red-500',
  };

  const stateLabels: Record<SessionState, string> = {
    idle: 'Ready',
    loading_config: 'Loading...',
    running: 'Running',
    awaiting_user_input: 'Awaiting Input',
    complete: 'Complete',
    error: 'Error',
  };

  return (
    <div className="flex-shrink-0 bg-slate-800 px-4 py-2 flex items-center justify-between text-xs text-slate-300">
      <div className="flex items-center gap-4">
        {/* State indicator */}
        <div className="flex items-center gap-2">
          <span className={`w-2 h-2 rounded-full ${stateColors[state]}`}></span>
          <span>{stateLabels[state]}</span>
        </div>

        {/* WebSocket connection */}
        <div className="flex items-center gap-2">
          <span
            className={`w-2 h-2 rounded-full ${
              isConnected ? 'bg-green-400' : 'bg-slate-500'
            }`}
          ></span>
          <span>WS: {isConnected ? 'Connected' : 'Disconnected'}</span>
        </div>

        {/* Session ID */}
        {sessionId && (
          <span className="text-slate-500">
            Session: {sessionId.slice(0, 8)}...
          </span>
        )}
      </div>

      {/* Error message */}
      {error && (
        <div className="text-red-400 truncate max-w-md">
          Error: {error}
        </div>
      )}
    </div>
  );
}
