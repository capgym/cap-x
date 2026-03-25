import { useEffect, useRef, useState } from 'react';
import type { ChatMessage, SessionState } from '../types/messages';
import { MessageList } from './MessageList';
import { ChatInput } from './ChatInput';

interface ChatPanelProps {
  messages: ChatMessage[];
  state: SessionState;
  onSendMessage: (text: string) => void;
  onResume: () => void;
  taskPrompt: string | null;
}

export function ChatPanel({
  messages,
  state,
  onSendMessage,
  onResume,
  taskPrompt,
}: ChatPanelProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [taskExpanded, setTaskExpanded] = useState(false);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTo({
        top: scrollRef.current.scrollHeight,
        behavior: 'smooth',
      });
    }
  }, [messages]);

  const canInput = state === 'awaiting_user_input';

  const taskLines = taskPrompt?.split('\n') || [];
  const charCount = taskPrompt?.length || 0;
  const needsExpansion = taskLines.length > 3 || charCount > 200;

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Task prompt display */}
      {taskPrompt && (
        <div className="flex-shrink-0 bg-surface-raised border-b border-surface-border">
          <button
            onClick={() => needsExpansion && setTaskExpanded(!taskExpanded)}
            className="w-full px-4 py-2.5 flex items-center justify-between hover:bg-surface-overlay/50 transition-colors"
          >
            <div className="flex items-center gap-2.5">
              <div className="w-5 h-5 rounded bg-accent/10 flex items-center justify-center">
                <svg className="w-3.5 h-3.5 text-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
                </svg>
              </div>
              <span className="text-xs font-semibold text-accent uppercase tracking-wide">Task</span>
            </div>
            {needsExpansion && (
              <span className="flex items-center gap-1 text-xs text-text-tertiary">
                {taskExpanded ? 'Collapse' : 'Expand'}
                <svg className={`w-3.5 h-3.5 transition-transform ${taskExpanded ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </span>
            )}
          </button>

          {/* Preview (collapsed) */}
          {!taskExpanded && (
            <div className="px-4 pb-3 relative">
              <div className="text-sm text-text-primary whitespace-pre-wrap overflow-hidden leading-relaxed" style={{ maxHeight: '4.5em' }}>
                {taskLines.slice(0, 3).join('\n')}
              </div>
              {needsExpansion && (
                <div className="absolute bottom-0 left-0 right-0 h-8 bg-gradient-to-t from-surface-raised to-transparent pointer-events-none" />
              )}
            </div>
          )}

          {/* Full content (expanded) */}
          {taskExpanded && (
            <div className="px-4 pb-4 max-h-64 overflow-y-auto">
              <p className="text-sm text-text-primary whitespace-pre-wrap leading-relaxed">{taskPrompt}</p>
            </div>
          )}
        </div>
      )}

      {/* Messages */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto px-4 py-4 space-y-4 bg-surface"
      >
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center py-12">
            <div className="w-14 h-14 rounded-2xl bg-surface-raised flex items-center justify-center mb-4">
              <svg className="w-7 h-7 text-text-muted" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
              </svg>
            </div>
            <p className="text-text-secondary font-medium text-sm">No messages yet</p>
            <p className="text-xs text-text-tertiary mt-1">Start a trial to begin the conversation</p>
          </div>
        ) : (
          <MessageList messages={messages} />
        )}
      </div>

      {/* Input area */}
      <div className="flex-shrink-0 border-t border-surface-border bg-surface-raised p-3">
        <ChatInput
          onSend={onSendMessage}
          onSkip={onResume}
          disabled={!canInput}
          placeholder={
            canInput
              ? 'Type your feedback...'
              : state === 'running'
              ? 'Model is generating...'
              : 'Waiting for trial to start...'
          }
        />
      </div>
    </div>
  );
}
