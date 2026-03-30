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
              <div className="w-5 h-5 rounded bg-accent/5 flex items-center justify-center">
                <svg className="w-3.5 h-3.5 text-[#D4A017]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
                </svg>
              </div>
              <span className="text-xs font-semibold text-[#D4A017] uppercase tracking-wide">Task</span>
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
                <div className="absolute bottom-0 left-0 right-0 h-8 bg-gradient-to-t from-surface-raised to-transparent pointer-events-none" style={{ '--tw-gradient-from': '#141414' } as React.CSSProperties} />
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
            {/* Gold accent bar */}
            <div className="w-10 h-0.5 bg-[#D4A017] rounded-full mb-6" />

            {/* Brand title */}
            <h2 className="text-2xl font-semibold text-[#E8E8E8] tracking-[0.2em] mb-1">CaP-X</h2>
            <p className="text-sm text-text-secondary mb-8">Code-as-Policy Agent Framework</p>

            {/* Steps */}
            <div className="flex flex-col gap-3 text-left w-full max-w-[220px]">
              <div className="flex items-center gap-3">
                <span className="text-xs font-medium text-[#D4A017] w-4 shrink-0">1.</span>
                <span className="text-sm text-text-secondary">Select a config</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="text-xs font-medium text-[#D4A017] w-4 shrink-0">2.</span>
                <span className="text-sm text-text-secondary">Start a trial</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="text-xs font-medium text-[#D4A017] w-4 shrink-0">3.</span>
                <span className="text-sm text-text-secondary">Watch the agent code</span>
              </div>
            </div>
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
