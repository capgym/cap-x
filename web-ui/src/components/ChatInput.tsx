import { useState, useRef, useEffect, KeyboardEvent } from 'react';

interface ChatInputProps {
  onSend: (text: string) => void;
  onSkip: () => void;
  disabled?: boolean;
  placeholder?: string;
}

export function ChatInput({ onSend, onSkip, disabled, placeholder }: ChatInputProps) {
  const [text, setText] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-grow textarea up to 4 lines
  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = 'auto';
    const lineHeight = 20;
    const maxHeight = lineHeight * 4 + 24; // 4 lines + padding
    textarea.style.height = Math.min(textarea.scrollHeight, maxHeight) + 'px';
  }, [text]);

  const handleSubmit = () => {
    const trimmed = text.trim();
    if (trimmed) {
      onSend(trimmed);
      setText('');
    } else {
      // Empty submit = skip/continue
      onSkip();
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const hasText = text.trim().length > 0;
  const isAwaitingInput = !disabled;

  return (
    <div className="relative">
      <textarea
        ref={textareaRef}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={disabled}
        placeholder={placeholder}
        rows={1}
        className="w-full pl-4 pr-24 py-3 bg-surface-sunken border border-surface-border rounded-md text-sm text-text-primary placeholder-text-tertiary resize-none overflow-y-hidden focus:outline-none focus:ring-1 focus:ring-accent/30 focus:border-accent/30 disabled:opacity-40 disabled:cursor-not-allowed transition-all leading-5"
      />
      {/* Submit button — right-aligned, vertically centered to textarea */}
      {isAwaitingInput && (
        <button
          onClick={handleSubmit}
          className={`absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
            hasText
              ? 'bg-accent text-black hover:bg-accent-light'
              : 'bg-surface-overlay text-text-secondary hover:bg-surface-border-light hover:text-text-primary border border-surface-border'
          }`}
          title={hasText ? 'Send feedback (Enter)' : 'Continue without feedback (Enter)'}
        >
          {hasText ? (
            <>
              Send
              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 10.5L12 3m0 0l7.5 7.5M12 3v18" />
              </svg>
            </>
          ) : (
            <>
              Continue
              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3" />
              </svg>
            </>
          )}
        </button>
      )}
    </div>
  );
}
