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

  const handleSend = () => {
    const trimmed = text.trim();
    if (trimmed) {
      onSend(trimmed);
      setText('');
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 relative">
        <textarea
          ref={textareaRef}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          placeholder={placeholder}
          rows={1}
          className="w-full pl-4 pr-12 py-3 bg-surface-sunken border border-surface-border rounded-2xl text-sm text-text-primary placeholder-text-tertiary resize-none overflow-y-hidden focus:outline-none focus:ring-2 focus:ring-accent/30 focus:border-accent/40 disabled:opacity-50 disabled:cursor-not-allowed transition-all leading-5"
        />
        {/* Send button inside input */}
        <button
          onClick={handleSend}
          disabled={disabled || !text.trim()}
          className="absolute right-2 top-1/2 -translate-y-1/2 w-8 h-8 rounded-full bg-accent text-white flex items-center justify-center hover:bg-accent-dark disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          title="Send"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" />
          </svg>
        </button>
      </div>
      <button
        onClick={onSkip}
        disabled={disabled}
        className="px-4 py-2.5 bg-surface-overlay text-text-secondary rounded-xl text-sm font-medium hover:bg-surface-border hover:text-text-primary disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
      >
        Skip
      </button>
    </div>
  );
}
